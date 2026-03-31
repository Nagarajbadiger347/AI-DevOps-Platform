"""Continuous monitoring loop — background worker for proactive anomaly detection.

Polls K8s and AWS on a configurable interval. When anomalies are detected,
it triggers the LangGraph pipeline automatically.

Enable via: ENABLE_MONITOR_LOOP=true in .env
Control remediation: AUTO_REMEDIATE_ON_MONITOR=true (default: false — alert only)
"""
from __future__ import annotations

import asyncio
import time

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Pod restart threshold to flag as anomaly
_RESTART_THRESHOLD = 5


def _detect_k8s_anomalies() -> list[str]:
    """Synchronous K8s health scan — returns list of anomaly descriptions."""
    issues: list[str] = []
    try:
        from app.plugins.k8s_checker import check_k8s_cluster, check_k8s_pods
        cluster = check_k8s_cluster()
        if isinstance(cluster, dict) and cluster.get("status") == "error":
            return []  # K8s not configured — skip silently

        pods_result = check_k8s_pods("default")
        pod_list: list = []
        if isinstance(pods_result, list):
            pod_list = pods_result
        elif isinstance(pods_result, dict):
            pod_list = pods_result.get("pods", [])

        for pod in pod_list:
            if not isinstance(pod, dict):
                continue
            restarts = pod.get("restart_count", 0) or 0
            if int(restarts) >= _RESTART_THRESHOLD:
                issues.append(
                    f"Pod {pod.get('name', 'unknown')} in namespace "
                    f"{pod.get('namespace', 'default')} has {restarts} restarts"
                )
            if pod.get("status") not in ("Running", "Completed", "Succeeded", None):
                issues.append(
                    f"Pod {pod.get('name', 'unknown')} is in {pod.get('status')} state"
                )
    except Exception as exc:
        logger.warning("monitor_k8s_scan_failed", error=str(exc))
    return issues


def _detect_aws_anomalies() -> list[str]:
    """Scan AWS for CloudWatch alarms, Lambda errors, ECS failures, RDS events, SQS depth."""
    issues: list[str] = []

    # CloudWatch: alarms in ALARM state
    try:
        from app.integrations.aws_ops import list_cloudwatch_alarms
        cw = list_cloudwatch_alarms(state="ALARM")
        if cw.get("success"):
            for alarm in cw.get("alarms", []):
                if alarm.get("state") == "ALARM":
                    issues.append(
                        f"CloudWatch alarm FIRING: {alarm.get('name', 'unknown')} — "
                        f"{alarm.get('description', '')}"
                    )
    except Exception:
        pass

    # Lambda: functions with recent errors (last 1 h)
    try:
        from app.integrations.aws_ops import list_lambda_functions, get_lambda_errors
        lambdas = list_lambda_functions()
        if lambdas.get("success"):
            for fn in lambdas.get("functions", [])[:20]:  # cap at 20 to avoid throttle
                name = fn.get("name", "")
                try:
                    err = get_lambda_errors(name, hours=1)
                    error_pts = err.get("metrics", {}).get("errors", [])
                    total_errors = sum(p.get("value", 0) for p in error_pts)
                    if err.get("success") and total_errors > 0:
                        issues.append(
                            f"Lambda errors detected: {name} had "
                            f"{int(total_errors)} error(s) in the last hour"
                        )
                except Exception:
                    pass
    except Exception:
        pass

    # ECS: stopped tasks with non-zero exit codes
    try:
        from app.integrations.aws_ops import get_stopped_ecs_tasks
        stopped = get_stopped_ecs_tasks()
        if stopped.get("success"):
            for t in stopped.get("stopped_tasks", []):
                reason = t.get("stop_reason", "")
                if reason and any(k in reason for k in ("Essential container", "OOM", "error", "failed")):
                    cluster = stopped.get("cluster", "unknown")
                    issues.append(
                        f"ECS task crash in cluster {cluster}: "
                        f"task {t.get('task_id', '')[:20]} — {reason}"
                    )
    except Exception:
        pass

    # RDS: recent failover / restart events
    try:
        from app.integrations.aws_ops import list_rds_instances, get_rds_events
        rds = list_rds_instances()
        if rds.get("success"):
            for db in rds.get("instances", [])[:10]:
                db_id = db.get("id", "")
                try:
                    evts = get_rds_events(db_id, hours=1)
                    for ev in evts.get("events", []):
                        msg = ev.get("message", "").lower()
                        if any(k in msg for k in ("failover", "restarting", "crash", "oom")):
                            issues.append(
                                f"RDS event on {db_id}: {ev.get('message', '')}"
                            )
                except Exception:
                    pass
    except Exception:
        pass

    # SQS: queues with depth > 1000
    try:
        from app.integrations.aws_ops import list_sqs_queues
        sqs = list_sqs_queues()
        if sqs.get("success"):
            for q in sqs.get("queues", [])[:15]:
                visible = q.get("visible", 0) or 0
                if visible > 1000:
                    issues.append(
                        f"SQS queue backlog: {q.get('name', q.get('url', 'unknown'))} "
                        f"has {visible} messages queued"
                    )
    except Exception:
        pass

    return issues


def _detect_anomalies() -> list[str]:
    """Aggregate anomaly detectors — K8s, Grafana, CloudWatch, Lambda, ECS, RDS, SQS."""
    anomalies = _detect_k8s_anomalies()

    # Grafana alerts
    try:
        from app.integrations.grafana import get_alerts
        gf = get_alerts()
        if gf.get("success"):
            for alert in gf.get("alerts", []):
                if alert.get("state") in ("alerting", "pending"):
                    anomalies.append(
                        f"Grafana alert firing: {alert.get('name', 'unknown')}"
                    )
    except Exception:
        pass

    # AWS anomalies
    anomalies.extend(_detect_aws_anomalies())

    return anomalies


async def monitoring_loop() -> None:
    """Async background task — runs forever, triggering pipelines on anomalies."""
    logger.info(
        "monitoring_loop_started",
        interval_seconds=settings.MONITOR_INTERVAL_SECONDS,
        auto_remediate=settings.AUTO_REMEDIATE_ON_MONITOR,
    )

    while True:
        await asyncio.sleep(settings.MONITOR_INTERVAL_SECONDS)

        try:
            # Run blocking scan in thread so we don't block the event loop
            anomalies = await asyncio.get_event_loop().run_in_executor(
                None, _detect_anomalies
            )
        except Exception as exc:
            logger.warning("monitor_scan_error", error=str(exc))
            continue

        if not anomalies:
            logger.info("monitor_scan_clean")
            continue

        logger.warning("monitor_anomalies_detected", count=len(anomalies))

        for description in anomalies:
            incident_id = f"monitor-{int(time.time())}"
            logger.warning("monitor_triggering_pipeline",
                           incident_id=incident_id,
                           description=description)
            try:
                # Run pipeline in executor to keep monitoring loop non-blocking
                from app.orchestrator.runner import run_pipeline
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: run_pipeline(
                        incident_id=incident_id,
                        description=description,
                        auto_remediate=settings.AUTO_REMEDIATE_ON_MONITOR,
                        metadata={"user": "monitor", "role": "admin"},
                    ),
                )
            except Exception as exc:
                logger.error("monitor_pipeline_failed",
                             incident_id=incident_id, error=str(exc))
