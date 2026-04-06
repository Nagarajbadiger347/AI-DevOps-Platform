"""Continuous monitoring loop — background worker for proactive anomaly detection.

Polls K8s and AWS on a configurable interval. When anomalies are detected,
it triggers the LangGraph pipeline automatically.

Features:
- Alert deduplication (10-minute suppression window per fingerprint)
- Async queue-based processing with 5-second inter-alert delay
- Exponential backoff on AWS/K8s API failures (30s → 60s → 120s, max 3 retries)
- External webhook injection via receive_external_alert()

Enable via: ENABLE_MONITOR_LOOP=true in .env
Control remediation: AUTO_REMEDIATE_ON_MONITOR=true (default: false — alert only)
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _vs(msg: str, show: bool = False) -> None:
    """Fire-and-forget write to the VS Code NsOps output channel."""
    try:
        from app.integrations.vscode import write_output
        write_output(msg, show=show)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Detector health tracking
# ---------------------------------------------------------------------------
_detector_health: dict[str, float] = {}   # detector_name → last_success_time
_DETECTOR_WARN_AFTER = 1800  # 30 minutes

# Pod restart threshold to flag as anomaly
_RESTART_THRESHOLD = 5

# Deduplication: tracks currently-firing alerts by fingerprint
_active_alerts: dict[str, dict] = {}

# Queue for event-driven alert processing
_alert_queue: asyncio.Queue = asyncio.Queue()

# Dedup suppression window in seconds
_DEDUP_WINDOW = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _make_fingerprint(alert_type: str, resource_id: str) -> str:
    """Create a stable dedup key from alert type and resource id."""
    raw = f"{alert_type}:{resource_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_duplicate(fingerprint: str) -> bool:
    """Return True if this fingerprint was seen within the dedup window."""
    entry = _active_alerts.get(fingerprint)
    if entry is None:
        return False
    age = time.time() - entry.get("last_seen", 0)
    return age < _DEDUP_WINDOW


def _record_alert(fingerprint: str, alert_type: str, resource_id: str) -> None:
    entry = _active_alerts.get(fingerprint, {"pipeline_triggered": 0})
    entry["alert_type"] = alert_type
    entry["resource_id"] = resource_id
    entry["last_seen"] = time.time()
    entry["pipeline_triggered"] = entry.get("pipeline_triggered", 0) + 1
    _active_alerts[fingerprint] = entry


def _resolve_alert(alert_type: str, resource_id: str) -> None:
    """Remove fingerprint from active alerts when state transitions to OK/resolved."""
    fp = _make_fingerprint(alert_type, resource_id)
    _active_alerts.pop(fp, None)
    _vs(f"✅ RESOLVED  [{alert_type}] {resource_id}")


def _enqueue_alert(alert_type: str, resource_id: str, description: str,
                   source: str = "monitor", extra: dict | None = None) -> bool:
    """Deduplicate and enqueue an alert. Returns True if enqueued."""
    fp = _make_fingerprint(alert_type, resource_id)
    if _is_duplicate(fp):
        logger.debug("alert_deduplicated", fingerprint=fp, alert_type=alert_type)
        return False
    _record_alert(fp, alert_type, resource_id)
    payload = {
        "alert_type": alert_type,
        "resource_id": resource_id,
        "description": description,
        "source": source,
        "fingerprint": fp,
        **(extra or {}),
    }
    _alert_queue.put_nowait(payload)
    _vs(f"🚨 ALERT  [{alert_type}] {resource_id} — {description}", show=True)
    return True


# ---------------------------------------------------------------------------
# K8s detection
# ---------------------------------------------------------------------------

def _detect_k8s_anomalies() -> list[dict]:
    """Synchronous K8s health scan — returns list of alert dicts."""
    alerts: list[dict] = []
    try:
        from app.plugins.k8s_checker import check_k8s_cluster, check_k8s_pods
        cluster = check_k8s_cluster()
        if isinstance(cluster, dict) and cluster.get("status") == "error":
            return []

        pods_result = check_k8s_pods("default")
        pod_list: list = []
        if isinstance(pods_result, list):
            pod_list = pods_result
        elif isinstance(pods_result, dict):
            pod_list = pods_result.get("pods", [])

        for pod in pod_list:
            if not isinstance(pod, dict):
                continue
            name = pod.get("name", "unknown")
            ns = pod.get("namespace", "default")
            restarts = int(pod.get("restart_count", 0) or 0)
            if restarts >= _RESTART_THRESHOLD:
                alerts.append({
                    "alert_type": "k8s_pod_restart",
                    "resource_id": f"{ns}/{name}",
                    "description": f"Pod {name} in namespace {ns} has {restarts} restarts",
                })
            status = pod.get("status")
            if status not in ("Running", "Completed", "Succeeded", None):
                alerts.append({
                    "alert_type": "k8s_pod_status",
                    "resource_id": f"{ns}/{name}",
                    "description": f"Pod {name} is in {status} state",
                })
    except Exception as exc:
        logger.warning("monitor_k8s_scan_failed", error=str(exc))

    # K8s node health
    try:
        from app.integrations.k8s_ops import list_deployments
        from app.plugins.k8s_checker import check_k8s_nodes
        nodes_result = check_k8s_nodes()
        node_list: list = []
        if isinstance(nodes_result, list):
            node_list = nodes_result
        elif isinstance(nodes_result, dict):
            node_list = nodes_result.get("nodes", [])
        for node in node_list:
            if not isinstance(node, dict):
                continue
            node_name = node.get("name", "unknown")
            ready = node.get("ready", True)
            if not ready:
                alerts.append({
                    "alert_type": "k8s_node_notready",
                    "resource_id": node_name,
                    "description": f"K8s node {node_name} is NotReady",
                })
            else:
                _resolve_alert("k8s_node_notready", node_name)
        _detector_health["k8s_nodes"] = time.time()
    except Exception as exc:
        logger.warning("detector_failed", extra={"detector": "k8s_nodes", "error": str(exc)})

    # K8s deployments with 0 available replicas
    try:
        from app.integrations.k8s_ops import list_deployments as _list_deps
        deps_result = _list_deps()
        dep_list: list = []
        if isinstance(deps_result, list):
            dep_list = deps_result
        elif isinstance(deps_result, dict):
            dep_list = deps_result.get("deployments", [])
        for dep in dep_list:
            if not isinstance(dep, dict):
                continue
            dep_name = dep.get("name", "unknown")
            ns = dep.get("namespace", "default")
            desired = int(dep.get("replicas", 0) or dep.get("desired", 0) or 0)
            available = int(dep.get("available", 0) or dep.get("ready", 0) or 0)
            if desired > 0 and available == 0:
                alerts.append({
                    "alert_type": "k8s_deployment_down",
                    "resource_id": f"{ns}/{dep_name}",
                    "description": f"Deployment {dep_name} in {ns} has 0/{desired} pods available",
                })
            elif desired > 0 and available >= desired:
                _resolve_alert("k8s_deployment_down", f"{ns}/{dep_name}")
        _detector_health["k8s_deployments"] = time.time()
    except Exception as exc:
        logger.warning("detector_failed", extra={"detector": "k8s_deployments", "error": str(exc)})

    return alerts


# ---------------------------------------------------------------------------
# EC2 detection
# ---------------------------------------------------------------------------

def _detect_ec2_anomalies() -> list[dict]:
    """Detect EC2 instances that are stopped or in an error state."""
    alerts: list[dict] = []
    try:
        from app.integrations.aws_ops import list_ec2_instances
        result = list_ec2_instances()
        if not result.get("success"):
            return alerts
        for inst in result.get("instances", []):
            state = (inst.get("state") or "").lower()
            iid = inst.get("id", "")
            name = inst.get("name") or iid
            if state == "stopped":
                alerts.append({
                    "alert_type": "ec2_stopped",
                    "resource_id": iid,
                    "description": f"EC2 instance {name} ({iid}) is in stopped state",
                })
            elif state in ("shutting-down", "terminated"):
                alerts.append({
                    "alert_type": "ec2_terminated",
                    "resource_id": iid,
                    "description": f"EC2 instance {name} ({iid}) is {state}",
                })
            elif state == "running":
                # Instance recovered — clear any prior stopped alert
                _resolve_alert("ec2_stopped", iid)
                _resolve_alert("ec2_terminated", iid)
        _detector_health["ec2"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "ec2", "error": str(e)})
    return alerts


# ---------------------------------------------------------------------------
# AWS detection
# ---------------------------------------------------------------------------

def _detect_aws_anomalies() -> list[dict]:
    """Scan AWS — CloudWatch, Lambda, ECS, RDS, SQS."""
    alerts: list[dict] = []

    # CloudWatch alarms in ALARM state
    try:
        from app.integrations.aws_ops import list_cloudwatch_alarms
        cw = list_cloudwatch_alarms(state="ALARM")
        if cw.get("success"):
            for alarm in cw.get("alarms", []):
                if alarm.get("state") == "ALARM":
                    name = alarm.get("name", "unknown")
                    alerts.append({
                        "alert_type": "cloudwatch_alarm",
                        "resource_id": name,
                        "description": (
                            f"CloudWatch alarm FIRING: {name} — "
                            f"{alarm.get('description', '')}"
                        ),
                    })
                elif alarm.get("state") == "OK":
                    _resolve_alert("cloudwatch_alarm", alarm.get("name", ""))
        _detector_health["cloudwatch"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "cloudwatch", "error": str(e)})

    # Lambda errors
    try:
        from app.integrations.aws_ops import list_lambda_functions, get_lambda_errors
        lambdas = list_lambda_functions()
        if lambdas.get("success"):
            for fn in lambdas.get("functions", [])[:20]:
                name = fn.get("name", "")
                try:
                    err = get_lambda_errors(name, hours=1)
                    error_pts = err.get("metrics", {}).get("errors", [])
                    total_errors = sum(p.get("value", 0) for p in error_pts)
                    if err.get("success") and total_errors > 0:
                        alerts.append({
                            "alert_type": "lambda_errors",
                            "resource_id": name,
                            "description": (
                                f"Lambda errors detected: {name} had "
                                f"{int(total_errors)} error(s) in the last hour"
                            ),
                        })
                except Exception as e:
                    logger.warning("detector_failed", extra={"detector": "lambda_errors", "error": str(e)})
        _detector_health["lambda"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "lambda", "error": str(e)})

    # ECS stopped tasks
    try:
        from app.integrations.aws_ops import get_stopped_ecs_tasks
        stopped = get_stopped_ecs_tasks()
        if stopped.get("success"):
            cluster_name = stopped.get("cluster", "unknown")
            for t in stopped.get("stopped_tasks", []):
                reason = t.get("stop_reason", "")
                if reason and any(k in reason for k in ("Essential container", "OOM", "error", "failed")):
                    task_id = t.get("task_id", "")[:20]
                    alerts.append({
                        "alert_type": "ecs_task_crash",
                        "resource_id": f"{cluster_name}/{task_id}",
                        "description": (
                            f"ECS task crash in cluster {cluster_name}: "
                            f"task {task_id} — {reason}"
                        ),
                    })
        _detector_health["ecs"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "ecs", "error": str(e)})

    # ECS service health — running count below desired
    try:
        from app.integrations.aws_ops import list_ecs_services, get_ecs_service_detail
        svcs = list_ecs_services()
        if svcs.get("success"):
            for svc in svcs.get("services", [])[:20]:
                svc_name = svc.get("service_name") or svc.get("name", "")
                cluster = svc.get("cluster_arn", "").split("/")[-1] or "default"
                desired = int(svc.get("desired_count", 0) or 0)
                running = int(svc.get("running_count", 0) or 0)
                if desired > 0 and running == 0:
                    alerts.append({
                        "alert_type": "ecs_service_down",
                        "resource_id": f"{cluster}/{svc_name}",
                        "description": (
                            f"ECS service {svc_name} in cluster {cluster} has "
                            f"0/{desired} tasks running"
                        ),
                    })
                elif desired > 0 and running < desired:
                    alerts.append({
                        "alert_type": "ecs_service_degraded",
                        "resource_id": f"{cluster}/{svc_name}",
                        "description": (
                            f"ECS service {svc_name} in cluster {cluster} is degraded: "
                            f"{running}/{desired} tasks running"
                        ),
                    })
                elif running >= desired and desired > 0:
                    _resolve_alert("ecs_service_down", f"{cluster}/{svc_name}")
                    _resolve_alert("ecs_service_degraded", f"{cluster}/{svc_name}")
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "ecs_services", "error": str(e)})

    # RDS events
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
                            alerts.append({
                                "alert_type": "rds_event",
                                "resource_id": db_id,
                                "description": f"RDS event on {db_id}: {ev.get('message', '')}",
                            })
                except Exception as e:
                    logger.warning("detector_failed", extra={"detector": "rds_events", "error": str(e)})
        _detector_health["rds"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "rds", "error": str(e)})

    # SQS backlog
    try:
        from app.integrations.aws_ops import list_sqs_queues
        sqs = list_sqs_queues()
        if sqs.get("success"):
            for q in sqs.get("queues", [])[:15]:
                visible = q.get("visible", 0) or 0
                if visible > 1000:
                    qname = q.get("name", q.get("url", "unknown"))
                    alerts.append({
                        "alert_type": "sqs_backlog",
                        "resource_id": qname,
                        "description": (
                            f"SQS queue backlog: {qname} has {visible} messages queued"
                        ),
                    })
        _detector_health["sqs"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "sqs", "error": str(e)})

    return alerts


# ---------------------------------------------------------------------------
# Grafana detection
# ---------------------------------------------------------------------------

def _detect_grafana_anomalies() -> list[dict]:
    alerts: list[dict] = []
    try:
        from app.plugins.grafana_checker import check_grafana
        result = check_grafana()
        if result.get("status") == "unavailable":
            return alerts
        if result.get("success"):
            for alert in result.get("details", {}).get("firing", []):
                name = alert.get("name", "unknown")
                sev  = alert.get("severity", "")
                desc = alert.get("summary", "") or f"Grafana alert firing: {name}"
                if sev:
                    desc = f"[{sev.upper()}] {desc}"
                alerts.append({
                    "alert_type": "grafana_alert",
                    "resource_id": name,
                    "description": desc,
                })
            # Resolve alerts no longer firing
            firing_names = {a.get("name", "") for a in result.get("details", {}).get("firing", [])}
            for alert_name in list(_active_alerts.keys()):
                entry = _active_alerts.get(alert_name, {})
                if entry.get("alert_type") == "grafana_alert" and entry.get("resource_id", "") not in firing_names:
                    _resolve_alert("grafana_alert", entry.get("resource_id", ""))
            _detector_health["grafana"] = time.time()
    except Exception as e:
        logger.warning("detector_failed", extra={"detector": "grafana", "error": str(e)})
    return alerts


# ---------------------------------------------------------------------------
# Detection with exponential backoff
# ---------------------------------------------------------------------------

async def _detect_with_backoff(detector_fn, max_retries: int = 3) -> list[dict]:
    """Run a blocking detector in an executor with exponential backoff on failure."""
    delays = [30, 60, 120]
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            return await loop.run_in_executor(None, detector_fn)
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = delays[attempt]
                logger.warning("detector_retry",
                               detector=detector_fn.__name__,
                               attempt=attempt + 1,
                               wait_seconds=wait,
                               error=str(exc))
                await asyncio.sleep(wait)
            else:
                logger.error("detector_max_retries_exceeded",
                             detector=detector_fn.__name__, error=str(exc))
    return []


# ---------------------------------------------------------------------------
# Detect and enqueue
# ---------------------------------------------------------------------------

async def _detect_and_enqueue() -> None:
    """Run all detectors and enqueue any new (non-duplicate) alerts."""
    k8s_alerts = await _detect_with_backoff(_detect_k8s_anomalies)
    ec2_alerts = await _detect_with_backoff(_detect_ec2_anomalies)
    aws_alerts = await _detect_with_backoff(_detect_aws_anomalies)
    gf_alerts  = await _detect_with_backoff(_detect_grafana_anomalies)

    all_alerts = k8s_alerts + ec2_alerts + aws_alerts + gf_alerts

    if not all_alerts:
        logger.info("monitor_scan_clean")
        return

    logger.warning("monitor_anomalies_detected", count=len(all_alerts))
    enqueued = 0
    for alert in all_alerts:
        queued = _enqueue_alert(
            alert_type=alert["alert_type"],
            resource_id=alert["resource_id"],
            description=alert["description"],
            source="monitor",
        )
        if queued:
            enqueued += 1

    if enqueued:
        logger.info("alerts_enqueued", count=enqueued)


# ---------------------------------------------------------------------------
# Alert queue processor
# ---------------------------------------------------------------------------

async def _process_alert_queue() -> None:
    """Consume alerts from the queue one at a time, with 5s delay between each."""
    while True:
        try:
            alert = await _alert_queue.get()
            incident_id = f"monitor-{int(time.time())}"
            logger.warning(
                "monitor_triggering_pipeline",
                incident_id=incident_id,
                description=alert["description"],
                source=alert.get("source", "monitor"),
                fingerprint=alert.get("fingerprint"),
            )
            _vs(f"⚙️  PIPELINE  [{incident_id}] Starting AI pipeline for: {alert['description']}")
            try:
                from app.orchestrator.runner import run_pipeline
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda a=alert, iid=incident_id: run_pipeline(
                        incident_id=iid,
                        description=a["description"],
                        auto_remediate=settings.AUTO_REMEDIATE_ON_MONITOR,
                        metadata={"user": "monitor", "role": "admin",
                                  "source": a.get("source", "monitor")},
                    ),
                )
            except Exception as exc:
                logger.error("monitor_pipeline_failed",
                             incident_id=incident_id, error=str(exc))
            finally:
                _alert_queue.task_done()
            # Brief pause between pipeline triggers
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("alert_queue_processor_error", error=str(exc))
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# External webhook injection
# ---------------------------------------------------------------------------

def receive_external_alert(source: str, payload: dict) -> None:
    """Inject an alert from an external source (Grafana, PagerDuty, etc.) into the queue.

    Designed to be called from API endpoint handlers.
    """
    alert_type = payload.get("alert_type", f"{source}_webhook")
    resource_id = payload.get("resource_id") or payload.get("name") or source
    description = payload.get("description") or payload.get("message") or f"{source} alert"

    enqueued = _enqueue_alert(
        alert_type=alert_type,
        resource_id=str(resource_id),
        description=str(description),
        source=source,
        extra=payload,
    )
    if enqueued:
        logger.info("external_alert_enqueued", source=source, resource_id=resource_id)
    else:
        logger.debug("external_alert_deduplicated", source=source, resource_id=resource_id)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def monitoring_loop() -> None:
    """Async background task — runs forever, detecting anomalies and triggering pipelines."""
    logger.info(
        "monitoring_loop_started",
        interval_seconds=settings.MONITOR_INTERVAL_SECONDS,
        auto_remediate=settings.AUTO_REMEDIATE_ON_MONITOR,
    )

    # Start the queue processor as a concurrent task
    processor_task = asyncio.ensure_future(_process_alert_queue())

    try:
        while True:
            await asyncio.sleep(settings.MONITOR_INTERVAL_SECONDS)

            try:
                await _detect_and_enqueue()
            except Exception as exc:
                logger.warning("monitor_scan_error", error=str(exc))

            # Warn if any detector hasn't succeeded in 30 minutes
            now = time.time()
            for det, last_ok in _detector_health.items():
                if now - last_ok > _DETECTOR_WARN_AFTER:
                    logger.warning(
                        "detector_stale",
                        extra={"detector": det, "last_success_seconds_ago": int(now - last_ok)},
                    )
    finally:
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass
