"""
Observer Agent — listens to events (alerts, pipeline failures, K8s events)
and triggers the appropriate workflow.
"""
from __future__ import annotations
import logging
import time
from typing import Callable

logger = logging.getLogger("nsops.agent.observer")


class ObserverAgent:
    """
    Processes incoming events from webhooks, Prometheus, GitLab, and K8s.
    Routes them to the appropriate workflow via registered handlers.
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def process_event(self, event: dict) -> dict:
        """
        Process an incoming event and trigger the appropriate workflow.

        Event schema:
          {
            "type": "k8s_alert|gitlab_pipeline|prometheus_alert|manual",
            "source": str,
            "payload": dict,
            "timestamp": float (optional)
          }
        """
        event_type = event.get("type", "unknown")
        payload = event.get("payload", {})
        source = event.get("source", "unknown")

        logger.info("[OBSERVER] event type=%s source=%s", event_type, source)

        result = {"event_type": event_type, "processed": False, "workflow_triggered": None}

        # Route by event type
        if event_type == "k8s_alert":
            result.update(self._handle_k8s_alert(payload))
        elif event_type == "gitlab_pipeline":
            result.update(self._handle_gitlab_pipeline(payload))
        elif event_type == "prometheus_alert":
            result.update(self._handle_prometheus_alert(payload))
        elif event_type == "manual_debug":
            result.update(self._handle_manual_debug(payload))
        else:
            result["error"] = f"Unknown event type: {event_type}"

        # Call registered handlers
        for handler in self._handlers.get(event_type, []):
            try:
                handler(event, result)
            except Exception as e:
                logger.warning("[OBSERVER] handler error: %s", e)

        return result

    def _handle_k8s_alert(self, payload: dict) -> dict:
        """Handle Kubernetes pod/deployment alerts."""
        namespace = payload.get("namespace", "default")
        pod = payload.get("pod_name") or payload.get("pod", "")
        labels = payload.get("labels", {})

        if not pod and labels:
            pod = labels.get("pod", labels.get("pod_name", ""))

        if not pod:
            return {"processed": False, "error": "No pod_name in k8s_alert payload"}

        logger.info("[OBSERVER] k8s_alert: triggering incident debug ns=%s pod=%s", namespace, pod)
        from app.workflows.incident_workflow import run_incident_debug
        result = run_incident_debug(namespace=namespace, pod_name=pod, dry_run=True)
        return {
            "processed": True,
            "workflow_triggered": "incident_debug",
            "workflow_result": {
                "failure_type": result.get("failure_type"),
                "severity": result.get("severity"),
                "root_cause": result.get("root_cause", "")[:200],
                "report_preview": result.get("report", "")[:500],
            },
        }

    def _handle_gitlab_pipeline(self, payload: dict) -> dict:
        """Handle GitLab pipeline failure webhook."""
        project_id = str(payload.get("project", {}).get("id", "") or payload.get("project_id", ""))
        pipeline_id = str(payload.get("object_attributes", {}).get("id", "") or payload.get("pipeline_id", ""))
        status = payload.get("object_attributes", {}).get("status", payload.get("status", ""))

        if status not in ("failed", "error"):
            return {"processed": False, "reason": f"Pipeline status '{status}' is not a failure"}

        logger.info("[OBSERVER] gitlab pipeline failure: project=%s pipeline=%s", project_id, pipeline_id)

        from app.tools.gitlab import GitLabTool
        gl = GitLabTool()
        logs = gl.get_pipeline_logs(project_id=project_id, pipeline_id=pipeline_id)

        from app.agents.debugger import DebuggerAgent
        debug = DebuggerAgent()
        log_text = str(logs.get("data", ""))
        analysis = debug.analyze_log_error(log_text, service=f"gitlab-pipeline-{pipeline_id}")

        return {
            "processed": True,
            "workflow_triggered": "pipeline_debug",
            "analysis": analysis,
        }

    def _handle_prometheus_alert(self, payload: dict) -> dict:
        """Handle Prometheus/Alertmanager alert."""
        alerts = payload.get("alerts", [payload])
        results = []
        for alert in alerts[:3]:  # process up to 3 alerts
            name = alert.get("labels", {}).get("alertname", "unknown")
            ns = alert.get("labels", {}).get("namespace", "default")
            pod = alert.get("labels", {}).get("pod", "")

            if pod:
                from app.workflows.incident_workflow import run_incident_debug
                r = run_incident_debug(namespace=ns, pod_name=pod, dry_run=True)
                results.append({"alert": name, "pod": pod, "failure": r.get("failure_type"),
                                 "severity": r.get("severity")})
            else:
                results.append({"alert": name, "reason": "No pod label in alert"})

        return {"processed": True, "workflow_triggered": "prometheus_alert", "results": results}

    def _handle_manual_debug(self, payload: dict) -> dict:
        """Handle manual debug trigger from API."""
        namespace = payload.get("namespace", "default")
        pod = payload.get("pod_name", "")
        dry_run = payload.get("dry_run", True)
        auto_fix = payload.get("auto_fix", False)

        if not pod:
            return {"processed": False, "error": "pod_name required for manual_debug"}

        from app.workflows.incident_workflow import run_incident_debug
        result = run_incident_debug(namespace=namespace, pod_name=pod,
                                    dry_run=dry_run, auto_fix=auto_fix)
        return {"processed": True, "workflow_triggered": "incident_debug", "result": result}


# Module-level singleton
_observer = ObserverAgent()


def get_observer() -> ObserverAgent:
    return _observer
