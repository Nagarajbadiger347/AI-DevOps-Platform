"""
Kubernetes Tool — wraps app/integrations/k8s_ops.py for agent use.
All methods return {"success": bool, "data": any, "error": str|None}.
"""
from __future__ import annotations
import logging

logger = logging.getLogger("nsops.tools.k8s")


class KubernetesTool:
    """Kubernetes operations tool used by LangGraph agents."""

    # ── Pod operations ────────────────────────────────────────────────────────

    def get_pods(self, namespace: str = "") -> dict:
        """List all pods, optionally filtered by namespace."""
        try:
            from app.integrations.k8s_ops import list_pods
            result = list_pods(namespace=namespace)
            logger.info("get_pods ns=%s count=%d", namespace or "all", len(result.get("pods", [])))
            return {"success": True, "data": result.get("pods", []), "error": None}
        except Exception as e:
            logger.error("get_pods failed: %s", e)
            return {"success": False, "data": [], "error": str(e)}

    def describe_pod(self, namespace: str, pod: str) -> dict:
        """Get detailed info about a specific pod (status, conditions, containers)."""
        try:
            from app.integrations.k8s_ops import list_pods
            # Filter pods to find the specific one
            result = list_pods(namespace=namespace)
            pods = result.get("pods", [])
            match = next((p for p in pods if p.get("name") == pod), None)
            if not match:
                return {"success": False, "data": None,
                        "error": f"Pod '{pod}' not found in namespace '{namespace}'"}
            logger.info("describe_pod ns=%s pod=%s status=%s", namespace, pod, match.get("status"))
            return {"success": True, "data": match, "error": None}
        except Exception as e:
            logger.error("describe_pod failed: %s", e)
            return {"success": False, "data": None, "error": str(e)}

    def get_logs(self, namespace: str, pod: str, container: str = "",
                 tail_lines: int = 200) -> dict:
        """Fetch logs from a pod/container."""
        try:
            from app.integrations.k8s_ops import get_pod_logs
            result = get_pod_logs(namespace=namespace, pod=pod,
                                  container=container, tail_lines=tail_lines)
            logs = result.get("logs", result.get("output", ""))
            logger.info("get_logs ns=%s pod=%s lines=%d", namespace, pod, len(logs.splitlines()))
            return {"success": True, "data": logs, "error": None}
        except Exception as e:
            logger.error("get_logs failed: %s", e)
            return {"success": False, "data": "", "error": str(e)}

    def get_events(self, namespace: str = "", limit: int = 30) -> dict:
        """Fetch cluster or namespace-scoped events."""
        try:
            from app.integrations.k8s_ops import get_cluster_events
            result = get_cluster_events(namespace=namespace, limit=limit)
            events = result.get("events", [])
            logger.info("get_events ns=%s count=%d", namespace or "all", len(events))
            return {"success": True, "data": events, "error": None}
        except Exception as e:
            logger.error("get_events failed: %s", e)
            return {"success": False, "data": [], "error": str(e)}

    def get_unhealthy_pods(self, namespace: str = "") -> dict:
        """Return only pods that are not Running/Succeeded."""
        try:
            from app.integrations.k8s_ops import get_unhealthy_pods
            result = get_unhealthy_pods(namespace=namespace)
            pods = result.get("pods", result.get("unhealthy_pods", []))
            return {"success": True, "data": pods, "error": None}
        except Exception as e:
            return {"success": False, "data": [], "error": str(e)}

    # ── Remediation ───────────────────────────────────────────────────────────

    def restart_pod(self, namespace: str, pod: str) -> dict:
        """Delete pod so it gets rescheduled (soft restart)."""
        try:
            from app.integrations.k8s_ops import delete_pod
            result = delete_pod(namespace=namespace, pod=pod)
            logger.info("restart_pod ns=%s pod=%s success=%s", namespace, pod, result.get("success"))
            return {"success": result.get("success", True), "data": result, "error": None}
        except Exception as e:
            logger.error("restart_pod failed: %s", e)
            return {"success": False, "data": None, "error": str(e)}

    def restart_deployment(self, namespace: str, deployment: str) -> dict:
        """Rolling restart of a deployment."""
        try:
            from app.integrations.k8s_ops import restart_deployment
            result = restart_deployment(namespace=namespace, deployment=deployment)
            return {"success": result.get("success", True), "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def scale_deployment(self, namespace: str, deployment: str, replicas: int) -> dict:
        """Scale a deployment to N replicas."""
        try:
            from app.integrations.k8s_ops import scale_deployment
            result = scale_deployment(namespace=namespace, deployment=deployment, replicas=replicas)
            return {"success": result.get("success", True), "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}

    def get_resource_usage(self, namespace: str = "default") -> dict:
        """Get CPU/memory resource usage."""
        try:
            from app.integrations.k8s_ops import get_resource_usage
            result = get_resource_usage(namespace=namespace)
            return {"success": True, "data": result, "error": None}
        except Exception as e:
            return {"success": False, "data": {}, "error": str(e)}
