"""K8sAgent — collects Kubernetes cluster state into pipeline state.

Read-only: no mutations. All K8s control-plane actions go through the Executor.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class K8sAgent(BaseAgent):
    """Wraps app.agents.infra.k8s_checker for the multi-agent pipeline."""

    def run(self, state: dict) -> dict:
        """Return a dict of K8s context (merged into state by the graph node)."""
        k8s_cfg   = state.get("metadata", {}).get("k8s_cfg") or {}
        namespace = k8s_cfg.get("namespace", "default")

        try:
            from app.agents.infra.k8s_checker import (
                check_k8s_cluster,
                check_k8s_pods,
                check_k8s_deployments,
            )
            cluster = check_k8s_cluster()
            # check_k8s_cluster returns one of: healthy | degraded | unavailable | error.
            # Only `healthy` and `degraded` mean we actually have a usable cluster.
            # `unavailable` (no KUBECONFIG) used to leak through as `_data_available=True`,
            # making the planner think it could schedule k8s_* actions on a non-existent
            # cluster.
            status = (cluster.get("status") if isinstance(cluster, dict) else "") or ""
            if status in ("error", "unavailable") or (isinstance(cluster, dict) and cluster.get("success") is False):
                return {
                    "_data_available": False,
                    "_reason": cluster.get("details") or cluster.get("error") or "K8s not configured",
                }
            pods    = check_k8s_pods(namespace)
            deploys = check_k8s_deployments(namespace)
            self._log("k8s_context_collected",
                      incident_id=state.get("incident_id", ""),
                      namespace=namespace)
            return {
                "_data_available": True,
                "cluster_summary": cluster,
                "pods":            pods,
                "deployments":     deploys,
                "namespace":       namespace,
            }
        except Exception as exc:
            self._warn("k8s_agent_error", error=str(exc))
            return {"_data_available": False, "_reason": str(exc)}
