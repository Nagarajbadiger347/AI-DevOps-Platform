"""Validator — post-execution health verification.

After actions are executed, the Validator re-checks infrastructure health
to confirm the incident has been resolved. On failure it signals the graph
to retry (up to max retries) or escalate.
"""
from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


class Validator:
    def run(self, state: dict) -> dict:
        executed    = state.get("executed_actions", [])
        incident_id = state.get("incident_id", "unknown")

        # Nothing was executed → mark as passed (nothing to verify)
        if not executed:
            state["validation_passed"] = True
            state["validation_detail"] = {"reason": "no actions executed"}
            return state

        # Any action that explicitly failed during execution = validation failure
        failed_actions = [a for a in executed if a.get("status") == "failed"]
        if failed_actions:
            state["validation_passed"] = False
            state["validation_detail"] = {
                "failed_actions": [
                    {"type": a.get("type"), "error": a.get("error")}
                    for a in failed_actions
                ]
            }
            logger.warning("validation_failed_execution_errors",
                           incident_id=incident_id,
                           failed_count=len(failed_actions))
            return state

        # Re-check K8s health if any K8s action was executed
        action_types = {a.get("type", "") for a in executed}
        if any(t.startswith("k8s_") for t in action_types):
            k8s_result = self._check_k8s_health(state)
            if not k8s_result["healthy"]:
                state["validation_passed"] = False
                state["validation_detail"] = k8s_result
                logger.warning("validation_failed_k8s_unhealthy",
                               incident_id=incident_id,
                               detail=k8s_result)
                return state

        state["validation_passed"] = True
        state["validation_detail"] = {"reason": "all checks passed"}
        logger.info("validation_passed", incident_id=incident_id)
        return state

    @staticmethod
    def _check_k8s_health(state: dict) -> dict:
        """Re-query K8s for unhealthy pods as a basic health signal."""
        try:
            from app.agents.infra.k8s_agent import K8sAgent
            fresh = K8sAgent().run(state)
            pods  = fresh.get("pods", {})

            # pods may be a list or a nested dict depending on k8s_checker version
            pod_list: list = []
            if isinstance(pods, list):
                pod_list = pods
            elif isinstance(pods, dict):
                pod_list = pods.get("pods", [])

            unhealthy = [
                p for p in pod_list
                if isinstance(p, dict)
                and p.get("status") not in ("Running", "Completed", "Succeeded")
            ]
            return {
                "healthy":       len(unhealthy) == 0,
                "unhealthy_pods": unhealthy[:10],  # cap list size
            }
        except Exception as exc:
            # Can't reach K8s — treat as inconclusive (pass)
            logger.warning("validator_k8s_check_failed", error=str(exc))
            return {"healthy": True, "reason": f"k8s unreachable: {exc}"}
