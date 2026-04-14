"""Validator — post-execution health verification.

After actions run, the Validator re-checks infrastructure state to confirm
the incident has been resolved. A failed check signals the graph to retry
(up to max retries) or escalate.

Checks performed:
  1. Any action that hard-failed during execution → fail immediately
  2. K8s actions → re-query for unhealthy pods
  3. (Extendable: add AWS health checks, SLO checks, etc.)
"""
from __future__ import annotations

from app.core.logging import get_logger, trace_id_var

logger = get_logger(__name__)


class Validator:

    def run(self, state: dict) -> dict:
        executed    = state.get("executed_actions", [])
        incident_id = state.get("incident_id", "unknown")
        trace_id    = state.get("trace_id") or trace_id_var.get("")

        # Nothing was executed — nothing to verify
        if not executed:
            state["validation_passed"] = True
            state["validation_detail"] = {"reason": "no actions executed"}
            logger.info("validation_skipped_no_actions", extra={
                "incident_id": incident_id, "trace_id": trace_id,
            })
            return state

        # Skip dry-run-only runs — all actions were simulated
        real_actions = [a for a in executed if a.get("status") != "dry_run_skipped"]
        if not real_actions:
            state["validation_passed"] = True
            state["validation_detail"] = {"reason": "dry_run — no real actions to verify"}
            logger.info("validation_skipped_dry_run", extra={
                "incident_id": incident_id, "trace_id": trace_id,
            })
            return state

        # Check for hard failures
        failed_actions = [a for a in real_actions if a.get("status") == "failed"]
        if failed_actions:
            detail = {
                "failed_actions": [
                    {"type": a.get("type"), "error": a.get("error"), "action_id": a.get("action_id")}
                    for a in failed_actions
                ]
            }
            state["validation_passed"] = False
            state["validation_detail"] = detail
            logger.warning("validation_failed_action_errors", extra={
                "incident_id":  incident_id,
                "trace_id":     trace_id,
                "failed_count": len(failed_actions),
                "actions":      [a.get("type") for a in failed_actions],
            })
            return state

        # K8s-specific health re-check
        action_types = {a.get("type", "") for a in real_actions}
        if any(t.startswith("k8s_") for t in action_types):
            k8s_result = self._check_k8s_health(state)
            if not k8s_result["healthy"]:
                state["validation_passed"] = False
                state["validation_detail"] = k8s_result
                logger.warning("validation_failed_k8s_unhealthy", extra={
                    "incident_id": incident_id,
                    "trace_id":    trace_id,
                    "detail":      k8s_result,
                })
                return state

        state["validation_passed"] = True
        state["validation_detail"] = {"reason": "all checks passed"}
        logger.info("validation_passed", extra={
            "incident_id":   incident_id,
            "trace_id":      trace_id,
            "actions_checked": len(real_actions),
        })
        return state

    @staticmethod
    def _check_k8s_health(state: dict) -> dict:
        """Re-query K8s for unhealthy pods as a post-execution health signal."""
        try:
            from app.agents.infra.k8s_agent import K8sAgent
            fresh    = K8sAgent().run(state)
            pods_raw = fresh.get("pods", {})

            pod_list: list = []
            if isinstance(pods_raw, list):
                pod_list = pods_raw
            elif isinstance(pods_raw, dict):
                pod_list = pods_raw.get("pods", [])

            unhealthy = [
                p for p in pod_list
                if isinstance(p, dict)
                and p.get("status") not in ("Running", "Completed", "Succeeded")
            ]
            return {
                "healthy":        len(unhealthy) == 0,
                "unhealthy_pods": unhealthy[:10],
            }
        except Exception as exc:
            logger.warning("validator_k8s_check_failed", extra={
                "error":    str(exc),
                "trace_id": trace_id_var.get(""),
            })
            return {
                "healthy": False,
                "reason":  "K8s unreachable — cannot verify state",
                "error":   str(exc),
            }
