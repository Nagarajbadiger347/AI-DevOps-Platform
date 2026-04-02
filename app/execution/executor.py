"""Executor — safe action execution layer.

Responsibilities:
  1. Receive the plan's action list from PipelineState
  2. Evaluate every action through PolicyEngine
  3. Execute allowed actions via ActionRegistry
  4. Record results (executed / blocked / failed) back into state
  5. Never call agents or the LLM
"""
from __future__ import annotations

from app.execution.action_registry import ACTION_REGISTRY
from app.policies.policy_engine import PolicyEngine
from app.core.logging import get_logger

logger = get_logger(__name__)

_policy = PolicyEngine()


class Executor:
    def run(self, state: dict) -> dict:
        actions   = state.get("plan", {}).get("actions", [])
        user      = state.get("metadata", {}).get("user", "system")
        role      = state.get("metadata", {}).get("role", "viewer")
        incident_id = state.get("incident_id", "unknown")

        state.setdefault("executed_actions", [])
        state.setdefault("blocked_actions",  [])
        state["retry_count"] = state.get("retry_count", 0) + 1

        dry_run = state.get("dry_run", False)

        if not actions:
            logger.info("executor_no_actions", extra={"incident_id": incident_id})
            return state

        if dry_run:
            logger.info("executor_dry_run", extra={"incident_id": incident_id, "action_count": len(actions)})
            for action in actions:
                action_type = action.get("type", "unknown")
                logger.info(
                    "dry_run_would_execute",
                    extra={"incident_id": incident_id, "action": action_type, "params": action},
                )
                state["executed_actions"].append(
                    {**action, "status": "dry_run_skipped",
                     "note": f"dry_run=True — action '{action_type}' was not executed"}
                )
            return state

        evaluations = _policy.evaluate_batch(actions, user=user, role=role)

        for action, allowed, reason in evaluations:
            action_type = action.get("type", "unknown")

            if not allowed:
                logger.warning("action_blocked", extra={"incident_id": incident_id, "action": action_type, "reason": reason, "user": user, "role": role})
                state["blocked_actions"].append({**action, "reason": reason})
                continue

            handler = ACTION_REGISTRY.get(action_type)
            if not handler:
                logger.error("unknown_action_type", extra={"incident_id": incident_id, "action_type": action_type})
                state["executed_actions"].append(
                    {**action, "status": "failed",
                     "error": f"No handler registered for '{action_type}'"}
                )
                continue

            try:
                result = handler(action)
                state["executed_actions"].append(
                    {**action, "status": "ok", "result": result}
                )
                logger.info("action_executed", extra={"incident_id": incident_id, "action": action_type})
            except Exception as exc:
                logger.error("action_failed", extra={
                    "incident_id": incident_id, "action": action_type, "error": str(exc),
                })
                state["executed_actions"].append(
                    {**action, "status": "failed", "error": str(exc)}
                )
                state.setdefault("errors", []).append(
                    f"Executor/{action_type}: {exc}"
                )

        return state
