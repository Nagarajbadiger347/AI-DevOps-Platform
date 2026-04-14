"""Executor — safe action execution layer.

Enforcement order per action:
  1. RBAC        — does this role have the permission required for this action type?
  2. PolicyEngine — is the action blocked/guardrailed by rules.json?
  3. Dry-run gate — log intent only, no real execution
  4. Dispatch     — call the registered handler
  5. Audit log    — record outcome regardless of result (executed/blocked/failed)
"""
from __future__ import annotations

import time
import uuid

from app.core.audit import audit_log
from app.execution.action_registry import ACTION_REGISTRY
from app.policies.policy_engine import PolicyEngine
from app.core.logging import get_logger

logger = get_logger(__name__)

_policy = PolicyEngine()


def _role_has_permission(role: str, action_type: str) -> tuple[bool, str]:
    """
    Check whether the given role holds the permission required for action_type.
    Returns (allowed: bool, reason: str).
    Delegates to PolicyEngine for the required-permission lookup, then checks
    ROLE_PERMISSIONS from security/rbac.py.
    """
    required = _policy.get_required_permission(action_type)
    if required is None:
        # No permission mapping defined → allow (unknown actions are caught later
        # by the handler registry check, not by RBAC).
        return True, ""

    try:
        from app.security.rbac import ROLE_PERMISSIONS
        if required in ROLE_PERMISSIONS.get(role, set()):
            return True, ""
        return False, f"role '{role}' lacks '{required}' permission for '{action_type}'"
    except Exception as exc:
        logger.warning("rbac_check_error", extra={"error": str(exc)})
        return False, f"RBAC check failed: {exc}"


class Executor:
    def run(self, state: dict) -> dict:
        actions     = state.get("plan", {}).get("actions", [])
        # Role resolution: state["role"] takes priority (set by service layer),
        # fall back to metadata for backwards compatibility.
        role        = state.get("role") or state.get("metadata", {}).get("role", "viewer")
        user        = state.get("user") or state.get("metadata", {}).get("user", "system")
        incident_id = state.get("incident_id", "unknown")
        dry_run     = state.get("dry_run", False)

        state.setdefault("executed_actions", [])
        state.setdefault("blocked_actions",  [])
        state.setdefault("errors",           [])
        state["retry_count"] = state.get("retry_count", 0) + 1

        if not actions:
            logger.info("executor_no_actions", extra={"incident_id": incident_id})
            return state

        logger.info("executor_start", extra={
            "incident_id":  incident_id,
            "action_count": len(actions),
            "dry_run":      dry_run,
            "role":         role,
            "user":         user,
        })

        for action in actions:
            action_id   = str(uuid.uuid4())[:8]
            action_type = action.get("type", "unknown")
            t_start     = time.monotonic()

            # ── 1. RBAC enforcement ──────────────────────────────────────────
            rbac_ok, rbac_reason = _role_has_permission(role, action_type)
            if not rbac_ok:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                logger.warning("action_rbac_blocked", extra={
                    "incident_id": incident_id,
                    "action":      action_type,
                    "reason":      rbac_reason,
                    "user":        user,
                    "role":        role,
                    "action_id":   action_id,
                })
                state["blocked_actions"].append({
                    **action,
                    "action_id":   action_id,
                    "reason":      rbac_reason,
                    "blocked_by":  "rbac",
                    "duration_ms": duration_ms,
                })
                audit_log(
                    user=user, action=action_type,
                    params={**action, "action_id": action_id},
                    result={"success": False, "error": rbac_reason},
                    source="executor",
                )
                continue

            # ── 2. Policy engine ─────────────────────────────────────────────
            allowed, policy_reason = _policy.evaluate(action, user=user, role=role)
            if not allowed:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                logger.warning("action_policy_blocked", extra={
                    "incident_id": incident_id,
                    "action":      action_type,
                    "reason":      policy_reason,
                    "user":        user,
                    "role":        role,
                    "action_id":   action_id,
                })
                state["blocked_actions"].append({
                    **action,
                    "action_id":   action_id,
                    "reason":      policy_reason,
                    "blocked_by":  "policy",
                    "duration_ms": duration_ms,
                })
                audit_log(
                    user=user, action=action_type,
                    params={**action, "action_id": action_id},
                    result={"success": False, "error": policy_reason},
                    source="executor",
                )
                continue

            # ── 3. Dry-run gate ──────────────────────────────────────────────
            if dry_run:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                logger.info("action_dry_run", extra={
                    "incident_id": incident_id,
                    "action":      action_type,
                    "action_id":   action_id,
                })
                state["executed_actions"].append({
                    **action,
                    "action_id":   action_id,
                    "status":      "dry_run_skipped",
                    "note":        f"dry_run=True — '{action_type}' was not executed",
                    "duration_ms": duration_ms,
                })
                audit_log(
                    user=user, action=action_type,
                    params={**action, "action_id": action_id},
                    result={"success": True, "dry_run": True},
                    source="executor", dry_run=True,
                )
                continue

            # ── 4. Dispatch ──────────────────────────────────────────────────
            handler = ACTION_REGISTRY.get(action_type)
            if not handler:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                error_msg = f"No handler registered for '{action_type}'"
                logger.error("unknown_action_type", extra={
                    "incident_id": incident_id,
                    "action_type": action_type,
                    "action_id":   action_id,
                })
                state["executed_actions"].append({
                    **action,
                    "action_id":   action_id,
                    "status":      "failed",
                    "error":       error_msg,
                    "duration_ms": duration_ms,
                })
                audit_log(
                    user=user, action=action_type,
                    params={**action, "action_id": action_id},
                    result={"success": False, "error": error_msg},
                    source="executor",
                )
                continue

            try:
                result      = handler(action)
                duration_ms = int((time.monotonic() - t_start) * 1000)
                state["executed_actions"].append({
                    **action,
                    "action_id":   action_id,
                    "status":      "ok",
                    "result":      result,
                    "duration_ms": duration_ms,
                })
                # ── 5. Audit log ─────────────────────────────────────────────
                audit_log(
                    user=user, action=action_type,
                    params={**action, "action_id": action_id},
                    result=result,
                    source="executor",
                )
                logger.info("action_executed", extra={
                    "incident_id": incident_id,
                    "action":      action_type,
                    "action_id":   action_id,
                    "duration_ms": duration_ms,
                    "success":     result.get("success", True),
                })
            except Exception as exc:
                duration_ms = int((time.monotonic() - t_start) * 1000)
                error_msg   = str(exc)
                logger.error("action_failed", extra={
                    "incident_id": incident_id,
                    "action":      action_type,
                    "action_id":   action_id,
                    "error":       error_msg,
                    "duration_ms": duration_ms,
                })
                state["executed_actions"].append({
                    **action,
                    "action_id":   action_id,
                    "status":      "failed",
                    "error":       error_msg,
                    "duration_ms": duration_ms,
                })
                state["errors"].append(f"Executor/{action_type}: {exc}")
                audit_log(
                    user=user, action=action_type,
                    params={**action, "action_id": action_id},
                    result={"success": False, "error": error_msg},
                    source="executor",
                )

        logger.info("executor_done", extra={
            "incident_id": incident_id,
            "executed":    len(state["executed_actions"]),
            "blocked":     len(state["blocked_actions"]),
            "errors":      len(state["errors"]),
        })
        return state
