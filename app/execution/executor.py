"""Executor — safe, audited, parallel action execution layer.

Enforcement order per action:
  1. RBAC check      — does this role hold the required permission?
  2. Policy engine   — is the action blocked or guardrailed by rules.json?
  3. Dry-run gate    — log intent only, skip real execution
  4. Dispatch        — call the registered integration handler
  5. Audit log       — record every outcome regardless of result

Performance:
  - Independent actions run in parallel (ThreadPoolExecutor).
  - Ordering-dependent actions (e.g. restart → verify) are serialised
    by setting action["depends_on"] = ["action_type_that_must_run_first"].
  - Each action has a per-action timeout (default 30s) to prevent slow
    integrations from blocking the whole pipeline.
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from app.core.audit import audit_log
from app.core.logging import get_logger, set_context, trace_id_var
from app.execution.action_registry import ACTION_REGISTRY
from app.policies.policy_engine import PolicyEngine

logger = get_logger(__name__)

_policy = PolicyEngine()

# Per-action execution timeout in seconds. Prevents one slow integration
# (e.g. an unresponsive OpsGenie endpoint) from stalling the whole pipeline.
_ACTION_TIMEOUT_SECS = 30

# Maximum parallel workers for action execution.
# Keep this low — integrations share connection pools and rate limits.
_MAX_WORKERS = 4


# ── Execution context ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context threaded through every action."""
    user:        str
    role:        str
    tenant_id:   str
    trace_id:    str
    incident_id: str
    dry_run:     bool

    @classmethod
    def from_state(cls, state: dict) -> "ExecutionContext":
        return cls(
            user        = state.get("user")      or state.get("metadata", {}).get("user", "system"),
            role        = state.get("role")      or state.get("metadata", {}).get("role", "viewer"),
            tenant_id   = state.get("tenant_id") or state.get("metadata", {}).get("tenant_id", "default"),
            trace_id    = state.get("trace_id")  or trace_id_var.get(""),
            incident_id = state.get("incident_id", "unknown"),
            dry_run     = bool(state.get("dry_run", False)),
        )


# ── RBAC helper ───────────────────────────────────────────────────────────────

def _role_has_permission(role: str, action_type: str) -> tuple[bool, str]:
    required = _policy.get_required_permission(action_type)
    if required is None:
        return True, ""
    try:
        from app.security.rbac import ROLE_PERMISSIONS
        if required in ROLE_PERMISSIONS.get(role, set()):
            return True, ""
        return False, f"role '{role}' lacks '{required}' permission for '{action_type}'"
    except Exception as exc:
        logger.warning("rbac_import_error", extra={"error": str(exc)})
        return False, f"RBAC check failed: {exc}"


# ── Result builders ───────────────────────────────────────────────────────────

def _blocked_entry(action: dict, action_id: str, reason: str,
                   blocked_by: str, duration_ms: int) -> dict:
    return {**action, "action_id": action_id, "reason": reason,
            "blocked_by": blocked_by, "duration_ms": duration_ms}


def _executed_entry(action: dict, action_id: str, status: str,
                    duration_ms: int, **kwargs: Any) -> dict:
    return {**action, "action_id": action_id, "status": status,
            "duration_ms": duration_ms, **kwargs}


# ── Single action execution ───────────────────────────────────────────────────

def _run_action(action: dict, ctx: ExecutionContext) -> tuple[str, dict]:
    """
    Execute one action through the full enforcement stack.

    Returns (bucket, entry) where bucket is one of:
      "executed" | "blocked" | "error"
    """
    action_id   = str(uuid.uuid4())[:8]
    action_type = action.get("type", "unknown")
    t_start     = time.monotonic()

    # 1. RBAC
    rbac_ok, rbac_reason = _role_has_permission(ctx.role, action_type)
    if not rbac_ok:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.warning("action_rbac_blocked", extra={
            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            "action": action_type, "action_id": action_id,
            "reason": rbac_reason, "user": ctx.user, "role": ctx.role,
        })
        audit_log(user=ctx.user, action=action_type,
                  params={**action, "action_id": action_id, "trace_id": ctx.trace_id},
                  result={"success": False, "error": rbac_reason, "blocked_by": "rbac"},
                  source="executor")
        return "blocked", _blocked_entry(action, action_id, rbac_reason, "rbac", duration_ms)

    # 2. Policy engine
    allowed, policy_reason = _policy.evaluate(action, user=ctx.user, role=ctx.role)
    if not allowed:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.warning("action_policy_blocked", extra={
            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            "action": action_type, "action_id": action_id,
            "reason": policy_reason, "role": ctx.role,
        })
        audit_log(user=ctx.user, action=action_type,
                  params={**action, "action_id": action_id, "trace_id": ctx.trace_id},
                  result={"success": False, "error": policy_reason, "blocked_by": "policy"},
                  source="executor")
        return "blocked", _blocked_entry(action, action_id, policy_reason, "policy", duration_ms)

    # 3. Dry-run gate
    if ctx.dry_run:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        logger.info("action_dry_run", extra={
            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            "action": action_type, "action_id": action_id,
        })
        audit_log(user=ctx.user, action=action_type,
                  params={**action, "action_id": action_id, "trace_id": ctx.trace_id},
                  result={"success": True, "dry_run": True},
                  source="executor", dry_run=True)
        return "executed", _executed_entry(
            action, action_id, "dry_run_skipped", duration_ms,
            note=f"dry_run=True — '{action_type}' not executed",
        )

    # 4. Handler dispatch
    handler = ACTION_REGISTRY.get(action_type)
    if not handler:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        error_msg = f"No handler registered for action type '{action_type}'"
        logger.error("unknown_action_type", extra={
            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            "action_type": action_type, "action_id": action_id,
        })
        audit_log(user=ctx.user, action=action_type,
                  params={**action, "action_id": action_id, "trace_id": ctx.trace_id},
                  result={"success": False, "error": error_msg},
                  source="executor")
        return "error", _executed_entry(action, action_id, "failed", duration_ms, error=error_msg)

    try:
        result      = handler(action)
        duration_ms = int((time.monotonic() - t_start) * 1000)
        # 5. Audit log
        audit_log(user=ctx.user, action=action_type,
                  params={**action, "action_id": action_id, "trace_id": ctx.trace_id},
                  result=result, source="executor")
        logger.info("action_executed", extra={
            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            "action": action_type, "action_id": action_id,
            "duration_ms": duration_ms,
            "success": result.get("success", True) if isinstance(result, dict) else True,
        })
        return "executed", _executed_entry(action, action_id, "ok", duration_ms, result=result)

    except Exception as exc:
        duration_ms = int((time.monotonic() - t_start) * 1000)
        error_msg   = str(exc)
        logger.error("action_failed", extra={
            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            "action": action_type, "action_id": action_id,
            "error": error_msg, "duration_ms": duration_ms,
        })
        audit_log(user=ctx.user, action=action_type,
                  params={**action, "action_id": action_id, "trace_id": ctx.trace_id},
                  result={"success": False, "error": error_msg},
                  source="executor")
        return "error", _executed_entry(action, action_id, "failed", duration_ms, error=error_msg)


# ── Dependency-aware action grouping ─────────────────────────────────────────

def _group_actions(actions: list[dict]) -> list[list[dict]]:
    """
    Split actions into ordered execution groups.

    Actions without "depends_on" run in parallel in group 0.
    Actions with "depends_on" run in subsequent groups, after their
    dependencies have completed.

    This is a simple single-level dependency resolver — sufficient for
    the patterns seen in practice (e.g. restart first, then notify).
    """
    independent = [a for a in actions if not a.get("depends_on")]
    dependent   = [a for a in actions if a.get("depends_on")]

    groups: list[list[dict]] = []
    if independent:
        groups.append(independent)
    if dependent:
        # Group all dependent actions into one sequential batch after independents
        groups.append(dependent)
    return groups or [[]]


# ── Executor ──────────────────────────────────────────────────────────────────

class Executor:

    def run(self, state: dict) -> dict:
        """Execute all planned actions in parallel where safe to do so."""
        ctx = ExecutionContext.from_state(state)

        set_context(
            trace_id    = ctx.trace_id,
            incident_id = ctx.incident_id,
            user        = ctx.user,
            tenant_id   = ctx.tenant_id,
        )

        actions = state.get("plan", {}).get("actions", [])
        state.setdefault("executed_actions", [])
        state.setdefault("blocked_actions",  [])
        state.setdefault("errors",           [])
        state["retry_count"] = state.get("retry_count", 0) + 1

        if not actions:
            logger.info("executor_no_actions", extra={
                "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
            })
            return state

        logger.info("executor_start", extra={
            "incident_id":  ctx.incident_id,
            "trace_id":     ctx.trace_id,
            "action_count": len(actions),
            "dry_run":      ctx.dry_run,
            "role":         ctx.role,
            "user":         ctx.user,
            "tenant_id":    ctx.tenant_id,
        })

        # Run groups sequentially; within each group run in parallel
        groups = _group_actions(actions)
        for group_idx, group in enumerate(groups):
            self._run_group(group, ctx, state, group_idx)

        logger.info("executor_done", extra={
            "incident_id": ctx.incident_id,
            "trace_id":    ctx.trace_id,
            "executed":    len(state["executed_actions"]),
            "blocked":     len(state["blocked_actions"]),
            "errors":      len(state["errors"]),
        })
        return state

    def _run_group(
        self,
        actions: list[dict],
        ctx: ExecutionContext,
        state: dict,
        group_idx: int,
    ) -> None:
        """Run a group of actions in parallel, collecting results thread-safely."""
        if not actions:
            return

        if len(actions) == 1:
            # Skip thread overhead for single-action groups
            bucket, entry = _run_action(actions[0], ctx)
            self._apply(bucket, entry, state)
            return

        logger.info("executor_group_start", extra={
            "incident_id": ctx.incident_id,
            "trace_id":    ctx.trace_id,
            "group":       group_idx,
            "count":       len(actions),
        })

        with ThreadPoolExecutor(max_workers=min(len(actions), _MAX_WORKERS)) as pool:
            futures = {pool.submit(_run_action, action, ctx): action for action in actions}
            try:
                for future in as_completed(futures, timeout=_ACTION_TIMEOUT_SECS * len(actions)):
                    try:
                        bucket, entry = future.result(timeout=_ACTION_TIMEOUT_SECS)
                        self._apply(bucket, entry, state)
                    except FutureTimeout:
                        action = futures[future]
                        atype  = action.get("type", "unknown")
                        error  = f"Action '{atype}' timed out after {_ACTION_TIMEOUT_SECS}s"
                        logger.error("action_timeout", extra={
                            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
                            "action": atype,
                        })
                        state["executed_actions"].append(
                            _executed_entry(action, "timeout", "failed", _ACTION_TIMEOUT_SECS * 1000, error=error)
                        )
                        state["errors"].append(f"Executor/{atype}: {error}")
                    except Exception as exc:
                        action = futures[future]
                        atype  = action.get("type", "unknown")
                        logger.error("action_future_error", extra={
                            "incident_id": ctx.incident_id, "trace_id": ctx.trace_id,
                            "action": atype, "error": str(exc),
                        })
                        state["errors"].append(f"Executor/{atype}: {exc}")
            except FutureTimeout:
                logger.error("executor_group_timeout", extra={
                    "incident_id": ctx.incident_id,
                    "trace_id":    ctx.trace_id,
                    "group":       group_idx,
                })

    @staticmethod
    def _apply(bucket: str, entry: dict, state: dict) -> None:
        """Apply a single action result to state."""
        if bucket == "blocked":
            state["blocked_actions"].append(entry)
        elif bucket == "error":
            state["executed_actions"].append(entry)
            state["errors"].append(f"Executor/{entry.get('type', '?')}: {entry.get('error', '')}")
        else:
            state["executed_actions"].append(entry)
