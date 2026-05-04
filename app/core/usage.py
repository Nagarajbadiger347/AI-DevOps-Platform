"""Usage metering helpers — called from chat, agent, and LLM layers.

Wraps record_usage() from tenants.store with fire-and-forget threading
so metering never blocks the request path.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _fire(fn, *args, **kwargs) -> None:
    """Run fn(*args, **kwargs) in a daemon thread — never raises."""
    def _run():
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("usage_meter_bg_error error=%s", exc)
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def meter_llm_tokens(tenant_id: str, tokens: int, workspace_id: str = "",
                     model: str = "", provider: str = "") -> None:
    """Record LLM token usage for the tenant."""
    if not tenant_id or tenant_id == "default":
        return
    _fire(_record, tenant_id, "llm_tokens", tokens, workspace_id,
          {"model": model, "provider": provider})


def meter_agent_run(tenant_id: str, agent_name: str = "", workspace_id: str = "") -> None:
    """Record one agent run for the tenant."""
    if not tenant_id or tenant_id == "default":
        return
    _fire(_record, tenant_id, "agent_run", 1, workspace_id, {"agent": agent_name})


def meter_action(tenant_id: str, action: str = "", workspace_id: str = "") -> None:
    """Record one chat/pipeline action execution."""
    if not tenant_id or tenant_id == "default":
        return
    _fire(_record, tenant_id, "action_executed", 1, workspace_id, {"action": action})


def meter_api_call(tenant_id: str, endpoint: str = "", workspace_id: str = "") -> None:
    """Record one API call (called from HTTP middleware)."""
    if not tenant_id or tenant_id == "default":
        return
    _fire(_record, tenant_id, "api_call", 1, workspace_id, {"endpoint": endpoint})


def _record(tenant_id: str, event_type: str, quantity: int,
            workspace_id: str, meta: dict) -> None:
    from app.tenants.store import record_usage
    record_usage(tenant_id, event_type, quantity, workspace_id, meta)
