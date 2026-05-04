"""Request-scoped context vars — available anywhere in the call stack without passing args.

Set by TenantMiddleware at the start of every request.
"""
from __future__ import annotations
import contextvars

_tenant_id_var:    contextvars.ContextVar[str] = contextvars.ContextVar("tenant_id",    default="")
_workspace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("workspace_id", default="")
_plan_name_var:    contextvars.ContextVar[str] = contextvars.ContextVar("plan_name",    default="starter")


def set_current_tenant(tenant_id: str, workspace_id: str = "", plan_name: str = "starter") -> None:
    _tenant_id_var.set(tenant_id)
    _workspace_id_var.set(workspace_id)
    _plan_name_var.set(plan_name)


def get_current_tenant() -> str:
    return _tenant_id_var.get()


def get_current_workspace() -> str:
    return _workspace_id_var.get()


def get_current_plan() -> str:
    return _plan_name_var.get()
