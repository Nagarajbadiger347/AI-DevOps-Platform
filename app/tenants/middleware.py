"""
Tenant middleware — extracts X-Tenant-ID header and makes tenant context
available to request handlers via request.state.tenant_id.

Also sets contextvars (app.core.context) so any downstream code
(agents, LLM calls, metering) can read the current tenant without
thread-unsafe globals or parameter drilling.

Usage:
    app.add_middleware(TenantMiddleware)

In a route:
    request.state.tenant_id  # e.g. "acme", "default"
    request.state.tenant     # Tenant | None
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_DEFAULT_TENANT = "default"


class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant_id = (
            request.headers.get("X-Tenant-ID")
            or request.query_params.get("tenant_id")
            or _DEFAULT_TENANT
        )
        request.state.tenant_id = tenant_id

        tenant       = None
        workspace_id = ""
        plan_name    = "starter"
        try:
            from app.tenants.store import get_tenant
            tenant = get_tenant(tenant_id)
            if tenant:
                workspace_id = tenant.workspace_id or ""
                plan_name    = tenant.plan_name or "starter"
        except Exception:
            pass
        request.state.tenant = tenant

        # Set context vars so any downstream code can read tenant without arg passing
        from app.core.context import set_current_tenant
        set_current_tenant(tenant_id, workspace_id, plan_name)

        response = await call_next(request)
        response.headers["X-Tenant-ID"] = tenant_id
        return response
