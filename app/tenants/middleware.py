"""
Tenant middleware — extracts X-Tenant-ID header and makes tenant context
available to request handlers via request.state.tenant_id.

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

        # Lazy-load tenant record (avoids import cycle at module load)
        try:
            from app.tenants.store import get_tenant
            request.state.tenant = get_tenant(tenant_id)
        except Exception:
            request.state.tenant = None

        response = await call_next(request)
        response.headers["X-Tenant-ID"] = tenant_id
        return response
