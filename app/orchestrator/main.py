"""
NsOps — AI DevOps Platform
Thin FastAPI application entry point.

All route logic lives in app/routes/*.py.
This file wires routers, middleware, and startup tasks only.
"""
import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from app.tenants.middleware import TenantMiddleware
from app.core.logging import TraceMiddleware
from app.core.config import settings
from app.core.ratelimit import rate_limit_check

# ── Import all routers ────────────────────────────────────────────────────────
from app.api import (
    auth, aws, k8s, security, webhooks, deploy,
    incidents, approvals, warroom, chat, github,
    cost, health, vscode, misc, websocket_routes, tenants, agentic,
)

logger = logging.getLogger("nsops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and graceful shutdown tasks."""
    # Track active requests (SRE: for graceful shutdown)
    app.state.active_requests = 0
    app.state.shutting_down = False
    
    # Start background cleanup loop for expired approvals
    async def _approval_cleanup():
        while True:
            await asyncio.sleep(300)
            try:
                from app.incident.approval import cleanup_expired
                cleanup_expired()
            except Exception as exc:
                logger.warning("approval_cleanup_error", extra={"error": str(exc)})

    # Start monitor loop if enabled
    _monitor_task = None
    if os.getenv("ENABLE_MONITOR_LOOP", "").lower() in ("1", "true", "yes"):
        async def _monitor_loop():
            from app.orchestrator.monitor import run_monitor_loop
            await asyncio.get_event_loop().run_in_executor(None, run_monitor_loop)

        _monitor_task = asyncio.create_task(_monitor_loop())

    _cleanup_task = asyncio.create_task(_approval_cleanup())

    # ── Database connectivity check ───────────────────────────────
    try:
        from app.core.database import health_check

        if health_check():
            logger.info(
                "database_connected url=%s",
                os.getenv("DATABASE_URL", "postgresql://localhost/nexusops").split("@")[-1]
            )
        else:
            logger.error("database_unreachable — check DATABASE_URL in .env")
    except Exception as exc:
        logger.error("database_startup_check_failed error=%s", exc)

    logger.info("NexusOps platform started — multi-tenant SaaS mode")
    
    yield  # ─────────────────────── App running ──────────────────────
    
    # ── Graceful Shutdown Phase (SRE optimization) ──────────────────
    logger.info("graceful_shutdown_started")
    app.state.shutting_down = True
    
    # Wait for in-flight requests (max 30 seconds)
    shutdown_timeout = 30
    start = time.time()
    while app.state.active_requests > 0 and time.time() - start < shutdown_timeout:
        await asyncio.sleep(0.1)
    
    if app.state.active_requests > 0:
        logger.warning(
            "graceful_shutdown_timeout",
            extra={"active_requests": app.state.active_requests}
        )
    
    # Cancel background tasks
    _cleanup_task.cancel()
    if _monitor_task:
        _monitor_task.cancel()
    
    # Close DB connections
    try:
        from app.core.database import _get_pool
        pool = _get_pool()
        if pool:
            pool.closeall()
            logger.info("database_pool_closed")
    except Exception as e:
        logger.error("database_pool_close_error error=%s", e)
    
    logger.info("graceful_shutdown_complete active_requests=%d", app.state.active_requests)


app = FastAPI(
    title="NsOps — AI DevOps Platform",
    description="Autonomous incident response, AI chat, multi-cloud observability, and approval workflows.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TraceMiddleware)
app.add_middleware(TenantMiddleware)


# ── Active Request Tracking Middleware (SRE: graceful shutdown) ────────────────

from starlette.middleware.base import BaseHTTPMiddleware


class ActiveRequestTracker(BaseHTTPMiddleware):
    """Track active requests for graceful shutdown."""

    async def dispatch(self, request: Request, call_next):
        if not hasattr(app.state, 'active_requests'):
            app.state.active_requests = 0
        
        app.state.active_requests += 1
        try:
            response = await call_next(request)
            return response
        finally:
            app.state.active_requests -= 1


app.add_middleware(ActiveRequestTracker)


# ── Prometheus Metrics Endpoint ────────────────────────────────────────────────
from fastapi.responses import Response
from app.core.metrics import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ── Rate limiting middleware ───────────────────────────────────────────────────

from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-endpoint rate limits using the RateLimiter from core.ratelimit."""

    async def dispatch(self, request: Request, call_next):
        # from app.core.metrics import rate_limit_exceeded_total
        
        # Health and static endpoints are exempt
        path = request.url.path
        if path in ("/health", "/", "/docs", "/redoc", "/openapi.json") or path.startswith("/static"):
            return await call_next(request)

        # Identify the caller: prefer JWT sub, fall back to IP
        identifier = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        tenant_id = request.headers.get("X-Tenant-ID", "unknown")
        
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from app.core.auth import decode_token
                payload = decode_token(auth_header[7:])
                identifier = payload.get("sub", identifier)
                tenant_id = payload.get("tenant_id", tenant_id)
            except Exception:
                pass

        allowed, remaining = rate_limit_check(identifier, path)
        if not allowed:
            # SRE: Track rate limit violations per tenant
            # rate_limit_exceeded_total.labels(tenant_id=tenant_id, endpoint=path).inc()
            logger.warning(
                "rate_limit_exceeded",
                extra={"identifier": identifier, "path": path, "tenant_id": tenant_id}
            )
            from app.core.exceptions import RateLimitExceeded
            raise RateLimitExceeded(retry_after=60)

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


app.add_middleware(RateLimitMiddleware)
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Trace-Id", "X-Tenant-Id", "X-User"],
)

# ── Static files (dashboard HTML/CSS/JS) ─────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "../static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Exception Handlers (SRE: structured error responses) ──────────────────────

from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch uncaught exceptions and return structured error response."""
    from app.core.exceptions import APIError, InternalServerError
    import uuid
    
    if isinstance(exc, APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )
    
    # Unexpected error
    error_id = str(uuid.uuid4())
    logger.error(
        "unhandled_exception",
        extra={"error_id": error_id, "error": str(exc)},
        exc_info=True
    )
    
    api_error = InternalServerError(error_id)
    return JSONResponse(
        status_code=api_error.status_code,
        content=api_error.to_dict(),
    )


# ── Dashboard HTML (served from static/index.html or inline fallback) ────────
_DASHBOARD_HTML: bytes | None = None
_DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "../static/index.html")


def _load_dashboard() -> bytes | None:
    # Always reload from disk (no-cache mode — browser gets fresh HTML every time)
    if os.path.isfile(_DASHBOARD_PATH):
        with open(_DASHBOARD_PATH, "rb") as f:
            return f.read()
    return None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request = None):
    html = _load_dashboard()
    if html:
        return HTMLResponse(
            content=html,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff",
            },
        )
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")


# ── Prometheus Metrics Endpoint ───────────────────────────────────────────────
# @app.get("/metrics", include_in_schema=False)
# async def metrics():
#     """Prometheus metrics endpoint for scraping."""
#     from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
#     from prometheus_client import REGISTRY
#     return generate_latest(REGISTRY)


# ── Register all routers ──────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(aws.router)
app.include_router(k8s.router)
app.include_router(security.router)
app.include_router(webhooks.router)
app.include_router(deploy.router)
app.include_router(incidents.router)
app.include_router(approvals.router)
app.include_router(warroom.router)
app.include_router(chat.router)
app.include_router(github.router)
app.include_router(cost.router)
app.include_router(vscode.router)
app.include_router(misc.router)
app.include_router(websocket_routes.router)
app.include_router(tenants.router)
app.include_router(agentic.router)
