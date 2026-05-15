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
from app.core.config import settings, validate_security
from app.core.ratelimit import rate_limit_check

# Production refuses to boot when webhook secrets / JWT_SECRET_KEY are missing.
validate_security()

# ── Import all routers ────────────────────────────────────────────────────────
# `deploy` and `saas` modules are intentionally NOT imported — every endpoint
# they exposed was orphaned (no UI caller, no external integration). Re-add
# them here if/when a UI flow needs them.
from app.api import (
    auth, aws, k8s, security, webhooks,
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

    # Run DB migrations on startup (idempotent)
    try:
        from app.core.schema import apply_migrations as _apply_migrations
        _apply_migrations()
    except Exception as exc:
        logger.warning("db_migrations_failed error=%s", exc)
    
    # Start background cleanup loop for expired approvals
    async def _approval_cleanup():
        while True:
            await asyncio.sleep(300)
            try:
                from app.incident.approval import cleanup_expired
                cleanup_expired()
            except Exception as exc:
                logger.warning("approval_cleanup_error", extra={"error": str(exc)})

    # Start monitor loop if enabled (reads from settings, not raw env)
    _monitor_task = None
    if settings.ENABLE_MONITOR_LOOP:
        from app.monitoring.loop import monitoring_loop as _monitoring_loop

        async def _run_monitor():
            try:
                await _monitoring_loop()
            except Exception as exc:
                logger.error("monitor_loop_crashed error=%s", exc, exc_info=True)

        _monitor_task = asyncio.create_task(_run_monitor())
        logger.info(
            "monitor_loop_enabled interval=%ds auto_remediate=%s",
            settings.MONITOR_INTERVAL_SECONDS,
            settings.AUTO_REMEDIATE_ON_MONITOR,
        )

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
    """Track active requests and emit SLO metrics (latency + status)."""

    async def dispatch(self, request: Request, call_next):
        import time
        from app.core.metrics import http_requests_total, http_request_duration_seconds

        if not hasattr(app.state, 'active_requests'):
            app.state.active_requests = 0

        # Normalise endpoint label: strip path params to avoid high cardinality
        path = request.url.path
        # collapse UUIDs and numeric IDs to placeholders
        import re
        path_label = re.sub(r"/[0-9a-f]{8}-[0-9a-f-]{27}", "/{id}", path)
        path_label = re.sub(r"/\d+", "/{id}", path_label)

        app.state.active_requests += 1
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            app.state.active_requests -= 1
            # Skip metrics / health noise
            if path not in ("/metrics", "/health", "/favicon.ico"):
                http_requests_total.labels(
                    method=request.method, endpoint=path_label, status=status
                ).inc()
                http_request_duration_seconds.labels(
                    method=request.method, endpoint=path_label
                ).observe(duration)


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
# Unversioned root routes — kept for Docker/k8s probes and websocket clients.
app.include_router(health.router)
app.include_router(websocket_routes.router)

# ── /v1 — all product API routes ─────────────────────────────────────────────
V1 = "/v1"

# Health is also exposed under /v1 so the dashboard can call it through the
# same versioned base URL as every other product endpoint.
app.include_router(health.router,    prefix=V1)

# Auth + users
app.include_router(auth.router,      prefix=V1)

# Infra actions
app.include_router(aws.router,       prefix=V1)
app.include_router(k8s.router,       prefix=V1)
app.include_router(github.router,    prefix=V1)
app.include_router(cost.router,      prefix=V1)
app.include_router(vscode.router,    prefix=V1)
app.include_router(agentic.router,   prefix=V1)

# Incidents + approvals + war room
app.include_router(incidents.router, prefix=V1)
app.include_router(approvals.router, prefix=V1 + "/approvals")
app.include_router(warroom.router,   prefix=V1)

# Chat / AI
app.include_router(chat.router,      prefix=V1)

# Platform admin
app.include_router(security.router,  prefix=V1)
app.include_router(tenants.router,   prefix=V1)
app.include_router(misc.router,      prefix=V1)

# Webhooks — external callers (PagerDuty, GitHub, Grafana, etc.) hit /v1/webhooks/*
app.include_router(webhooks.router,  prefix=V1)
