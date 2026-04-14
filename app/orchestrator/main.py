"""
NsOps — AI DevOps Platform
Thin FastAPI application entry point.

All route logic lives in app/routes/*.py.
This file wires routers, middleware, and startup tasks only.
"""
import os
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

# ── Import all routers ────────────────────────────────────────────────────────
from app.routes import (
    auth, aws, k8s, security, webhooks, deploy,
    incidents, approvals, warroom, chat, github,
    cost, health, vscode, misc, websocket_routes, tenants, agentic,
)

logger = logging.getLogger("nsops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown tasks."""
    # Start background cleanup loop for expired approvals
    async def _approval_cleanup():
        while True:
            await asyncio.sleep(300)
            try:
                from app.incident.approval import cleanup_expired
                cleanup_expired()
            except Exception:
                pass

    # Start monitor loop if enabled
    _monitor_task = None
    if os.getenv("ENABLE_MONITOR_LOOP", "").lower() in ("1", "true", "yes"):
        async def _monitor_loop():
            from app.orchestrator.monitor import run_monitor_loop
            await asyncio.get_event_loop().run_in_executor(None, run_monitor_loop)

        _monitor_task = asyncio.create_task(_monitor_loop())

    _cleanup_task = asyncio.create_task(_approval_cleanup())

    logger.info("NsOps platform started")
    yield

    _cleanup_task.cancel()
    if _monitor_task:
        _monitor_task.cancel()


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
app.add_middleware(TenantMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (dashboard HTML/CSS/JS) ─────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "../static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


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
