"""Structured JSON logging with per-request trace context.

Every log line automatically carries: ts, level, logger, msg,
plus any of correlation_id / incident_id / user / trace_id that are set
for the current async task or thread.

Usage:
    from app.core.logging import get_logger, set_context, TraceMiddleware

    logger = get_logger(__name__)
    set_context(trace_id="abc", incident_id="INC-001", user="alice")
    logger.info("something happened", extra={"key": "value"})
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

# ── Per-request context vars ──────────────────────────────────────────────────
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
incident_id_var:    ContextVar[str] = ContextVar("incident_id",    default="")
user_var:           ContextVar[str] = ContextVar("user",           default="")
trace_id_var:       ContextVar[str] = ContextVar("trace_id",       default="")
tenant_id_var:      ContextVar[str] = ContextVar("tenant_id",      default="")


def set_context(
    correlation_id: str = "",
    incident_id:    str = "",
    user:           str = "",
    trace_id:       str = "",
    tenant_id:      str = "",
) -> None:
    """Set logging context for the current async task / thread.

    Safe to call with partial arguments — only non-empty strings are set.
    """
    if correlation_id: correlation_id_var.set(correlation_id)
    if incident_id:    incident_id_var.set(incident_id)
    if user:           user_var.set(user)
    if trace_id:       trace_id_var.set(trace_id)
    if tenant_id:      tenant_id_var.set(tenant_id)


def clear_context() -> None:
    """Reset all context vars to their defaults (useful in test teardowns)."""
    correlation_id_var.set("")
    incident_id_var.set("")
    user_var.set("")
    trace_id_var.set("")
    tenant_id_var.set("")


def new_correlation_id() -> str:
    """Generate a new correlation ID, set it in context, and return it."""
    cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


def new_trace_id() -> str:
    """Generate a new trace ID, set it in context, and return it."""
    tid = str(uuid.uuid4())
    trace_id_var.set(tid)
    return tid


# ── JSON formatter ────────────────────────────────────────────────────────────

_NOISE_FIELDS = frozenset({
    # Standard LogRecord instance attributes — never re-emit these
    "args", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "filename", "module",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "taskName",
    "levelno", "levelname", "pathname", "name", "message",
    # Already mapped to our own keys
    "msg", "levelname",
})


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base: dict[str, Any] = {
            "ts":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Inject all non-empty context vars
        if cid := correlation_id_var.get(""):
            base["correlation_id"] = cid
        if iid := incident_id_var.get(""):
            base["incident_id"] = iid
        if usr := user_var.get(""):
            base["user"] = usr
        if tid := trace_id_var.get(""):
            base["trace_id"] = tid
        if ten := tenant_id_var.get(""):
            base["tenant_id"] = ten

        # Merge extra kwargs from the logger call (fields added via extra={...})
        # Use record.__dict__ but exclude all standard LogRecord instance fields
        # by name — checking against LogRecord.__dict__ (class dict) is wrong
        # because instance fields like levelname/msg/args are NOT in the class dict.
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in _NOISE_FIELDS
            and not k.startswith("_")
        }
        base.update(extra)

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)

        return json.dumps(base, default=str)


# ── Logger factory ────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a JSON-structured logger. Idempotent — safe to call multiple times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ── FastAPI / Starlette middleware ────────────────────────────────────────────

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as _Request

    class TraceMiddleware(BaseHTTPMiddleware):
        """Injects a trace_id into every request's log context.

        - Reads  X-Trace-Id from inbound request headers.
        - Generates a UUID if the header is absent.
        - Echoes the trace ID in the response header for client-side correlation.
        - Also extracts X-Tenant-Id and X-User if present.
        """

        async def dispatch(self, request: _Request, call_next):
            trace_id  = request.headers.get("X-Trace-Id")  or str(uuid.uuid4())
            tenant_id = request.headers.get("X-Tenant-Id") or ""
            user      = request.headers.get("X-User")      or ""

            request.state.trace_id = trace_id
            set_context(
                correlation_id=trace_id,
                trace_id=trace_id,
                tenant_id=tenant_id,
                user=user,
            )

            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            return response

except ImportError:
    # Non-web contexts (tests, CLI scripts) — middleware unavailable but module
    # remains importable.
    pass
