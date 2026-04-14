"""Structured JSON logging with per-request trace context.

Changes from original:
  - Added:  incident_id_var, user_var context vars (thread through every log line)
  - Added:  set_context() helper — call once per request/pipeline run
  - Added:  TraceMiddleware — FastAPI middleware that injects trace context from headers
  - Kept:   correlation_id_var, get_logger(), new_correlation_id() (unchanged behaviour)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

# ── Per-request context vars ──────────────────────────────────────────────────
# Set these once at the start of each request/pipeline run; they then appear
# automatically in every log line emitted during that run.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
incident_id_var:    ContextVar[str] = ContextVar("incident_id",    default="")
user_var:           ContextVar[str] = ContextVar("user",           default="")


def set_context(
    correlation_id: str = "",
    incident_id:    str = "",
    user:           str = "",
) -> None:
    """Set the logging context for the current async task / thread."""
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if incident_id:
        incident_id_var.set(incident_id)
    if user:
        user_var.set(user)


def new_correlation_id() -> str:
    """Generate a new correlation ID and set it in context. Returns the ID."""
    cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


# ── JSON formatter ────────────────────────────────────────────────────────────

_NOISE_FIELDS = frozenset({
    "args", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "filename", "module",
    "created", "msecs", "relativeCreated", "thread",
    "threadName", "processName", "process", "taskName",
    "levelno", "pathname", "name", "message",
})


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base: dict[str, Any] = {
            "ts":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg":   record.getMessage(),
        }

        # Inject request-scoped context if set
        if cid := correlation_id_var.get(""):
            base["correlation_id"] = cid
        if iid := incident_id_var.get(""):
            base["incident_id"] = iid
        if usr := user_var.get(""):
            base["user"] = usr

        # Merge any extra kwargs passed to the logger call
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in logging.LogRecord.__dict__
            and k not in _NOISE_FIELDS
            and not k.startswith("_")
        }
        base.update(extra)

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)

        return json.dumps(base, default=str)


# ── Logger factory ────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ── FastAPI middleware ────────────────────────────────────────────────────────

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as _Request

    class TraceMiddleware(BaseHTTPMiddleware):
        """
        Injects trace context into every request's log lines.
        Reads X-Trace-Id from inbound headers; generates one if absent.
        Echoes the trace ID in the response header so clients can correlate logs.
        """

        async def dispatch(self, request: _Request, call_next):
            trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
            request.state.trace_id = trace_id
            set_context(correlation_id=trace_id)

            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            return response

except ImportError:
    # Starlette not installed — middleware simply won't be available.
    # This keeps the module importable in non-web contexts (tests, scripts).
    pass
