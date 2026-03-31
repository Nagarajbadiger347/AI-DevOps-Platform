"""Structured JSON logging with per-request correlation IDs."""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

# Set this at the start of each request / pipeline run
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = correlation_id_var.get("")
        if cid:
            base["correlation_id"] = cid
        # Merge any extra kwargs passed to the logger call
        extra = {k: v for k, v in record.__dict__.items()
                 if k not in logging.LogRecord.__dict__ and not k.startswith("_")}
        # Remove standard LogRecord noise
        for noise in ("args", "exc_info", "exc_text", "stack_info",
                      "lineno", "funcName", "filename", "module",
                      "created", "msecs", "relativeCreated", "thread",
                      "threadName", "processName", "process", "taskName",
                      "levelno", "pathname", "name", "message"):
            extra.pop(noise, None)
        base.update(extra)
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def new_correlation_id() -> str:
    cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid
