"""Audit log — records every mutating action with who, what, when, result.

Writes to:
  1. Structured JSON log file: logs/audit.jsonl  (one JSON object per line)
  2. Structured logger (stdout) for log aggregators (Datadog, CloudWatch Logs, etc.)

Usage:
    from app.core.audit import audit_log
    audit_log(user="alice", action="restart_deployment",
              params={"namespace": "prod", "deployment": "payment"},
              result={"success": True}, source="chat")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger("audit")

_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "audit.jsonl"


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(exist_ok=True)


def audit_log(
    user: str,
    action: str,
    params: dict,
    result: dict,
    source: str = "chat",          # "chat" | "api" | "monitor" | "webhook"
    dry_run: bool = False,
) -> None:
    """Write one audit record.  Never raises — audit failures must not break the app."""
    try:
        _ensure_log_dir()
        record = {
            "ts":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user":     user or "unknown",
            "action":   action,
            "params":   params,
            "success":  result.get("success", False) if isinstance(result, dict) else True,
            "error":    result.get("error")           if isinstance(result, dict) else None,
            "source":   source,
            "dry_run":  dry_run,
        }
        # Append to JSONL file (primary store — do this first)
        with _LOG_FILE.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        # Log line for aggregators (standard Python logger format)
        logger.info("audit_action user=%s action=%s success=%s source=%s",
                    record["user"], record["action"], record["success"], record["source"])
    except Exception:
        pass  # audit must never crash the main flow


def get_audit_log(limit: int = 100, user: str = "", action: str = "") -> list[dict]:
    """Read recent audit entries, newest first. Optionally filter by user or action."""
    try:
        _ensure_log_dir()
        if not _LOG_FILE.exists():
            return []
        lines = _LOG_FILE.read_text().strip().splitlines()
        records = []
        for line in reversed(lines):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if user   and r.get("user")   != user:   continue
            if action and r.get("action") != action: continue
            records.append(r)
            if len(records) >= limit:
                break
        return records
    except Exception:
        return []
