"""Conversation memory — PostgreSQL backed with tenant isolation.

Maintains per-session message history.

Usage:
    from app.chat.memory import get_or_create_session, add_message, get_history

    session = get_or_create_session("user-abc-123", tenant_id="acme")
    add_message("user-abc-123", "user", "What is the CPU usage?", tenant_id="acme")
    history = get_history("user-abc-123", tenant_id="acme")
"""
from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

_DEFAULT_MAX_MSGS = int(20)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role:       str
    content:    str
    timestamp:  str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata:   dict = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationSession:
    session_id:  str
    tenant_id:   str = "default"
    messages:    list[Message] = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    last_active: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    context:     dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db():
    from app.core.database import execute, execute_one
    return execute, execute_one


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_session(session_id: str, tenant_id: str = "default") -> ConversationSession:
    execute, execute_one = _db()
    row = execute_one(
        "SELECT * FROM chat_sessions WHERE session_id = %s AND tenant_id = %s",
        (session_id, tenant_id)
    )
    if row:
        return ConversationSession(
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            created_at=str(row["created_at"]),
            last_active=str(row["last_active"]),
            context=row.get("context") or {},
        )
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    execute(
        "INSERT INTO chat_sessions (session_id, tenant_id, created_at, last_active) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
        (session_id, tenant_id, now, now)
    )
    return ConversationSession(session_id=session_id, tenant_id=tenant_id)


def add_message(
    session_id: str,
    role: str,
    content: str,
    tenant_id: str = "default",
    metadata: Optional[dict] = None,
) -> Message:
    execute, _ = _db()
    msg = Message(role=role, content=content, metadata=metadata or {})
    execute(
        """
        INSERT INTO chat_messages (message_id, session_id, tenant_id, role, content, metadata, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (msg.message_id, session_id, tenant_id, role, content,
         json.dumps(msg.metadata), msg.timestamp)
    )
    execute(
        "UPDATE chat_sessions SET last_active = NOW() WHERE session_id = %s AND tenant_id = %s",
        (session_id, tenant_id)
    )
    return msg


def get_history(
    session_id: str,
    tenant_id: str = "default",
    max_messages: int = _DEFAULT_MAX_MSGS,
) -> list[Message]:
    execute, _ = _db()
    rows = execute(
        """
        SELECT * FROM chat_messages
        WHERE session_id = %s AND tenant_id = %s
        ORDER BY created_at DESC LIMIT %s
        """,
        (session_id, tenant_id, max_messages)
    )
    rows.reverse()
    return [
        Message(
            role=r["role"],
            content=r["content"],
            timestamp=str(r["created_at"]),
            metadata=r.get("metadata") or {},
            message_id=r["message_id"],
        )
        for r in rows
    ]


def get_history_as_dicts(
    session_id: str,
    tenant_id: str = "default",
    max_messages: int = _DEFAULT_MAX_MSGS,
) -> list[dict]:
    return [
        {"role": m.role, "content": m.content}
        for m in get_history(session_id, tenant_id, max_messages)
    ]


def clear_session(session_id: str, tenant_id: str = "default") -> None:
    execute, _ = _db()
    execute(
        "DELETE FROM chat_messages WHERE session_id = %s AND tenant_id = %s",
        (session_id, tenant_id)
    )


def list_sessions(tenant_id: str = "default", limit: int = 50) -> list[dict]:
    execute, _ = _db()
    rows = execute(
        "SELECT session_id, created_at, last_active FROM chat_sessions WHERE tenant_id = %s ORDER BY last_active DESC LIMIT %s",
        (tenant_id, limit)
    )
    return [{"session_id": r["session_id"], "last_active": str(r["last_active"])} for r in rows]


def get_session_count(tenant_id: str = "default") -> int:
    execute, _ = _db()
    rows = execute("SELECT COUNT(*) as cnt FROM chat_sessions WHERE tenant_id = %s", (tenant_id,))
    return rows[0]["cnt"] if rows else 0
