"""Conversation memory for the AI chatbot.

Maintains per-session message history with:
- In-memory LRU cache (fast reads, max 100 sessions)
- SQLite persistence (survives server restarts)

Usage:
    from app.chat.memory import get_or_create_session, add_message, get_history

    session = get_or_create_session("user-abc-123")
    add_message("user-abc-123", "user", "What is the CPU usage?")
    history = get_history("user-abc-123")
"""
from __future__ import annotations

import datetime
import json
import sqlite3
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_MAX_SESSIONS     = int(100)
_DEFAULT_MAX_MSGS = int(20)

# SQLite database path — stored next to this file
_DB_PATH = Path(__file__).parent / "chat_history.db"
_DB_LOCK = threading.Lock()   # serialise writes; reads use their own connections

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversation message."""
    role:      str          # "user" | "assistant" | "system"
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata:  dict = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationSession:
    """All state for a single chatbot session."""
    session_id:  str
    messages:    list[Message] = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    last_active: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    context:     dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create tables if they don't exist. Called once at module load."""
    with _DB_LOCK, _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                last_active TEXT NOT NULL,
                context_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id  TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, timestamp);
        """)


def _db_save_message(session_id: str, msg: Message) -> None:
    """Persist a single message to SQLite (non-blocking on lock)."""
    try:
        with _DB_LOCK, _get_conn() as conn:
            # Upsert session row
            conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, created_at, last_active, context_json) "
                "VALUES (?,?,?,'{}')",
                (session_id, datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            conn.execute(
                "UPDATE sessions SET last_active=? WHERE session_id=?",
                (datetime.datetime.now(datetime.timezone.utc).isoformat(), session_id),
            )
            # Insert message
            conn.execute(
                "INSERT OR IGNORE INTO messages"
                "(message_id, session_id, role, content, timestamp, metadata_json) "
                "VALUES (?,?,?,?,?,?)",
                (msg.message_id, session_id, msg.role, msg.content,
                 msg.timestamp, json.dumps(msg.metadata or {})),
            )
    except Exception as exc:
        logger.warning("db_save_message_failed: %s", exc)


def _db_load_session(session_id: str, max_messages: int = _DEFAULT_MAX_MSGS) -> list[Message]:
    """Load the most recent messages for a session from SQLite."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp, metadata_json, message_id "
                "FROM messages WHERE session_id=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (session_id, max_messages),
            ).fetchall()
        return [
            Message(
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata_json"] or "{}"),
                message_id=r["message_id"],
            )
            for r in reversed(rows)   # oldest first
        ]
    except Exception as exc:
        logger.warning("db_load_session_failed: %s", exc)
        return []


def _db_list_recent_sessions(limit: int = 50) -> list[dict]:
    """Return recently active sessions from SQLite."""
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, last_active, context_json "
                "FROM sessions ORDER BY last_active DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "last_active": r["last_active"],
                "context": json.loads(r["context_json"] or "{}"),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.warning("db_list_sessions_failed: %s", exc)
        return []


# Initialise DB on import
try:
    _init_db()
except Exception as _db_init_exc:
    logger.warning("chat_db_init_failed: %s — history will be in-memory only", _db_init_exc)

# ---------------------------------------------------------------------------
# In-memory LRU store
# ---------------------------------------------------------------------------
_sessions: OrderedDict[str, ConversationSession] = OrderedDict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log(level_fn, msg: str, **kwargs) -> None:
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)


def _touch(session_id: str) -> None:
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
        _sessions[session_id].last_active = datetime.datetime.now(datetime.timezone.utc).isoformat()


def _evict_if_needed() -> None:
    while len(_sessions) >= _MAX_SESSIONS:
        evicted_id, _ = _sessions.popitem(last=False)
        _log(logger.info, "session_evicted_lru", session_id=evicted_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_session(session_id: str) -> ConversationSession:
    """Return an existing session or create a new one (restores from SQLite if available)."""
    if session_id in _sessions:
        _touch(session_id)
        return _sessions[session_id]

    _evict_if_needed()
    session = ConversationSession(session_id=session_id)

    # Restore from SQLite if this session existed before
    persisted = _db_load_session(session_id, max_messages=_DEFAULT_MAX_MSGS)
    if persisted:
        session.messages = persisted
        _log(logger.info, "session_restored_from_db",
             session_id=session_id, message_count=len(persisted))

    _sessions[session_id] = session
    _log(logger.info, "session_created", session_id=session_id, total_sessions=len(_sessions))
    return session


def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict = None,
) -> Message:
    """Append a message to a session's history and persist it to SQLite."""
    session = get_or_create_session(session_id)
    msg = Message(
        role=role,
        content=content,
        metadata=metadata or {},
    )
    session.messages.append(msg)
    _touch(session_id)

    # Persist to SQLite in a background thread to avoid blocking the response
    t = threading.Thread(target=_db_save_message, args=(session_id, msg), daemon=True)
    t.start()

    return msg


def get_history(session_id: str, max_messages: int = _DEFAULT_MAX_MSGS) -> list[Message]:
    """Return the most recent messages for a session.

    Checks in-memory cache first; falls back to SQLite if the session
    was evicted from RAM (e.g. after a server restart).
    """
    if session_id in _sessions:
        _touch(session_id)
        return _sessions[session_id].messages[-max_messages:]

    # Session not in RAM — try SQLite
    persisted = _db_load_session(session_id, max_messages=max_messages)
    if persisted:
        # Reload into RAM for subsequent calls
        _evict_if_needed()
        session = ConversationSession(session_id=session_id, messages=persisted)
        _sessions[session_id] = session
        return persisted

    return []


def set_context(session_id: str, key: str, value: Any) -> None:
    session = get_or_create_session(session_id)
    session.context[key] = value
    _touch(session_id)


def get_context(session_id: str, key: str, default: Any = None) -> Any:
    session = _sessions.get(session_id)
    if session is None:
        return default
    return session.context.get(key, default)


def clear_session(session_id: str) -> None:
    """Remove a session from RAM and delete its history from SQLite."""
    if session_id in _sessions:
        del _sessions[session_id]
    try:
        with _DB_LOCK, _get_conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    except Exception as exc:
        logger.warning("db_clear_session_failed: %s", exc)
    _log(logger.info, "session_cleared", session_id=session_id)


def list_sessions() -> list[dict]:
    """Return summary dicts for all active sessions (RAM + recent SQLite)."""
    # Merge in-memory sessions with recent SQLite sessions
    seen = set()
    result = []
    for sid, session in _sessions.items():
        seen.add(sid)
        result.append({
            "session_id":    sid,
            "message_count": len(session.messages),
            "created_at":    session.created_at,
            "last_active":   session.last_active,
            "context_keys":  list(session.context.keys()),
            "source":        "memory",
        })
    for db_sess in _db_list_recent_sessions():
        if db_sess["session_id"] not in seen:
            result.append({
                "session_id":    db_sess["session_id"],
                "message_count": None,   # not counted without loading all messages
                "created_at":    db_sess["created_at"],
                "last_active":   db_sess["last_active"],
                "context_keys":  [],
                "source":        "db",
            })
    return result
