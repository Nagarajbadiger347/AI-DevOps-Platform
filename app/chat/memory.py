"""Conversation memory for the AI chatbot.

Maintains per-session message history with LRU eviction (max 100 sessions).
Sessions hold typed ``Message`` objects and an arbitrary ``context`` dict for
incident war-room state.

Usage:
    from app.chat.memory import get_or_create_session, add_message, get_history

    session = get_or_create_session("user-abc-123")
    add_message("user-abc-123", "user", "What is the CPU usage?")
    history = get_history("user-abc-123")
"""
from __future__ import annotations

import datetime
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
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

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single conversation message."""
    role:      str          # "user" | "assistant" | "system"
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    metadata:  dict = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationSession:
    """All state for a single chatbot session."""
    session_id:  str
    messages:    list[Message] = field(default_factory=list)
    created_at:  str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    last_active: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    context:     dict = field(default_factory=dict)  # incident context, war-room ID, etc.


# ---------------------------------------------------------------------------
# In-memory LRU store
# OrderedDict is used so we can evict the least-recently-used session when
# the cap is exceeded.
# ---------------------------------------------------------------------------
_sessions: OrderedDict[str, ConversationSession] = OrderedDict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _log(level_fn, msg: str, **kwargs) -> None:
    """Emit a structured log message compatible with stdlib logging."""
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)

def _touch(session_id: str) -> None:
    """Move session to end of OrderedDict (most recently used) and update timestamp."""
    if session_id in _sessions:
        _sessions.move_to_end(session_id)
        _sessions[session_id].last_active = datetime.datetime.utcnow().isoformat() + "Z"


def _evict_if_needed() -> None:
    """Remove the least-recently-used session if we are at capacity."""
    while len(_sessions) >= _MAX_SESSIONS:
        evicted_id, _ = _sessions.popitem(last=False)
        _log(logger.info, "session_evicted_lru", session_id=evicted_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_session(session_id: str) -> ConversationSession:
    """Return an existing session or create a new one.

    Args:
        session_id: Unique identifier for the session (e.g. user ID, UUID).

    Returns:
        ConversationSession for the given ID.
    """
    if session_id in _sessions:
        _touch(session_id)
        return _sessions[session_id]

    _evict_if_needed()
    session = ConversationSession(session_id=session_id)
    _sessions[session_id] = session
    _log(logger.info, "session_created", session_id=session_id, total_sessions=len(_sessions))
    return session


def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict = None,
) -> Message:
    """Append a message to a session's history.

    Args:
        session_id: Target session.
        role:       "user", "assistant", or "system".
        content:    Message text.
        metadata:   Optional dict of extra data (e.g. tool calls, cost).

    Returns:
        The newly created Message.
    """
    session = get_or_create_session(session_id)
    msg = Message(
        role=role,
        content=content,
        metadata=metadata or {},
    )
    session.messages.append(msg)
    _touch(session_id)
    return msg


def get_history(session_id: str, max_messages: int = _DEFAULT_MAX_MSGS) -> list[Message]:
    """Return the most recent messages for a session.

    Args:
        session_id:   Target session.
        max_messages: Maximum number of messages to return (most recent).

    Returns:
        List of Message objects in chronological order.
    """
    session = _sessions.get(session_id)
    if session is None:
        return []
    _touch(session_id)
    return session.messages[-max_messages:]


def set_context(session_id: str, key: str, value: Any) -> None:
    """Set an arbitrary key in the session's context dict.

    Args:
        session_id: Target session.
        key:        Context key.
        value:      Value to store.
    """
    session = get_or_create_session(session_id)
    session.context[key] = value
    _touch(session_id)


def get_context(session_id: str, key: str, default: Any = None) -> Any:
    """Retrieve a value from the session's context dict.

    Args:
        session_id: Target session.
        key:        Context key.
        default:    Value returned if the key is absent.

    Returns:
        Stored value or default.
    """
    session = _sessions.get(session_id)
    if session is None:
        return default
    return session.context.get(key, default)


def clear_session(session_id: str) -> None:
    """Remove a session and all its history from memory.

    Args:
        session_id: Session to clear.
    """
    if session_id in _sessions:
        del _sessions[session_id]
        _log(logger.info, "session_cleared", session_id=session_id)


def list_sessions() -> list[dict]:
    """Return summary dicts for all active sessions.

    Returns:
        List of dicts with session_id, message_count, created_at, last_active,
        and context keys.
    """
    result = []
    for sid, session in _sessions.items():
        result.append({
            "session_id":    sid,
            "message_count": len(session.messages),
            "created_at":    session.created_at,
            "last_active":   session.last_active,
            "context_keys":  list(session.context.keys()),
        })
    return result
