"""Simple in-memory rate limiter — no Redis needed for single-instance deployments.

Default limits (configurable via .env):
  RATE_LIMIT_CHAT_PER_MIN   = 20   # max chat messages per user per minute
  RATE_LIMIT_ACTION_PER_MIN = 10   # max mutating actions per user per minute

For multi-instance deployments, swap the deque store for a Redis backend.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Tuple

from app.core.config import settings


# Read limits from settings (with fallback defaults)
_CHAT_LIMIT   = int(getattr(settings, "RATE_LIMIT_CHAT_PER_MIN",   20))
_ACTION_LIMIT = int(getattr(settings, "RATE_LIMIT_ACTION_PER_MIN", 10))
_WINDOW       = 60  # seconds

# {(user, bucket): deque of timestamps}
_store: dict[tuple, deque] = defaultdict(deque)


def _check(user: str, bucket: str, limit: int) -> Tuple[bool, int]:
    """Return (allowed, requests_remaining). Thread-safe enough for single-process."""
    now  = time.time()
    key  = (user, bucket)
    dq   = _store[key]
    # Drop entries older than the window
    while dq and dq[0] < now - _WINDOW:
        dq.popleft()
    if len(dq) >= limit:
        return False, 0
    dq.append(now)
    return True, limit - len(dq)


def check_chat(user: str) -> Tuple[bool, int]:
    """Check the general chat rate limit. Returns (allowed, remaining)."""
    return _check(user, "chat", _CHAT_LIMIT)


def check_action(user: str) -> Tuple[bool, int]:
    """Check the mutating-action rate limit. Returns (allowed, remaining)."""
    return _check(user, "action", _ACTION_LIMIT)


def get_usage(user: str) -> dict:
    """Return current usage stats for a user."""
    now = time.time()
    def _count(bucket: str) -> int:
        dq = _store.get((user, bucket), deque())
        return sum(1 for t in dq if t >= now - _WINDOW)
    return {
        "user":             user,
        "chat_used":        _count("chat"),
        "chat_limit":       _CHAT_LIMIT,
        "action_used":      _count("action"),
        "action_limit":     _ACTION_LIMIT,
        "window_seconds":   _WINDOW,
    }
