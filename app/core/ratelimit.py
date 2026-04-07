"""Redis-backed rate limiter with in-memory fallback for single-instance deployments.

Uses a sliding window algorithm via Redis sorted sets.
Falls back to in-memory deque store if Redis is unavailable.

Per-endpoint limits:
  /chat                         → 60 req/min
  /incidents/run, /v2/incident/run → 30 req/min
  /warroom/create, /warroom/ask → 20 req/min
  default                       → 120 req/min
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Tuple

# ---------------------------------------------------------------------------
# Redis client (optional) — imported lazily so the app starts without Redis
# ---------------------------------------------------------------------------
_redis_client = None
_redis_available = False
_redis_last_check: float = 0.0
_REDIS_RETRY_INTERVAL = 30  # seconds between reconnect attempts

def _get_redis():
    """Return a live Redis client, or None if unavailable.

    Supports:
    - Single node:  REDIS_URL=redis://localhost:6379
    - Sentinel HA:  REDIS_SENTINEL_HOSTS=host1:26379,host2:26379
                    REDIS_SENTINEL_MASTER=mymaster

    Re-checks connectivity every 30 s so the app recovers automatically
    when Redis comes back after a failure.
    """
    global _redis_client, _redis_available, _redis_last_check

    now = time.time()

    # Live client — periodic liveness probe
    if _redis_client is not None and _redis_available:
        if now - _redis_last_check > _REDIS_RETRY_INTERVAL:
            try:
                _redis_client.ping()
                _redis_last_check = now
            except Exception:
                import logging
                logging.getLogger("ratelimit").warning(
                    "Redis connection lost — rate limiter falling back to in-memory store"
                )
                _redis_available = False
                _redis_client = None
        return _redis_client if _redis_available else None

    # Don't hammer a down Redis — retry only every 30 s
    if not _redis_available and (now - _redis_last_check) < _REDIS_RETRY_INTERVAL:
        return None
    _redis_last_check = now

    try:
        import redis as redis_py

        # Sentinel HA mode
        sentinel_hosts_raw = os.environ.get("REDIS_SENTINEL_HOSTS", "")
        if sentinel_hosts_raw:
            sentinel_hosts = [
                (h.split(":")[0], int(h.split(":")[1]))
                for h in sentinel_hosts_raw.split(",")
                if ":" in h
            ]
            master_name = os.environ.get("REDIS_SENTINEL_MASTER", "mymaster")
            sentinel = redis_py.Sentinel(sentinel_hosts, socket_timeout=2, socket_connect_timeout=2)
            client = sentinel.master_for(master_name, socket_timeout=2)
        else:
            url = os.environ.get("REDIS_URL", "redis://localhost:6379")
            client = redis_py.from_url(url, socket_connect_timeout=2, socket_timeout=2)

        client.ping()
        _redis_client = client
        _redis_available = True
        import logging
        logging.getLogger("ratelimit").info("Redis connected successfully")
    except Exception as exc:
        import logging
        logging.getLogger("ratelimit").info(
            "Redis unavailable (%s) — using in-memory rate limiter", exc
        )
        _redis_available = False
        _redis_client = None

    return _redis_client if _redis_available else None


# ---------------------------------------------------------------------------
# In-memory fallback store
# ---------------------------------------------------------------------------
_store: dict[str, deque] = defaultdict(deque)
_store_check_counter: int = 0
_MAX_STORE_KEYS = 10_000


# ---------------------------------------------------------------------------
# Core rate limiter class
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter backed by Redis sorted sets.

    Falls back to an in-memory deque store when Redis is unreachable.
    """

    def check(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Return (allowed, remaining).

        allowed   – True if the request is within limit.
        remaining – how many more requests are allowed in the current window.
        """
        prefixed = f"rl:{key}"
        r = _get_redis()
        if r is not None:
            return self._check_redis(r, prefixed, limit, window_seconds)
        return self._check_memory(prefixed, limit, window_seconds)

    # ------------------------------------------------------------------
    # Redis path — sliding window with sorted sets
    # ------------------------------------------------------------------

    @staticmethod
    def _check_redis(r, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        try:
            now = time.time()
            window_start = now - window_seconds
            pipe = r.pipeline()
            # Remove entries older than the window
            pipe.zremrangebyscore(key, "-inf", window_start)
            # Add current request timestamp (score = ts, member = unique ts string)
            pipe.zadd(key, {str(now): now})
            # Count entries in window
            pipe.zcard(key)
            # Set TTL so the key expires automatically
            pipe.expire(key, window_seconds + 1)
            results = pipe.execute()
            count = results[2]
            if count > limit:
                # Remove the entry we just added — request is rejected
                r.zrem(key, str(now))
                return False, 0
            remaining = max(0, limit - count)
            return True, remaining
        except Exception:
            # Redis error mid-request — fail open to avoid blocking users
            return True, limit

    # ------------------------------------------------------------------
    # In-memory path — simple deque sliding window
    # ------------------------------------------------------------------

    @staticmethod
    def _check_memory(key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        global _store_check_counter
        _store_check_counter += 1

        # Periodic cleanup: every 100 requests or when store is too large
        if _store_check_counter % 100 == 0 or len(_store) > _MAX_STORE_KEYS:
            RateLimiter._cleanup_store(window_seconds)

        now = time.time()
        dq = _store[key]
        while dq and dq[0] < now - window_seconds:
            dq.popleft()
        if len(dq) >= limit:
            return False, 0
        dq.append(now)
        return True, max(0, limit - len(dq))

    @staticmethod
    def _cleanup_store(window_seconds: int = 60) -> None:
        """Remove stale entries from the in-memory store.

        Deletes keys whose entire deque is older than the window.
        If the store still exceeds _MAX_STORE_KEYS after that, evict the
        oldest keys until we're back under the limit.
        """
        now = time.time()
        cutoff = now - window_seconds
        stale_keys = [k for k, dq in list(_store.items()) if not dq or dq[-1] < cutoff]
        for k in stale_keys:
            del _store[k]

        # Hard guard: if still over limit, evict the oldest (smallest last-seen)
        if len(_store) > _MAX_STORE_KEYS:
            sorted_keys = sorted(_store.keys(), key=lambda k: _store[k][-1] if _store[k] else 0)
            for k in sorted_keys[: len(_store) - _MAX_STORE_KEYS]:
                del _store[k]


# Module-level singleton
_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Per-endpoint config
# ---------------------------------------------------------------------------

_ENDPOINT_LIMITS: dict[str, tuple[int, int]] = {
    "/chat":              (60,  60),
    "/incidents/run":     (30,  60),
    "/v2/incident/run":   (30,  60),
    "/warroom/create":    (20,  60),
    "/warroom/ask":       (20,  60),
}
_DEFAULT_LIMIT = (120, 60)


def rate_limit_check(identifier: str, endpoint: str) -> Tuple[bool, int]:
    """Check rate limit for a given identifier (user/IP) and endpoint.

    Returns (allowed, remaining).
    """
    limit, window = _ENDPOINT_LIMITS.get(endpoint, _DEFAULT_LIMIT)
    key = f"{identifier}:{endpoint}"
    return _limiter.check(key, limit, window)


# ---------------------------------------------------------------------------
# Backwards-compatible helpers kept for existing callers
# ---------------------------------------------------------------------------

from app.core.config import settings  # noqa: E402

_CHAT_LIMIT   = int(getattr(settings, "RATE_LIMIT_CHAT_PER_MIN",   20))
_ACTION_LIMIT = int(getattr(settings, "RATE_LIMIT_ACTION_PER_MIN", 10))
_WINDOW       = 60


def check_chat(user: str) -> Tuple[bool, int]:
    """Legacy helper — check the general chat rate limit."""
    return _limiter.check(f"{user}:chat", _CHAT_LIMIT, _WINDOW)


def check_action(user: str) -> Tuple[bool, int]:
    """Legacy helper — check the mutating-action rate limit."""
    return _limiter.check(f"{user}:action", _ACTION_LIMIT, _WINDOW)


def get_usage(user: str) -> dict:
    """Return current usage stats for a user (best-effort from memory store)."""
    now = time.time()

    def _count(suffix: str) -> int:
        key = f"rl:{user}:{suffix}"
        dq = _store.get(key, deque())
        return sum(1 for t in dq if t >= now - _WINDOW)

    return {
        "user":           user,
        "chat_used":      _count("chat"),
        "chat_limit":     _CHAT_LIMIT,
        "action_used":    _count("action"),
        "action_limit":   _ACTION_LIMIT,
        "window_seconds": _WINDOW,
    }
