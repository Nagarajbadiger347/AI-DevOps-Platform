"""LLM response cache — short-TTL in-memory cache for identical LLM calls.

Prevents redundant LLM calls when the same incident type and context appear
within the TTL window (e.g. alert storms that fire the same incident 5 times
in 3 minutes). Each cache entry is keyed by a hash of (system_prompt, user_prompt)
so different prompts always get fresh responses.

TTL is intentionally short (5 minutes) to prevent stale plans being returned
for incidents where infra state has changed.

Thread-safe: uses a lock for all cache reads and writes.
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Cache TTL in seconds. Keep short — infra state can change rapidly.
_DEFAULT_TTL = int(300)  # 5 minutes

# Maximum cache entries. Old entries are evicted when this is exceeded.
_MAX_ENTRIES = 256


class LLMCache:
    """Thread-safe in-memory LLM response cache."""

    def __init__(self, ttl: int = _DEFAULT_TTL, max_entries: int = _MAX_ENTRIES) -> None:
        self._ttl        = ttl
        self._max        = max_entries
        self._store: dict[str, dict] = {}
        self._lock       = threading.Lock()
        self._hits       = 0
        self._misses     = 0

    @staticmethod
    def make_key(system: str, prompt: str) -> str:
        """Deterministic cache key from system prompt + user prompt."""
        raw = f"{system.strip()}\n\n{prompt.strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() > entry["exp"]:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            logger.info("llm_cache_hit", extra={
                "key":       key[:12],
                "hits":      self._hits,
                "misses":    self._misses,
                "hit_rate":  f"{self._hits / max(1, self._hits + self._misses):.0%}",
            })
            return entry["value"]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            # Evict expired entries if we're at capacity
            if len(self._store) >= self._max:
                now = time.monotonic()
                expired = [k for k, v in self._store.items() if v["exp"] < now]
                for k in expired:
                    del self._store[k]
                # If still at capacity, evict the oldest entry
                if len(self._store) >= self._max:
                    oldest = min(self._store, key=lambda k: self._store[k]["exp"])
                    del self._store[oldest]
            self._store[key] = {"value": value, "exp": time.monotonic() + self._ttl}

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits":      self._hits,
                "misses":    self._misses,
                "hit_rate":  round(self._hits / max(1, total), 3),
                "size":      len(self._store),
                "max":       self._max,
                "ttl_secs":  self._ttl,
            }


# Module-level singleton — shared across all agents in the same process
llm_cache = LLMCache()
