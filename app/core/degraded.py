"""Degraded-mode health manager.

Checks Postgres, Redis, and LLM reachability at startup and on demand.
When a dependency is unavailable the platform enters a partial-function mode:
  - DB down  → pipeline disabled; read-only metric/log endpoints still work
  - Redis down → rate-limiting and LLM cache skipped (non-fatal)
  - LLM down  → pipeline disabled; observability endpoints still work

Usage:
    from app.core.degraded import system_health, is_degraded, DegradedError

    if is_degraded("database"):
        raise DegradedError("database")

    health = system_health()   # -> SystemHealth dataclass
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

Component = Literal["database", "redis", "llm"]

_CHECK_INTERVAL = 30        # re-probe every 30 s
_PROBE_TIMEOUT  = 5         # max seconds per probe

@dataclass
class ComponentStatus:
    name: str
    ok: bool
    error: str = ""
    last_checked: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        return time.time() - self.last_checked


@dataclass
class SystemHealth:
    database: ComponentStatus
    redis:    ComponentStatus
    llm:      ComponentStatus

    @property
    def pipeline_available(self) -> bool:
        """Pipeline requires both DB and LLM."""
        return self.database.ok and self.llm.ok

    @property
    def mode(self) -> str:
        if self.pipeline_available:
            return "full"
        if not self.database.ok and not self.llm.ok:
            return "telemetry-only"
        if not self.database.ok:
            return "no-persistence"
        return "no-ai"

    def to_dict(self) -> dict:
        return {
            "mode":               self.mode,
            "pipeline_available": self.pipeline_available,
            "components": {
                "database": {"ok": self.database.ok, "error": self.database.error},
                "redis":    {"ok": self.redis.ok,    "error": self.redis.error},
                "llm":      {"ok": self.llm.ok,      "error": self.llm.error},
            },
        }


class DegradedError(RuntimeError):
    """Raised when a required component is unavailable."""

    def __init__(self, component: str):
        self.component = component
        super().__init__(
            f"Platform is in degraded mode — {component} is unavailable. "
            f"AI pipeline and memory operations are disabled. "
            f"Read-only observability endpoints remain functional."
        )


# ── Probes ────────────────────────────────────────────────────────────────────

def _probe_database() -> ComponentStatus:
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
            fut = exe.submit(_db_ping)
            ok, err = fut.result(timeout=_PROBE_TIMEOUT)
        return ComponentStatus("database", ok, err)
    except concurrent.futures.TimeoutError:
        return ComponentStatus("database", False, "probe timed out")
    except Exception as exc:
        return ComponentStatus("database", False, str(exc))


def _db_ping() -> tuple[bool, str]:
    try:
        from app.core.database import health_check
        return (True, "") if health_check() else (False, "health_check returned False")
    except Exception as exc:
        return (False, str(exc))


def _probe_redis() -> ComponentStatus:
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
            fut = exe.submit(_redis_ping)
            ok, err = fut.result(timeout=_PROBE_TIMEOUT)
        return ComponentStatus("redis", ok, err)
    except concurrent.futures.TimeoutError:
        return ComponentStatus("redis", False, "probe timed out")
    except Exception as exc:
        return ComponentStatus("redis", False, str(exc))


def _redis_ping() -> tuple[bool, str]:
    try:
        import redis as _redis
        from app.core.config import settings
        r = _redis.from_url(settings.REDIS_URL, socket_connect_timeout=3)
        r.ping()
        return (True, "")
    except Exception as exc:
        return (False, str(exc))


def _probe_llm() -> ComponentStatus:
    try:
        import os
        keys = [
            os.getenv("ANTHROPIC_API_KEY", ""),
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("GROQ_API_KEY", ""),
        ]
        if not any(k.strip() for k in keys):
            # Check Ollama as last resort
            try:
                import urllib.request
                from app.core.config import settings
                urllib.request.urlopen(f"{settings.OLLAMA_HOST}/api/tags", timeout=3)
                return ComponentStatus("llm", True, "")
            except Exception:
                return ComponentStatus("llm", False, "no LLM API key configured and Ollama unreachable")
        return ComponentStatus("llm", True, "")
    except Exception as exc:
        return ComponentStatus("llm", False, str(exc))


# ── Health state (module-level singleton) ─────────────────────────────────────

_state: SystemHealth | None = None
_lock  = threading.Lock()


def _build_health() -> SystemHealth:
    db    = _probe_database()
    redis = _probe_redis()
    llm   = _probe_llm()
    return SystemHealth(database=db, redis=redis, llm=llm)


def system_health(force_refresh: bool = False) -> SystemHealth:
    """Return current system health, re-probing if stale or forced."""
    global _state
    with _lock:
        if _state is None or force_refresh or _state.database.age > _CHECK_INTERVAL:
            _state = _build_health()
            if not _state.pipeline_available:
                logger.warning(
                    "degraded_mode_active",
                    extra={
                        "mode":     _state.mode,
                        "database": _state.database.ok,
                        "redis":    _state.redis.ok,
                        "llm":      _state.llm.ok,
                    },
                )
            else:
                logger.debug("system_health_ok", extra={"mode": _state.mode})
    return _state


def is_degraded(component: Component) -> bool:
    """Return True if the given component is currently unavailable."""
    h = system_health()
    return not getattr(h, component).ok


def require_pipeline() -> None:
    """Raise DegradedError if the AI pipeline cannot run right now."""
    h = system_health()
    if not h.database.ok:
        raise DegradedError("database")
    if not h.llm.ok:
        raise DegradedError("llm")


def _background_refresh() -> None:
    """Periodically refresh health state so is_degraded() stays current."""
    import time as _t
    while True:
        _t.sleep(_CHECK_INTERVAL)
        try:
            system_health(force_refresh=True)
        except Exception:
            pass


# Start background refresh thread on module import
_refresh_thread = threading.Thread(target=_background_refresh, daemon=True, name="health-refresh")
_refresh_thread.start()
