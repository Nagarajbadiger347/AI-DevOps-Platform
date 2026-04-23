"""Pipeline event bus — in-process pub/sub for real-time pipeline progress.

Architecture:
  Publisher  → pipeline nodes call  emit(incident_id, event)
  Subscriber → WebSocket handlers / SSE endpoints call  subscribe(incident_id)

Each incident gets its own asyncio.Queue per subscriber so multiple dashboard
tabs or WebSocket connections each receive every event independently.

Events are also kept in a short rolling buffer (last 50 per incident) so a
client that connects mid-pipeline can catch up immediately.

Thread-safety: pipeline nodes run in ThreadPoolExecutor threads, so emit()
uses loop.call_soon_threadsafe() to push events onto the async event loop.
"""
from __future__ import annotations

import asyncio
import datetime
import threading
from collections import defaultdict, deque
from typing import AsyncIterator

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Event schema ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_event(
    incident_id: str,
    stage: str,
    status: str,           # started | progress | completed | failed | info
    message: str,
    data: dict | None = None,
) -> dict:
    """Build a structured pipeline event."""
    return {
        "incident_id": incident_id,
        "stage":       stage,
        "status":      status,
        "message":     message,
        "data":        data or {},
        "ts":          _now(),
    }


# ── Event bus ─────────────────────────────────────────────────────────────────

class PipelineEventBus:
    """In-process pub/sub bus for pipeline progress events."""

    def __init__(self) -> None:
        # incident_id → list of asyncio.Queue (one per subscriber)
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        # incident_id → rolling buffer of last 50 events (for late joiners)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._lock = threading.Lock()
        # The running event loop — set on first emit from async context
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Publisher API (called from pipeline threads) ──────────────────────────

    def emit(self, incident_id: str, event: dict) -> None:
        """Publish an event. Thread-safe — works from any thread."""
        with self._lock:
            self._history[incident_id].append(event)
            queues = list(self._subscribers.get(incident_id, []))

        logger.info("pipeline_event_emitted", extra={
            "incident_id": incident_id,
            "stage":       event.get("stage"),
            "status":      event.get("status"),
        })

        if not queues:
            return

        loop = self._get_loop()
        if loop and loop.is_running():
            for q in queues:
                loop.call_soon_threadsafe(q.put_nowait, event)

    def emit_stage(
        self,
        incident_id: str,
        stage: str,
        status: str,
        message: str,
        data: dict | None = None,
    ) -> None:
        """Convenience wrapper — builds event then emits."""
        self.emit(incident_id, make_event(incident_id, stage, status, message, data))

    # ── Subscriber API (called from async WebSocket / SSE handlers) ───────────

    async def subscribe(self, incident_id: str) -> AsyncIterator[dict]:
        """Async generator — yields events for incident_id as they arrive.

        Replays the history buffer first so late-joining clients catch up.
        Ends when a terminal event (status="done") is received.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._set_loop()

        with self._lock:
            history = list(self._history.get(incident_id, []))
            self._subscribers[incident_id].append(queue)

        try:
            # Replay history for late joiners
            for event in history:
                yield event

            # Stream live events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event
                    if event.get("status") == "done":
                        break
                except asyncio.TimeoutError:
                    # Send keepalive so SSE/WS connections don't time out
                    yield make_event(incident_id, "keepalive", "info", "heartbeat")
        finally:
            with self._lock:
                subs = self._subscribers.get(incident_id, [])
                if queue in subs:
                    subs.remove(queue)

    def history(self, incident_id: str) -> list[dict]:
        """Return the buffered event history for an incident."""
        with self._lock:
            return list(self._history.get(incident_id, []))

    def close_incident(self, incident_id: str) -> None:
        """Send terminal event and clean up history after a delay."""
        terminal = make_event(incident_id, "pipeline", "done", "Pipeline finished")
        self.emit(incident_id, terminal)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _set_loop(self) -> None:
        """Capture the running event loop from the async context."""
        try:
            loop = asyncio.get_event_loop()
            with self._lock:
                self._loop = loop
        except RuntimeError:
            pass

    def _get_loop(self) -> asyncio.AbstractEventLoop | None:
        with self._lock:
            return self._loop


# Module-level singleton — imported everywhere
bus = PipelineEventBus()
