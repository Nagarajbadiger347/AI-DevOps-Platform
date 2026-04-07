"""LLM-powered war room AI for active incident response.

Wraps chat.intelligence and chat.memory to provide a focused AI assistant
scoped to a specific incident.  Each war room has its own session so the
LLM retains the full incident context throughout the response lifecycle.

Usage:
    from app.incident.war_room_intelligence import (
        create_war_room_session,
        answer_war_room_question,
    )

    wr = create_war_room_session(
        incident_id="INC-042",
        description="Payment service latency spike p99 > 10s",
        pipeline_state={"root_cause": "DB connection pool exhausted"},
        slack_channel="#incidents",
    )
    answer = answer_war_room_question(wr.war_room_id, "What should we check next?", "alice")
"""
from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from app.chat.intelligence import chat_with_intelligence
    _INTELLIGENCE_AVAILABLE = True
except ImportError:
    _INTELLIGENCE_AVAILABLE = False

try:
    from app.chat.memory import (
        get_or_create_session,
        add_message,
        get_history,
        set_context,
    )
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False

try:
    from app.integrations.slack import post_message
    _SLACK_AVAILABLE = True
except ImportError:
    _SLACK_AVAILABLE = False

try:
    from app.llm.factory import LLMFactory
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WarRoomSession:
    """All state for an active incident war room."""
    war_room_id:          str
    incident_id:          str
    incident_description: str
    pipeline_state:       dict
    slack_channel:        str
    created_at:           str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    participants:         list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistent store — JSON file backed, survives restarts
# ---------------------------------------------------------------------------
import os as _os
import pathlib as _pathlib

_WR_STORE_PATH = _pathlib.Path(
    _os.getenv("WAR_ROOM_STORE_PATH", "") or
    _pathlib.Path(__file__).resolve().parents[2] / "data" / "war_rooms.json"
)

_war_rooms: dict[str, WarRoomSession] = {}


def _wr_save() -> None:
    """Persist all war rooms to disk."""
    try:
        _WR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            wr_id: {
                "war_room_id":          wr.war_room_id,
                "incident_id":          wr.incident_id,
                "incident_description": wr.incident_description,
                "pipeline_state":       wr.pipeline_state,
                "slack_channel":        wr.slack_channel,
                "created_at":           wr.created_at,
                "participants":         wr.participants,
            }
            for wr_id, wr in _war_rooms.items()
        }
        tmp = _WR_STORE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(_WR_STORE_PATH)
    except Exception:
        pass


def _wr_load() -> None:
    """Load war rooms from disk on startup."""
    try:
        with open(_WR_STORE_PATH) as f:
            data = json.load(f)
        for wr_id, d in data.items():
            wr = WarRoomSession(
                war_room_id          = d["war_room_id"],
                incident_id          = d["incident_id"],
                incident_description = d["incident_description"],
                pipeline_state       = d.get("pipeline_state", {}),
                slack_channel        = d.get("slack_channel", ""),
                created_at           = d.get("created_at", ""),
                participants         = d.get("participants", []),
            )
            _war_rooms[wr_id] = wr
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass


_wr_load()  # load on import


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------



def _log(level_fn, msg: str, **kwargs) -> None:
    """Emit a structured log message compatible with stdlib logging."""
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)

def _war_room_session_id(war_room_id: str) -> str:
    """Map a war room ID to its chat session ID."""
    return f"war_room::{war_room_id}"


def _build_incident_context(wr: WarRoomSession) -> dict:
    """Build the incident context dict injected into every LLM call.

    Includes a ``context_captured_at`` timestamp so the LLM knows the age
    of the data it is working with.
    """
    return {
        "war_room_id":          wr.war_room_id,
        "incident_id":          wr.incident_id,
        "incident_description": wr.incident_description,
        "root_cause":           wr.pipeline_state.get("root_cause", "Under investigation"),
        "actions_taken":        wr.pipeline_state.get("actions_taken",
                                wr.pipeline_state.get("executed_actions", [])),
        "current_status":       wr.pipeline_state.get("status", "active"),
        "severity":             wr.pipeline_state.get("severity", "SEV2"),
        "slack_channel":        wr.slack_channel,
        "participants":         wr.participants,
        "pipeline_state":       wr.pipeline_state,
        "war_room_started_at":  wr.created_at,
        "context_captured_at":  datetime.datetime.utcnow().isoformat() + "Z",
    }


def _post_to_slack_safe(channel: str, text: str) -> None:
    """Post to Slack, silently swallowing any errors."""
    if not _SLACK_AVAILABLE or not channel:
        return
    try:
        result = post_message(channel=channel, text=text)
        if not result.get("success"):
            _log(logger.warning, "war_room_slack_post_failed", error=result.get("error"))
    except Exception as exc:
        _log(logger.warning, "war_room_slack_exception", error=str(exc))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_war_room_session(
    incident_id: str,
    description: str,
    pipeline_state: dict,
    slack_channel: str,
) -> WarRoomSession:
    """Create and register a new war room session.

    An initial message with incident context is seeded into the session's
    conversation memory so the LLM has full awareness from the first question.

    Args:
        incident_id:    Incident identifier (e.g. "INC-042").
        description:    Short description of the incident.
        pipeline_state: Dict with root_cause, actions_taken, severity, etc.
        slack_channel:  Slack channel for posting answers (e.g. "#incidents").

    Returns:
        New WarRoomSession.
    """
    war_room_id = str(uuid.uuid4())
    wr = WarRoomSession(
        war_room_id=war_room_id,
        incident_id=incident_id,
        incident_description=description,
        pipeline_state=pipeline_state,
        slack_channel=slack_channel,
    )
    _war_rooms[war_room_id] = wr
    _wr_save()  # persist immediately

    # Pre-seed the memory session with an incident summary system message
    if _MEMORY_AVAILABLE:
        session_id = _war_room_session_id(war_room_id)
        get_or_create_session(session_id)
        set_context(session_id, "war_room_id",  war_room_id)
        set_context(session_id, "incident_id",  incident_id)
        context_dump = json.dumps(_build_incident_context(wr), indent=2, default=str)
        add_message(
            session_id,
            "system",
            f"War room initialized for incident {incident_id}.\n\n"
            f"Incident context:\n{context_dump}",
            metadata={"type": "war_room_init"},
        )

    _log(logger.info, 
        "war_room_created",
        war_room_id=war_room_id,
        incident_id=incident_id,
        channel=slack_channel,
    )
    return wr


def get_war_room(war_room_id: str) -> Optional[WarRoomSession]:
    """Retrieve an active war room by ID.

    Args:
        war_room_id: UUID of the war room.

    Returns:
        WarRoomSession or None if not found.
    """
    return _war_rooms.get(war_room_id)


def list_active_war_rooms() -> list[dict]:
    """Return summary dicts for all active war rooms.

    Returns:
        List of dicts with war_room_id, incident_id, description,
        slack_channel, created_at, and participant count.
    """
    return [
        {
            "war_room_id":   wr.war_room_id,
            "incident_id":   wr.incident_id,
            "description":   wr.incident_description,
            "slack_channel": wr.slack_channel,
            "created_at":    wr.created_at,
            "participants":  len(wr.participants),
            "severity":      wr.pipeline_state.get("severity", "SEV2"),
        }
        for wr in _war_rooms.values()
        if wr.pipeline_state.get("status") != "resolved"
    ]


def answer_war_room_question(
    war_room_id: str,
    question: str,
    asked_by: str,
) -> str:
    """Answer an arbitrary question in the context of an active war room.

    Uses chat_with_intelligence with the full incident context pre-loaded.
    Posts the answer to the war room's Slack channel.

    Args:
        war_room_id: UUID of the war room.
        question:    The question asked by the team member.
        asked_by:    Username or display name of the asker.

    Returns:
        Answer string from the LLM.
    """
    wr = _war_rooms.get(war_room_id)
    if wr is None:
        return f"War room {war_room_id} not found."

    # Track participant
    if asked_by and asked_by not in wr.participants:
        wr.participants.append(asked_by)

    if not _INTELLIGENCE_AVAILABLE:
        return "Chat intelligence module is not available (check LLM configuration)."

    session_id       = _war_room_session_id(war_room_id)
    incident_context = _build_incident_context(wr)

    _log(logger.info, 
        "war_room_question",
        war_room_id=war_room_id,
        asked_by=asked_by,
        question_length=len(question),
    )

    answer = chat_with_intelligence(
        message=question,
        session_id=session_id,
        incident_context=incident_context,
    )

    # Post to Slack
    slack_text = (
        f":robot_face: *War Room AI* | _{wr.incident_id}_\n"
        f"*{asked_by} asked:* {question}\n\n"
        f"{answer}"
    )
    _post_to_slack_safe(wr.slack_channel, slack_text)

    _log(logger.info, 
        "war_room_answer_posted",
        war_room_id=war_room_id,
        answer_length=len(answer),
    )
    return answer


def generate_incident_timeline(war_room_id: str) -> list[dict]:
    """Reconstruct a timeline of what happened during the incident.

    Pulls events from the war room's conversation history and pipeline state.

    Args:
        war_room_id: UUID of the war room.

    Returns:
        List of event dicts sorted by timestamp: {timestamp, event, actor, source}.
    """
    wr = _war_rooms.get(war_room_id)
    if wr is None:
        return []

    events: list[dict] = []

    # Seed from pipeline state
    events.append({
        "timestamp": wr.created_at,
        "event":     f"War room created for incident {wr.incident_id}: {wr.incident_description}",
        "actor":     "system",
        "source":    "war_room",
    })

    # Actions taken from pipeline
    actions_taken = wr.pipeline_state.get(
        "actions_taken", wr.pipeline_state.get("executed_actions", [])
    )
    if isinstance(actions_taken, list):
        for i, action in enumerate(actions_taken):
            ts = action.get("executed_at", action.get("timestamp", wr.created_at))
            events.append({
                "timestamp": ts,
                "event":     (
                    f"Action executed: {action.get('action_type', action.get('type', 'unknown'))} — "
                    f"{action.get('description', '')}"
                ),
                "actor":     "pipeline",
                "source":    "pipeline_state",
            })

    # Messages from conversation memory
    if _MEMORY_AVAILABLE:
        session_id = _war_room_session_id(war_room_id)
        history    = get_history(session_id, max_messages=100)
        for msg in history:
            role = getattr(msg, "role", "")
            if role in ("user", "assistant"):
                events.append({
                    "timestamp": getattr(msg, "timestamp", ""),
                    "event":     getattr(msg, "content", "")[:200],
                    "actor":     role,
                    "source":    "conversation",
                })

    # Sort by timestamp (best-effort)
    def _sort_key(e: dict) -> str:
        return e.get("timestamp") or ""

    events.sort(key=_sort_key)
    return events


def suggest_next_steps(war_room_id: str) -> str:
    """Ask the LLM to suggest what the team should do next.

    Args:
        war_room_id: UUID of the war room.

    Returns:
        LLM-generated string of suggested next steps.
    """
    wr = _war_rooms.get(war_room_id)
    if wr is None:
        return f"War room {war_room_id} not found."

    if not _LLM_AVAILABLE:
        return "LLM not available — cannot generate suggestions."

    incident_context = _build_incident_context(wr)
    timeline         = generate_incident_timeline(war_room_id)

    timeline_str = "\n".join(
        f"  {e['timestamp']} [{e['actor']}] {e['event']}"
        for e in timeline[-20:]  # last 20 events
    ) or "  No timeline data."

    prompt = f"""You are an expert incident commander reviewing an active incident.

=== INCIDENT CONTEXT ===
{json.dumps(incident_context, indent=2, default=str)[:1500]}

=== RECENT TIMELINE ===
{timeline_str}

Based on the current state of this incident, provide 3–5 specific, actionable next steps
the response team should take RIGHT NOW. Be concrete — name the actual commands, checks, or
escalations required. Format as a numbered list."""

    try:
        llm      = LLMFactory.get()
        response = llm.complete(
            prompt,
            system="You are an expert incident commander. Be direct and concise.",
            max_tokens=600,
        )
        suggestion = response.content.strip()
    except Exception as exc:
        _log(logger.error, "suggest_next_steps_failed", error=str(exc), war_room_id=war_room_id)
        suggestion = f"Could not generate suggestions: {exc}"

    # Post to Slack
    _post_to_slack_safe(
        wr.slack_channel,
        f":bulb: *Suggested Next Steps* | _{wr.incident_id}_\n{suggestion}",
    )

    return suggestion
