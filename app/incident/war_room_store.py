"""Shared war room state store — importable by both main.py and runner.py."""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Optional

_WR_STORE = Path(
    os.getenv("WAR_ROOM_STORE_PATH", "") or
    (Path(__file__).resolve().parents[2] / "data" / "war_rooms.json")
)


@dataclasses.dataclass
class WarRoomSession:
    war_room_id:          str
    incident_id:          str
    incident_description: str
    pipeline_state:       dict
    slack_channel:        str
    created_at:           str = dataclasses.field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    participants: list = dataclasses.field(default_factory=list)


# Single in-process store — shared across all importers in the same process
WAR_ROOMS: dict[str, WarRoomSession] = {}


def save() -> None:
    try:
        _WR_STORE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            wid: {k: getattr(wr, k) for k in dataclasses.fields(wr)}
            for wid, wr in WAR_ROOMS.items()
        }
        tmp = _WR_STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(_WR_STORE)
    except Exception:
        pass


def load() -> None:
    try:
        with open(_WR_STORE) as f:
            data = json.load(f)
        for wid, d in data.items():
            WAR_ROOMS[wid] = WarRoomSession(
                war_room_id=d["war_room_id"],
                incident_id=d["incident_id"],
                incident_description=d["incident_description"],
                pipeline_state=d.get("pipeline_state", {}),
                slack_channel=d.get("slack_channel", ""),
                created_at=d.get("created_at", ""),
                participants=d.get("participants", []),
            )
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass


def create(incident_id: str, description: str, pipeline_state: dict,
           slack_channel: str) -> WarRoomSession:
    """Create or reuse the war room for this incident (one war room per incident)."""
    # Reuse existing active war room for the same incident
    for wr in WAR_ROOMS.values():
        if wr.incident_id == incident_id and wr.pipeline_state.get("status") != "resolved":
            # Update pipeline state with latest data
            wr.pipeline_state.update(pipeline_state)
            if slack_channel:
                wr.slack_channel = slack_channel
            save()
            return wr

    wr = WarRoomSession(
        war_room_id=str(uuid.uuid4()),
        incident_id=incident_id,
        incident_description=description,
        pipeline_state=pipeline_state,
        slack_channel=slack_channel,
    )
    WAR_ROOMS[wr.war_room_id] = wr
    save()

    # Pre-seed chat memory with incident context
    try:
        from app.chat.memory import get_or_create_session, add_message, set_context
        sid = f"war_room::{wr.war_room_id}"
        get_or_create_session(sid)
        set_context(sid, "war_room_id", wr.war_room_id)
        set_context(sid, "incident_id", incident_id)
        add_message(
            sid, "system",
            f"War room opened for incident {incident_id}: {description}\n"
            f"Context: {json.dumps(pipeline_state, default=str)[:1500]}",
            metadata={"type": "war_room_init"},
        )
    except Exception:
        pass

    return wr


def answer(war_room_id: str, question: str, asked_by: str) -> str:
    """Answer a question in the context of a war room."""
    wr = WAR_ROOMS.get(war_room_id)
    if not wr:
        return f"War room {war_room_id} not found."
    if asked_by and asked_by not in wr.participants:
        wr.participants.append(asked_by)
    try:
        from app.chat.intelligence import chat_with_intelligence
        incident_context = {
            "war_room_id":          wr.war_room_id,
            "incident_id":          wr.incident_id,
            "incident_description": wr.incident_description,
            "root_cause":           wr.pipeline_state.get("root_cause", "Under investigation"),
            "actions_taken":        wr.pipeline_state.get("actions_taken",
                                    wr.pipeline_state.get("executed_actions", [])),
            "current_status":       wr.pipeline_state.get("status", "active"),
            "severity":             wr.pipeline_state.get("severity", "SEV2"),
            "slack_channel":        wr.slack_channel,
            "pipeline_state":       wr.pipeline_state,
        }
        result = chat_with_intelligence(
            message=question,
            session_id=f"war_room::{war_room_id}",
            incident_context=incident_context,
        )
    except Exception as exc:
        result = f"Could not process question: {exc}"

    # Post answer to Slack
    try:
        if wr.slack_channel:
            from app.integrations.slack import post_message
            post_message(
                channel=wr.slack_channel,
                text=f":robot_face: *NsOps AI* | _{wr.incident_id}_\n"
                     f"*{asked_by} asked:* {question}\n\n{result}",
            )
    except Exception:
        pass

    return result


def timeline(war_room_id: str) -> list:
    wr = WAR_ROOMS.get(war_room_id)
    if not wr:
        return []
    events = [{"timestamp": wr.created_at,
               "event": f"War room created for {wr.incident_id}: {wr.incident_description}",
               "actor": "system", "source": "war_room"}]
    for a in (wr.pipeline_state.get("actions_taken") or wr.pipeline_state.get("executed_actions") or []):
        if isinstance(a, dict):
            events.append({"timestamp": a.get("executed_at", wr.created_at),
                           "event": f"Action: {a.get('type','?')} — {a.get('description','')}",
                           "actor": "pipeline", "source": "pipeline_state"})
    try:
        from app.chat.memory import get_history
        for msg in get_history(f"war_room::{war_room_id}", max_messages=100):
            role = getattr(msg, "role", "")
            if role in ("user", "assistant"):
                events.append({"timestamp": getattr(msg, "timestamp", ""),
                               "event": getattr(msg, "content", "")[:200],
                               "actor": role, "source": "conversation"})
    except Exception:
        pass
    events.sort(key=lambda e: e.get("timestamp") or "")
    return events


load()  # load persisted war rooms on import
