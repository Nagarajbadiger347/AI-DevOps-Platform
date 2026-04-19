"""
War room and Slack events routes.
Paths: /warroom/*, /slack/events
"""
import os
import asyncio
import datetime as _datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.api.deps import (
    require_developer, require_viewer, AuthContext,
    _WAR_ROOMS, _wr_save, _create_war_room, _answer_war_room_question, _wr_timeline,
    _RECENT_RESULTS,
)
from app.integrations.slack import create_incident_channel, post_incident_summary, post_message
from app.integrations.universal_collector import collect_all_context

router = APIRouter(tags=["warroom"])

_SLACK_BOT_USER_ID: str = os.getenv("SLACK_BOT_USER_ID", "")


class WarRoomRequest(BaseModel):
    incident_id:   str
    description:   str
    severity:      str = "high"
    post_to_slack: bool = True


class WarRoomQuestion(BaseModel):
    question: str
    asked_by: Optional[str] = None


class SlackSendRequest(BaseModel):
    message:  str
    sent_by:  str = "user"


@router.post("/warroom/create")
def warroom_create(req: WarRoomRequest, auth: AuthContext = Depends(require_developer)):
    """Create a war room: collect universal context, run AI analysis, create Slack channel, post findings."""
    context: dict = {}
    try:
        context = collect_all_context(hours=2)
    except Exception:
        pass

    from app.llm.claude import synthesize_incident
    synthesis = synthesize_incident({
        "incident_id":    req.incident_id,
        "description":    req.description,
        "severity":       req.severity,
        "aws_context":    context.get("aws", {}),
        "k8s_context":    context.get("k8s", {}),
        "github_context": context.get("github", {}),
    })

    slack_channel = ""
    slack_info = None

    if req.post_to_slack:
        channel_result = create_incident_channel(req.incident_id, topic=f"{req.severity.upper()} — {req.description[:80]}")
        if channel_result.get("success"):
            channel_id   = channel_result["channel_id"]
            slack_channel = channel_result.get("channel_name", channel_id)
            post_incident_summary(
                channel     = channel_id,
                incident_id = req.incident_id,
                summary     = synthesis.get("summary", req.description),
                findings    = synthesis.get("findings", []),
                severity    = synthesis.get("severity", req.severity),
                actions     = synthesis.get("actions_to_take", []),
            )
            slack_info = {
                "channel_name": channel_result.get("channel_name"),
                "channel_url":  channel_result.get("channel_url"),
            }
        else:
            slack_info = {"error": channel_result.get("error")}

    # Extract key resource identifiers from AWS context for AI reference
    aws_ctx = context.get("aws", {})
    ec2_data = aws_ctx.get("ec2", {})  # context["aws"]["ec2"] from universal_collector
    instances = ec2_data.get("instances", []) if isinstance(ec2_data, dict) else []
    instance_ids = [
        {"id": i.get("id",""), "name": i.get("name",""), "state": i.get("state",""), "type": i.get("type","")}
        for i in instances if isinstance(i, dict) and i.get("id")
    ]

    pipeline_state = {
        "root_cause":     synthesis.get("root_cause", "Under investigation"),
        "summary":        synthesis.get("summary", req.description),
        "findings":       synthesis.get("findings", []),
        "fix_suggestion": synthesis.get("fix_suggestion", synthesis.get("recommended_actions", "")),
        "severity":       synthesis.get("severity", req.severity),
        "status":         "active",
        "actions_taken":  synthesis.get("actions_to_take", []),
        "ec2_instances":  instance_ids,
        "aws_region":     ec2_data.get("region", "") if isinstance(ec2_data, dict) else "",
        "aws_context":    aws_ctx,
        "k8s_context":    context.get("k8s", {}),
        "github_context": context.get("github", {}),
    }
    war_room = _create_war_room(
        incident_id    = req.incident_id,
        description    = req.description,
        pipeline_state = pipeline_state,
        slack_channel  = slack_channel,
    )

    return {
        "war_room_id":   war_room.war_room_id,
        "incident_id":   req.incident_id,
        "slack_channel": slack_channel,
        "analysis":      synthesis,
        "sources":       context.get("configured", []),
        "slack":         slack_info,
        "created_at":    war_room.created_at,
    }


@router.post("/warroom/{war_room_id}/ask")
async def ask_war_room_ai(war_room_id: str, req: WarRoomQuestion, auth: AuthContext = Depends(require_viewer)):
    if war_room_id not in _WAR_ROOMS:
        raise HTTPException(status_code=404, detail=f"War room {war_room_id} not found")
    try:
        answer = _answer_war_room_question(war_room_id, req.question, req.asked_by or auth.username)
        return {"answer": answer, "war_room_id": war_room_id}
    except Exception as e:
        return {"answer": f"War room AI unavailable: {e}", "war_room_id": war_room_id}


@router.get("/warroom/{war_room_id}/history")
def get_war_room_history(war_room_id: str, auth: AuthContext = Depends(require_viewer)):
    """Return full chat history for a war room, including messages mirrored from Slack."""
    if war_room_id not in _WAR_ROOMS:
        raise HTTPException(status_code=404, detail=f"War room {war_room_id} not found")
    try:
        from app.chat.memory import get_history
        raw = get_history(f"war_room::{war_room_id}", max_messages=200) or []
        messages = [
            {"role": m.role, "content": m.content,
             "metadata": m.metadata or {}, "ts": m.timestamp}
            for m in raw
        ]
        return {"war_room_id": war_room_id, "history": messages or []}
    except Exception as exc:
        return {"war_room_id": war_room_id, "history": [], "error": str(exc)}


@router.get("/warroom/active")
def list_active_war_rooms(auth: AuthContext = Depends(require_viewer)):
    return {"war_rooms": [
        {"war_room_id": wr.war_room_id, "incident_id": wr.incident_id,
         "description": wr.incident_description, "slack_channel": wr.slack_channel,
         "created_at": wr.created_at, "participants": len(wr.participants),
         "severity": wr.pipeline_state.get("severity", "SEV2")}
        for wr in _WAR_ROOMS.values()
        if wr.pipeline_state.get("status") != "resolved"
    ]}


@router.get("/warroom/resolved")
def list_resolved_war_rooms(auth: AuthContext = Depends(require_viewer)):
    return {"war_rooms": [
        {"war_room_id": wr.war_room_id, "incident_id": wr.incident_id,
         "description": wr.incident_description, "slack_channel": wr.slack_channel,
         "created_at": wr.created_at, "participants": len(wr.participants),
         "severity": wr.pipeline_state.get("severity", "low")}
        for wr in _WAR_ROOMS.values()
        if wr.pipeline_state.get("status") == "resolved"
    ]}


@router.get("/warroom/{war_room_id}/timeline")
def get_war_room_timeline(war_room_id: str, auth: AuthContext = Depends(require_viewer)):
    events = _wr_timeline(war_room_id)
    return {"timeline": events, "count": len(events)}


@router.get("/warroom/{war_room_id}/slack-history")
def get_war_room_slack_history(war_room_id: str, limit: int = 30, auth: AuthContext = Depends(require_viewer)):
    """Fetch recent messages from the Slack channel linked to this war room."""
    try:
        wr = _WAR_ROOMS.get(war_room_id)
        channel = wr.slack_channel if wr else ""
        if not channel:
            return {"messages": [], "channel": "", "note": "No Slack channel linked to this war room"}
        from app.integrations.slack import _client as _slack_client
        sc = _slack_client()
        ch_id = channel
        if not channel.startswith("C"):
            name = channel.lstrip("#")
            result = sc.conversations_list(types="public_channel,private_channel", limit=500)
            for ch in result.get("channels", []):
                if ch["name"] == name:
                    ch_id = ch["id"]
                    break
        resp = sc.conversations_history(channel=ch_id, limit=limit)
        messages = []
        for m in reversed(resp.get("messages", [])):
            if m.get("subtype"):
                continue
            ts = float(m.get("ts", 0))
            time_str = _datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
            username = m.get("username") or m.get("user", "unknown")
            messages.append({"username": username, "text": m.get("text", ""), "time": time_str, "ts": m.get("ts", "")})
        return {"messages": messages, "channel": channel, "count": len(messages)}
    except Exception as e:
        return {"messages": [], "channel": "", "error": str(e)}


@router.post("/warroom/{war_room_id}/slack-send")
def send_war_room_slack_message(war_room_id: str, req: SlackSendRequest, auth: AuthContext = Depends(require_viewer)):
    try:
        wr = _WAR_ROOMS.get(war_room_id)
        channel = wr.slack_channel if wr else ""
        if not channel:
            raise HTTPException(status_code=400, detail="No Slack channel linked to this war room")
        text = f"*{req.sent_by}* (via NsOps): {req.message}"
        result = post_message(channel=channel, text=text)
        return {"success": result.get("ok", False), "channel": channel}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/warroom/{war_room_id}/resolve")
def resolve_war_room(war_room_id: str, auth: AuthContext = Depends(require_developer)):
    wr = _WAR_ROOMS.get(war_room_id)
    if not wr:
        raise HTTPException(status_code=404, detail=f"War room {war_room_id} not found")
    wr.pipeline_state["status"] = "resolved"
    _wr_save()
    try:
        if wr.slack_channel:
            post_message(channel=wr.slack_channel, text=f":white_check_mark: War room for *{wr.incident_id}* marked as resolved by {auth.username}.")
    except Exception:
        pass
    return {"success": True, "war_room_id": war_room_id, "resolved_by": auth.username}


@router.delete("/warroom/{war_room_id}")
def delete_war_room(war_room_id: str, auth: AuthContext = Depends(require_developer)):
    """Permanently delete a war room."""
    if war_room_id not in _WAR_ROOMS:
        raise HTTPException(status_code=404, detail=f"War room {war_room_id} not found")
    del _WAR_ROOMS[war_room_id]
    _wr_save()
    return {"success": True, "war_room_id": war_room_id, "deleted_by": auth.username}


@router.post("/slack/events", include_in_schema=False)
async def slack_events_webhook(request: Request):
    """Receive Slack Events API payloads and route channel messages to war room AI."""
    body = await request.json()

    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    event = body.get("event", {})
    etype = event.get("type")

    if etype not in ("message", "message.channels", "message.groups"):
        return {"ok": True}
    if event.get("subtype"):
        return {"ok": True}
    bot_id  = event.get("bot_id") or ""
    user_id = event.get("user") or ""
    if bot_id or user_id == _SLACK_BOT_USER_ID:
        return {"ok": True}

    channel_id = event.get("channel", "")
    text = (event.get("text") or "").strip()
    username = user_id or "slack_user"

    if not text or not channel_id:
        return {"ok": True}

    ch_name_resolved = ""
    try:
        from app.integrations.slack import _client as _sc, SLACK_BOT_TOKEN
        if SLACK_BOT_TOKEN:
            sc = _sc()
            info = sc.conversations_info(channel=channel_id)
            ch_name_resolved = info.get("channel", {}).get("name", "")
            if user_id:
                try:
                    uinfo = sc.users_info(user=user_id)
                    username = uinfo.get("user", {}).get("real_name") or uinfo.get("user", {}).get("name") or user_id
                except Exception:
                    pass
    except Exception:
        pass

    matched_wr = None
    for wr in _WAR_ROOMS.values():
        if wr.pipeline_state.get("status") == "resolved":
            continue
        ch = wr.slack_channel.lstrip("#")
        if ch in (channel_id, ch_name_resolved):
            matched_wr = wr
            break

    if matched_wr:
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _answer_war_room_question(matched_wr.war_room_id, text, username)
        )
        return {"ok": True}

    import re as _re
    inc_match = _re.match(r'^inc[-_](.+)$', ch_name_resolved or "")
    if not inc_match:
        return {"ok": True}

    incident_id = inc_match.group(1)
    cached = _RECENT_RESULTS.get(incident_id, {})
    plan = cached.get("plan") or {}
    incident_context = {
        "incident_id":          incident_id,
        "incident_description": cached.get("description") or incident_id,
        "root_cause":           plan.get("root_cause") or "Under investigation",
        "actions_taken":        cached.get("executed_actions") or [],
        "blocked_actions":      cached.get("blocked_actions") or [],
        "current_status":       cached.get("status") or "unknown",
        "risk":                 plan.get("risk") or "unknown",
        "confidence":           plan.get("confidence") or 0,
        "plan_actions":         plan.get("actions") or [],
        "aws_available":        bool((cached.get("aws_context") or {}).get("_data_available")),
        "k8s_available":        bool((cached.get("k8s_context") or {}).get("_data_available")),
    }

    def _answer_incident_channel(q: str, u: str, ch: str, ctx: dict) -> None:
        try:
            from app.chat.intelligence import chat_with_intelligence
            from app.integrations.slack import post_message as _pm
            answer = chat_with_intelligence(
                message=q,
                session_id=f"slack_incident::{ch}",
                incident_context=ctx,
            )
            _pm(channel=ch, text=f":robot_face: *NsOps AI* | _{ctx['incident_id']}_\n*{u} asked:* {q}\n\n{answer}")
        except Exception:
            pass

    asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _answer_incident_channel(text, username, channel_id, incident_context)
    )

    return {"ok": True}
