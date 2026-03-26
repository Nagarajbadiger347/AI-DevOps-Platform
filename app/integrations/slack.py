"""Slack integration — war room creation, incident reporting, threaded chat.

Requires env vars:
  SLACK_BOT_TOKEN  xoxb-... (bot token with channels:write, chat:write, groups:write)
  SLACK_CHANNEL    default channel for general alerts (e.g. #general)
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL   = os.getenv("SLACK_CHANNEL", "#general")


def _client() -> WebClient:
    return WebClient(token=SLACK_BOT_TOKEN)


def _safe_channel_name(raw: str) -> str:
    """Convert to valid Slack channel name: lowercase, hyphens, max 80 chars."""
    name = raw.lower().strip()
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name[:80]


# ── Basic messaging ───────────────────────────────────────────

def post_message(channel: str, text: str, blocks=None) -> dict:
    """Post a message to a channel."""
    if not SLACK_BOT_TOKEN:
        return {"success": False, "error": "SLACK_BOT_TOKEN not configured"}
    try:
        resp = _client().chat_postMessage(channel=channel, text=text, blocks=blocks)
        return {"success": True, "ts": resp["ts"], "channel": resp["channel"]}
    except SlackApiError as e:
        return {"success": False, "error": str(e), "detail": e.response.get("error")}


def post_thread_reply(channel: str, thread_ts: str, text: str) -> dict:
    """Reply in a thread."""
    if not SLACK_BOT_TOKEN:
        return {"success": False, "error": "SLACK_BOT_TOKEN not configured"}
    try:
        resp = _client().chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)
        return {"success": True, "ts": resp["ts"]}
    except SlackApiError as e:
        return {"success": False, "error": str(e)}


# ── War room ──────────────────────────────────────────────────

def create_war_room(topic: str = "War Room", members=None) -> dict:
    """Post a war room announcement to the default Slack channel.

    For full war room (dedicated channel), use create_incident_channel().
    """
    if not SLACK_BOT_TOKEN:
        return {"error": "SLACK_BOT_TOKEN not configured"}
    try:
        msg = f"🚨 War room opened: *{topic}*\nParticipants: {members or 'TBD'}"
        resp = _client().chat_postMessage(channel=SLACK_CHANNEL, text=msg)
        return {
            "room_url": f"https://slack.com/app_redirect?channel={SLACK_CHANNEL.strip('#')}",
            "ts": resp.get("ts"),
            "message": msg,
        }
    except SlackApiError as e:
        return {"error": str(e), "details": e.response.get("error")}


def create_incident_channel(incident_id: str, topic: str = "") -> dict:
    """Create a dedicated Slack channel for an incident (e.g. #inc-001).

    Returns the channel ID and URL for deep-linking.
    Requires channels:manage or groups:write scope on the bot token.
    """
    if not SLACK_BOT_TOKEN:
        return {"success": False, "error": "SLACK_BOT_TOKEN not configured"}

    channel_name = _safe_channel_name(f"inc-{incident_id}")
    sc = _client()

    # Try creating a public channel; fall back to private if permission denied
    try:
        resp = sc.conversations_create(name=channel_name, is_private=False)
    except SlackApiError as e:
        if e.response.get("error") == "name_taken":
            # Channel already exists — find it
            try:
                found = sc.conversations_list(types="public_channel,private_channel", limit=200)
                for ch in found.get("channels", []):
                    if ch["name"] == channel_name:
                        return {
                            "success": True,
                            "channel_id":   ch["id"],
                            "channel_name": ch["name"],
                            "channel_url":  f"https://slack.com/app_redirect?channel={ch['id']}",
                            "already_existed": True,
                        }
            except Exception:
                pass
            return {"success": False, "error": f"Channel #{channel_name} already exists but could not be found"}
        # If missing scope, try private
        try:
            resp = sc.conversations_create(name=channel_name, is_private=True)
        except SlackApiError as e2:
            return {"success": False, "error": str(e2), "detail": e2.response.get("error")}

    channel = resp["channel"]
    if topic:
        try:
            sc.conversations_setTopic(channel=channel["id"], topic=topic)
        except Exception:
            pass

    return {
        "success":      True,
        "channel_id":   channel["id"],
        "channel_name": channel["name"],
        "channel_url":  f"https://slack.com/app_redirect?channel={channel['id']}",
    }


def post_incident_summary(channel: str, incident_id: str, summary: str,
                          findings: list, severity: str = "high",
                          actions: list = None) -> dict:
    """Post a formatted incident analysis to a Slack channel using Block Kit."""
    if not SLACK_BOT_TOKEN:
        return {"success": False, "error": "SLACK_BOT_TOKEN not configured"}

    sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity.lower(), "⚪")

    findings_text = "\n".join(f"• {f}" for f in (findings or [])[:8]) or "_No findings_"
    actions_text  = "\n".join(f"• {a if isinstance(a, str) else a.get('reason', str(a))}" for a in (actions or [])[:5]) or "_No actions_"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"{sev_emoji} Incident {incident_id} — AI Analysis"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Summary:* {summary}"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Findings:*\n{findings_text}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Recommended Actions:*\n{actions_text}"}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "🤖 AI DevOps Intelligence Platform"}]},
    ]

    return post_message(channel, text=f"Incident {incident_id} analysis", blocks=blocks)
