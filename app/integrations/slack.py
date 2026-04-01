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
                          actions: list = None, root_cause: str = "",
                          confidence: float = 0.0, infra_context: dict = None,
                          pr_links: list = None) -> dict:
    """Post a rich incident analysis to a Slack channel using Block Kit.

    Includes root cause, findings, infra snapshot, PR links, and action plan.
    """
    if not SLACK_BOT_TOKEN:
        return {"success": False, "error": "SLACK_BOT_TOKEN not configured"}

    sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity.lower(), "⚪")
    conf_bar  = "█" * int((confidence or 0) * 10) + "░" * (10 - int((confidence or 0) * 10))

    findings_text = "\n".join(f"• {f}" for f in (findings or [])[:10]) or "_No findings detected_"

    # Build action lines with type + reason
    action_lines = []
    for a in (actions or [])[:8]:
        if isinstance(a, str):
            action_lines.append(f"• {a}")
        else:
            atype  = a.get("type", "action")
            reason = a.get("reason", "")
            params = a.get("params", {})
            line = f"• *[{atype}]* {reason}"
            if params:
                detail = ", ".join(f"{k}={v}" for k, v in params.items() if v)
                if detail:
                    line += f" `{detail}`"
            action_lines.append(line)
    actions_text = "\n".join(action_lines) or "_No actions required_"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"{sev_emoji} Incident {incident_id} — AI War Room"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Severity:*\n{sev_emoji} {severity.upper()}"},
            {"type": "mrkdwn", "text": f"*AI Confidence:*\n`{conf_bar}` {int((confidence or 0)*100)}%"},
        ]},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*📋 Summary*\n{summary}"}},
    ]

    if root_cause:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*🔍 Root Cause*\n{root_cause}"}})

    blocks.append({"type": "divider"})
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*🔬 Findings*\n{findings_text}"}})

    # Infrastructure snapshot
    if infra_context:
        infra_lines = []
        aws = infra_context.get("aws", {})
        k8s = infra_context.get("k8s", {})
        gh  = infra_context.get("github", {})
        if aws.get("alarms"):
            firing = [a for a in aws.get("alarms", []) if str(a.get("state", "")).upper() == "ALARM"]
            if firing:
                infra_lines.append(f"• *AWS Alarms firing:* {len(firing)} — " + ", ".join(a.get("name", "") for a in firing[:3]))
        if aws.get("instances"):
            unhealthy = [i for i in aws.get("instances", []) if i.get("state") not in ("running", "stopped")]
            if unhealthy:
                infra_lines.append(f"• *EC2 unhealthy:* {', '.join(i.get('id','') for i in unhealthy[:3])}")
        if k8s.get("unhealthy_pods"):
            infra_lines.append(f"• *K8s unhealthy pods:* {len(k8s['unhealthy_pods'])} — " + ", ".join(p.get("name","") for p in k8s["unhealthy_pods"][:3]))
        if gh.get("recent_commits"):
            last = gh["recent_commits"][0] if gh["recent_commits"] else {}
            if last:
                infra_lines.append(f"• *Last commit:* `{last.get('sha','')[:7]}` {last.get('message','')[:60]} by {last.get('author','')}")
        if infra_lines:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "*🏗️ Infrastructure Snapshot*\n" + "\n".join(infra_lines)}})

    # PR links
    if pr_links:
        pr_text = "\n".join(f"• <{url}|{title}>" for title, url in pr_links)
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*🔀 Related Pull Requests*\n{pr_text}"}})

    blocks.append({"type": "divider"})
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*⚡ Action Plan*\n{actions_text}"}})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "🤖 AI DevOps Intelligence Platform — automated incident analysis"}]})

    return post_message(channel, text=f"{sev_emoji} Incident {incident_id}: {summary}", blocks=blocks)
