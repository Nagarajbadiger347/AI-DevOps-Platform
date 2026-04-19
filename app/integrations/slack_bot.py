"""Slack War Room Bot — answers questions asked inside incident channels.

When someone asks a question in a #inc-* channel (e.g. "which PR raised this?"
or "show last 30 min Grafana alerts"), this bot fetches live data and replies
in the thread.

Requires env vars:
  SLACK_BOT_TOKEN       xoxb-... (existing)
  SLACK_SIGNING_SECRET  from Slack app settings → Basic Information
  SLACK_BOT_USER_ID     your bot's Slack user ID (to avoid replying to itself)

Enable in Slack app settings:
  Event Subscriptions → Request URL: https://your-domain/slack/events
  Subscribe to bot events: message.channels, message.groups, app_mention
"""

import hashlib
import hmac
import json
import logging
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.integrations.slack import post_message, post_thread_reply, _client

log = logging.getLogger("slack_bot")

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
SLACK_BOT_USER_ID    = os.getenv("SLACK_BOT_USER_ID", "")
SLACK_BOT_TOKEN      = os.getenv("SLACK_BOT_TOKEN", "")


# ── Signature verification ─────────────────────────────────────

def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """Verify the request came from Slack using HMAC-SHA256."""
    if not SLACK_SIGNING_SECRET:
        return True  # Skip verification in dev if secret not set
    if abs(time.time() - int(timestamp)) > 300:
        return False  # Replay attack protection
    base = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(
        SLACK_SIGNING_SECRET.encode(), base.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


# ── Intent detection ───────────────────────────────────────────

_INTENTS = {
    "pr":          re.compile(r"pr|pull request|which (commit|change|deploy)|raised this|what changed", re.I),
    "logs":        re.compile(r"log|grafana|kibana|splunk|last \d+\s*(min|hour|day)", re.I),
    "k8s":         re.compile(r"pod|deploy|k8s|kubernetes|namespace|restart|replica|crash|oom", re.I),
    "aws":         re.compile(r"ec2|ecs|lambda|rds|s3|alarm|cloudwatch|sqs|dynamo|aws", re.I),
    "alerts":      re.compile(r"alert|firing|alarm|pagerduty|opsgenie|oncall|on.call", re.I),
    "summary":     re.compile(r"summar|what.s happening|status|overview|root cause|why", re.I),
    "actions":     re.compile(r"what (should|do) (we|i)|next step|action|fix|resolve|remediat", re.I),
}

def _detect_intent(text: str) -> list[str]:
    return [intent for intent, pattern in _INTENTS.items() if pattern.search(text)]


# ── Data fetchers ──────────────────────────────────────────────

def _fetch_pr_context(question: str) -> str:
    try:
        from app.integrations.github import get_recent_commits, get_recent_prs
        hours = 48
        m = re.search(r"last\s+(\d+)\s*(day|hour|week)", question, re.I)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            hours = n * (24 if "day" in unit else 168 if "week" in unit else 1)

        commits = get_recent_commits(hours=hours)
        prs     = get_recent_prs(hours=hours)

        lines = []
        for pr in (prs.get("pull_requests") or prs.get("prs", []))[:5]:
            url    = pr.get("url") or pr.get("html_url", "")
            title  = pr.get("title", "")
            author = pr.get("author") or pr.get("user", {}).get("login", "")
            merged = pr.get("merged_at") or pr.get("updated_at", "")
            lines.append(f"• <{url}|{title}> by {author} — {merged[:10] if merged else ''}")

        for c in (commits.get("commits", []))[:3]:
            sha  = c.get("sha", "")[:7]
            msg  = c.get("message", "")[:80]
            auth = c.get("author", "")
            lines.append(f"• commit `{sha}` — {msg} ({auth})")

        return "\n".join(lines) if lines else "No recent PRs or commits found."
    except Exception as e:
        return f"Could not fetch PR data: {e}"


def _fetch_grafana_context(question: str) -> str:
    try:
        from app.integrations.grafana import get_firing_alerts, get_annotations
        hours = 1
        m = re.search(r"last\s+(\d+)\s*(min|hour|day)", question, re.I)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            hours = max(1, n // 60 if "min" in unit else n * 24 if "day" in unit else n)

        alerts = get_firing_alerts()
        annots = get_annotations(hours=hours)

        lines = []
        for a in (alerts.get("firing_alerts", []))[:5]:
            lines.append(f"• *FIRING* {a.get('name','')} — {a.get('summary','')} [{a.get('severity','')}]")
        for a in (annots.get("annotations", []))[:3]:
            lines.append(f"• Annotation: {a.get('text','')[:100]} ({a.get('tags',[])})")

        return "\n".join(lines) if lines else "No firing Grafana alerts right now."
    except Exception as e:
        return f"Could not fetch Grafana data: {e}"


def _fetch_k8s_context(question: str) -> str:
    try:
        from app.integrations.k8s_ops import get_unhealthy_pods, list_deployments
        ns = "default"
        m = re.search(r"namespace[=:\s]+(\S+)", question, re.I)
        if m:
            ns = m.group(1)

        unhealthy = get_unhealthy_pods()
        deps      = list_deployments(ns)

        lines = []
        for p in (unhealthy.get("unhealthy_pods") or unhealthy.get("pods", []))[:5]:
            lines.append(f"• Pod `{p.get('name','')}` in `{p.get('namespace','')}` — {p.get('phase','')} ({p.get('reason','')})")
        for d in (deps.get("deployments", []))[:4]:
            ready = d.get("ready_replicas", 0)
            total = d.get("replicas", 0)
            lines.append(f"• Deployment `{d.get('name','')}` — {ready}/{total} ready")

        return "\n".join(lines) if lines else "All K8s pods appear healthy."
    except Exception as e:
        return f"Could not fetch K8s data: {e}"


def _fetch_aws_context(question: str) -> str:
    try:
        from app.integrations.aws_ops import list_cloudwatch_alarms, list_ec2_instances
        alarms  = list_cloudwatch_alarms(state="ALARM")
        ec2     = list_ec2_instances()

        lines = []
        for a in (alarms.get("alarms", []))[:5]:
            lines.append(f"• Alarm `{a.get('name','')}` — {a.get('state','')} — {a.get('reason','')[:80]}")
        for i in (ec2.get("instances", []))[:4]:
            if i.get("state") != "running":
                lines.append(f"• EC2 `{i.get('id','')}` ({i.get('name','')}) — {i.get('state','')}")

        return "\n".join(lines) if lines else "No AWS alarms firing, all EC2 instances running."
    except Exception as e:
        return f"Could not fetch AWS data: {e}"


# ── AI answer builder ──────────────────────────────────────────

def _get_war_room_context(incident_id: str) -> dict:
    """Fetch the war room context for this incident."""
    try:
        from app.incident.war_room_store import WAR_ROOMS
        wr = next((w for w in WAR_ROOMS.values() if w.incident_id == incident_id), None)
        if wr:
            return {
                "war_room_id":   wr.war_room_id,
                "description":   wr.incident_description,
                "root_cause":    wr.pipeline_state.get("root_cause", ""),
                "summary":       wr.pipeline_state.get("summary", ""),
                "severity":      wr.pipeline_state.get("severity", ""),
                "status":        wr.pipeline_state.get("status", "active"),
                "findings":      wr.pipeline_state.get("findings", []),
                "fix_suggestion":wr.pipeline_state.get("fix_suggestion", ""),
                "executed_actions": wr.pipeline_state.get("executed_actions") or wr.pipeline_state.get("actions_taken", []),
                "blocked_actions":  wr.pipeline_state.get("blocked_actions", []),
            }
    except Exception:
        pass
    return {}


def _build_answer(question: str, intents: list, data_snippets: dict, incident_id: str) -> str:
    """Pass question + war room context + live data to LLM for a coherent Slack reply."""
    try:
        from app.llm.claude import _llm

        # Pull stored war room context (root cause, summary, actions)
        wr_ctx = _get_war_room_context(incident_id)

        # Build incident context block
        incident_lines = [f"*Incident ID:* {incident_id}"]
        if wr_ctx.get("description"):
            incident_lines.append(f"*Description:* {wr_ctx['description']}")
        if wr_ctx.get("severity"):
            incident_lines.append(f"*Severity:* {wr_ctx['severity'].upper()}")
        if wr_ctx.get("status"):
            incident_lines.append(f"*Status:* {wr_ctx['status'].upper()}")
        if wr_ctx.get("root_cause"):
            incident_lines.append(f"*Root Cause:* {wr_ctx['root_cause']}")
        if wr_ctx.get("summary"):
            incident_lines.append(f"*AI Summary:* {wr_ctx['summary']}")
        if wr_ctx.get("fix_suggestion"):
            incident_lines.append(f"*Recommended Fix:* {wr_ctx['fix_suggestion']}")
        executed = wr_ctx.get("executed_actions", [])
        if executed:
            acts = ", ".join(a.get("type", "?") for a in executed[:5])
            incident_lines.append(f"*Actions Executed:* {acts}")
        blocked = wr_ctx.get("blocked_actions", [])
        if blocked:
            blk = ", ".join(a.get("type", "?") for a in blocked[:5])
            incident_lines.append(f"*Blocked (need approval):* {blk}")

        incident_context = "\n".join(incident_lines)

        # Live infra data from fetchers
        live_data = "\n\n".join(
            f"=== {k.upper()} ===\n{v}"
            for k, v in data_snippets.items() if v
        )

        system = (
            "You are a DevOps AI assistant inside a Slack war room for an active incident. "
            "You have access to the incident's root cause, AI analysis, and live infrastructure data. "
            "Answer the engineer's question directly and specifically based on THIS incident's context. "
            "Do NOT give generic advice — reference the actual incident details, root cause, and actions taken. "
            "Use Slack mrkdwn formatting (*bold*, `code`, • bullets). "
            "Keep the reply under 400 words. If data is missing, say what you know and what to check next."
        )
        prompt = (
            f"=== INCIDENT CONTEXT ===\n{incident_context}\n\n"
            f"=== LIVE INFRA DATA ===\n{live_data or 'No live data available.'}\n\n"
            f"=== ENGINEER QUESTION ===\n{question}"
        )
        return _llm(system, [{"role": "user", "content": prompt}], max_tokens=700)
    except Exception as e:
        raw = "\n\n".join(f"*{k.upper()}*\n{v}" for k, v in data_snippets.items() if v)
        return raw or f"Could not process question: {e}"


# ── Main bot handler ───────────────────────────────────────────

def handle_slack_event(payload: dict) -> dict:
    """Process a Slack Events API payload. Returns dict for the HTTP response."""

    # URL verification challenge (one-time setup)
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    event = payload.get("event", {})
    etype = event.get("type", "")

    # Only handle actual messages, not bot messages or edits
    if etype not in ("message", "app_mention"):
        return {"ok": True}
    if event.get("bot_id") or event.get("subtype"):
        return {"ok": True}
    # Don't reply to ourselves
    if SLACK_BOT_USER_ID and event.get("user") == SLACK_BOT_USER_ID:
        return {"ok": True}

    channel  = event.get("channel", "")
    user_id  = event.get("user", "")
    text     = re.sub(r"<@\w+>", "", event.get("text", "")).strip()
    thread   = event.get("thread_ts") or event.get("ts", "")

    if not text or not channel:
        return {"ok": True}

    # Only respond in war room incident channels (inc-*) or when directly mentioned
    channel_name = ""
    try:
        info = _client().conversations_info(channel=channel)
        channel_name = info.get("channel", {}).get("name", "")
    except Exception:
        pass

    is_warroom  = channel_name.startswith("inc-")
    is_mention  = etype == "app_mention"
    if not is_warroom and not is_mention:
        return {"ok": True}

    # Extract incident ID from channel name (inc-<incident_id>)
    incident_id = channel_name.replace("inc-", "", 1) if is_warroom else "unknown"

    # Detect what data to fetch
    intents = _detect_intent(text)
    if not intents:
        intents = ["summary"]  # Default: give a general status

    # Fetch relevant data in parallel
    from concurrent.futures import ThreadPoolExecutor
    fetchers = {
        "pr":      _fetch_pr_context,
        "logs":    _fetch_grafana_context,
        "k8s":     _fetch_k8s_context,
        "aws":     _fetch_aws_context,
        "alerts":  _fetch_grafana_context,
        "summary": _fetch_aws_context,
        "actions": _fetch_k8s_context,
    }

    data_snippets = {}
    to_fetch = {i: fetchers[i] for i in intents if i in fetchers}
    # Deduplicate same fetcher
    seen_fetchers = set()
    deduped = {}
    for key, fn in to_fetch.items():
        if fn not in seen_fetchers:
            deduped[key] = fn
            seen_fetchers.add(fn)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn, text): key for key, fn in deduped.items()}
        for future in futures:
            key = futures[future]
            try:
                data_snippets[key] = future.result(timeout=10)
            except Exception as e:
                data_snippets[key] = f"Fetch timed out: {e}"

    # Build AI answer
    answer = _build_answer(text, intents, data_snippets, incident_id)

    # Post reply in thread
    post_thread_reply(channel=channel, thread_ts=thread, text=answer)
    log.info("War room bot replied in %s (intents: %s)", channel_name, intents)

    # Mirror Q&A into the in-app war room chat so UI users see Slack conversations too
    _mirror_to_war_room(incident_id, user_id, text, answer)

    return {"ok": True}


def _mirror_to_war_room(incident_id: str, slack_user: str, question: str, answer: str) -> None:
    """Push the Slack Q&A into the matching in-app war room chat history."""
    try:
        from app.incident.war_room_store import WAR_ROOMS
        from app.chat.memory import get_or_create_session, add_message

        # Find the war room for this incident
        wr = next((w for w in WAR_ROOMS.values() if w.incident_id == incident_id), None)
        if not wr:
            return

        sid = f"war_room::{wr.war_room_id}"
        get_or_create_session(sid)

        # Add user question (tagged as from Slack)
        add_message(sid, "user", question, metadata={
            "source":     "slack",
            "slack_user": slack_user,
            "channel":    f"inc-{incident_id}",
        })
        # Add bot answer
        add_message(sid, "assistant", answer, metadata={
            "source": "slack_bot",
        })
        log.info("Mirrored Slack Q&A to war room %s", wr.war_room_id)
    except Exception as exc:
        log.warning("mirror_to_war_room failed incident=%s error=%s", incident_id, exc)
