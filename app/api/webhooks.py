"""
Webhook receivers (GitHub, PagerDuty, Grafana, CloudWatch, OpsGenie).
Paths: /webhooks/*
"""
import os
from typing import Optional, Any

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class GitHubWebhookPayload(BaseModel):
    action: str = ""
    ref: str = ""
    commits: list = []
    pull_request: dict = {}
    repository: dict = {}


class PagerDutyWebhookPayload(BaseModel):
    messages: list = []


@router.post("/github", tags=["Webhooks"])
async def webhook_github(
    request: Request,
    payload: GitHubWebhookPayload,
    x_github_event: str = Header("", alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header("", alias="X-Hub-Signature-256"),
):
    """Receive GitHub push/PR events and trigger pipeline automatically."""
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "").strip()
    if webhook_secret and x_hub_signature_256:
        import hmac, hashlib
        body = await request.body()
        expected = "sha256=" + hmac.new(
            webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, x_hub_signature_256):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Invalid webhook signature"})
    event = x_github_event
    if event == "push":
        commits = payload.commits or []
        if not commits:
            return {"status": "skipped", "reason": "no commits"}
        desc = f"GitHub push to {payload.ref}: {commits[0].get('message', '')[:120]}"
        incident_id = f"gh-push-{payload.ref.split('/')[-1]}-{len(commits)}c"
        from app.orchestrator.runner import run_pipeline
        result = run_pipeline(
            incident_id=incident_id,
            description=desc,
            severity="medium",
            auto_remediate=False,
        )
        return {"status": "triggered", "incident_id": incident_id, "pipeline": result.get("status")}
    elif event == "pull_request":
        pr = payload.pull_request
        if payload.action not in ("opened", "synchronize"):
            return {"status": "skipped", "reason": f"action={payload.action}"}
        pr_num = pr.get("number")
        if pr_num:
            from app.llm.claude import review_pr
            from app.integrations.github import get_pr_for_review, post_pr_review_comment
            pr_data = get_pr_for_review(pr_num)
            if pr_data.get("success"):
                review = review_pr(pr_data)
                post_pr_review_comment(pr_num, review)
                return {"status": "reviewed", "pr": pr_num}
        return {"status": "skipped", "reason": "no pr number"}
    return {"status": "ignored", "event": event}


@router.post("/pagerduty", tags=["Webhooks"])
async def webhook_pagerduty(
    request: Request,
    x_pagerduty_signature: str = Header("", alias="X-PagerDuty-Signature"),
):
    """Receive PagerDuty v2/v3 incident webhooks and auto-triage.

    Supports both legacy messages[] format and v3 event.data format.
    Verifies HMAC-SHA256 signature when PAGERDUTY_WEBHOOK_SECRET is set.
    On trigger: creates Slack war room, collects context, posts enriched brief,
    then fires the AI pipeline asynchronously.
    """
    raw_body = await request.body()

    # Signature verification
    pd_secret = os.getenv("PAGERDUTY_WEBHOOK_SECRET", "").strip()
    if pd_secret and x_pagerduty_signature:
        import hmac, hashlib
        sig_body = f"v1:{raw_body.decode()}"
        expected = "v1=" + hmac.new(pd_secret.encode(), raw_body, hashlib.sha256).hexdigest()
        sigs = x_pagerduty_signature.split(",")
        if not any(hmac.compare_digest(expected, s.strip()) for s in sigs):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=401, content={"detail": "Invalid PagerDuty signature"})

    try:
        body = await request.json() if not raw_body else __import__("json").loads(raw_body)
    except Exception:
        body = {}

    # Normalise payload — support both v2 (messages[]) and v3 (event.data)
    incidents_to_triage: list[dict] = []

    # v3 format: {"event": {"event_type": "incident.triggered", "data": {...}}}
    if "event" in body:
        ev = body["event"]
        if ev.get("event_type", "").startswith("incident.trigger"):
            data = ev.get("data", {})
            incidents_to_triage.append({
                "id":      data.get("id", "pd-unknown"),
                "title":   data.get("title", "PagerDuty incident"),
                "urgency": data.get("urgency", "high"),
                "service": data.get("service", {}).get("summary", ""),
                "url":     data.get("html_url", ""),
                "details": data.get("body", {}).get("details", ""),
            })

    # v2 format: {"messages": [{"event": "incident.trigger", "incident": {...}}]}
    for msg in body.get("messages", []):
        if msg.get("event", msg.get("type", "")).startswith("incident.trigger"):
            inc = msg.get("incident", {})
            incidents_to_triage.append({
                "id":      inc.get("id", "pd-unknown"),
                "title":   inc.get("title", "PagerDuty incident"),
                "urgency": inc.get("urgency", "high"),
                "service": inc.get("service", {}).get("name", ""),
                "url":     inc.get("html_url", ""),
                "details": inc.get("body", {}).get("details", ""),
            })

    if not incidents_to_triage:
        return {"status": "ignored", "reason": "no trigger events found"}

    triggered = []
    for inc in incidents_to_triage:
        incident_id = f"pd-{inc['id']}"
        service     = inc["service"]
        urgency     = inc["urgency"]
        title       = inc["title"]

        severity_map = {"high": "high", "low": "medium", "critical": "critical"}
        severity = severity_map.get(urgency, "high")

        description = f"[PagerDuty] {title}"
        if service:
            description += f" | service={service}"
        if inc.get("details"):
            description += f" | {inc['details'][:200]}"

        # 1. Create Slack war room immediately — on-call engineer sees it before AI finishes
        war_room_channel = None
        try:
            from app.integrations.slack import create_incident_channel, post_message
            channel_name = f"inc-pd-{inc['id'][:8].lower()}"
            ch = create_incident_channel(channel_name)
            war_room_channel = ch.get("channel") or ch.get("channel_id")
            if war_room_channel:
                post_message(
                    war_room_channel,
                    f":rotating_light: *PagerDuty Alert Auto-Triaged*\n"
                    f"*Incident:* {title}\n"
                    f"*Service:* {service or 'unknown'}\n"
                    f"*Urgency:* {urgency}\n"
                    f"*PagerDuty URL:* {inc.get('url', 'n/a')}\n\n"
                    f"_AI pipeline started — context enrichment in progress..._"
                )
        except Exception:
            pass

        # 2. Collect change timeline context synchronously (fast, 2h window)
        change_summary = ""
        try:
            from app.integrations.universal_collector import collect_all_context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                ctx_future = exe.submit(collect_all_context, 2)
                try:
                    ctx = ctx_future.result(timeout=8)
                    commits = ctx.get("github", {}).get("commits", {}).get("commits", [])
                    ct_events = ctx.get("aws", {}).get("cloudtrail", {}).get("events", [])
                    recent_changes = []
                    for c in commits[:5]:
                        recent_changes.append(f"  • commit: {c.get('message','')[:80]} ({c.get('author','')})")
                    for e in ct_events[:5]:
                        recent_changes.append(f"  • aws: {e.get('event_name','')} by {e.get('user','')}")
                    if recent_changes:
                        change_summary = "Recent changes (last 2h):\n" + "\n".join(recent_changes)
                except concurrent.futures.TimeoutError:
                    change_summary = "(context collection timed out)"
        except Exception:
            pass

        # 3. Post enriched brief to war room
        if war_room_channel and change_summary:
            try:
                from app.integrations.slack import post_message
                post_message(war_room_channel, f":mag: *Change Timeline*\n{change_summary}")
            except Exception:
                pass

        # 4. Fire AI pipeline (background — don't block webhook response)
        try:
            import threading
            from app.orchestrator.runner import run_pipeline
            threading.Thread(
                target=run_pipeline,
                kwargs={
                    "incident_id":    incident_id,
                    "description":    description,
                    "severity":       severity,
                    "auto_remediate": False,
                    "metadata": {
                        "source":          "pagerduty",
                        "pagerduty_id":    inc["id"],
                        "service":         service,
                        "war_room_channel": war_room_channel or "",
                        "pd_url":          inc.get("url", ""),
                    },
                },
                daemon=True,
            ).start()
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

        triggered.append({"incident_id": incident_id, "war_room": war_room_channel})

    return {"status": "triggered", "count": len(triggered), "incidents": triggered}


def _require_webhook_token(header_value: str, env_var: str, source: str) -> None:
    """Reject the request unless X-Webhook-Token matches the secret in `env_var`.

    Same contract as the GitHub/OpsGenie path: when the env var is set, every
    request must carry it; when unset, the endpoint is OPEN (matching the
    historical behaviour) — but the platform refuses to start in production
    when these env vars are missing (see app.core.config.validate_security).
    """
    import os as _os
    expected = _os.getenv(env_var, "").strip()
    if not expected:
        return  # dev-mode: no secret configured
    import hmac
    if not hmac.compare_digest(header_value or "", expected):
        from fastapi import HTTPException as _HTTPE
        raise _HTTPE(status_code=401, detail=f"{source} webhook: invalid X-Webhook-Token")


@router.post("/grafana", tags=["webhooks"])
async def webhook_grafana(
    request: Request,
    x_webhook_token: str = Header("", alias="X-Webhook-Token"),
):
    """Receive Grafana alert webhooks. Requires X-Webhook-Token matching
    GRAFANA_WEBHOOK_TOKEN when that env var is set."""
    _require_webhook_token(x_webhook_token, "GRAFANA_WEBHOOK_TOKEN", "grafana")
    try:
        body = await request.json()
    except Exception:
        body = {}
    alerts = body.get("alerts", [])
    triggered = []
    for alert in alerts:
        if alert.get("status") == "firing":
            from app.orchestrator.runner import run_pipeline
            result = run_pipeline(
                incident_id=f"grafana-{alert.get('fingerprint', 'unknown')}",
                description=f"Grafana alert: {alert.get('labels', {}).get('alertname', 'unknown')}",
                severity="high",
                auto_remediate=False,
            )
            triggered.append(result.get("status"))
    return {"status": "processed", "triggered": len(triggered)}


@router.post("/cloudwatch", tags=["webhooks"])
async def webhook_cloudwatch(
    request: Request,
    x_webhook_token: str = Header("", alias="X-Webhook-Token"),
):
    """Receive CloudWatch alarm SNS notifications. Requires X-Webhook-Token
    matching CLOUDWATCH_WEBHOOK_TOKEN when that env var is set."""
    _require_webhook_token(x_webhook_token, "CLOUDWATCH_WEBHOOK_TOKEN", "cloudwatch")
    try:
        body = await request.json()
    except Exception:
        body = {}
    alarm_name = body.get("AlarmName", "unknown")
    new_state = body.get("NewStateValue", "")
    if new_state == "ALARM":
        from app.orchestrator.runner import run_pipeline
        result = run_pipeline(
            incident_id=f"cw-{alarm_name[:40]}",
            description=f"CloudWatch alarm triggered: {alarm_name}",
            severity="high",
            auto_remediate=False,
        )
        return {"status": "triggered", "alarm": alarm_name, "pipeline": result.get("status")}
    return {"status": "ignored", "alarm": alarm_name, "state": new_state}


@router.post("/opsgenie", tags=["webhooks"])
async def webhook_opsgenie(
    request: Request,
    x_og_token: str = Header("", alias="X-OG-Token"),
):
    """Receive OpsGenie v2 alert webhooks and auto-triage.

    Verifies X-OG-Token bearer token when OPSGENIE_WEBHOOK_TOKEN is set.
    On Create/Acknowledge: creates Slack war room, collects change timeline,
    posts enriched brief, fires AI pipeline in background thread.
    """
    og_token = os.getenv("OPSGENIE_WEBHOOK_TOKEN", "").strip()
    if og_token and x_og_token and not __import__("hmac").compare_digest(og_token, x_og_token):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Invalid OpsGenie token"})

    try:
        body = await request.json()
    except Exception:
        body = {}

    action  = body.get("action", "")
    alert   = body.get("alert", {})
    alert_id = alert.get("alertId", "unknown")
    message  = alert.get("message", "OpsGenie alert")
    source   = alert.get("source", "")
    tags     = alert.get("tags", [])
    priority = alert.get("priority", "P3")

    # Only triage on alert creation or first acknowledgement
    if action not in ("Create", "Acknowledge"):
        return {"status": "ignored", "action": action}

    incident_id = f"opsgenie-{alert_id}"
    description = f"[OpsGenie] {message}"
    if source:
        description += f" | source={source}"
    if tags:
        description += f" | tags={','.join(tags[:5])}"

    priority_severity = {"P1": "critical", "P2": "high", "P3": "high", "P4": "medium", "P5": "low"}
    severity = priority_severity.get(priority, "high")

    # Create Slack war room
    war_room_channel = None
    try:
        from app.integrations.slack import create_incident_channel, post_message
        channel_name = f"inc-og-{alert_id[:8].lower()}"
        ch = create_incident_channel(channel_name)
        war_room_channel = ch.get("channel") or ch.get("channel_id")
        if war_room_channel:
            post_message(
                war_room_channel,
                f":rotating_light: *OpsGenie Alert Auto-Triaged*\n"
                f"*Alert:* {message}\n"
                f"*Priority:* {priority}\n"
                f"*Source:* {source or 'unknown'}\n"
                f"*Tags:* {', '.join(tags) if tags else 'none'}\n\n"
                f"_AI pipeline started — context enrichment in progress..._"
            )
    except Exception:
        pass

    # Collect change timeline
    change_summary = ""
    try:
        from app.integrations.universal_collector import collect_all_context
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
            ctx_future = exe.submit(collect_all_context, 2)
            try:
                ctx = ctx_future.result(timeout=8)
                commits = ctx.get("github", {}).get("commits", {}).get("commits", [])
                ct_events = ctx.get("aws", {}).get("cloudtrail", {}).get("events", [])
                recent_changes = []
                for c in commits[:5]:
                    recent_changes.append(f"  • commit: {c.get('message','')[:80]} ({c.get('author','')})")
                for e in ct_events[:5]:
                    recent_changes.append(f"  • aws: {e.get('event_name','')} by {e.get('user','')}")
                if recent_changes:
                    change_summary = "Recent changes (last 2h):\n" + "\n".join(recent_changes)
            except concurrent.futures.TimeoutError:
                change_summary = "(context collection timed out)"
    except Exception:
        pass

    if war_room_channel and change_summary:
        try:
            from app.integrations.slack import post_message
            post_message(war_room_channel, f":mag: *Change Timeline*\n{change_summary}")
        except Exception:
            pass

    # Fire AI pipeline in background
    try:
        import threading
        from app.orchestrator.runner import run_pipeline
        threading.Thread(
            target=run_pipeline,
            kwargs={
                "incident_id":    incident_id,
                "description":    description,
                "severity":       severity,
                "auto_remediate": False,
                "metadata": {
                    "source":           "opsgenie",
                    "opsgenie_id":      alert_id,
                    "priority":         priority,
                    "war_room_channel": war_room_channel or "",
                },
            },
            daemon=True,
        ).start()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    return {"status": "triggered", "incident_id": incident_id, "war_room": war_room_channel}
