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
async def webhook_pagerduty(payload: PagerDutyWebhookPayload):
    """Receive PagerDuty incident trigger webhooks."""
    for msg in payload.messages:
        inc = msg.get("incident", {})
        inc_id = inc.get("id", "pd-unknown")
        title = inc.get("title", "PagerDuty incident")
        urgency = inc.get("urgency", "high")
        from app.orchestrator.runner import run_pipeline
        result = run_pipeline(
            incident_id=f"pd-{inc_id}",
            description=title,
            severity=urgency,
            auto_remediate=False,
        )
        return {"status": "triggered", "incident_id": f"pd-{inc_id}", "pipeline": result.get("status")}
    return {"status": "no_messages"}


@router.post("/grafana", tags=["webhooks"])
async def webhook_grafana(request: Request):
    """Receive Grafana alert webhooks."""
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
async def webhook_cloudwatch(request: Request):
    """Receive CloudWatch alarm SNS notifications."""
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
async def webhook_opsgenie(request: Request):
    """Receive OpsGenie alert webhooks."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    alert = body.get("alert", {})
    alert_id = alert.get("alertId", "unknown")
    message = alert.get("message", "OpsGenie alert")
    action = body.get("action", "")
    if action in ("Create", "Acknowledge"):
        from app.orchestrator.runner import run_pipeline
        result = run_pipeline(
            incident_id=f"opsgenie-{alert_id}",
            description=f"OpsGenie: {message}",
            severity="high",
            auto_remediate=False,
        )
        return {"status": "triggered", "alert_id": alert_id}
    return {"status": "ignored", "action": action}
