"""
Deployment assessment, Jira webhook, and Jira-to-PR routes.
Paths: /deploy/*, /jira/*
"""
import re
from typing import Optional, List

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.api.deps import require_developer, AuthContext, _rbac_guard
from app.llm.claude import assess_deployment, interpret_jira_for_pr
from app.agents.infra.k8s_checker import check_k8s_cluster, check_k8s_pods, check_k8s_deployments
from app.integrations.aws_ops import list_cloudwatch_alarms
from app.memory.vector_db import search_similar_incidents
from app.integrations.jira import add_comment as jira_add_comment
from app.integrations.github import create_incident_pr

router = APIRouter(tags=["deploy"])

_AUTO_PR_ISSUE_TYPES = {"change request", "change-request", "task", "story"}
_AUTO_PR_LABELS      = {"auto-pr", "auto_pr", "create-pr"}


class DeployAssessRequest(BaseModel):
    deployment:  str
    namespace:   str = "default"
    new_image:   str = ""
    description: str = ""
    hours:       int = 2


class JiraWebhookPayload(BaseModel):
    webhookEvent: str = ""
    issue: dict = {}


@router.post("/deploy/assess")
def deploy_assess(req: DeployAssessRequest, x_user: Optional[str] = Header(default=None)):
    """Pre-deployment risk assessment — go / no-go decision before any deploy."""
    _rbac_guard(x_user, "deploy")

    from app.integrations.github import get_recent_commits as _get_recent_commits
    k8s_state = {
        "cluster":     check_k8s_cluster(),
        "pods":        check_k8s_pods(req.namespace),
        "deployments": check_k8s_deployments(req.namespace),
    }
    aws_alarms     = list_cloudwatch_alarms(state="ALARM")
    recent_commits = _get_recent_commits(hours=req.hours)
    past_incidents = search_similar_incidents(
        f"{req.deployment} {req.description}", n_results=3
    )

    assessment = assess_deployment({
        "deployment":       req.deployment,
        "namespace":        req.namespace,
        "new_image":        req.new_image,
        "description":      req.description,
        "k8s_state":        k8s_state,
        "recent_incidents": past_incidents,
        "aws_alarms":       aws_alarms,
        "recent_commits":   recent_commits,
    })

    return {
        "deployment": req.deployment,
        "namespace":  req.namespace,
        "new_image":  req.new_image,
        "assessment": assessment,
    }


@router.post("/aws/assess-deployment")
def deploy_assess_alias(req: DeployAssessRequest, x_user: Optional[str] = Header(default=None)):
    """Alias for /deploy/assess."""
    return deploy_assess(req, x_user=x_user)


@router.post("/jira/webhook")
def jira_webhook(payload: JiraWebhookPayload):
    """Jira webhook receiver — auto-creates a GitHub PR for change-request tickets."""
    issue_fields = payload.issue.get("fields", {})
    issue_key    = payload.issue.get("key", "")

    if not issue_key:
        return {"skipped": True, "reason": "No issue key in payload"}

    issue_type = (issue_fields.get("issuetype", {}).get("name", "") or "").lower()
    labels     = [str(l).lower() for l in (issue_fields.get("labels") or [])]

    if issue_type not in _AUTO_PR_ISSUE_TYPES and not any(l in _AUTO_PR_LABELS for l in labels):
        return {
            "skipped":   True,
            "issue_key": issue_key,
            "reason":    f"Issue type '{issue_type}' with labels {labels} does not trigger auto-PR",
        }

    jira_data = {
        "key":         issue_key,
        "summary":     issue_fields.get("summary", ""),
        "description": issue_fields.get("description", "") or "",
        "issue_type":  issue_fields.get("issuetype", {}).get("name", ""),
        "reporter":    (issue_fields.get("reporter") or {}).get("displayName", ""),
        "labels":      labels,
    }

    pr_plan = interpret_jira_for_pr(jira_data)
    if pr_plan.get("error"):
        return {"error": pr_plan["error"], "issue_key": issue_key}

    pr_result = create_incident_pr(
        incident_id  = issue_key,
        title        = pr_plan.get("pr_title", jira_data["summary"]),
        body         = pr_plan.get("pr_body", ""),
        file_changes = pr_plan.get("file_patches") or None,
    )

    comment_result = {"skipped": True}
    if pr_result.get("success"):
        pr_url = pr_result.get("url", "")
        comment_body = (
            f"🤖 *AI DevOps Platform* automatically created a GitHub PR for this ticket.\n\n"
            f"*PR:* [{pr_plan.get('pr_title')}|{pr_url}]\n"
            f"*Branch:* `{pr_result.get('branch')}`\n\n"
            f"_Confidence: {pr_plan.get('confidence', 0):.0%}_\n"
            + (f"\n*Target files:* {', '.join(pr_plan.get('target_files', []))}"
               if pr_plan.get("target_files") else "")
        )
        comment_result = jira_add_comment(issue_key, comment_body)

    return {
        "issue_key":      issue_key,
        "pr_plan":        pr_plan,
        "pr_created":     pr_result,
        "jira_commented": comment_result,
    }


@router.post("/jira/incident")
def jira_incident(summary: str = "AI DevOps Incident", description: str = "Created via NsOps"):
    from app.integrations.jira import create_incident
    result = create_incident(summary=summary, description=description)
    if "error" in result:
        return {"jira_incident": result, "ok": False}
    return {"jira_incident": result, "ok": True}


@router.post("/deploy/jira-to-pr")
def deploy_jira_to_pr(payload: JiraWebhookPayload):
    """Alias: same as /jira/webhook."""
    return jira_webhook(payload)
