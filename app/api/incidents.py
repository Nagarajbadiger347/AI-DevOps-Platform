"""
Incident pipeline, memory, and post-mortem routes.
Paths: /incident/*, /incidents/*, /memory/*
"""
import re
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import (
    require_developer, require_viewer, optional_auth, AuthContext,
    _PENDING_PIPELINE_STATES, _RECENT_RESULTS, _cache_result,
    IncidentRunRequest, Event,
)
from app.agents.incident_pipeline import run_incident_pipeline
from app.orchestrator.runner import run_pipeline as run_pipeline_v2
from app.memory.vector_db import store_incident, search_similar_incidents
from app.memory.long_term import get_trends, get_trend_report
from app.integrations.jira import create_incident
from app.integrations.opsgenie import notify_on_call
from app.integrations.github import create_issue, create_pull_request

logger = logging.getLogger(__name__)
router = APIRouter(tags=["incidents"])

_SAFE_BRANCH = re.compile(r'^[\w\-./]+$')


# ── Request / Response schemas ────────────────────────────────────────────────

class IncidentRunV2Request(BaseModel):
    incident_id:    str
    description:    str
    auto_remediate: bool = False
    aws_cfg:        Optional[Dict[str, Any]] = None
    k8s_cfg:        Optional[Dict[str, Any]] = None
    hours:          int  = 2
    slack_channel:  str  = "#incidents"
    llm_provider:   str  = ""
    metadata:       Optional[Dict[str, Any]] = None
    # NOTE: user/role removed — always resolved from JWT auth, never from request body


class PostMortemRequest(BaseModel):
    incident_id:       str
    description:       Optional[str] = None
    root_cause:        Optional[str] = None
    severity:          str = "SEV2"
    started_at:        Optional[str] = None
    resolved_at:       Optional[str] = None
    actions_taken:     List[Any] = Field(default_factory=list)
    validation:        Dict[str, Any] = Field(default_factory=dict)
    errors:            List[Any] = Field(default_factory=list)
    save_to_disk:      bool = False


class PostMortemSlackRequest(BaseModel):
    message: str
    channel: Optional[str] = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_incident_from_memory(incident_id: str) -> Optional[dict]:
    """
    Fetch a single incident from the vector DB by exact incident_id match.
    Searches with the incident_id as query and filters by metadata field — avoids
    scanning 200 unrelated records.
    """
    import json
    try:
        results = search_similar_incidents(incident_id, n_results=20)
        if results and isinstance(results[0], list):
            results = results[0]
        for hit in results:
            meta = hit.get("metadata", hit) if isinstance(hit, dict) else {}
            if str(meta.get("incident_id", "") or hit.get("id", "")) == str(incident_id):
                payload = hit.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = {}
                return {"hit": hit, "meta": meta, "payload": payload}
    except Exception as exc:
        logger.warning("memory_lookup_failed incident_id=%s error=%s", incident_id, exc)
    return None


def _handle_approval_flow(result: dict, slack_channel: str) -> dict:
    """
    Create an approval request, persist state, post to Slack, and send email.
    Returns the (possibly mutated) result dict with updated correlation_id.
    All sub-steps are best-effort; failures are logged, not swallowed silently.
    """
    import datetime

    cid = result.get("correlation_id")
    if not cid:
        return result

    _PENDING_PIPELINE_STATES[cid] = result

    try:
        from app.incident.approval import create_approval_request, post_approval_to_slack
        plan = result.get("plan") or {}
        approval = create_approval_request(
            incident_id  = result.get("incident_id", "unknown"),
            actions      = plan.get("actions", []),
            plan         = plan.get("summary") or result.get("approval_reason") or "AI-generated remediation plan",
            risk_score   = float(result.get("risk_score") or plan.get("risk_score") or 0.5),
            cost_report  = result.get("cost_report"),
            requested_by = (result.get("metadata") or {}).get("user", "pipeline"),
        )

        # Re-key pending state under the canonical approval correlation_id
        _PENDING_PIPELINE_STATES[approval.correlation_id] = result
        _PENDING_PIPELINE_STATES.pop(cid, None)
        result["correlation_id"] = approval.correlation_id

        # Persist to vector DB so the incident survives a restart
        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            store_incident({
                "id":          result.get("incident_id", "unknown"),
                "type":        "pipeline_v2",
                "source":      "langgraph_orchestrator",
                "created_at":  now,
                "description": result.get("description", ""),
                "risk":        plan.get("risk", ""),
                "status":      "awaiting_approval",
                "confidence":  float(plan.get("confidence", 0.5)),
                "payload": {
                    "description":       result.get("description", ""),
                    "root_cause":        plan.get("root_cause", ""),
                    "summary":           plan.get("summary", ""),
                    "risk":              plan.get("risk", ""),
                    "confidence":        float(plan.get("confidence", 0.5)),
                    "actions_executed":  0,
                    "validation_passed": False,
                    "status":            "awaiting_approval",
                    "created_at":        now,
                    "correlation_id":    approval.correlation_id,
                },
            })
        except Exception as exc:
            logger.warning("approval_memory_store_failed incident=%s error=%s",
                           result.get("incident_id"), exc)

        if slack_channel:
            try:
                post_approval_to_slack(approval, slack_channel)
            except Exception as exc:
                logger.warning("approval_slack_post_failed channel=%s error=%s",
                               slack_channel, exc)

        try:
            from app.integrations.email import send_approval_required
            send_approval_required(
                incident_id     = result.get("incident_id", "unknown"),
                description     = result.get("description", ""),
                risk            = plan.get("risk", "unknown"),
                confidence      = float(plan.get("confidence", 0)),
                actions         = plan.get("actions", []),
                approval_reason = result.get("approval_reason", ""),
            )
        except Exception as exc:
            logger.warning("approval_email_failed incident=%s error=%s",
                           result.get("incident_id"), exc)

    except Exception as exc:
        logger.error("approval_flow_failed incident=%s error=%s",
                     result.get("incident_id"), exc)

    return result


def _send_completion_notification(result: dict) -> None:
    """Send email on pipeline completion. Best-effort."""
    try:
        from app.integrations.email import send_incident_completed
        plan = result.get("plan") or {}
        send_incident_completed(
            incident_id       = result.get("incident_id", "unknown"),
            description       = result.get("description", ""),
            risk              = plan.get("risk", "unknown"),
            status            = result.get("status", "completed"),
            root_cause        = plan.get("root_cause", ""),
            summary           = plan.get("summary", ""),
            actions_executed  = len(result.get("executed_actions", [])),
            validation_passed = bool(result.get("validation_passed", False)),
        )
    except Exception as exc:
        logger.warning("completion_email_failed incident=%s error=%s",
                       result.get("incident_id"), exc)


# ── Integration shim endpoints ────────────────────────────────────────────────

@router.post("/incident/jira")
def incident_jira(
    summary: str = "AI DevOps Incident",
    description: str = "Created via NsOps",
    _: AuthContext = Depends(require_developer),
):
    result = create_incident(summary=summary, description=description)
    if "error" in result:
        return {"jira_incident": result, "ok": False}
    return {"jira_incident": result, "ok": True}


@router.post("/incident/opsgenie")
def incident_opsgenie(_: AuthContext = Depends(require_developer)):
    result = notify_on_call()
    return {"opsgenie_notify": result}


@router.post("/incident/github/issue")
def incident_github_issue(_: AuthContext = Depends(require_developer)):
    result = create_issue()
    return {"github_issue": result}


@router.post("/incident/github/pr")
def incident_github_pr(
    head: str,
    base: str = "main",
    _: AuthContext = Depends(require_developer),
):
    if not _SAFE_BRANCH.match(head) or not _SAFE_BRANCH.match(base):
        raise HTTPException(status_code=400, detail="Invalid branch name")
    result = create_pull_request(head, base)
    return {"github_pr": result}


# ── Memory endpoints ──────────────────────────────────────────────────────────

@router.get("/memory/incidents")
def memory_incidents_list(
    limit: int = 10,
    _: AuthContext = Depends(require_viewer),
):
    try:
        results = search_similar_incidents("", n_results=limit)
        if results and isinstance(results[0], list):
            results = results[0]
        return {"incidents": results or []}
    except Exception as exc:
        logger.error("memory_list_failed error=%s", exc)
        return {"incidents": [], "error": str(exc)}


@router.get("/memory/incidents/trends")
def memory_incidents_trends(
    _: AuthContext = Depends(require_viewer),
):
    """Return trend analysis across all stored incidents."""
    try:
        trends = get_trends()
        report = get_trend_report()
        return {"trends": trends, "report": report}
    except Exception as exc:
        logger.error("memory_trends_failed error=%s", exc)
        return {"trends": {}, "report": "", "error": str(exc)}


@router.get("/memory/incidents/search")
def memory_incidents_search(
    q: str = "",
    n: int = 10,
    _: AuthContext = Depends(require_viewer),
):
    """Semantic search over stored incidents with similarity scores."""
    try:
        results = search_similar_incidents(q or "incident", n_results=min(n, 50))
        if results and isinstance(results[0], list):
            results = results[0]
        return {"incidents": results or [], "query": q, "count": len(results or [])}
    except Exception as exc:
        logger.error("memory_search_failed error=%s", exc)
        return {"incidents": [], "error": str(exc)}


@router.post("/memory/incidents")
def memory_incident(
    incident: Event,
    _: AuthContext = Depends(require_developer),
):
    record = store_incident(incident.model_dump())
    return {"stored": record}


# ── Primary pipeline endpoint ─────────────────────────────────────────────────

@router.post("/incident/run")
def incident_run(
    req: IncidentRunRequest,
    x_user: Optional[str] = Header(default=None),
    auth: Optional[AuthContext] = Depends(optional_auth),
):
    """End-to-end autonomous incident response pipeline (LangGraph v2)."""
    # ── Resolve identity from JWT first, headers second, never from body ──
    resolved_user = (auth.username if auth else None) or x_user or "system"
    resolved_role = (auth.role if auth else None) or "viewer"

    if req.auto_remediate:
        if resolved_role not in ("admin", "developer", "super_admin"):
            raise HTTPException(
                status_code=403,
                detail="'deploy' permission required for auto_remediate",
            )

    # ── LLM provider override — thread-local, not global env mutation ──
    # Pass provider through metadata so the pipeline can set it per-run.
    metadata = dict(req.metadata or {})
    metadata.update({
        "user":          resolved_user,
        "role":          resolved_role,
        "aws_cfg":       req.aws_cfg or (req.aws.model_dump() if req.aws else {}),
        "k8s_cfg":       req.k8s_cfg or (req.k8s.model_dump() if req.k8s else {}),
        "hours":         req.hours,
        "slack_channel": req.slack_channel,
    })
    if req.llm_provider:
        metadata["llm_provider"] = req.llm_provider

    result = run_pipeline_v2(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        dry_run        = req.dry_run,
        metadata       = metadata,
    )

    is_awaiting = (
        result.get("requires_human_approval")
        and result.get("status") not in ("completed", "failed", "escalated")
    )

    if is_awaiting:
        result["status"] = "awaiting_approval"
        slack_channel = (result.get("metadata") or {}).get("slack_channel", "")
        result = _handle_approval_flow(result, slack_channel)
    else:
        _send_completion_notification(result)

    _cache_result(result.get("incident_id", ""), result)

    if getattr(req, "create_slack_channel", False):
        try:
            from app.orchestrator.runner import _slack_notify_pipeline
            _slack_notify_pipeline(req.incident_id, req.description, result,
                                   triggered_by=resolved_user)
        except Exception as exc:
            logger.warning("slack_channel_create_failed error=%s", exc)

    return result


@router.post("/v2/incident/run")
def incident_run_v2(
    req: IncidentRunV2Request,
    auth: AuthContext = Depends(require_developer),
):
    """Stable v2 alias — role always comes from JWT, never the request body."""
    unified = IncidentRunRequest(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        hours          = req.hours,
        user           = auth.username,
        role           = auth.role,
        aws_cfg        = req.aws_cfg,
        k8s_cfg        = req.k8s_cfg,
        slack_channel  = req.slack_channel,
        llm_provider   = req.llm_provider,
        metadata       = req.metadata,
    )
    return incident_run(unified, x_user=None, auth=auth)


# ── Unified workflow endpoint (primary for UI) ────────────────────────────────

@router.post("/incidents/run")
def incidents_run(
    req: IncidentRunRequest,
    auth: Optional[AuthContext] = Depends(optional_auth),
):
    """Unified LangGraph incident pipeline — handles K8s, AWS, GitHub, and all notification actions."""
    import time

    aws_cfg = req.aws_cfg or (req.aws.model_dump() if req.aws else {})
    k8s_cfg = req.k8s_cfg or (req.k8s.model_dump() if req.k8s else {})

    t0 = time.monotonic()
    try:
        from app.workflows.unified_workflow import run_unified
        result = run_unified(
            incident_id    = req.incident_id,
            description    = req.description,
            severity       = req.severity or "high",
            namespace      = k8s_cfg.get("namespace", "default"),
            pod_name       = k8s_cfg.get("pod_name", ""),
            aws_cfg        = aws_cfg,
            hours          = req.hours or 2,
            dry_run        = req.dry_run or False,
            auto_fix       = False,
            auto_remediate = req.auto_remediate or False,
        )
    except Exception as exc:
        logger.error("unified_workflow_failed incident=%s error=%s", req.incident_id, exc)
        return {"status": "failed", "error": str(exc), "incident_id": req.incident_id}

    elapsed = round(time.monotonic() - t0, 2)
    inc_id = result.get("incident_id") or req.incident_id or ""
    _cache_result(inc_id, result)

    try:
        store_incident({
            "incident_id": inc_id,
            "description": req.description,
            "severity":    result.get("severity_ai") or req.severity or "medium",
            "root_cause":  result.get("root_cause", ""),
            "status":      "completed" if result.get("success") else "failed",
            "confidence":  result.get("confidence", 0.0),
        })
    except Exception as exc:
        logger.warning("incident_memory_store_failed incident=%s error=%s", inc_id, exc)

    root_cause     = result.get("root_cause", "")
    fix_suggestion = result.get("fix_suggestion", "")
    confidence     = result.get("confidence", 0.0)
    findings       = result.get("findings", [])
    actions_taken  = result.get("actions_taken", [])
    severity_ai    = result.get("severity_ai") or req.severity

    plan_actions = []
    if fix_suggestion:
        plan_actions.append({"description": fix_suggestion, "type": "fix", "risk": "low"})
    for f in findings:
        plan_actions.append({"description": f, "type": "finding", "risk": "low"})

    return {
        "status":         "completed" if result.get("success") else "failed",
        "incident_id":    inc_id,
        "summary":        result.get("summary", root_cause[:100] if root_cause else ""),
        "root_cause":     root_cause,
        "failure_type":   result.get("failure_type", ""),
        "ai_severity":    severity_ai,
        "confidence":     confidence,
        "findings":       findings,
        "fix_suggestion": fix_suggestion,
        "actions_taken":  actions_taken,
        "report":         result.get("report", ""),
        "steps_taken":    result.get("steps_taken", []),
        "errors":         result.get("errors", []),
        "elapsed_s":      elapsed,
        "plan": {
            "root_cause":     root_cause,
            "summary":        result.get("summary", ""),
            "risk":           severity_ai or "medium",
            "confidence":     confidence,
            "actions":        plan_actions,
            "reasoning":      result.get("report", ""),
            "findings":       findings,
            "fix_suggestion": fix_suggestion,
        },
        "risk_level": severity_ai or "medium",
        "observability": {
            "k8s_collected":    result.get("k8s_data", {}).get("_data_available", False),
            "aws_collected":    result.get("aws_data", {}).get("_data_available", False),
            "github_collected": result.get("github_data", {}).get("_data_available", False),
        },
    }


@router.post("/incidents/run/async")
async def incidents_run_async(
    req: IncidentRunRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(require_viewer),
):
    """Fire-and-forget — returns job ID immediately, runs pipeline in background."""
    import uuid

    job_id = f"job-{uuid.uuid4().hex[:8]}"

    def _run():
        try:
            run_incident_pipeline(
                incident_id    = req.incident_id,
                description    = req.description,
                severity       = req.severity,
                aws_config     = {},
                k8s_config     = {},
                auto_remediate = req.auto_remediate,
            )
        except Exception as exc:
            logger.error("async_pipeline_failed job=%s incident=%s error=%s",
                         job_id, req.incident_id, exc)

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "accepted", "incident_id": req.incident_id}


# ── Incident result & management ──────────────────────────────────────────────

@router.get("/incidents/{incident_id}/result")
def get_incident_result(
    incident_id: str,
    auth: AuthContext = Depends(require_viewer),
):
    # Fast path: in-memory cache hit
    result = _RECENT_RESULTS.get(incident_id)
    if result:
        return result

    # Slow path: search vector DB (bounded scan, not 200 records)
    found = _get_incident_from_memory(incident_id)
    if found:
        payload = found["payload"]
        return {
            "incident_id":      incident_id,
            "status":           payload.get("status") or found["hit"].get("status", "unknown"),
            "from_memory":      True,
            "plan": {
                "root_cause":  payload.get("root_cause", ""),
                "summary":     payload.get("summary", ""),
                "risk":        payload.get("risk", "unknown"),
                "confidence":  payload.get("confidence", 0),
                "actions":     [],
            },
            "description":      payload.get("description", found["hit"].get("description", "")),
            "executed_actions": [],
            "blocked_actions":  [],
            "errors":           [],
            "aws_context":      {"_data_available": False},
            "k8s_context":      {"_data_available": False},
            "github_context":   {"_data_available": False},
            "created_at":       payload.get("created_at", ""),
        }

    raise HTTPException(status_code=404, detail="Incident not found")


@router.delete("/incidents/{incident_id}")
def delete_incident(
    incident_id: str,
    auth: AuthContext = Depends(require_developer),
):
    removed_cache = incident_id in _RECENT_RESULTS
    _RECENT_RESULTS.pop(incident_id, None)
    from app.memory.vector_db import delete_incident as _db_delete
    db_result = _db_delete(incident_id)
    return {
        "deleted":            True,
        "incident_id":        incident_id,
        "removed_from_cache": removed_cache,
        "removed_from_db":    db_result.get("deleted", False),
    }


# ── Post-mortem endpoints ─────────────────────────────────────────────────────

@router.post("/incidents/{incident_id}/post-mortem")
def generate_post_mortem_endpoint(
    incident_id: str,
    req: PostMortemRequest,
    auth: AuthContext = Depends(require_viewer),
):
    """Generate an AI-written blameless post-mortem for a resolved incident."""
    try:
        from app.incident.post_mortem import generate_post_mortem, format_as_markdown, save_post_mortem
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Post-mortem module unavailable: {exc}")

    state: dict = {
        "incident_id":   incident_id,
        "description":   req.description or "",
        "root_cause":    req.root_cause or "",
        "severity":      req.severity,
        "started_at":    req.started_at or "",
        "resolved_at":   req.resolved_at or "",
        "actions_taken": req.actions_taken,
        "validation":    req.validation,
        "errors":        req.errors,
    }

    # Enrich from memory if description / root_cause missing
    if not state["description"] or not state["root_cause"]:
        found = _get_incident_from_memory(incident_id)
        if found:
            meta = found["meta"]
            state["description"]   = state["description"]   or meta.get("description", "")
            state["root_cause"]    = state["root_cause"]    or meta.get("root_cause", "") or meta.get("rca", "")
            state["severity"]      = state["severity"]      or meta.get("severity", req.severity)
            state["actions_taken"] = state["actions_taken"] or meta.get("actions_taken", []) or meta.get("executed_actions", [])
            state["started_at"]    = state["started_at"]    or meta.get("started_at", "") or meta.get("created_at", "")
            state["resolved_at"]   = state["resolved_at"]   or meta.get("resolved_at", "") or meta.get("completed_at", "")

    try:
        pm       = generate_post_mortem(state)
        markdown = format_as_markdown(pm)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Post-mortem generation failed: {exc}")

    result = {
        "incident_id":          pm.incident_id,
        "title":                pm.title,
        "severity":             pm.severity,
        "duration_minutes":     pm.duration_minutes,
        "root_cause":           pm.root_cause,
        "contributing_factors": pm.contributing_factors,
        "impact":               pm.impact,
        "resolution":           pm.resolution,
        "action_items":         [vars(a) for a in pm.action_items],
        "lessons_learned":      pm.lessons_learned,
        "prevention_steps":     pm.prevention_steps,
        "generated_at":         pm.generated_at,
        "markdown":             markdown,
    }

    if req.save_to_disk:
        try:
            result["saved_to"] = save_post_mortem(pm)
        except Exception as exc:
            logger.warning("post_mortem_save_failed incident=%s error=%s", incident_id, exc)

    return result


@router.post("/incidents/{incident_id}/postmortem-slack")
def send_postmortem_to_slack(
    incident_id: str,  # kept as path param for route matching
    req: PostMortemSlackRequest,
    _: AuthContext = Depends(require_viewer),
):
    try:
        from app.integrations.slack import post_message
        from app.core.config import settings

        channel = req.channel or settings.SLACK_CHANNEL or "#incidents"
        result = post_message(channel=channel, text=req.message)
        if result:
            return {"success": True, "channel": channel}
        return {"success": False, "error": "Slack not configured or message send failed"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
