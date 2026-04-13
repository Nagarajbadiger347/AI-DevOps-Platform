"""
Incident pipeline, memory, and post-mortem routes.
Paths: /incident/*, /incidents/*, /memory/*
"""
import re
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.routes.deps import (
    require_developer, require_viewer, optional_auth, AuthContext,
    _rbac_guard, _PENDING_PIPELINE_STATES, _RECENT_RESULTS, _cache_result,
    IncidentRunRequest, Event,
)
from app.agents.incident_pipeline import run_incident_pipeline
from app.orchestrator.runner import run_pipeline as run_pipeline_v2
from app.memory.vector_db import store_incident, search_similar_incidents
from app.integrations.jira import create_incident
from app.integrations.opsgenie import notify_on_call
from app.integrations.github import create_issue, create_pull_request

router = APIRouter(tags=["incidents"])

_SAFE_BRANCH = re.compile(r'^[\w\-./]+$')


class IncidentRunV2Request(BaseModel):
    incident_id:    str
    description:    str
    auto_remediate: bool = False
    user:           str  = "system"
    role:           str  = "viewer"
    aws_cfg:        Optional[Dict[str, Any]] = None
    k8s_cfg:        Optional[Dict[str, Any]] = None
    hours:          int  = 2
    slack_channel:  str  = "#incidents"
    llm_provider:   str  = ""
    metadata:       Optional[Dict[str, Any]] = None


class PostMortemRequest(BaseModel):
    incident_id:       str
    description:       Optional[str] = None
    root_cause:        Optional[str] = None
    severity:          Optional[str] = "SEV2"
    started_at:        Optional[str] = None
    resolved_at:       Optional[str] = None
    actions_taken:     Optional[list] = []
    validation:        Optional[dict] = {}
    errors:            Optional[list] = []
    save_to_disk:      bool = False


@router.post("/incident/jira")
def incident_jira(summary: str = "AI DevOps Incident", description: str = "Created via NsOps"):
    result = create_incident(summary=summary, description=description)
    if "error" in result:
        return {"jira_incident": result, "ok": False}
    return {"jira_incident": result, "ok": True}


@router.post("/incident/opsgenie")
def incident_opsgenie():
    result = notify_on_call()
    return {"opsgenie_notify": result}


@router.post("/incident/github/issue")
def incident_github_issue():
    result = create_issue()
    return {"github_issue": result}


@router.post("/incident/github/pr")
def incident_github_pr(head: str, base: str = "main"):
    if not _SAFE_BRANCH.match(head) or not _SAFE_BRANCH.match(base):
        raise HTTPException(status_code=400, detail="Invalid branch name")
    result = create_pull_request(head, base)
    return {"github_pr": result}


@router.get("/memory/incidents")
def memory_incidents_list(limit: int = 10):
    try:
        results = search_similar_incidents("", n_results=limit)
        if results and isinstance(results[0], list):
            results = results[0]
        return {"incidents": results or []}
    except Exception as exc:
        return {"incidents": [], "error": str(exc)}


@router.post("/memory/incidents")
def memory_incident(incident: Event):
    record = store_incident(incident.model_dump())
    return {"stored": record}


@router.post("/incident/run")
def incident_run(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None),
                 auth: Optional[AuthContext] = Depends(optional_auth)):
    """End-to-end autonomous incident response pipeline (LangGraph v2)."""
    resolved_user = (auth.username if auth else None) or x_user or req.user or "system"
    resolved_role = (auth.role if auth else None) or req.role or "admin"

    if req.auto_remediate:
        if auth and auth.role not in ("admin", "developer"):
            raise HTTPException(status_code=403, detail="deploy permission required for auto_remediate")
        elif not auth:
            _rbac_guard(x_user, "deploy")

    import os as _os
    if req.llm_provider:
        _os.environ["LLM_PROVIDER"] = req.llm_provider

    aws_cfg = req.aws_cfg or (req.aws.model_dump() if req.aws else {})
    k8s_cfg = req.k8s_cfg or (req.k8s.model_dump() if req.k8s else {})

    result = run_pipeline_v2(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        dry_run        = req.dry_run,
        metadata       = {
            "user":          resolved_user,
            "role":          resolved_role,
            "aws_cfg":       aws_cfg,
            "k8s_cfg":       k8s_cfg,
            "hours":         req.hours,
            "slack_channel": req.slack_channel,
        },
    )

    if result.get("requires_human_approval") and result.get("status") not in ("completed", "failed", "escalated"):
        result["status"] = "awaiting_approval"

    if result.get("status") == "awaiting_approval" or result.get("requires_human_approval"):
        cid = result.get("correlation_id")
        if cid:
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
                _PENDING_PIPELINE_STATES[approval.correlation_id] = result
                _PENDING_PIPELINE_STATES.pop(cid, None)
                result["correlation_id"] = approval.correlation_id

                try:
                    import datetime as _dt2
                    _plan2 = result.get("plan") or {}
                    store_incident({
                        "id":          result.get("incident_id", "unknown"),
                        "type":        "pipeline_v2",
                        "source":      "langgraph_orchestrator",
                        "created_at":  _dt2.datetime.now(_dt2.timezone.utc).isoformat(),
                        "description": result.get("description", ""),
                        "risk":        _plan2.get("risk", ""),
                        "status":      "awaiting_approval",
                        "confidence":  float(_plan2.get("confidence", 0.5)),
                        "payload": {
                            "description":       result.get("description", ""),
                            "root_cause":        _plan2.get("root_cause", ""),
                            "summary":           _plan2.get("summary", ""),
                            "risk":              _plan2.get("risk", ""),
                            "confidence":        float(_plan2.get("confidence", 0.5)),
                            "actions_executed":  0,
                            "validation_passed": False,
                            "status":            "awaiting_approval",
                            "created_at":        _dt2.datetime.now(_dt2.timezone.utc).isoformat(),
                            "correlation_id":    approval.correlation_id,
                        },
                    })
                except Exception:
                    pass

                slack_ch = (result.get("metadata") or {}).get("slack_channel", "")
                if slack_ch:
                    try:
                        post_approval_to_slack(approval, slack_ch)
                    except Exception:
                        pass

                try:
                    from app.integrations.email import send_approval_required
                    _plan = result.get("plan") or {}
                    send_approval_required(
                        incident_id     = result.get("incident_id", "unknown"),
                        description     = result.get("description", ""),
                        risk            = _plan.get("risk", "unknown"),
                        confidence      = float(_plan.get("confidence", 0)),
                        actions         = _plan.get("actions", []),
                        approval_reason = result.get("approval_reason", ""),
                    )
                except Exception:
                    pass
            except Exception as _approval_exc:
                import logging as _log
                _log.getLogger(__name__).warning("approval_creation_failed: %s", _approval_exc)
    else:
        try:
            from app.integrations.email import send_incident_completed
            _plan = result.get("plan") or {}
            send_incident_completed(
                incident_id       = result.get("incident_id", "unknown"),
                description       = result.get("description", ""),
                risk              = _plan.get("risk", "unknown"),
                status            = result.get("status", "completed"),
                root_cause        = _plan.get("root_cause", ""),
                summary           = _plan.get("summary", ""),
                actions_executed  = len(result.get("executed_actions", [])),
                validation_passed = bool(result.get("validation_passed", False)),
            )
        except Exception:
            pass

    _cache_result(result.get("incident_id", ""), result)

    if getattr(req, 'create_slack_channel', False):
        try:
            from app.orchestrator.runner import _slack_notify_pipeline
            _slack_notify_pipeline(req.incident_id, req.description, result,
                                   triggered_by=resolved_user)
        except Exception:
            pass

    return result


@router.post("/v2/incident/run")
def incident_run_v2(req: IncidentRunV2Request, auth: AuthContext = Depends(require_developer)):
    """Alias for /incidents/run — kept for backwards compatibility."""
    unified = IncidentRunRequest(
        incident_id    = req.incident_id,
        description    = req.description,
        auto_remediate = req.auto_remediate,
        hours          = req.hours,
        user           = req.user,
        role           = req.role,
        aws_cfg        = req.aws_cfg,
        k8s_cfg        = req.k8s_cfg,
        slack_channel  = req.slack_channel,
        llm_provider   = req.llm_provider,
        metadata       = req.metadata,
    )
    return incident_run(unified, x_user=None, auth=auth)


@router.get("/incidents/{incident_id}/result")
def get_incident_result(incident_id: str, auth: AuthContext = Depends(require_viewer)):
    result = _RECENT_RESULTS.get(incident_id)
    if result:
        return result

    try:
        results = search_similar_incidents("", n_results=200)
        if results and isinstance(results[0], list):
            results = results[0]
        incident = next((r for r in results if r.get("id") == incident_id), None)
        if incident:
            import json as _json
            payload = incident.get("payload", {})
            if isinstance(payload, str):
                try:
                    payload = _json.loads(payload)
                except Exception:
                    payload = {}
            return {
                "incident_id": incident_id,
                "status":      payload.get("status") or incident.get("status", "unknown"),
                "from_memory": True,
                "plan": {
                    "root_cause":  payload.get("root_cause", ""),
                    "summary":     payload.get("summary", ""),
                    "risk":        payload.get("risk", "unknown"),
                    "confidence":  payload.get("confidence", 0),
                    "actions":     [],
                },
                "description":      payload.get("description", incident.get("description", "")),
                "executed_actions": [],
                "blocked_actions":  [],
                "errors":           [],
                "aws_context":      {"_data_available": False},
                "k8s_context":      {"_data_available": False},
                "github_context":   {"_data_available": False},
                "created_at":       payload.get("created_at", ""),
            }
    except Exception:
        pass

    raise HTTPException(status_code=404, detail="Incident not found")


@router.delete("/incidents/{incident_id}", tags=["incidents"])
def delete_incident(incident_id: str, auth: AuthContext = Depends(require_developer)):
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


@router.post("/incidents/run")
def incidents_run_alias(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None),
                        auth: Optional[AuthContext] = Depends(optional_auth)):
    """Primary incident pipeline endpoint (same as /incident/run)."""
    return incident_run(req, x_user, auth)


@router.post("/incidents/run/async")
async def incidents_run_async(req: IncidentRunRequest, x_user: Optional[str] = Header(default=None)):
    """Fire-and-forget async pipeline — returns job ID immediately."""
    import asyncio, uuid
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, lambda: run_incident_pipeline(
        incident_id=req.incident_id, description=req.description,
        severity=req.severity, aws_config={}, k8s_config={}, auto_remediate=req.auto_remediate,
    ))
    return {"job_id": job_id, "status": "accepted", "incident_id": req.incident_id}


@router.post("/incidents/{incident_id}/post-mortem")
def generate_post_mortem_endpoint(
    incident_id: str,
    req: PostMortemRequest,
    auth: AuthContext = Depends(require_viewer),
):
    """Generate an AI-written blameless post-mortem for a resolved incident."""
    try:
        from app.incident.post_mortem import generate_post_mortem, format_as_markdown, save_post_mortem

        state: dict = {
            "incident_id":   incident_id,
            "description":   req.description or "",
            "root_cause":    req.root_cause or "",
            "severity":      req.severity,
            "started_at":    req.started_at or "",
            "resolved_at":   req.resolved_at or "",
            "actions_taken": req.actions_taken or [],
            "validation":    req.validation or {},
            "errors":        req.errors or [],
        }

        if not state["description"] or not state["root_cause"]:
            try:
                hits = search_similar_incidents(incident_id, n_results=5)
                if hits and isinstance(hits[0], list):
                    hits = hits[0]
                for hit in hits:
                    meta = hit.get("metadata", hit) if isinstance(hit, dict) else {}
                    if str(meta.get("incident_id", "")) == str(incident_id):
                        state.setdefault("description",   meta.get("description", state["description"]) or state["description"])
                        state.setdefault("root_cause",    meta.get("root_cause", "") or meta.get("rca", ""))
                        state.setdefault("severity",      meta.get("severity", state["severity"]))
                        state.setdefault("actions_taken", meta.get("actions_taken", []) or meta.get("executed_actions", []))
                        state.setdefault("started_at",    meta.get("started_at", "") or meta.get("created_at", ""))
                        state.setdefault("resolved_at",   meta.get("resolved_at", "") or meta.get("completed_at", ""))
                        break
            except Exception:
                pass

        pm       = generate_post_mortem(state)
        markdown = format_as_markdown(pm)
        result   = {
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
            saved_path = save_post_mortem(pm)
            result["saved_to"] = saved_path
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-mortem generation failed: {e}")


@router.post("/incidents/{incident_id}/postmortem-slack")
def send_postmortem_to_slack(incident_id: str, req: dict, auth: AuthContext = Depends(require_viewer)):
    try:
        from app.integrations.slack import post_message
        from app.core.config import settings
        message = req.get("message", "")
        channel = settings.SLACK_CHANNEL or "#incidents"
        result = post_message(channel=channel, text=message)
        if result:
            return {"success": True, "channel": channel}
        return {"success": False, "error": "Slack not configured or send failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
