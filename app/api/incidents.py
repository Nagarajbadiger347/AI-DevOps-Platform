"""
Incident pipeline, memory, and post-mortem routes.

Canonical endpoints:
  POST /incidents/run              — primary pipeline (LangGraph + unified workflow)
  POST /incidents/run/async        — fire-and-forget variant
  GET  /incidents/{id}/result      — poll result
  DELETE /incidents/{id}           — remove from cache + memory
  POST /incidents/{id}/post-mortem — generate post-mortem
  POST /incidents/{id}/postmortem-slack — post to Slack

  GET  /memory/incidents           — list stored incidents
  GET  /memory/incidents/search    — semantic search with similarity scores
  GET  /memory/incidents/trends    — trend analysis
  POST /memory/incidents           — store an incident manually

"""
import re
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import (
    require_developer, require_viewer, optional_auth, AuthContext,
    _PENDING_PIPELINE_STATES, _RECENT_RESULTS, _cache_result,
    IncidentRunRequest, Event,
)
from app.orchestrator.runner import run_pipeline
from app.memory.vector_db import store_incident, search_similar_incidents
from app.memory.long_term import get_trends, get_trend_report

logger = logging.getLogger(__name__)
router = APIRouter(tags=["incidents"])

_SAFE_BRANCH = re.compile(r'^[\w\-./]+$')


# ── Schemas ───────────────────────────────────────────────────────────────────

class PostMortemRequest(BaseModel):
    incident_id:   str
    description:   Optional[str] = None
    root_cause:    Optional[str] = None
    severity:      str = "SEV2"
    started_at:    Optional[str] = None
    resolved_at:   Optional[str] = None
    actions_taken: List[Any] = Field(default_factory=list)
    validation:    Dict[str, Any] = Field(default_factory=dict)
    errors:        List[Any] = Field(default_factory=list)
    save_to_disk:  bool = False


class PostMortemSlackRequest(BaseModel):
    message: str
    channel: Optional[str] = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_incident_from_memory(incident_id: str) -> Optional[dict]:
    """Fetch a single incident from ChromaDB by exact incident_id match."""
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
    """Create approval request, persist state, post to Slack, send email. Best-effort."""
    import datetime
    cid = result.get("correlation_id")
    if not cid:
        return result

    _PENDING_PIPELINE_STATES[cid] = result

    try:
        from app.incident.approval import create_approval_request, post_approval_to_slack
        plan     = result.get("plan") or {}
        approval = create_approval_request(
            incident_id  = result.get("incident_id", "unknown"),
            actions      = plan.get("actions", []),
            plan         = plan.get("summary") or result.get("approval_reason") or "AI-generated plan",
            risk_score   = float(result.get("risk_score") or plan.get("risk_score") or 0.5),
            cost_report  = result.get("cost_report"),
            requested_by = (result.get("metadata") or {}).get("user", "pipeline"),
        )
        _PENDING_PIPELINE_STATES[approval.correlation_id] = result
        _PENDING_PIPELINE_STATES.pop(cid, None)
        result["correlation_id"] = approval.correlation_id

        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            store_incident({
                "id": result.get("incident_id", "unknown"), "type": "pipeline_v2",
                "source": "langgraph_orchestrator", "created_at": now,
                "description": result.get("description", ""), "status": "awaiting_approval",
                "confidence": float(plan.get("confidence", 0.5)),
                "payload": {
                    "description": result.get("description", ""),
                    "root_cause": plan.get("root_cause", ""),
                    "summary": plan.get("summary", ""),
                    "risk": plan.get("risk", ""),
                    "confidence": float(plan.get("confidence", 0.5)),
                    "status": "awaiting_approval",
                    "created_at": now,
                    "correlation_id": approval.correlation_id,
                },
            })
        except Exception as exc:
            logger.warning("approval_memory_store_failed incident=%s error=%s",
                           result.get("incident_id"), exc)

        if slack_channel:
            try:
                post_approval_to_slack(approval, slack_channel)
            except Exception as exc:
                logger.warning("approval_slack_post_failed error=%s", exc)

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
            logger.warning("approval_email_failed error=%s", exc)

    except Exception as exc:
        logger.error("approval_flow_failed incident=%s error=%s",
                     result.get("incident_id"), exc)
    return result


def _send_completion_notification(result: dict) -> None:
    """Send completion email. Best-effort."""
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


def _run_pipeline_from_request(req: IncidentRunRequest, auth: Optional[AuthContext]) -> dict:
    """Shared pipeline execution logic used by all run endpoints."""
    resolved_user = (auth.username if auth else None) or "system"
    resolved_role = (auth.role    if auth else None) or "viewer"

    if req.auto_remediate and resolved_role not in ("admin", "developer", "super_admin"):
        raise HTTPException(status_code=403,
                            detail="'deploy' permission required for auto_remediate")

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

    result = run_pipeline(
        incident_id    = req.incident_id,
        description    = req.description,
        user           = resolved_user,
        role           = resolved_role,
        severity       = req.severity or "medium",
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
        slack_channel = metadata.get("slack_channel", "")
        result = _handle_approval_flow(result, slack_channel)
    else:
        _send_completion_notification(result)
        try:
            import datetime as _dt
            plan = result.get("plan") or {}
            now = _dt.datetime.now(_dt.timezone.utc).isoformat()
            store_incident({
                "id": result.get("incident_id", "unknown"), "type": "pipeline_v2",
                "source": "langgraph_orchestrator", "created_at": now,
                "description": result.get("description", ""), "status": result.get("status", "completed"),
                "confidence": float(plan.get("confidence", 0.5)),
                "payload": {
                    "description": result.get("description", ""),
                    "root_cause": plan.get("root_cause", ""),
                    "summary": plan.get("summary", ""),
                    "risk": plan.get("risk", ""),
                    "confidence": float(plan.get("confidence", 0.5)),
                    "status": result.get("status", "completed"),
                    "created_at": now,
                },
            })
        except Exception as exc:
            logger.warning("incident_memory_store_failed incident=%s error=%s",
                           result.get("incident_id"), exc)

    _cache_result(result.get("incident_id", ""), result)
    return result


# ── Canonical pipeline endpoint ───────────────────────────────────────────────

@router.post("/incidents/run")
def incidents_run(
    req: IncidentRunRequest,
    auth: Optional[AuthContext] = Depends(optional_auth),
):
    """Run the LangGraph incident pipeline. Single canonical endpoint for all pipeline runs."""
    return _run_pipeline_from_request(req, auth)


@router.post("/incidents/run/async")
async def incidents_run_async(
    req: IncidentRunRequest,
    background_tasks: BackgroundTasks,
    auth: Optional[AuthContext] = Depends(optional_auth),
):
    """Fire-and-forget — returns immediately, runs pipeline in background.
    Poll GET /incidents/{incident_id}/result for the outcome.
    """
    def _run():
        try:
            _run_pipeline_from_request(req, auth)
        except Exception as exc:
            logger.error("async_pipeline_failed incident=%s error=%s",
                         req.incident_id, exc)

    background_tasks.add_task(_run)
    return {"status": "accepted", "incident_id": req.incident_id,
            "poll": f"/incidents/{req.incident_id}/result"}


# ── Integration shim endpoints ────────────────────────────────────────────────

@router.post("/incident/jira")
def incident_jira(
    summary: str = "AI DevOps Incident",
    description: str = "Created via NexusOps",
    _: AuthContext = Depends(require_developer),
):
    from app.integrations.jira import create_incident
    result = create_incident(summary=summary, description=description)
    return {"jira_incident": result, "ok": "error" not in result}


@router.post("/incident/opsgenie")
def incident_opsgenie(_: AuthContext = Depends(require_developer)):
    from app.integrations.opsgenie import notify_on_call
    return {"opsgenie_notify": notify_on_call()}


@router.post("/incident/github/pr")
def incident_github_pr(
    head: str,
    base: str = "main",
    _: AuthContext = Depends(require_developer),
):
    if not _SAFE_BRANCH.match(head) or not _SAFE_BRANCH.match(base):
        raise HTTPException(status_code=400, detail="Invalid branch name")
    from app.integrations.github import create_pull_request
    return {"github_pr": create_pull_request(head, base)}


# ── Memory endpoints ──────────────────────────────────────────────────────────

@router.get("/memory/incidents")
def memory_incidents_list(limit: int = 10, auth: AuthContext = Depends(require_viewer)):
    try:
        results = search_similar_incidents("incident", n_results=limit, tenant_id=auth.tenant_id)
        if results and isinstance(results[0], list):
            results = results[0]
        return {"incidents": results or []}
    except Exception as exc:
        logger.error("memory_list_failed error=%s", exc)
        return {"incidents": [], "error": str(exc)}


@router.get("/memory/incidents/trends")
def memory_incidents_trends(auth: AuthContext = Depends(require_viewer)):
    """Trend analysis across all stored incidents — MTTR, recurring causes, frequency."""
    try:
        from app.memory.long_term import get_trends as _get_trends, get_trend_report as _get_trend_report
        return {"trends": _get_trends(auth.tenant_id), "report": _get_trend_report(auth.tenant_id)}
    except Exception as exc:
        logger.error("memory_trends_failed error=%s", exc)
        return {"trends": {}, "report": "", "error": str(exc)}


@router.get("/memory/incidents/search")
def memory_incidents_search(
    q: str = "",
    n: int = 10,
    auth: AuthContext = Depends(require_viewer),
):
    """Semantic search over stored incidents with similarity scores."""
    try:
        results = search_similar_incidents(q or "incident", n_results=min(n, 50), tenant_id=auth.tenant_id)
        if results and isinstance(results[0], list):
            results = results[0]
        return {"incidents": results or [], "query": q, "count": len(results or [])}
    except Exception as exc:
        logger.error("memory_search_failed error=%s", exc)
        return {"incidents": [], "error": str(exc)}


@router.post("/memory/incidents")
def memory_incident_store(incident: Event, auth: AuthContext = Depends(require_developer)):
    return {"stored": store_incident(incident.model_dump(), tenant_id=auth.tenant_id)}


# ── Incident result & management ──────────────────────────────────────────────

@router.get("/incidents/{incident_id}/result")
def get_incident_result(
    incident_id: str,
    _: AuthContext = Depends(require_viewer),
):
    result = _RECENT_RESULTS.get(incident_id)
    if result:
        return result

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
            "description":      payload.get("description", ""),
            "executed_actions": [],
            "blocked_actions":  [],
            "errors":           [],
            "created_at":       payload.get("created_at", ""),
        }

    raise HTTPException(status_code=404, detail="Incident not found")


@router.delete("/incidents/{incident_id}")
def delete_incident_endpoint(
    incident_id: str,
    _: AuthContext = Depends(require_developer),
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


# ── Post-mortem ───────────────────────────────────────────────────────────────

@router.post("/incidents/{incident_id}/post-mortem")
def generate_post_mortem_endpoint(
    incident_id: str,
    req: PostMortemRequest,
    _: AuthContext = Depends(require_viewer),
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

    # Enrich from memory if fields are missing
    if not state["description"] or not state["root_cause"]:
        found = _get_incident_from_memory(incident_id)
        if found:
            meta = found["meta"]
            state["description"]   = state["description"]   or meta.get("description", "")
            state["root_cause"]    = state["root_cause"]    or meta.get("root_cause", "") or meta.get("rca", "")
            state["severity"]      = state["severity"]      or meta.get("severity", req.severity)
            state["actions_taken"] = state["actions_taken"] or meta.get("actions_taken", [])
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


@router.get("/incidents/{incident_id}/stream")
async def stream_incident_events(
    incident_id: str,
    _: AuthContext = Depends(require_viewer),
):
    """SSE stream of real-time pipeline progress for an incident.

    Clients connect with `EventSource('/incidents/{id}/stream')`.
    Each event is a JSON-encoded pipeline stage update.
    The stream ends with a terminal `status=done` event.
    """
    from app.core.pipeline_events import bus

    async def _event_generator() -> AsyncIterator[str]:
        try:
            async for event in bus.subscribe(incident_id):
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") == "done":
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/incidents/{incident_id}/postmortem-slack")
def send_postmortem_to_slack(
    incident_id: str,
    req: PostMortemSlackRequest,
    _: AuthContext = Depends(require_viewer),
):
    try:
        from app.integrations.slack import post_message
        from app.core.config import settings
        channel = req.channel or settings.SLACK_CHANNEL or "#incidents"
        result  = post_message(channel=channel, text=req.message)
        return {"success": bool(result), "channel": channel}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
