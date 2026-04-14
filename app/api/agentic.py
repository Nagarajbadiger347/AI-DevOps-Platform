"""
Agentic AI routes — LangGraph multi-agent endpoints.

Paths:
  POST /debug-pod          → K8s incident debugging workflow
  POST /agent/plan         → Planner agent
  POST /agent/execute      → Executor agent (single action)
  POST /agent/observe      → Observer agent (event routing)
  GET  /agent/workflows    → List available workflows
"""
from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.api.deps import require_viewer, require_operator, AuthContext

router = APIRouter(tags=["agentic"])
logger = logging.getLogger("nsops.routes.agentic")


# ── Request / Response models ─────────────────────────────────────────────────

class DebugPodRequest(BaseModel):
    namespace: str = Field(..., description="Kubernetes namespace", example="default")
    pod_name: str  = Field(..., description="Pod name to debug",    example="my-app-abc123")
    dry_run:  bool = Field(True,  description="If true, executor logs actions but does NOT execute")
    auto_fix: bool = Field(False, description="If true, executor attempts to fix the issue")


class PlanRequest(BaseModel):
    task: str = Field(..., description="Natural language task description")
    context: Optional[dict] = None


class ExecuteRequest(BaseModel):
    action: str  = Field(..., description="Action name, e.g. restart_pod")
    params: dict = Field(default_factory=dict)
    dry_run: bool = True
    approved: bool = False


class ObserveRequest(BaseModel):
    type: str    = Field(..., description="Event type: k8s_alert|gitlab_pipeline|prometheus_alert|manual_debug")
    source: str  = Field("api", description="Source identifier")
    payload: dict = Field(default_factory=dict)


# ── POST /debug-pod ───────────────────────────────────────────────────────────

@router.post("/debug-pod")
async def debug_pod(
    req: DebugPodRequest,
    auth: AuthContext = Depends(require_viewer),
):
    """
    **Core use case** — K8s Incident Debugging Workflow (LangGraph).

    Flow:
      Planner → Gather Data → Debugger → [Executor?] → Reporter

    Steps:
      1. Planner validates input and checks memory for similar incidents
      2. Gather Data fetches pod details, logs, events via K8s API
      3. Debugger runs LLM analysis → failure_type, root_cause, fix_suggestion
      4. Executor (if auto_fix=true) executes the fix
      5. Reporter generates structured report, stores in memory, notifies Slack if critical

    **dry_run=true (default):** Safe to run in production — no changes made.
    **auto_fix=true:** Executor will attempt fix. Requires operator role.

    Example response includes: failure_type, severity, root_cause, fix_suggestion, report
    """
    if req.auto_fix and not req.dry_run:
        # Auto-fix without dry_run requires operator+
        if auth.role not in ("admin", "operator"):
            raise HTTPException(
                status_code=403,
                detail="auto_fix=true with dry_run=false requires operator or admin role"
            )

    t0 = time.time()
    logger.info("/debug-pod ns=%s pod=%s dry_run=%s auto_fix=%s user=%s",
                req.namespace, req.pod_name, req.dry_run, req.auto_fix, auth.username)

    try:
        from app.workflows.unified_workflow import run_unified
        result = run_unified(
            namespace=req.namespace,
            pod_name=req.pod_name,
            dry_run=req.dry_run,
            auto_fix=req.auto_fix,
        )
    except Exception as e:
        logger.error("/debug-pod error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}")

    elapsed = round(time.time() - t0, 2)

    return {
        "success":         result.get("success", False),
        "namespace":       req.namespace,
        "pod_name":        req.pod_name,
        "failure_type":    result.get("failure_type", "Unknown"),
        "severity":        result.get("severity_ai", "unknown"),
        "root_cause":      result.get("root_cause", ""),
        "fix_suggestion":  result.get("fix_suggestion", ""),
        "fix_executed":    result.get("fix_executed", False),
        "fix_result":      result.get("fix_result", {}),
        "actions_taken":   result.get("actions_taken", []),
        "report":          result.get("report", ""),
        "steps_taken":     result.get("steps_taken", []),
        "step_timings":    result.get("step_timings", {}),
        "errors":          result.get("errors", []),
        "elapsed_s":       elapsed,
        "dry_run":         req.dry_run,
        "auto_fix":        req.auto_fix,
    }


# ── POST /agent/plan ──────────────────────────────────────────────────────────

@router.post("/agent/plan")
async def agent_plan(
    req: PlanRequest,
    auth: AuthContext = Depends(require_viewer),
):
    """Planner agent — decomposes a task into agent-executable steps."""
    try:
        from app.agents.planner import PlannerAgent
        result = PlannerAgent().plan(req.task, req.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /agent/execute ───────────────────────────────────────────────────────

@router.post("/agent/execute")
async def agent_execute(
    req: ExecuteRequest,
    auth: AuthContext = Depends(require_operator),
):
    """
    Executor agent — runs a single action with RBAC and dry-run support.
    Requires operator role.
    """
    try:
        from app.agents.executor import ExecutorAgent
        result = ExecutorAgent().execute(
            action=req.action,
            params=req.params,
            dry_run=req.dry_run,
            user_role=auth.role,
            approved=req.approved,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /agent/observe ───────────────────────────────────────────────────────

@router.post("/agent/observe")
async def agent_observe(
    req: ObserveRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(require_viewer),
):
    """
    Observer agent — processes an event and triggers the appropriate workflow.
    Supports: k8s_alert, gitlab_pipeline, prometheus_alert, manual_debug
    """
    try:
        from app.agents.observer import get_observer
        # Run heavy workflows in background for webhook responsiveness
        event = {"type": req.type, "source": req.source, "payload": req.payload}
        if req.type in ("k8s_alert", "prometheus_alert"):
            background_tasks.add_task(_run_observer, event)
            return {"accepted": True, "type": req.type, "message": "Event accepted, processing in background"}
        else:
            result = get_observer().process_event(event)
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_observer(event: dict):
    try:
        from app.agents.observer import get_observer
        get_observer().process_event(event)
    except Exception as e:
        logger.error("background observer error: %s", e)


# ── GET /agent/workflows ──────────────────────────────────────────────────────

@router.get("/agent/workflows")
def list_workflows(auth: AuthContext = Depends(require_viewer)):
    """List available LangGraph workflows and their capabilities."""
    return {
        "workflows": [
            {
                "id":          "unified_incident",
                "name":        "Unified Incident Workflow",
                "endpoint":    "POST /debug-pod  or  POST /incidents/run",
                "description": "Single LangGraph graph handling ALL incident types: K8s pod debug, AWS, GitHub, Jira, Slack, OpsGenie",
                "agents":      ["planner", "gather_all", "debugger", "executor", "reporter"],
                "inputs":      ["incident_id", "description", "severity", "namespace", "pod_name", "aws_cfg", "dry_run", "auto_fix", "auto_remediate"],
                "outputs":     ["failure_type", "severity_ai", "root_cause", "fix_suggestion", "findings", "actions_taken", "report"],
                "supports_auto_fix":       True,
                "supports_dry_run":        True,
                "supports_auto_remediate": True,
                "sources":     ["kubernetes", "aws", "github"],
                "actions":     ["k8s_restart", "k8s_scale", "github_pr", "jira_ticket", "slack_warroom", "opsgenie_alert"],
            }
        ],
        "agents": [
            {"id": "planner",  "description": "Decomposes tasks into steps. POST /agent/plan"},
            {"id": "debugger", "description": "Analyzes logs and events for root cause"},
            {"id": "executor", "description": "Executes actions with RBAC. POST /agent/execute"},
            {"id": "observer", "description": "Routes events to workflows. POST /agent/observe"},
            {"id": "reporter", "description": "Formats and delivers reports"},
        ],
        "event_types": ["k8s_alert", "gitlab_pipeline", "prometheus_alert", "manual_debug"],
    }
