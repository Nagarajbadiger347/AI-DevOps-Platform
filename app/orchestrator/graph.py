"""LangGraph orchestration graph — defines the full autonomous pipeline workflow.

Flow:
  collect_context
    → plan
    → [decide | escalate]          ← escalate if confidence < 0.3
    → [execute | awaiting_approval | store_memory]
    → validate
    → [store_memory | execute(retry+backoff) | escalate]
    → store_memory → END
    → escalate     → END
"""
from __future__ import annotations

import datetime
import time

from langgraph.graph import StateGraph, END

from app.orchestrator.state import PipelineState
from app.agents.infra.aws_agent import AWSAgent
from app.agents.infra.k8s_agent import K8sAgent
from app.agents.scm.github_agent import GitHubAgent
from app.agents.planner.agent import PlannerAgent
from app.agents.decision.agent import DecisionAgent
from app.agents.memory.agent import MemoryAgent
from app.execution.executor import Executor
from app.execution.validator import Validator
from app.core.logging import get_logger, set_context

logger = get_logger(__name__)

_MAX_RETRIES = 3
_MIN_CONFIDENCE_TO_DECIDE = 0.3   # below this → escalate immediately, skip execute
_BACKOFF_BASE = 2.0                # retry delay: 2s, 4s, 8s


# ── Node functions ────────────────────────────────────────────────────────────

def collect_context(state: PipelineState) -> PipelineState:
    """Parallel context collection from AWS, K8s, and GitHub."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    set_context(
        incident_id=state.get("incident_id", ""),
        user=state.get("metadata", {}).get("user", ""),
    )

    tasks = {
        "aws":    lambda: AWSAgent().run(state),
        "k8s":    lambda: K8sAgent().run(state),
        "github": lambda: GitHubAgent().run(state),
    }
    results: dict = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures, timeout=25):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"_data_available": False, "_reason": str(exc)}
                state.setdefault("errors", []).append(f"collect_context/{name}: {exc}")

    state["aws_context"]    = results.get("aws",    {})
    state["k8s_context"]    = results.get("k8s",    {})
    state["github_context"] = results.get("github", {})

    state["similar_incidents"] = MemoryAgent.retrieve_similar(
        state.get("description", ""), n=5
    )

    logger.info("context_collected", extra={
        "incident_id":       state.get("incident_id"),
        "aws_ok":            state["aws_context"].get("_data_available", False),
        "k8s_ok":            state["k8s_context"].get("_data_available", False),
        "github_ok":         state["github_context"].get("_data_available", False),
        "similar_incidents": len(state["similar_incidents"]),
    })
    return state


def plan(state: PipelineState) -> PipelineState:
    result = PlannerAgent().run(state)
    confidence = result.get("plan", {}).get("confidence", 0.0) if isinstance(result, dict) else 0.0
    logger.info("plan_generated", extra={
        "incident_id": state.get("incident_id"),
        "confidence":  confidence,
        "risk":        result.get("plan", {}).get("risk") if isinstance(result, dict) else None,
        "action_count": len(result.get("plan", {}).get("actions", [])) if isinstance(result, dict) else 0,
    })
    if isinstance(result, dict):
        state.update(result)
    return state


def decide(state: PipelineState) -> PipelineState:
    result = DecisionAgent().run(state)
    if isinstance(result, dict):
        state.update(result)
    return state


def execute(state: PipelineState) -> PipelineState:
    """Execute actions with exponential backoff on retries."""
    retry = state.get("retry_count", 0)
    if retry > 0:
        delay = _BACKOFF_BASE ** retry   # 2s, 4s, 8s
        logger.info("execute_retry_backoff", extra={
            "incident_id": state.get("incident_id"),
            "retry":       retry,
            "delay_s":     delay,
        })
        time.sleep(delay)

    return Executor().run(state)


def validate(state: PipelineState) -> PipelineState:
    return Validator().run(state)


def store_memory(state: PipelineState) -> PipelineState:
    state["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state["status"] = "completed"
    return MemoryAgent().run(state)


def escalate(state: PipelineState) -> PipelineState:
    """Notify on-call and Slack when the pipeline cannot auto-remediate."""
    incident_id = state.get("incident_id", "unknown")
    last_error  = (state.get("errors") or ["unknown"])[-1]

    logger.warning("escalating_incident", extra={
        "incident_id": incident_id,
        "retry_count": state.get("retry_count", 0),
        "last_error":  last_error,
        "confidence":  state.get("plan", {}).get("confidence"),
    })

    try:
        from app.integrations.slack import post_message
        post_message(
            channel=state.get("metadata", {}).get("slack_channel", "#incidents"),
            text=(
                f":sos: *Escalation* — incident `{incident_id}` failed after "
                f"{state.get('retry_count', 0)} retries.\n"
                f"Last error: `{last_error}`\n"
                f"Plan risk: `{state.get('plan', {}).get('risk', 'unknown')}`"
            ),
        )
    except Exception as exc:
        logger.warning("escalate_slack_failed", extra={"error": str(exc)})

    try:
        from app.integrations.opsgenie import notify_on_call
        notify_on_call(
            message=f"Auto-remediation failed for incident {incident_id}: {last_error}",
            alias=f"escalation-{incident_id}",
        )
    except Exception as exc:
        logger.warning("escalate_opsgenie_failed", extra={"error": str(exc)})

    state["status"]       = "escalated"
    state["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return state


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_plan(state: PipelineState) -> str:
    """Escalate immediately if confidence is too low — don't waste a decide + execute cycle."""
    confidence = state.get("plan", {}).get("confidence", 0.0)
    if confidence < _MIN_CONFIDENCE_TO_DECIDE:
        state.setdefault("errors", []).append(
            f"Plan confidence {confidence:.2f} below threshold {_MIN_CONFIDENCE_TO_DECIDE} — escalating"
        )
        logger.warning("plan_confidence_too_low", extra={
            "incident_id": state.get("incident_id"),
            "confidence":  confidence,
            "threshold":   _MIN_CONFIDENCE_TO_DECIDE,
        })
        return "escalate"
    return "decide"


def _route_after_decide(state: PipelineState) -> str:
    if state.get("requires_human_approval"):
        state["status"]           = "awaiting_approval"
        state["approval_deadline"] = time.time() + 1800   # 30-minute window
        state["completed_at"]     = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info("pipeline_awaiting_approval", extra={
            "incident_id":       state.get("incident_id"),
            "reason":            state.get("approval_reason"),
            "approval_deadline": state["approval_deadline"],
        })
        return "awaiting_approval"

    actions = state.get("plan", {}).get("actions", [])
    if not actions:
        return "store_memory"

    return "execute"


def _route_after_validate(state: PipelineState) -> str:
    if state.get("validation_passed"):
        return "store_memory"

    retry = state.get("retry_count", 0)
    if retry < _MAX_RETRIES:
        logger.info("pipeline_retrying", extra={
            "incident_id": state.get("incident_id"),
            "retry_count": retry,
            "next_delay_s": _BACKOFF_BASE ** (retry + 1),
        })
        return "execute"

    logger.warning("max_retries_exhausted", extra={
        "incident_id": state.get("incident_id"),
        "max_retries": _MAX_RETRIES,
    })
    return "escalate"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    g.add_node("collect_context", collect_context)
    g.add_node("plan",            plan)
    g.add_node("decide",          decide)
    g.add_node("execute",         execute)
    g.add_node("validate",        validate)
    g.add_node("store_memory",    store_memory)
    g.add_node("escalate",        escalate)

    g.set_entry_point("collect_context")
    g.add_edge("collect_context", "plan")

    # Confidence gate — low confidence goes straight to escalate
    g.add_conditional_edges(
        "plan",
        _route_after_plan,
        {
            "decide":   "decide",
            "escalate": "escalate",
        },
    )

    g.add_conditional_edges(
        "decide",
        _route_after_decide,
        {
            "execute":           "execute",
            "store_memory":      "store_memory",
            "awaiting_approval": END,
        },
    )

    g.add_edge("execute", "validate")

    # Feedback loop — failed validation retries execute with backoff
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "store_memory": "store_memory",
            "execute":      "execute",
            "escalate":     "escalate",
        },
    )

    g.add_edge("store_memory", END)
    g.add_edge("escalate",     END)

    return g.compile()


# Singleton compiled graph — imported by runner.py
pipeline_graph = build_graph()
