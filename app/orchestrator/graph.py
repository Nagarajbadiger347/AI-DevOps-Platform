"""LangGraph orchestration graph.

Pipeline flow:
  collect_context
    → plan
    → [decide | escalate]          — escalate if confidence < MIN_CONFIDENCE
    → [execute | awaiting_approval | store_memory]
    → validate
    → [store_memory | execute (retry+backoff) | escalate]
    → store_memory → END
    → escalate     → END

Design rules:
  - Nodes are thin: they call agents / executor / validator and update state.
  - No integration calls in node functions — integrations go through Executor.
  - All logging carries trace_id and incident_id.
  - Retry backoff: BACKOFF_BASE ** retry_count seconds (2s, 4s, 8s).
  - Confidence gate: plans below MIN_CONFIDENCE skip execute and escalate directly.
"""
from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END

from app.core.logging import get_logger, set_context
from app.orchestrator.state import PipelineState
from app.agents.infra.aws_agent import AWSAgent
from app.agents.infra.k8s_agent import K8sAgent
from app.agents.scm.github_agent import GitHubAgent
from app.agents.planner.agent import PlannerAgent
from app.agents.decision.agent import DecisionAgent
from app.agents.memory.agent import MemoryAgent
from app.execution.executor import Executor
from app.execution.validator import Validator

logger = get_logger(__name__)

_MAX_RETRIES          = 3
_MIN_CONFIDENCE       = 0.3    # plans below this confidence → escalate immediately
_BACKOFF_BASE         = 2.0    # retry delays: 2s, 4s, 8s
_CONTEXT_TIMEOUT_SECS = 10     # per-collector timeout — pipeline continues with partial data after this
_APPROVAL_WINDOW_SECS = 1800   # 30-minute approval deadline


# ── Node helpers ───────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _set_pipeline_context(state: PipelineState) -> None:
    """Propagate state identity fields into log context vars."""
    set_context(
        trace_id    = state.get("trace_id", ""),
        correlation_id = state.get("correlation_id", ""),
        incident_id = state.get("incident_id", ""),
        user        = state.get("user", "") or state.get("metadata", {}).get("user", ""),
        tenant_id   = state.get("tenant_id", ""),
    )


# ── Node functions ─────────────────────────────────────────────────────────────

def collect_context(state: PipelineState) -> PipelineState:
    """Parallel context collection from AWS, K8s, and GitHub."""
    _set_pipeline_context(state)

    tasks = {
        "aws":    lambda: AWSAgent().run(state),
        "k8s":    lambda: K8sAgent().run(state),
        "github": lambda: GitHubAgent().run(state),
    }
    results: dict = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures, timeout=_CONTEXT_TIMEOUT_SECS):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"_data_available": False, "_reason": str(exc)}
                state.setdefault("errors", []).append(f"collect_context/{name}: {exc}")
                logger.warning("context_collector_failed", extra={
                    "incident_id": state.get("incident_id"),
                    "trace_id":    state.get("trace_id"),
                    "collector":   name,
                    "error":       str(exc),
                })

    state["aws_context"]    = results.get("aws",    {})
    state["k8s_context"]    = results.get("k8s",    {})
    state["github_context"] = results.get("github", {})

    state["similar_incidents"] = MemoryAgent.retrieve_similar(
        state.get("description", ""), n=5
    )

    logger.info("context_collected", extra={
        "incident_id":       state.get("incident_id"),
        "trace_id":          state.get("trace_id"),
        "aws_ok":            state["aws_context"].get("_data_available", False),
        "k8s_ok":            state["k8s_context"].get("_data_available", False),
        "github_ok":         state["github_context"].get("_data_available", False),
        "similar_incidents": len(state["similar_incidents"]),
    })
    return state


def plan(state: PipelineState) -> PipelineState:
    """Run the PlannerAgent and merge its result into state."""
    _set_pipeline_context(state)

    try:
        result     = PlannerAgent().run(state)
        plan_data  = result.get("plan", {}) if isinstance(result, dict) else {}
        confidence = plan_data.get("confidence", 0.0)
        actions    = plan_data.get("actions", [])
        risk       = plan_data.get("risk")

        logger.info("plan_generated", extra={
            "incident_id":  state.get("incident_id"),
            "trace_id":     state.get("trace_id"),
            "confidence":   confidence,
            "risk":         risk,
            "action_count": len(actions),
        })

        if isinstance(result, dict):
            state.update(result)

    except Exception as exc:
        logger.error("plan_node_failed", extra={
            "incident_id": state.get("incident_id"),
            "trace_id":    state.get("trace_id"),
            "error":       str(exc),
        })
        state.setdefault("errors", []).append(f"plan: {exc}")
        # Inject a stub plan so routing can still escalate cleanly
        state.setdefault("plan", {"confidence": 0.0, "risk": "unknown", "actions": []})

    return state


def decide(state: PipelineState) -> PipelineState:
    """Run the DecisionAgent to score risk and set approval requirements."""
    _set_pipeline_context(state)

    try:
        result = DecisionAgent().run(state)
        if isinstance(result, dict):
            state.update(result)
        logger.info("decision_made", extra={
            "incident_id":           state.get("incident_id"),
            "trace_id":              state.get("trace_id"),
            "risk_score":            state.get("risk_score"),
            "requires_approval":     state.get("requires_human_approval"),
        })
    except Exception as exc:
        logger.error("decide_node_failed", extra={
            "incident_id": state.get("incident_id"),
            "trace_id":    state.get("trace_id"),
            "error":       str(exc),
        })
        state.setdefault("errors", []).append(f"decide: {exc}")
        state["requires_human_approval"] = True
        state["approval_reason"] = f"Decision agent failed: {exc}"

    return state


def execute(state: PipelineState) -> PipelineState:
    """Execute planned actions with exponential backoff on retries."""
    _set_pipeline_context(state)

    retry = state.get("retry_count", 0)
    if retry > 0:
        delay = _BACKOFF_BASE ** retry
        logger.info("execute_retry_backoff", extra={
            "incident_id": state.get("incident_id"),
            "trace_id":    state.get("trace_id"),
            "retry":       retry,
            "delay_s":     delay,
        })
        time.sleep(delay)

    return Executor().run(state)


def validate(state: PipelineState) -> PipelineState:
    """Verify that executed actions achieved the desired state."""
    _set_pipeline_context(state)
    return Validator().run(state)


def store_memory(state: PipelineState) -> PipelineState:
    """Mark the pipeline complete and persist to ChromaDB."""
    _set_pipeline_context(state)
    state["completed_at"] = _now_iso()
    state["status"]       = "completed"

    try:
        result = MemoryAgent().run(state)
        if isinstance(result, dict):
            state.update(result)
    except Exception as exc:
        logger.warning("store_memory_failed", extra={
            "incident_id": state.get("incident_id"),
            "trace_id":    state.get("trace_id"),
            "error":       str(exc),
        })

    logger.info("pipeline_completed", extra={
        "incident_id":     state.get("incident_id"),
        "trace_id":        state.get("trace_id"),
        "status":          state.get("status"),
        "actions_executed": len(state.get("executed_actions", [])),
        "actions_blocked":  len(state.get("blocked_actions", [])),
    })
    return state


def escalate(state: PipelineState) -> PipelineState:
    """Notify on-call and Slack when the pipeline cannot auto-remediate."""
    _set_pipeline_context(state)

    incident_id = state.get("incident_id", "unknown")
    last_error  = (state.get("errors") or ["unknown"])[-1]
    retry_count = state.get("retry_count", 0)
    confidence  = state.get("plan", {}).get("confidence")

    logger.warning("escalating_incident", extra={
        "incident_id": incident_id,
        "trace_id":    state.get("trace_id"),
        "retry_count": retry_count,
        "last_error":  last_error,
        "confidence":  confidence,
    })

    try:
        from app.integrations.slack import post_message
        post_message(
            channel=state.get("metadata", {}).get("slack_channel", "#incidents"),
            text=(
                f":sos: *Escalation* — incident `{incident_id}` requires human attention.\n"
                f"Retries: {retry_count} · Last error: `{last_error}`\n"
                f"Plan confidence: `{confidence}`\n"
                f"Trace ID: `{state.get('trace_id', 'n/a')}`"
            ),
        )
    except Exception as exc:
        logger.warning("escalate_slack_failed", extra={
            "incident_id": incident_id,
            "error":       str(exc),
        })

    try:
        from app.integrations.opsgenie import notify_on_call
        notify_on_call(
            message=f"Auto-remediation failed for {incident_id}: {last_error}",
            alias=f"escalation-{incident_id}",
        )
    except Exception as exc:
        logger.warning("escalate_opsgenie_failed", extra={
            "incident_id": incident_id,
            "error":       str(exc),
        })

    state["status"]       = "escalated"
    state["completed_at"] = _now_iso()
    return state


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_plan(state: PipelineState) -> str:
    """Low-confidence plans skip execute entirely and go straight to escalate."""
    confidence = state.get("plan", {}).get("confidence", 0.0)
    if confidence < _MIN_CONFIDENCE:
        state.setdefault("errors", []).append(
            f"Plan confidence {confidence:.2f} < threshold {_MIN_CONFIDENCE} — escalating"
        )
        logger.warning("plan_confidence_too_low", extra={
            "incident_id": state.get("incident_id"),
            "trace_id":    state.get("trace_id"),
            "confidence":  confidence,
            "threshold":   _MIN_CONFIDENCE,
        })
        return "escalate"
    return "decide"


def _route_after_decide(state: PipelineState) -> str:
    if state.get("requires_human_approval"):
        state["status"]            = "awaiting_approval"
        state["approval_deadline"] = time.time() + _APPROVAL_WINDOW_SECS
        state["completed_at"]      = _now_iso()
        logger.info("pipeline_awaiting_approval", extra={
            "incident_id":       state.get("incident_id"),
            "trace_id":          state.get("trace_id"),
            "reason":            state.get("approval_reason"),
            "deadline_unix":     state["approval_deadline"],
        })
        return "awaiting_approval"

    if not state.get("plan", {}).get("actions"):
        return "store_memory"

    return "execute"


def _route_after_validate(state: PipelineState) -> str:
    if state.get("validation_passed"):
        return "store_memory"

    retry = state.get("retry_count", 0)
    if retry < _MAX_RETRIES:
        logger.info("pipeline_retrying", extra={
            "incident_id": state.get("incident_id"),
            "trace_id":    state.get("trace_id"),
            "retry_count": retry,
            "next_delay_s": _BACKOFF_BASE ** (retry + 1),
        })
        return "execute"

    logger.warning("max_retries_exhausted", extra={
        "incident_id": state.get("incident_id"),
        "trace_id":    state.get("trace_id"),
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

    g.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"decide": "decide", "escalate": "escalate"},
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
