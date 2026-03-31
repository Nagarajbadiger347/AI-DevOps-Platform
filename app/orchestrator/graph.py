"""LangGraph orchestration graph — defines the full autonomous pipeline workflow.

Flow:
  collect_context
    → plan
    → decide
    → [execute | store_memory | END(awaiting_approval)]
    → validate
    → [store_memory | execute(retry) | escalate]
    → store_memory → END
    → escalate     → END
"""
from __future__ import annotations

import datetime

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
from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3

# ── Node functions ───────────────────────────────────────────────────────────


def collect_context(state: PipelineState) -> PipelineState:
    """Parallel context collection from AWS, K8s, and GitHub."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = {
        "aws":    lambda: AWSAgent().run(state),
        "k8s":    lambda: K8sAgent().run(state),
        "github": lambda: GitHubAgent().run(state),
    }
    results: dict = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = {"_data_available": False, "_reason": str(exc)}
                state.setdefault("errors", []).append(f"collect_context/{name}: {exc}")

    state["aws_context"]    = results.get("aws",    {})
    state["k8s_context"]    = results.get("k8s",    {})
    state["github_context"] = results.get("github", {})

    # Retrieve similar incidents from memory to inform the planner
    state["similar_incidents"] = MemoryAgent.retrieve_similar(
        state.get("description", ""), n=5
    )

    logger.info(
        "context_collected",
        incident_id=state.get("incident_id"),
        aws_ok=state["aws_context"].get("_data_available", False),
        k8s_ok=state["k8s_context"].get("_data_available", False),
        github_ok=state["github_context"].get("_data_available", False),
        similar_incidents=len(state["similar_incidents"]),
    )
    return state


def plan(state: PipelineState) -> PipelineState:
    return PlannerAgent().run(state)


def decide(state: PipelineState) -> PipelineState:
    return DecisionAgent().run(state)


def execute(state: PipelineState) -> PipelineState:
    return Executor().run(state)


def validate(state: PipelineState) -> PipelineState:
    return Validator().run(state)


def store_memory(state: PipelineState) -> PipelineState:
    state["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return MemoryAgent().run(state)


def escalate(state: PipelineState) -> PipelineState:
    """Notify on-call and Slack when auto-remediation has exhausted retries."""
    incident_id = state.get("incident_id", "unknown")
    last_error  = (state.get("errors") or ["unknown"])[-1]

    logger.warning("escalating_incident",
                   incident_id=incident_id,
                   retry_count=state.get("retry_count", 0),
                   last_error=last_error)

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
        logger.warning("escalate_slack_failed", error=str(exc))

    try:
        from app.integrations.opsgenie import notify_on_call
        notify_on_call(
            message=f"Auto-remediation failed for incident {incident_id}: {last_error}",
            alias=f"escalation-{incident_id}",
        )
    except Exception as exc:
        logger.warning("escalate_opsgenie_failed", error=str(exc))

    state["status"]       = "escalated"
    state["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return state


# ── Routing functions ────────────────────────────────────────────────────────


def _route_after_decide(state: PipelineState) -> str:
    if state.get("requires_human_approval"):
        logger.info("pipeline_awaiting_approval",
                    incident_id=state.get("incident_id"),
                    reason=state.get("approval_reason"))
        return "awaiting_approval"

    actions = state.get("plan", {}).get("actions", [])
    if not actions:
        return "store_memory"

    return "execute"


def _route_after_validate(state: PipelineState) -> str:
    if state.get("validation_passed"):
        return "store_memory"
    if state.get("retry_count", 0) < _MAX_RETRIES:
        logger.info("pipeline_retrying",
                    incident_id=state.get("incident_id"),
                    retry_count=state.get("retry_count"))
        return "execute"
    return "escalate"


# ── Graph assembly ───────────────────────────────────────────────────────────


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
    g.add_edge("plan",            "decide")

    g.add_conditional_edges(
        "decide",
        _route_after_decide,
        {
            "execute":            "execute",
            "store_memory":       "store_memory",
            "awaiting_approval":  END,
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
