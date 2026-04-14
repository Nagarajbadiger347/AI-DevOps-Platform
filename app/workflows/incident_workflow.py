"""
Kubernetes Incident Debugging Workflow — LangGraph multi-agent implementation.

Flow:
  planner → gather_data → debugger → [executor?] → reporter → END

State moves through nodes; each node reads from state, writes back updates.
The LLM is used ONLY for reasoning (analysis, root cause, fix suggestion).
All data collection uses real tool calls.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Annotated, TypedDict, Any

from langgraph.graph import StateGraph, END

logger = logging.getLogger("nsops.workflow.incident")


# ── State ─────────────────────────────────────────────────────────────────────

class IncidentState(TypedDict):
    # Input
    namespace: str
    pod_name: str
    dry_run: bool
    auto_fix: bool                  # whether executor should attempt fix

    # Gathered data
    pod_details: dict
    logs: str
    events: list[dict]
    resource_usage: dict

    # Analysis outputs
    failure_type: str               # e.g. CrashLoopBackOff, OOMKilled, ImagePullBackOff
    root_cause: str
    fix_suggestion: str
    severity: str                   # critical/high/medium/low

    # Execution
    fix_executed: bool
    fix_result: dict

    # Observability
    steps_taken: list[str]
    step_timings: dict[str, float]
    errors: list[str]

    # Output
    report: str
    success: bool


# ── Helper: LLM call ──────────────────────────────────────────────────────────

def _llm(system: str, user: str, max_tokens: int = 1024) -> str:
    """Call the platform LLM (Groq-first) for reasoning."""
    try:
        from app.chat.intelligence import _llm_call
        return _llm_call(user, system=system, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"[LLM unavailable: {e}]"


# ── Helper: Memory ────────────────────────────────────────────────────────────

def _recall_similar_incidents(description: str) -> str:
    """Search memory for similar past incidents."""
    try:
        from app.memory.vector_db import search_similar_incidents
        results = search_similar_incidents(description, n_results=3)
        if not results:
            return ""
        parts = ["Similar past incidents:"]
        for r in results[:3]:
            doc = r if isinstance(r, str) else r.get("document", str(r))
            parts.append(f"  • {doc[:200]}")
        return "\n".join(parts)
    except Exception:
        return ""


def _store_incident(description: str, resolution: str) -> None:
    """Persist incident + resolution to vector memory."""
    try:
        from app.memory.vector_db import store_incident
        store_incident(description, resolution)
    except Exception:
        pass


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def planner_node(state: IncidentState) -> dict:
    """
    Plans the debugging steps.
    Validates input, checks memory for similar incidents, decides if auto-fix is safe.
    """
    t0 = time.time()
    steps = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))
    timings = dict(state.get("step_timings", {}))

    steps.append("planner: validating input and planning debug steps")
    logger.info("[PLANNER] ns=%s pod=%s dry_run=%s", state["namespace"], state["pod_name"], state.get("dry_run"))

    # Recall similar past incidents from memory
    memory_hint = _recall_similar_incidents(
        f"pod {state['pod_name']} namespace {state['namespace']} crash"
    )
    if memory_hint:
        steps.append(f"planner: found memory context — {memory_hint[:100]}")
        logger.info("[PLANNER] memory_hint found")

    timings["planner"] = round(time.time() - t0, 2)
    return {
        "steps_taken": steps,
        "step_timings": timings,
        "errors": errors,
        # Inject memory hint as part of pod_details for downstream use
        "pod_details": {"_memory_hint": memory_hint},
    }


# ── Node 2: Gather Data ───────────────────────────────────────────────────────

def gather_data_node(state: IncidentState) -> dict:
    """
    Fetches pod details, logs, events, and resource usage in parallel.
    Uses KubernetesTool — no LLM involved.
    """
    t0 = time.time()
    steps = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))

    from app.tools.kubernetes import KubernetesTool
    k8s = KubernetesTool()

    ns = state["namespace"]
    pod = state["pod_name"]

    # 1. Pod details
    steps.append(f"gather: fetching pod details for {pod} in {ns}")
    pod_result = k8s.describe_pod(namespace=ns, pod=pod)
    pod_details = state.get("pod_details", {})
    if pod_result["success"]:
        pod_details.update(pod_result["data"] or {})
        logger.info("[GATHER] pod_details status=%s", pod_details.get("status"))
    else:
        errors.append(f"describe_pod: {pod_result['error']}")
        # Try listing all pods to find it
        all_pods = k8s.get_pods(namespace=ns)
        if all_pods["success"]:
            match = next((p for p in all_pods["data"] if p.get("name") == pod), None)
            if match:
                pod_details.update(match)

    # 2. Logs
    steps.append(f"gather: fetching logs for {pod}")
    logs_result = k8s.get_logs(namespace=ns, pod=pod, tail_lines=200)
    logs = logs_result["data"] if logs_result["success"] else ""
    if not logs_result["success"]:
        errors.append(f"get_logs: {logs_result['error']}")

    # 3. Events
    steps.append(f"gather: fetching events for namespace {ns}")
    events_result = k8s.get_events(namespace=ns, limit=30)
    all_events = events_result["data"] if events_result["success"] else []
    # Filter events relevant to this pod
    pod_events = [e for e in all_events
                  if pod in str(e.get("involved_object", "")) or pod in str(e.get("name", ""))]
    events = pod_events or all_events[:10]
    if not events_result["success"]:
        errors.append(f"get_events: {events_result['error']}")

    # 4. Resource usage
    steps.append(f"gather: fetching resource usage for {ns}")
    usage_result = k8s.get_resource_usage(namespace=ns)
    resource_usage = usage_result["data"] if usage_result["success"] else {}

    timings = dict(state.get("step_timings", {}))
    timings["gather_data"] = round(time.time() - t0, 2)

    logger.info("[GATHER] logs_len=%d events=%d errors=%d",
                len(logs), len(events), len(errors))

    return {
        "pod_details": pod_details,
        "logs": logs,
        "events": events,
        "resource_usage": resource_usage,
        "steps_taken": steps,
        "step_timings": timings,
        "errors": errors,
    }


# ── Node 3: Debugger ──────────────────────────────────────────────────────────

def debugger_node(state: IncidentState) -> dict:
    """
    Analyzes gathered data using LLM reasoning.
    Determines failure type, root cause, severity, and fix suggestion.
    """
    t0 = time.time()
    steps = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))

    steps.append("debugger: analyzing failure with LLM")
    logger.info("[DEBUGGER] starting analysis")

    pod_details = state.get("pod_details", {})
    logs = state.get("logs", "")
    events = state.get("events", [])
    memory_hint = pod_details.pop("_memory_hint", "")

    # Build analysis context
    events_text = json.dumps(events[:10], indent=2, default=str) if events else "No events found."
    pod_text = json.dumps({k: v for k, v in pod_details.items()
                           if k not in ("_memory_hint",)}, indent=2, default=str)
    logs_snippet = logs[-3000:] if len(logs) > 3000 else logs  # last 3000 chars most relevant

    context = f"""
Pod: {state['pod_name']} | Namespace: {state['namespace']}

=== Pod Details ===
{pod_text}

=== Recent Logs (last 200 lines) ===
{logs_snippet or 'No logs available'}

=== Kubernetes Events ===
{events_text}

=== Memory: Similar Past Incidents ===
{memory_hint or 'No similar past incidents found.'}
""".strip()

    system = """You are an expert Kubernetes SRE debugging a pod failure.

Analyze the provided pod details, logs, and events. Respond ONLY with valid JSON:
{
  "failure_type": "<CrashLoopBackOff|OOMKilled|ImagePullBackOff|Pending|Error|Unknown>",
  "severity": "<critical|high|medium|low>",
  "root_cause": "<1-3 sentence root cause explanation>",
  "fix_suggestion": "<specific actionable fix — include exact commands if relevant>",
  "can_auto_fix": <true|false>,
  "auto_fix_action": "<restart_pod|restart_deployment|scale_deployment|none>"
}

Rules:
- can_auto_fix = true ONLY for CrashLoopBackOff or simple restarts
- Be specific — use real values from the data
- If logs are empty, say so in root_cause"""

    raw = _llm(system, context, max_tokens=600)

    # Parse JSON from LLM response
    failure_type = "Unknown"
    severity = "medium"
    root_cause = "Could not determine root cause."
    fix_suggestion = "Review pod logs and events manually."
    can_auto_fix = False
    auto_fix_action = "none"

    try:
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        failure_type = parsed.get("failure_type", failure_type)
        severity = parsed.get("severity", severity)
        root_cause = parsed.get("root_cause", root_cause)
        fix_suggestion = parsed.get("fix_suggestion", fix_suggestion)
        can_auto_fix = parsed.get("can_auto_fix", False)
        auto_fix_action = parsed.get("auto_fix_action", "none")
    except Exception as e:
        # LLM didn't return clean JSON — extract what we can
        errors.append(f"debugger_json_parse: {e} — raw: {raw[:200]}")
        # Heuristic fallback from raw text
        lower = raw.lower()
        if "crashloopbackoff" in lower:
            failure_type = "CrashLoopBackOff"
        elif "oomkilled" in lower or "oom" in lower:
            failure_type = "OOMKilled"
        elif "imagepull" in lower:
            failure_type = "ImagePullBackOff"
        root_cause = raw[:500] if raw else root_cause

    steps.append(f"debugger: failure_type={failure_type} severity={severity}")
    logger.info("[DEBUGGER] failure=%s severity=%s can_auto_fix=%s",
                failure_type, severity, can_auto_fix)

    # Only auto-fix if explicitly enabled AND LLM says it's safe
    auto_fix = state.get("auto_fix", False) and can_auto_fix

    timings = dict(state.get("step_timings", {}))
    timings["debugger"] = round(time.time() - t0, 2)

    return {
        "failure_type": failure_type,
        "severity": severity,
        "root_cause": root_cause,
        "fix_suggestion": fix_suggestion,
        "auto_fix": auto_fix,
        "steps_taken": steps,
        "step_timings": timings,
        "errors": errors,
        "pod_details": pod_details,
    }


# ── Node 4: Executor ──────────────────────────────────────────────────────────

def executor_node(state: IncidentState) -> dict:
    """
    Optionally executes the suggested fix.
    Respects dry_run flag — in dry-run mode, only logs what it would do.
    """
    t0 = time.time()
    steps = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))

    from app.tools.kubernetes import KubernetesTool
    k8s = KubernetesTool()

    ns = state["namespace"]
    pod = state["pod_name"]
    dry_run = state.get("dry_run", True)
    failure_type = state.get("failure_type", "Unknown")

    fix_result = {"executed": False, "dry_run": dry_run}

    # Determine action based on failure type
    action = "restart_pod"  # default safe action
    if failure_type == "OOMKilled":
        # OOM needs scale or resource limit change — just restart for now
        action = "restart_pod"
    elif failure_type in ("CrashLoopBackOff", "Error"):
        action = "restart_pod"
    elif failure_type == "ImagePullBackOff":
        # Can't fix image pull with a restart — skip auto-fix
        action = "none"
        steps.append("executor: ImagePullBackOff — skipping auto-fix (image issue requires manual fix)")

    if action == "none":
        fix_result["reason"] = "Auto-fix not applicable for this failure type"
    elif dry_run:
        steps.append(f"executor: DRY-RUN — would {action} pod={pod} ns={ns}")
        fix_result["action"] = action
        fix_result["message"] = f"[DRY-RUN] Would execute: {action} on {pod} in {ns}"
        logger.info("[EXECUTOR] dry_run action=%s ns=%s pod=%s", action, ns, pod)
    else:
        steps.append(f"executor: executing {action} on {pod} in {ns}")
        logger.info("[EXECUTOR] executing action=%s ns=%s pod=%s", action, ns, pod)
        if action == "restart_pod":
            result = k8s.restart_pod(namespace=ns, pod=pod)
        elif action == "restart_deployment":
            deployment = state.get("pod_details", {}).get("owner", pod.rsplit("-", 2)[0])
            result = k8s.restart_deployment(namespace=ns, deployment=deployment)
        else:
            result = {"success": False, "error": f"Unknown action: {action}"}

        fix_result.update({"executed": True, "action": action, **result})
        if not result.get("success"):
            errors.append(f"executor: {result.get('error', 'unknown error')}")

    timings = dict(state.get("step_timings", {}))
    timings["executor"] = round(time.time() - t0, 2)

    return {
        "fix_executed": fix_result.get("executed", False),
        "fix_result": fix_result,
        "steps_taken": steps,
        "step_timings": timings,
        "errors": errors,
    }


# ── Node 5: Reporter ──────────────────────────────────────────────────────────

def reporter_node(state: IncidentState) -> dict:
    """
    Generates the final structured report.
    Stores incident in memory for future recall.
    Optionally sends Slack notification for critical incidents.
    """
    t0 = time.time()
    steps = list(state.get("steps_taken", []))

    steps.append("reporter: generating final report")

    ns = state["namespace"]
    pod = state["pod_name"]
    failure_type = state.get("failure_type", "Unknown")
    severity = state.get("severity", "unknown")
    root_cause = state.get("root_cause", "N/A")
    fix_suggestion = state.get("fix_suggestion", "N/A")
    fix_result = state.get("fix_result", {})
    errors = state.get("errors", [])
    timings = state.get("step_timings", {})
    total_time = sum(timings.values())

    # Build report
    status_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")

    fix_status = ""
    if fix_result.get("dry_run"):
        fix_status = f"\n**Fix (DRY-RUN):** {fix_result.get('message', 'Would execute fix')}"
    elif fix_result.get("executed"):
        ok = fix_result.get("success", False)
        fix_status = f"\n**Fix Executed:** {'✅ Success' if ok else '❌ Failed'} — {fix_result.get('action', '')}"
    else:
        fix_status = "\n**Fix:** Not executed (set auto_fix=true to enable)"

    error_section = ""
    if errors:
        error_section = f"\n**Warnings:** {'; '.join(errors[:3])}"

    report = f"""## {status_emoji} K8s Incident Report

**Pod:** `{pod}` | **Namespace:** `{ns}`
**Failure Type:** `{failure_type}` | **Severity:** {severity.upper()}

### Root Cause
{root_cause}

### Suggested Fix
{fix_suggestion}
{fix_status}
{error_section}

### Debug Steps Taken
{chr(10).join(f'  {i+1}. {s}' for i, s in enumerate(steps))}

### Timing
{chr(10).join(f'  - {k}: {v}s' for k, v in timings.items())}
Total: {total_time:.1f}s
""".strip()

    # Store in memory for future recall
    _store_incident(
        description=f"pod {pod} ns {ns} failure: {failure_type} — {root_cause[:150]}",
        resolution=fix_suggestion[:300],
    )

    # Notify Slack for critical incidents
    if severity == "critical":
        try:
            from app.integrations.slack import send_slack_message
            send_slack_message(
                f"{status_emoji} *Critical K8s Incident*\n"
                f"Pod: `{pod}` | NS: `{ns}` | Type: `{failure_type}`\n"
                f"Root cause: {root_cause[:200]}"
            )
            steps.append("reporter: sent Slack notification (critical severity)")
        except Exception:
            pass

    timings["reporter"] = round(time.time() - t0, 2)
    logger.info("[REPORTER] report generated len=%d", len(report))

    return {
        "report": report,
        "success": True,
        "steps_taken": steps,
        "step_timings": timings,
    }


# ── Conditional routing ───────────────────────────────────────────────────────

def should_execute(state: IncidentState) -> str:
    """Route to executor if auto_fix is enabled, else go straight to reporter."""
    if state.get("auto_fix", False) and not state.get("dry_run", True):
        return "executor"
    if state.get("dry_run", True) and state.get("auto_fix", False):
        # Still run executor in dry-run mode so we log what would happen
        return "executor"
    return "reporter"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_incident_graph() -> Any:
    """Compile and return the LangGraph incident debugging workflow."""
    graph = StateGraph(IncidentState)

    graph.add_node("planner",     planner_node)
    graph.add_node("gather_data", gather_data_node)
    graph.add_node("debugger",    debugger_node)
    graph.add_node("executor",    executor_node)
    graph.add_node("reporter",    reporter_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",     "gather_data")
    graph.add_edge("gather_data", "debugger")
    graph.add_conditional_edges("debugger", should_execute, {
        "executor": "executor",
        "reporter": "reporter",
    })
    graph.add_edge("executor", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


# Singleton compiled graph — built once at import time
_incident_graph = None


def get_incident_graph():
    global _incident_graph
    if _incident_graph is None:
        _incident_graph = build_incident_graph()
    return _incident_graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_incident_debug(
    namespace: str,
    pod_name: str,
    dry_run: bool = True,
    auto_fix: bool = False,
) -> dict:
    """
    Entry point for the K8s incident debugging workflow.

    Args:
        namespace:  Kubernetes namespace
        pod_name:   Pod to debug
        dry_run:    If True, executor logs actions but does not execute
        auto_fix:   If True, executor attempts to fix the issue

    Returns:
        Full IncidentState dict with report, root_cause, fix_suggestion, etc.
    """
    initial_state: IncidentState = {
        "namespace":      namespace,
        "pod_name":       pod_name,
        "dry_run":        dry_run,
        "auto_fix":       auto_fix,
        "pod_details":    {},
        "logs":           "",
        "events":         [],
        "resource_usage": {},
        "failure_type":   "",
        "root_cause":     "",
        "fix_suggestion": "",
        "severity":       "medium",
        "fix_executed":   False,
        "fix_result":     {},
        "steps_taken":    [],
        "step_timings":   {},
        "errors":         [],
        "report":         "",
        "success":        False,
    }

    graph = get_incident_graph()
    logger.info("Starting incident debug: ns=%s pod=%s dry_run=%s auto_fix=%s",
                namespace, pod_name, dry_run, auto_fix)

    t_start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = round(time.time() - t_start, 2)

    logger.info("Incident debug complete: failure=%s severity=%s elapsed=%.1fs",
                final_state.get("failure_type"), final_state.get("severity"), elapsed)

    return dict(final_state)
