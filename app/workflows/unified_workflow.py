"""
Unified Incident Workflow — single LangGraph graph for ALL incident types.

Replaces both:
  - incident_workflow.py   (K8s-only LangGraph)
  - incident_pipeline.py   (sequential AWS+K8s+GitHub pipeline)

Graph:
  planner → gather_all → debugger → executor? → reporter → END

Input determines what gets collected and executed:
  - pod_name set  → detailed K8s pod debug (logs, events, describe)
  - aws_cfg set   → AWS context (EC2, CloudWatch, alarms)
  - description   → general incident (all sources collected)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

logger = logging.getLogger("nsops.workflow.unified")


# ── State ─────────────────────────────────────────────────────────────────────

class UnifiedState(TypedDict):
    # Input
    incident_id:    str
    description:    str
    severity:       str          # reported: critical/high/medium/low

    # K8s targeting (optional)
    namespace:      str
    pod_name:       str

    # AWS targeting (optional)
    aws_cfg:        dict         # {resource_type, resource_id, log_group}
    hours:          int          # lookback hours

    # Flags
    dry_run:        bool
    auto_fix:       bool         # K8s executor (restart/scale)
    auto_remediate: bool         # notification actions (Jira/Slack/OpsGenie/PR)

    # Gathered data
    k8s_data:       dict
    aws_data:       dict
    github_data:    dict

    # Analysis
    failure_type:   str          # CrashLoopBackOff | OOMKilled | General | ...
    root_cause:     str
    summary:        str
    findings:       list
    fix_suggestion: str
    severity_ai:    str          # AI-assessed severity
    confidence:     float
    actions_to_take: list        # [{type, params, reason}]

    # Execution
    actions_taken:  list
    fix_executed:   bool
    fix_result:     dict

    # Observability
    steps_taken:    list
    step_timings:   dict
    errors:         list
    memory_hint:    str

    # Output
    report:         str
    success:        bool


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(system: str, user: str, max_tokens: int = 1024) -> str:
    try:
        from app.chat.intelligence import _llm_call
        return _llm_call(user, system=system, max_tokens=max_tokens, temperature=0.3)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return f"[LLM unavailable: {e}]"


# ── Memory helpers ────────────────────────────────────────────────────────────

def _recall(description: str) -> str:
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


def _store(description: str, resolution: str) -> None:
    try:
        from app.memory.vector_db import store_incident
        store_incident(description, resolution)
    except Exception:
        pass


# ── Data collectors ───────────────────────────────────────────────────────────

def _collect_k8s(namespace: str, pod_name: str) -> dict:
    """Fetch K8s data. If pod_name given, does deep pod debug. Otherwise cluster overview."""
    try:
        from app.tools.kubernetes import KubernetesTool
        k8s = KubernetesTool()

        if pod_name:
            pod_r   = k8s.describe_pod(namespace=namespace, pod=pod_name)
            logs_r  = k8s.get_logs(namespace=namespace, pod=pod_name, tail_lines=200)
            events_r = k8s.get_events(namespace=namespace, limit=30)
            usage_r = k8s.get_resource_usage(namespace=namespace)

            all_events = events_r.get("data") or [] if events_r.get("success") else []
            pod_events = [e for e in all_events
                          if pod_name in str(e.get("involved_object", "")) or
                             pod_name in str(e.get("name", ""))]

            return {
                "_data_available": True,
                "_mode": "pod_debug",
                "namespace": namespace,
                "pod_name": pod_name,
                "pod_details": pod_r.get("data") or {} if pod_r.get("success") else {},
                "logs": logs_r.get("data") or "" if logs_r.get("success") else "",
                "events": pod_events or all_events[:10],
                "resource_usage": usage_r.get("data") or {} if usage_r.get("success") else {},
                "errors": [
                    e for e in [
                        pod_r.get("error") if not pod_r.get("success") else None,
                        logs_r.get("error") if not logs_r.get("success") else None,
                    ] if e
                ],
            }
        else:
            # Cluster overview
            pods_r  = k8s.get_pods(namespace=namespace)
            unhealthy_r = k8s.get_unhealthy_pods(namespace=namespace)
            return {
                "_data_available": pods_r.get("success", False),
                "_mode": "cluster_overview",
                "namespace": namespace,
                "pods": pods_r.get("data") or [],
                "unhealthy_pods": unhealthy_r.get("data") or [],
            }
    except Exception as e:
        return {"_data_available": False, "_reason": str(e)}


def _collect_aws(aws_cfg: dict, hours: int) -> dict:
    if not aws_cfg or not aws_cfg.get("resource_type"):
        # Try basic EC2 + alarms even without explicit config
        try:
            from app.tools.aws import AWSTool
            aws = AWSTool()
            ec2_r    = aws.list_ec2()
            alarms_r = aws.get_alarms()
            return {
                "_data_available": True,
                "_mode": "overview",
                "ec2_instances": ec2_r.get("data") or [] if ec2_r.get("success") else [],
                "alarms": alarms_r.get("data") or [] if alarms_r.get("success") else [],
            }
        except Exception as e:
            return {"_data_available": False, "_reason": str(e)}
    try:
        from app.agents.incident_pipeline import _collect_aws as _pipeline_aws
        result = _pipeline_aws(aws_cfg, hours)
        return result
    except Exception as e:
        return {"_data_available": False, "_reason": str(e)}


def _collect_github(hours: int) -> dict:
    try:
        from app.agents.incident_pipeline import _collect_github as _pipeline_github
        return _pipeline_github(hours)
    except Exception as e:
        return {"_data_available": False, "_reason": str(e)}


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def planner_node(state: UnifiedState) -> dict:
    t0 = time.time()
    steps  = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))

    incident_id = state.get("incident_id") or f"INC-{uuid.uuid4().hex[:8].upper()}"
    description = state.get("description") or ""
    pod_name    = state.get("pod_name") or ""
    namespace   = state.get("namespace") or "default"

    steps.append(f"planner: starting incident {incident_id}")
    logger.info("[PLANNER] incident=%s pod=%s ns=%s", incident_id, pod_name, namespace)

    # Build search query from available context
    search_query = description or f"pod {pod_name} ns {namespace} failure"
    memory_hint = _recall(search_query)
    if memory_hint:
        steps.append("planner: found similar past incidents in memory")

    timings = dict(state.get("step_timings", {}))
    timings["planner"] = round(time.time() - t0, 2)

    return {
        "incident_id": incident_id,
        "namespace":   namespace,
        "steps_taken": steps,
        "step_timings": timings,
        "errors":      errors,
        "memory_hint": memory_hint,
    }


# ── Node 2: Gather All ────────────────────────────────────────────────────────

def gather_all_node(state: UnifiedState) -> dict:
    """Collect K8s + AWS + GitHub data in parallel using ThreadPoolExecutor."""
    t0 = time.time()
    steps  = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))

    namespace = state.get("namespace") or "default"
    pod_name  = state.get("pod_name") or ""
    aws_cfg   = state.get("aws_cfg") or {}
    hours     = state.get("hours") or 2

    steps.append("gather: collecting K8s + AWS + GitHub data in parallel")
    logger.info("[GATHER] ns=%s pod=%s aws=%s", namespace, pod_name, bool(aws_cfg))

    k8s_data = aws_data = github_data = {}

    def _k8s():    return _collect_k8s(namespace, pod_name)
    def _aws():    return _collect_aws(aws_cfg, hours)
    def _github(): return _collect_github(hours)

    pool = ThreadPoolExecutor(max_workers=3)
    futures = {
        pool.submit(_k8s):    "k8s",
        pool.submit(_aws):    "aws",
        pool.submit(_github): "github",
    }
    try:
        for fut in as_completed(futures, timeout=20):
            name = futures[fut]
            try:
                result = fut.result()
                if name == "k8s":
                    k8s_data = result
                    steps.append(f"gather: K8s {'ok' if result.get('_data_available') else 'unavailable'} ({result.get('_mode','?')})")
                    if result.get("errors"):
                        errors.extend(result["errors"])
                elif name == "aws":
                    aws_data = result
                    steps.append(f"gather: AWS {'ok' if result.get('_data_available') else 'unavailable'}")
                elif name == "github":
                    github_data = result
                    steps.append(f"gather: GitHub {'ok' if result.get('_data_available') else 'unavailable'}")
            except Exception as e:
                errors.append(f"gather_{name}: {e}")
    except TimeoutError:
        errors.append("gather: timeout after 20s — using partial data")
    finally:
        pool.shutdown(wait=False)

    timings = dict(state.get("step_timings", {}))
    timings["gather_all"] = round(time.time() - t0, 2)

    return {
        "k8s_data":    k8s_data,
        "aws_data":    aws_data,
        "github_data": github_data,
        "steps_taken": steps,
        "step_timings": timings,
        "errors":      errors,
    }


# ── Node 3: Debugger ──────────────────────────────────────────────────────────

def debugger_node(state: UnifiedState) -> dict:
    """LLM analyses all gathered data. Returns root_cause, findings, actions_to_take."""
    t0 = time.time()
    steps  = list(state.get("steps_taken", []))
    errors = list(state.get("errors", []))

    steps.append("debugger: analysing all data with LLM")
    logger.info("[DEBUGGER] starting analysis")

    k8s_data    = state.get("k8s_data", {})
    aws_data    = state.get("aws_data", {})
    github_data = state.get("github_data", {})
    memory_hint = state.get("memory_hint", "")
    description = state.get("description", "")
    pod_name    = state.get("pod_name", "")
    namespace   = state.get("namespace", "")

    # Build context for LLM — include whatever is available
    sections = []

    if description:
        sections.append(f"=== Incident Description ===\n{description}")

    if k8s_data.get("_data_available"):
        if k8s_data.get("_mode") == "pod_debug":
            logs = k8s_data.get("logs", "")
            logs_snippet = logs[-3000:] if len(logs) > 3000 else logs
            sections.append(f"""=== Kubernetes Pod: {pod_name} / {namespace} ===
Pod Details: {json.dumps(k8s_data.get('pod_details', {}), indent=2, default=str)[:800]}
Logs (last 200 lines):
{logs_snippet or 'No logs available'}
Events:
{json.dumps(k8s_data.get('events', [])[:10], indent=2, default=str)[:600]}""")
        else:
            unhealthy = k8s_data.get("unhealthy_pods", [])
            sections.append(f"=== Kubernetes Cluster ({namespace}) ===\nUnhealthy pods: {json.dumps(unhealthy[:5], default=str)[:400]}")

    if aws_data.get("_data_available"):
        ec2 = aws_data.get("ec2_instances", [])
        alarms = aws_data.get("alarms", [])
        sections.append(f"=== AWS ===\nEC2 ({len(ec2)} instances): {json.dumps(ec2[:3], default=str)[:400]}\nAlarms: {json.dumps(alarms[:5], default=str)[:400]}")

    if github_data.get("_data_available"):
        commits = github_data.get("recent_commits", {})
        sections.append(f"=== GitHub Recent Commits ===\n{json.dumps(commits, default=str)[:400]}")

    if memory_hint:
        sections.append(f"=== Memory: Similar Past Incidents ===\n{memory_hint}")

    context = "\n\n".join(sections) if sections else "No observability data available."

    system = """You are an expert SRE debugging an incident.

Analyse the provided context and respond ONLY with valid JSON:
{
  "failure_type": "<CrashLoopBackOff|OOMKilled|ImagePullBackOff|Pending|HighCPU|HighMemory|NetworkError|DeploymentFailed|General|Unknown>",
  "severity": "<critical|high|medium|low>",
  "summary": "<1 sentence summary>",
  "root_cause": "<2-3 sentence root cause explanation>",
  "fix_suggestion": "<specific actionable fix — include commands if relevant>",
  "confidence": <0.0-1.0>,
  "findings": ["<finding 1>", "<finding 2>"],
  "actions_to_take": [
    {"type": "<k8s_restart|k8s_scale|github_pr|jira_ticket|slack_warroom|opsgenie_alert|none>",
     "params": {},
     "reason": "<why>"}
  ],
  "can_auto_fix": <true|false>
}

Rules:
- can_auto_fix=true ONLY for CrashLoopBackOff or simple restarts
- actions_to_take should list ALL recommended actions (K8s + notifications)
- Be specific — use real values from the data
- If no data available, base analysis on description only"""

    raw = _llm(system, context, max_tokens=800)

    # Defaults
    failure_type   = "General" if not pod_name else "Unknown"
    severity_ai    = state.get("severity") or "medium"
    summary        = description[:100] if description else "Incident analysis"
    root_cause     = "Insufficient data to determine root cause."
    fix_suggestion = "Review logs and events manually."
    confidence     = 0.5
    findings       = []
    actions_to_take = []
    can_auto_fix   = False

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        failure_type    = parsed.get("failure_type", failure_type)
        severity_ai     = parsed.get("severity", severity_ai)
        summary         = parsed.get("summary", summary)
        root_cause      = parsed.get("root_cause", root_cause)
        fix_suggestion  = parsed.get("fix_suggestion", fix_suggestion)
        confidence      = float(parsed.get("confidence", confidence))
        findings        = parsed.get("findings", [])
        actions_to_take = parsed.get("actions_to_take", [])
        can_auto_fix    = parsed.get("can_auto_fix", False)
    except Exception as e:
        errors.append(f"debugger_json_parse: {e} — raw[:100]: {raw[:100]}")
        lower = raw.lower()
        if "crashloopbackoff" in lower:    failure_type = "CrashLoopBackOff"
        elif "oomkilled" in lower:         failure_type = "OOMKilled"
        elif "imagepull" in lower:         failure_type = "ImagePullBackOff"
        root_cause = raw[:500] if raw else root_cause

    # Only allow auto_fix if LLM says it's safe
    auto_fix = state.get("auto_fix", False) and can_auto_fix

    steps.append(f"debugger: failure_type={failure_type} severity={severity_ai} confidence={confidence:.1f}")
    logger.info("[DEBUGGER] failure=%s severity=%s actions=%d", failure_type, severity_ai, len(actions_to_take))

    timings = dict(state.get("step_timings", {}))
    timings["debugger"] = round(time.time() - t0, 2)

    return {
        "failure_type":    failure_type,
        "severity_ai":     severity_ai,
        "summary":         summary,
        "root_cause":      root_cause,
        "fix_suggestion":  fix_suggestion,
        "confidence":      confidence,
        "findings":        findings,
        "actions_to_take": actions_to_take,
        "auto_fix":        auto_fix,
        "steps_taken":     steps,
        "step_timings":    timings,
        "errors":          errors,
    }


# ── Node 4: Executor ──────────────────────────────────────────────────────────

def executor_node(state: UnifiedState) -> dict:
    """
    Executes ALL recommended actions:
    - K8s: restart_pod / scale (requires auto_fix=True)
    - Notifications: Jira, Slack, OpsGenie, GitHub PR (requires auto_remediate=True)
    Respects dry_run — logs what it would do without executing.
    """
    t0 = time.time()
    steps       = list(state.get("steps_taken", []))
    errors      = list(state.get("errors", []))
    actions_taken = list(state.get("actions_taken", []))

    dry_run        = state.get("dry_run", True)
    auto_fix       = state.get("auto_fix", False)
    auto_remediate = state.get("auto_remediate", False)
    failure_type   = state.get("failure_type", "General")
    namespace      = state.get("namespace", "default")
    pod_name       = state.get("pod_name", "")
    incident_id    = state.get("incident_id", "INC-unknown")
    actions_to_take = state.get("actions_to_take", [])
    severity_ai    = state.get("severity_ai", "medium")

    fix_executed = False
    fix_result   = {}

    steps.append(f"executor: processing {len(actions_to_take)} recommended actions")
    logger.info("[EXECUTOR] dry_run=%s auto_fix=%s auto_remediate=%s", dry_run, auto_fix, auto_remediate)

    # ── K8s remediation ───────────────────────────────────────────────────────
    if pod_name and failure_type not in ("ImagePullBackOff", "General", "Unknown"):
        k8s_action = "restart_pod"
        if failure_type == "ImagePullBackOff":
            steps.append("executor: ImagePullBackOff — skipping K8s auto-fix (image issue)")
        elif not auto_fix:
            steps.append(f"executor: K8s fix available but auto_fix=false — set auto_fix=true to enable")
            fix_result = {"skipped": True, "reason": "auto_fix=false"}
        elif dry_run:
            steps.append(f"executor: DRY-RUN — would {k8s_action} pod={pod_name} ns={namespace}")
            fix_result = {"dry_run": True, "action": k8s_action, "message": f"[DRY-RUN] Would {k8s_action} {pod_name} in {namespace}"}
            fix_executed = False
        else:
            try:
                from app.tools.kubernetes import KubernetesTool
                k8s = KubernetesTool()
                result = k8s.restart_pod(namespace=namespace, pod=pod_name)
                fix_result = {"executed": True, "action": k8s_action, **result}
                fix_executed = result.get("success", False)
                steps.append(f"executor: K8s {k8s_action} — {'ok' if fix_executed else 'failed'}")
                if not fix_executed:
                    errors.append(f"k8s_{k8s_action}: {result.get('error', 'unknown')}")
            except Exception as e:
                errors.append(f"executor_k8s: {e}")
                fix_result = {"executed": False, "error": str(e)}

    # ── Notification / integration actions ────────────────────────────────────
    for action in actions_to_take:
        action_type = action.get("type", "none")
        params      = action.get("params", {})
        reason      = action.get("reason", "")

        if action_type in ("none", "k8s_restart", "k8s_scale"):
            # k8s handled above; none = skip
            continue

        if not auto_remediate:
            actions_taken.append({"type": action_type, "skipped": True, "reason": "auto_remediate=false"})
            steps.append(f"executor: skipped {action_type} (auto_remediate=false)")
            continue

        if dry_run:
            actions_taken.append({"type": action_type, "dry_run": True, "reason": reason})
            steps.append(f"executor: DRY-RUN — would execute {action_type}")
            continue

        try:
            if action_type == "jira_ticket":
                from app.integrations.jira import create_incident
                res = create_incident(
                    summary=params.get("title", f"[{incident_id}] {state.get('summary', '')}"),
                    description=params.get("body", state.get("root_cause", "")),
                )
                actions_taken.append({"type": "jira_ticket", "result": res})
                steps.append(f"executor: Jira ticket {'created' if res.get('success') else 'failed'}")

            elif action_type == "slack_warroom":
                from app.integrations.slack import create_war_room
                res = create_war_room(topic=f"Incident {incident_id} | {severity_ai.upper()} | {state.get('summary', '')}")
                actions_taken.append({"type": "slack_warroom", "result": res})
                steps.append(f"executor: Slack war room {'created' if res.get('success') else 'failed'}")

            elif action_type == "opsgenie_alert":
                from app.integrations.opsgenie import notify_on_call
                res = notify_on_call(
                    message=params.get("message", f"Incident {incident_id}: {state.get('summary', '')}"),
                    alias=f"incident-{incident_id}",
                )
                actions_taken.append({"type": "opsgenie_alert", "result": res})
                steps.append(f"executor: OpsGenie alert {'sent' if res.get('success') else 'failed'}")

            elif action_type == "github_pr":
                from app.integrations.github import create_incident_pr
                res = create_incident_pr(
                    incident_id=incident_id,
                    title=params.get("title", f"Incident {incident_id} — AI suggested fix"),
                    body=state.get("root_cause", "") + f"\n\n*Auto-generated by NsOps for incident `{incident_id}`*",
                    file_changes=params.get("file_patches"),
                )
                actions_taken.append({"type": "github_pr", "result": res})
                steps.append(f"executor: GitHub PR {'created' if res.get('success') else 'failed'}")

        except Exception as e:
            errors.append(f"executor_{action_type}: {e}")
            actions_taken.append({"type": action_type, "error": str(e)})

    timings = dict(state.get("step_timings", {}))
    timings["executor"] = round(time.time() - t0, 2)

    return {
        "fix_executed":  fix_executed,
        "fix_result":    fix_result,
        "actions_taken": actions_taken,
        "steps_taken":   steps,
        "step_timings":  timings,
        "errors":        errors,
    }


# ── Node 5: Reporter ──────────────────────────────────────────────────────────

def reporter_node(state: UnifiedState) -> dict:
    t0 = time.time()
    steps = list(state.get("steps_taken", []))
    steps.append("reporter: generating report")

    incident_id    = state.get("incident_id", "INC-?")
    pod_name       = state.get("pod_name", "")
    namespace      = state.get("namespace", "")
    failure_type   = state.get("failure_type", "General")
    severity_ai    = state.get("severity_ai", "unknown")
    root_cause     = state.get("root_cause", "N/A")
    fix_suggestion = state.get("fix_suggestion", "N/A")
    summary        = state.get("summary", "")
    findings       = state.get("findings", [])
    fix_result     = state.get("fix_result", {})
    actions_taken  = state.get("actions_taken", [])
    errors         = state.get("errors", [])
    timings        = state.get("step_timings", {})

    sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity_ai, "⚪")
    target = f"`{pod_name}` in `{namespace}`" if pod_name else f"Incident `{incident_id}`"

    fix_status = ""
    if fix_result.get("dry_run"):
        fix_status = f"\n**Fix (DRY-RUN):** {fix_result.get('message', 'Would execute fix')}"
    elif fix_result.get("executed"):
        ok = fix_result.get("success", False)
        fix_status = f"\n**Fix Executed:** {'✅ Success' if ok else '❌ Failed'} — {fix_result.get('action', '')}"
    elif fix_result.get("skipped"):
        fix_status = f"\n**Fix:** Skipped — {fix_result.get('reason', '')}"
    else:
        fix_status = "\n**Fix:** Not executed (enable auto_fix or auto_remediate)"

    findings_md = ""
    if findings:
        findings_md = "\n### Findings\n" + "\n".join(f"- {f}" for f in findings)

    actions_md = ""
    if actions_taken:
        lines = []
        for a in actions_taken:
            if a.get("skipped"):
                lines.append(f"- ⏭ `{a['type']}` — skipped ({a.get('reason', '')})")
            elif a.get("dry_run"):
                lines.append(f"- 🔵 `{a['type']}` — dry-run ({a.get('reason', '')})")
            elif a.get("error"):
                lines.append(f"- ❌ `{a['type']}` — {a['error']}")
            else:
                ok = (a.get("result") or {}).get("success", False)
                lines.append(f"- {'✅' if ok else '❌'} `{a['type']}`")
        actions_md = "\n### Actions\n" + "\n".join(lines)

    errors_md = ""
    if errors:
        errors_md = f"\n**Warnings:** {'; '.join(errors[:3])}"

    total_time = sum(timings.values())

    report = f"""## {sev_emoji} Incident Report — {incident_id}

**Target:** {target}
**Failure Type:** `{failure_type}` | **Severity:** {severity_ai.upper()}
{f'**Summary:** {summary}' if summary else ''}

### Root Cause
{root_cause}

### Suggested Fix
{fix_suggestion}
{fix_status}
{findings_md}
{actions_md}
{errors_md}

### Steps Taken
{chr(10).join(f'  {i+1}. {s}' for i, s in enumerate(steps))}

### Timing
{chr(10).join(f'  - {k}: {v}s' for k, v in timings.items())}
Total: {total_time:.1f}s
""".strip()

    # Store in memory
    _store(
        description=f"{failure_type} — {root_cause[:150]}",
        resolution=fix_suggestion[:300],
    )

    # Slack alert for critical
    if severity_ai == "critical":
        try:
            from app.integrations.slack import send_slack_message
            send_slack_message(
                f"{sev_emoji} *Critical Incident — {incident_id}*\n"
                f"Type: `{failure_type}` | Target: {target}\n"
                f"Root cause: {root_cause[:200]}"
            )
            steps.append("reporter: Slack critical alert sent")
        except Exception:
            pass

    timings["reporter"] = round(time.time() - t0, 2)
    logger.info("[REPORTER] done len=%d", len(report))

    return {
        "report":      report,
        "success":     True,
        "steps_taken": steps,
        "step_timings": timings,
    }


# ── Conditional routing ───────────────────────────────────────────────────────

def _should_execute(state: UnifiedState) -> str:
    """Run executor if auto_fix or auto_remediate is set (even in dry-run — shows what would happen)."""
    if state.get("auto_fix") or state.get("auto_remediate") or state.get("dry_run"):
        return "executor"
    return "reporter"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_unified_graph() -> Any:
    graph = StateGraph(UnifiedState)

    graph.add_node("planner",    planner_node)
    graph.add_node("gather_all", gather_all_node)
    graph.add_node("debugger",   debugger_node)
    graph.add_node("executor",   executor_node)
    graph.add_node("reporter",   reporter_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "gather_all")
    graph.add_edge("gather_all", "debugger")
    graph.add_conditional_edges("debugger", _should_execute, {
        "executor": "executor",
        "reporter": "reporter",
    })
    graph.add_edge("executor", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


_graph = None

def get_unified_graph():
    global _graph
    if _graph is None:
        _graph = build_unified_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_unified(
    incident_id:    str  = None,
    description:    str  = "",
    severity:       str  = "medium",
    namespace:      str  = "default",
    pod_name:       str  = "",
    aws_cfg:        dict = None,
    hours:          int  = 2,
    dry_run:        bool = True,
    auto_fix:       bool = False,
    auto_remediate: bool = False,
) -> dict:
    """
    Single entry point for ALL incident types.

    K8s pod debug:    run_unified(namespace="prod", pod_name="api-abc123", dry_run=True)
    General incident: run_unified(description="High CPU on payment service", severity="high")
    Full pipeline:    run_unified(description="...", auto_remediate=True, aws_cfg={...})
    """
    initial: UnifiedState = {
        "incident_id":    incident_id or f"INC-{uuid.uuid4().hex[:8].upper()}",
        "description":    description,
        "severity":       severity,
        "namespace":      namespace,
        "pod_name":       pod_name,
        "aws_cfg":        aws_cfg or {},
        "hours":          hours,
        "dry_run":        dry_run,
        "auto_fix":       auto_fix,
        "auto_remediate": auto_remediate,
        "k8s_data":       {},
        "aws_data":       {},
        "github_data":    {},
        "failure_type":   "",
        "root_cause":     "",
        "summary":        "",
        "findings":       [],
        "fix_suggestion": "",
        "severity_ai":    severity,
        "confidence":     0.5,
        "actions_to_take": [],
        "actions_taken":  [],
        "fix_executed":   False,
        "fix_result":     {},
        "steps_taken":    [],
        "step_timings":   {},
        "errors":         [],
        "memory_hint":    "",
        "report":         "",
        "success":        False,
    }

    t_start = time.time()
    graph = get_unified_graph()
    logger.info("run_unified: incident=%s pod=%s description=%s",
                initial["incident_id"], pod_name, description[:60])

    final = graph.invoke(initial)
    elapsed = round(time.time() - t_start, 2)
    logger.info("run_unified done: failure=%s severity=%s elapsed=%.1fs",
                final.get("failure_type"), final.get("severity_ai"), elapsed)

    return dict(final)
