"""
Debugger Agent — analyzes logs, K8s events, and metrics to identify root cause.
Can be used standalone or as a node in a LangGraph workflow.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

logger = logging.getLogger("nsops.agent.debugger")

# Known K8s failure signatures for fast heuristic detection
_FAILURE_SIGNATURES = {
    "CrashLoopBackOff":  ["crashloopbackoff", "back-off restarting", "restart count"],
    "OOMKilled":         ["oomkilled", "out of memory", "memory limit exceeded", "killed process"],
    "ImagePullBackOff":  ["imagepullbackoff", "errimagepull", "failed to pull image", "image not found"],
    "Pending":           ["insufficient cpu", "insufficient memory", "0/1 nodes available", "unschedulable"],
    "ContainerCreating": ["containercreatring", "containercreating"],
    "Error":             ["error", "failed", "exception", "traceback"],
}


class DebuggerAgent:
    """
    Analyzes K8s pod failures, AWS errors, or pipeline failures.
    Uses heuristic detection + LLM reasoning for root cause analysis.
    """

    def analyze_pod_failure(
        self,
        pod_name: str,
        namespace: str,
        logs: str,
        events: list[dict],
        pod_details: dict,
        memory_hint: str = "",
    ) -> dict:
        """
        Full root cause analysis for a pod failure.
        Returns structured diagnosis.
        """
        from app.chat.intelligence import _llm_call

        # Fast heuristic: detect failure type from logs + events
        failure_type = self._detect_failure_type(logs, events, pod_details)
        logger.info("[DEBUGGER] heuristic failure_type=%s pod=%s", failure_type, pod_name)

        # Build rich context for LLM
        events_text = json.dumps(events[:10], indent=2, default=str) if events else "None"
        logs_snippet = logs[-2500:] if len(logs) > 2500 else logs
        details_text = json.dumps(
            {k: v for k, v in pod_details.items() if k not in ("_memory_hint",)},
            indent=2, default=str
        )[:800]

        system = """You are a Kubernetes SRE expert performing root cause analysis.
Analyze the pod details, logs, and events. Return ONLY valid JSON:
{
  "failure_type": "<type>",
  "severity": "<critical|high|medium|low>",
  "root_cause": "<2-3 sentences>",
  "immediate_fix": "<what to do right now — be specific>",
  "long_term_fix": "<architectural or config change to prevent recurrence>",
  "commands": ["<kubectl command 1>", "<kubectl command 2>"],
  "can_auto_restart": <true|false>
}"""

        user = f"""Pod: {pod_name} | Namespace: {namespace}
Detected failure type (heuristic): {failure_type}

Pod Details:
{details_text}

Recent Logs:
{logs_snippet or 'No logs available'}

Kubernetes Events:
{events_text}

Past Similar Incidents:
{memory_hint or 'None found'}"""

        raw = _llm_call(user, system=system, max_tokens=700, temperature=0.2)

        result = {
            "failure_type": failure_type,
            "severity": "high",
            "root_cause": "Analysis failed",
            "immediate_fix": "Review logs and events manually",
            "long_term_fix": "",
            "commands": [],
            "can_auto_restart": False,
        }

        try:
            clean = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(clean)
            result.update(parsed)
            # Trust heuristic over LLM if it found something specific
            if failure_type != "Unknown" and result["failure_type"] == "Unknown":
                result["failure_type"] = failure_type
        except Exception as e:
            logger.warning("[DEBUGGER] JSON parse failed: %s", e)
            result["root_cause"] = raw[:500]

        logger.info("[DEBUGGER] analysis complete: type=%s severity=%s",
                    result["failure_type"], result["severity"])
        return result

    def analyze_log_error(self, log_text: str, service: str = "") -> dict:
        """Analyze arbitrary log text for errors and root cause."""
        from app.chat.intelligence import _llm_call

        system = """Analyze this log output and identify errors.
Return JSON: {"error_type": str, "severity": str, "root_cause": str, "fix": str}"""
        raw = _llm_call(
            f"Service: {service}\n\nLogs:\n{log_text[-2000:]}",
            system=system, max_tokens=400, temperature=0.2
        )
        try:
            return json.loads(raw.strip().strip("```json").strip("```").strip())
        except Exception:
            return {"error_type": "Unknown", "severity": "medium",
                    "root_cause": raw[:300], "fix": "Review logs manually"}

    # ── Heuristics ────────────────────────────────────────────────────────────

    def _detect_failure_type(self, logs: str, events: list, pod_details: dict) -> str:
        """Fast heuristic detection before calling LLM."""
        combined = (logs + " " + json.dumps(events, default=str) +
                    " " + json.dumps(pod_details, default=str)).lower()

        for failure, keywords in _FAILURE_SIGNATURES.items():
            if any(kw in combined for kw in keywords):
                return failure

        # Check pod status field directly
        status = str(pod_details.get("status", "")).lower()
        reason = str(pod_details.get("reason", "")).lower()
        for failure, keywords in _FAILURE_SIGNATURES.items():
            if any(kw in status or kw in reason for kw in keywords):
                return failure

        return "Unknown"
