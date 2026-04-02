"""PlannerAgent — generates a structured, executable remediation plan via LLM."""
from __future__ import annotations

import json
import re

from app.agents.base import BaseAgent
from app.llm.factory import LLMFactory, mark_rate_limited
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are an expert SRE. You produce lean, precise incident remediation plans. "
    "RULES:\n"
    "1. Only recommend actions for infrastructure that is marked AVAILABLE in the data summary. "
    "If Kubernetes is NOT AVAILABLE, output zero k8s_* actions. "
    "If AWS is NOT AVAILABLE, output zero aws_* actions.\n"
    "2. Use ONLY real resource identifiers (IDs, names, IPs) from the context. "
    "Never invent example values like i-0abc123, api-server, production, etc.\n"
    "3. Fewer, precise actions beat many vague ones. Only include actions with clear evidence.\n"
    "4. If the symptom is 'EC2 instance stopped/not running' and you have the instance list, "
    "recommend investigating that specific instance and restarting it — nothing else.\n"
    "5. Set confidence below 0.5 when key data is missing.\n"
    "6. Similar past incidents are BACKGROUND ONLY. NEVER mention them in root_cause or summary. "
    "Root cause must be based purely on the AWS/K8s/GitHub data in this incident."
)

_PROMPT_TEMPLATE = """
=== DATA AVAILABILITY ===
AWS data:       {aws_available}
Kubernetes:     {k8s_available}
GitHub:         {github_available}

=== INCIDENT ===
ID:          {incident_id}
Description: {description}

=== AWS CONTEXT ===
{aws_context}

=== KUBERNETES CONTEXT ===
{k8s_context}

=== GITHUB CONTEXT ===
{github_context}

=== SIMILAR PAST INCIDENTS (background context only) ===
{similar_incidents}
NOTE: Similar incidents are for pattern recognition only.
NEVER cite them as the root cause of THIS incident.
The root cause must come solely from the AWS/K8s/GitHub context above.

=== TASK ===
Look at the DATA AVAILABILITY section first.
- If a source is NOT AVAILABLE → include ZERO actions for that infrastructure type.
- Only reference real values from the context above — no placeholders.
- For each action write a clear "description" of exactly what to do and why.

Return ONLY valid JSON (no markdown fences):
{{
  "actions": [
    {{
      "type": "investigate",
      "description": "SSH into instance i-0e2f6f0b7cb4446fb and check system logs to determine why it stopped",
      "target": "EC2 i-0e2f6f0b7cb4446fb",
      "estimated_cost_delta": 0
    }},
    {{
      "type": "aws_restart",
      "description": "Start the stopped EC2 instance i-0e2f6f0b7cb4446fb",
      "instance_id": "i-0e2f6f0b7cb4446fb",
      "estimated_cost_delta": 0
    }},
    {{
      "type": "slack_notify",
      "channel": "incidents",
      "description": "Notify on-call team that EC2 i-0e2f6f0b7cb4446fb is down and being investigated",
      "message": "EC2 instance i-0e2f6f0b7cb4446fb is in stopped state. Investigating root cause."
    }}
  ],
  "confidence": 0.75,
  "risk": "medium",
  "root_cause": "Specific description using ONLY real data from the context. If unknown, say what IS known and what is missing.",
  "summary": "One sentence summary with real resource names from the context.",
  "reasoning": "Step-by-step: what data points led to what conclusion.",
  "data_gaps": ["Only list genuinely missing data types, not specific fabricated resource names"]
}}

Allowed risk levels: low | medium | high | critical
Allowed action types (use ONLY for available infrastructure):
  AWS available    → investigate, aws_restart, aws_scale, slack_notify, create_jira, opsgenie_alert, create_pr
  K8s available    → investigate, k8s_restart, k8s_scale, slack_notify, create_jira, opsgenie_alert, create_pr
  Neither          → investigate (generic steps only), slack_notify, create_jira, opsgenie_alert
"""

# Fabricated placeholder patterns to detect and strip
_FAKE_PATTERNS = re.compile(
    r'i-0abc123|api-server|<[^>]+>|your-namespace|production namespace|'
    r'your-deployment|example\.|placeholder',
    re.IGNORECASE
)


def _data_summary(ctx: dict) -> str:
    if not ctx or ctx.get("_data_available") is False:
        reason = ctx.get("_reason", "not configured") if ctx else "not configured"
        return f"NOT AVAILABLE ({reason})"
    return "AVAILABLE"


def _clean_actions(actions: list, aws_ok: bool, k8s_ok: bool) -> list:
    """Hard-remove actions for unavailable infrastructure."""
    cleaned = []
    for a in actions:
        atype = (a.get("type") or "").lower()
        # Drop K8s actions if K8s not available
        if atype in ("k8s_restart", "k8s_scale") and not k8s_ok:
            continue
        # Drop AWS-specific actions if AWS not available
        if atype in ("aws_restart", "aws_scale") and not aws_ok:
            continue
        cleaned.append(a)
    return cleaned


def _strip_fabricated(plan: dict) -> dict:
    """Remove obviously fabricated identifiers from data_gaps."""
    gaps = plan.get("data_gaps", [])
    if gaps:
        cleaned = [g for g in gaps if not _FAKE_PATTERNS.search(g)]
        plan["data_gaps"] = cleaned
    return plan


class PlannerAgent(BaseAgent):
    """Uses LLM to generate a structured JSON execution plan."""

    def run(self, state: dict) -> dict:
        aws_ctx    = state.get("aws_context", {})
        k8s_ctx    = state.get("k8s_context", {})
        github_ctx = state.get("github_context", {})

        aws_ok    = bool(aws_ctx and aws_ctx.get("_data_available"))
        k8s_ok    = bool(k8s_ctx and k8s_ctx.get("_data_available"))
        github_ok = bool(github_ctx and github_ctx.get("_data_available"))

        # Early exit: if ALL integrations are unavailable, skip the LLM call
        if not aws_ok and not k8s_ok and not github_ok:
            self._warn(
                "planner_all_integrations_unavailable",
                incident_id=state.get("incident_id"),
            )
            state["plan"] = {
                "actions": [],
                "confidence": 0.0,
                "risk": "unknown",
                "root_cause": (
                    "No infrastructure data available — AWS, Kubernetes, and GitHub "
                    "integrations are not configured or unreachable. "
                    "Please add credentials in the Secrets panel and restart."
                ),
                "summary": "All integrations unavailable — cannot analyse incident without data.",
                "data_gaps": [
                    "AWS context unavailable — provide aws_cfg or check AWS credentials",
                    "Kubernetes context unavailable — kubectl not configured",
                    "GitHub context unavailable — check GITHUB_TOKEN",
                ],
            }
            return state

        prompt = _PROMPT_TEMPLATE.format(
            incident_id       = state.get("incident_id", "unknown"),
            description       = state.get("description", ""),
            aws_available     = _data_summary(aws_ctx),
            k8s_available     = _data_summary(k8s_ctx),
            github_available  = _data_summary(github_ctx),
            aws_context       = json.dumps(aws_ctx, default=str)[:3000],
            k8s_context       = json.dumps(k8s_ctx, default=str)[:3000],
            github_context    = json.dumps(github_ctx, default=str)[:2000],
            similar_incidents = json.dumps(state.get("similar_incidents", []), default=str)[:1500],
        )

        last_exc = None
        for attempt in range(4):
            try:
                llm = LLMFactory.get()
            except RuntimeError as exc:
                last_exc = exc
                break
            try:
                response = llm.complete(prompt, system=_SYSTEM, max_tokens=2500)
                plan = self._parse_json(response.content)
                if not plan:
                    raise ValueError("Empty plan returned by LLM")

                # Hard-filter: remove actions for unavailable infrastructure
                plan["actions"] = _clean_actions(plan.get("actions", []), aws_ok, k8s_ok)

                # Strip fabricated identifiers from data_gaps
                plan = _strip_fabricated(plan)

                # Cap confidence when no real infra data
                if not aws_ok and not k8s_ok:
                    plan["confidence"] = min(plan.get("confidence", 0.3), 0.4)
                    if not plan.get("data_gaps"):
                        gaps = []
                        if not aws_ok:
                            gaps.append("AWS context unavailable — provide aws_cfg or check AWS credentials")
                        if not k8s_ok:
                            gaps.append("Kubernetes context unavailable — kubectl not configured")
                        if not github_ok:
                            gaps.append("GitHub context unavailable — check GITHUB_TOKEN")
                        plan["data_gaps"] = gaps

                state["plan"] = plan
                self._log(
                    "plan_generated",
                    incident_id=state.get("incident_id"),
                    actions=len(plan.get("actions", [])),
                    risk=plan.get("risk"),
                    confidence=plan.get("confidence"),
                    provider=response.provider,
                )
                return state
            except Exception as exc:
                err_str = str(exc)
                last_exc = exc
                if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower():
                    provider_key = getattr(llm, "_force_provider", None) or "groq"
                    mark_rate_limited(str(provider_key) if provider_key else "groq", err_str)
                    self._warn("planner_rate_limited_retrying", provider=str(provider_key), attempt=attempt + 1)
                    continue
                break

        self._warn("planner_agent_failed", error=str(last_exc))
        state.setdefault("errors", []).append(f"PlannerAgent: {last_exc}")
        state["plan"] = {"actions": [], "confidence": 0.0, "risk": "unknown",
                         "root_cause": "Planning failed", "summary": str(last_exc)}
        return state
