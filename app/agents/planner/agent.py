"""PlannerAgent — generates a structured, executable remediation plan via LLM."""
from __future__ import annotations

import json

from app.agents.base import BaseAgent
from app.llm.factory import LLMFactory
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = "You are an expert SRE and DevOps automation planner."

_PROMPT_TEMPLATE = """
An incident has been reported. You have been given infrastructure observability
data and a list of similar past incidents from memory. Produce a remediation plan.

=== INCIDENT ===
ID:          {incident_id}
Description: {description}

=== AWS CONTEXT ===
{aws_context}

=== KUBERNETES CONTEXT ===
{k8s_context}

=== GITHUB CONTEXT ===
{github_context}

=== SIMILAR PAST INCIDENTS (from memory) ===
{similar_incidents}

=== INSTRUCTIONS ===
Return ONLY valid JSON — no markdown, no prose:
{{
  "actions": [
    {{"type": "k8s_restart",    "namespace": "...", "deployment": "..."}},
    {{"type": "k8s_scale",      "namespace": "...", "deployment": "...", "replicas": 3}},
    {{"type": "create_jira",    "summary": "...", "description": "..."}},
    {{"type": "slack_notify",   "channel": "...", "message": "..."}},
    {{"type": "create_pr",      "title": "...", "body": "..."}},
    {{"type": "opsgenie_alert", "message": "...", "priority": "P2"}}
  ],
  "confidence": 0.87,
  "risk": "medium",
  "root_cause": "...",
  "summary": "...",
  "reasoning": "..."
}}

Allowed risk levels: low | medium | high | critical
Only include actions that are clearly warranted by the data.
Omit the actions array entry entirely rather than using type "none".
"""


class PlannerAgent(BaseAgent):
    """Uses LLM to generate a structured JSON execution plan."""

    def run(self, state: dict) -> dict:
        llm = LLMFactory.get()
        prompt = _PROMPT_TEMPLATE.format(
            incident_id       = state.get("incident_id", "unknown"),
            description       = state.get("description", ""),
            aws_context       = json.dumps(state.get("aws_context", {}), default=str)[:3000],
            k8s_context       = json.dumps(state.get("k8s_context", {}), default=str)[:3000],
            github_context    = json.dumps(state.get("github_context", {}), default=str)[:2000],
            similar_incidents = json.dumps(state.get("similar_incidents", []), default=str)[:1500],
        )

        try:
            response = llm.complete(prompt, system=_SYSTEM, max_tokens=2000)
            plan = self._parse_json(response.content)
            if not plan:
                raise ValueError("Empty plan returned by LLM")

            state["plan"] = plan
            self._log(
                "plan_generated",
                incident_id=state.get("incident_id"),
                actions=len(plan.get("actions", [])),
                risk=plan.get("risk"),
                confidence=plan.get("confidence"),
                provider=response.provider,
            )
        except Exception as exc:
            self._warn("planner_agent_failed", error=str(exc))
            state.setdefault("errors", []).append(f"PlannerAgent: {exc}")
            state["plan"] = {"actions": [], "confidence": 0.0, "risk": "unknown",
                             "root_cause": "Planning failed", "summary": str(exc)}
        return state
