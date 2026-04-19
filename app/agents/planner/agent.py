"""PlannerAgent — generates a structured, executable remediation plan via LLM."""
from __future__ import annotations

import json
import re

from app.agents.base import BaseAgent
from app.llm.factory import LLMFactory, mark_rate_limited
from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are an expert SRE. You produce lean, precise, EXECUTABLE incident remediation plans. "
    "RULES:\n"
    "1. Only recommend actions for infrastructure that is marked AVAILABLE in the data summary. "
    "If Kubernetes is NOT AVAILABLE, output zero k8s_* actions. "
    "If AWS is NOT AVAILABLE, output zero ec2_*/ecs_*/lambda_*/rds_* actions.\n"
    "2. Use ONLY real resource identifiers (IDs, names) from the context. "
    "Never invent example values like i-0abc123, api-server, production, etc.\n"
    "3. PREFER executable actions over 'investigate'. If you can see a stopped EC2 instance ID, "
    "use ec2_start to actually start it. If an ECS service is down, use ecs_redeploy. "
    "Only use 'investigate' when you genuinely cannot determine the right fix from the data.\n"
    "4. Fewer, precise actions beat many vague ones. Only include actions with clear evidence.\n"
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

=== INCIDENT FOCUS ===
{focus_hint}

=== TASK ===
Look at the DATA AVAILABILITY section AND the INCIDENT FOCUS above.
- If a source is NOT AVAILABLE → include ZERO actions for that infrastructure type.
- If a source is NOT in the INCIDENT FOCUS → include ZERO actions for that infrastructure type.
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
      "type": "ec2_start",
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

Allowed action types — use ONLY types matching available infrastructure:

  EC2 actions (AWS required):
    ec2_start       → {{"type":"ec2_start","instance_id":"i-xxx","description":"..."}}
    ec2_reboot      → {{"type":"ec2_reboot","instance_id":"i-xxx","description":"..."}}
    ec2_stop        → {{"type":"ec2_stop","instance_id":"i-xxx","description":"..."}}

  ECS actions (AWS required):
    ecs_redeploy    → {{"type":"ecs_redeploy","cluster":"my-cluster","service":"my-svc","description":"..."}}
    ecs_scale       → {{"type":"ecs_scale","cluster":"my-cluster","service":"my-svc","desired_count":2,"description":"..."}}

  Lambda actions (AWS required):
    lambda_invoke   → {{"type":"lambda_invoke","function_name":"my-fn","payload":{{}},"description":"..."}}

  RDS actions (AWS required):
    rds_reboot      → {{"type":"rds_reboot","db_instance_id":"my-db","description":"..."}}

  Kubernetes actions (K8s required):
    k8s_restart     → {{"type":"k8s_restart","deployment":"my-dep","namespace":"default","description":"..."}}
    k8s_scale       → {{"type":"k8s_scale","deployment":"my-dep","namespace":"default","replicas":3,"description":"..."}}

  Always available:
    investigate     → {{"type":"investigate","description":"...","target":"..."}} — use ONLY when action type unknown
    slack_notify    → {{"type":"slack_notify","channel":"#incidents","message":"...","description":"..."}}
    slack_warroom   → {{"type":"slack_warroom","incident_id":"...","description":"...","severity":"..."}} — create a war room for coordination
    create_jira     → {{"type":"create_jira","summary":"...","description":"..."}}
    opsgenie_alert  → {{"type":"opsgenie_alert","message":"...","alias":"...","description":"..."}}
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


_AWS_ACTION_TYPES = {"ec2_start", "ec2_stop", "ec2_reboot", "ecs_scale", "ecs_redeploy", "lambda_invoke", "rds_reboot"}
_K8S_ACTION_TYPES = {"k8s_restart", "k8s_scale"}

# Keywords that signal a specific infrastructure focus
_EC2_KEYWORDS  = {"ec2", "instance", "vm", "virtual machine", "server", "host", "stopped", "terminated", "ami"}
_ECS_KEYWORDS  = {"ecs", "container", "task", "fargate", "service", "cluster"}
_LAMBDA_KEYWORDS = {"lambda", "function", "serverless", "invocation", "cold start"}
_RDS_KEYWORDS  = {"rds", "database", "db", "postgres", "mysql", "aurora", "sql"}
_K8S_KEYWORDS  = {"kubernetes", "k8s", "pod", "pods", "deployment", "kubectl", "namespace", "node", "helm", "crashloopbackoff", "oomkilled", "evicted"}


def _classify_incident(description: str) -> dict:
    """Return which infrastructure layers are relevant to this incident description.

    Returns a dict with keys: ec2, ecs, lambda, rds, k8s, general
    All True means we don't have enough signal — show everything.
    """
    desc = description.lower()
    words = set(desc.replace("-", " ").replace("_", " ").split())

    ec2    = bool(words & _EC2_KEYWORDS)
    ecs    = bool(words & _ECS_KEYWORDS)
    lam    = bool(words & _LAMBDA_KEYWORDS)
    rds    = bool(words & _RDS_KEYWORDS)
    k8s    = bool(words & _K8S_KEYWORDS)
    aws    = ec2 or ecs or lam or rds or any(w in desc for w in ("aws", "amazon", "s3", "cloudwatch", "alb", "sns", "sqs"))
    general = not (aws or k8s)   # no signal found → keep everything

    return {"ec2": ec2, "ecs": ecs, "lambda": lam, "rds": rds, "k8s": k8s, "aws": aws, "general": general}


def _focus_hint(classification: dict) -> str:
    """Generate a FOCUS instruction for the LLM prompt based on classification."""
    if classification["general"]:
        return "No specific infrastructure type detected — analyze all available data."

    parts = []
    if classification["ec2"]:
        parts.append("EC2 instances")
    if classification["ecs"]:
        parts.append("ECS services/tasks")
    if classification["lambda"]:
        parts.append("Lambda functions")
    if classification["rds"]:
        parts.append("RDS databases")
    if classification["k8s"]:
        parts.append("Kubernetes workloads")

    focused = ", ".join(parts) if parts else "AWS resources"
    ignore = []
    if not classification["k8s"] and not classification["general"]:
        ignore.append("Kubernetes")
    if not classification["aws"] and not classification["general"]:
        ignore.append("AWS")

    hint = f"FOCUS ONLY on: {focused}."
    if ignore:
        hint += f" IGNORE and output ZERO actions for: {', '.join(ignore)} — it is NOT relevant to this incident even if data is available."
    return hint


def _clean_actions(actions: list, aws_ok: bool, k8s_ok: bool, classification: dict | None = None) -> list:
    """Hard-remove actions for unavailable infrastructure, and optionally for irrelevant infra."""
    cleaned = []
    for a in actions:
        atype = (a.get("type") or "").lower()
        # Remove actions for unavailable infra
        if atype in _K8S_ACTION_TYPES and not k8s_ok:
            continue
        if atype in _AWS_ACTION_TYPES and not aws_ok:
            continue
        # Remove actions for irrelevant infra based on incident classification
        if classification and not classification["general"]:
            if atype in _K8S_ACTION_TYPES and not classification["k8s"]:
                continue
            ec2_types = {"ec2_start", "ec2_stop", "ec2_reboot"}
            ecs_types  = {"ecs_scale", "ecs_redeploy"}
            if atype in ec2_types and not classification["ec2"] and not classification["general"]:
                continue
            if atype in ecs_types and not classification["ecs"] and not classification["general"]:
                continue
            if atype == "lambda_invoke" and not classification["lambda"] and not classification["general"]:
                continue
            if atype == "rds_reboot" and not classification["rds"] and not classification["general"]:
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

        description = state.get("description", "")
        classification = _classify_incident(description)

        prompt = _PROMPT_TEMPLATE.format(
            incident_id       = state.get("incident_id", "unknown"),
            description       = description,
            aws_available     = _data_summary(aws_ctx),
            k8s_available     = _data_summary(k8s_ctx),
            github_available  = _data_summary(github_ctx),
            aws_context       = json.dumps(aws_ctx, default=str)[:2000],
            k8s_context       = json.dumps(k8s_ctx, default=str)[:1500],
            github_context    = json.dumps(github_ctx, default=str)[:1000],
            similar_incidents = json.dumps(state.get("similar_incidents", []), default=str)[:800],
            focus_hint        = _focus_hint(classification),
        )

        from app.llm.factory import get_global_provider
        _global = get_global_provider()
        _provider_order = [_global, "groq", "claude", "openai"] if _global else ["groq", "claude", "openai"]
        # deduplicate while preserving order
        seen = set(); _provider_order = [p for p in _provider_order if not (p in seen or seen.add(p))]
        last_exc = None
        for attempt in range(4):
            try:
                preferred = _provider_order[min(attempt, len(_provider_order) - 1)]
                llm = LLMFactory.get(preferred=preferred)
            except RuntimeError as exc:
                last_exc = exc
                break
            try:
                response = llm.complete(prompt, system=_SYSTEM, max_tokens=1500)
                plan = self._parse_json(response.content)
                if not plan:
                    raise ValueError("Empty plan returned by LLM")

                # Hard-filter: remove actions for unavailable OR irrelevant infrastructure
                plan["actions"] = _clean_actions(plan.get("actions", []), aws_ok, k8s_ok, classification)

                # Enrich slack_warroom actions with incident context from state
                for a in plan.get("actions", []):
                    if a.get("type") == "slack_warroom":
                        a.setdefault("incident_id", state.get("incident_id", "unknown"))
                        a.setdefault("severity", state.get("severity", "medium"))
                        a.setdefault("description", state.get("description", ""))

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
                _is_transient = (
                    "429" in err_str
                    or "rate_limit" in err_str.lower()
                    or "rate limit" in err_str.lower()
                    or "no credits" in err_str.lower()
                    or "credit balance" in err_str.lower()
                    or "billing" in err_str.lower()
                    or "falling back to next available provider" in err_str.lower()
                    or "timeout" in err_str.lower()
                    or "timed out" in err_str.lower()
                    or "read timeout" in err_str.lower()
                    or "connect timeout" in err_str.lower()
                )
                if _is_transient:
                    provider_key = getattr(llm, "_force_provider", None) or preferred
                    # insufficient_quota = long-term issue, cool down for 24h
                    if "insufficient_quota" in err_str or "exceeded your current quota" in err_str:
                        from app.llm.factory import _rate_limited_until
                        import time as _t
                        _rate_limited_until[str(provider_key)] = _t.monotonic() + 86400
                    else:
                        mark_rate_limited(str(provider_key) if provider_key else preferred, err_str)
                    self._warn("planner_rate_limited_retrying", provider=str(provider_key), attempt=attempt + 1)
                    continue
                break

        self._warn("planner_agent_failed", error=str(last_exc))
        state.setdefault("errors", []).append(f"PlannerAgent: {last_exc}")
        state["plan"] = {"actions": [], "confidence": 0.0, "risk": "unknown",
                         "root_cause": "Planning failed", "summary": str(last_exc)}
        return state
