"""Intelligent chatbot with tool-calling capability.

The LLM is given tool descriptions in the system prompt and can emit
``[TOOL_CALL: tool_name({"key": "value"})]`` tokens in its response.
This module parses those tokens, executes the real integration calls, and
feeds results back to the LLM for a final polished answer.

Up to 3 tool-call rounds are performed per user message.

Usage:
    from app.chat.intelligence import chat_with_intelligence
    answer = chat_with_intelligence("How many pods are running?", session_id="sess-1")
"""
from __future__ import annotations

import datetime
import json
import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Optional integration imports
# ---------------------------------------------------------------------------
try:
    from app.llm.factory import LLMFactory
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    from app.chat.memory import add_message, get_history, get_or_create_session
    _MEMORY_AVAILABLE = True
except ImportError:
    _MEMORY_AVAILABLE = False

try:
    from app.integrations import k8s_ops
    _K8S_AVAILABLE = True
except ImportError:
    _K8S_AVAILABLE = False

try:
    from app.integrations import aws_ops
    _AWS_AVAILABLE = True
except ImportError:
    _AWS_AVAILABLE = False

try:
    from app.integrations import github as github_ops
    _GITHUB_AVAILABLE = True
except ImportError:
    _GITHUB_AVAILABLE = False

try:
    from app.integrations import grafana
    _GRAFANA_AVAILABLE = True
except ImportError:
    _GRAFANA_AVAILABLE = False

try:
    from app.memory.vector_db import search_similar_incidents as query_incidents
    _MEMORY_VECTOR_AVAILABLE = True
except ImportError:
    _MEMORY_VECTOR_AVAILABLE = False

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-level EC2 context cache (instance_id → name per session)
# ---------------------------------------------------------------------------
import time as _time

_EC2_CACHE_TTL = 300  # 5 minutes

# Format: session_id → {"instances": [...], "expires_at": float}
_ec2_session_cache: dict[str, dict] = {}

# ── Generic tool result cache (read-only tools, 60s TTL) ──────────────────────
# key: (tool_name, frozenset(params.items())) → {"result": str, "expires_at": float}
_TOOL_CACHE: dict[tuple, dict] = {}
_TOOL_CACHE_TTL = 60  # seconds — short enough to stay fresh, long enough to avoid duplicate calls

# Tools whose results can be safely cached (no side effects)
_CACHEABLE_TOOLS = frozenset({
    "list_ec2_instances", "get_ec2_status", "list_rds_instances",
    "list_lambda_functions", "list_ecs_services", "list_s3_buckets",
    "list_cloudwatch_alarms", "get_cloudwatch_alarms", "list_sqs_queues",
    "list_dynamodb_tables", "get_cost_by_service", "get_cost_optimization_data",
    "query_k8s_pods", "query_k8s_nodes", "query_k8s_deployments",
    "list_github_repos", "get_recent_commits", "list_jira_issues",
    "get_aws_costs", "query_aws_resources", "get_infra_overview",
})


def _tool_cache_get(tool_name: str, params: dict) -> str | None:
    if tool_name not in _CACHEABLE_TOOLS:
        return None
    key = (tool_name, frozenset(sorted(params.items())))
    entry = _TOOL_CACHE.get(key)
    if entry and _time.time() < entry["expires_at"]:
        return entry["result"]
    return None


def _tool_cache_set(tool_name: str, params: dict, result: str) -> None:
    if tool_name not in _CACHEABLE_TOOLS:
        return
    key = (tool_name, frozenset(sorted(params.items())))
    _TOOL_CACHE[key] = {"result": result, "expires_at": _time.time() + _TOOL_CACHE_TTL}
    # Evict old entries to prevent unbounded growth
    if len(_TOOL_CACHE) > 200:
        now = _time.time()
        expired = [k for k, v in _TOOL_CACHE.items() if v["expires_at"] < now]
        for k in expired:
            del _TOOL_CACHE[k]


def _get_cached_instances(session_id: str) -> list[dict]:
    """Return cached EC2 instances if not expired, else empty list."""
    entry = _ec2_session_cache.get(session_id)
    if entry and _time.time() < entry["expires_at"]:
        return entry["instances"]
    return []


def _set_cached_instances(session_id: str, instances: list[dict]) -> None:
    """Store EC2 instances in the session cache with TTL."""
    _ec2_session_cache[session_id] = {
        "instances": instances,
        "expires_at": _time.time() + _EC2_CACHE_TTL,
    }


def _clear_ec2_cache(session_id: str) -> None:
    """Invalidate EC2 cache for a session (call after mutating actions)."""
    _ec2_session_cache.pop(session_id, None)

_PRONOUN_IDS = {
    "that", "it", "the", "this", "instance", "one", "them", "those",
    "my", "our", "ec2", "above", "same", "server", "machine", "box",
    "host", "node", "vm", "system", "running", "stopped", "down",
    "unknown", "none", "null", "n/a", "na", "unspecified", "not", "working",
}

def _resolve_instance_id(raw_id: str, session_id: str) -> str:
    """If raw_id looks like a pronoun/vague word, try to return the last known instance id."""
    if not raw_id:
        return raw_id
    # Already looks like a real instance ID (i-xxx) or Name tag (not a common word)
    normalized = raw_id.strip().lower().replace(" ", "")
    if raw_id.startswith("i-") or (raw_id and not all(w in _PRONOUN_IDS for w in raw_id.lower().split())):
        # Check if it's genuinely a pronoun phrase
        words = set(raw_id.lower().split())
        if not words.issubset(_PRONOUN_IDS):
            return raw_id  # looks like a real name/id
    # Try to resolve from session cache (respects TTL)
    instances = _get_cached_instances(session_id)
    if instances:
        # Return the first (and usually only) instance
        return instances[0].get("id") or instances[0].get("instance_id") or raw_id
    return raw_id


def _auto_resolve_single_instance(session_id: str, action: str) -> str:
    """Return a helpful message when no specific instance ID was given for a mutating action."""
    if not _AWS_AVAILABLE:
        return "AWS integration not available."
    cached = _get_cached_instances(session_id)
    if not cached:
        # Try to list and cache
        try:
            result = aws_ops.list_ec2_instances()
            instances = result.get("instances", [])
            if instances:
                simplified = [
                    {"id": i.get("id",""), "name": i.get("name",""), "state": i.get("state","")}
                    for i in instances
                ]
                _set_cached_instances(session_id, simplified)
                cached = simplified
        except Exception:
            pass
    if not cached:
        return "No EC2 instances found in your account."
    if len(cached) == 1:
        iid = cached[0]["id"]
        return f"Found one instance: {iid} ({cached[0].get('name','unnamed')}). Please confirm — say 'yes, {action} {iid}' to proceed."
    names = ", ".join(f'{i["id"]} ({i.get("name","unnamed")}, {i.get("state","?")})' for i in cached[:5])
    return f"Multiple instances found: {names}. Which one should I {action}?"


def _cache_ec2_instances(session_id: str, result_json: str) -> None:
    """Parse list_ec2_instances result and store in session cache with TTL."""
    try:
        data = json.loads(result_json)
        instances = (
            data.get("instances")
            or data.get("ec2_instances", {}).get("instances")
            or (data if isinstance(data, list) else [])
        )
        simplified = [
            {"id": i.get("id") or i.get("instance_id", ""),
             "name": i.get("name") or i.get("Name", "")}
            for i in (instances or []) if isinstance(i, dict)
        ]
        if simplified:
            _set_cached_instances(session_id, simplified)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tool schema converters — for native API function calling
# ---------------------------------------------------------------------------

def _build_anthropic_tools() -> list[dict]:
    """Convert TOOLS registry to Anthropic's native input_schema format."""
    result = []
    for t in TOOLS:
        props = {
            k: {"type": "string", "description": str(v)}
            for k, v in t["parameters"].items()
        }
        result.append({
            "name": t["name"],
            "description": t["description"],
            "input_schema": {
                "type": "object",
                "properties": props,
            },
        })
    return result


def _build_openai_tools() -> list[dict]:
    """Convert TOOLS registry to OpenAI/Groq function-calling schema format."""
    result = []
    for t in TOOLS:
        props = {
            k: {"type": "string", "description": str(v)}
            for k, v in t["parameters"].items()
        }
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                },
            },
        })
    return result


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "query_k8s_pods",
        "description": "Get current pod status across all namespaces or a specific namespace.",
        "parameters": {
            "namespace": "Optional namespace name (default: all namespaces)",
        },
    },
    {
        "name": "query_k8s_logs",
        "description": "Get recent logs from a specific Kubernetes pod.",
        "parameters": {
            "namespace": "Namespace of the pod",
            "pod": "Pod name",
            "tail_lines": "Number of log lines to fetch (default: 50)",
        },
    },
    {
        "name": "query_aws_alarms",
        "description": "Get all CloudWatch alarms and their current state (OK/ALARM/INSUFFICIENT_DATA).",
        "parameters": {
            "state": "Optional filter: ALARM, OK, or INSUFFICIENT_DATA",
            "region": "Optional AWS region (default: us-east-1)",
        },
    },
    {
        "name": "query_aws_metrics",
        "description": "Get a specific CloudWatch metric value for a resource.",
        "parameters": {
            "namespace": "CloudWatch namespace (e.g. AWS/EC2)",
            "metric_name": "Metric name (e.g. CPUUtilization)",
            "dimensions": "List of dimension dicts [{Name: ..., Value: ...}]",
            "period": "Period in seconds (default: 300)",
            "hours": "How many hours back to query (default: 1)",
        },
    },
    {
        "name": "list_github_repos",
        "description": (
            "List all GitHub repositories accessible with the configured token. "
            "Use this when the user asks 'list my repos', 'what repos do I have', "
            "'show my GitHub repositories', 'what's on my GitHub', etc."
        ),
        "parameters": {},
    },
    {
        "name": "query_github_recent",
        "description": "Get recent commits and pull requests from the configured GitHub repository.",
        "parameters": {
            "hours": "How many hours back to look (default: 24)",
        },
    },
    {
        "name": "get_commit_diff",
        "description": "Get the full code diff (files changed, lines added/removed) for a specific commit SHA. Use this when user asks 'what changed in commit X', 'show me the code changes', 'what files were modified'.",
        "parameters": {
            "sha": "The commit SHA hash",
            "repo_name": "Optional repository name (uses configured default if omitted)",
        },
    },
    {
        "name": "query_grafana_alerts",
        "description": "Get all currently firing Grafana alerts.",
        "parameters": {},
    },
    {
        "name": "query_incidents_memory",
        "description": "Search past incidents in vector memory for similar patterns.",
        "parameters": {
            "query": "Search query describing the incident or symptom",
            "top_k": "Number of results to return (default: 3)",
        },
    },
    {
        "name": "get_current_cost",
        "description": "Get current AWS cost estimate for the last 7 days and monthly projection.",
        "parameters": {
            "region": "Optional AWS region (default: us-east-1)",
        },
    },
    {
        "name": "list_ec2_instances",
        "description": "List all EC2 instances and their current state (running/stopped/etc). Supports region filtering.",
        "parameters": {
            "region": "AWS region code (e.g. us-east-1, ap-southeast-1). Leave empty for default region.",
        },
    },
    {
        "name": "list_aws_resources",
        "description": (
            "List ALL AWS resources (EC2, ECS, RDS, Lambda, S3, SQS, DynamoDB) in a given region or set of regions. "
            "Use this when the user asks 'what do I have in Asia', 'show all resources in eu-west-1', "
            "'what AWS services are running in us-west-2', 'list everything in Singapore', etc. "
            "Covers: EC2, ECS, RDS, Lambda, S3, SQS, DynamoDB."
        ),
        "parameters": {
            "region": "AWS region code or region group keyword: 'asia', 'eu', 'europe', 'us', 'all', or specific code like 'ap-southeast-1'. Defaults to all regions.",
        },
    },
    {
        "name": "start_ec2_instance",
        "description": "Start a stopped EC2 instance. Use when user says 'start that', 'start the instance', 'bring it up', etc. Requires confirmation — always confirm with the user before executing.",
        "parameters": {
            "instance_id": "The EC2 instance ID (e.g. i-0abc123) or Name tag",
        },
    },
    {
        "name": "stop_ec2_instance",
        "description": "Stop a running EC2 instance. Requires confirmation before executing.",
        "parameters": {
            "instance_id": "The EC2 instance ID or Name tag",
        },
    },
    {
        "name": "reboot_ec2_instance",
        "description": "Reboot an EC2 instance. Requires confirmation before executing.",
        "parameters": {
            "instance_id": "The EC2 instance ID or Name tag",
        },
    },
    {
        "name": "get_ec2_instance_status",
        "description": "Get the detailed status and health checks of a specific EC2 instance.",
        "parameters": {
            "instance_id": "The EC2 instance ID or Name tag",
        },
    },
    # ── ECS ──────────────────────────────────────────────────────────────────
    {
        "name": "list_ecs_services",
        "description": "List all ECS services in a cluster and their running/desired task counts.",
        "parameters": {"cluster": "ECS cluster name (default: default)"},
    },
    {
        "name": "scale_ecs_service",
        "description": "Scale an ECS service to a desired task count. Requires confirmation.",
        "parameters": {
            "cluster": "ECS cluster name",
            "service": "ECS service name",
            "desired_count": "Number of tasks to run",
        },
    },
    {
        "name": "redeploy_ecs_service",
        "description": "Force a new ECS deployment (rolling restart). Requires confirmation.",
        "parameters": {
            "cluster": "ECS cluster name",
            "service": "ECS service name",
        },
    },
    # ── Lambda ───────────────────────────────────────────────────────────────
    {
        "name": "list_lambda_functions",
        "description": "List all Lambda functions and their runtime/last-modified info.",
        "parameters": {"region": "Optional AWS region"},
    },
    {
        "name": "get_lambda_errors",
        "description": "Get recent error count for a Lambda function from CloudWatch.",
        "parameters": {
            "function_name": "Lambda function name",
            "hours": "Hours back to check (default: 1)",
        },
    },
    # ── RDS ──────────────────────────────────────────────────────────────────
    {
        "name": "list_rds_instances",
        "description": "List all RDS database instances and their status.",
        "parameters": {"region": "Optional AWS region"},
    },
    {
        "name": "reboot_rds_instance",
        "description": "Reboot an RDS instance (brief downtime). Requires confirmation.",
        "parameters": {"db_instance_id": "RDS instance identifier"},
    },
    # ── Jira ─────────────────────────────────────────────────────────────────
    {
        "name": "create_jira_ticket",
        "description": "Create a Jira ticket for an incident or task. Requires confirmation.",
        "parameters": {
            "summary": "Ticket title/summary",
            "description": "Detailed description",
            "issue_type": "Bug, Task, or Story (default: Bug)",
        },
    },
    {
        "name": "get_jira_issue",
        "description": "Get details of a specific Jira issue by key (e.g. DEVOPS-123).",
        "parameters": {"issue_key": "Jira issue key"},
    },
    # ── OpsGenie ─────────────────────────────────────────────────────────────
    {
        "name": "page_oncall",
        "description": "Page the on-call engineer via OpsGenie. Requires confirmation.",
        "parameters": {
            "message": "Alert message to send",
            "priority": "P1, P2, P3, P4, or P5 (default: P3)",
        },
    },
    # ── Slack ─────────────────────────────────────────────────────────────────
    {
        "name": "send_slack_message",
        "description": "Send a message to a Slack channel.",
        "parameters": {
            "channel": "Channel name or ID (e.g. #incidents)",
            "message": "Message text to send",
        },
    },
    {
        "name": "create_war_room",
        "description": "Create a Slack war room channel for an incident and invite team members.",
        "parameters": {
            "incident_id": "Incident ID (e.g. INC-001)",
            "topic": "Channel topic/description",
        },
    },
    {
        "name": "get_slack_channel_history",
        "description": "Get recent messages from a Slack channel (e.g. a war room).",
        "parameters": {
            "channel": "Channel name or ID",
            "limit": "Number of messages to fetch (default: 20)",
        },
    },
    # ── GitHub ───────────────────────────────────────────────────────────────
    {
        "name": "create_github_issue",
        "description": "Create a GitHub issue to track a problem or task. Requires confirmation.",
        "parameters": {
            "title": "Issue title",
            "body": "Issue description",
            "labels": "Optional comma-separated labels (e.g. bug,incident)",
        },
    },
    {
        "name": "list_github_prs",
        "description": "List recent open pull requests from GitHub.",
        "parameters": {"hours": "How many hours back to look (default: 48)"},
    },
    # ── Cost Estimation ──────────────────────────────────────────────────────
    {
        "name": "estimate_aws_cost",
        "description": (
            "Estimate the monthly and annual AWS cost for any resource or infrastructure description. "
            "Uses the official AWS Pricing API for live prices. "
            "Use when the user asks: 'how much will it cost', 'what is the price of', "
            "'estimate cost for', 'how much does X instance cost', 'compare pricing', "
            "or asks about any AWS service pricing. "
            "Supports EC2, RDS, Lambda, ECS Fargate, S3 — with real AWS on-demand rates."
        ),
        "parameters": {
            "description": "Natural language description of the resource(s) to estimate. "
                           "Examples: '3 t3.medium instances', 'db.r5.large postgres 200gb multi-az', "
                           "'lambda 512mb 10 million invocations', 'fargate 2 vcpu 4gb 5 tasks', '500gb s3'",
            "region": "AWS region code (default: us-east-1). E.g. us-west-2, eu-west-1.",
        },
    },
    {
        "name": "estimate_terraform_cost",
        "description": (
            "Estimate monthly cost from a Terraform plan JSON. "
            "Use when the user shares Terraform code or asks about cost impact of infrastructure-as-code changes. "
            "Parses aws_instance, aws_db_instance, aws_lambda_function, aws_ecs_service, etc."
        ),
        "parameters": {
            "plan_json": "Terraform plan JSON string (output of 'terraform show -json')",
            "region":    "AWS region code (default: us-east-1)",
        },
    },

    # ── CloudWatch Logs ──────────────────────────────────────────────────────
    {
        "name": "list_log_groups",
        "description": (
            "List available CloudWatch log groups. Use when user asks 'what logs do I have', "
            "'show me log groups', or before searching logs without knowing the group name."
        ),
        "parameters": {},
    },
    {
        "name": "get_recent_logs",
        "description": (
            "Fetch recent CloudWatch log events from a log group. Use when user asks 'show me logs', "
            "'what's in the logs', 'show me the last X minutes of logs', 'show errors in logs', "
            "'what happened between 2pm and 3pm', or any log viewing request."
        ),
        "parameters": {
            "log_group": "CloudWatch log group name (e.g. /aws/lambda/my-function)",
            "hours": "How many hours back to look (default: 1)",
            "filter_pattern": "Optional CloudWatch filter pattern (e.g. ERROR, WARN, timeout)",
        },
    },
    {
        "name": "search_logs",
        "description": (
            "Search CloudWatch logs for a pattern or keyword. Use when user asks 'find errors in logs', "
            "'search for timeout', 'look for OOM in logs', 'any exceptions today', or any log search."
        ),
        "parameters": {
            "log_group": "CloudWatch log group name",
            "pattern": "Search pattern or keyword (e.g. ERROR, NullPointer, timeout, 500)",
            "hours": "How many hours back to search (default: 1)",
        },
    },
    # ── Grafana ──────────────────────────────────────────────────────────────
    {
        "name": "get_grafana_status",
        "description": (
            "Get Grafana health — firing alerts, datasource status, recent annotations. "
            "Use when user asks about Grafana, dashboards, or monitoring alerts."
        ),
        "parameters": {},
    },
    # ── VS Code ──────────────────────────────────────────────────────────────
    {
        "name": "vscode_notify",
        "description": (
            "Send a notification popup to VS Code. Use when user asks to notify in VS Code, "
            "'alert the editor', or 'send to IDE'."
        ),
        "parameters": {
            "message": "Notification message text",
            "level": "info, warning, or error (default: info)",
        },
    },
    {
        "name": "vscode_open_file",
        "description": "Open a file in VS Code at a specific line.",
        "parameters": {
            "file_path": "Absolute path to the file",
            "line": "Optional line number to jump to",
        },
    },
    # ── S3 / SQS / DynamoDB ─────────────────────────────────────────────────
    {
        "name": "list_s3_buckets",
        "description": "List all S3 buckets with their names and creation dates.",
        "parameters": {},
    },
    {
        "name": "list_sqs_queues",
        "description": "List all SQS queues with message counts (visible, in-flight, delayed).",
        "parameters": {},
    },
    {
        "name": "list_dynamodb_tables",
        "description": "List all DynamoDB tables with status, item count, and billing mode.",
        "parameters": {},
    },
    # ── Cost Intelligence ────────────────────────────────────────────────────
    {
        "name": "get_cost_by_service",
        "description": (
            "Get a detailed AWS cost breakdown by service (EC2, RDS, S3, Lambda, etc.) for the last N days. "
            "Use when user asks 'what's my biggest cost', 'what am I spending on S3', 'break down my bill', "
            "'which service costs the most', 'show my AWS spending by service'."
        ),
        "parameters": {
            "days": "Number of days to look back (default: 30)",
            "region": "Optional AWS region for the Cost Explorer client",
        },
    },
    {
        "name": "get_cost_optimization",
        "description": (
            "Analyse the current AWS infrastructure and return specific cost optimization recommendations. "
            "Use when user asks 'how can I save money', 'cost optimization tips', 'reduce my AWS bill', "
            "'what's wasting money', 'how to cut costs', 'any unused resources'."
        ),
        "parameters": {
            "region": "Optional AWS region to check",
        },
    },
    # ── CloudTrail ───────────────────────────────────────────────────────────
    {
        "name": "get_cloudtrail_events",
        "description": (
            "Fetch AWS CloudTrail audit events — who started/stopped/modified what and when. "
            "Use this when the user asks who did something, what changed, or for audit history. "
            "Supports flexible time ranges (hours or days) and filters by resource, event type, or user."
        ),
        "parameters": {
            "days": "Number of days back to look (e.g. 7, 30, 90). Takes priority over hours.",
            "hours": "Number of hours back to look (default: 24 if days not set).",
            "resource_name": "Filter by resource ID or name (e.g. i-0abc1234)",
            "event_name": "Filter by API event name (e.g. StopInstances, StartInstances, TerminateInstances)",
            "username": "Filter by IAM username or role",
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------



def _log(level_fn, msg: str, **kwargs) -> None:
    """Emit a structured log message compatible with stdlib logging."""
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)

def execute_tool(tool_name: str, params: dict, session_id: str = "") -> str:
    """Execute a named tool with given parameters and return a string result.

    Args:
        tool_name: One of the tool names defined in TOOLS.
        params:    Dict of parameters for the tool.

    Returns:
        String representation of the tool result.
    """
    # Return cached result if available (avoids redundant AWS/K8s API calls)
    cached = _tool_cache_get(tool_name, params)
    if cached is not None:
        return cached

    try:
        if tool_name == "query_k8s_pods":
            if not _K8S_AVAILABLE:
                return "Kubernetes integration not available."
            ns = params.get("namespace", "")
            result = k8s_ops.list_pods(namespace=ns)
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "query_k8s_logs":
            if not _K8S_AVAILABLE:
                return "Kubernetes integration not available."
            result = k8s_ops.get_pod_logs(
                namespace=params.get("namespace", "default"),
                pod=params.get("pod", ""),
                tail_lines=int(params.get("tail_lines", 50)),
            )
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "query_aws_alarms":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_cloudwatch_alarms(
                state=params.get("state", ""),
                region=params.get("region", ""),
            )
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "query_aws_metrics":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            dims = params.get("dimensions", [])
            if isinstance(dims, str):
                try:
                    dims = json.loads(dims)
                except Exception:
                    dims = []
            result = aws_ops.get_metric(
                namespace=params.get("namespace", "AWS/EC2"),
                metric_name=params.get("metric_name", "CPUUtilization"),
                dimensions=dims,
                period=int(params.get("period", 300)),
                hours=int(params.get("hours", 1)),
            )
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "list_github_repos":
            if not _GITHUB_AVAILABLE:
                return "GitHub integration not available."
            result = github_ops.list_repos()
            return json.dumps(result, default=str)[:3000]

        elif tool_name == "query_github_recent":
            if not _GITHUB_AVAILABLE:
                return "GitHub integration not available."
            hours = int(params.get("hours", 24))
            commits = github_ops.get_recent_commits(hours=hours)
            prs     = github_ops.get_recent_prs(hours=hours * 12)
            return json.dumps({"commits": commits, "prs": prs}, default=str)[:2000]

        elif tool_name == "get_commit_diff":
            if not _GITHUB_AVAILABLE:
                return "GitHub integration not available."
            sha = params.get("sha", "")
            if not sha:
                return "sha is required to fetch a commit diff."
            repo_name = params.get("repo_name", "")
            result = github_ops.get_commit_diff(sha=sha, repo_name=repo_name)
            if not result.get("success"):
                return f"Could not fetch diff: {result.get('error','unknown error')}"
            files = result.get("files", [])
            lines = [f"Commit {sha[:7]} — {result.get('message','')[:80]}",
                     f"Author: {result.get('author','?')} at {result.get('date','?')[:10]}",
                     f"Files changed: {len(files)}"]
            for f in files[:8]:
                patch_preview = (f.get("patch") or "")[:300]
                lines.append(f"\n  {f['filename']} ({f['status']}, +{f['additions']} -{f['deletions']})")
                if patch_preview:
                    lines.append(f"  ```\n{patch_preview}\n  ```")
            return "\n".join(lines)[:2500]

        elif tool_name == "query_grafana_alerts":
            if not _GRAFANA_AVAILABLE:
                return "Grafana integration not available."
            result = grafana.get_firing_alerts()
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "query_incidents_memory":
            if not _MEMORY_VECTOR_AVAILABLE:
                return "Incident vector memory not available."
            result = query_incidents(
                query=params.get("query", ""),
                n_results=int(params.get("top_k", 3)),
            )
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "list_ec2_instances":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            region = params.get("region", "")
            result = aws_ops.list_ec2_instances(region=region) if region else aws_ops.list_ec2_instances()
            result_str = json.dumps(result, default=str)[:2000]
            if session_id:
                _cache_ec2_instances(session_id, result_str)
            return result_str

        elif tool_name == "list_aws_resources":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            import boto3 as _boto3
            region_input = (params.get("region") or "").lower().strip()

            # Resolve region groups to specific region codes
            _REGION_GROUPS = {
                "asia":      ["ap-southeast-1","ap-southeast-2","ap-northeast-1","ap-northeast-2","ap-south-1"],
                "apac":      ["ap-southeast-1","ap-southeast-2","ap-northeast-1","ap-northeast-2","ap-south-1"],
                "singapore": ["ap-southeast-1"],
                "tokyo":     ["ap-northeast-1"],
                "sydney":    ["ap-southeast-2"],
                "mumbai":    ["ap-south-1"],
                "seoul":     ["ap-northeast-2"],
                "eu":        ["eu-west-1","eu-west-2","eu-central-1","eu-north-1","eu-west-3"],
                "europe":    ["eu-west-1","eu-west-2","eu-central-1","eu-north-1","eu-west-3"],
                "ireland":   ["eu-west-1"],
                "frankfurt": ["eu-central-1"],
                "london":    ["eu-west-2"],
                "us":        ["us-east-1","us-east-2","us-west-1","us-west-2"],
                "virginia":  ["us-east-1"],
                "oregon":    ["us-west-2"],
                "all":       ["us-east-1","us-east-2","us-west-1","us-west-2",
                               "eu-west-1","eu-west-2","eu-central-1",
                               "ap-southeast-1","ap-southeast-2","ap-northeast-1","ap-south-1"],
            }
            if region_input in _REGION_GROUPS:
                regions = _REGION_GROUPS[region_input]
            elif region_input:
                regions = [region_input]
            else:
                regions = [os.getenv("AWS_REGION", "us-east-1")]

            summary_lines = []
            for r in regions:
                parts = [f"**{r}**:"]
                # EC2
                try:
                    ec2_r = aws_ops.list_ec2_instances(region=r)
                    insts = ec2_r.get("instances", [])
                    if insts:
                        running = sum(1 for i in insts if i.get("state") == "running")
                        parts.append(f"EC2 {running}/{len(insts)} running")
                except Exception:
                    pass
                # RDS
                try:
                    rds_cl = _boto3.client("rds", region_name=r)
                    rds_resp = rds_cl.describe_db_instances()
                    dbs = rds_resp.get("DBInstances", [])
                    if dbs:
                        parts.append(f"RDS {len(dbs)} db")
                except Exception:
                    pass
                # Lambda
                try:
                    lam_cl = _boto3.client("lambda", region_name=r)
                    lam_resp = lam_cl.list_functions(MaxItems=20)
                    fns = lam_resp.get("Functions", [])
                    if fns:
                        parts.append(f"Lambda {len(fns)}+ functions")
                except Exception:
                    pass
                # ECS
                try:
                    ecs_cl = _boto3.client("ecs", region_name=r)
                    cls_resp = ecs_cl.list_clusters()
                    if cls_resp.get("clusterArns"):
                        total_svcs = 0
                        for carn in cls_resp["clusterArns"]:
                            svc_r = ecs_cl.list_services(cluster=carn.split("/")[-1])
                            total_svcs += len(svc_r.get("serviceArns", []))
                        if total_svcs:
                            parts.append(f"ECS {total_svcs} services")
                except Exception:
                    pass
                # SQS
                try:
                    sqs_cl = _boto3.client("sqs", region_name=r)
                    sqs_resp = sqs_cl.list_queues()
                    queues = sqs_resp.get("QueueUrls", [])
                    if queues:
                        parts.append(f"SQS {len(queues)} queues")
                except Exception:
                    pass
                # DynamoDB
                try:
                    ddb_cl = _boto3.client("dynamodb", region_name=r)
                    ddb_resp = ddb_cl.list_tables()
                    tables = ddb_resp.get("TableNames", [])
                    if tables:
                        parts.append(f"DynamoDB {len(tables)} tables")
                except Exception:
                    pass

                if len(parts) > 1:
                    summary_lines.append(" — ".join(parts))
                else:
                    summary_lines.append(f"**{r}**: no resources found")

            # S3 is global — list once
            try:
                s3_cl = _boto3.client("s3", region_name="us-east-1")
                s3_resp = s3_cl.list_buckets()
                buckets = s3_resp.get("Buckets", [])
                if buckets:
                    summary_lines.append(f"**S3 (global)**: {len(buckets)} buckets")
            except Exception:
                pass

            if not summary_lines:
                return f"No AWS resources found in the requested region(s): {', '.join(regions)}"
            header = f"AWS Resources across {', '.join(regions)}:\n"
            return header + "\n".join(f"  • {l}" for l in summary_lines)

        elif tool_name == "start_ec2_instance":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            instance_id = _resolve_instance_id(params.get("instance_id", ""), session_id)
            if not instance_id or not instance_id.startswith("i-"):
                return _auto_resolve_single_instance(session_id, "start")
            result = aws_ops.start_ec2_instance(instance_id=instance_id)
            _clear_ec2_cache(session_id)  # state changed — invalidate cache
            return json.dumps(result, default=str)[:1000]

        elif tool_name == "stop_ec2_instance":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            instance_id = _resolve_instance_id(params.get("instance_id", ""), session_id)
            if not instance_id or not instance_id.startswith("i-"):
                return _auto_resolve_single_instance(session_id, "stop")
            result = aws_ops.stop_ec2_instance(instance_id=instance_id)
            _clear_ec2_cache(session_id)  # state changed — invalidate cache
            return json.dumps(result, default=str)[:1000]

        elif tool_name == "reboot_ec2_instance":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            instance_id = _resolve_instance_id(params.get("instance_id", ""), session_id)
            if not instance_id or not instance_id.startswith("i-"):
                return _auto_resolve_single_instance(session_id, "reboot")
            result = aws_ops.reboot_ec2_instance(instance_id=instance_id)
            _clear_ec2_cache(session_id)  # state changed — invalidate cache
            return json.dumps(result, default=str)[:1000]

        elif tool_name == "get_ec2_instance_status":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            instance_id = _resolve_instance_id(params.get("instance_id", ""), session_id)
            # If still no real instance ID, list instances first so the LLM can pick the right one
            if not instance_id or not instance_id.startswith("i-"):
                instances_raw = aws_ops.list_ec2_instances()
                instances = instances_raw.get("instances", [])
                if not instances:
                    return "No EC2 instances found in your account."
                if len(instances) == 1:
                    instance_id = instances[0]["id"]
                else:
                    names = ", ".join(f'{i["id"]} ({i.get("name","unnamed")}, {i.get("state","?")})' for i in instances[:5])
                    return f"Multiple EC2 instances found: {names}. Please specify which one you mean."
            result = aws_ops.get_ec2_status_checks(instance_id=instance_id)
            return json.dumps(result, default=str)[:1000]

        elif tool_name == "get_current_cost":
            try:
                import boto3
                from botocore.exceptions import BotoCoreError, ClientError
                region = params.get("region", "us-east-1")
                ce = boto3.client("ce", region_name=region)
                import datetime as _dt
                end   = _dt.date.today()
                start = end - _dt.timedelta(days=7)
                resp  = ce.get_cost_and_usage(
                    TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
                    Granularity="MONTHLY",
                    Metrics=["UnblendedCost"],
                )
                total = sum(
                    float(r["Total"]["UnblendedCost"]["Amount"])
                    for r in resp.get("ResultsByTime", [])
                )
                monthly = total / 7 * 30
                return (
                    f"Last 7 days: ${total:.2f}. "
                    f"Estimated monthly projection: ${monthly:.2f}."
                )
            except Exception as exc:
                return f"Could not fetch AWS cost data: {exc}"

        # ── S3 / SQS / DynamoDB ─────────────────────────────────────────────
        elif tool_name == "list_s3_buckets":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_s3_buckets()
            if not result.get("success"):
                return f"Could not list S3 buckets: {result.get('error','unknown error')}"
            buckets = result.get("buckets", [])
            if not buckets:
                return "No S3 buckets found in your account."
            lines = [f"S3 Buckets ({len(buckets)} total):"]
            for b in buckets[:20]:
                created = b.get("created", "")[:10]
                lines.append(f"  • {b['name']} (created {created})")
            return "\n".join(lines)

        elif tool_name == "list_sqs_queues":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_sqs_queues()
            if not result.get("success"):
                return f"Could not list SQS queues: {result.get('error','unknown error')}"
            queues = result.get("queues", [])
            if not queues:
                return "No SQS queues found."
            lines = [f"SQS Queues ({len(queues)} total):"]
            for q in queues[:15]:
                vis    = q.get("visible", "?")
                inflight = q.get("in_flight", "?")
                lines.append(f"  • {q['name']} — {vis} visible, {inflight} in-flight")
            return "\n".join(lines)

        elif tool_name == "list_dynamodb_tables":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_dynamodb_tables()
            if not result.get("success"):
                return f"Could not list DynamoDB tables: {result.get('error','unknown error')}"
            tables = result.get("tables", [])
            if not tables:
                return "No DynamoDB tables found."
            lines = [f"DynamoDB Tables ({len(tables)} total):"]
            for t in tables[:15]:
                items = t.get("item_count", 0)
                billing = t.get("billing", "PROVISIONED")
                lines.append(f"  • {t['name']} — {t.get('status','?')} — {items:,} items — {billing}")
            return "\n".join(lines)

        elif tool_name == "get_cost_by_service":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            days   = int(params.get("days", 30))
            region = params.get("region", "")
            result = aws_ops.get_cost_by_service(days=days, region=region)
            if not result.get("success"):
                return f"Could not fetch cost data: {result.get('error','unknown error')} — Note: Cost Explorer must be enabled in your AWS account."
            lines = [
                f"AWS Cost Breakdown — last {days} days",
                f"  Total: ${result['total_usd']:,.2f}",
                f"  Daily avg: ${result['daily_avg_usd']:,.4f}",
                f"  Monthly projection: ${result['monthly_projection_usd']:,.2f}",
                "",
                "By service (top spenders):",
            ]
            for svc in result.get("by_service", [])[:15]:
                pct = (svc["cost_usd"] / result["total_usd"] * 100) if result["total_usd"] else 0
                lines.append(f"  • {svc['service']}: ${svc['cost_usd']:,.4f} ({pct:.1f}%)")
            return "\n".join(lines)

        elif tool_name == "get_cost_optimization":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            region = params.get("region", "")
            result = aws_ops.get_cost_optimization_data(region=region)
            if not result.get("success"):
                return "Could not collect optimization data."
            findings = result.get("findings", [])
            if not findings:
                return json.dumps({
                    "summary": "No obvious cost waste detected.",
                    "ec2": result.get("ec2", {}),
                    "rds": result.get("rds", {}),
                    "lambda": result.get("lambda", {}),
                    "s3": result.get("s3", {}),
                }, default=str)[:2000]
            return json.dumps(result, default=str)[:3000]

        # ── ECS ──────────────────────────────────────────────────────────────
        elif tool_name == "list_ecs_services":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            cluster = params.get("cluster", "default")
            result = aws_ops.list_ecs_services(cluster=cluster)
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "scale_ecs_service":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            cluster = params.get("cluster", "default")
            service = params.get("service", "")
            count   = int(params.get("desired_count", 1))
            if not service:
                return "service name is required to scale ECS."
            result = aws_ops.scale_ecs_service(cluster=cluster, service=service, desired_count=count)
            return json.dumps(result, default=str)[:1000]

        elif tool_name == "redeploy_ecs_service":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            cluster = params.get("cluster", "default")
            service = params.get("service", "")
            if not service:
                return "service name is required to redeploy ECS."
            result = aws_ops.force_new_ecs_deployment(cluster=cluster, service=service)
            return json.dumps(result, default=str)[:1000]

        # ── Lambda ───────────────────────────────────────────────────────────
        elif tool_name == "list_lambda_functions":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_lambda_functions(region=params.get("region", ""))
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "get_lambda_errors":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            fn = params.get("function_name", "")
            if not fn:
                return "function_name is required."
            result = aws_ops.get_lambda_errors(function_name=fn, hours=int(params.get("hours", 1)))
            return json.dumps(result, default=str)[:1000]

        # ── RDS ──────────────────────────────────────────────────────────────
        elif tool_name == "list_rds_instances":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_rds_instances(region=params.get("region", ""))
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "reboot_rds_instance":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            db_id = params.get("db_instance_id", "")
            if not db_id:
                return "db_instance_id is required to reboot RDS."
            result = aws_ops.reboot_rds_instance(db_instance_id=db_id)
            return json.dumps(result, default=str)[:1000]

        # ── Jira ─────────────────────────────────────────────────────────────
        elif tool_name == "create_jira_ticket":
            try:
                from app.integrations import jira as jira_ops
                summary     = params.get("summary", "AI-generated ticket")
                description = params.get("description", "")
                result = jira_ops.create_incident(summary=summary, description=description)
                return json.dumps(result, default=str)[:1000]
            except Exception as exc:
                return f"Jira not configured or failed: {exc}"

        elif tool_name == "get_jira_issue":
            try:
                from app.integrations import jira as jira_ops
                key = params.get("issue_key", "")
                if not key:
                    return "issue_key is required."
                result = jira_ops.get_issue(issue_key=key)
                return json.dumps(result, default=str)[:1500]
            except Exception as exc:
                return f"Jira not configured or failed: {exc}"

        # ── OpsGenie ─────────────────────────────────────────────────────────
        elif tool_name == "page_oncall":
            try:
                from app.integrations.opsgenie import notify_on_call
                message  = params.get("message", "Incident alert from NsOps")
                priority = params.get("priority", "P3")
                result = notify_on_call(message=message)
                return f"On-call paged successfully: {result}"
            except Exception as exc:
                return f"OpsGenie not configured or failed: {exc}"

        # ── Slack ─────────────────────────────────────────────────────────────
        elif tool_name == "send_slack_message":
            try:
                from app.integrations.slack import post_message
                channel = params.get("channel", "#general")
                message = params.get("message", "")
                if not message:
                    return "message is required."
                result = post_message(channel=channel, text=message)
                return json.dumps(result, default=str)[:500]
            except Exception as exc:
                return f"Slack not configured or failed: {exc}"

        elif tool_name == "create_war_room":
            try:
                from app.integrations.slack import create_war_room as slack_war_room
                incident_id = params.get("incident_id", "")
                topic       = params.get("topic", "Incident War Room")
                result = slack_war_room(topic=topic, incident_id=incident_id)
                if result.get("success"):
                    ch = result.get("channel_name") or result.get("channel_id", "")
                    return f"War room created: #{ch}. Channel ID: {result.get('channel_id','')}. Team can now collaborate there."
                return f"War room creation result: {json.dumps(result, default=str)}"
            except Exception as exc:
                return f"Slack not configured or failed: {exc}"

        elif tool_name == "get_slack_channel_history":
            try:
                from app.integrations.slack import _client as _slack_client
                channel = params.get("channel", "")
                limit   = int(params.get("limit", 20))
                if not channel:
                    return "channel is required."
                sc = _slack_client()
                # resolve channel name → ID if needed
                if channel.startswith("#"):
                    channel = channel.lstrip("#")
                    ch_list = sc.conversations_list(types="public_channel,private_channel", limit=200)
                    for ch in ch_list.get("channels", []):
                        if ch["name"] == channel:
                            channel = ch["id"]
                            break
                resp = sc.conversations_history(channel=channel, limit=limit)
                messages = resp.get("messages", [])
                if not messages:
                    return "No messages found in this channel."
                lines = []
                for m in reversed(messages):
                    ts = m.get("ts", "")
                    user = m.get("user", m.get("username", "unknown"))
                    text = m.get("text", "")
                    lines.append(f"[{user}]: {text}")
                return "\n".join(lines[:20])
            except Exception as exc:
                return f"Slack not configured or failed: {exc}"

        # ── GitHub ───────────────────────────────────────────────────────────
        elif tool_name == "create_github_issue":
            if not _GITHUB_AVAILABLE:
                return "GitHub integration not available."
            title  = params.get("title", "AI-generated issue")
            body   = params.get("body", "")
            labels_raw = params.get("labels", "")
            labels = [l.strip() for l in labels_raw.split(",") if l.strip()] if labels_raw else []
            result = github_ops.create_issue(title=title, body=body, labels=labels)
            return json.dumps(result, default=str)[:1000]

        elif tool_name == "list_github_prs":
            if not _GITHUB_AVAILABLE:
                return "GitHub integration not available."
            hours = int(params.get("hours", 48))
            result = github_ops.get_recent_prs(hours=hours)
            return json.dumps(result, default=str)[:2000]

        elif tool_name == "estimate_aws_cost":
            from app.cost.pricing import estimate_from_description
            description = params.get("description", "")
            region      = params.get("region", os.getenv("AWS_REGION", "us-east-1"))
            if not description:
                return "Please provide a description of the resource(s) to estimate (e.g. '3 t3.medium instances')."
            result = estimate_from_description(description, region)
            # If no instance type specified, show a full pricing menu so user can pick
            if not result.get("resources") or any("defaulted" in w for w in result.get("warnings", [])):
                # Fetch live prices for common types
                from app.cost.pricing import estimate_ec2_cost
                tiers = [
                    ("t3.nano",    "0.5 GB",  "Dev / scratch"),
                    ("t3.micro",   "1 GB",    "Dev / low traffic"),
                    ("t3.small",   "2 GB",    "Light workload"),
                    ("t3.medium",  "4 GB",    "Small web app"),
                    ("t3.large",   "8 GB",    "Medium workload"),
                    ("m5.large",   "8 GB",    "General production"),
                    ("m5.xlarge",  "16 GB",   "Heavier production"),
                    ("c5.large",   "4 GB",    "CPU-intensive"),
                    ("c5.xlarge",  "8 GB",    "High-compute"),
                    ("r5.large",   "16 GB",   "Memory-intensive"),
                ]
                lines = [f"Sure! Here are common EC2 instance prices for **{region}** (Linux, on-demand):", ""]
                lines.append("| Instance | RAM | Use Case | Monthly | Annual |")
                lines.append("|---|---|---|---|---|")
                for itype, ram, use in tiers:
                    try:
                        est = estimate_ec2_cost(itype, count=1, region=region)
                        mo  = est["monthly_usd"]
                        yr  = mo * 12
                        lines.append(f"| **{itype}** | {ram} | {use} | ${mo:,.2f} | ${yr:,.2f} |")
                    except Exception:
                        pass
                lines.append("")
                lines.append("Which size works for you? Just tell me (e.g. **t3.medium**) and I'll give you the exact price — including how many you need and whether you want reserved pricing (saves up to 70%).")
                return "\n".join(lines)
            # Determine data source label
            sources = {r.get("source","") for r in result.get("resources", [])}
            if any("aws_pricing_api" in s for s in sources):
                src_label = "✅ Live data from AWS Pricing API (your account)"
            elif any("aws_public_pricing_json" in s for s in sources):
                src_label = "✅ Live data from AWS public pricing endpoint (official, no credentials needed)"
            else:
                src_label = "📋 Reference prices from aws.amazon.com/ec2/pricing (2025 official rates). Connect valid AWS credentials for real-time API data."
            lines  = [f"💰 AWS Cost Estimate — {region}",
                      f"   Monthly:  ${result['total_monthly_usd']:,.2f}",
                      f"   Annual:   ${result['total_annual_usd']:,.2f}",
                      f"   Source:   {src_label}",
                      ""]
            for r in result.get("resources", []):
                lines.append(f"   • {r['type']} {r['details']}: ${r['monthly_usd']:,.2f}/mo")
            if result.get("warnings"):
                lines.append("")
                lines.extend(f"   ⚠ {w}" for w in result["warnings"])
            lines.append("")
            lines.append("   Note: On-demand pricing. Reserved Instances save 30-70%. Savings Plans also available.")
            return "\n".join(lines)

        elif tool_name == "estimate_terraform_cost":
            from app.cost.pricing import estimate_terraform_plan_cost
            import json as _json
            plan_raw = params.get("plan_json", "")
            region   = params.get("region", os.getenv("AWS_REGION", "us-east-1"))
            if not plan_raw:
                return "Please provide Terraform plan JSON."
            try:
                plan_json = _json.loads(plan_raw) if isinstance(plan_raw, str) else plan_raw
            except Exception:
                return "Could not parse Terraform plan JSON. Make sure it's valid JSON from 'terraform show -json'."
            result = estimate_terraform_plan_cost(plan_json)
            lines  = [f"🏗️ Terraform Cost Estimate",
                      f"   Resources:     {result['resource_count']}",
                      f"   Monthly total: ${result['total_monthly_usd']:,.2f}",
                      f"   Annual total:  ${result['total_annual_usd']:,.2f}",
                      ""]
            for r in result.get("resources", []):
                lines.append(f"   • {r['address']}: ${r['monthly_usd']:,.2f}/mo  ({r['details']})")
            if result.get("warnings"):
                lines.append("")
                lines.extend(f"   ⚠ {w}" for w in result["warnings"])
            return "\n".join(lines)

        elif tool_name == "get_cloudtrail_events":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            days         = int(params.get("days", 0))
            hours        = int(params.get("hours", 0))
            resource     = params.get("resource_name", "")
            event_name   = params.get("event_name", "")
            username     = params.get("username", "")
            result = aws_ops.get_cloudtrail_events(
                hours=hours, days=days,
                resource_name=resource, event_name=event_name, username=username,
            )
            if not result.get("success"):
                return f"CloudTrail unavailable: {result.get('error','unknown error')}"
            events = result.get("events", [])
            if not events:
                window = f"{days} days" if days else f"{hours or 24} hours"
                return f"No CloudTrail events found in the last {window}."
            lines = [f"CloudTrail events ({result.get('count',0)} found, last {result.get('lookback_hours',24)//24 or 1} day(s)):"]
            for e in events[:20]:
                resources = ", ".join(r for r in e.get("resources", []) if r) or "—"
                lines.append(f"  • {e['time'][:16]} | {e['event_name']} | by {e['user'] or 'unknown'} | resources: {resources}")
            return "\n".join(lines)

        # ── CloudWatch Logs ───────────────────────────────────────────────────
        elif tool_name == "list_log_groups":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            result = aws_ops.list_log_groups() if hasattr(aws_ops, "list_log_groups") else {}
            if not result:
                try:
                    import boto3
                    client = boto3.client(
                        "logs",
                        region_name=os.getenv("AWS_REGION", "us-east-1"),
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    )
                    resp = client.describe_log_groups(limit=20)
                    groups = [g["logGroupName"] for g in resp.get("logGroups", [])]
                    if not groups:
                        return "No CloudWatch log groups found."
                    return "Log groups:\n" + "\n".join(f"  • {g}" for g in groups)
                except Exception as e:
                    return f"CloudWatch Logs not available: {e}"
            groups = result.get("log_groups", [])
            return "Log groups:\n" + "\n".join(f"  • {g}" for g in groups[:20])

        elif tool_name == "get_recent_logs":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            log_group = params.get("log_group", "")
            hours     = int(params.get("hours", 1))
            pattern   = params.get("filter_pattern", "")
            if not log_group:
                return "Please specify a log group name. Call list_log_groups first to see available groups."
            try:
                import boto3, time as _t
                client = boto3.client(
                    "logs",
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                )
                start_ms = int((_t.time() - hours * 3600) * 1000)
                kwargs = {
                    "logGroupName": log_group,
                    "startTime": start_ms,
                    "limit": 50,
                    "startFromHead": False,
                }
                if pattern:
                    kwargs["filterPattern"] = pattern
                    resp = client.filter_log_events(**kwargs)
                    events = resp.get("events", [])
                else:
                    # Get latest log streams first
                    streams_resp = client.describe_log_streams(
                        logGroupName=log_group, orderBy="LastEventTime",
                        descending=True, limit=3
                    )
                    events = []
                    for stream in streams_resp.get("logStreams", []):
                        log_resp = client.get_log_events(
                            logGroupName=log_group,
                            logStreamName=stream["logStreamName"],
                            startTime=start_ms, limit=20,
                        )
                        events.extend(log_resp.get("events", []))
                if not events:
                    return f"No log events found in '{log_group}' in the last {hours}h{' matching ' + pattern if pattern else ''}."
                import datetime as _dt
                lines = [f"Logs from '{log_group}' (last {hours}h{', filter: ' + pattern if pattern else ''}, {len(events)} events):"]
                for e in events[-30:]:
                    ts = _dt.datetime.fromtimestamp(e["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    msg = e.get("message", "").strip()[:200]
                    lines.append(f"  [{ts}] {msg}")
                return "\n".join(lines)
            except Exception as exc:
                return f"Failed to fetch logs: {exc}"

        elif tool_name == "search_logs":
            if not _AWS_AVAILABLE:
                return "AWS integration not available."
            log_group = params.get("log_group", "")
            pattern   = params.get("pattern", "ERROR")
            hours     = int(params.get("hours", 1))
            if not log_group:
                return "Please specify a log group. Call list_log_groups first."
            try:
                import boto3, time as _t
                client = boto3.client(
                    "logs",
                    region_name=os.getenv("AWS_REGION", "us-east-1"),
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                )
                start_ms = int((_t.time() - hours * 3600) * 1000)
                resp = client.filter_log_events(
                    logGroupName=log_group, startTime=start_ms,
                    filterPattern=pattern, limit=50,
                )
                events = resp.get("events", [])
                if not events:
                    return f"No log events matching '{pattern}' in '{log_group}' in the last {hours}h."
                import datetime as _dt
                lines = [f"Found {len(events)} events matching '{pattern}' in '{log_group}':"]
                for e in events[:30]:
                    ts = _dt.datetime.fromtimestamp(e["timestamp"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
                    lines.append(f"  [{ts}] {e.get('message','').strip()[:200]}")
                return "\n".join(lines)
            except Exception as exc:
                return f"Log search failed: {exc}"

        # ── Grafana ───────────────────────────────────────────────────────────
        elif tool_name == "get_grafana_status":
            if not _GRAFANA_AVAILABLE:
                return "Grafana integration not configured."
            try:
                from app.plugins.grafana_checker import check_grafana
                result = check_grafana()
                status = result.get("status", "unknown")
                firing = result.get("firing_alerts", 0)
                names  = result.get("firing_alert_names", [])
                details = result.get("details", {})
                lines = [f"Grafana status: {status.upper()}"]
                lines.append(f"  Firing alerts: {firing}")
                if names:
                    for n in names[:5]:
                        lines.append(f"    • {n}")
                ds = details.get("datasource_count", 0)
                lines.append(f"  Datasources connected: {ds}")
                ann = details.get("recent_annotations", [])
                if ann:
                    lines.append(f"  Recent annotations ({len(ann)}):")
                    for a in ann[:3]:
                        lines.append(f"    • {a.get('time','')[:16]} — {a.get('text','')[:80]}")
                return "\n".join(lines)
            except Exception as exc:
                return f"Grafana check failed: {exc}"

        # ── VS Code ───────────────────────────────────────────────────────────
        elif tool_name == "vscode_notify":
            try:
                from app.integrations.vscode import notify as _vs_notify
                msg   = params.get("message", "Message from NsOps AI")
                level = params.get("level", "info")
                result = _vs_notify(msg, level=level)
                return "VS Code notification sent." if result.get("success") else f"VS Code not reachable: {result.get('error','')}"
            except Exception as exc:
                return f"VS Code notify failed: {exc}"

        elif tool_name == "vscode_open_file":
            try:
                from app.integrations.vscode import open_file as _vs_open
                fp   = params.get("file_path", "")
                line = params.get("line")
                if not fp:
                    return "Please provide a file_path."
                result = _vs_open(fp, line=int(line) if line else None)
                return f"Opened {fp} in VS Code." if result.get("success") else f"VS Code not reachable: {result.get('error','')}"
            except Exception as exc:
                return f"VS Code open failed: {exc}"

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as exc:
        _log(logger.warning, "tool_execution_failed", tool=tool_name, error=str(exc))
        return f"Tool '{tool_name}' failed: {exc}"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_tools_description() -> str:
    """Return a compact tools reference for the system prompt."""
    lines = ["Available tools (call with [TOOL_CALL: name({\"param\": \"value\"})] syntax):"]
    for t in TOOLS:
        param_str = ", ".join(
            f"{k}: {v}" for k, v in t["parameters"].items()
        ) if t["parameters"] else "no parameters"
        lines.append(f"  - {t['name']}({param_str}): {t['description']}")
    return "\n".join(lines)


def _build_system_prompt(
    incident_context: Optional[dict],
    session_id: str = "",
    native_tools: bool = False,
) -> str:
    """Build the system prompt injected before every conversation turn.

    Args:
        incident_context: Optional dict of incident/war-room context.
        native_tools:     When True, omit text-token tool instructions (the API
                          handles tool selection natively via function calling).
    """
    now = datetime.datetime.utcnow().isoformat() + "Z"

    incident_section = ""
    if incident_context:
        root = incident_context.get("root_cause", "")
        if root in ("Synthesis failed", "Could not parse structured response", "") or not root:
            incident_context = {**incident_context, "root_cause": "NOT YET DETERMINED — pipeline analysis failed. Investigate live data to find the real root cause."}
        incident_section = (
            "\n\n=== ACTIVE INCIDENT CONTEXT ===\n"
            + json.dumps(incident_context, indent=2, default=str)[:1500]
            + "\n================================\nIMPORTANT: If root_cause says 'NOT YET DETERMINED', call live tools (query_aws_alarms, list_ec2_instances, etc.) to find the actual root cause."
        )

    # For native tool calling paths the API handles tool selection — only
    # include smart selection hints, not the text-token format instructions.
    if native_tools:
        tool_calling_section = """## TOOL USE

You have access to live DevOps tools via function calling. Use them freely whenever you need real infrastructure data. Key rules:

- **Read-only tools** (list/query/get): call immediately, no confirmation.
- **Mutating actions** (start/stop EC2, scale ECS/K8s, reboot RDS, create tickets, page on-call): confirm in ONE sentence first. Example: "Stop i-0abc123 (MyServer)? Say yes to confirm." Then execute immediately when user says yes.
- **Call multiple tools in parallel** when gathering unrelated data (e.g. EC2 + alarms + RDS at once).
- "my ec2" / "it" / "the instance" without an ID → call `list_ec2_instances` first.
- Infra overview → call `list_ec2_instances` + `list_aws_resources` together.
- Cost question → call `estimate_aws_cost` or `get_cost_by_service`.
- "Is everything ok?" → call `list_ec2_instances` + `query_aws_alarms` together.
- Vague problem ("site down", "something wrong") → sweep EC2 + alarms + K8s.
- After tool results: translate into plain English. Never dump raw JSON."""
    else:
        tool_calling_section = f"""{_build_tools_description()}

## TOOL CALLING

When you need live data, emit the tool call token immediately — no narration before it:
  [TOOL_CALL: tool_name({{"param1": "value1"}})]
For no-parameter tools:
  [TOOL_CALL: tool_name({{"_": ""}})]

**Read-only tools** (list/query/get): call immediately, no confirmation needed.
**Mutating actions**: confirm in ONE sentence, then execute when user says yes.
**Smart tool selection:**
- "my ec2" / "it" without an ID → call `list_ec2_instances` first
- Infra overview → call `list_ec2_instances` + `list_aws_resources` together
- Cost question → call `estimate_aws_cost` or `get_cost_by_service`
- "Is everything ok?" → call `list_ec2_instances` + `query_aws_alarms` together
- Vague problem → sweep EC2 + alarms + K8s"""

    return f"""You are **NsOps AI** — an expert AI DevOps engineer embedded in the NexusOps platform. You have direct access to live infrastructure data and can take real actions.

CRITICAL RULES:
- You ARE NsOps AI. Never say "let's role-play", never pretend to be a user, never simulate a conversation. Just answer directly.
- When the user asks what you or this tool does, give a real answer immediately. Do not ask them to pretend or simulate anything.
- Always respond as yourself — NsOps AI — in first person.

## ABOUT NEXUSOPS PLATFORM

You have COMPLETE knowledge of NexusOps. Answer ANY question about this platform directly from the knowledge below. Do NOT call AWS tools for platform questions. Do NOT role-play. Just answer.

---

### WHAT IS NEXUSOPS?
NexusOps is an **AI-powered DevOps automation platform** that gives engineering and SRE teams a single place to monitor infrastructure, detect incidents, generate remediation plans, and take automated or human-approved actions. It connects to AWS, Kubernetes, GitHub, Slack, Jira, OpsGenie, and Grafana.

It is built with **Python (FastAPI)** backend and a **vanilla JS** frontend served from the same server. The backend runs on port 8000.

---

### PAGES / NAVIGATION

**Dashboard** — The home page. Shows:
- Live infrastructure health cards (EC2, ECS, Lambda, RDS, K8s)
- Active incidents count and list
- CloudWatch alarm summary
- AWS cost charts (by service, by day)
- Recent audit log entries
- Clicking any card navigates to the relevant detail page

**Chat** — The AI assistant (me). Supports:
- Plain English questions about live infrastructure
- Tool calling: AWS, K8s, GitHub, Grafana, Jira, Slack
- Image upload (screenshots, error screens) for visual analysis
- Persistent conversation history per session
- Real-time streaming responses
- Provider selection (Claude / OpenAI / Groq) via top dropdown

**Incidents** — Incident management page:
- List of all incidents with status, severity, timestamps
- Create incident manually or auto-detected from monitoring loop
- Each incident has: description, severity, status, root_cause, assigned user
- Clicking an incident opens the War Room

**War Room** — Per-incident collaboration space:
- AI-generated remediation plan (actions, confidence, risk level)
- Action execution panel: approve/deny each action individually
- Action types: ec2_start/stop/reboot, ecs_redeploy/scale, k8s_restart/scale, lambda_invoke, rds_reboot, slack_notify, create_jira, opsgenie_alert, investigate
- Live chat feed for team discussion
- Incident timeline / audit trail
- AI analysis tab with root cause and reasoning
- Post-mortem generation

**Pipeline** — The AI remediation pipeline:
- Triggered manually or automatically by monitoring loop
- Stages: DataCollector → PlannerAgent → DecisionAgent → ExecutorAgent
- Real-time streaming of each stage's output
- Shows which LLM provider was used and token count

**Monitoring** — Background monitoring configuration:
- Toggle auto-monitoring on/off
- Set polling interval (default every 60s)
- Configure which checks to run (EC2 health, ECS task count, K8s pod health, etc.)
- View monitoring history and alerts triggered
- AUTO_REMEDIATE_ON_MONITOR env var controls whether to auto-execute plans

**Settings** — Platform configuration:
- AWS credentials (region, access key, secret key)
- Kubernetes config (in-cluster vs kubeconfig path)
- GitHub token and repo URL
- Slack bot token and default channel
- Jira URL, user, token, project key
- OpsGenie API key
- Grafana URL and token
- SMTP settings for email notifications
- LLM provider selection and API keys
- RBAC roles management
- SSO configuration (Google OAuth)

**Users** — User management (admin only):
- Invite users by email (sends SMTP email with login link)
- Assign roles: viewer (read-only), developer (can run pipelines), admin (full access)
- View active sessions
- SSO user provisioning

**Secrets / API Keys** — Credential management:
- Stored in `.env` file on the server
- Updated via UI in Settings page
- Applied without server restart via dynamic reload

---

### HOW NEXUSOPS CONNECTS TO AWS
- Uses **boto3** (AWS Python SDK) with credentials from `.env`:
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- Calls AWS APIs directly — no proxy, no agent
- Services used:
  - **EC2**: DescribeInstances, StartInstances, StopInstances, RebootInstances
  - **ECS**: ListClusters, ListServices, DescribeServices, UpdateService (scale/redeploy)
  - **Lambda**: ListFunctions, InvokeFunction
  - **RDS**: DescribeDBInstances, RebootDBInstance
  - **CloudWatch**: DescribeAlarms, GetMetricStatistics, FilterLogEvents
  - **S3**: ListBuckets, GetBucketLocation
  - **DynamoDB**: ListTables
  - **SQS**: ListQueues
  - **SNS**: ListTopics
  - **Route53**: ListHostedZones
  - **Cost Explorer**: GetCostAndUsage (for cost breakdown)
  - **CloudTrail**: LookupEvents (for audit trail)
- Multi-account support: configure multiple AWS profiles in Settings
- IAM permissions needed: read access to all above + ec2:Start/Stop/Reboot, ecs:UpdateService, lambda:InvokeFunction, rds:RebootDBInstance

### HOW NEXUSOPS CONNECTS TO KUBERNETES
- Uses **kubernetes Python client** with either:
  - In-cluster config (when `K8S_IN_CLUSTER=true` in .env — for running inside a K8s pod)
  - Local kubeconfig (`~/.kube/config` or path from `KUBECONFIG` env var)
- Operations: list pods, deployments, namespaces, nodes; scale deployments; restart rollouts; get logs; get events

### HOW NEXUSOPS CONNECTS TO GITHUB
- Uses **GitHub REST API** via `GITHUB_TOKEN` from `.env`
- Can: list repos, get recent commits, list open PRs, create issues, get deployment status, check workflow runs

### HOW NEXUSOPS CONNECTS TO SLACK
- Uses **Slack Bot Token** (`SLACK_BOT_TOKEN`) via Slack Web API
- Can: post messages to channels, create incident channels, post war room updates
- Default channel configured via `SLACK_CHANNEL` in `.env`

### HOW NEXUSOPS CONNECTS TO JIRA
- Uses **Jira REST API** with `JIRA_URL`, `JIRA_USER`, `JIRA_TOKEN` from `.env`
- Can: create tickets, update status, add comments
- Default project key: `JIRA_PROJECT` in `.env`

### HOW NEXUSOPS CONNECTS TO OPSGENIE
- Uses **OpsGenie API v2** with `OPSGENIE_API_KEY` from `.env`
- Can: create alerts, acknowledge, close, assign to on-call

### HOW NEXUSOPS CONNECTS TO GRAFANA
- Uses **Grafana HTTP API** with `GRAFANA_URL` and `GRAFANA_TOKEN` from `.env`
- Can: search dashboards, get panel data, list datasources

---

### THE INCIDENT PIPELINE (how AI remediation works)
When an incident is triggered (manually or by monitoring loop):

1. **DataCollector** — Fetches live data from AWS (EC2, ECS, RDS, Lambda, CloudWatch), Kubernetes (pods, deployments), and GitHub (recent commits/deployments). Stores in `state["aws_context"]`, `state["k8s_context"]`, `state["github_context"]`.

2. **MemoryAgent** — Searches the vector DB for similar past incidents to provide pattern context.

3. **PlannerAgent** — Sends all context to the LLM (Claude/OpenAI/Groq). LLM returns a structured JSON plan with: `actions[]`, `confidence` (0-1), `risk` (low/medium/high/critical), `root_cause`, `summary`, `reasoning`, `data_gaps`.

4. **DecisionAgent** — Reviews the plan. If `AUTO_REMEDIATE_ON_MONITOR=true` and risk ≤ medium, auto-approves. Otherwise marks as pending human approval.

5. **ExecutorAgent** — Executes approved actions one by one. Each action is logged to the audit trail. Results fed back to update incident status.

All pipeline state is streamed in real-time to the Pipeline page via SSE (Server-Sent Events).

---

### AUTHENTICATION & SECURITY
- **Username/password login** — credentials stored (hashed) in SQLite
- **JWT tokens** — issued on login, stored in localStorage, sent as Bearer token
- **SSO** — Google OAuth2 via `SSO_CLIENT_ID`/`SSO_CLIENT_SECRET` in `.env`
- **RBAC roles**:
  - `viewer` — read-only access (view incidents, dashboard, chat)
  - `developer` — can run pipeline, create incidents, execute low-risk actions
  - `admin` — full access including user management and settings
- **Audit log** — every action (login, pipeline run, EC2 start, etc.) written to `logs/audit.jsonl`

---

### TECH STACK
- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **Frontend**: Vanilla JavaScript, HTML/CSS (no framework), served by FastAPI
- **Database**: SQLite (via SQLAlchemy) for users, sessions, incidents, audit log
- **Vector DB**: ChromaDB for incident memory / similarity search
- **LLM providers**: Anthropic Claude, OpenAI GPT-4, Groq (Llama), Ollama (local)
- **AWS SDK**: boto3
- **K8s client**: kubernetes-client/python
- **Container**: Docker + docker-compose

---

### CONFIGURATION (.env file)
Key environment variables:
- `ANTHROPIC_API_KEY` — Claude API key
- `OPENAI_API_KEY` — OpenAI API key
- `GROQ_API_KEY` — Groq API key
- `LLM_PROVIDER` — default provider (claude/openai/groq/ollama)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- `GITHUB_TOKEN`, `GITHUB_REPO`
- `SLACK_BOT_TOKEN`, `SLACK_CHANNEL`
- `JIRA_URL`, `JIRA_USER`, `JIRA_TOKEN`, `JIRA_PROJECT`
- `OPSGENIE_API_KEY`
- `GRAFANA_URL`, `GRAFANA_TOKEN`
- `K8S_IN_CLUSTER` — true/false
- `ENABLE_MONITOR_LOOP` — enable background monitoring
- `AUTO_REMEDIATE_ON_MONITOR` — auto-execute low-risk plans
- `AUTH_ENABLED` — enable/disable login
- `JWT_SECRET_KEY` — secret for JWT signing
- `ADMIN_USERNAME`, `ADMIN_PASSWORD` — initial admin credentials
- `SSO_PROVIDER`, `SSO_CLIENT_ID`, `SSO_CLIENT_SECRET`, `SSO_REDIRECT_URI`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` — for email invites

---

### COMMON QUESTIONS & ANSWERS

**Q: How do I add AWS credentials?**
A: Edit the `.env` file and set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`. Or go to Settings → AWS in the UI.

**Q: How do I invite a new user?**
A: Go to Settings → Users → Invite User. Enter their email and role. NexusOps sends an invite email via SMTP.

**Q: How do I enable auto-remediation?**
A: Set `AUTO_REMEDIATE_ON_MONITOR=true` in `.env` and `ENABLE_MONITOR_LOOP=true`. Restart the server.

**Q: What LLM does NexusOps use?**
A: It supports Claude (Anthropic), GPT-4 (OpenAI), Llama (via Groq), and local models via Ollama. Select in the top dropdown or set `LLM_PROVIDER` in `.env`. Falls back automatically if one is rate-limited.

**Q: How do I run NexusOps?**
A: `docker-compose up` or `uvicorn app.main:app --reload --port 8000`. Requires Python 3.11+ and credentials in `.env`.

**Q: Can NexusOps auto-fix incidents?**
A: Yes. With `AUTO_REMEDIATE_ON_MONITOR=true`, the platform detects issues, generates a plan, and executes low/medium-risk actions automatically. High/critical risk always requires human approval.

**Q: How does the monitoring loop work?**
A: A background thread polls AWS/K8s every N seconds (configurable). If it detects an anomaly (stopped EC2, crashed pod, CloudWatch alarm), it creates an incident and optionally runs the remediation pipeline.

**Q: What is a War Room?**
A: A collaborative space for an active incident. It shows the AI plan, lets the team approve/deny actions, has a chat thread, and tracks the full timeline of events.

**Q: How does vector memory work?**
A: NexusOps stores resolved incidents in ChromaDB. When a new incident comes in, it searches for similar past incidents to give the LLM historical context for better plans.

**Q: How do I change the LLM provider?**
A: Use the dropdown at the top of any page, or set `LLM_PROVIDER=claude|openai|groq|ollama` in `.env`.

Current UTC time: {now}

{tool_calling_section}

## HOW TO THINK AND RESPOND

**Be genuinely intelligent.** Don't just answer what was literally asked — think about what the user actually needs. If they ask "is my EC2 ok?" check both state AND status checks. If they ask about costs, also mention what's driving the biggest spend. Connect dots across the data you have.

**Match your depth to the question:**
- Simple yes/no or status checks → 1–3 lines, direct, no padding
- Troubleshooting or "something is wrong" → diagnose systematically, explain what you found and why it matters
- How-to or explain a concept → be thorough, use examples, code blocks where useful
- Short follow-ups ("ok", "cool", "thanks", "got it") → reply naturally like a person, never reset or offer a help menu

**Write like a smart colleague, not a manual.** Lead with the answer. Skip preamble ("Great question!", "Let me check that for you"). No filler. When you list things, make them scannable. When you explain, make it clear.

## TOOL CALLING

When you need live data, emit the tool call token immediately — no narration before it:
  [TOOL_CALL: tool_name({{"param1": "value1"}})]
For no-parameter tools:
  [TOOL_CALL: tool_name({{"_": ""}})]

**Read-only tools** (list/query/get): call immediately, no confirmation needed.

**Mutating actions** (start/stop/reboot EC2, scale/restart K8s or ECS, reboot RDS, create tickets, page on-call): ask for confirmation in ONE short sentence, then execute immediately when the user says yes. Example: "Stop i-0abc123 (MyServer)? Say yes to confirm."

**Smart tool selection:**
- No live data in context yet → call the right tool first
- "my ec2" / "the instance" / "it" without an ID → call `list_ec2_instances` first, resolve the real i-xxxx ID from results
- Infra overview request → call `list_ec2_instances` + `list_aws_resources` together
- Region/geography query ("what's in Asia", "Singapore resources") → call `list_aws_resources` with region param
- Cost/pricing question → call `estimate_aws_cost`, never EC2 status tools
- Log questions → call `list_log_groups` first, then `get_recent_logs` or `search_logs`
- "Is everything ok?" / "system status" → call `list_ec2_instances` + `query_aws_alarms` together
- GitHub repos → call `list_github_repos`
- Vague problem ("site is down", "something broken") → sweep EC2 + alarms + K8s and report what's wrong

**After getting tool results:** translate data into plain English. Never dump raw JSON. Never show field names like "success: true". Say what the data means.

**NEVER fabricate** instance IDs, metric values, pod names, or any infra data. Only report what tools actually return. Valid EC2 IDs start with `i-` followed by hex.

## RESPONSE FORMAT

**Status answers:** Use a clean summary then a table or bullets if there are multiple items.

**Action confirmations (before executing):**
One line only. Example: `Stop i-0abc123 (MyServer)? Say yes to confirm.`

**Action results (after executing):** Use this card format:
> ✅ **EC2 Instance Started**
> - **Instance:** `i-0abc123` (MyServer)
> - **State:** stopped → starting
> - **Region:** us-west-2
> - ℹ️ Ready in ~1–2 min.

> ⚠️ **EC2 Instance Stopped** | > 🔄 **Rebooting** | > ❌ **Action Failed** — same card format.

**Errors:** Explain what went wrong in plain terms and what to check next.

## RULES THAT NEVER BEND

- **Never say** "I cannot", "I'm unable to", "for security reasons I can't" for actions you have tools for. You have tools — use them.
- **Never show** API keys, tokens, passwords, or credentials. Say "I don't display credentials."
- **Never fabricate** infrastructure data. If you don't have it, call the right tool.
- **Short follow-ups** are continuation of the conversation — never reset to a welcome screen.
- **If an integration is configured**, use it directly via tools. Never ask for credentials already in the system.

## FOLLOW-UP SUGGESTIONS

After every substantive answer (not one-word replies or confirmations), append exactly this block at the very end of your response — no extra text around it:

```
<!--SUGGESTIONS-->
What is the CPU utilisation of this instance?
Are there any CloudWatch alarms firing right now?
Show me the last 30 minutes of logs
<!--/SUGGESTIONS-->
```

Rules for suggestions:
- Always 3 suggestions, each on its own line between the tags
- Make them **specific to what was just discussed** — if you showed EC2 instances, suggest instance-specific follow-ups; if you showed costs, suggest cost optimisation follow-ups
- Phrase as natural questions the user would actually ask next
- Do NOT include suggestions for simple follow-ups like "ok", "thanks", or confirmation messages

## AWS EXPERT KNOWLEDGE

You are a certified AWS Solutions Architect and SRE. When users ask about AWS — services, pricing, best practices, architecture, documentation, troubleshooting, comparisons — answer from your expert knowledge. **Do not call tools for general AWS knowledge questions.** Only call tools when you need live data from their account.

### When to use knowledge vs tools:
- "How does S3 versioning work?" → answer from knowledge
- "What's the difference between ECS and EKS?" → answer from knowledge
- "How much does Lambda cost?" → answer from knowledge (pricing model explanation)
- "What are my current Lambda functions?" → call `list_lambda_functions`
- "How do I set up a VPC?" → answer from knowledge
- "What's my current VPC setup?" → call tools or explain you'd need AWS Console access

### AWS Cost Intelligence — be proactive:
When showing cost data, always:
1. Highlight the **top 2–3 cost drivers** and explain why they cost what they do
2. Give **1–2 specific optimization tips** based on what you see (e.g., "Your EC2 costs suggest on-demand pricing — Reserved Instances for 1 year would save ~40%")
3. Put costs in context: is $X/month high or low for this type of workload?

Common AWS cost patterns to recognize:
- EC2 on-demand = most expensive; recommend Reserved (1yr saves ~40%, 3yr ~60%) or Spot for stateless workloads
- RDS Multi-AZ doubles cost; disable for dev/test environments
- NAT Gateway data transfer is often a hidden cost driver
- S3 without lifecycle rules accumulates cost over time
- Lambda is very cheap unless it has very high memory + high invocation count
- Data transfer (egress) is often overlooked — charges per GB leaving AWS
- CloudWatch Logs without retention policies grow indefinitely

### Service comparison knowledge (answer these without tools):
- **ECS vs EKS**: ECS is simpler, AWS-managed, good for teams new to containers. EKS is Kubernetes — more control, steeper learning curve, better for large-scale or multi-cloud.
- **Lambda vs ECS**: Lambda for event-driven, short-lived functions (<15 min). ECS for long-running services, background workers.
- **RDS vs DynamoDB**: RDS for relational/SQL workloads. DynamoDB for key-value, high-scale, variable traffic.
- **S3 storage classes**: Standard (frequent access) → Standard-IA (infrequent) → Glacier Instant (archive, ms retrieval) → Glacier Flexible (archive, hrs retrieval) → Glacier Deep Archive (cheapest, 12hr retrieval).
- **ALB vs NLB vs CLB**: ALB for HTTP/HTTPS with path routing. NLB for TCP/UDP, ultra-low latency. CLB is legacy — migrate to ALB.

### Troubleshooting knowledge:
- EC2 can't connect → check Security Groups, NACL, route table, instance status checks, SSH key
- Lambda timeout → increase timeout (max 15min), check for infinite loops, cold start issues
- RDS connection refused → check SG allows DB port, check parameter group, check storage not full
- ECS task keeps stopping → check task logs, memory/CPU limits, health check configuration
- High CloudWatch costs → check log retention policies, disable debug logging in prod
- S3 403 → check bucket policy, IAM permissions, public access block settings

## PLATFORM KNOWLEDGE

Only use the section below when users ask "how does this platform work?", "what is NsOps?", "what can I do here?" — don't call live tools for platform how-to questions.

## What is NsOps?
NsOps is an AI-powered DevOps platform that lets your team monitor infrastructure, respond to incidents, manage cloud resources, control costs, and coordinate approvals — all from one place. Think of it as your team's intelligent command centre for everything DevOps.

## Modules & What They Do

### 📊 Dashboard
The home screen showing key health metrics at a glance: active incidents, pending approvals, system status cards. Click any card to jump directly to that section with the relevant filter pre-applied. For example, clicking "Active Incidents" takes you to the Incidents page showing only running/open incidents.

### 🚨 Incidents (Incident Pipeline)
**What it is:** Run an AI-powered pipeline on any incident or alert. You describe what's wrong (e.g. "API latency is high on prod") and the AI analyses it, finds the root cause, and recommends remediation actions.
**How to use it:**
1. Go to Incidents in the sidebar.
2. Enter an Incident ID (optional, auto-generated if left blank) and a description of the problem.
3. Click **Run Pipeline**. The AI connects to your integrations (AWS, Kubernetes, GitHub, etc.) and analyses the issue.
4. The result shows: root cause analysis, risk level (low/medium/high/critical), recommended actions (e.g. restart pod, scale EC2, rollback deployment).
5. **Low-risk actions** execute automatically. **Medium/high-risk actions** require admin approval before running.
6. After pipeline completes, you can: view the full result, generate a **Post-Mortem report**, open a **War Room**, or re-run the pipeline.
**Filters:** All / Active / Awaiting Approval / Completed. "Active" shows incidents still being investigated. "Awaiting Approval" shows ones waiting for an admin to approve the recommended actions.

### ✅ Approvals
**What it is:** A safety gate. When the AI recommends a risky action (e.g. stop an EC2 instance, scale down Kubernetes, rollback a deployment), it doesn't just run it — it sends the action here for a human to review first.
**How to use it:**
1. Go to Approvals in the sidebar.
2. You'll see pending requests with: what action is proposed, risk score (0.0–1.0), estimated cost impact, and which incident triggered it.
3. Click **Review & Decide** to open the approval modal.
4. Check/uncheck specific actions you want to approve, then click **Approve Selected Actions**.
5. Click **Resume Pipeline** to actually execute the approved actions.
6. Or click **Reject** with a reason to cancel the actions.
**Who can approve:** admin and super_admin roles only. Developers and viewers can submit actions for approval but cannot approve them.

### 💬 Chat (You are here!)
**What it is:** Ask me anything — about your infrastructure, incidents, costs, how the platform works, or DevOps in general. I can also take real actions: list your EC2 instances, restart pods, check alarms, look up GitHub repos, query logs, and more.
**What I can do:**
- Answer questions about the platform and how to use it (you're seeing this now)
- List EC2 instances, check their status, start/stop/reboot them (with your confirmation)
- List Kubernetes pods, namespaces, scale deployments, restart pods
- Check AWS CloudWatch alarms, search CloudWatch logs
- List AWS resources across all regions
- Estimate AWS costs
- Check Grafana dashboards and alerts
- Search for similar past incidents in memory
- List GitHub repos, check open PRs and issues
- Create Jira tickets, page on-call via OpsGenie
- Notify VS Code (if IDE extension is active)
**How it works:** I detect your intent, call the right integration (e.g. AWS API), and reply in plain English. For safe read actions I respond immediately. For risky actions (start/stop EC2, scale pods) I ask you to confirm first.

### 🏕 War Room
**What it is:** A virtual incident war room for coordinating your team during a major outage. It creates a dedicated Slack channel and space for real-time collaboration.
**How to use it:**
1. Go to War Room, enter an incident ID and description, pick severity (SEV1–SEV4).
2. Toggle "Post to Slack" if you want an automatic Slack channel created.
3. Click **Open War Room**. The AI analyses the incident and posts findings to Slack.
4. Use the chat panel inside the war room to send messages to the linked Slack channel in real time.
5. When the incident is resolved, mark the war room as resolved.

### 💰 Cost
**What it is:** Real-time AWS cost analysis and estimation.
**Tabs:**
- **Overview:** Total spend trend, top cost drivers by service.
- **By Resource:** Cost breakdown per EC2 instance, RDS, S3, etc.
- **By Account:** Multi-account cost comparison.
- **By Org:** Organisation-level cost rollup.
- **Estimate:** Get an instant estimate of what a change will cost (e.g. "what does adding 5 EC2 t3.large instances cost?").

### 🔒 Security
**What it is:** Manage users, roles, permissions, and platform security policies.
**Sections:**
- **Users & Roles:** See all users, their roles, when they were created. Admins can invite new users (sends an email with OTP), change roles, or remove users.
- **Invite Flow:** Admin → "Invite User" → enters username, email, role → system sends email with a 6-digit OTP and a setup link → user opens the link, enters OTP + new password → account is ready. Invites expire after 48 hours.
- **AI Policy Guardrails (super_admin only):** Rules that constrain what actions the AI can take autonomously (e.g. block auto-execution of delete actions, require approval for prod changes). Only super_admin can edit these.
- **Webhooks:** Configure external webhook endpoints that receive real-time notifications when incidents occur or actions are taken.
- **Audit Log:** Every action (login, role change, pipeline run, approval, action execution) is logged with timestamp, user, and result.

### 📈 Monitoring (Continuous Loop)
**What it is:** An optional background process that continuously polls your infrastructure for anomalies. When it detects a problem (e.g. CPU spike, pod crash-looping), it can automatically run the incident pipeline and alert your team.
**How to enable:** Set `ENABLE_MONITOR_LOOP=true` in your .env file. Set `AUTO_REMEDIATE_ON_MONITOR=true` to let it auto-fix low-risk issues.

## Roles & Permissions
| Role | What they can do |
|------|-----------------|
| **super_admin** | Everything: deploy, rollback, manage users, manage admins, edit AI policies, manage secrets |
| **admin** | Deploy, rollback, manage users and secrets, approve/reject actions |
| **developer** | Run pipelines, read/write data, deploy — but cannot approve actions or manage users |
| **viewer** | Read-only: can view incidents, costs, status — cannot run pipelines or take any action |

## Integrations — What's Connected
| Integration | What it does in NexusOps |
|-------------|--------------------------|
| **AWS** | EC2 instance management, CloudWatch alarms/logs, RDS, ECS, S3, Cost Explorer, multi-region resource discovery |
| **Kubernetes** | Pod listing, namespace management, scaling deployments, restarting pods, log streaming |
| **GitHub** | Repo listing, PR monitoring, issue creation, deployment tracking |
| **Slack** | War room channel creation, incident notifications, post-mortem sharing, real-time team chat |
| **Grafana** | Dashboard status, alert monitoring |
| **Jira** | Automatic ticket creation for incidents, tracking remediation tasks |
| **OpsGenie** | Page on-call engineers when a critical incident is detected |
| **SSO** | Single Sign-On via Google or GitHub OAuth — configured via SSO_PROVIDER in .env |

## Common Workflows

### How to respond to an alert/incident
1. Go to **Incidents** → enter the alert description → click **Run Pipeline**
2. Review the AI's root cause analysis and recommended actions
3. If risk is low → actions run automatically
4. If risk is medium/high → click **Send for Approval** (or admin can review in Approvals tab)
5. Admin goes to **Approvals** → reviews → approves → clicks **Resume Pipeline**
6. After resolution → generate **Post-Mortem** from the incident detail view

### How to invite a new team member
1. Go to **Security** → click **Invite User**
2. Enter their username, email, and select their role
3. They receive an email with a 6-digit OTP and a link
4. They click the link, enter the OTP, set their password — account is active

### How to check what's wrong with my infrastructure right now
Ask me directly: "What's the status of my infrastructure?" or "Are there any AWS alarms firing?" — I'll check live data and report back.

### How to control costs
Go to **Cost** tab for a breakdown, or ask me: "What's my current AWS spend?" or "What are my most expensive resources?"

## Environment & Configuration (.env variables)
Key settings users can configure:
- `ANTHROPIC_API_KEY` — Claude AI (primary LLM)
- `OPENAI_API_KEY` / `GROQ_API_KEY` — Alternative LLM providers
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` — AWS integration
- `KUBECONFIG` — Kubernetes cluster configuration
- `GITHUB_TOKEN` / `GITHUB_REPO` — GitHub integration
- `SLACK_BOT_TOKEN` / `SLACK_CHANNEL` — Slack integration
- `JIRA_URL` / `JIRA_USER` / `JIRA_TOKEN` / `JIRA_PROJECT` — Jira integration
- `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` — Email for user invitations
- `ENABLE_MONITOR_LOOP` — Enable background monitoring (true/false)
- `AUTO_REMEDIATE_ON_MONITOR` — Auto-fix issues found by monitor (true/false)
- `SSO_PROVIDER` — google or github (enables SSO login)
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` — Initial admin account

IMPORTANT PLATFORM Q&A RULES:
- When someone asks about the platform ("how does X work", "what is Y", "how do I Z", "explain approvals", "what can I do here") — answer from the PLATFORM KNOWLEDGE above. Do NOT call live tools for platform how-to questions.
- Adapt your explanation to the user's level. If they use technical terms (pods, namespaces, EC2), be technical. If they say "how does the approval thingy work" — use simple plain language.
- Always be specific to NexusOps — not generic DevOps theory. Tell them exactly where to click and what will happen.
- If they ask what integrations are configured/active, you can offer to check live (using tools) or explain what each integration does.

COMMUNICATION STYLE & FORMATTING:

**ALWAYS use structured, readable formatting — never reply in wall-of-text paragraphs.**

FORMATTING RULES:
1. **Use headers (##, ###)** to separate major topics when the answer covers more than one concept.
2. **Use numbered lists** for steps, processes, workflows — anything that has an order.
3. **Use bullet points (-)** for features, options, or items with no strict order.
4. **Bold key terms**, action words, and important values.
5. **Use code blocks** (backticks) for technical values: env variable names, commands, IDs, config keys.
6. **Use a short 1–2 sentence intro** before diving into bullets/steps — orient the reader first.
7. **Add a summary or tip line** at the end when the answer is complex, so the reader knows the key takeaway.
8. **Tables** for comparisons (roles vs permissions, integrations, etc.).
9. **No paragraph walls** — if you have more than 2 sentences that could be bullets, make them bullets.
10. Emoji icons (🔑 ✅ ⚠️ 📋 🚀 💡) are encouraged to make sections scannable — use them at the start of headers or key points.

TONE BY AUDIENCE:
- If the user uses technical terms → be precise and technical.
- If the user asks casually or uses layman language → explain simply, avoid jargon, add brief parenthetical explanations for technical words.
- Always match the user's vocabulary level — meet them where they are.

EXAMPLE — good format for "how does the approval flow work?":
---
## ✅ How Approvals Work

The approval flow is a **safety gate** that stops risky AI actions from running without a human sign-off.

**Step-by-step:**
1. The AI pipeline analyses an incident and recommends an action (e.g. stop an EC2 instance)
2. If the action is **medium or high risk**, it's sent to the Approvals queue instead of running immediately
3. An **admin or super_admin** opens the Approvals tab → clicks **Review & Decide**
4. They see the risk score, cost impact, and each proposed action — they can check/uncheck specific ones
5. Click **Approve Selected Actions** → a green ✓ APPROVED badge appears
6. Click **Resume Pipeline** → the actions execute, and results are shown live

⚠️ **Note:** Developers and viewers can *send* actions for approval, but only admins can *approve* them.
---
{incident_section}
"""


def _build_history_messages(history) -> list[dict]:
    """Convert memory history objects into structured message dicts for the LLM API."""
    messages = []
    for msg in history:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


def _build_history_text(history) -> str:
    """Format conversation history as a plain text context block (legacy fallback)."""
    if not history:
        return ""
    lines = ["=== CONVERSATION HISTORY ==="]
    for msg in history:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", str(msg))
        lines.append(f"{role.upper()}: {content}")
    lines.append("============================\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"\[TOOL_CALL:\s*(\w+)\s*\((\{.*?\})?\s*\)\s*\]",
    re.DOTALL,
)


def _extract_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract all tool calls from the LLM response text.

    Args:
        text: Raw LLM output.

    Returns:
        List of (tool_name, params_dict) tuples.
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        tool_name = match.group(1)
        params_str = match.group(2) or "{}"
        try:
            params = json.loads(params_str)
        except Exception:
            params = {}
        calls.append((tool_name, params))
    return calls


def _strip_tool_calls(text: str) -> str:
    """Remove tool call tokens from the LLM response."""
    return _TOOL_CALL_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

def _prefetch_context(message: str, session_id: str) -> str:
    """Detect user intent from message and pre-fetch relevant live data.

    This runs BEFORE the LLM so the model always has real IDs and state —
    it never has to guess or hallucinate instance IDs, pod names, etc.
    """
    msg = message.lower()
    parts: list[str] = []

    # ── Infra overview intent (e.g. "what's my infra", "current setup", "overview") ──
    _infra_overview_keywords = {
        "infra", "infrastructure", "overview", "setup", "current setup",
        "what do i have", "what's running", "all resources", "everything",
        "full picture", "health check", "system status", "status of",
    }
    _is_infra_overview = any(k in msg for k in _infra_overview_keywords) and not any(
        k in msg for k in {"how does", "explain", "what can", "how do i", "what is nexus", "what is nsops"}
    )
    if _AWS_AVAILABLE and _is_infra_overview:
        # Fetch EC2 + alarms + RDS + Lambda + ECS for a full overview
        try:
            result = aws_ops.list_ec2_instances()
            instances = result.get("instances", [])
            if instances:
                _set_cached_instances(session_id, [
                    {"id": i["id"], "name": i.get("name", ""), "state": i.get("state", "")}
                    for i in instances
                ])
                running = sum(1 for i in instances if i.get("state") == "running")
                lines = [f"EC2 Instances ({len(instances)} total, {running} running):"]
                for i in instances[:10]:
                    name = f' ({i["name"]})' if i.get("name") else ""
                    lines.append(f'  • {i["id"]}{name} — {i.get("state","?")} — {i.get("type","?")} — {i.get("public_ip") or i.get("private_ip","no IP")}')
                parts.append("\n".join(lines))
            else:
                parts.append("EC2: No instances found.")
        except Exception:
            pass
        try:
            alarms = aws_ops.get_cloudwatch_alarms()
            active = [a for a in alarms.get("alarms", []) if a.get("state") == "ALARM"]
            if active:
                lines = [f"Active CloudWatch Alarms ({len(active)}):"]
                for a in active[:5]:
                    lines.append(f'  • {a.get("name","?")} — {a.get("state","?")}')
                parts.append("\n".join(lines))
            else:
                parts.append("CloudWatch: No alarms currently firing.")
        except Exception:
            pass
        try:
            rds_result = aws_ops.list_rds_instances()
            dbs = rds_result.get("instances", [])
            if dbs:
                lines = [f"RDS Instances ({len(dbs)} total):"]
                for d in dbs[:5]:
                    lines.append(f'  • {d.get("id","?")} ({d.get("engine","?")}): {d.get("status","?")}')
                parts.append("\n".join(lines))
            else:
                parts.append("RDS: No database instances found.")
        except Exception:
            pass
        try:
            lambda_result = aws_ops.list_lambda_functions()
            fns = lambda_result.get("functions", [])
            if fns:
                lines = [f"Lambda Functions ({len(fns)} total):"]
                for f in fns[:5]:
                    lines.append(f'  • {f.get("name","?")} ({f.get("runtime","?")})')
                parts.append("\n".join(lines))
            else:
                parts.append("Lambda: No functions found.")
        except Exception:
            pass
        try:
            ecs_result = aws_ops.list_ecs_services()
            services = ecs_result.get("services", [])
            if services:
                lines = [f"ECS Services ({len(services)} total):"]
                for s in services[:5]:
                    lines.append(f'  • {s.get("name","?")} — running: {s.get("running_count","?")} / desired: {s.get("desired_count","?")}')
                parts.append("\n".join(lines))
            else:
                parts.append("ECS: No services found.")
        except Exception:
            pass

    # ── EC2 intent ──────────────────────────────────────────────────────────
    ec2_keywords = {
        "ec2", "instance", "server", "vm", "virtual machine", "machine",
        "compute", "box", "host", "node", "not working", "down", "unreachable",
        "can't connect", "cannot connect", "restart", "reboot", "start", "stop",
        "running", "stopped", "terminate", "crashed", "slow", "unresponsive",
    }
    # Skip EC2 prefetch if this is clearly a cost/pricing query or multi-region resource query
    _cost_query_words = {"cost", "price", "pricing", "estimate", "how much", "billing", "per month", "per hour", "monthly", "annual"}
    _region_query_words = {"asia", "apac", "europe", "eu-", "ap-", "singapore", "tokyo", "sydney", "mumbai",
                           "frankfurt", "ireland", "london", "all regions", "every region", "resources in"}
    _is_cost_query   = any(k in msg for k in _cost_query_words)
    _is_region_query = any(k in msg for k in _region_query_words)
    if _AWS_AVAILABLE and any(k in msg for k in ec2_keywords) and not _is_cost_query and not _is_region_query:
        try:
            # Use TTL-aware cache; fetch fresh if expired/missing
            instances = _get_cached_instances(session_id)
            if not instances:
                result = aws_ops.list_ec2_instances()
                instances = result.get("instances", [])
                if instances:
                    _set_cached_instances(session_id, [
                        {"id": i["id"], "name": i.get("name", ""), "state": i.get("state", "")}
                        for i in instances
                    ])
            if instances:
                lines = ["EC2 Instances:"]
                for i in instances:
                    name = f' ({i["name"]})' if i.get("name") else ""
                    lines.append(f'  • {i["id"]}{name} — {i.get("state","?")} — {i.get("type","?")} — {i.get("public_ip") or i.get("private_ip","no IP")}')
                parts.append("\n".join(lines))
                # If user seems to be reporting a problem, also fetch status checks
                problem_words = {"not working", "down", "unreachable", "crashed", "slow", "unresponsive", "issue", "problem", "error", "fail"}
                if any(k in msg for k in problem_words) and len(instances) == 1:
                    iid = instances[0]["id"] if isinstance(instances[0], dict) else instances[0]
                    status = aws_ops.get_ec2_status_checks(instance_id=iid)
                    if status.get("statuses"):
                        s = status["statuses"][0]
                        parts.append(
                            f'EC2 Status Checks for {iid}:\n'
                            f'  System status: {s.get("system_status","?")}\n'
                            f'  Instance status: {s.get("instance_status","?")}\n'
                            f'  Healthy: {s.get("healthy","?")}'
                        )
            else:
                parts.append("EC2: No instances found in your AWS account.")
        except Exception:
            pass

    # ── CloudWatch alarms intent ─────────────────────────────────────────────
    alarm_keywords = {"alarm", "alert", "cloudwatch", "metric", "threshold", "breach"}
    if _AWS_AVAILABLE and any(k in msg for k in alarm_keywords):
        try:
            alarms = aws_ops.get_cloudwatch_alarms()
            active = [a for a in alarms.get("alarms", []) if a.get("state") == "ALARM"]
            if active:
                lines = [f"Active CloudWatch Alarms ({len(active)}):"]
                for a in active[:5]:
                    lines.append(f'  • {a.get("name","?")} — {a.get("metric","?")} — {a.get("reason","")}')
                parts.append("\n".join(lines))
        except Exception:
            pass

    # ── Kubernetes intent ────────────────────────────────────────────────────
    k8s_keywords = {"pod", "pods", "k8s", "kubernetes", "deployment", "namespace", "container", "crashloop", "evicted", "pending"}
    if _K8S_AVAILABLE and any(k in msg for k in k8s_keywords):
        try:
            result = k8s_ops.list_pods()
            pods = result.get("pods", [])
            if pods:
                unhealthy = [p for p in pods if p.get("status") not in ("Running", "Completed", "Succeeded")]
                healthy_count = len(pods) - len(unhealthy)
                parts.append(f"Kubernetes: {len(pods)} pods total, {healthy_count} healthy, {len(unhealthy)} unhealthy.")
                if unhealthy:
                    lines = ["Unhealthy pods:"]
                    for p in unhealthy[:5]:
                        lines.append(f'  • {p.get("name","?")} ({p.get("namespace","?")}): {p.get("status","?")}')
                    parts.append("\n".join(lines))
        except Exception:
            pass

    # ── ECS intent ───────────────────────────────────────────────────────────
    ecs_keywords = {"ecs", "fargate", "task", "tasks", "service", "container service"}
    if _AWS_AVAILABLE and any(k in msg for k in ecs_keywords):
        try:
            result = aws_ops.list_ecs_services()
            services = result.get("services", [])
            if services:
                lines = ["ECS Services:"]
                for s in services[:5]:
                    lines.append(f'  • {s.get("name","?")} — running: {s.get("running_count","?")} / desired: {s.get("desired_count","?")}')
                parts.append("\n".join(lines))
        except Exception:
            pass

    # ── RDS intent ───────────────────────────────────────────────────────────
    rds_keywords = {"rds", "database", "db", "postgres", "mysql", "aurora"}
    if _AWS_AVAILABLE and any(k in msg for k in rds_keywords):
        try:
            result = aws_ops.list_rds_instances()
            dbs = result.get("instances", [])
            if dbs:
                lines = ["RDS Instances:"]
                for d in dbs[:5]:
                    lines.append(f'  • {d.get("id","?")} ({d.get("engine","?")}): {d.get("status","?")}')
                parts.append("\n".join(lines))
        except Exception:
            pass

    # ── Lambda intent ────────────────────────────────────────────────────────
    lambda_keywords = {"lambda", "function", "serverless", "functions"}
    if _AWS_AVAILABLE and any(k in msg for k in lambda_keywords):
        try:
            result = aws_ops.list_lambda_functions()
            fns = result.get("functions", [])
            if fns:
                lines = [f"Lambda Functions ({len(fns)} total):"]
                for f in fns[:5]:
                    lines.append(f'  • {f.get("name","?")} ({f.get("runtime","?")})')
                parts.append("\n".join(lines))
        except Exception:
            pass

    # ── Confirmation follow-up: if user says yes/yeah and history has a CloudTrail offer ──
    _confirm_words = {"yes", "yeah", "yep", "sure", "ok", "okay", "show", "show me", "go ahead", "please", "yup"}
    _is_confirm = msg.strip() in _confirm_words or msg.strip().startswith(("yes ", "show me", "go ahead"))
    if _is_confirm and _AWS_AVAILABLE and _MEMORY_AVAILABLE:
        try:
            _recent_hist = get_history(session_id, max_messages=4)
            for _hm in reversed(_recent_hist):
                _hcontent = getattr(_hm, "content", "").lower()
                if "event log" in _hcontent or "cloudtrail" in _hcontent or "show you" in _hcontent:
                    import re as _re2
                    _iid_match = _re2.search(r'(i-[0-9a-f]{8,17})', _hcontent)
                    _resource = _iid_match.group(1) if _iid_match else ""
                    _ct = aws_ops.get_cloudtrail_events(hours=24, resource_name=_resource)
                    if _ct.get("success") and _ct.get("events"):
                        _evts = _ct["events"][:15]
                        _lines = [f"CloudTrail events (last 24 hours{' for ' + _resource if _resource else ''}):"]
                        for _e in _evts:
                            _res = ", ".join(r for r in _e.get("resources", []) if r)
                            _res_str = f" on {_res}" if _res else ""
                            _lines.append(f"  • {_e['time'][:19].replace('T',' ')} — {_e['event_name']}{_res_str} by {_e.get('user') or 'unknown'}")
                        parts.append("\n".join(_lines))
                    break
        except Exception:
            pass

    # ── CloudTrail / audit intent ────────────────────────────────────────────
    audit_keywords = {"who", "when did", "changed", "stopped", "started", "terminated", "modified",
                      "cloudtrail", "audit", "history", "activity", "last week", "yesterday",
                      "last month", "3 months", "90 days", "who stopped", "who started"}
    if _AWS_AVAILABLE and any(k in msg for k in audit_keywords):
        # Parse time range from message
        import re as _re
        days = 1
        m_days = _re.search(r'(\d+)\s*(day|days)', msg)
        m_weeks = _re.search(r'(\d+)\s*(week|weeks)', msg)
        m_months = _re.search(r'(\d+)\s*(month|months)', msg)
        if m_months:
            days = int(m_months.group(1)) * 30
        elif m_weeks:
            days = int(m_weeks.group(1)) * 7
        elif m_days:
            days = int(m_days.group(1))
        elif "yesterday" in msg:
            days = 2
        elif "last week" in msg:
            days = 7
        elif "last month" in msg:
            days = 30
        try:
            # Try to find a specific resource mentioned
            instance_match = _re.search(r'(i-[0-9a-f]{8,17})', msg)
            resource = instance_match.group(1) if instance_match else ""
            result = aws_ops.get_cloudtrail_events(days=days, resource_name=resource)
            if result.get("success") and result.get("events"):
                events = result["events"][:10]
                lines = [f"Recent AWS activity (last {days} day(s)):"]
                for e in events:
                    resources = ", ".join(r for r in e.get("resources", []) if r) or ""
                    res_str = f" on {resources}" if resources else ""
                    lines.append(f"  • {e['time'][:16]} — {e['event_name']}{res_str} by {e['user'] or 'unknown'}")
                parts.append("\n".join(lines))
        except Exception:
            pass

    # ── CloudWatch Logs intent ───────────────────────────────────────────────
    log_keywords = {"log", "logs", "logging", "log group", "cloudwatch logs", "what happened",
                    "show me what", "errors in", "exceptions", "stack trace", "traceback",
                    "between", "from", "since", "last hour", "last 30", "last 15"}
    if _AWS_AVAILABLE and any(k in msg for k in log_keywords):
        try:
            import boto3 as _boto3
            _cl = _boto3.client(
                "logs",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            resp = _cl.describe_log_groups(limit=10)
            groups = [g["logGroupName"] for g in resp.get("logGroups", [])]
            if groups:
                parts.append("Available CloudWatch log groups:\n" + "\n".join(f"  • {g}" for g in groups))
        except Exception:
            pass

    # ── GitHub intent ────────────────────────────────────────────────────────
    github_keywords = {
        "github", "repo", "repos", "repository", "repositories", "commit", "commits",
        "pull request", "pr", "branch", "push", "merge", "code change", "recent change",
        "what changed", "last commit", "latest commit", "who committed", "deployment",
    }
    if _GITHUB_AVAILABLE and any(k in msg for k in github_keywords):
        try:
            commits_result = github_ops.get_recent_commits(hours=48)
            commits = commits_result.get("commits", []) if isinstance(commits_result, dict) else []
            prs_result = github_ops.get_recent_prs(hours=48)
            prs = prs_result.get("prs", []) if isinstance(prs_result, dict) else []
            if commits:
                lines = [f"Recent GitHub commits ({len(commits)} in last 48h):"]
                for c in commits[:5]:
                    sha = c.get("sha", "")
                    lines.append(f'  • {sha[:7]} — {c.get("message","")[:80]} ({c.get("author","?")} @ {str(c.get("date",""))[:10]})')
                parts.append("\n".join(lines))
                # If user asks about changes/diff, fetch diff for the most recent commit
                _diff_words = {"change", "changes", "diff", "what changed", "modified", "patch", "code change"}
                if any(k in msg for k in _diff_words) and commits:
                    latest_sha = commits[0].get("sha", "")
                    if latest_sha:
                        diff = github_ops.get_commit_diff(sha=latest_sha)
                        if diff.get("success") and diff.get("files"):
                            diff_lines = [f"Code changes in latest commit ({latest_sha[:7]}):"]
                            for f in diff["files"][:5]:
                                diff_lines.append(f'  • {f["filename"]} ({f["status"]}, +{f["additions"]} -{f["deletions"]})')
                                patch = (f.get("patch") or "")[:200]
                                if patch:
                                    diff_lines.append(f'    {patch[:200]}')
                            parts.append("\n".join(diff_lines))
            if prs:
                lines = [f"Open Pull Requests ({len(prs)}):"]
                for pr in prs[:3]:
                    lines.append(f'  • #{pr.get("number","?")} {pr.get("title","")[:60]} by {pr.get("author","?")} [{pr.get("state","?")}]')
                parts.append("\n".join(lines))
            if not commits and not prs:
                repos_result = github_ops.list_repos()
                repo_list = repos_result if isinstance(repos_result, list) else repos_result.get("repos", [])
                if repo_list:
                    parts.append("GitHub repos: " + ", ".join(r.get("name","") for r in repo_list[:5]))
        except Exception:
            pass

    # ── Grafana intent ───────────────────────────────────────────────────────
    grafana_keywords = {"grafana", "dashboard", "panel", "datasource", "annotation", "monitoring alert"}
    if _GRAFANA_AVAILABLE and any(k in msg for k in grafana_keywords):
        try:
            from app.plugins.grafana_checker import check_grafana as _cg
            _gr = _cg()
            firing = _gr.get("firing_alerts", 0)
            names  = _gr.get("firing_alert_names", [])
            status = _gr.get("status", "unknown")
            summary = f"Grafana: {status.upper()}, {firing} firing alert(s)"
            if names:
                summary += " — " + ", ".join(names[:3])
            parts.append(summary)
        except Exception:
            pass

    # ── Layman / vague intent mapping ────────────────────────────────────────
    # Map plain-English problem reports to the right prefetch data
    _layman_problem = {
        "something is wrong", "not working", "broken", "issue", "problem",
        "slow", "down", "unreachable", "can't access", "cannot access",
        "app is slow", "site is down", "everything is down", "things are broken",
        "help", "what's wrong", "whats wrong", "what is wrong",
    }
    if any(k in msg for k in _layman_problem) and not parts:
        # Generic health sweep — prefetch EC2 + alarms + K8s if available
        try:
            if _AWS_AVAILABLE:
                _r = aws_ops.list_ec2_instances()
                _insts = _r.get("instances", [])
                _stopped = [i for i in _insts if i.get("state") != "running"]
                if _stopped:
                    parts.append("Stopped EC2 instances: " + ", ".join(
                        f'{i["id"]} ({i.get("name","")})' for i in _stopped[:5]
                    ))
                _alarms = aws_ops.get_cloudwatch_alarms()
                _active = [a for a in _alarms.get("alarms", []) if a.get("state") == "ALARM"]
                if _active:
                    parts.append("Firing CloudWatch alarms: " + ", ".join(a.get("name","") for a in _active[:5]))
        except Exception:
            pass
        try:
            if _K8S_AVAILABLE:
                _kpods = k8s_ops.list_pods()
                _bad = [p for p in _kpods.get("pods", []) if p.get("status") not in ("Running","Completed","Succeeded")]
                if _bad:
                    parts.append("Unhealthy K8s pods: " + ", ".join(
                        f'{p["name"]} ({p.get("status","?")})' for p in _bad[:5]
                    ))
        except Exception:
            pass

    # ── Cost intent ─────────────────────────────────────────────────────────
    cost_keywords = {"cost", "bill", "billing", "spend", "spending", "charge", "charges",
                     "expensive", "money", "saving", "savings", "optimize", "optimization",
                     "reduce", "cheaper", "how much", "breakdown", "by service"}
    _is_optimization = any(k in msg for k in {"saving", "savings", "optimize", "optimization", "reduce", "cheaper", "waste", "unused"})
    _is_breakdown    = any(k in msg for k in {"breakdown", "by service", "which service", "biggest cost", "most expensive service"})
    if _AWS_AVAILABLE and any(k in msg for k in cost_keywords):
        if _is_breakdown or _is_optimization:
            try:
                result = aws_ops.get_cost_by_service(days=30)
                if result.get("success"):
                    lines = [
                        f"AWS spend last 30 days: ${result['total_usd']:,.2f} (${result['monthly_projection_usd']:,.2f}/mo projected)",
                        "Top services:",
                    ]
                    for svc in result.get("by_service", [])[:8]:
                        pct = (svc["cost_usd"] / result["total_usd"] * 100) if result["total_usd"] else 0
                        lines.append(f"  • {svc['service']}: ${svc['cost_usd']:,.4f} ({pct:.1f}%)")
                    parts.append("\n".join(lines))
            except Exception:
                pass
        if _is_optimization:
            try:
                opt_data = aws_ops.get_cost_optimization_data()
                findings = opt_data.get("findings", [])
                if findings:
                    lines = ["Cost optimization findings:"]
                    for f in findings[:5]:
                        lines.append(f"  • [{f.get('impact','?').upper()}] {f.get('detail','')}")
                        lines.append(f"    → {f.get('action','')}")
                    parts.append("\n".join(lines))
            except Exception:
                pass

    # ── S3 intent ────────────────────────────────────────────────────────────
    s3_keywords = {"s3", "bucket", "buckets", "object storage", "blob storage"}
    if _AWS_AVAILABLE and any(k in msg for k in s3_keywords) and not _is_cost_query:
        try:
            result = aws_ops.list_s3_buckets()
            buckets = result.get("buckets", [])
            if buckets:
                parts.append(f"S3 Buckets ({len(buckets)} total): " + ", ".join(b["name"] for b in buckets[:10]))
            else:
                parts.append("S3: No buckets found.")
        except Exception:
            pass

    # ── SQS intent ───────────────────────────────────────────────────────────
    sqs_keywords = {"sqs", "queue", "queues", "message queue"}
    if _AWS_AVAILABLE and any(k in msg for k in sqs_keywords):
        try:
            result = aws_ops.list_sqs_queues()
            queues = result.get("queues", [])
            if queues:
                lines = [f"SQS Queues ({len(queues)} total):"]
                for q in queues[:5]:
                    lines.append(f"  • {q['name']} — {q.get('visible','?')} visible msgs")
                parts.append("\n".join(lines))
        except Exception:
            pass

    # ── DynamoDB intent ──────────────────────────────────────────────────────
    dynamo_keywords = {"dynamodb", "dynamo", "nosql", "ddb"}
    if _AWS_AVAILABLE and any(k in msg for k in dynamo_keywords):
        try:
            result = aws_ops.list_dynamodb_tables()
            tables = result.get("tables", [])
            if tables:
                lines = [f"DynamoDB Tables ({len(tables)} total):"]
                for t in tables[:5]:
                    lines.append(f"  • {t['name']} — {t.get('status','?')} — {t.get('item_count',0):,} items")
                parts.append("\n".join(lines))
        except Exception:
            pass

    return "\n\n".join(parts)


_CRED_PATTERNS = re.compile(
    r'(sk-ant-[A-Za-z0-9\-_]{10,}|'
    r'sk-[A-Za-z0-9]{20,}|'
    r'ghp_[A-Za-z0-9]{20,}|'
    r'ghs_[A-Za-z0-9]{20,}|'
    r'AKIA[A-Z0-9]{16}|'
    r'xoxb-[A-Za-z0-9\-]{20,}|'
    r'xoxp-[A-Za-z0-9\-]{20,}|'
    r'grok-[A-Za-z0-9\-]{20,}|'
    r'[A-Za-z0-9+/]{40,}={0,2}'  # generic long base64 that looks like a secret
    r')',
    re.IGNORECASE,
)

_SHORT_FOLLOWUPS = {
    "ok", "oky", "okay", "cool", "got it", "thanks", "thank you", "sure",
    "yes", "no", "yep", "nope", "alright", "nice", "great", "sounds good",
    "perfect", "good", "noted", "ack", "k", "understood",
}


def _redact_credentials(text: str) -> str:
    """Replace any credential-like strings in LLM output with [REDACTED]."""
    return _CRED_PATTERNS.sub("[REDACTED]", text)


_SUGGESTIONS_RE = re.compile(
    r"<!--SUGGESTIONS-->\s*(.*?)\s*<!--/SUGGESTIONS-->",
    re.DOTALL,
)


def _extract_suggestions(text: str) -> tuple[str, list[str]]:
    """Split the LLM response into (clean_answer, [suggestion1, suggestion2, suggestion3]).

    Returns the answer with the suggestions block stripped out, plus up to 3
    suggestion strings.  If no suggestions block is present returns (text, []).
    """
    match = _SUGGESTIONS_RE.search(text)
    if not match:
        return text, []

    suggestions = [
        s.strip() for s in match.group(1).splitlines() if s.strip()
    ][:3]
    clean = _SUGGESTIONS_RE.sub("", text).strip()
    return clean, suggestions


def _is_short_followup(message: str) -> bool:
    """Return True if the message is a short acknowledgement with no real query."""
    stripped = message.strip().lower().rstrip("!.,")
    return stripped in _SHORT_FOLLOWUPS or (len(stripped.split()) <= 2 and stripped in _SHORT_FOLLOWUPS)


def _parallel_execute(tool_uses: list[dict], session_id: str) -> list[dict]:
    """Execute multiple tool calls concurrently using a thread pool.

    Args:
        tool_uses: List of dicts with keys: id, name, input
        session_id: Session ID passed through to execute_tool for caching

    Returns:
        List of dicts with keys: tool_use_id, name, result
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run(tool_use: dict) -> dict:
        result = execute_tool(tool_use["name"], tool_use["input"], session_id=session_id)
        # Cache successful results for read-only tools
        _tool_cache_set(tool_use["name"], tool_use["input"], result)
        return {"tool_use_id": tool_use["id"], "name": tool_use["name"], "result": result}

    with ThreadPoolExecutor(max_workers=min(len(tool_uses), 5)) as pool:
        futures = {pool.submit(_run, tu): tu for tu in tool_uses}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result(timeout=60))
            except Exception as exc:
                tu = futures[future]
                results.append({
                    "tool_use_id": tu["id"],
                    "name": tu["name"],
                    "result": f"Tool execution failed: {exc}",
                })
    return results


def _chat_anthropic_native(
    system_prompt: str,
    history_messages: list[dict],
    user_turn: str,
    session_id: str,
    vision_content: list | None = None,
) -> str:
    """Run a full multi-turn tool loop using the Anthropic native tools API.

    Uses structured tool_use / tool_result content blocks — no regex parsing.
    Independent tool calls in each round are executed in parallel.
    If vision_content is provided, it is used as the first user message content
    (list of image + text blocks); otherwise plain user_turn string is used.
    """
    from app.llm.claude import _anthropic_client

    anthropic_tools = _build_anthropic_tools()
    first_user_content = vision_content if vision_content is not None else user_turn
    messages = list(history_messages) + [{"role": "user", "content": first_user_content}]

    for _round in range(6):  # allow more rounds — native calling is reliable
        try:
            response = _anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system_prompt,
                tools=anthropic_tools,
                messages=messages,
                temperature=0.65,
            )
        except Exception as exc:
            err = str(exc)
            if "529" in err or "overloaded" in err.lower():
                return "The AI is temporarily overloaded. Please wait a moment and try again, or switch models using the dropdown."
            if "429" in err or "rate_limit" in err.lower():
                return "Rate limit hit. Please wait a moment and try again, or switch to a different model."
            raise  # let caller handle other errors

        if response.stop_reason == "end_turn":
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

        if response.stop_reason == "tool_use":
            # Append the full assistant turn (raw SDK objects — required by Anthropic API)
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [
                {"id": block.id, "name": block.name, "input": block.input}
                for block in response.content
                if block.type == "tool_use"
            ]

            _log(logger.info, "native_tool_calls",
                 round=_round + 1,
                 tools=[tu["name"] for tu in tool_uses],
                 parallel=len(tool_uses) > 1,
                 session_id=session_id)

            # Execute all tools for this round in parallel
            results = _parallel_execute(tool_uses, session_id)

            # Feed results back as a user turn with tool_result blocks
            tool_result_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r["tool_use_id"],
                    "content": str(r["result"])[:4000],
                }
                for r in results
            ]
            messages.append({"role": "user", "content": tool_result_blocks})

        else:
            # Unexpected stop reason — extract any text and return
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            ) or "Unexpected response from AI."

    return "Reached maximum tool-calling rounds without a final answer."


def _chat_anthropic_stream(
    system_prompt: str,
    history_messages: list[dict],
    user_turn: str,
    session_id: str,
    vision_content: list | None = None,
    on_tool_event=None,
):
    """Streaming generator version of _chat_anthropic_native.

    Yields text tokens in real-time as they arrive from the Anthropic API.
    For tool-call rounds (non-final), uses regular blocking API so results feed
    back quickly; only the final text response streams token-by-token.

    Args:
        on_tool_event: optional callable(event_dict) — called with
            {"type": "thinking", "text": "..."} SSE events during tool rounds
            so the frontend can show progress while AWS calls run.

    Yields:
        str tokens — individual text deltas from the final response
    """
    from app.llm.claude import _anthropic_client

    anthropic_tools = _build_anthropic_tools()
    first_user_content = vision_content if vision_content is not None else user_turn
    messages = list(history_messages) + [{"role": "user", "content": first_user_content}]

    for _round in range(6):
        # ── Check if this round will need tool calls first ───────────────
        # Run non-streaming for tool rounds (fast); stream the final text answer
        try:
            probe = _anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system_prompt,
                tools=anthropic_tools,
                messages=messages,
                temperature=0.65,
            )
        except Exception as exc:
            err = str(exc)
            if "529" in err or "overloaded" in err.lower():
                yield "The AI is temporarily overloaded. Please wait a moment and try again."
                return
            if "429" in err or "rate_limit" in err.lower():
                yield "Rate limit hit. Please wait a moment and try again."
                return
            raise

        if probe.stop_reason == "end_turn":
            # This is the final response — re-request with streaming for real token delivery
            try:
                with _anthropic_client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=2048,
                    system=system_prompt,
                    tools=anthropic_tools,
                    messages=messages,
                    temperature=0.65,
                ) as stream:
                    for text_delta in stream.text_stream:
                        yield text_delta
            except Exception:
                # Fall back to the already-fetched probe response
                yield "".join(
                    block.text for block in probe.content if hasattr(block, "text")
                )
            return

        if probe.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": probe.content})
            tool_uses = [
                {"id": block.id, "name": block.name, "input": block.input}
                for block in probe.content
                if block.type == "tool_use"
            ]
            tool_names = [tu["name"] for tu in tool_uses]
            _log(logger.info, "stream_tool_calls", round=_round + 1,
                 tools=tool_names, session_id=session_id)

            # Notify frontend so user sees "fetching data..." immediately
            if on_tool_event:
                label = ", ".join(
                    t.replace("_", " ").replace("get ", "").replace("list ", "")
                    for t in tool_names
                )
                on_tool_event({"type": "thinking", "text": f"Fetching {label}..."})

            results = _parallel_execute(tool_uses, session_id)
            tool_result_blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": r["tool_use_id"],
                    "content": str(r["result"])[:4000],
                }
                for r in results
            ]
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            yield "".join(
                block.text for block in probe.content if hasattr(block, "text")
            ) or "Unexpected response from AI."
            return

    yield "Reached maximum tool-calling rounds without a final answer."


def _chat_openai_native(
    system_prompt: str,
    history_messages: list[dict],
    user_turn: str,
    session_id: str,
    model: str = "",
) -> str:
    """Run a full multi-turn tool loop using the OpenAI-compatible tools API.

    Works with both OpenAI (gpt-4o) and Groq (llama-3.3-70b-versatile).
    Independent tool calls in each round are executed in parallel.
    """
    from app.llm.openai import _client as _openai_client, _OPENAI_MODEL

    openai_tools = _build_openai_tools()
    _model = model or _OPENAI_MODEL

    messages: list = (
        [{"role": "system", "content": system_prompt}]
        + list(history_messages)
        + [{"role": "user", "content": user_turn}]
    )

    for _round in range(6):
        try:
            response = _openai_client.chat.completions.create(
                model=_model,
                max_tokens=2048,
                temperature=0.65,
                tools=openai_tools,
                tool_choice="auto",
                messages=messages,
            )
        except Exception as exc:
            raise

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content or ""

        if choice.finish_reason == "tool_calls":
            # Append the full assistant message object (SDK type, JSON-serializable)
            messages.append(choice.message)

            tool_uses = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments or "{}"),
                }
                for tc in (choice.message.tool_calls or [])
            ]

            _log(logger.info, "native_tool_calls_openai",
                 round=_round + 1,
                 tools=[tu["name"] for tu in tool_uses],
                 parallel=len(tool_uses) > 1,
                 session_id=session_id)

            results = _parallel_execute(tool_uses, session_id)

            # One tool message per result (OpenAI format)
            for r in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": r["tool_use_id"],
                    "content": str(r["result"])[:4000],
                })

        else:
            return choice.message.content or ""

    return "Reached maximum tool-calling rounds without a final answer."


def _handle_llm_error(exc: Exception, llm) -> tuple[str, bool]:
    """Return (error_message, should_break). Empty string means retry."""
    err_str = str(exc)
    provider_name = llm.__class__.__name__.replace("Provider", "")
    if "credit balance" in err_str.lower() or "no credits" in err_str.lower() or "billing" in err_str.lower():
        return "", False  # signal caller to switch provider and retry
    if "529" in err_str or "overloaded" in err_str.lower():
        return (
            f"The **{provider_name}** AI is temporarily overloaded. Please wait a moment and try again.\n\n"
            "**What you can do:**\n"
            "- Wait 10–30 seconds and retry your message\n"
            "- Switch to a different model using the dropdown at the top (e.g. Claude → GPT-4 or Groq)"
        ), True
    if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower() or "insufficient_quota" in err_str:
        return "", False  # signal caller to switch provider and retry
    if "ollama" in err_str.lower() and ("not running" in err_str.lower() or "no models" in err_str.lower()):
        return (
            "**Ollama is not running** or has no models loaded.\n\n"
            "To use Ollama:\n"
            "1. Install it from [ollama.com](https://ollama.com)\n"
            "2. Run `ollama pull llama3` to download a model\n"
            "3. Start it with `ollama serve`\n"
            "4. Restart the server, then try again\n\n"
            "Or switch to a different model in the dropdown."
        ), True
    if "api_key" in err_str.lower() or "unauthorized" in err_str.lower() or "401" in err_str:
        return "The AI model key is invalid or missing. Check your API key in the `.env` file.", True
    return f"Something went wrong talking to the AI. Technical detail: `{exc}`", True


# ---------------------------------------------------------------------------
# Pre-LLM interceptor: platform / identity questions answered from code
# ---------------------------------------------------------------------------
_PLATFORM_KEYWORDS = re.compile(
    r'\b('
    r'what (is|are|does) (this|the|your|nexusops|nsops|devops|ai) (tool|platform|system|app|product|software|assistant|bot|do|does)'
    r'|what (can|do) (you|this) (do|does|offer|provide|help|handle)'
    r'|tell me (about|more about) (this|the|yourself|nexusops|nsops|you|your)'
    r'|introduce yourself|who are you|what are you'
    r'|explain (this|the|nexusops|nsops) (tool|platform|app|system|product)?'
    r'|how (does|do|did|can) (this|nexusops|nsops|the platform|the tool|the system|it) (work|function|operate|run)'
    r'|how nexusops (works?|connects?|integrates?|monitors?|handles?|manages?|uses?)'
    r'|how (does|do) nexusops (connect|integrate|monitor|handle|manage|use|work)'
    r'|nexusops (connects?|integrates?|monitors?|handles?) (to|with|aws|kubernetes|github|k8s)'
    r'|capabilities|features|functionality|overview'
    r'|what (is|are) nexusops|what is nsops'
    r'|how .{0,30} (aws|kubernetes|k8s|github|grafana|slack|jira) .{0,30} (connect|integrat|work|use|monitor)'
    r'|connect.{0,20}aws|aws.{0,20}connect|integrat.{0,20}aws|aws.{0,20}integrat'
    r')\b',
    re.IGNORECASE,
)

_PLATFORM_DESCRIPTION = """**NexusOps** is an AI-powered DevOps platform that monitors, analyzes, and auto-remediates infrastructure incidents.

**Core Capabilities:**

🔍 **Incident Detection & Monitoring**
- Continuously monitors AWS, Kubernetes, and GitHub for anomalies
- Detects unhealthy EC2 instances, ECS task failures, Lambda errors, RDS issues, pod crashes, and failed deployments
- Configurable alert thresholds with auto-escalation

🤖 **AI-Powered Remediation**
- Generates structured remediation plans with confidence scores
- Executes approved actions: restart pods, scale services, reboot instances, redeploy ECS tasks
- Approval workflow — human-in-the-loop or fully automated mode

💬 **Intelligent Chat Assistant (that's me!)**
- Ask questions about your live infrastructure in plain English
- I can list EC2 instances, check pod health, review recent GitHub deployments, query Grafana metrics, and more
- I call live AWS/K8s/GitHub APIs to give you real-time answers

📊 **War Room & Collaboration**
- Shared incident war rooms for team coordination
- Auto-generates Jira tickets, Slack notifications, and OpsGenie alerts
- Audit log of every automated action taken

🔗 **Integrations**
- AWS (EC2, ECS, Lambda, RDS, CloudWatch)
- Kubernetes (pods, deployments, namespaces)
- GitHub / GitLab (commits, deployments, PR status)
- Grafana, Slack, Jira, OpsGenie

---

**How NexusOps connects to AWS:**
NexusOps uses the **AWS SDK (boto3)** with credentials from your `.env` file (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`). It calls AWS APIs directly to:
- **EC2**: list instances, start/stop/reboot them
- **ECS**: list clusters/services/tasks, redeploy or scale services
- **Lambda**: list functions, invoke them
- **RDS**: list DB instances, reboot them
- **CloudWatch**: fetch metrics and alarms

No agent or proxy is needed — NexusOps talks directly to AWS REST APIs using your IAM credentials.

---

**How to use me:** Just ask! Try _"show me stopped EC2 instances"_, _"which pods are crashing?"_, _"what deployments happened in the last hour?"_, or _"create an incident for the API being down"_."""

_AWS_CONNECT_KEYWORDS = re.compile(
    r'(how|why|where|what).{0,40}(aws|amazon).{0,40}(connect|integrat|work|setup|config|credential|key|access)',
    re.IGNORECASE,
)


def _maybe_answer_platform_question(message: str) -> str | None:
    """Return a hardcoded platform description if the message is asking about NexusOps.

    This bypasses the LLM entirely so the answer is always correct regardless
    of which model is active (Groq/Llama often ignores system prompts on this).
    Returns None if the message is not a platform question.
    """
    msg = message.strip()
    # Skip very short messages that might be coincidental keyword matches
    if len(msg) < 5:
        return None
    if _PLATFORM_KEYWORDS.search(msg) or _AWS_CONNECT_KEYWORDS.search(msg):
        return _PLATFORM_DESCRIPTION
    return None


def chat_with_intelligence(
    message: str,
    session_id: str,
    incident_context: Optional[dict] = None,
    preferred_provider: Optional[str] = None,
    image_data: Optional[str] = None,   # base64-encoded image
    image_type: Optional[str] = None,   # e.g. "image/png", "image/jpeg"
) -> tuple[str, list[str]]:
    """Process a user message with optional tool calling and return a final answer.

    Flow:
    1. Build structured message history for proper multi-turn conversation context.
    2. Pre-fetch live infra data so LLM never has to guess IDs.
    3. Send to LLM with full history as structured messages.
    4. If LLM emits tool calls, execute them and re-query (up to 3 rounds).
    5. Store and return the final answer.
    """
    if not _LLM_AVAILABLE:
        return "LLM is not configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY."

    # ── Pre-LLM interceptor: answer platform questions without calling any LLM ──
    _platform_answer = _maybe_answer_platform_question(message)
    if _platform_answer:
        if _MEMORY_AVAILABLE:
            get_or_create_session(session_id)
            add_message(session_id, "user", message)
            add_message(session_id, "assistant", _platform_answer)
        return _platform_answer, []

    # Retrieve conversation history (increased to 20 for better context)
    history = []
    if _MEMORY_AVAILABLE:
        history = get_history(session_id, max_messages=20)

    # ── Intent pre-fetch: get live infra data before the LLM runs ───────────
    prefetched_context = _prefetch_context(message, session_id)

    # ── Detect active provider to pick the right execution path ────────────
    # Import provider state — read-only, no side effects
    try:
        from app.llm.claude import _anthropic_client, _ANTHROPIC_KEY_VALID, ANTHROPIC_API_KEY
        from app.llm.openai import _client as _openai_client
    except ImportError:
        _anthropic_client = None
        _ANTHROPIC_KEY_VALID = False
        ANTHROPIC_API_KEY = ""
        _openai_client = None

    _pref = (preferred_provider or "").lower()

    # Native Anthropic path: key is valid, not explicitly overridden to another provider
    _use_anthropic = (
        bool(ANTHROPIC_API_KEY and _ANTHROPIC_KEY_VALID and _anthropic_client)
        and _pref not in ("openai", "groq", "ollama")
    )

    # Native OpenAI path: openai client configured, explicitly chosen or Claude not available
    _use_openai = (
        not _use_anthropic
        and _openai_client is not None
        and _pref not in ("groq", "ollama")
    )

    # Build history and user turn (shared across all paths)
    history_messages = _build_history_messages(history)

    text_body = message
    if prefetched_context:
        text_body = (
            f"{message}\n\n"
            f"=== LIVE INFRASTRUCTURE DATA (auto-fetched) ===\n"
            f"{prefetched_context}\n"
            f"================================================\n"
            f"Use the above live data to answer. Only call tools for data NOT already shown above."
        )

    # user_turn is a plain string for text-only paths.
    user_turn = text_body

    # For Anthropic native path with an image, build a multi-part content block list.
    # This enables Claude's vision capability (screenshots, dashboards, error screens).
    if image_data and image_type:
        _media = image_type if image_type.startswith("image/") else f"image/{image_type}"
        user_turn_vision: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _media,
                    "data": image_data,
                },
            },
            {"type": "text", "text": text_body},
        ]
    else:
        user_turn_vision = None   # use plain string path

    final_answer = ""  # will be set by whichever provider succeeds

    # ── Native Anthropic function calling (primary path) ────────────────────
    if _use_anthropic:
        system_prompt = _build_system_prompt(incident_context, native_tools=True)
        _log(logger.info, "chat_provider", provider="anthropic_native", session_id=session_id)
        try:
            final_answer = _chat_anthropic_native(
                system_prompt, history_messages, user_turn, session_id,
                vision_content=user_turn_vision,
            )
        except Exception as exc:
            err_str = str(exc)
            _log(logger.error, "anthropic_native_failed", error=err_str)
            final_answer = ""
            # For billing/auth errors, fall through to next provider instead of returning error
            _anthropic_hard_fail = (
                "credit balance" in err_str.lower() or
                "no credits" in err_str.lower() or
                "too low" in err_str.lower() or
                "api_key" in err_str.lower() or
                "401" in err_str or
                "529" in err_str or
                "overloaded" in err_str.lower()
            )
            if not _anthropic_hard_fail:
                # Unknown error — surface it
                final_answer = f"Anthropic error: {err_str[:200]}"
                return _redact_credentials(final_answer), []
            # Otherwise fall through to OpenAI / Groq below
            _log(logger.warning, "anthropic_unavailable_falling_back", error=err_str[:80])

    # ── Native OpenAI function calling (secondary path) ──────────────────────
    if not final_answer and _use_openai:
        system_prompt = _build_system_prompt(incident_context, native_tools=True)
        _log(logger.info, "chat_provider", provider="openai_native", session_id=session_id)
        try:
            final_answer = _chat_openai_native(
                system_prompt, history_messages, user_turn, session_id
            )
        except Exception as exc:
            err_str = str(exc)
            _log(logger.error, "openai_native_failed", error=err_str)
            final_answer = f"OpenAI error: {exc}"

    # ── Text-token fallback (Groq / Ollama / no native client) ───────────────
    if not final_answer:
        system_prompt = _build_system_prompt(incident_context, native_tools=False)
        _log(logger.info, "chat_provider", provider="text_token_fallback", session_id=session_id)

        llm = None
        for _ in range(4):
            try:
                llm = LLMFactory.get(preferred=preferred_provider or None)
                break
            except RuntimeError as exc:
                err = str(exc)
                if "no credits" in err.lower() or "falling back" in err.lower():
                    continue
                return (
                    "No AI model is available right now. Please configure at least one of: "
                    "ANTHROPIC_API_KEY (Claude), OPENAI_API_KEY (OpenAI), GROQ_API_KEY (Groq/Llama), "
                    "or start Ollama locally."
                )
        if llm is None:
            return (
                "No AI model is available right now. Please configure at least one of: "
                "ANTHROPIC_API_KEY (Claude), OPENAI_API_KEY (OpenAI), GROQ_API_KEY (Groq/Llama), "
                "or start Ollama locally."
            )

        tool_results_accumulated: list[str] = []
        final_answer = ""
        round_num = 0

        for round_num in range(3):
            current_prompt = user_turn
            if tool_results_accumulated:
                tool_summary = "\n---\n".join(tool_results_accumulated)
                current_prompt = (
                    f"{user_turn}\n\n=== TOOL RESULTS ===\n{tool_summary}\n====================\n\n"
                    f"Based on the tool results above, give your final answer now."
                )

            try:
                response = llm.complete(
                    current_prompt,
                    system=system_prompt,
                    max_tokens=2048,
                    messages=history_messages,
                    temperature=0.65,
                )
                raw_response = response.content
            except Exception as exc:
                _log(logger.error, "llm_chat_failed", error=str(exc), round=round_num)
                err_msg, should_break = _handle_llm_error(exc, llm)
                if not err_msg and not should_break:
                    # Current provider failed — try next one in order
                    _failed = getattr(llm, "_force_provider", None) or preferred_provider or ""
                    _fallback_order = [p for p in ("groq", "ollama", "openai", "claude") if p != _failed]
                    _switched = False
                    for _fb in _fallback_order:
                        try:
                            llm = LLMFactory.get(preferred=_fb)
                            _switched = True
                            break
                        except RuntimeError:
                            continue
                    if _switched:
                        continue
                    final_answer = "All AI providers are unavailable. Please add credits to Anthropic, OpenAI, or use Groq/Ollama."
                    break
                final_answer = err_msg
                break

            tool_calls = _extract_tool_calls(raw_response)
            if not tool_calls:
                final_answer = _strip_tool_calls(raw_response)
                break

            _log(logger.info, "tool_calls_detected",
                 round=round_num + 1, tools=[t[0] for t in tool_calls], session_id=session_id)

            # Run tool calls in parallel even on the fallback path
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
            with ThreadPoolExecutor(max_workers=min(len(tool_calls), 5)) as _pool:
                _futures = {
                    _pool.submit(execute_tool, tn, params, session_id): tn
                    for tn, params in tool_calls
                }
                for _f in _as_completed(_futures):
                    tn = _futures[_f]
                    try:
                        tool_results_accumulated.append(f"[{tn}] => {_f.result(timeout=45)}")
                    except Exception as _exc:
                        tool_results_accumulated.append(f"[{tn}] => Tool failed: {_exc}")

            if round_num == 2:
                tool_summary = "\n---\n".join(tool_results_accumulated)
                final_prompt = (
                    f"{user_turn}\n\n=== TOOL RESULTS ===\n{tool_summary}\n====================\n\n"
                    f"You have all the data. Provide your complete final answer now."
                )
                try:
                    resp = llm.complete(final_prompt, system=system_prompt,
                                        max_tokens=2048, messages=history_messages, temperature=0.65)
                    final_answer = _strip_tool_calls(resp.content)
                except Exception as exc:
                    final_answer = f"Error generating final answer: {exc}"
                break

    if not final_answer:
        final_answer = "I was unable to generate a response. Please try again."

    # Extract follow-up suggestions embedded by the LLM, then redact credentials
    final_answer, suggestions = _extract_suggestions(final_answer)
    final_answer = _redact_credentials(final_answer)

    if _MEMORY_AVAILABLE:
        add_message(session_id, "user",      message,      metadata={"incident_context": bool(incident_context)})
        add_message(session_id, "assistant", final_answer, metadata={"provider": "anthropic_native" if _use_anthropic else "openai_native" if _use_openai else "text_token"})

    return final_answer, suggestions
