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


def _build_system_prompt(incident_context: Optional[dict], session_id: str = "") -> str:
    """Build the system prompt injected before every conversation turn.

    Args:
        incident_context: Optional dict of incident/war-room context.

    Returns:
        System prompt string.
    """
    now = datetime.datetime.utcnow().isoformat() + "Z"

    incident_section = ""
    if incident_context:
        incident_section = (
            "\n\n=== ACTIVE INCIDENT CONTEXT ===\n"
            + json.dumps(incident_context, indent=2, default=str)[:1500]
            + "\n================================"
        )

    return f"""You are NsOps AI, a friendly and knowledgeable DevOps assistant.
Current UTC time: {now}

{_build_tools_description()}

COMMUNICATION STYLE:
- Always explain things in plain, simple language that anyone can understand — not just engineers.
- Avoid jargon. When you must use a technical term, briefly explain it in parentheses (e.g. "pods (the small containers your app runs in)").
- Be warm, clear, and direct. Lead with the key takeaway, then add detail.
- Use short bullet points for lists of items. Bold the most important words.
- If someone asks a vague question, give them a helpful answer rather than asking for clarification — make a reasonable assumption and state it.

TOOL CALLING:
When you need live data, emit ONLY the tool call token — no introduction, no "let me check", no narration before it.
Use EXACTLY this format on its own line:
  [TOOL_CALL: tool_name({{"param1": "value1"}})]
For tools with no parameters use:
  [TOOL_CALL: tool_name({{"key": "value"}})]

Rules:
- READ-ONLY tools (list, query, get): Call immediately, no confirmation, no preamble. Just emit the token then the answer.
- MUTATING ACTIONS (start/stop/reboot EC2, restart/scale K8s, scale/redeploy ECS, reboot RDS, create Jira ticket, create GitHub issue, page on-call, create war room): Ask the user to confirm first, THEN call the tool after they say yes.
- FOLLOW-UP CONTEXT: When user says "that", "it", "the instance", "the pod" — resolve from CONVERSATION HISTORY. Never ask what they mean if the answer is in history.
- NO SPECIFIC ID? When the user mentions EC2 or K8s without a specific instance ID or pod name (e.g. "my ec2 is down", "ec2 not working"), ALWAYS call list_ec2_instances or list_k8s_instances FIRST — do NOT call any other EC2 tool yet. Never pass words like "my", "ec2", "instance", "server", "unknown", "none", "it", "that" as an instance ID — they are not valid IDs. A valid EC2 instance ID always starts with "i-" followed by hex digits (e.g. i-0abc1234def567890). If you don't have a valid i-xxxx ID, call list_ec2_instances first.
- Only call tools when you genuinely need live data.
- You may chain up to 3 tool calls per message.
- After receiving tool results, give a direct plain-English answer using the real values. No raw JSON, no field names.
- If a tool returns an error, explain it simply and suggest what to check.
- Never make up metric values.
- REGION QUERIES: When user asks about resources in a specific region or geography (e.g. "what's in Asia", "resources in Singapore", "show eu-west-1", "everything in us-west-2"), ALWAYS call list_aws_resources with the region parameter. Supported region keywords: "asia", "eu", "europe", "us", "all", specific region codes like "ap-southeast-1", city names like "singapore", "tokyo", "frankfurt", "london".
- COST QUERIES: When user asks about pricing or costs, ALWAYS call estimate_aws_cost — never EC2 status tools.
- GITHUB REPOS: When user asks "list my repos", "what repos do I have", "show GitHub repos", "what's on my GitHub" — call list_github_repos immediately. No confirmation needed.
- INFRA OVERVIEW: When user asks "how does my infra look", "what's my infrastructure", "overview of my setup" — call list_ec2_instances AND list_aws_resources (region="all") together.
- LOGS: When user asks "show me logs", "what's in the logs", "any errors in logs", "show me what happened between X and Y" — ALWAYS call list_log_groups first to see what CloudWatch log groups exist, then call get_recent_logs or search_logs with the correct group. Do NOT use query_k8s_logs unless the user specifically mentions Kubernetes pods or namespaces. Never make up log content.
- LOG TIME RANGE: When user says "between 2pm and 3pm", "from 10am to 11am", "last 30 minutes", "last hour" — convert to an hours value and pass to get_recent_logs.
- GRAFANA: When user asks about Grafana, dashboards, or monitoring alerts (not CloudWatch) — call get_grafana_status.
- VSCODE: When user asks to notify VS Code, open a file in the editor, or send something to the IDE — call vscode_notify or vscode_open_file.
- LAYMAN PROMPTS: When user says vague things like "something is wrong", "my app is slow", "site is down", "help" — don't ask for clarification. Sweep all available health data (EC2, K8s, alarms) and report what you find.
- GENERIC HEALTH: When user asks "is everything ok?", "how are things?", "system status" — call list_ec2_instances and query_aws_alarms together to give a full picture.{incident_section}
"""


def _build_history_text(history) -> str:
    """Format conversation history as a plain text context block."""
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

    # ── Cost intent ──────────────────────────────────────────────────────────
    cost_keywords = {"cost", "bill", "billing", "spend", "spending", "charge", "charges", "expensive", "money", "price"}
    if any(k in msg for k in cost_keywords):
        parts.append("Tip: Use the cost dashboard or ask me to check current AWS costs.")

    return "\n\n".join(parts)


def chat_with_intelligence(
    message: str,
    session_id: str,
    incident_context: Optional[dict] = None,
    preferred_provider: Optional[str] = None,
) -> str:
    """Process a user message with optional tool calling and return a final answer.

    Flow:
    1. Build prompt from system context + history + user message.
    2. Send to LLM.
    3. If LLM emits tool calls, execute them and re-query (up to 3 rounds).
    4. Store user message and final assistant response in memory.
    5. Return final answer.

    Args:
        message:          The user's message.
        session_id:       Session identifier for memory tracking.
        incident_context: Optional dict injected as incident war-room context.

    Returns:
        Final assistant response string.
    """
    if not _LLM_AVAILABLE:
        return "LLM is not configured. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY."

    # Retrieve conversation history
    history = []
    if _MEMORY_AVAILABLE:
        history = get_history(session_id, max_messages=15)

    # ── Intent pre-fetch ────────────────────────────────────────────────────
    # Detect what the user is asking about and fetch live data automatically
    # so the LLM always has real context — never needs to guess IDs.
    prefetched_context = _prefetch_context(message, session_id)

    system_prompt = _build_system_prompt(incident_context)
    history_text  = _build_history_text(history)

    # Build initial user prompt — inject pre-fetched context so LLM has real data
    if prefetched_context:
        full_prompt = f"{history_text}USER: {message}\n\n=== LIVE CONTEXT (auto-fetched) ===\n{prefetched_context}\n==================================\n\nASSISTANT:"
    else:
        full_prompt = f"{history_text}USER: {message}\n\nASSISTANT:"

    # Try preferred provider first; if it fails due to credits/billing, retry with fallback
    _tried_providers: list[str] = []
    llm = None
    for _attempt in range(4):  # try up to 4 providers in chain
        try:
            llm = LLMFactory.get(preferred=preferred_provider or None)
            break
        except RuntimeError as exc:
            err = str(exc)
            if "no credits" in err.lower() or "falling back" in err.lower():
                # Already marked rate-limited in ClaudeProvider.complete — retry factory
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

    for round_num in range(3):
        # Add any accumulated tool results to the prompt
        if tool_results_accumulated:
            tool_block = "\n\n=== TOOL RESULTS ===\n"
            tool_block += "\n---\n".join(tool_results_accumulated)
            tool_block += "\n====================\n\nBased on the above results, answer the user's question:\nASSISTANT:"
            prompt_with_tools = full_prompt.rstrip("ASSISTANT:").rstrip() + tool_block
        else:
            prompt_with_tools = full_prompt

        try:
            response = llm.complete(
                prompt_with_tools,
                system=system_prompt,
                max_tokens=1024,
            )
            raw_response = response.content
        except Exception as exc:
            _log(logger.error, "llm_chat_failed", error=str(exc), round=round_num)
            err_str = str(exc)
            # Credit/billing error — auto-switch to next available provider
            if "credit balance" in err_str.lower() or "no credits" in err_str.lower() or "billing" in err_str.lower():
                try:
                    llm = LLMFactory.get(preferred=None)  # let factory pick next available
                    continue
                except RuntimeError:
                    final_answer = "All AI providers are unavailable (no credits or API keys). Please add credits to your Anthropic account or configure GROQ_API_KEY."
                    break
            if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower():
                # Extract wait time if present
                import re as _re
                wait_match = _re.search(r"try again in (\d+m\d+s|\d+s|\d+ seconds?)", err_str, _re.IGNORECASE)
                wait_hint = f" Try again in {wait_match.group(1)}." if wait_match else " Please wait a minute and try again."
                provider_name = llm.__class__.__name__.replace("Provider", "")
                final_answer = (
                    f"The **{provider_name}** AI is temporarily busy — you've hit the daily/minute usage limit.{wait_hint}\n\n"
                    "**What you can do:**\n"
                    "- Switch to a different model using the dropdown at the top\n"
                    "- Wait the suggested time and retry\n"
                    "- Add an API key for another provider (Claude, OpenAI) in your `.env` file"
                )
            elif "ollama" in err_str.lower() and ("not running" in err_str.lower() or "no models" in err_str.lower()):
                final_answer = (
                    "**Ollama is not running** or has no models loaded.\n\n"
                    "To use Ollama:\n"
                    "1. Install it from [ollama.com](https://ollama.com)\n"
                    "2. Run `ollama pull llama3` to download a model\n"
                    "3. Start it with `ollama serve`\n"
                    "4. Restart the server, then try again\n\n"
                    "Or switch to a different model in the dropdown."
                )
            elif "api_key" in err_str.lower() or "unauthorized" in err_str.lower() or "401" in err_str:
                final_answer = "The AI model key is invalid or missing. Check your API key in the `.env` file."
            else:
                final_answer = f"Something went wrong talking to the AI. Technical detail: `{exc}`"
            break

        # Check for tool calls
        tool_calls = _extract_tool_calls(raw_response)

        if not tool_calls:
            # No tool calls — this is the final answer
            final_answer = _strip_tool_calls(raw_response)
            break

        # Execute tool calls
        _log(logger.info, 
            "tool_calls_detected",
            round=round_num + 1,
            tools=[t[0] for t in tool_calls],
            session_id=session_id,
        )

        round_results = []
        for tool_name, params in tool_calls:
            result = execute_tool(tool_name, params, session_id=session_id)
            round_results.append(f"[{tool_name}] => {result}")

        tool_results_accumulated.extend(round_results)

        # If this is round 3, force a final answer
        if round_num == 2:
            tool_block = "\n\n=== TOOL RESULTS ===\n"
            tool_block += "\n---\n".join(tool_results_accumulated)
            tool_block += (
                "\n====================\n\n"
                "You have reached the maximum tool-call rounds. "
                "Provide your final answer now without calling more tools:\nASSISTANT:"
            )
            final_prompt = full_prompt.rstrip("ASSISTANT:").rstrip() + tool_block
            try:
                resp = llm.complete(final_prompt, system=system_prompt, max_tokens=1024)
                final_answer = _strip_tool_calls(resp.content)
            except Exception as exc:
                final_answer = f"Error generating final answer: {exc}"
            break

    if not final_answer:
        final_answer = "I was unable to generate a response. Please try again."

    # Store in memory
    if _MEMORY_AVAILABLE:
        add_message(session_id, "user",      message,      metadata={"incident_context": bool(incident_context)})
        add_message(session_id, "assistant", final_answer, metadata={"tool_rounds": round_num + 1})

    return final_answer
