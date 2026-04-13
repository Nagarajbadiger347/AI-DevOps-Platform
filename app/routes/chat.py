"""
Chat, streaming, and session routes.
Paths: /chat/*, /chat/stream
"""
import asyncio
from typing import Optional, Dict, List

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.routes.deps import (
    require_viewer, AuthContext,
)

router = APIRouter(tags=["chat"])

_chat_action_count = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatPayload(BaseModel):
    message: str
    history: List[ChatMessage] = []
    provider: str = ""
    confirmed: bool = False
    pending_action: Optional[str] = None
    pending_params: Optional[Dict] = None
    dry_run: bool = False
    session_id: Optional[str] = None
    incident_context: Optional[Dict] = None
    image_data: Optional[str] = None
    image_type: Optional[str] = None


# ── K8s handlers (imported lazily to avoid circular imports) ──────────────────
def _k8s():
    from app.integrations import k8s_ops as _k
    return _k


def _aws():
    from app.integrations import aws_ops as _a
    return _a


# ── Action catalogue ──────────────────────────────────────────────────────────
def _build_catalogue():
    """Return the action catalogue dict. Built lazily to allow all integrations to load."""
    from app.integrations.k8s_ops import (
        restart_deployment, scale_deployment, delete_pod, get_pod_logs,
        list_pods, list_deployments, list_namespaces, get_cluster_events,
        get_unhealthy_pods, get_resource_usage, cordon_node, uncordon_node,
    )
    from app.integrations.aws_ops import (
        list_ec2_instances, get_ec2_instance_info, get_ec2_status_checks,
        start_ec2_instance, stop_ec2_instance, reboot_ec2_instance,
        list_ecs_services, get_ecs_service_detail, scale_ecs_service,
        force_new_ecs_deployment, get_stopped_ecs_tasks,
        list_lambda_functions, get_lambda_errors, invoke_lambda,
        list_rds_instances, get_rds_instance_detail, get_rds_events, reboot_rds_instance,
        list_cloudwatch_alarms, set_alarm_state, search_logs, get_recent_logs,
        get_cloudtrail_events, list_sqs_queues, get_sqs_queue_depth,
        list_s3_buckets, list_dynamodb_tables,
    )
    from app.integrations.github import create_issue
    from app.integrations.slack import post_message
    from app.agents.incident_pipeline import run_incident_pipeline
    from app.integrations.jira import create_incident
    from app.integrations.opsgenie import notify_on_call

    def _list_repos():
        from app.integrations.github import list_repos
        return list_repos()

    def _get_recent_commits(**kw):
        from app.integrations.github import get_recent_commits
        return get_recent_commits(**kw)

    def _get_recent_prs(**kw):
        from app.integrations.github import get_recent_prs
        return get_recent_prs(**kw)

    return {
        "restart_deployment": {
            "desc": "Rolling restart a Kubernetes deployment",
            "params": ["namespace", "deployment"],
            "handler": lambda p: restart_deployment(p["namespace"], p["deployment"]),
        },
        "scale_deployment": {
            "desc": "Scale a Kubernetes deployment to N replicas",
            "params": ["namespace", "deployment", "replicas"],
            "handler": lambda p: scale_deployment(p["namespace"], p["deployment"], int(p["replicas"])),
        },
        "delete_pod": {
            "desc": "Delete a pod so Kubernetes reschedules it",
            "params": ["namespace", "pod"],
            "handler": lambda p: delete_pod(p["namespace"], p["pod"]),
        },
        "get_pod_logs": {
            "desc": "Fetch recent logs from a specific pod",
            "params": ["namespace", "pod"],
            "handler": lambda p: get_pod_logs(p["namespace"], p["pod"], tail_lines=int(p.get("lines", 100))),
        },
        "list_pods": {
            "desc": "List all pods and their status in a namespace",
            "params": ["namespace"],
            "handler": lambda p: list_pods(p.get("namespace", "")),
        },
        "list_deployments": {
            "desc": "List all deployments and their replica health",
            "params": ["namespace"],
            "handler": lambda p: list_deployments(p.get("namespace", "")),
        },
        "list_namespaces": {
            "desc": "List all Kubernetes namespaces",
            "params": [],
            "handler": lambda _: list_namespaces(),
        },
        "get_cluster_events": {
            "desc": "Get warning events from the cluster",
            "params": ["namespace"],
            "handler": lambda p: get_cluster_events(p.get("namespace", "")),
        },
        "get_unhealthy_pods": {
            "desc": "Get all pods that are not healthy or crash-looping",
            "params": ["namespace"],
            "handler": lambda p: get_unhealthy_pods(p.get("namespace", "")),
        },
        "get_resource_usage": {
            "desc": "Get CPU and memory requests/limits for pods in a namespace",
            "params": ["namespace"],
            "handler": lambda p: get_resource_usage(p.get("namespace", "default")),
        },
        "cordon_node": {
            "desc": "Cordon a node to stop new pods scheduling on it",
            "params": ["node"],
            "handler": lambda p: cordon_node(p["node"]),
        },
        "uncordon_node": {
            "desc": "Uncordon a node to allow pods to schedule on it again",
            "params": ["node"],
            "handler": lambda p: uncordon_node(p["node"]),
        },
        "list_ec2": {
            "desc": "List all EC2 instances",
            "params": ["region"],
            "handler": lambda p: list_ec2_instances(region=p.get("region", "")),
        },
        "get_ec2_info": {
            "desc": "Get full details for a specific EC2 instance",
            "params": ["instance_id"],
            "handler": lambda p: get_ec2_instance_info(p.get("instance_id", "")),
        },
        "get_ec2_status": {
            "desc": "Get system and instance status checks for EC2",
            "params": ["instance_id"],
            "handler": lambda p: get_ec2_status_checks(p.get("instance_id", "")),
        },
        "start_ec2": {
            "desc": "Start a stopped EC2 instance",
            "params": ["instance_id"],
            "handler": lambda p: start_ec2_instance(p.get("instance_id", ""), region=p.get("region", "")),
        },
        "stop_ec2": {
            "desc": "Stop a running EC2 instance",
            "params": ["instance_id"],
            "handler": lambda p: stop_ec2_instance(p.get("instance_id", ""), region=p.get("region", "")),
        },
        "reboot_ec2": {
            "desc": "Reboot an EC2 instance",
            "params": ["instance_id"],
            "handler": lambda p: reboot_ec2_instance(p.get("instance_id", ""), region=p.get("region", "")),
        },
        "list_ecs_services": {
            "desc": "List ECS services and their task counts",
            "params": ["cluster"],
            "handler": lambda p: list_ecs_services(p.get("cluster", "default")),
        },
        "get_ecs_service": {
            "desc": "Get detailed status for a specific ECS service",
            "params": ["cluster", "service"],
            "handler": lambda p: get_ecs_service_detail(p.get("cluster", "default"), p["service"]),
        },
        "scale_ecs_service": {
            "desc": "Scale an ECS service to a desired task count",
            "params": ["cluster", "service", "count"],
            "handler": lambda p: scale_ecs_service(p.get("cluster", "default"), p["service"], int(p["count"])),
        },
        "redeploy_ecs_service": {
            "desc": "Force a new ECS deployment",
            "params": ["cluster", "service"],
            "handler": lambda p: force_new_ecs_deployment(p.get("cluster", "default"), p["service"]),
        },
        "get_stopped_ecs_tasks": {
            "desc": "List recently stopped ECS tasks and their stop reasons",
            "params": ["cluster"],
            "handler": lambda p: get_stopped_ecs_tasks(p.get("cluster", "default")),
        },
        "list_lambda": {
            "desc": "List all Lambda functions",
            "params": ["region"],
            "handler": lambda p: list_lambda_functions(region=p.get("region", "")),
        },
        "get_lambda_errors": {
            "desc": "Get error and throttle metrics for a Lambda function",
            "params": ["function_name"],
            "handler": lambda p: get_lambda_errors(p["function_name"]),
        },
        "invoke_lambda": {
            "desc": "Invoke a Lambda function and return its response",
            "params": ["function_name", "payload"],
            "handler": lambda p: invoke_lambda(p["function_name"], p.get("payload", {})),
        },
        "list_rds": {
            "desc": "List all RDS database instances",
            "params": ["region"],
            "handler": lambda p: list_rds_instances(region=p.get("region", "")),
        },
        "get_rds_detail": {
            "desc": "Get detailed status for a specific RDS instance",
            "params": ["db_instance_id"],
            "handler": lambda p: get_rds_instance_detail(p["db_instance_id"]),
        },
        "get_rds_events": {
            "desc": "Get recent RDS events",
            "params": ["db_instance_id"],
            "handler": lambda p: get_rds_events(p["db_instance_id"]),
        },
        "reboot_rds": {
            "desc": "Reboot an RDS database instance",
            "params": ["db_instance_id"],
            "handler": lambda p: reboot_rds_instance(p["db_instance_id"]),
        },
        "get_alarms": {
            "desc": "List CloudWatch alarms",
            "params": ["state", "region"],
            "handler": lambda p: list_cloudwatch_alarms(p.get("state", ""), region=p.get("region", "")),
        },
        "get_firing_alarms": {
            "desc": "Get only alarms currently in ALARM state",
            "params": ["region"],
            "handler": lambda p: list_cloudwatch_alarms("ALARM", region=p.get("region", "")),
        },
        "set_alarm_state": {
            "desc": "Manually set a CloudWatch alarm state",
            "params": ["alarm_name", "state"],
            "handler": lambda p: set_alarm_state(p["alarm_name"], p["state"]),
        },
        "search_logs": {
            "desc": "Search CloudWatch logs for a pattern",
            "params": ["log_group", "pattern", "hours"],
            "handler": lambda p: search_logs(p["log_group"], p["pattern"], int(p.get("hours", 1))),
        },
        "get_recent_logs": {
            "desc": "Get recent log events from a CloudWatch log group",
            "params": ["log_group", "minutes"],
            "handler": lambda p: get_recent_logs(p["log_group"], int(p.get("minutes", 30))),
        },
        "get_cloudtrail": {
            "desc": "Get recent CloudTrail API events",
            "params": ["hours"],
            "handler": lambda p: get_cloudtrail_events(int(p.get("hours", 1))),
        },
        "list_sqs": {
            "desc": "List SQS queues and their message counts",
            "params": [],
            "handler": lambda _: list_sqs_queues(),
        },
        "get_sqs_depth": {
            "desc": "Get message depth for a specific SQS queue",
            "params": ["queue_url"],
            "handler": lambda p: get_sqs_queue_depth(p["queue_url"]),
        },
        "list_s3": {
            "desc": "List S3 buckets",
            "params": [],
            "handler": lambda _: list_s3_buckets(),
        },
        "list_dynamodb": {
            "desc": "List DynamoDB tables",
            "params": [],
            "handler": lambda _: list_dynamodb_tables(),
        },
        "list_repos": {
            "desc": "List GitHub repositories",
            "params": [],
            "handler": lambda _: _list_repos(),
        },
        "get_recent_commits": {
            "desc": "Get recent commits from GitHub",
            "params": ["hours"],
            "handler": lambda p: _get_recent_commits(hours=int(p.get("hours", 24))),
        },
        "get_recent_prs": {
            "desc": "Get recently merged pull requests",
            "params": ["hours"],
            "handler": lambda p: _get_recent_prs(hours=int(p.get("hours", 48))),
        },
        "create_github_issue": {
            "desc": "Create a GitHub issue",
            "params": ["title", "body"],
            "handler": lambda p: create_issue(p.get("title", "AI-generated issue"), p.get("body", "")),
        },
        "post_slack": {
            "desc": "Post a message to a Slack channel",
            "params": ["channel", "message"],
            "handler": lambda p: post_message(p.get("channel", "#general"), p["message"]),
        },
        "run_pipeline": {
            "desc": "Run the full autonomous incident response pipeline",
            "params": ["description", "severity"],
            "handler": lambda p: run_incident_pipeline(
                incident_id=p.get("incident_id", f"chat-{__import__('time').strftime('%H%M%S')}"),
                description=p["description"],
                severity=p.get("severity", "high"),
                aws_config={}, k8s_config={}, auto_remediate=False,
            ),
        },
        "create_jira_ticket": {
            "desc": "Create a Jira incident ticket",
            "params": ["summary", "description"],
            "handler": lambda p: create_incident(
                summary=p.get("summary", "AI DevOps Incident"),
                description=p.get("description", p.get("summary", "")),
            ),
        },
        "notify_oncall": {
            "desc": "Page the on-call team via OpsGenie",
            "params": ["message", "priority"],
            "handler": lambda p: notify_on_call(p.get("message", ""), p.get("priority", "P2")),
        },
        "debug_and_fix": {
            "desc": "Collect full context and run AI root cause analysis + automated fix",
            "params": ["description", "severity"],
            "handler": lambda p: run_incident_pipeline(
                incident_id=f"debug-{__import__('time').strftime('%H%M%S')}",
                description=p.get("description", "infrastructure issue"),
                severity=p.get("severity", "high"),
                aws_config={}, k8s_config={}, auto_remediate=True,
            ),
        },
        "estimate_cost": {
            "desc": "Estimate monthly and annual AWS cost for a resource description",
            "params": ["description", "region"],
            "handler": lambda p: __import__('app.cost.pricing', fromlist=['estimate_from_description']).estimate_from_description(
                p.get("description", ""), p.get("region", __import__('os').getenv("AWS_REGION", "us-east-1"))
            ),
        },
    }


_ACTION_CATALOGUE = None


def _get_catalogue():
    global _ACTION_CATALOGUE
    if _ACTION_CATALOGUE is None:
        try:
            _ACTION_CATALOGUE = _build_catalogue()
        except Exception:
            _ACTION_CATALOGUE = {}
    return _ACTION_CATALOGUE


_INTENT_SYSTEM = """You are an intent classifier for a DevOps automation platform.

Decide: is the user asking to DO something, or asking a question?

If DO → output:
{"intent": "action", "action": "<name>", "params": {<key>: <value>}}

If QUESTION → output:
{"intent": "question"}

Available actions (ONLY these — never invent others):

KUBERNETES:
- restart_deployment: namespace, deployment
- scale_deployment: namespace, deployment, replicas (int)
- delete_pod: namespace, pod
- get_pod_logs: namespace, pod, container (optional), tail_lines (optional int)
- list_pods: namespace (optional)
- list_deployments: namespace (optional)
- list_namespaces: (no params)
- get_cluster_events: namespace (optional), limit (optional int)
- get_unhealthy_pods: namespace (optional)
- get_resource_usage: namespace (optional)
- cordon_node: node
- uncordon_node: node

EC2:
- list_ec2: (no params)
- get_ec2_info: instance_id
- get_ec2_status: instance_id
- start_ec2: instance_id
- stop_ec2: instance_id
- reboot_ec2: instance_id

ECS:
- list_ecs_services: cluster (optional)
- get_ecs_service: cluster, service
- scale_ecs_service: cluster, service, desired_count (int)
- redeploy_ecs_service: cluster, service
- get_stopped_ecs_tasks: cluster (optional)

LAMBDA:
- list_lambda: (no params)
- get_lambda_errors: function_name, hours (optional int)
- invoke_lambda: function_name, payload (optional JSON string)

RDS:
- list_rds: (no params)
- get_rds_detail: db_instance_id
- get_rds_events: db_instance_id, hours (optional int)
- reboot_rds: db_instance_id

CLOUDWATCH / LOGS:
- get_alarms: (no params)
- get_firing_alarms: (no params)
- set_alarm_state: alarm_name, state (OK/ALARM/INSUFFICIENT_DATA), reason
- search_logs: log_group, query, hours (optional int)
- get_recent_logs: log_group, tail_lines (optional int)
- get_cloudtrail: hours (optional int)

SQS / S3 / DYNAMODB:
- list_sqs: (no params)
- get_sqs_depth: queue_url
- list_s3: (no params)
- list_dynamodb: (no params)

GITHUB:
- list_repos: (no params)
- get_recent_commits: hours (optional int), branch (optional)
- get_recent_prs: hours (optional int)
- create_github_issue: title, body

INCIDENTS / PIPELINE:
- run_pipeline: description, severity (critical/high/medium/low), incident_id (optional)
- create_jira_ticket: summary, description
- notify_oncall: message, priority (P1/P2/P3)
- debug_and_fix: service, error_description

COST ESTIMATION:
- estimate_cost: description (natural language), region (optional)

Rules:
- Extract params literally from the message
- REGION: if the user mentions a region, always include "region": "<value>" in params
- COST QUERIES: If the user asks "how much does X cost", always use estimate_cost action
- If a required param is missing and cannot be inferred, output {"intent": "question"} instead
- GENERAL QUERIES: Phrases like "check my infra", "check infrastructure", "show my setup", "what's running", "how is my infra", "infrastructure status", "check health" are QUESTIONS not actions — output {"intent": "question"}
- Output ONLY valid JSON, no markdown, nothing else"""


def _detect_intent(message: str, force_provider: str = "", conv_context: str = "") -> dict:
    import json as _json
    from app.llm.claude import _llm, _extract_json
    content = message
    if conv_context:
        content = (
            f"=== RECENT CONVERSATION ===\n{conv_context}\n"
            f"=== NEW MESSAGE ===\n{message}"
        )
    try:
        raw = _llm(_INTENT_SYSTEM, [{"role": "user", "content": content}],
                   max_tokens=400, force_provider=force_provider)
        return _json.loads(_extract_json(raw))
    except Exception:
        return {"intent": "question"}


_CONFIRM_REQUIRED = {
    "restart_deployment", "scale_deployment", "delete_pod",
    "cordon_node", "uncordon_node",
    "start_ec2", "stop_ec2", "reboot_ec2",
    "scale_ecs_service", "redeploy_ecs_service",
    "invoke_lambda",
    "reboot_rds",
    "set_alarm_state",
    "create_github_issue", "create_jira_ticket",
    "run_pipeline", "notify_oncall", "debug_and_fix",
}


def _confirmation_message(action_name: str, params: dict) -> str:
    p = params or {}
    descriptions = {
        "restart_deployment":   f"rolling restart deployment **{p.get('deployment','?')}** in namespace **{p.get('namespace','?')}**",
        "scale_deployment":     f"scale deployment **{p.get('deployment','?')}** in **{p.get('namespace','?')}** to **{p.get('replicas','?')}** replicas",
        "delete_pod":           f"delete pod **{p.get('pod','?')}** in namespace **{p.get('namespace','?')}** (will be rescheduled)",
        "cordon_node":          f"cordon node **{p.get('node','?')}** — no new pods will be scheduled on it",
        "uncordon_node":        f"uncordon node **{p.get('node','?')}** — allow scheduling again",
        "start_ec2":            f"start EC2 instance **{p.get('instance_id','?')}**",
        "stop_ec2":             f"stop EC2 instance **{p.get('instance_id','?')}**",
        "reboot_ec2":           f"reboot EC2 instance **{p.get('instance_id','?')}**",
        "scale_ecs_service":    f"scale ECS service **{p.get('service','?')}** in cluster **{p.get('cluster','?')}** to **{p.get('desired_count','?')}** tasks",
        "redeploy_ecs_service": f"force a new deployment of ECS service **{p.get('service','?')}** in cluster **{p.get('cluster','?')}**",
        "invoke_lambda":        f"invoke Lambda function **{p.get('function_name','?')}**",
        "reboot_rds":           f"reboot RDS instance **{p.get('db_instance_id','?')}** (brief downtime expected)",
        "set_alarm_state":      f"set CloudWatch alarm **{p.get('alarm_name','?')}** to state **{p.get('state','?')}**",
        "create_github_issue":  f"create GitHub issue: **{p.get('title','?')}**",
        "create_jira_ticket":   f"create Jira ticket: **{p.get('summary','?')}**",
        "run_pipeline":         f"run the full incident pipeline — *{p.get('description','?')}* (severity: {p.get('severity','?')})",
        "notify_oncall":        f"page on-call with priority **{p.get('priority','?')}**: _{p.get('message','?')}_",
        "debug_and_fix":        f"run automated debug & fix for **{p.get('service','?')}**: _{p.get('error_description','?')}_",
    }
    desc = descriptions.get(action_name, f"**{action_name.replace('_',' ')}** with params: {p}")
    return f"\u26a0\ufe0f I\u2019m about to {desc}.\n\n**Confirm?** Reply **yes** to proceed or **no** to cancel."


def _build_action_reply(action_name: str, user_msg: str, action_result: dict,
                        force_prov: str, _llm, _j) -> str:
    succeeded = action_result.get("success", False) if isinstance(action_result, dict) else True
    result_json = _j.dumps(action_result, default=str, indent=2)
    ACTION_REPLY_SYSTEM = (
        "You are a DevOps assistant. The user asked you to perform an operation. "
        "The ACTUAL result from executing that operation is shown below as JSON — use ONLY the values in it. "
        "RESPONSE LENGTH: "
        "If success=true and the result is simple: 1-2 sentences confirming what happened using real values. "
        "If success=true and the result contains a list: present it in a clean readable format. "
        "If success=false or there is an error key: state it failed, quote the exact error message. "
        "FORMATTING: NEVER show raw JSON. Translate everything to natural English. "
        "Use **bold** for resource names and states. "
        "Use \u2705 for success, \u274c for failure, \u26a0\ufe0f for warnings. "
        "NEVER claim success if success=false. NEVER fabricate values."
    )
    try:
        return _llm(
            ACTION_REPLY_SYSTEM,
            [{"role": "user", "content":
                f"User asked: {user_msg}\n\nOperation: {action_name}\nResult:\n{result_json}"}],
            max_tokens=600,
            force_provider=force_prov,
        )
    except Exception:
        if succeeded:
            return "\u2705 `{}` completed. {}".format(action_name, _j.dumps(
                {k: v for k, v in action_result.items() if k != "success"}, default=str))
        else:
            return "\u274c `{}` failed: {}".format(action_name, action_result.get("error", "unknown error"))


def _chat_inner(payload: ChatPayload, x_user: str):
    global _chat_action_count
    import json as _j
    from app.llm.claude import _provider, _llm
    from app.core.ratelimit import check_chat, check_action
    from app.core.audit import audit_log

    force_prov = payload.provider or ""
    history = [{"role": m.role, "content": m.content} for m in payload.history]
    catalogue = _get_catalogue()

    allowed, remaining = check_chat(x_user)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — max 20 messages per minute.")

    action_result = None
    action_taken  = None
    reply         = ""
    _suggestions  = []

    # Path A: user confirmed a pending action
    if payload.confirmed and payload.pending_action:
        action_name = payload.pending_action
        params      = payload.pending_params or {}
        action_def  = catalogue.get(action_name)
        if action_def:
            act_ok, _ = check_action(x_user)
            if not act_ok:
                raise HTTPException(status_code=429, detail="Action rate limit exceeded — max 10 operations per minute.")
            if payload.dry_run:
                reply = f"**Dry-run:** Would execute `{action_name}` with params `{_j.dumps(params)}`.\nNo changes made."
            else:
                try:
                    action_result = action_def["handler"](params)
                    action_taken  = action_name
                    _chat_action_count += 1
                except Exception as exc:
                    action_result = {"success": False, "error": str(exc)}
                audit_log(user=x_user, action=action_name, params=params,
                          result=action_result or {}, source="chat")
                reply = _build_action_reply(action_name, payload.message, action_result, force_prov, _llm, _j)
        else:
            reply = "Sorry, I could not find that operation. Please try again."

    # Path B: cancellation
    if not reply:
        _cancel = {"no", "cancel", "nope", "stop", "abort", "never mind", "nevermind"}
        if payload.message.lower().strip().rstrip(".,!") in _cancel:
            reply = "Got it — operation cancelled."

    if not reply:
        _conv_context = ""
        try:
            from app.chat.memory import get_history as _get_hist
            _sid_for_ctx = payload.session_id or f"chat-{x_user}"
            _hist = _get_hist(_sid_for_ctx, max_messages=6)
            if _hist:
                _conv_context = "\n".join(
                    f"{getattr(m,'role','?').upper()}: {getattr(m,'content','')[:300]}"
                    for m in _hist
                )
        except Exception:
            pass
        intent_data = _detect_intent(payload.message, force_prov, conv_context=_conv_context)

        if intent_data.get("intent") == "action":
            action_name = intent_data.get("action", "")
            params      = intent_data.get("params", {})
            action_def  = catalogue.get(action_name)

            # EC2 instance ID validation
            if action_name in ("start_ec2", "stop_ec2", "reboot_ec2"):
                raw_iid = params.get("instance_id", "")
                if not raw_iid or not str(raw_iid).startswith("i-"):
                    from app.chat.intelligence import _ec2_session_cache
                    sid = payload.session_id or f"chat-{x_user}"
                    cached = _ec2_session_cache.get(sid, [])
                    if not cached:
                        try:
                            from app.integrations.aws_ops import list_ec2_instances
                            result = list_ec2_instances()
                            instances = result.get("instances", [])
                            if instances:
                                _ec2_session_cache[sid] = [
                                    {"id": i["id"], "name": i.get("name",""), "state": i.get("state","")}
                                    for i in instances
                                ]
                                cached = _ec2_session_cache[sid]
                        except Exception:
                            pass
                    if len(cached) == 1:
                        params = dict(params)
                        params["instance_id"] = cached[0]["id"]
                    elif len(cached) > 1:
                        names = ", ".join(f'{i["id"]} ({i.get("name","?")}, {i.get("state","?")})' for i in cached[:5])
                        reply = f"Multiple EC2 instances found: {names}\n\nWhich one should I {action_name.replace('_ec2','')}? Please specify the instance ID."
                        used_provider = force_prov or _provider or "none"
                        return {"reply": reply, "sources": [], "llm_provider": used_provider,
                                "action_taken": None, "action_result": None, "action_count": _chat_action_count,
                                "pending_action": None, "pending_params": None, "needs_confirm": False}
                    else:
                        reply = "I couldn't find any EC2 instances in your AWS account."
                        used_provider = force_prov or _provider or "none"
                        return {"reply": reply, "sources": [], "llm_provider": used_provider,
                                "action_taken": None, "action_result": None, "action_count": _chat_action_count,
                                "pending_action": None, "pending_params": None, "needs_confirm": False}

            if not action_def:
                reply = (
                    f"I understood you want to **{action_name.replace('_', ' ')}**, "
                    f"but that operation is not available yet. "
                    f"I can restart/scale K8s deployments, start/stop/reboot EC2, manage ECS/Lambda/RDS, "
                    f"query CloudWatch logs and alarms, create GitHub issues or Jira tickets, "
                    f"and run the full incident pipeline."
                )
            elif action_name in _CONFIRM_REQUIRED:
                if payload.dry_run:
                    reply = f"**Dry-run:** Would execute `{action_name}` with params `{_j.dumps(params)}`.\nNo changes made."
                else:
                    reply = _confirmation_message(action_name, params)
                    used_provider = force_prov or _provider or "none"
                    return {
                        "reply":          reply,
                        "sources":        [],
                        "llm_provider":   used_provider,
                        "action_taken":   None,
                        "action_result":  None,
                        "action_count":   _chat_action_count,
                        "pending_action": action_name,
                        "pending_params": params,
                        "needs_confirm":  True,
                    }
            else:
                try:
                    action_result = action_def["handler"](params)
                    action_taken  = action_name
                    _chat_action_count += 1
                except Exception as exc:
                    action_result = {"success": False, "error": str(exc)}
                audit_log(user=x_user, action=action_name, params=params,
                          result=action_result or {}, source="chat")
                reply = _build_action_reply(action_name, payload.message, action_result, force_prov, _llm, _j)

    # Path C: general question / conversation
    if not reply:
        import uuid as _uuid, logging as _log_m
        sid = payload.session_id or f"chat-{x_user}-default"
        try:
            from app.chat.intelligence import chat_with_intelligence
            _result = chat_with_intelligence(
                message=payload.message,
                session_id=sid,
                incident_context=payload.incident_context,
                preferred_provider=force_prov or None,
                image_data=payload.image_data or None,
                image_type=payload.image_type or None,
            )
            if isinstance(_result, tuple):
                reply, _suggestions = _result
            else:
                reply, _suggestions = _result, []
        except Exception as exc:
            _log_m.getLogger("chat").error("chat_with_intelligence failed: %s", exc, exc_info=True)
            fallback_history = history
            try:
                from app.chat.memory import get_history as _gh
                _mh = _gh(sid, max_messages=10)
                if _mh:
                    fallback_history = [
                        {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                        for m in _mh
                    ]
            except Exception:
                pass
            context: dict = {}
            try:
                from app.integrations.universal_collector import collect_all_context
                context = collect_all_context(hours=2)
            except Exception:
                pass
            from app.llm.claude import chat_devops
            reply = chat_devops(payload.message, fallback_history, context, force_provider=force_prov)

    used_provider = force_prov or "none"
    import uuid as _uuid2
    sid_out = payload.session_id or f"chat-{x_user}-{_uuid2.uuid4().hex[:8]}"
    return {
        "reply":          reply,
        "answer":         reply,
        "session_id":     sid_out,
        "sources":        [],
        "llm_provider":   used_provider,
        "action_taken":   action_taken,
        "action_result":  action_result,
        "action_count":   _chat_action_count,
        "pending_action": None,
        "pending_params": None,
        "needs_confirm":  False,
        "suggestions":    _suggestions,
    }


@router.post("/chat")
def chat(payload: ChatPayload, auth: AuthContext = Depends(require_viewer)):
    """Conversational DevOps AI with confirmation flow, rate limiting, audit log, and dry-run."""
    try:
        return _chat_inner(payload, auth.username)
    except HTTPException:
        raise
    except Exception as exc:
        import traceback, logging, uuid
        err_id = uuid.uuid4().hex[:8]
        logging.getLogger("chat").error("Unhandled chat error [%s]: %s\n%s", err_id, exc, traceback.format_exc())
        return {
            "reply": f"An unexpected error occurred. Please try again. (ref: {err_id})",
            "sources": [], "llm_provider": "none", "action_taken": None,
            "action_result": None, "action_count": _chat_action_count,
            "pending_action": None, "pending_params": None, "needs_confirm": False,
        }


@router.post("/chat/stream")
async def chat_stream(payload: ChatPayload, auth: AuthContext = Depends(require_viewer)):
    """SSE streaming chat endpoint — fast Groq-first streaming with Anthropic fallback."""
    import json as _json
    import threading as _threading
    from app.chat.intelligence import (
        _chat_anthropic_stream,
        _is_dead,
        _build_system_prompt,
        _extract_suggestions,
        _SYSTEM_PROMPT,
    )

    async def _event_generator():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_stream():
            try:
                from app.chat.memory import get_history as _gh, add_message as _am

                sid = payload.session_id or f"chat-{auth.username}-default"
                mem = _gh(sid, max_messages=20)
                history_messages = [
                    {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                    for m in mem
                ]
                _am(sid, "user", payload.message)

                requested_provider = (payload.provider or "").lower()

                # Use Anthropic streaming only if explicitly requested and not dead
                use_anthropic_stream = (
                    requested_provider == "anthropic" and
                    not _is_dead("anthropic")
                )

                if use_anthropic_stream:
                    vision = None
                    if payload.image_data and payload.image_type:
                        media = payload.image_type if payload.image_type.startswith("image/") else f"image/{payload.image_type}"
                        vision = [
                            {"type": "image", "source": {"type": "base64", "media_type": media, "data": payload.image_data}},
                            {"type": "text", "text": payload.message},
                        ]

                    full_text = ""
                    for token in _chat_anthropic_stream(
                        _SYSTEM_PROMPT, history_messages, payload.message, sid,
                        vision_content=vision,
                    ):
                        full_text += token
                        loop.call_soon_threadsafe(queue.put_nowait,
                            _json.dumps({"type": "chunk", "text": token}))

                    clean_text, suggestions = _extract_suggestions(full_text)
                    _am(sid, "assistant", clean_text or full_text)
                    loop.call_soon_threadsafe(queue.put_nowait,
                        _json.dumps({"type": "done", "suggestions": suggestions,
                                     "session_id": sid, "llm_provider": "anthropic"}))
                else:
                    # Fast path: Groq-first, send full reply as one chunk
                    result = _chat_inner(payload, auth.username)
                    reply = result.get("reply", "")
                    suggestions = result.get("suggestions", [])
                    loop.call_soon_threadsafe(queue.put_nowait,
                        _json.dumps({"type": "chunk", "text": reply}))
                    loop.call_soon_threadsafe(queue.put_nowait,
                        _json.dumps({"type": "done", "suggestions": suggestions,
                                     "session_id": sid, "llm_provider": result.get("llm_provider", "groq")}))

            except Exception as exc:
                import logging as _lg
                _lg.getLogger("chat.stream").error("stream_failed: %s", exc, exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait,
                    _json.dumps({"type": "error", "message": "AI response failed. Please try again."}))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        _threading.Thread(target=_run_stream, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(_event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/chat/action_count")
def chat_action_count():
    return {"count": _chat_action_count}


@router.get("/chat/sessions")
def list_chat_sessions(auth: AuthContext = Depends(require_viewer)):
    try:
        from app.chat.memory import list_sessions
        return {"sessions": list_sessions()}
    except Exception as e:
        return {"sessions": [], "error": str(e)}


@router.get("/chat/history/{session_id}")
def get_chat_history(session_id: str, auth: AuthContext = Depends(require_viewer)):
    """Return stored conversation history for a session."""
    try:
        from app.chat.memory import get_history
        msgs = get_history(session_id, max_messages=50)
        history = []
        for m in msgs:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", str(m))
            history.append({"role": role, "content": content})
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        return {"session_id": session_id, "messages": [], "error": str(e)}
