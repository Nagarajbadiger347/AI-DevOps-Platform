"""ActionRegistry — maps action type strings to integration handler functions.

Each handler receives the action dict and returns a result dict.
All handlers are synchronous (integrations use blocking SDKs).
Add new action types here without touching the Executor.
"""
from __future__ import annotations

from typing import Callable


# ── Kubernetes ──────────────────────────────────────────────────────────────

def _k8s_restart(action: dict) -> dict:
    from app.integrations.k8s_ops import restart_deployment
    return restart_deployment(
        namespace=action.get("namespace", "default"),
        deployment=action["deployment"],
    )


def _k8s_scale(action: dict) -> dict:
    from app.integrations.k8s_ops import scale_deployment
    return scale_deployment(
        namespace=action.get("namespace", "default"),
        deployment=action["deployment"],
        replicas=int(action.get("replicas", 2)),
    )


# ── AWS EC2 ─────────────────────────────────────────────────────────────────

def _ec2_start(action: dict) -> dict:
    from app.integrations.aws_ops import start_ec2_instance
    return start_ec2_instance(
        instance_id=action["instance_id"],
        region=action.get("region", ""),
    )


def _ec2_stop(action: dict) -> dict:
    from app.integrations.aws_ops import stop_ec2_instance
    return stop_ec2_instance(
        instance_id=action["instance_id"],
        region=action.get("region", ""),
    )


def _ec2_reboot(action: dict) -> dict:
    from app.integrations.aws_ops import reboot_ec2_instance
    return reboot_ec2_instance(
        instance_id=action["instance_id"],
        region=action.get("region", ""),
    )


# ── AWS ECS ─────────────────────────────────────────────────────────────────

def _ecs_scale(action: dict) -> dict:
    from app.integrations.aws_ops import scale_ecs_service
    return scale_ecs_service(
        cluster=action.get("cluster", "default"),
        service=action["service"],
        desired_count=int(action.get("desired_count", action.get("replicas", 1))),
    )


def _ecs_redeploy(action: dict) -> dict:
    from app.integrations.aws_ops import force_new_ecs_deployment
    return force_new_ecs_deployment(
        cluster=action.get("cluster", "default"),
        service=action["service"],
    )


# ── AWS Lambda ──────────────────────────────────────────────────────────────

def _lambda_invoke(action: dict) -> dict:
    from app.integrations.aws_ops import invoke_lambda
    return invoke_lambda(
        function_name=action["function_name"],
        payload=action.get("payload", {}),
    )


# ── AWS RDS ─────────────────────────────────────────────────────────────────

def _rds_reboot(action: dict) -> dict:
    from app.integrations.aws_ops import reboot_rds_instance
    return reboot_rds_instance(
        db_instance_id=action["db_instance_id"],
    )


# ── Investigate (manual / informational) ─────────────────────────────────────

def _investigate(action: dict) -> dict:
    """Investigate action — no automation possible, just logs the recommendation."""
    return {
        "success": True,
        "action": f"Manual investigation required: {action.get('description', '')}",
        "target": action.get("target", ""),
        "note": "This is a manual step — no automated action was taken.",
    }


# ── Notifications / Ticketing ────────────────────────────────────────────────

def _create_jira(action: dict) -> dict:
    from app.integrations.jira import create_incident
    return create_incident(
        summary=action.get("summary", "AI-generated incident ticket"),
        description=action.get("description", ""),
    )


def _slack_notify(action: dict) -> dict:
    from app.integrations.slack import post_message
    return post_message(
        channel=action.get("channel", "#incidents"),
        text=action.get("message", ""),
    )


def _create_pr(action: dict) -> dict:
    from app.integrations.github import create_incident_pr
    return create_incident_pr(
        incident_id=action.get("incident_id", "ai-generated"),
        title=action.get("title", "AI-generated fix"),
        body=action.get("body", ""),
        file_changes=action.get("file_patches") or action.get("file_changes"),
    )


def _opsgenie_alert(action: dict) -> dict:
    from app.integrations.opsgenie import notify_on_call
    return notify_on_call(
        message=action.get("message", ""),
        alias=action.get("alias", "ai-alert"),
    )


# ── Registry ────────────────────────────────────────────────────────────────

ACTION_REGISTRY: dict[str, Callable[[dict], dict]] = {
    # Manual / informational
    "investigate":    _investigate,
    # Kubernetes
    "k8s_restart":    _k8s_restart,
    "k8s_scale":      _k8s_scale,
    # EC2
    "ec2_start":      _ec2_start,
    "ec2_stop":       _ec2_stop,
    "ec2_reboot":     _ec2_reboot,
    # ECS
    "ecs_scale":      _ecs_scale,
    "ecs_redeploy":   _ecs_redeploy,
    # Lambda
    "lambda_invoke":  _lambda_invoke,
    # RDS
    "rds_reboot":     _rds_reboot,
    # Notifications / tickets
    "create_jira":    _create_jira,
    "slack_notify":   _slack_notify,
    "create_pr":      _create_pr,
    "opsgenie_alert": _opsgenie_alert,
}
