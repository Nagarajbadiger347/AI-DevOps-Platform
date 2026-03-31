"""ActionRegistry — maps action type strings to integration handler functions.

Each handler receives the action dict and returns a result dict.
All handlers are synchronous (integrations use blocking SDKs).
Add new action types here without touching the Executor.
"""
from __future__ import annotations

from typing import Callable


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
    # incident_id not available in action, use title as fallback key
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
    "k8s_restart":    _k8s_restart,
    "k8s_scale":      _k8s_scale,
    "create_jira":    _create_jira,
    "slack_notify":   _slack_notify,
    "create_pr":      _create_pr,
    "opsgenie_alert": _opsgenie_alert,
}
