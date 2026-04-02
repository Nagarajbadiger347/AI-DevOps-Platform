"""Webhook processors for external alerting systems.

Supported sources:
  - Grafana (v9+ alert webhook format)
  - AWS CloudWatch via SNS
  - OpsGenie
  - PagerDuty (v3 webhooks)

Each processor parses the incoming payload, normalises it, and (where
appropriate) injects it into the monitoring pipeline via receive_external_alert().
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def _inject(source: str, alert: dict) -> None:
    """Safely call receive_external_alert — imported lazily to avoid circular deps."""
    try:
        from app.monitoring.loop import receive_external_alert
        receive_external_alert(source, alert)
    except Exception as exc:
        logger.warning("webhook_inject_failed", extra={"source": source, "error": str(exc)})


# ---------------------------------------------------------------------------
# Grafana (v9+ unified alerting)
# ---------------------------------------------------------------------------

def process_grafana_webhook(payload: dict) -> dict:
    """Parse a Grafana alert webhook (v9+ format).

    Expected top-level keys: status, alerts (array), title, message.
    Each alert has: labels, annotations, status, generatorURL.
    """
    status = payload.get("status", "").lower()  # "firing" | "resolved"
    processed_alerts = []

    for alert in payload.get("alerts", []):
        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        alert_name = labels.get("alertname") or annotations.get("summary") or "unknown"
        state = alert.get("status", status).lower()
        generator_url = alert.get("generatorURL", "")

        processed = {
            "alert_type": "grafana_alert",
            "name": alert_name,
            "resource_id": alert_name,
            "state": state,
            "labels": labels,
            "annotations": annotations,
            "generator_url": generator_url,
            "description": (
                annotations.get("description")
                or annotations.get("summary")
                or f"Grafana alert {alert_name} is {state}"
            ),
        }
        processed_alerts.append(processed)

        if state == "firing":
            _inject("grafana", processed)

    result = {
        "source": "grafana",
        "status": status,
        "alert_count": len(processed_alerts),
        "alerts": processed_alerts,
    }
    return result


# ---------------------------------------------------------------------------
# AWS CloudWatch via SNS notification
# ---------------------------------------------------------------------------

def process_cloudwatch_webhook(payload: dict) -> dict:
    """Parse an AWS SNS → CloudWatch alarm notification.

    SNS wraps the CloudWatch JSON in a 'Message' string field.
    """
    # SNS wraps the real payload in a JSON-encoded 'Message' field
    message_raw = payload.get("Message", "")
    if isinstance(message_raw, str):
        try:
            message = json.loads(message_raw)
        except (json.JSONDecodeError, ValueError):
            message = {}
    else:
        message = message_raw or {}

    alarm_name = message.get("AlarmName") or payload.get("AlarmName", "unknown")
    state = message.get("NewStateValue") or payload.get("NewStateValue", "UNKNOWN")
    metric_name = (
        message.get("Trigger", {}).get("MetricName")
        or message.get("MetricName", "")
    )
    threshold = (
        message.get("Trigger", {}).get("Threshold")
        or message.get("Threshold", "")
    )
    reason = message.get("NewStateReason") or payload.get("NewStateReason", "")

    processed = {
        "alert_type": "cloudwatch_alarm",
        "name": alarm_name,
        "resource_id": alarm_name,
        "state": state,
        "metric": metric_name,
        "threshold": threshold,
        "reason": reason,
        "description": (
            f"CloudWatch alarm {alarm_name} entered {state} state. "
            f"Metric: {metric_name}, Threshold: {threshold}. Reason: {reason}"
        ),
    }

    if state == "ALARM":
        _inject("cloudwatch", processed)

    return processed


# ---------------------------------------------------------------------------
# OpsGenie
# ---------------------------------------------------------------------------

def process_opsgenie_webhook(payload: dict) -> dict:
    """Parse an OpsGenie webhook payload."""
    alert_data = payload.get("alert", payload)  # some versions nest under 'alert'
    alert_id = alert_data.get("alertId") or alert_data.get("id", "unknown")
    message = alert_data.get("message") or payload.get("message", "")
    priority = alert_data.get("priority") or payload.get("priority", "P3")
    source = alert_data.get("source") or payload.get("source", "opsgenie")
    tags = alert_data.get("tags") or payload.get("tags", [])

    processed = {
        "alert_type": "opsgenie_alert",
        "name": message or alert_id,
        "resource_id": str(alert_id),
        "alert_id": alert_id,
        "message": message,
        "priority": priority,
        "source": source,
        "tags": tags,
        "description": f"OpsGenie alert [{priority}]: {message} (id={alert_id})",
    }

    _inject("opsgenie", processed)
    return processed


# ---------------------------------------------------------------------------
# PagerDuty (v3 webhooks)
# ---------------------------------------------------------------------------

def process_pagerduty_webhook(payload: dict) -> dict:
    """Parse a PagerDuty v3 webhook payload.

    PD v3 sends a list of events under 'events'; each event has 'event_type'
    and 'data' containing the incident details.
    """
    processed_events = []

    events = payload.get("events", [payload])  # gracefully handle single-event payloads
    for event in events:
        event_type = event.get("event_type", "")
        data = event.get("data", event)

        incident_id = data.get("id", "unknown")
        title = data.get("title") or data.get("summary", "")
        urgency = data.get("urgency", "high")
        status = data.get("status", "triggered")
        service_name = (
            data.get("service", {}).get("name")
            if isinstance(data.get("service"), dict)
            else data.get("service", "unknown")
        )

        processed = {
            "alert_type": "pagerduty_incident",
            "name": title or incident_id,
            "resource_id": incident_id,
            "incident_id": incident_id,
            "title": title,
            "urgency": urgency,
            "service_name": service_name,
            "status": status,
            "event_type": event_type,
            "description": (
                f"PagerDuty incident [{urgency}] on {service_name}: "
                f"{title} (id={incident_id}, status={status})"
            ),
        }
        processed_events.append(processed)
        _inject("pagerduty", processed)

    return {
        "source": "pagerduty",
        "event_count": len(processed_events),
        "events": processed_events,
    }
