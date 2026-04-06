"""Grafana health checker.

Aggregates Grafana alert state, datasource connectivity, and recent annotations
into a unified health summary — same contract as aws_checker.py and k8s_checker.py.

Return schema:
    {
        "status":  "healthy" | "degraded" | "unavailable" | "error",
        "success": bool,
        "details": {
            "firing_alerts":   int,
            "total_alerts":    int,
            "firing":          list[dict],   # alert name/severity/summary
            "datasources":     int,
            "recent_events":   list[dict],   # annotations from last 2h
        },
        "firing_alert_names": list[str] | None,
    }

Requires env vars:
    GRAFANA_URL    e.g. http://localhost:3000
    GRAFANA_TOKEN  service account token or API key
"""
from __future__ import annotations


def check_grafana() -> dict:
    """Full Grafana health check — alerts, datasources, recent annotations.

    Returns:
        status  "healthy"     no firing alerts, datasources reachable
                "degraded"    one or more alerts firing
                "unavailable" GRAFANA_URL / GRAFANA_TOKEN not set
                "error"       Grafana API returned an error
    """
    try:
        from app.integrations.grafana import get_alerts, get_datasources, get_annotations
    except ImportError:
        return {
            "status":  "unavailable",
            "success": False,
            "details": "Grafana integration module not available",
        }

    # ── Connectivity check ────────────────────────────────────────────────────
    alerts_result = get_alerts()

    if not alerts_result.get("success"):
        err = alerts_result.get("error", "unknown error")
        if "not configured" in err.lower():
            return {
                "status":  "unavailable",
                "success": False,
                "details": "Grafana not configured — set GRAFANA_URL and GRAFANA_TOKEN in .env",
            }
        return {
            "status":  "error",
            "success": False,
            "details": err,
        }

    # ── Alerts ────────────────────────────────────────────────────────────────
    all_alerts    = alerts_result.get("alerts", [])
    firing_alerts = [
        a for a in all_alerts
        if (a.get("state") or "").lower() in ("alerting", "firing", "pending")
    ]
    firing_names  = [a.get("name", "?") for a in firing_alerts[:10]]

    firing_detail = [
        {
            "name":      a.get("name", ""),
            "severity":  a.get("severity", ""),
            "summary":   a.get("summary", ""),
            "namespace": a.get("namespace", ""),
            "service":   a.get("service", ""),
            "starts_at": a.get("starts_at", ""),
        }
        for a in firing_alerts
    ]

    # ── Datasources ───────────────────────────────────────────────────────────
    datasource_count = 0
    try:
        ds_result = get_datasources()
        if ds_result.get("success"):
            datasource_count = ds_result.get("count", 0)
    except Exception:
        pass

    # ── Recent annotations (deployments, incidents tagged in dashboards) ──────
    recent_events: list[dict] = []
    try:
        ann_result = get_annotations(hours=2)
        if ann_result.get("success"):
            recent_events = [
                {
                    "text": a.get("text", ""),
                    "tags": a.get("tags", []),
                    "time": a.get("time"),
                }
                for a in ann_result.get("annotations", [])[:10]
            ]
    except Exception:
        pass

    # ── Overall status ────────────────────────────────────────────────────────
    overall = "degraded" if firing_alerts else "healthy"

    return {
        "status":             overall,
        "success":            True,
        "firing_alerts":      len(firing_alerts),
        "total_alerts":       len(all_alerts),
        "firing_alert_names": firing_names if firing_names else None,
        "details": {
            "firing":        firing_detail,
            "datasources":   datasource_count,
            "recent_events": recent_events,
        },
    }
