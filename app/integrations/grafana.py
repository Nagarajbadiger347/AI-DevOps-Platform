"""Grafana integration — fetch firing alerts, annotations, and dashboard data.

Requires env vars:
  GRAFANA_URL   e.g. http://localhost:3000
  GRAFANA_TOKEN service account token (or API key)
"""

import json
import os
import urllib.request as _urllib
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GRAFANA_URL   = os.getenv("GRAFANA_URL", "").rstrip("/")
GRAFANA_TOKEN = os.getenv("GRAFANA_TOKEN", "")


def _get(path: str) -> dict:
    if not GRAFANA_URL or not GRAFANA_TOKEN:
        return {"success": False, "error": "Grafana not configured (GRAFANA_URL / GRAFANA_TOKEN)"}
    url = f"{GRAFANA_URL}{path}"
    req = _urllib.Request(url, headers={
        "Authorization": f"Bearer {GRAFANA_TOKEN}",
        "Content-Type": "application/json",
    })
    try:
        r = _urllib.urlopen(req, timeout=3)
        return {"success": True, "data": json.loads(r.read())}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_alerts() -> dict:
    """Fetch all Grafana alert rules and their current state."""
    result = _get("/api/alertmanager/grafana/api/v2/alerts")
    if not result["success"]:
        return result
    alerts = result["data"] if isinstance(result["data"], list) else []
    parsed = []
    for a in alerts:
        labels = a.get("labels", {})
        parsed.append({
            "name":       labels.get("alertname", ""),
            "state":      a.get("status", {}).get("state", ""),
            "severity":   labels.get("severity", ""),
            "namespace":  labels.get("namespace", ""),
            "service":    labels.get("service", ""),
            "summary":    a.get("annotations", {}).get("summary", ""),
            "starts_at":  a.get("startsAt", ""),
        })
    return {"success": True, "alerts": parsed, "count": len(parsed)}


def get_firing_alerts() -> dict:
    """Return only ALERTING (firing) alerts from Grafana."""
    result = get_alerts()
    if not result["success"]:
        return result
    firing = [a for a in result["alerts"] if a["state"].lower() in ("alerting", "firing")]
    return {"success": True, "firing_alerts": firing, "count": len(firing)}


def get_annotations(hours: int = 2) -> dict:
    """Fetch recent annotations (deployments, incidents marked in Grafana dashboards)."""
    import time
    from_ms = int((time.time() - hours * 3600) * 1000)
    to_ms   = int(time.time() * 1000)
    result  = _get(f"/api/annotations?from={from_ms}&to={to_ms}&limit=50")
    if not result["success"]:
        return result
    annotations = [
        {
            "id":      a.get("id"),
            "text":    a.get("text", ""),
            "tags":    a.get("tags", []),
            "time":    a.get("time"),
            "type":    a.get("type", ""),
            "dashboard": a.get("dashboardUID", ""),
        }
        for a in (result["data"] if isinstance(result["data"], list) else [])
    ]
    return {"success": True, "annotations": annotations, "count": len(annotations)}


def get_datasources() -> dict:
    """List configured Grafana datasources."""
    result = _get("/api/datasources")
    if not result["success"]:
        return result
    sources = [
        {"id": d.get("id"), "name": d.get("name"), "type": d.get("type"), "url": d.get("url")}
        for d in (result["data"] if isinstance(result["data"], list) else [])
    ]
    return {"success": True, "datasources": sources, "count": len(sources)}
