"""GitLab integration — pipelines, deployments, merge requests, events.

Requires env vars:
  GITLAB_URL      e.g. https://gitlab.com  (or self-hosted)
  GITLAB_TOKEN    personal access token or project access token
  GITLAB_PROJECT  project ID or namespace/project slug
"""

import json
import os
import urllib.request as _urllib
import urllib.parse as _parse
import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GITLAB_URL     = os.getenv("GITLAB_URL", "https://gitlab.com").rstrip("/")
GITLAB_TOKEN   = os.getenv("GITLAB_TOKEN", "")
GITLAB_PROJECT = os.getenv("GITLAB_PROJECT", "")


def _get(path: str, params: dict = None) -> dict:
    if not GITLAB_TOKEN or not GITLAB_PROJECT:
        return {"success": False, "error": "GitLab not configured (GITLAB_TOKEN / GITLAB_PROJECT)"}
    url = f"{GITLAB_URL}/api/v4{path}"
    if params:
        url += "?" + _parse.urlencode(params)
    req = _urllib.Request(url, headers={
        "PRIVATE-TOKEN": GITLAB_TOKEN,
        "Content-Type": "application/json",
    })
    try:
        r = _urllib.urlopen(req, timeout=10)
        return {"success": True, "data": json.loads(r.read())}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _project_path() -> str:
    return "/projects/" + _parse.quote(GITLAB_PROJECT, safe="")


def list_pipelines(hours: int = 6) -> dict:
    """List recent pipelines for the configured project."""
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).isoformat()
    result = _get(f"{_project_path()}/pipelines", {"updated_after": since, "per_page": 20, "order_by": "updated_at"})
    if not result["success"]:
        return result
    pipelines = [
        {
            "id":         p["id"],
            "status":     p["status"],
            "ref":        p.get("ref", ""),
            "sha":        p.get("sha", "")[:10],
            "created_at": p.get("created_at", ""),
            "updated_at": p.get("updated_at", ""),
            "web_url":    p.get("web_url", ""),
        }
        for p in (result["data"] if isinstance(result["data"], list) else [])
    ]
    return {"success": True, "pipelines": pipelines, "count": len(pipelines)}


def get_failed_pipelines(hours: int = 6) -> dict:
    """Return only failed/canceled pipelines in the last N hours."""
    result = list_pipelines(hours=hours)
    if not result["success"]:
        return result
    failed = [p for p in result["pipelines"] if p["status"] in ("failed", "canceled")]
    return {"success": True, "failed_pipelines": failed, "count": len(failed)}


def list_merge_requests(state: str = "opened") -> dict:
    """List merge requests by state: opened | closed | merged."""
    result = _get(f"{_project_path()}/merge_requests", {"state": state, "per_page": 20, "order_by": "updated_at"})
    if not result["success"]:
        return result
    mrs = [
        {
            "iid":        mr["iid"],
            "title":      mr["title"],
            "state":      mr["state"],
            "author":     mr.get("author", {}).get("username", ""),
            "source_branch": mr.get("source_branch", ""),
            "target_branch": mr.get("target_branch", ""),
            "created_at": mr.get("created_at", ""),
            "web_url":    mr.get("web_url", ""),
        }
        for mr in (result["data"] if isinstance(result["data"], list) else [])
    ]
    return {"success": True, "merge_requests": mrs, "count": len(mrs)}


def list_deployments(hours: int = 24) -> dict:
    """List recent deployments for the configured project."""
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).isoformat()
    result = _get(f"{_project_path()}/deployments", {"updated_after": since, "per_page": 10, "order_by": "updated_at", "sort": "desc"})
    if not result["success"]:
        return result
    deploys = [
        {
            "id":          d["id"],
            "status":      d.get("status", ""),
            "environment": d.get("environment", {}).get("name", ""),
            "ref":         d.get("ref", ""),
            "sha":         d.get("sha", "")[:10],
            "created_at":  d.get("created_at", ""),
            "web_url":     d.get("deployable", {}).get("web_url", ""),
        }
        for d in (result["data"] if isinstance(result["data"], list) else [])
    ]
    return {"success": True, "deployments": deploys, "count": len(deploys)}


def get_project_events(hours: int = 6) -> dict:
    """Get recent project events (pushes, comments, merges)."""
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d")
    result = _get(f"{_project_path()}/events", {"after": since, "per_page": 30})
    if not result["success"]:
        return result
    events = [
        {
            "action":     e.get("action_name", ""),
            "author":     e.get("author", {}).get("username", ""),
            "target":     e.get("target_title", ""),
            "target_type": e.get("target_type", ""),
            "created_at": e.get("created_at", ""),
        }
        for e in (result["data"] if isinstance(result["data"], list) else [])
    ]
    return {"success": True, "events": events, "count": len(events)}
