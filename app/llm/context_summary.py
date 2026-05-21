"""Compact prompt-ready summaries of observability context.

The naive approach (used to be `json.dumps(ctx, default=str)[:2000]`) does
two bad things:

1. Wastes input tokens on ARNs, timestamps, and metadata the model doesn't
   need to plan a remediation.
2. Brute character-truncates JSON, often slicing off the most important
   field (e.g. `state: stopped`) when dict ordering puts it last.

These helpers extract only the SRE-relevant signal:
  - alarms currently firing
  - resources in a bad state (stopped / unhealthy / crashlooping)
  - recent changes (deploys, commits, config edits)

Output is a plain dict so callers can either JSON-encode or render it
themselves. Keep this file dependency-free — pure Python, no LLM calls.
"""
from __future__ import annotations

import json
from typing import Any


def _take(seq, n: int) -> list:
    """Safe slice that tolerates None / non-list inputs."""
    if not isinstance(seq, list):
        return []
    return seq[:n]


def summarize_aws_for_prompt(ctx: dict | None, max_items: int = 5) -> dict:
    """Reduce an AWS context blob to its SRE-relevant signal.

    Source keys produced by app.integrations.aws_ops / universal_collector:
      instances:[{id,state,...}], alarms:[{name,state,...}],
      ecs_services:[{service_name,running_count,desired_count}],
      lambda_metrics:[{function_name,errors}],
      rds_instances:[{identifier,status}],
      cloudtrail_events:[{event_name,event_time,resource}],
      log_groups, s3_buckets, dynamodb_tables (mostly inventory; drop)
    """
    if not isinstance(ctx, dict):
        return {"_data_available": False}

    summary: dict[str, Any] = {
        "_data_available": bool(ctx.get("_data_available", True)),
        "region": ctx.get("region", ""),
    }

    # Alarms firing — the single highest-signal field.
    alarms = ctx.get("alarms") or ctx.get("cloudwatch_alarms")
    if isinstance(alarms, list):
        firing = [
            {"name": a.get("name"), "state": a.get("state"), "reason": (a.get("reason") or "")[:120]}
            for a in alarms if isinstance(a, dict) and str(a.get("state", "")).upper() == "ALARM"
        ]
        if firing:
            summary["alarms_firing"] = _take(firing, max_items)

    # EC2 instances not in `running` state.
    instances = ctx.get("instances")
    if isinstance(instances, list):
        bad = [
            {"id": i.get("id"), "state": i.get("state"), "type": i.get("instance_type")}
            for i in instances if isinstance(i, dict) and i.get("state") and i.get("state") != "running"
        ]
        if bad:
            summary["instances_unhealthy"] = _take(bad, max_items)
        summary["instance_count"] = len(instances)

    # ECS services with running != desired.
    ecs = ctx.get("ecs_services")
    if isinstance(ecs, list):
        misscaled = [
            {"service": s.get("service_name"), "running": s.get("running_count"),
             "desired": s.get("desired_count")}
            for s in ecs if isinstance(s, dict)
            and s.get("running_count") is not None
            and s.get("desired_count") is not None
            and s["running_count"] != s["desired_count"]
        ]
        if misscaled:
            summary["ecs_misscaled"] = _take(misscaled, max_items)

    # Lambdas with non-zero errors.
    lambdas = ctx.get("lambda_metrics")
    if isinstance(lambdas, list):
        err_fns = [
            {"function": l.get("function_name"), "errors": l.get("errors") or l.get("error_count")}
            for l in lambdas if isinstance(l, dict)
            and (l.get("errors") or l.get("error_count") or 0)
        ]
        if err_fns:
            summary["lambda_errors"] = _take(err_fns, max_items)

    # RDS in non-available status.
    rds = ctx.get("rds_instances")
    if isinstance(rds, list):
        bad_rds = [
            {"id": r.get("identifier"), "status": r.get("status")}
            for r in rds if isinstance(r, dict)
            and r.get("status") and r.get("status") != "available"
        ]
        if bad_rds:
            summary["rds_unhealthy"] = _take(bad_rds, max_items)

    # Recent CloudTrail mutations (most useful for "what changed?").
    trail = ctx.get("cloudtrail_events")
    if isinstance(trail, list) and trail:
        recent = [
            {"event": e.get("event_name") or e.get("eventName"),
             "user": e.get("user_identity") or e.get("user"),
             "resource": (e.get("resource") or e.get("resource_name") or "")[:80],
             "time": e.get("event_time") or e.get("eventTime")}
            for e in trail if isinstance(e, dict)
        ]
        if recent:
            summary["recent_changes"] = _take(recent, max_items)

    if "note" in ctx:
        summary["note"] = ctx["note"]
    return summary


def summarize_k8s_for_prompt(ctx: dict | None, max_items: int = 5) -> dict:
    """Reduce a Kubernetes context blob to its SRE-relevant signal."""
    if not isinstance(ctx, dict):
        return {"_data_available": False}

    summary: dict[str, Any] = {
        "_data_available": bool(ctx.get("_data_available", True)),
    }

    pods = ctx.get("pods")
    if isinstance(pods, list):
        unhealthy = [
            {"name": p.get("name"), "namespace": p.get("namespace"),
             "phase": p.get("phase"), "restarts": p.get("restarts"),
             "ready": p.get("ready")}
            for p in pods if isinstance(p, dict) and (
                p.get("phase") not in ("Running", "Succeeded", None)
                or p.get("restarts", 0) >= 3
                or p.get("ready") is False
            )
        ]
        if unhealthy:
            summary["pods_unhealthy"] = _take(unhealthy, max_items)
        summary["pod_count"] = len(pods)

    deps = ctx.get("deployments")
    if isinstance(deps, list):
        bad_deps = [
            {"name": d.get("name"), "namespace": d.get("namespace"),
             "ready": d.get("ready"), "desired": d.get("desired")}
            for d in deps if isinstance(d, dict)
            and d.get("desired") is not None
            and d.get("ready") != d.get("desired")
        ]
        if bad_deps:
            summary["deployments_misscaled"] = _take(bad_deps, max_items)

    events = ctx.get("events")
    if isinstance(events, list):
        warnings = [
            {"reason": e.get("reason"), "object": e.get("name"),
             "message": (e.get("message") or "")[:140], "count": e.get("count")}
            for e in events if isinstance(e, dict)
        ]
        if warnings:
            summary["warning_events"] = _take(warnings, max_items)

    if "note" in ctx:
        summary["note"] = ctx["note"]
    return summary


def summarize_github_for_prompt(ctx: dict | None, max_items: int = 3) -> dict:
    """Reduce a GitHub context blob to its SRE-relevant signal — recent
    commits, recent merged PRs, failing workflow runs."""
    if not isinstance(ctx, dict):
        return {"_data_available": False}

    summary: dict[str, Any] = {
        "_data_available": bool(ctx.get("_data_available", True)),
        "repo": ctx.get("repo", ""),
    }

    commits = _take(ctx.get("recent_commits"), max_items)
    if commits:
        summary["recent_commits"] = [
            {"sha": (c.get("sha") or "")[:7],
             "author": c.get("author"),
             "message": (c.get("message") or "")[:120]}
            for c in commits if isinstance(c, dict)
        ]

    prs = _take(ctx.get("recent_prs") or ctx.get("merged_prs"), max_items)
    if prs:
        summary["recent_prs"] = [
            {"number": p.get("number"), "title": (p.get("title") or "")[:120],
             "merged_at": p.get("merged_at")}
            for p in prs if isinstance(p, dict)
        ]

    workflows = ctx.get("workflow_runs")
    if isinstance(workflows, list):
        failing = [
            {"name": w.get("name"), "status": w.get("status"),
             "conclusion": w.get("conclusion"), "url": w.get("html_url")}
            for w in workflows if isinstance(w, dict)
            and w.get("conclusion") in ("failure", "cancelled", "timed_out")
        ]
        if failing:
            summary["workflows_failing"] = _take(failing, max_items)

    return summary


def render_summary_for_prompt(summary: dict, label: str = "") -> str:
    """Render a summary dict as compact JSON, with an optional label prefix.

    Returns an empty string when the summary has no actionable signal —
    callers can use that to skip emitting a whole section."""
    if not summary or not isinstance(summary, dict):
        return ""
    keys = {k for k in summary if k != "_data_available" and not k.startswith("_")}
    if not keys:
        return ""
    body = json.dumps(summary, default=str, separators=(",", ":"))
    return f"{label}: {body}" if label else body
