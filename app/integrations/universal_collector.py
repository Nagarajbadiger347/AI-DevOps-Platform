"""Universal context collector — gathers observability data from every configured
integration in parallel and returns a single structured dict for AI analysis.

Integrations attempted (each silently skipped if not configured):
  AWS      — EC2, ECS, Lambda, RDS, S3, SQS, DynamoDB, Route53, CloudWatch, CloudTrail
  Grafana  — firing alerts, annotations
  Kubernetes — pods, deployments, cluster events
  GitHub   — recent commits, recent PRs
  GitLab   — recent pipelines, deployments
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

# ── Lazy imports so missing/unconfigured integrations don't crash ─────────────

def _try(fn: Callable, *args, **kwargs):
    """Call fn(*args, **kwargs); return None on any exception."""
    try:
        result = fn(*args, **kwargs)
        # Treat results that explicitly report failure as None
        if isinstance(result, dict) and result.get("success") is False:
            return None
        return result
    except Exception:
        return None


def collect_all_context(hours: int = 2) -> dict:
    """Collect observability data from every configured integration in parallel.

    Returns:
        {
          "configured": ["aws", "grafana", "k8s", "github", ...],
          "aws": { "ec2": ..., "alarms": ..., ... },
          "grafana": { ... },
          "k8s": { ... },
          "github": { ... },
          "gitlab": { ... },
        }
    """
    from app.integrations.aws_ops import (
        list_ec2_instances, get_ec2_status_checks, list_cloudwatch_alarms,
        list_ecs_services, get_stopped_ecs_tasks, list_lambda_functions,
        list_rds_instances, get_cloudtrail_events, list_s3_buckets,
        list_sqs_queues, list_dynamodb_tables, list_route53_healthchecks,
        list_sns_topics,
    )
    from app.integrations.grafana import get_firing_alerts, get_annotations
    from app.integrations.k8s_ops import (
        list_pods, list_deployments, get_cluster_events, get_unhealthy_pods,
    )
    from app.integrations.github import get_recent_commits, get_recent_prs
    from app.integrations.gitlab_ops import list_pipelines, get_failed_pipelines

    # Define all tasks
    task_map: dict[str, Callable] = {
        # AWS
        "aws_ec2":          lambda: list_ec2_instances(),
        "aws_ec2_status":   lambda: get_ec2_status_checks(),
        "aws_alarms":       lambda: list_cloudwatch_alarms(),
        "aws_alarms_firing": lambda: list_cloudwatch_alarms(state="ALARM"),
        "aws_ecs":          lambda: list_ecs_services(),
        "aws_ecs_stopped":  lambda: get_stopped_ecs_tasks(),
        "aws_lambda":       lambda: list_lambda_functions(),
        "aws_rds":          lambda: list_rds_instances(),
        "aws_cloudtrail":   lambda: get_cloudtrail_events(hours=hours),
        "aws_s3":           lambda: list_s3_buckets(),
        "aws_sqs":          lambda: list_sqs_queues(),
        "aws_dynamodb":     lambda: list_dynamodb_tables(),
        "aws_route53":      lambda: list_route53_healthchecks(),
        "aws_sns":          lambda: list_sns_topics(),
        # Grafana
        "grafana_alerts":      lambda: get_firing_alerts(),
        "grafana_annotations": lambda: get_annotations(hours=hours),
        # Kubernetes
        "k8s_pods":         lambda: list_pods(),
        "k8s_deployments":  lambda: list_deployments(),
        "k8s_events":       lambda: get_cluster_events(),
        "k8s_unhealthy":    lambda: get_unhealthy_pods(),
        # GitHub
        "github_commits": lambda: get_recent_commits(hours=hours),
        "github_prs":     lambda: get_recent_prs(hours=hours * 12),
        # GitLab
        "gitlab_pipelines":       lambda: list_pipelines(hours=hours),
        "gitlab_failed_pipelines": lambda: get_failed_pipelines(hours=hours),
    }

    results: dict = {}
    configured_sources: set = set()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_try, fn): name for name, fn in task_map.items()}
        for future in as_completed(futures, timeout=30):
            name   = futures[future]
            result = future.result()
            if result is not None:
                results[name] = result
                source = name.split("_")[0]
                configured_sources.add(source)

    # Structure into nested groups
    context: dict = {"configured": sorted(configured_sources)}

    def _group(prefix: str) -> dict:
        return {k[len(prefix)+1:]: v for k, v in results.items() if k.startswith(prefix + "_")}

    for source in ("aws", "grafana", "k8s", "github", "gitlab"):
        grp = _group(source)
        if grp:
            context[source] = grp

    # Build unified change timeline — ordered newest-first, cross-source
    context["change_timeline"] = _build_change_timeline(results, hours)

    return context


def _build_change_timeline(results: dict, hours: int) -> dict:
    """Merge GitHub commits, CloudTrail events, and K8s deployment changes into
    a single ordered timeline for AI root-cause correlation.

    Returns:
        {
          "window_hours": 2,
          "total_changes": 12,
          "events": [
            {"ts": "2026-05-04T10:30:00Z", "source": "github", "type": "commit",
             "summary": "fix: increase memory limit", "actor": "alice", "ref": "main"},
            {"ts": "2026-05-04T10:25:00Z", "source": "aws_cloudtrail", "type": "UpdateService",
             "summary": "ECS UpdateService payments-api", "actor": "ci-deploy"},
            ...
          ]
        }
    """
    import datetime
    events: list[dict] = []

    # GitHub commits
    gh_commits = results.get("github_commits", {})
    for c in (gh_commits.get("commits", []) if isinstance(gh_commits, dict) else []):
        ts = c.get("timestamp") or c.get("date") or c.get("authored_at") or ""
        events.append({
            "ts":      ts,
            "source":  "github",
            "type":    "commit",
            "summary": (c.get("message") or "")[:120],
            "actor":   c.get("author") or c.get("committer") or "",
            "ref":     c.get("branch") or c.get("ref") or "",
            "sha":     (c.get("sha") or c.get("id") or "")[:8],
        })

    # GitHub PRs merged in window
    gh_prs = results.get("github_prs", {})
    for pr in (gh_prs.get("prs", []) if isinstance(gh_prs, dict) else []):
        if pr.get("state") == "closed" and pr.get("merged"):
            ts = pr.get("merged_at") or pr.get("closed_at") or ""
            events.append({
                "ts":      ts,
                "source":  "github",
                "type":    "pr_merged",
                "summary": f"PR #{pr.get('number')}: {(pr.get('title') or '')[:100]}",
                "actor":   pr.get("user") or pr.get("author") or "",
                "ref":     pr.get("base") or "",
            })

    # AWS CloudTrail — filter to deployment/mutation events (skip reads)
    _CT_WRITE_PREFIXES = (
        "Update", "Create", "Delete", "Put", "Set", "Modify",
        "Register", "Deregister", "Deploy", "Run", "Start", "Stop", "Terminate",
    )
    ct_events = results.get("aws_cloudtrail", {})
    for e in (ct_events.get("events", []) if isinstance(ct_events, dict) else []):
        event_name = e.get("event_name") or e.get("eventName") or ""
        if not any(event_name.startswith(p) for p in _CT_WRITE_PREFIXES):
            continue
        ts = e.get("event_time") or e.get("eventTime") or ""
        resource = ""
        for r in e.get("resources", []):
            if isinstance(r, dict):
                resource = r.get("resource_name") or r.get("ARN") or ""
                break
        events.append({
            "ts":      str(ts),
            "source":  "aws_cloudtrail",
            "type":    event_name,
            "summary": f"{event_name} {resource}".strip(),
            "actor":   e.get("user") or e.get("username") or e.get("userIdentity", {}).get("arn", "") if isinstance(e.get("userIdentity"), dict) else e.get("user", ""),
            "region":  e.get("aws_region") or e.get("awsRegion") or "",
        })

    # K8s deployments with recent changes (rollout events)
    k8s_events = results.get("k8s_events", {})
    for ev in (k8s_events.get("events", []) if isinstance(k8s_events, dict) else []):
        if ev.get("kind") in ("Deployment", "ReplicaSet") and ev.get("reason") in (
            "ScalingReplicaSet", "Killing", "Pulled", "Started",
        ):
            events.append({
                "ts":      ev.get("last_seen") or ev.get("first_seen") or "",
                "source":  "kubernetes",
                "type":    ev.get("reason", ""),
                "summary": f"{ev.get('kind','')}/{ev.get('name','')} — {ev.get('message','')[:100]}",
                "actor":   "k8s-controller",
                "namespace": ev.get("namespace", ""),
            })

    # Sort newest-first (ISO timestamps sort lexicographically)
    events.sort(key=lambda e: e.get("ts") or "", reverse=True)

    return {
        "window_hours":  hours,
        "total_changes": len(events),
        "events":        events[:50],   # cap at 50 to keep AI context compact
    }


def summarize_health(context: dict) -> dict:
    """Extract quick health signals from universal context for topbar / status display."""
    issues = []

    # AWS EC2 unhealthy instances
    for s in (context.get("aws", {}).get("ec2_status", {}).get("statuses", [])):
        if not s.get("healthy"):
            issues.append(f"EC2 {s['instance_id']} status check failing")

    # AWS alarms firing
    for a in (context.get("aws", {}).get("alarms_firing", {}).get("alarms", [])):
        issues.append(f"CloudWatch ALARM: {a['name']}")

    # Grafana alerts
    for a in (context.get("grafana", {}).get("alerts", {}).get("firing_alerts", [])):
        issues.append(f"Grafana alert: {a.get('name', '?')} ({a.get('severity', '')})")

    # K8s unhealthy pods
    for p in (context.get("k8s", {}).get("unhealthy", {}).get("unhealthy_pods", [])):
        issues.append(f"K8s pod {p['name']} in {p['namespace']}: {p['phase']}")

    # K8s warning events (top 3)
    for e in (context.get("k8s", {}).get("events", {}).get("events", []))[:3]:
        issues.append(f"K8s event: {e['reason']} on {e['kind']}/{e['name']}")

    # GitLab failed pipelines
    for p in (context.get("gitlab", {}).get("failed_pipelines", {}).get("failed_pipelines", [])):
        issues.append(f"GitLab pipeline {p['id']} failed on {p.get('ref','?')}")

    return {
        "healthy":      len(issues) == 0,
        "issue_count":  len(issues),
        "issues":       issues[:20],
        "sources":      context.get("configured", []),
    }
