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

    return context


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
