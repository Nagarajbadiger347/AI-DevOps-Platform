"""AWS infrastructure health checker.

Aggregates read-only status across the most important AWS services and returns
a unified health summary.  All actual API calls are delegated to
``app.integrations.aws_ops`` so credentials and boto3 config live in one place.

Return schema (all functions):
    {
        "status":  "healthy" | "degraded" | "error" | "unavailable",
        "success": bool,
        "details": dict | str,   # dict on success, error string on failure
    }
"""
from __future__ import annotations


def check_aws_infrastructure() -> dict:
    """Full AWS health check — EC2, ECS, Lambda, RDS, ALB, CloudWatch, S3, SQS,
    DynamoDB, Route53, SNS.

    Returns a unified status summary across all configured services.
    Partial failures (some services unavailable) result in ``degraded`` rather
    than ``error`` so the caller can distinguish "no credentials" from
    "credentials work but some resources have issues".
    """
    from app.integrations.aws_ops import (
        list_ec2_instances,
        list_s3_buckets,
        list_cloudwatch_alarms,
        list_ecs_services,
        list_lambda_functions,
        list_rds_instances,
        list_sqs_queues,
        list_dynamodb_tables,
        list_route53_healthchecks,
        list_sns_topics,
    )

    collectors = {
        "ec2":        (list_ec2_instances,       "instances"),
        "s3":         (list_s3_buckets,           "buckets"),
        "cloudwatch": (list_cloudwatch_alarms,    "alarms"),
        "ecs":        (list_ecs_services,          "services"),
        "lambda":     (list_lambda_functions,      "functions"),
        "rds":        (list_rds_instances,         "instances"),
        "sqs":        (list_sqs_queues,            "queues"),
        "dynamodb":   (list_dynamodb_tables,       "tables"),
        "route53":    (list_route53_healthchecks,  "health_checks"),
        "sns":        (list_sns_topics,            "topics"),
    }

    details: dict = {}
    errors:  list = []
    unavailable: list = []

    for service, (fn, count_key) in collectors.items():
        try:
            result = fn()
            if result.get("success"):
                items = result.get(count_key, [])
                svc_detail: dict = {
                    "count":   result.get("count", len(items)),
                    "healthy": True,
                }
                # EC2: enrich with running/stopped breakdown
                if service == "ec2" and items:
                    svc_detail["running"] = sum(
                        1 for i in items if (i.get("state") or "").lower() == "running"
                    )
                    svc_detail["stopped"] = sum(
                        1 for i in items if (i.get("state") or "").lower() == "stopped"
                    )
                # ECS: flag services with 0 running tasks
                if service == "ecs" and items:
                    down = [
                        s.get("service_name") or s.get("name", "?")
                        for s in items
                        if int(s.get("desired_count", 0) or 0) > 0
                        and int(s.get("running_count", 0) or 0) == 0
                    ]
                    if down:
                        svc_detail["services_down"] = down
                        svc_detail["healthy"] = False
                        errors.append(f"ecs: {len(down)} service(s) with 0 running tasks")
                # Route53: flag unhealthy health checks
                if service == "route53" and items:
                    unhealthy = [
                        hc.get("id", "?")
                        for hc in items
                        if str(hc.get("status", "")).upper() not in ("HEALTHY", "")
                    ]
                    if unhealthy:
                        svc_detail["unhealthy_checks"] = unhealthy
                        svc_detail["healthy"] = False
                        errors.append(f"route53: {len(unhealthy)} unhealthy health check(s)")
                details[service] = svc_detail
            else:
                err = result.get("error", "unknown error")
                if "credentials" in err.lower() or "not configured" in err.lower() or "no module" in err.lower():
                    unavailable.append(service)
                    details[service] = {"healthy": None, "note": "not configured"}
                else:
                    errors.append(f"{service}: {err}")
                    details[service] = {"healthy": False, "error": err}
        except Exception as exc:
            errors.append(f"{service}: {exc}")
            details[service] = {"healthy": False, "error": str(exc)}

    # ALB/ELB health — separate call since it uses a different list function
    try:
        import boto3 as _boto3
        elb = _boto3.client("elbv2")
        lbs = elb.describe_load_balancers().get("LoadBalancers", [])
        unhealthy_lbs = [
            lb["LoadBalancerName"]
            for lb in lbs
            if lb.get("State", {}).get("Code") != "active"
        ]
        details["alb"] = {
            "count":   len(lbs),
            "healthy": len(unhealthy_lbs) == 0,
        }
        if unhealthy_lbs:
            details["alb"]["unhealthy"] = unhealthy_lbs
            errors.append(f"alb: {len(unhealthy_lbs)} load balancer(s) not active")
    except Exception:
        unavailable.append("alb")
        details["alb"] = {"healthy": None, "note": "not configured or unavailable"}

    # CloudWatch firing alarms — enrich with alarm names
    firing_alarms = 0
    firing_alarm_names: list[str] = []
    try:
        from app.integrations.aws_ops import list_cloudwatch_alarms
        cw = list_cloudwatch_alarms()
        if cw.get("success"):
            firing = [a for a in cw.get("alarms", []) if a.get("state") == "ALARM"]
            firing_alarms = len(firing)
            firing_alarm_names = [a.get("name", "?") for a in firing[:10]]
            if "cloudwatch" in details:
                details["cloudwatch"]["firing_alarms"] = firing_alarms
                if firing_alarm_names:
                    details["cloudwatch"]["firing_alarm_names"] = firing_alarm_names
    except Exception:
        pass

    # Overall status
    configured = [s for s in details if details[s].get("healthy") is not None]
    if not configured:
        return {
            "status":  "unavailable",
            "success": False,
            "details": "No AWS credentials configured — set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or use an IAM role",
        }

    if errors:
        overall = "degraded"
    elif firing_alarms > 0:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status":              overall,
        "success":             len(errors) == 0,
        "details":             details,
        "firing_alarms":       firing_alarms,
        "firing_alarm_names":  firing_alarm_names if firing_alarm_names else None,
        "unavailable":         unavailable,
        "error_summary":       errors if errors else None,
    }
