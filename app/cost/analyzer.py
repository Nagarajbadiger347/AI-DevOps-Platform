"""Cost analyzer — real AWS Cost Explorer integration.

Calls the AWS Cost Explorer API to fetch:
  - Current month-to-date spend
  - Last 6 months trend (month-by-month)
  - Service-level breakdown (EC2, RDS, Lambda, S3, etc.)
  - 30-day forecast to end-of-month
  - Top 10 most expensive services

Falls back to estimation-only mode when AWS credentials are missing or
Cost Explorer is not enabled (requires opt-in in AWS console).

Pricing reference: https://aws.amazon.com/pricing/
Cost Explorer API: https://docs.aws.amazon.com/cost-management/latest/APIReference/
"""
from __future__ import annotations

import os
import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _BOTO3_AVAILABLE = True
except ImportError:
    _BOTO3_AVAILABLE = False

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_VCPU_HOUR_USD          = float(os.getenv("COST_VCPU_HOUR_USD", "0.048"))
_GB_HOUR_USD            = float(os.getenv("COST_GB_HOUR_USD", "0.006"))
_HOURS_PER_MONTH        = 730.0
_APPROVAL_THRESHOLD_USD = float(os.getenv("COST_APPROVAL_THRESHOLD_USD", "500.0"))
_DEFAULT_VCPU_PER_REPLICA = 0.25
_DEFAULT_GB_PER_REPLICA   = 0.512
_RESTART_DISRUPTION_MINS  = 2.0

# AWS service display names
_SERVICE_LABELS = {
    "Amazon Elastic Compute Cloud - Compute": "EC2",
    "Amazon Relational Database Service":     "RDS",
    "AWS Lambda":                             "Lambda",
    "Amazon Simple Storage Service":          "S3",
    "Amazon Elastic Container Service":       "ECS",
    "Amazon Elastic Kubernetes Service":      "EKS",
    "Amazon CloudWatch":                      "CloudWatch",
    "Amazon Route 53":                        "Route 53",
    "Amazon Virtual Private Cloud":           "VPC",
    "AWS Key Management Service":             "KMS",
    "Amazon DynamoDB":                        "DynamoDB",
    "Amazon ElastiCache":                     "ElastiCache",
    "Amazon Simple Queue Service":            "SQS",
    "Amazon Simple Notification Service":     "SNS",
    "Amazon API Gateway":                     "API Gateway",
    "Amazon Elastic Load Balancing":          "Load Balancer",
    "AWS Secrets Manager":                    "Secrets Manager",
    "Amazon Elastic Block Store":             "EBS",
    "Amazon CloudFront":                      "CloudFront",
    "AWS WAF":                                "WAF",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ActionCost:
    action_type: str
    description: str
    monthly_delta_usd: float
    one_time_usd: float = 0.0
    notes: str = ""


@dataclass
class ServiceCost:
    service: str           # human-readable label
    service_raw: str       # raw AWS service name
    amount_usd: float
    unit: str = "USD"


@dataclass
class MonthlySpend:
    month: str             # e.g. "2025-11"
    amount_usd: float


@dataclass
class CostReport:
    # Live AWS data
    current_monthly_spend: float          # month-to-date
    forecast_month_end: float             # projected end-of-month total
    last_month_spend: float               # previous full month
    service_breakdown: list[ServiceCost]  # top services this month
    monthly_trend: list[MonthlySpend]     # last 6 months
    cost_explorer_available: bool

    # Action impact estimates
    total_estimated_monthly_delta: float
    per_action_costs: list[ActionCost]

    warnings: list[str]
    approved: bool
    generated_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
# AWS Cost Explorer helpers
# ---------------------------------------------------------------------------

def _get_ce_client(aws_cfg: dict):
    if not _BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not installed")
    region = (aws_cfg or {}).get("region", os.getenv("AWS_REGION", "us-east-1"))
    kwargs: dict = {"region_name": region}
    if (aws_cfg or {}).get("aws_access_key_id"):
        kwargs["aws_access_key_id"]     = aws_cfg["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = aws_cfg.get("aws_secret_access_key", "")
    # Cost Explorer is a global service, must use us-east-1
    kwargs["region_name"] = "us-east-1"
    return boto3.client("ce", **kwargs)


def _fetch_current_month_spend(ce) -> float:
    """Month-to-date spend for the current calendar month."""
    today = datetime.date.today()
    start = today.replace(day=1)
    end   = today
    if start == end:
        end = today + datetime.timedelta(days=1)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )
    total = sum(
        float(r["Total"]["UnblendedCost"]["Amount"])
        for r in resp.get("ResultsByTime", [])
    )
    return round(total, 2)


def _fetch_last_month_spend(ce) -> float:
    """Total spend for the previous full calendar month."""
    today = datetime.date.today()
    end   = today.replace(day=1)
    start = (end - datetime.timedelta(days=1)).replace(day=1)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )
    total = sum(
        float(r["Total"]["UnblendedCost"]["Amount"])
        for r in resp.get("ResultsByTime", [])
    )
    return round(total, 2)


def _fetch_service_breakdown(ce) -> list[ServiceCost]:
    """Top services by spend this month."""
    today = datetime.date.today()
    start = today.replace(day=1)
    end   = today
    if start == end:
        end = today + datetime.timedelta(days=1)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    breakdown: list[ServiceCost] = []
    for result in resp.get("ResultsByTime", []):
        for group in result.get("Groups", []):
            svc_raw = group["Keys"][0]
            amount  = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if amount < 0.01:
                continue
            label = _SERVICE_LABELS.get(svc_raw, svc_raw.replace("Amazon ", "").replace("AWS ", ""))
            breakdown.append(ServiceCost(
                service=label,
                service_raw=svc_raw,
                amount_usd=round(amount, 2),
            ))
    breakdown.sort(key=lambda x: x.amount_usd, reverse=True)
    return breakdown[:10]


def _fetch_monthly_trend(ce) -> list[MonthlySpend]:
    """Last 6 complete months of spend."""
    today = datetime.date.today()
    end   = today.replace(day=1)             # start of this month
    start = (end - datetime.timedelta(days=180)).replace(day=1)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )
    trend: list[MonthlySpend] = []
    for result in resp.get("ResultsByTime", []):
        month  = result["TimePeriod"]["Start"][:7]   # "YYYY-MM"
        amount = float(result["Total"]["UnblendedCost"]["Amount"])
        trend.append(MonthlySpend(month=month, amount_usd=round(amount, 2)))
    trend.sort(key=lambda x: x.month)
    return trend


def _fetch_forecast(ce) -> float:
    """Projected spend to end of current month."""
    today = datetime.date.today()
    start = today + datetime.timedelta(days=1)
    # End of current month
    if today.month == 12:
        end = today.replace(year=today.year + 1, month=1, day=1)
    else:
        end = today.replace(month=today.month + 1, day=1)

    if start >= end:
        return 0.0

    try:
        resp = ce.get_cost_forecast(
            TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
            Metric="UNBLENDED_COST",
            Granularity="MONTHLY",
        )
        return round(float(resp["Total"]["Amount"]), 2)
    except (BotoCoreError, ClientError):
        # Forecast requires at least a few days of spend history
        return 0.0


def _fetch_all_cost_data(aws_cfg: dict) -> dict:
    """Single entry point — fetches everything from Cost Explorer."""
    ce = _get_ce_client(aws_cfg)
    current     = _fetch_current_month_spend(ce)
    last_month  = _fetch_last_month_spend(ce)
    breakdown   = _fetch_service_breakdown(ce)
    trend       = _fetch_monthly_trend(ce)
    forecast    = _fetch_forecast(ce)
    # forecast from CE is remaining spend; add MTD for full month projection
    forecast_total = round(current + forecast, 2) if forecast else 0.0
    return {
        "current_monthly_spend": current,
        "last_month_spend":      last_month,
        "service_breakdown":     breakdown,
        "monthly_trend":         trend,
        "forecast_month_end":    forecast_total,
    }


# ---------------------------------------------------------------------------
# Action cost estimators (unchanged logic, kept for pipeline use)
# ---------------------------------------------------------------------------

def _estimate_k8s_scale(action: dict) -> ActionCost:
    current_replicas = int(action.get("current_replicas", 1))
    target_replicas  = int(action.get("replicas", action.get("target_replicas", current_replicas)))
    delta_replicas   = target_replicas - current_replicas
    vcpu = float(action.get("vcpu_per_replica", _DEFAULT_VCPU_PER_REPLICA))
    gb   = float(action.get("memory_gb_per_replica", _DEFAULT_GB_PER_REPLICA))
    monthly_delta = delta_replicas * (vcpu * _VCPU_HOUR_USD + gb * _GB_HOUR_USD) * _HOURS_PER_MONTH
    direction = "up" if delta_replicas > 0 else "down"
    return ActionCost(
        action_type="k8s_scale",
        description=f"Scale {action.get('deployment','deployment')} {direction}: {current_replicas}→{target_replicas} replicas",
        monthly_delta_usd=monthly_delta,
        notes=f"Delta: ${monthly_delta:+.2f}/month ({abs(delta_replicas)} replica(s) × {vcpu} vCPU + {gb:.2f}GB).",
    )


def _estimate_aws_reboot(action: dict) -> ActionCost:
    revenue_per_hour = float(action.get("revenue_per_hour_usd", 0.0))
    downtime_mins    = float(action.get("estimated_downtime_minutes", 5.0))
    downtime_cost    = revenue_per_hour * (downtime_mins / 60.0)
    return ActionCost(
        action_type="aws_reboot",
        description=f"Reboot {action.get('instance_id', action.get('resource_id', 'AWS resource'))}",
        monthly_delta_usd=0.0,
        one_time_usd=downtime_cost,
        notes=f"~{downtime_mins:.0f} min downtime. Revenue impact: ${downtime_cost:.2f}." if revenue_per_hour else f"~{downtime_mins:.0f} min downtime.",
    )


def _estimate_k8s_restart(action: dict) -> ActionCost:
    return ActionCost(
        action_type="k8s_restart",
        description=f"Rolling restart of {action.get('deployment','deployment')}",
        monthly_delta_usd=0.0,
        notes=f"~{_RESTART_DISRUPTION_MINS:.0f} min partial disruption. No persistent cost change.",
    )


def _estimate_generic(action: dict) -> ActionCost:
    return ActionCost(
        action_type=action.get("type", "unknown"),
        description=str(action.get("description", action.get("type", "action"))),
        monthly_delta_usd=0.0,
        notes="No cost model for this action type.",
    )


_ESTIMATORS = {
    "k8s_scale":   _estimate_k8s_scale,
    "aws_reboot":  _estimate_aws_reboot,
    "aws_restart": _estimate_aws_reboot,
    "k8s_restart": _estimate_k8s_restart,
    "create_pr":   _estimate_generic,
    "slack_notify": _estimate_generic,
    "investigate": _estimate_generic,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_action_costs(
    actions: list[dict],
    aws_cfg: dict = None,
) -> CostReport:
    """Estimate cost impact of proposed actions + fetch real AWS spend data.

    Calls AWS Cost Explorer for live data. Falls back gracefully when
    credentials are missing or Cost Explorer is not enabled.
    """
    aws_cfg = aws_cfg or {}
    per_action_costs: list[ActionCost] = []
    warnings: list[str] = []

    # Fetch real AWS cost data
    live: dict = {}
    ce_available = False
    try:
        live = _fetch_all_cost_data(aws_cfg)
        ce_available = True
        logger.info("cost_explorer_fetch_ok", extra={
            "mtd": live["current_monthly_spend"],
            "forecast": live["forecast_month_end"],
        })
    except Exception as exc:
        warnings.append(f"AWS Cost Explorer unavailable — using estimates only. ({exc})")
        logger.warning("cost_explorer_unavailable", extra={"error": str(exc)})

    # Estimate each action
    for action in actions:
        action_type = action.get("action_type", action.get("type", "unknown")).lower()
        estimator   = _ESTIMATORS.get(action_type, _estimate_generic)
        try:
            cost = estimator(action)
            per_action_costs.append(cost)
        except Exception as exc:
            warnings.append(f"Could not estimate cost for '{action_type}': {exc}")
            per_action_costs.append(ActionCost(
                action_type=action_type,
                description=str(action.get("description", action_type)),
                monthly_delta_usd=0.0,
                notes=f"Estimation failed: {exc}",
            ))

    total_monthly_delta = sum(c.monthly_delta_usd for c in per_action_costs)

    for cost in per_action_costs:
        if cost.monthly_delta_usd > 100:
            warnings.append(f"High delta: {cost.action_type} adds ${cost.monthly_delta_usd:.2f}/month.")
        if cost.one_time_usd > 50:
            warnings.append(f"Downtime cost: {cost.action_type} = ${cost.one_time_usd:.2f}.")

    if total_monthly_delta > _APPROVAL_THRESHOLD_USD:
        warnings.append(
            f"Monthly delta ${total_monthly_delta:.2f} exceeds ${_APPROVAL_THRESHOLD_USD:.0f} threshold — approval required."
        )

    approved = total_monthly_delta <= _APPROVAL_THRESHOLD_USD

    return CostReport(
        current_monthly_spend=live.get("current_monthly_spend", 0.0),
        forecast_month_end=live.get("forecast_month_end", 0.0),
        last_month_spend=live.get("last_month_spend", 0.0),
        service_breakdown=live.get("service_breakdown", []),
        monthly_trend=live.get("monthly_trend", []),
        cost_explorer_available=ce_available,
        total_estimated_monthly_delta=total_monthly_delta,
        per_action_costs=per_action_costs,
        warnings=warnings,
        approved=approved,
    )


def fetch_cost_dashboard(aws_cfg: dict = None) -> dict:
    """Fetch full cost dashboard data — called directly by the UI endpoint.

    Returns a plain dict ready for JSON serialisation.
    """
    aws_cfg = aws_cfg or {}
    try:
        live = _fetch_all_cost_data(aws_cfg)
        return {
            "available": True,
            "current_monthly_spend": live["current_monthly_spend"],
            "last_month_spend":      live["last_month_spend"],
            "forecast_month_end":    live["forecast_month_end"],
            "service_breakdown": [
                {"service": s.service, "amount_usd": s.amount_usd}
                for s in live["service_breakdown"]
            ],
            "monthly_trend": [
                {"month": m.month, "amount_usd": m.amount_usd}
                for m in live["monthly_trend"]
            ],
        }
    except Exception as exc:
        logger.warning("cost_dashboard_unavailable", extra={"error": str(exc)})
        return {
            "available": False,
            "error": str(exc),
            "current_monthly_spend": 0.0,
            "last_month_spend":      0.0,
            "forecast_month_end":    0.0,
            "service_breakdown":     [],
            "monthly_trend":         [],
        }


def format_cost_report(report: CostReport) -> str:
    lines = ["=" * 60, "  COST IMPACT REPORT", "=" * 60]
    if report.cost_explorer_available:
        lines.append(f"  Month-to-date spend : ${report.current_monthly_spend:,.2f}")
        lines.append(f"  Last month total    : ${report.last_month_spend:,.2f}")
        if report.forecast_month_end:
            lines.append(f"  Forecast (month-end): ${report.forecast_month_end:,.2f}")
        if report.service_breakdown:
            lines.append("")
            lines.append("  Top services this month:")
            for s in report.service_breakdown[:5]:
                lines.append(f"    {s.service:<30} ${s.amount_usd:,.2f}")
    else:
        lines.append("  AWS spend           : unavailable (Cost Explorer not reached)")

    lines.append(f"  Estimated delta     : ${report.total_estimated_monthly_delta:+,.2f}/month")
    lines.append(f"  Auto-approved       : {'YES' if report.approved else 'NO — human approval required'}")
    lines.append(f"  Generated           : {report.generated_at}")

    if report.per_action_costs:
        lines.append("")
        lines.append("  Per-action breakdown:")
        lines.append("  " + "-" * 56)
        for i, ac in enumerate(report.per_action_costs, 1):
            lines.append(f"  [{i}] {ac.action_type.upper()}: {ac.description}")
            lines.append(f"      Monthly delta : ${ac.monthly_delta_usd:+,.2f}")
            if ac.one_time_usd:
                lines.append(f"      One-time cost : ${ac.one_time_usd:,.2f}")
            if ac.notes:
                lines.append(f"      Notes         : {ac.notes}")

    if report.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"  ⚠  {w}")

    lines.append("=" * 60)
    return "\n".join(lines)
