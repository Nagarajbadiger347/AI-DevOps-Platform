"""
Cost analysis and estimation routes.
Paths: /cost/*
"""
import os
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.routes.deps import require_viewer, AuthContext

router = APIRouter(tags=["cost"])


class CostAnalysisRequest(BaseModel):
    actions: List[Dict[str, Any]]
    aws_cfg: Optional[Dict[str, Any]] = None


class TerraformCostRequest(BaseModel):
    plan_json: Dict[str, Any]
    region: Optional[str] = None


class PriceEstimateRequest(BaseModel):
    description: str
    region: Optional[str] = None


@router.get("/cost/explorer")
def cost_explorer(days: int = 30, auth: AuthContext = Depends(require_viewer)):
    """Real AWS spend breakdown by service from Cost Explorer."""
    from app.cost.pricing import get_cost_explorer_summary
    return get_cost_explorer_summary(days=days)


@router.post("/cost/analyze")
async def analyze_costs_endpoint(req: CostAnalysisRequest, auth: AuthContext = Depends(require_viewer)):
    try:
        from app.cost.analyzer import analyze_action_costs, format_cost_report
        import dataclasses as _dc
        report = analyze_action_costs(req.actions, req.aws_cfg)
        report_dict = _dc.asdict(report)
        return {"report": report_dict, "formatted": format_cost_report(report)}
    except Exception as e:
        return {"report": None, "formatted": f"Cost analysis unavailable: {e}", "error": str(e)}


@router.post("/cost/terraform")
async def terraform_cost_endpoint(req: TerraformCostRequest, auth: AuthContext = Depends(require_viewer)):
    """Estimate monthly cost from a Terraform plan JSON output."""
    try:
        from app.cost.pricing import estimate_terraform_plan_cost
        result = estimate_terraform_plan_cost(req.plan_json)
        return result
    except Exception as e:
        return {"error": str(e), "total_monthly_usd": 0.0, "resources": []}


@router.post("/cost/estimate")
async def price_estimate_endpoint(req: PriceEstimateRequest, auth: AuthContext = Depends(require_viewer)):
    """Estimate AWS cost from a natural language description."""
    try:
        from app.cost.pricing import estimate_from_description
        region = req.region or os.getenv("AWS_REGION", "us-east-1")
        result = estimate_from_description(req.description, region)
        return result
    except Exception as e:
        return {"error": str(e), "total_monthly_usd": 0.0, "resources": []}


@router.get("/cost/dashboard")
async def cost_dashboard_endpoint(auth: AuthContext = Depends(require_viewer)):
    """Full cost dashboard: MTD spend, forecast, service breakdown, 6-month trend."""
    try:
        from app.cost.analyzer import fetch_cost_dashboard
        return fetch_cost_dashboard()
    except Exception as e:
        return {"available": False, "error": str(e), "current_monthly_spend": 0.0,
                "last_month_spend": 0.0, "forecast_month_end": 0.0,
                "service_breakdown": [], "monthly_trend": []}


@router.get("/cost/resources")
async def cost_resources_endpoint(auth: AuthContext = Depends(require_viewer)):
    """List all AWS resources with estimated monthly cost (EC2, RDS, Lambda, ECS)."""
    resources = []

    def _boto(svc):
        import boto3
        return boto3.client(
            svc,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        )

    try:
        ec2 = _boto("ec2")
        paginator = ec2.get_paginator("describe_instances")
        EC2_RATES = {
            "t2.micro":0.0116,"t2.small":0.023,"t2.medium":0.0464,"t2.large":0.0928,
            "t3.micro":0.0104,"t3.small":0.0208,"t3.medium":0.0416,"t3.large":0.0832,
            "t3.xlarge":0.1664,"t3.2xlarge":0.3328,
            "m5.large":0.096,"m5.xlarge":0.192,"m5.2xlarge":0.384,"m5.4xlarge":0.768,
            "m6i.large":0.096,"m6i.xlarge":0.192,"m6i.2xlarge":0.384,
            "c5.large":0.085,"c5.xlarge":0.17,"c5.2xlarge":0.34,"c5.4xlarge":0.68,
            "r5.large":0.126,"r5.xlarge":0.252,"r5.2xlarge":0.504,"r5.4xlarge":1.008,
        }
        for page in paginator.paginate():
            for res in page["Reservations"]:
                for inst in res["Instances"]:
                    if inst.get("State", {}).get("Name") not in ("running", "stopped"):
                        continue
                    itype = inst.get("InstanceType", "")
                    name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), "")
                    hourly = EC2_RATES.get(itype, 0.05)
                    monthly = hourly * 730
                    resources.append({
                        "service": "EC2",
                        "resource_id": inst["InstanceId"],
                        "name": name,
                        "region": os.getenv("AWS_REGION", "us-east-1"),
                        "instance_type": itype,
                        "details": inst.get("State", {}).get("Name", ""),
                        "monthly_usd": round(monthly, 2),
                    })
    except Exception:
        pass

    try:
        rds = _boto("rds")
        RDS_RATES = {
            "db.t3.micro":0.017,"db.t3.small":0.034,"db.t3.medium":0.068,"db.t3.large":0.136,
            "db.m5.large":0.171,"db.m5.xlarge":0.342,"db.m5.2xlarge":0.684,
            "db.r5.large":0.24,"db.r5.xlarge":0.48,"db.r5.2xlarge":0.96,"db.r5.4xlarge":1.92,
        }
        for db in rds.describe_db_instances().get("DBInstances", []):
            itype = db.get("DBInstanceClass", "")
            hourly = RDS_RATES.get(itype, 0.1)
            multi_az = db.get("MultiAZ", False)
            monthly = hourly * 730 * (2 if multi_az else 1)
            resources.append({
                "service": "RDS",
                "resource_id": db["DBInstanceIdentifier"],
                "name": db["DBInstanceIdentifier"],
                "region": os.getenv("AWS_REGION", "us-east-1"),
                "instance_type": itype,
                "details": f"{db.get('Engine','')} {'Multi-AZ' if multi_az else 'Single-AZ'}",
                "monthly_usd": round(monthly, 2),
            })
    except Exception:
        pass

    try:
        lam = _boto("lambda")
        paginator = lam.get_paginator("list_functions")
        for page in paginator.paginate():
            for fn in page["Functions"]:
                mem_mb = fn.get("MemorySize", 128)
                gb_seconds = (mem_mb / 1024) * 0.5 * 1_000_000
                compute_cost = gb_seconds * 0.0000166667
                request_cost = 1_000_000 * 0.0000002
                monthly = round(compute_cost + request_cost, 4)
                resources.append({
                    "service": "Lambda",
                    "resource_id": fn["FunctionArn"].split(":")[-1],
                    "name": fn["FunctionName"],
                    "region": os.getenv("AWS_REGION", "us-east-1"),
                    "instance_type": f"{mem_mb}MB",
                    "details": fn.get("Runtime", ""),
                    "monthly_usd": monthly,
                })
    except Exception:
        pass

    try:
        ecs = _boto("ecs")
        clusters = ecs.list_clusters().get("clusterArns", [])
        for cluster_arn in clusters[:5]:
            svc_arns = ecs.list_services(cluster=cluster_arn).get("serviceArns", [])
            if not svc_arns:
                continue
            for svc in ecs.describe_services(cluster=cluster_arn, services=svc_arns[:10]).get("services", []):
                tasks = svc.get("runningCount", 0)
                monthly = tasks * (0.5 * 0.04048 + 1 * 0.004445) * 730
                resources.append({
                    "service": "ECS",
                    "resource_id": svc["serviceArn"].split("/")[-1],
                    "name": svc["serviceName"],
                    "region": os.getenv("AWS_REGION", "us-east-1"),
                    "instance_type": f"{tasks} tasks",
                    "details": svc.get("launchType", "FARGATE"),
                    "monthly_usd": round(monthly, 2),
                })
    except Exception:
        pass

    if not resources:
        return {"resources": [], "error": "No resources found — check AWS credentials and region configuration"}

    resources.sort(key=lambda x: x["monthly_usd"], reverse=True)
    total = sum(r["monthly_usd"] for r in resources)
    return {"resources": resources, "total_monthly_usd": round(total, 2), "count": len(resources)}
