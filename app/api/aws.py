"""
AWS infrastructure routes.
Paths: /check/aws, /aws/*
"""
import os
from typing import Optional, List
from pydantic import BaseModel

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import require_developer, require_viewer, AuthContext, _rbac_guard
from app.agents.infra.aws_checker import check_aws_infrastructure
from app.integrations.aws_ops import (
    list_ec2_instances, get_ec2_status_checks, get_ec2_console_output,
    start_ec2_instance, stop_ec2_instance, reboot_ec2_instance,
    scale_ecs_service, get_ecs_service_detail, force_new_ecs_deployment,
    invoke_lambda, set_alarm_state, get_sqs_queue_depth,
    reboot_rds_instance, get_rds_instance_detail,
    list_log_groups, get_recent_logs, search_logs,
    list_cloudwatch_alarms, get_metric,
    list_ecs_services, get_stopped_ecs_tasks,
    list_lambda_functions, get_lambda_errors,
    list_rds_instances, get_rds_events,
    get_cloudtrail_events,
    collect_diagnosis_context, get_scaling_metrics,
    list_s3_buckets, list_sqs_queues, list_dynamodb_tables,
    list_route53_healthchecks, list_sns_topics,
)
from app.llm.claude import diagnose_aws_resource, predict_scaling, assess_deployment

router = APIRouter(tags=["aws"])


class AWSMetricRequest(BaseModel):
    namespace: str
    metric_name: str
    dimensions: list
    hours: int = 1
    period: int = 300
    stat: str = "Average"


class AWSDiagnoseRequest(BaseModel):
    resource_type: str   # ec2 | ecs | lambda | rds | alb
    resource_id: str
    log_group: str = ""
    hours: int = 1


class PredictScalingRequest(BaseModel):
    resource_type: str   # ecs | ec2 | alb | lambda
    resource_id:   str
    hours:         int = 6


@router.get("/check/aws")
def aws_check():
    result = check_aws_infrastructure()
    return {"aws_check": result}


@router.get("/aws/ec2/instances")
def aws_ec2_list(state: str = "", region: str = ""):
    result = list_ec2_instances(state, region=region) if region else list_ec2_instances(state)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"ec2_instances": result}


@router.get("/aws/ec2/status")
def aws_ec2_status(instance_id: str = ""):
    result = get_ec2_status_checks(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"status_checks": result}


@router.get("/aws/ec2/console")
def aws_ec2_console(instance_id: str):
    result = get_ec2_console_output(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"console_output": result}


@router.post("/aws/ec2/{instance_id}/start")
def aws_ec2_start(instance_id: str, auth: AuthContext = Depends(require_developer)):
    result = start_ec2_instance(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to start instance"))
    return result


@router.post("/aws/ec2/{instance_id}/stop")
def aws_ec2_stop(instance_id: str, auth: AuthContext = Depends(require_developer)):
    result = stop_ec2_instance(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to stop instance"))
    return result


@router.post("/aws/ec2/{instance_id}/reboot")
def aws_ec2_reboot(instance_id: str, auth: AuthContext = Depends(require_developer)):
    result = reboot_ec2_instance(instance_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to reboot instance"))
    return result


@router.get("/aws/logs/groups")
def aws_log_groups(prefix: str = "", limit: int = 50):
    result = list_log_groups(prefix, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"log_groups": result}


@router.get("/aws/logs/recent")
def aws_logs_recent(log_group: str, minutes: int = 30, limit: int = 100):
    result = get_recent_logs(log_group, minutes, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"logs": result}


@router.get("/aws/logs/search")
def aws_logs_search(log_group: str, pattern: str, hours: int = 1, limit: int = 100):
    result = search_logs(log_group, pattern, hours, limit)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"logs": result}


@router.get("/aws/cloudwatch/alarms")
def aws_cw_alarms(state: str = ""):
    valid = {"", "OK", "ALARM", "INSUFFICIENT_DATA"}
    if state.upper() not in valid:
        raise HTTPException(status_code=400, detail=f"state must be one of {valid - {''}}")
    result = list_cloudwatch_alarms(state)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"cloudwatch_alarms": result}


@router.post("/aws/cloudwatch/metrics")
def aws_cw_metrics(req: AWSMetricRequest):
    result = get_metric(req.namespace, req.metric_name, req.dimensions,
                        req.hours, req.period, req.stat)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"metric": result}


@router.get("/aws/cloudwatch/logs")
def aws_cw_logs(log_group: str = "", hours: int = 1, limit: int = 100):
    """Alias: search CloudWatch logs with optional filter."""
    if log_group:
        result = get_recent_logs(log_group, hours * 60, limit)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return {"logs": result}
    groups = list_log_groups("", 20)
    return {"log_groups": groups, "hint": "Provide ?log_group=<name> to get logs"}


@router.get("/aws/ecs/services")
def aws_ecs_services(region: str = ""):
    import boto3 as _b3
    _region = region or os.getenv("AWS_REGION", "us-east-1")
    try:
        _ecs = _b3.client("ecs", region_name=_region)
        clusters_resp = _ecs.list_clusters()
        all_services = []
        for carn in clusters_resp.get("clusterArns", []):
            cname = carn.split("/")[-1]
            svc_resp = _ecs.list_services(cluster=cname, maxResults=100)
            if svc_resp.get("serviceArns"):
                desc = _ecs.describe_services(cluster=cname, services=svc_resp["serviceArns"])
                for s in desc.get("services", []):
                    all_services.append({
                        "service_name":  s["serviceName"],
                        "cluster_name":  cname,
                        "status":        s["status"],
                        "running_count": s["runningCount"],
                        "desired_count": s["desiredCount"],
                    })
        return {"services": all_services, "count": len(all_services), "region": _region}
    except Exception as e:
        return {"services": [], "count": 0, "note": str(e), "region": _region}


@router.get("/aws/ecs/stopped-tasks")
def aws_ecs_stopped(cluster: str = "default", limit: int = 20):
    result = get_stopped_ecs_tasks(cluster, limit)
    if not result.get("success"):
        return {"stopped_tasks": {"success": True, "cluster": cluster, "stopped_tasks": [], "count": 0, "note": result.get("error")}}
    return {"stopped_tasks": result}


@router.get("/aws/lambda/functions")
def aws_lambda_list(region: str = ""):
    import boto3 as _b3
    _region = region or os.getenv("AWS_REGION", "us-east-1")
    try:
        lam = _b3.client("lambda", region_name=_region)
        paginator = lam.get_paginator("list_functions")
        fns = []
        for page in paginator.paginate():
            for f in page.get("Functions", []):
                fns.append({
                    "function_name": f["FunctionName"],
                    "runtime":       f.get("Runtime", "--"),
                    "memory_size":   f.get("MemorySize", 128),
                    "timeout":       f.get("Timeout", 3),
                    "last_modified": f.get("LastModified", ""),
                    "description":   f.get("Description", ""),
                })
        return {"functions": fns, "count": len(fns), "region": _region}
    except Exception as e:
        return {"functions": [], "count": 0, "note": str(e), "region": _region}


@router.get("/aws/lambda/errors")
def aws_lambda_errors(function_name: str = "", hours: int = 1):
    if function_name:
        result = get_lambda_errors(function_name, hours)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return {"lambda_metrics": [result]}
    all_fns = list_lambda_functions()
    if not all_fns.get("success"):
        raise HTTPException(status_code=400, detail=all_fns.get("error"))
    metrics = []
    for fn in all_fns.get("functions", []):
        r = get_lambda_errors(fn["name"], hours)
        if r.get("success"):
            metrics.append(r)
    return {"lambda_metrics": metrics, "count": len(metrics)}


@router.get("/aws/rds/instances")
def aws_rds_list(region: str = ""):
    import boto3 as _b3
    _region = region or os.getenv("AWS_REGION", "us-east-1")
    try:
        rds = _b3.client("rds", region_name=_region)
        paginator = rds.get_paginator("describe_db_instances")
        instances = []
        for page in paginator.paginate():
            for db in page.get("DBInstances", []):
                instances.append({
                    "identifier":     db["DBInstanceIdentifier"],
                    "engine":         db["Engine"],
                    "engine_version": db.get("EngineVersion", ""),
                    "instance_class": db["DBInstanceClass"],
                    "status":         db["DBInstanceStatus"],
                    "multi_az":       db.get("MultiAZ", False),
                    "endpoint":       (db.get("Endpoint") or {}).get("Address", ""),
                    "storage_gb":     db.get("AllocatedStorage", 0),
                })
        return {"instances": instances, "count": len(instances), "region": _region}
    except Exception as e:
        return {"instances": [], "count": 0, "note": str(e), "region": _region}


@router.get("/aws/rds/events")
def aws_rds_events(db_instance_id: str = "", hours: int = 24):
    if db_instance_id:
        result = get_rds_events(db_instance_id, hours)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return {"rds_events": [result]}
    all_dbs = list_rds_instances()
    if not all_dbs.get("success"):
        raise HTTPException(status_code=400, detail=all_dbs.get("error"))
    all_events = []
    for db in all_dbs.get("instances", []):
        r = get_rds_events(db["id"], hours)
        if r.get("success"):
            all_events.append(r)
    return {"rds_events": all_events, "count": len(all_events)}


@router.get("/aws/elb/target-health")
def aws_elb_health(region: str = ""):
    import boto3 as _b3
    _region = region or os.getenv("AWS_REGION", "us-east-1")
    try:
        elb = _b3.client("elbv2", region_name=_region)
        lbs_resp = elb.describe_load_balancers()
        lbs = []
        for lb in lbs_resp.get("LoadBalancers", []):
            tg_resp = elb.describe_target_groups(LoadBalancerArn=lb["LoadBalancerArn"])
            healthy = unhealthy = 0
            for tg in tg_resp.get("TargetGroups", []):
                health = elb.describe_target_health(TargetGroupArn=tg["TargetGroupArn"])
                for t in health.get("TargetHealthDescriptions", []):
                    if t["TargetHealth"]["State"] == "healthy":
                        healthy += 1
                    else:
                        unhealthy += 1
            lbs.append({
                "name":      lb["LoadBalancerName"],
                "type":      lb["Type"],
                "scheme":    lb.get("Scheme", ""),
                "state":     lb["State"]["Code"],
                "dns":       lb.get("DNSName", ""),
                "healthy":   healthy,
                "unhealthy": unhealthy,
            })
        return {"load_balancers": lbs, "count": len(lbs), "region": _region}
    except Exception as e:
        return {"load_balancers": [], "count": 0, "note": str(e), "region": _region}


@router.get("/aws/cloudtrail/events")
def aws_cloudtrail(hours: int = 1, resource_name: str = ""):
    result = get_cloudtrail_events(hours, resource_name)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"cloudtrail_events": result}


@router.get("/aws/s3/buckets")
def aws_s3_buckets():
    result = list_s3_buckets()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"s3_buckets": result}


@router.get("/aws/sqs/queues")
def aws_sqs_queues(region: str = ""):
    import boto3 as _b3
    _region = region or os.getenv("AWS_REGION", "us-east-1")
    try:
        sqs = _b3.client("sqs", region_name=_region)
        resp = sqs.list_queues()
        queues = []
        for url in resp.get("QueueUrls", []):
            attrs = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["All"]).get("Attributes", {})
            queues.append({
                "url":  url,
                "name": url.split("/")[-1],
                "approximate_number_of_messages": attrs.get("ApproximateNumberOfMessages", "0"),
                "fifo": url.endswith(".fifo"),
            })
        return {"queues": queues, "count": len(queues), "region": _region}
    except Exception as e:
        return {"queues": [], "count": 0, "note": str(e), "region": _region}


@router.get("/aws/dynamodb/tables")
def aws_dynamodb_tables(region: str = ""):
    import boto3 as _b3
    _region = region or os.getenv("AWS_REGION", "us-east-1")
    try:
        ddb = _b3.client("dynamodb", region_name=_region)
        paginator = ddb.get_paginator("list_tables")
        tables = []
        for page in paginator.paginate():
            for name in page.get("TableNames", []):
                desc = ddb.describe_table(TableName=name).get("Table", {})
                tables.append({
                    "name":       name,
                    "status":     desc.get("TableStatus", "ACTIVE"),
                    "items":      desc.get("ItemCount", 0),
                    "size_bytes": desc.get("TableSizeBytes", 0),
                })
        return {"tables": tables, "count": len(tables), "region": _region}
    except Exception as e:
        return {"tables": [], "count": 0, "note": str(e), "region": _region}


@router.get("/aws/route53/healthchecks")
def aws_route53_healthchecks():
    result = list_route53_healthchecks()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"route53_healthchecks": result}


@router.get("/aws/route53/health")
def aws_route53_health():
    """Alias for /aws/route53/healthchecks."""
    return aws_route53_healthchecks()


@router.get("/aws/sns/topics")
def aws_sns_topics():
    result = list_sns_topics()
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return {"sns_topics": result}


@router.post("/aws/diagnose")
def aws_diagnose(req: AWSDiagnoseRequest):
    valid_types = {"ec2", "ecs", "lambda", "rds", "alb"}
    if req.resource_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"resource_type must be one of {valid_types}")
    obs = collect_diagnosis_context(req.resource_type, req.resource_id, req.log_group, req.hours)
    diagnosis = diagnose_aws_resource(obs)
    return {"resource": req.resource_id, "type": req.resource_type, "diagnosis": diagnosis, "raw_context": obs}


@router.post("/aws/predict-scaling")
def aws_predict_scaling(req: PredictScalingRequest):
    metrics = get_scaling_metrics(req.resource_type, req.resource_id, req.hours)
    prediction = predict_scaling(metrics)
    return {
        "resource_type":  req.resource_type,
        "resource_id":    req.resource_id,
        "hours_analysed": req.hours,
        "prediction":     prediction,
    }


@router.get("/aws/context")
def aws_context(resource_type: str = "", resource_id: str = "", hours: int = 1):
    """Collect full AWS observability context for a resource."""
    from app.integrations.aws_ops import collect_diagnosis_context
    result = collect_diagnosis_context(resource_type, resource_id, "", hours)
    return {"context": result}


@router.get("/aws/synthesize")
def aws_synthesize(hours: int = 1):
    """Synthesize overall AWS health across all resource types."""
    from app.integrations.universal_collector import collect_all_context, summarize_health
    ctx = collect_all_context(hours=hours)
    summary = summarize_health(ctx)
    return {"summary": summary, "context": ctx}


@router.get("/aws/cost/summary")
def aws_cost_summary(months: int = 1):
    """Alias for /cost/explorer — AWS Cost Explorer summary."""
    import boto3 as _b3, datetime as _dt
    try:
        ce = _b3.client("ce", region_name="us-east-1")
        end = _dt.date.today()
        start = end.replace(day=1)
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": str(start), "End": str(end)},
            Granularity="MONTHLY",
            Metrics=["BlendedCost"],
        )
        total = sum(
            float(r["Total"]["BlendedCost"]["Amount"])
            for r in resp.get("ResultsByTime", [])
        )
        return {"total_cost_usd": round(total, 2), "period": f"{start} to {end}"}
    except Exception as e:
        return {"total_cost_usd": 0, "note": str(e)}


@router.post("/aws/assess-deployment")
def aws_assess_deployment_alias(req: dict):
    """Alias — use POST /deploy/assess instead."""
    raise HTTPException(status_code=301, detail="Use POST /deploy/assess")
