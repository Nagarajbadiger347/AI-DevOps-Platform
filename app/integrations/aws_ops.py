"""AWS Observability — read-only data collection for AI-driven root cause analysis.

Covers: EC2, CloudWatch Logs, CloudWatch Metrics, CloudWatch Alarms,
        ECS, Lambda, RDS, ELB, CloudTrail.

Credentials are picked up automatically from:
  - Environment variables: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  - IAM instance role (when running on EC2/ECS/Lambda)
  - ~/.aws/credentials file
"""

import datetime
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def _client(service: str):
    return boto3.client(service, region_name=AWS_REGION)

def _now():
    return datetime.datetime.utcnow()

def _hours_ago(h: int):
    return _now() - datetime.timedelta(hours=h)


# ── EC2 ──────────────────────────────────────────────────────

def list_ec2_instances(state: str = "") -> dict:
    """List EC2 instances with health state, optionally filtered by run state."""
    try:
        ec2 = _client("ec2")
        filters = [{"Name": "instance-state-name", "Values": [state]}] if state else []
        resp = ec2.describe_instances(Filters=filters)
        instances = []
        for res in resp["Reservations"]:
            for i in res["Instances"]:
                name = next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"), "")
                instances.append({
                    "id":          i["InstanceId"],
                    "name":        name,
                    "type":        i["InstanceType"],
                    "state":       i["State"]["Name"],
                    "public_ip":   i.get("PublicIpAddress", ""),
                    "private_ip":  i.get("PrivateIpAddress", ""),
                    "launch_time": i["LaunchTime"].isoformat(),
                    "az":          i["Placement"]["AvailabilityZone"],
                })
        return {"success": True, "region": AWS_REGION, "instances": instances, "count": len(instances)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_ec2_status_checks(instance_id: str) -> dict:
    """Get system and instance status checks — identifies hardware/OS-level failures."""
    try:
        ec2 = _client("ec2")
        resp = ec2.describe_instance_status(InstanceIds=[instance_id], IncludeAllInstances=True)
        statuses = resp.get("InstanceStatuses", [])
        if not statuses:
            return {"success": False, "error": f"Instance {instance_id} not found"}
        s = statuses[0]
        return {
            "success":        True,
            "instance_id":    instance_id,
            "instance_state": s["InstanceState"]["Name"],
            "system_status":  s["SystemStatus"]["Status"],     # ok | impaired | insufficient-data
            "instance_status": s["InstanceStatus"]["Status"],  # ok | impaired | insufficient-data
            "system_events":  [
                {"code": e["Code"], "description": e["Description"]}
                for e in s["SystemStatus"].get("Events", [])
            ],
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_ec2_console_output(instance_id: str) -> dict:
    """Retrieve the last console/serial output of an EC2 instance (kernel panics, boot errors)."""
    try:
        ec2 = _client("ec2")
        resp = ec2.get_console_output(InstanceId=instance_id)
        import base64
        raw = resp.get("Output", "")
        decoded = base64.b64decode(raw).decode("utf-8", errors="replace") if raw else "(no output available)"
        return {"success": True, "instance_id": instance_id, "output": decoded}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── CloudWatch Logs ───────────────────────────────────────────

def list_log_groups(prefix: str = "", limit: int = 50) -> dict:
    """List available CloudWatch Log Groups."""
    try:
        logs = _client("logs")
        kwargs = {"limit": limit}
        if prefix:
            kwargs["logGroupNamePrefix"] = prefix
        resp = logs.describe_log_groups(**kwargs)
        groups = [
            {
                "name":              g["logGroupName"],
                "retention_days":    g.get("retentionInDays", "never"),
                "stored_bytes":      g.get("storedBytes", 0),
            }
            for g in resp.get("logGroups", [])
        ]
        return {"success": True, "log_groups": groups, "count": len(groups)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_recent_logs(log_group: str, minutes: int = 30, limit: int = 100) -> dict:
    """Fetch the most recent log events from a log group across all streams."""
    try:
        logs   = _client("logs")
        start  = int((_now() - datetime.timedelta(minutes=minutes)).timestamp() * 1000)
        end    = int(_now().timestamp() * 1000)
        resp   = logs.filter_log_events(
            logGroupName=log_group,
            startTime=start,
            endTime=end,
            limit=limit,
        )
        events = [
            {
                "timestamp": datetime.datetime.utcfromtimestamp(e["timestamp"] / 1000).isoformat(),
                "stream":    e.get("logStreamName", ""),
                "message":   e["message"].strip(),
            }
            for e in resp.get("events", [])
        ]
        return {
            "success":   True,
            "log_group": log_group,
            "minutes":   minutes,
            "events":    events,
            "count":     len(events),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def search_logs(log_group: str, pattern: str, hours: int = 1, limit: int = 100) -> dict:
    """Search log events by a filter pattern (CloudWatch Insights syntax or simple keyword)."""
    try:
        logs  = _client("logs")
        start = int(_hours_ago(hours).timestamp() * 1000)
        end   = int(_now().timestamp() * 1000)
        resp  = logs.filter_log_events(
            logGroupName=log_group,
            filterPattern=pattern,
            startTime=start,
            endTime=end,
            limit=limit,
        )
        events = [
            {
                "timestamp": datetime.datetime.utcfromtimestamp(e["timestamp"] / 1000).isoformat(),
                "stream":    e.get("logStreamName", ""),
                "message":   e["message"].strip(),
            }
            for e in resp.get("events", [])
        ]
        return {
            "success":   True,
            "log_group": log_group,
            "pattern":   pattern,
            "events":    events,
            "count":     len(events),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── CloudWatch Metrics ────────────────────────────────────────

def list_cloudwatch_alarms(state: str = "") -> dict:
    """List CloudWatch alarms, optionally filtered by state (OK, ALARM, INSUFFICIENT_DATA)."""
    try:
        cw = _client("cloudwatch")
        kwargs = {}
        if state:
            kwargs["StateValue"] = state.upper()
        resp = cw.describe_alarms(**kwargs)
        alarms = [
            {
                "name":        a["AlarmName"],
                "state":       a["StateValue"],
                "metric":      a.get("MetricName", ""),
                "namespace":   a.get("Namespace", ""),
                "threshold":   a.get("Threshold"),
                "operator":    a.get("ComparisonOperator", ""),
                "description": a.get("AlarmDescription", ""),
                "updated":     a["StateUpdatedTimestamp"].isoformat(),
            }
            for a in resp.get("MetricAlarms", [])
        ]
        return {"success": True, "alarms": alarms, "count": len(alarms)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_metric(namespace: str, metric_name: str, dimensions: list,
               hours: int = 1, period: int = 300, stat: str = "Average") -> dict:
    """Fetch CloudWatch metric datapoints for any AWS resource."""
    try:
        cw    = _client("cloudwatch")
        start = _hours_ago(hours)
        end   = _now()
        resp  = cw.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[stat],
        )
        points = sorted(
            [{"time": p["Timestamp"].isoformat(), "value": round(p[stat], 4)} for p in resp["Datapoints"]],
            key=lambda x: x["time"],
        )
        return {
            "success":    True,
            "namespace":  namespace,
            "metric":     metric_name,
            "stat":       stat,
            "datapoints": points,
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── ECS ──────────────────────────────────────────────────────

def list_ecs_services(cluster: str = "default") -> dict:
    """List ECS services with running vs desired task counts."""
    try:
        ecs   = _client("ecs")
        arns  = ecs.list_services(cluster=cluster).get("serviceArns", [])
        if not arns:
            return {"success": True, "cluster": cluster, "services": [], "count": 0}
        svcs  = ecs.describe_services(cluster=cluster, services=arns).get("services", [])
        result = [
            {
                "name":         s["serviceName"],
                "status":       s["status"],
                "desired":      s["desiredCount"],
                "running":      s["runningCount"],
                "pending":      s["pendingCount"],
                "task_def":     s["taskDefinition"].split("/")[-1],
                "launch_type":  s.get("launchType", ""),
            }
            for s in svcs
        ]
        return {"success": True, "cluster": cluster, "services": result, "count": len(result)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_stopped_ecs_tasks(cluster: str = "default", limit: int = 20) -> dict:
    """List recently stopped ECS tasks and their stop reasons — key for crash diagnosis."""
    try:
        ecs  = _client("ecs")
        arns = ecs.list_tasks(cluster=cluster, desiredStatus="STOPPED", maxResults=limit).get("taskArns", [])
        if not arns:
            return {"success": True, "cluster": cluster, "stopped_tasks": [], "count": 0}
        tasks = ecs.describe_tasks(cluster=cluster, tasks=arns).get("tasks", [])
        result = [
            {
                "task_id":      t["taskArn"].split("/")[-1],
                "task_def":     t["taskDefinitionArn"].split("/")[-1],
                "stop_code":    t.get("stopCode", ""),
                "stop_reason":  t.get("stoppedReason", ""),
                "stopped_at":   t["stoppedAt"].isoformat() if t.get("stoppedAt") else "",
                "containers":   [
                    {
                        "name":         c["name"],
                        "exit_code":    c.get("exitCode"),
                        "reason":       c.get("reason", ""),
                    }
                    for c in t.get("containers", [])
                ],
            }
            for t in tasks
        ]
        return {"success": True, "cluster": cluster, "stopped_tasks": result, "count": len(result)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── Lambda ───────────────────────────────────────────────────

def list_lambda_functions() -> dict:
    """List Lambda functions with runtime and last modified."""
    try:
        lam  = _client("lambda")
        resp = lam.list_functions()
        fns  = [
            {
                "name":          f["FunctionName"],
                "runtime":       f.get("Runtime", ""),
                "memory_mb":     f["MemorySize"],
                "timeout_sec":   f["Timeout"],
                "last_modified": f["LastModified"],
                "description":   f.get("Description", ""),
            }
            for f in resp.get("Functions", [])
        ]
        return {"success": True, "functions": fns, "count": len(fns)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_lambda_errors(function_name: str, hours: int = 1) -> dict:
    """Get error count, throttle count, and duration for a Lambda function."""
    dims = [{"Name": "FunctionName", "Value": function_name}]
    metrics = {}
    for metric in ["Errors", "Throttles", "Duration", "Invocations"]:
        result = get_metric("AWS/Lambda", metric, dims, hours=hours, period=300,
                            stat="Sum" if metric != "Duration" else "Average")
        metrics[metric.lower()] = result.get("datapoints", [])

    return {
        "success":       True,
        "function":      function_name,
        "hours":         hours,
        "metrics":       metrics,
    }


# ── RDS ──────────────────────────────────────────────────────

def list_rds_instances() -> dict:
    """List RDS database instances with status and engine info."""
    try:
        rds  = _client("rds")
        resp = rds.describe_db_instances()
        dbs  = [
            {
                "id":               d["DBInstanceIdentifier"],
                "engine":           d["Engine"],
                "engine_version":   d["EngineVersion"],
                "class":            d["DBInstanceClass"],
                "status":           d["DBInstanceStatus"],
                "az":               d["AvailabilityZone"],
                "multi_az":         d["MultiAZ"],
                "storage_gb":       d["AllocatedStorage"],
                "endpoint":         d.get("Endpoint", {}).get("Address", ""),
            }
            for d in resp.get("DBInstances", [])
        ]
        return {"success": True, "instances": dbs, "count": len(dbs)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_rds_events(db_instance_id: str, hours: int = 24) -> dict:
    """Get recent RDS events for a DB instance (failovers, reboots, errors)."""
    try:
        rds   = _client("rds")
        start = _hours_ago(hours)
        resp  = rds.describe_events(
            SourceIdentifier=db_instance_id,
            SourceType="db-instance",
            StartTime=start,
            EndTime=_now(),
        )
        events = [
            {
                "time":     e["Date"].isoformat(),
                "message":  e["Message"],
                "categories": e.get("EventCategories", []),
            }
            for e in resp.get("Events", [])
        ]
        return {"success": True, "db_instance": db_instance_id, "events": events, "count": len(events)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── ELB / ALB ─────────────────────────────────────────────────

def get_target_health(target_group_arn: str) -> dict:
    """Get health of all targets in an ALB target group."""
    try:
        elb  = _client("elbv2")
        resp = elb.describe_target_health(TargetGroupArn=target_group_arn)
        targets = [
            {
                "id":          t["Target"]["Id"],
                "port":        t["Target"].get("Port", ""),
                "state":       t["TargetHealth"]["State"],
                "reason":      t["TargetHealth"].get("Reason", ""),
                "description": t["TargetHealth"].get("Description", ""),
            }
            for t in resp.get("TargetHealthDescriptions", [])
        ]
        unhealthy = [t for t in targets if t["state"] != "healthy"]
        return {
            "success":   True,
            "targets":   targets,
            "unhealthy": unhealthy,
            "count":     len(targets),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── CloudTrail ────────────────────────────────────────────────

def get_cloudtrail_events(hours: int = 1, resource_name: str = "") -> dict:
    """Get recent CloudTrail API events — useful for spotting who changed what before an incident."""
    try:
        ct     = _client("cloudtrail")
        start  = _hours_ago(hours)
        kwargs = {"StartTime": start, "EndTime": _now(), "MaxResults": 50}
        if resource_name:
            kwargs["LookupAttributes"] = [{"AttributeKey": "ResourceName", "AttributeValue": resource_name}]
        resp   = ct.lookup_events(**kwargs)
        events = [
            {
                "time":          e["EventTime"].isoformat(),
                "event_name":    e["EventName"],
                "user":          e.get("Username", ""),
                "source_ip":     e.get("CloudTrailEvent", "{}"),
                "resources":     [r.get("ResourceName", "") for r in e.get("Resources", [])],
            }
            for e in resp.get("Events", [])
        ]
        return {"success": True, "events": events, "count": len(events)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── AI Diagnosis Aggregator ───────────────────────────────────

def collect_diagnosis_context(resource_type: str, resource_id: str,
                               log_group: str = "", hours: int = 1) -> dict:
    """Collect all relevant observability data for a resource to feed into AI analysis."""
    ctx: dict = {
        "resource_type": resource_type,
        "resource_id":   resource_id,
        "region":        AWS_REGION,
        "hours":         hours,
    }

    if resource_type == "ec2":
        ctx["status_checks"] = get_ec2_status_checks(resource_id)
        ctx["console_output"] = get_ec2_console_output(resource_id)
        ctx["cpu_metrics"]    = get_metric(
            "AWS/EC2", "CPUUtilization",
            [{"Name": "InstanceId", "Value": resource_id}], hours=hours
        )
        ctx["network_in"]  = get_metric(
            "AWS/EC2", "NetworkIn",
            [{"Name": "InstanceId", "Value": resource_id}], hours=hours
        )
        ctx["status_check_failed"] = get_metric(
            "AWS/EC2", "StatusCheckFailed",
            [{"Name": "InstanceId", "Value": resource_id}], hours=hours
        )

    elif resource_type == "ecs":
        ctx["stopped_tasks"] = get_stopped_ecs_tasks(cluster=resource_id)
        ctx["services"]      = list_ecs_services(cluster=resource_id)

    elif resource_type == "lambda":
        ctx["lambda_errors"] = get_lambda_errors(resource_id, hours=hours)

    elif resource_type == "rds":
        ctx["rds_events"] = get_rds_events(resource_id, hours=hours)
        ctx["cpu_metrics"] = get_metric(
            "AWS/RDS", "CPUUtilization",
            [{"Name": "DBInstanceIdentifier", "Value": resource_id}], hours=hours
        )
        ctx["db_connections"] = get_metric(
            "AWS/RDS", "DatabaseConnections",
            [{"Name": "DBInstanceIdentifier", "Value": resource_id}], hours=hours
        )

    elif resource_type == "alb":
        ctx["target_health"] = get_target_health(resource_id)
        ctx["5xx_errors"] = get_metric(
            "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
            [{"Name": "TargetGroup", "Value": resource_id}], hours=hours, stat="Sum"
        )

    # Always include: firing alarms + recent CloudTrail changes
    ctx["active_alarms"]     = list_cloudwatch_alarms(state="ALARM")
    ctx["cloudtrail_events"] = get_cloudtrail_events(hours=hours, resource_name=resource_id)

    if log_group:
        ctx["recent_logs"]  = get_recent_logs(log_group, minutes=hours * 60)
        ctx["error_logs"]   = search_logs(log_group, "ERROR", hours=hours)

    return ctx


# ── Predictive Scaling Context ────────────────────────────────

def get_scaling_metrics(resource_type: str, resource_id: str, hours: int = 6) -> dict:
    """Collect CPU, memory, request-count and error metrics for scaling prediction.

    Returns time-series data that Claude can analyse to decide if scaling is needed.
    Supports: ecs, ec2, alb, lambda.
    """
    ctx: dict = {
        "resource_type": resource_type,
        "resource_id":   resource_id,
        "region":        AWS_REGION,
        "hours":         hours,
    }

    if resource_type == "ecs":
        ctx["cpu"]    = get_metric(
            "AWS/ECS", "CPUUtilization",
            [{"Name": "ClusterName", "Value": resource_id}], hours=hours, period=300
        )
        ctx["memory"] = get_metric(
            "AWS/ECS", "MemoryUtilization",
            [{"Name": "ClusterName", "Value": resource_id}], hours=hours, period=300
        )
        ctx["services"] = list_ecs_services(resource_id)

    elif resource_type == "ec2":
        ctx["cpu"] = get_metric(
            "AWS/EC2", "CPUUtilization",
            [{"Name": "InstanceId", "Value": resource_id}], hours=hours, period=300
        )
        ctx["network_in"] = get_metric(
            "AWS/EC2", "NetworkIn",
            [{"Name": "InstanceId", "Value": resource_id}], hours=hours, period=300
        )

    elif resource_type == "alb":
        ctx["request_count"] = get_metric(
            "AWS/ApplicationELB", "RequestCount",
            [{"Name": "LoadBalancer", "Value": resource_id}],
            hours=hours, period=300, stat="Sum"
        )
        ctx["target_response_time"] = get_metric(
            "AWS/ApplicationELB", "TargetResponseTime",
            [{"Name": "LoadBalancer", "Value": resource_id}], hours=hours, period=300
        )
        ctx["5xx_errors"] = get_metric(
            "AWS/ApplicationELB", "HTTPCode_Target_5XX_Count",
            [{"Name": "LoadBalancer", "Value": resource_id}],
            hours=hours, period=300, stat="Sum"
        )

    elif resource_type == "lambda":
        ctx["invocations"] = get_metric(
            "AWS/Lambda", "Invocations",
            [{"Name": "FunctionName", "Value": resource_id}],
            hours=hours, period=300, stat="Sum"
        )
        ctx["duration"] = get_metric(
            "AWS/Lambda", "Duration",
            [{"Name": "FunctionName", "Value": resource_id}],
            hours=hours, period=300, stat="Average"
        )
        ctx["throttles"] = get_metric(
            "AWS/Lambda", "Throttles",
            [{"Name": "FunctionName", "Value": resource_id}],
            hours=hours, period=300, stat="Sum"
        )

    ctx["active_alarms"] = list_cloudwatch_alarms(state="ALARM")
    return ctx
