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
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from botocore.config import Config as _BotoCfg

_BOTO_CFG = _BotoCfg(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2})


def _client(service: str):
    region = os.getenv("AWS_REGION", "us-west-2")
    return boto3.client(service, region_name=region, config=_BOTO_CFG)

def _now():
    return datetime.datetime.now(datetime.timezone.utc)

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
        return {"success": True, "region": os.getenv("AWS_REGION", "us-west-2"), "instances": instances, "count": len(instances)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_ec2_status_checks(instance_id: str = "") -> dict:
    """Get system and instance status checks for all instances (or a specific one)."""
    try:
        ec2 = _client("ec2")
        kwargs = {"IncludeAllInstances": True}
        if instance_id:
            kwargs["InstanceIds"] = [instance_id]
        resp = ec2.describe_instance_status(**kwargs)
        statuses = resp.get("InstanceStatuses", [])
        if not statuses:
            return {"success": True, "statuses": [], "count": 0, "message": "No instances found"}
        results = []
        for s in statuses:
            results.append({
                "instance_id":     s["InstanceId"],
                "instance_state":  s["InstanceState"]["Name"],
                "system_status":   s["SystemStatus"]["Status"],
                "instance_status": s["InstanceStatus"]["Status"],
                "healthy":         s["SystemStatus"]["Status"] == "ok" and s["InstanceStatus"]["Status"] == "ok",
                "events":          [
                    {"code": e["Code"], "description": e["Description"]}
                    for e in s["SystemStatus"].get("Events", [])
                ],
            })
        return {"success": True, "statuses": results, "count": len(results)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def start_ec2_instance(instance_id: str) -> dict:
    """Start a stopped EC2 instance."""
    try:
        ec2 = _client("ec2")
        resp = ec2.start_instances(InstanceIds=[instance_id])
        change = resp["StartingInstances"][0]
        return {
            "success":       True,
            "instance_id":   instance_id,
            "previous_state": change["PreviousState"]["Name"],
            "current_state":  change["CurrentState"]["Name"],
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "instance_id": instance_id, "error": str(e)}


def stop_ec2_instance(instance_id: str) -> dict:
    """Stop a running EC2 instance."""
    try:
        ec2 = _client("ec2")
        resp = ec2.stop_instances(InstanceIds=[instance_id])
        change = resp["StoppingInstances"][0]
        return {
            "success":       True,
            "instance_id":   instance_id,
            "previous_state": change["PreviousState"]["Name"],
            "current_state":  change["CurrentState"]["Name"],
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "instance_id": instance_id, "error": str(e)}


def reboot_ec2_instance(instance_id: str) -> dict:
    """Reboot a running EC2 instance."""
    try:
        ec2 = _client("ec2")
        ec2.reboot_instances(InstanceIds=[instance_id])
        return {"success": True, "instance_id": instance_id, "action": "rebooted"}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "instance_id": instance_id, "error": str(e)}


def get_ec2_console_output(instance_id: str) -> dict:
    """Retrieve the last console/serial output of an EC2 instance (kernel panics, boot errors)."""
    try:
        ec2 = _client("ec2")
        resp = ec2.get_console_output(InstanceId=instance_id)
        import base64
        raw = resp.get("Output", "")
        if not raw:
            decoded = "(no output available)"
        elif isinstance(raw, bytes):
            decoded = raw.decode("utf-8", errors="replace")
        else:
            # boto3 returns the output already decoded as a string in newer SDK versions
            try:
                decoded = base64.b64decode(raw.encode("ascii")).decode("utf-8", errors="replace")
            except (ValueError, UnicodeEncodeError):
                decoded = raw  # already a plain string
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
        "region":        os.getenv("AWS_REGION", "us-west-2"),
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


# ── S3 ────────────────────────────────────────────────────────

def list_s3_buckets() -> dict:
    """List S3 buckets with region and creation date."""
    try:
        s3   = _client("s3")
        resp = s3.list_buckets()
        buckets = [
            {
                "name":         b["Name"],
                "created":      b["CreationDate"].isoformat(),
            }
            for b in resp.get("Buckets", [])
        ]
        return {"success": True, "buckets": buckets, "count": len(buckets)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── SQS ───────────────────────────────────────────────────────

def list_sqs_queues() -> dict:
    """List SQS queues with approximate message counts and DLQ depths."""
    try:
        sqs  = _client("sqs")
        resp = sqs.list_queues()
        urls = resp.get("QueueUrls", [])
        queues = []
        attrs  = ["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible",
                  "ApproximateNumberOfMessagesDelayed", "QueueArn"]
        for url in urls[:30]:
            try:
                qa = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=attrs)["Attributes"]
                queues.append({
                    "name":    url.split("/")[-1],
                    "url":     url,
                    "visible": int(qa.get("ApproximateNumberOfMessages", 0)),
                    "in_flight": int(qa.get("ApproximateNumberOfMessagesNotVisible", 0)),
                    "delayed": int(qa.get("ApproximateNumberOfMessagesDelayed", 0)),
                })
            except (BotoCoreError, ClientError):
                queues.append({"name": url.split("/")[-1], "url": url})
        return {"success": True, "queues": queues, "count": len(queues)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── DynamoDB ──────────────────────────────────────────────────

def list_dynamodb_tables() -> dict:
    """List DynamoDB tables with status and item count."""
    try:
        ddb   = _client("dynamodb")
        names = ddb.list_tables().get("TableNames", [])
        tables = []
        for name in names[:20]:
            try:
                info = ddb.describe_table(TableName=name)["Table"]
                tables.append({
                    "name":       name,
                    "status":     info.get("TableStatus", ""),
                    "item_count": info.get("ItemCount", 0),
                    "size_bytes": info.get("TableSizeBytes", 0),
                    "billing":    info.get("BillingModeSummary", {}).get("BillingMode", "PROVISIONED"),
                })
            except (BotoCoreError, ClientError):
                tables.append({"name": name})
        return {"success": True, "tables": tables, "count": len(tables)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── Route53 Health Checks ─────────────────────────────────────

def list_route53_healthchecks() -> dict:
    """List Route53 health checks with current status."""
    try:
        r53    = boto3.client("route53")  # route53 is global, no region
        checks = r53.list_health_checks().get("HealthChecks", [])
        cw     = _client("cloudwatch")
        results = []
        for hc in checks[:20]:
            hc_id = hc["Id"]
            config = hc.get("HealthCheckConfig", {})
            # Get health status
            try:
                status = r53.get_health_check_status(HealthCheckId=hc_id)
                checks_status = status.get("HealthCheckObservations", [])
                healthy = all(
                    o.get("StatusReport", {}).get("Status", "").startswith("Success")
                    for o in checks_status
                )
            except Exception:
                healthy = None
            results.append({
                "id":       hc_id,
                "type":     config.get("Type", ""),
                "endpoint": config.get("FullyQualifiedDomainName", config.get("IPAddress", "")),
                "port":     config.get("Port", ""),
                "path":     config.get("ResourcePath", ""),
                "healthy":  healthy,
            })
        return {"success": True, "health_checks": results, "count": len(results)}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── SNS ───────────────────────────────────────────────────────

def list_sns_topics() -> dict:
    """List SNS topics."""
    try:
        sns    = _client("sns")
        topics = sns.list_topics().get("Topics", [])
        return {
            "success": True,
            "topics": [{"arn": t["TopicArn"], "name": t["TopicArn"].split(":")[-1]} for t in topics],
            "count": len(topics),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── Predictive Scaling Context ────────────────────────────────

def get_scaling_metrics(resource_type: str, resource_id: str, hours: int = 6) -> dict:
    """Collect CPU, memory, request-count and error metrics for scaling prediction.

    Returns time-series data that Claude can analyse to decide if scaling is needed.
    Supports: ecs, ec2, alb, lambda.
    """
    ctx: dict = {
        "resource_type": resource_type,
        "resource_id":   resource_id,
        "region":        os.getenv("AWS_REGION", "us-west-2"),
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


# ── EC2 Extended ──────────────────────────────────────────────

def get_ec2_instance_info(instance_id: str) -> dict:
    """Get full details for a single EC2 instance."""
    try:
        ec2 = _client("ec2")
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        for res in resp["Reservations"]:
            for i in res["Instances"]:
                name = next((t["Value"] for t in i.get("Tags", []) if t["Key"] == "Name"), "")
                return {
                    "success":     True,
                    "instance_id": i["InstanceId"],
                    "name":        name,
                    "type":        i["InstanceType"],
                    "state":       i["State"]["Name"],
                    "public_ip":   i.get("PublicIpAddress", ""),
                    "private_ip":  i.get("PrivateIpAddress", ""),
                    "az":          i["Placement"]["AvailabilityZone"],
                    "launch_time": i["LaunchTime"].isoformat(),
                    "vpc_id":      i.get("VpcId", ""),
                    "subnet_id":   i.get("SubnetId", ""),
                    "security_groups": [sg["GroupName"] for sg in i.get("SecurityGroups", [])],
                }
        return {"success": False, "error": f"Instance {instance_id} not found"}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── ECS Extended ──────────────────────────────────────────────

def scale_ecs_service(cluster: str, service: str, desired_count: int) -> dict:
    """Scale an ECS service to the desired task count."""
    try:
        ecs = _client("ecs")
        ecs.update_service(cluster=cluster, service=service, desiredCount=desired_count)
        return {
            "success":       True,
            "cluster":       cluster,
            "service":       service,
            "desired_count": desired_count,
            "message":       f"ECS service {service} scaled to {desired_count} task(s)",
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_ecs_service_detail(cluster: str, service: str) -> dict:
    """Get detailed status for a single ECS service."""
    try:
        ecs = _client("ecs")
        resp = ecs.describe_services(cluster=cluster, services=[service])
        svcs = resp.get("services", [])
        if not svcs:
            return {"success": False, "error": f"Service {service} not found in cluster {cluster}"}
        s = svcs[0]
        return {
            "success":        True,
            "service":        s["serviceName"],
            "cluster":        cluster,
            "status":         s["status"],
            "desired":        s["desiredCount"],
            "running":        s["runningCount"],
            "pending":        s["pendingCount"],
            "task_def":       s["taskDefinition"].split("/")[-1],
            "launch_type":    s.get("launchType", ""),
            "deployments":    len(s.get("deployments", [])),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def force_new_ecs_deployment(cluster: str, service: str) -> dict:
    """Force a new ECS deployment (restarts all tasks)."""
    try:
        ecs = _client("ecs")
        ecs.update_service(cluster=cluster, service=service, forceNewDeployment=True)
        return {
            "success": True,
            "cluster": cluster,
            "service": service,
            "message": f"New deployment forced for ECS service {service}",
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── Lambda Extended ───────────────────────────────────────────

def invoke_lambda(function_name: str, payload: dict = None) -> dict:
    """Invoke a Lambda function synchronously and return its response."""
    import json
    try:
        lam = _client("lambda")
        resp = lam.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload or {}).encode(),
        )
        status_code = resp.get("StatusCode", 0)
        body_raw = resp["Payload"].read().decode("utf-8")
        try:
            body = json.loads(body_raw)
        except Exception:
            body = body_raw
        return {
            "success":     status_code == 200,
            "function":    function_name,
            "status_code": status_code,
            "response":    body,
            "error_type":  resp.get("FunctionError", None),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── CloudWatch Alarms Extended ────────────────────────────────

def set_alarm_state(alarm_name: str, state: str, reason: str = "Set by AI DevOps") -> dict:
    """Manually set a CloudWatch alarm state (OK/ALARM/INSUFFICIENT_DATA)."""
    valid = {"OK", "ALARM", "INSUFFICIENT_DATA"}
    if state.upper() not in valid:
        return {"success": False, "error": f"state must be one of {valid}"}
    try:
        cw = _client("cloudwatch")
        cw.set_alarm_state(
            AlarmName=alarm_name,
            StateValue=state.upper(),
            StateReason=reason,
        )
        return {"success": True, "alarm": alarm_name, "new_state": state.upper()}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── SQS Extended ──────────────────────────────────────────────

def get_sqs_queue_depth(queue_url: str) -> dict:
    """Get message counts for an SQS queue."""
    try:
        sqs = _client("sqs")
        resp = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible",
                            "ApproximateNumberOfMessagesDelayed"],
        )
        attrs = resp.get("Attributes", {})
        return {
            "success":        True,
            "queue_url":      queue_url,
            "visible":        int(attrs.get("ApproximateNumberOfMessages", 0)),
            "in_flight":      int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)),
            "delayed":        int(attrs.get("ApproximateNumberOfMessagesDelayed", 0)),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


# ── RDS Extended ──────────────────────────────────────────────

def reboot_rds_instance(db_instance_id: str) -> dict:
    """Reboot an RDS DB instance."""
    try:
        rds = _client("rds")
        rds.reboot_db_instance(DBInstanceIdentifier=db_instance_id)
        return {"success": True, "db_instance_id": db_instance_id,
                "message": f"RDS instance {db_instance_id} reboot initiated"}
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}


def get_rds_instance_detail(db_instance_id: str) -> dict:
    """Get detailed status for a single RDS instance."""
    try:
        rds = _client("rds")
        resp = rds.describe_db_instances(DBInstanceIdentifier=db_instance_id)
        instances = resp.get("DBInstances", [])
        if not instances:
            return {"success": False, "error": f"RDS instance {db_instance_id} not found"}
        i = instances[0]
        return {
            "success":          True,
            "db_instance_id":   i["DBInstanceIdentifier"],
            "engine":           i["Engine"],
            "engine_version":   i["EngineVersion"],
            "status":           i["DBInstanceStatus"],
            "instance_class":   i["DBInstanceClass"],
            "multi_az":         i.get("MultiAZ", False),
            "storage_gb":       i.get("AllocatedStorage", 0),
            "endpoint":         i.get("Endpoint", {}).get("Address", ""),
        }
    except (BotoCoreError, ClientError) as e:
        return {"success": False, "error": str(e)}

