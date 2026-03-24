from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.orchestrator.main import app
from app.security import rbac as _rbac

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Nagaraj" in response.text
    assert "AI DevOps" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_correlate_no_events():
    response = client.post("/correlate", json=[])
    assert response.status_code == 400
    assert response.json()["detail"] == "No events provided"


def test_correlate_sample_event():
    # Single error event → type_counts["error"]=1 > 0.5*1 → "Multiple errors detected"
    payload = [{"id": "1", "type": "error", "source": "app", "payload": {"msg": "boom"}}]
    response = client.post("/correlate", json=payload)
    assert response.status_code == 200
    assert "correlation" in response.json()
    assert response.json()["correlation"]["root_cause"] == "Multiple errors detected"


def test_llm_analyze():
    # Without ANTHROPIC_API_KEY configured, the fallback rca is "Sample root cause"
    body = {"incident_id": "i1", "details": {"info": "test"}}
    response = client.post("/llm/analyze", json=body)
    assert response.status_code == 200
    assert response.json()["analysis"]["rca"] == "Sample root cause"


def test_check_aws_linux():
    # Mock boto3 calls so the test doesn't require real AWS credentials
    mock_ec2 = MagicMock()
    mock_ec2.describe_instances.return_value = {"Reservations": []}
    mock_s3 = MagicMock()
    mock_s3.list_buckets.return_value = {"Buckets": []}
    mock_cw = MagicMock()
    mock_cw.describe_alarms.return_value = {"MetricAlarms": []}

    def mock_boto_client(service, **kwargs):
        return {"ec2": mock_ec2, "s3": mock_s3, "cloudwatch": mock_cw}[service]

    with patch("boto3.client", side_effect=mock_boto_client):
        r1 = client.get("/check/aws")
        r2 = client.get("/check/linux")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["aws_check"]["status"] == "healthy"
    assert r2.json()["linux_check"]["status"] == "healthy"


def test_incident_endpoints():
    r2 = client.post("/incident/jira")
    r3 = client.post("/incident/opsgenie")
    assert r2.status_code == 200
    assert r3.status_code == 200
    # War-room: mock Slack so we get a real room_url back
    mock_slack = MagicMock()
    mock_slack.chat_postMessage.return_value = {"ts": "12345"}
    with patch("app.integrations.slack.WebClient", return_value=mock_slack), \
         patch("app.integrations.slack.SLACK_BOT_TOKEN", "mock-token"):
        r1 = client.post("/incident/war-room")
    assert r1.status_code == 200
    assert r1.json()["war_room"]["room_url"].startswith("https://")


def test_memory_incident():
    event = {"id": "i1", "type": "alert", "source": "sys", "payload": {"a": 1}}
    with patch("app.memory.vector_db.collection") as mock_col:
        mock_col.add.return_value = None
        response = client.post("/memory/incidents", json=event)
    assert response.status_code == 200
    assert response.json()["stored"]["stored"] is True


def test_security_check():
    # Assign a role first, then check access
    _rbac.assign_role("alice", "developer")
    body = {"user": "alice", "action": "deploy"}
    response = client.post("/security/check", json=body)
    assert response.status_code == 200
    assert response.json()["access"]["allowed"] is True


# ── Kubernetes tests ─────────────────────────────────────────

def _mock_k8s_node(name="node-1", ready=True):
    node = MagicMock()
    node.metadata.name = name
    node.metadata.labels = {"node-role.kubernetes.io/control-plane": ""}
    node.status.node_info.kubelet_version = "v1.28.0"
    cond = MagicMock()
    cond.type = "Ready"
    cond.status = "True" if ready else "False"
    node.status.conditions = [cond]
    return node

def _mock_k8s_pod(name="pod-1", phase="Running", ready=True, restarts=0):
    pod = MagicMock()
    pod.metadata.name = name
    pod.metadata.namespace = "default"
    pod.status.phase = phase
    cs = MagicMock()
    cs.ready = ready
    cs.restart_count = restarts
    pod.status.container_statuses = [cs]
    return pod

def _mock_k8s_deployment(name="dep-1", desired=2, ready=2):
    dep = MagicMock()
    dep.metadata.name = name
    dep.metadata.namespace = "default"
    dep.spec.replicas = desired
    dep.status.replicas = desired
    dep.status.ready_replicas = ready
    return dep


def test_k8s_cluster_check():
    mock_node = _mock_k8s_node()
    mock_pod  = _mock_k8s_pod()
    mock_dep  = _mock_k8s_deployment()

    with patch("app.plugins.k8s_checker._load_config", return_value=True), \
         patch("app.plugins.k8s_checker.client.CoreV1Api") as mock_core, \
         patch("app.plugins.k8s_checker.client.AppsV1Api") as mock_apps:
        mock_core.return_value.list_node.return_value.items = [mock_node]
        mock_core.return_value.list_pod_for_all_namespaces.return_value.items = [mock_pod]
        mock_apps.return_value.list_deployment_for_all_namespaces.return_value.items = [mock_dep]
        response = client.get("/check/k8s")

    assert response.status_code == 200
    data = response.json()["k8s_check"]
    assert data["status"] == "healthy"
    assert data["details"]["nodes"]["total"] == 1
    assert data["details"]["pods"]["running"] == 1


def test_k8s_nodes():
    mock_node = _mock_k8s_node("worker-1", ready=True)
    with patch("app.plugins.k8s_checker._load_config", return_value=True), \
         patch("app.plugins.k8s_checker.client.CoreV1Api") as mock_core:
        mock_core.return_value.list_node.return_value.items = [mock_node]
        response = client.get("/check/k8s/nodes")

    assert response.status_code == 200
    nodes = response.json()["k8s_nodes"]["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["ready"] is True


def test_k8s_pods():
    mock_pod = _mock_k8s_pod("api-pod", "Running")
    with patch("app.plugins.k8s_checker._load_config", return_value=True), \
         patch("app.plugins.k8s_checker.client.CoreV1Api") as mock_core:
        mock_core.return_value.list_namespaced_pod.return_value.items = [mock_pod]
        response = client.get("/check/k8s/pods?namespace=default")

    assert response.status_code == 200
    pods = response.json()["k8s_pods"]["pods"]
    assert pods[0]["phase"] == "Running"


def test_k8s_deployments():
    mock_dep = _mock_k8s_deployment("my-app", desired=3, ready=3)
    with patch("app.plugins.k8s_checker._load_config", return_value=True), \
         patch("app.plugins.k8s_checker.client.AppsV1Api") as mock_apps:
        mock_apps.return_value.list_namespaced_deployment.return_value.items = [mock_dep]
        response = client.get("/check/k8s/deployments?namespace=default")

    assert response.status_code == 200
    deps = response.json()["k8s_deployments"]["deployments"]
    assert deps[0]["available"] is True


def test_k8s_restart():
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client.AppsV1Api") as mock_apps:
        mock_apps.return_value.patch_namespaced_deployment.return_value = MagicMock()
        response = client.post("/k8s/restart", json={"namespace": "default", "deployment": "my-app"})

    assert response.status_code == 200
    assert response.json()["result"]["success"] is True


def test_k8s_scale():
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client.AppsV1Api") as mock_apps:
        mock_apps.return_value.patch_namespaced_deployment_scale.return_value = MagicMock()
        response = client.post("/k8s/scale", json={"namespace": "default", "deployment": "my-app", "replicas": 3})

    assert response.status_code == 200
    assert response.json()["result"]["success"] is True


def test_k8s_logs():
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client.CoreV1Api") as mock_core:
        mock_core.return_value.read_namespaced_pod_log.return_value = "line1\nline2\n"
        response = client.get("/k8s/logs?namespace=default&pod=my-pod")

    assert response.status_code == 200
    assert "line1" in response.json()["result"]["logs"]


# ── AWS Observability tests ───────────────────────────────────
import datetime as dt

def _mock_instance(instance_id="i-123", state="running", name="web-server"):
    return {
        "InstanceId": instance_id, "InstanceType": "t3.micro",
        "State": {"Name": state}, "PublicIpAddress": "1.2.3.4",
        "PrivateIpAddress": "10.0.0.1", "LaunchTime": dt.datetime(2024, 1, 1),
        "Placement": {"AvailabilityZone": "us-east-1a"},
        "Tags": [{"Key": "Name", "Value": name}],
    }


def test_aws_ec2_list():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.describe_instances.return_value = {
            "Reservations": [{"Instances": [_mock_instance()]}]
        }
        response = client.get("/aws/ec2/instances")
    assert response.status_code == 200
    data = response.json()["ec2_instances"]
    assert data["count"] == 1
    assert data["instances"][0]["id"] == "i-123"


def test_aws_ec2_status():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.describe_instance_status.return_value = {
            "InstanceStatuses": [{
                "InstanceState": {"Name": "running"},
                "SystemStatus": {"Status": "ok", "Events": []},
                "InstanceStatus": {"Status": "ok"},
            }]
        }
        response = client.get("/aws/ec2/status?instance_id=i-123")
    assert response.status_code == 200
    data = response.json()["status_checks"]
    assert data["system_status"] == "ok"
    assert data["instance_status"] == "ok"


def test_aws_ec2_console():
    import base64
    encoded = base64.b64encode(b"kernel panic: not syncing").decode()
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.get_console_output.return_value = {"Output": encoded}
        response = client.get("/aws/ec2/console?instance_id=i-123")
    assert response.status_code == 200
    assert "kernel panic" in response.json()["console_output"]["output"]


def test_aws_log_groups():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.describe_log_groups.return_value = {
            "logGroups": [{"logGroupName": "/app/api", "storedBytes": 1024}]
        }
        response = client.get("/aws/logs/groups")
    assert response.status_code == 200
    assert response.json()["log_groups"]["count"] == 1


def test_aws_logs_recent():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.filter_log_events.return_value = {
            "events": [{"timestamp": 1700000000000, "logStreamName": "stream-1", "message": "ERROR: timeout"}]
        }
        response = client.get("/aws/logs/recent?log_group=/app/api&minutes=30")
    assert response.status_code == 200
    assert response.json()["logs"]["count"] == 1
    assert "ERROR" in response.json()["logs"]["events"][0]["message"]


def test_aws_logs_search():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.filter_log_events.return_value = {
            "events": [{"timestamp": 1700000000000, "logStreamName": "stream-1", "message": "ERROR: DB connection failed"}]
        }
        response = client.get("/aws/logs/search?log_group=/app/api&pattern=ERROR")
    assert response.status_code == 200
    assert "DB connection" in response.json()["logs"]["events"][0]["message"]


def test_aws_cloudwatch_alarms():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.describe_alarms.return_value = {
            "MetricAlarms": [{
                "AlarmName": "high-cpu", "StateValue": "ALARM",
                "MetricName": "CPUUtilization", "Namespace": "AWS/EC2",
                "AlarmDescription": "CPU > 80%", "Threshold": 80.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "StateUpdatedTimestamp": dt.datetime(2024, 1, 1),
            }]
        }
        response = client.get("/aws/cloudwatch/alarms?state=ALARM")
    assert response.status_code == 200
    alarms = response.json()["cloudwatch_alarms"]["alarms"]
    assert alarms[0]["state"] == "ALARM"
    assert alarms[0]["threshold"] == 80.0


def test_aws_cloudwatch_metrics():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.get_metric_statistics.return_value = {
            "Datapoints": [{"Timestamp": dt.datetime(2024, 1, 1), "Average": 72.3}]
        }
        response = client.post("/aws/cloudwatch/metrics", json={
            "namespace": "AWS/EC2", "metric_name": "CPUUtilization",
            "dimensions": [{"Name": "InstanceId", "Value": "i-123"}],
        })
    assert response.status_code == 200
    assert response.json()["metric"]["datapoints"][0]["value"] == 72.3


def test_aws_ecs_services():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.list_services.return_value = {"serviceArns": ["arn:aws:ecs:us-east-1:123:service/api"]}
        mock_boto.return_value.describe_services.return_value = {"services": [{
            "serviceName": "api", "status": "ACTIVE",
            "desiredCount": 2, "runningCount": 2, "pendingCount": 0,
            "taskDefinition": "arn:aws:ecs:us-east-1:123:task-definition/api:5",
            "launchType": "FARGATE",
        }]}
        response = client.get("/aws/ecs/services?cluster=prod")
    assert response.status_code == 200
    svc = response.json()["ecs_services"]["services"][0]
    assert svc["running"] == 2


def test_aws_ecs_stopped_tasks():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.list_tasks.return_value = {"taskArns": ["arn:task/abc"]}
        mock_boto.return_value.describe_tasks.return_value = {"tasks": [{
            "taskArn": "arn:task/abc",
            "taskDefinitionArn": "arn:task-def/api:5",
            "stopCode": "EssentialContainerExited",
            "stoppedReason": "Essential container in task exited",
            "stoppedAt": dt.datetime(2024, 1, 1),
            "containers": [{"name": "app", "exitCode": 1, "reason": "OOMKilled"}],
        }]}
        response = client.get("/aws/ecs/stopped-tasks?cluster=prod")
    assert response.status_code == 200
    task = response.json()["stopped_tasks"]["stopped_tasks"][0]
    assert task["stop_code"] == "EssentialContainerExited"
    assert task["containers"][0]["reason"] == "OOMKilled"


def test_aws_lambda_list():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.list_functions.return_value = {"Functions": [{
            "FunctionName": "my-func", "Runtime": "python3.11",
            "MemorySize": 256, "Timeout": 30,
            "LastModified": "2024-01-01T00:00:00", "Description": "",
        }]}
        response = client.get("/aws/lambda/functions")
    assert response.status_code == 200
    assert response.json()["lambda_functions"]["count"] == 1


def test_aws_lambda_errors():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        # Datapoints must include both Sum (for Errors/Throttles/Invocations) and Average (for Duration)
        mock_boto.return_value.get_metric_statistics.return_value = {
            "Datapoints": [{"Timestamp": dt.datetime(2024, 1, 1), "Sum": 5.0, "Average": 120.0}]
        }
        response = client.get("/aws/lambda/errors?function_name=my-func&hours=1")
    assert response.status_code == 200
    assert response.json()["lambda_metrics"]["function"] == "my-func"


def test_aws_rds_instances():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.describe_db_instances.return_value = {"DBInstances": [{
            "DBInstanceIdentifier": "prod-db", "Engine": "postgres",
            "EngineVersion": "15.3", "DBInstanceClass": "db.t3.medium",
            "DBInstanceStatus": "available", "AvailabilityZone": "us-east-1a",
            "MultiAZ": True, "AllocatedStorage": 100,
            "Endpoint": {"Address": "prod-db.abc.us-east-1.rds.amazonaws.com"},
        }]}
        response = client.get("/aws/rds/instances")
    assert response.status_code == 200
    db = response.json()["rds_instances"]["instances"][0]
    assert db["status"] == "available"
    assert db["multi_az"] is True


def test_aws_rds_events():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.describe_events.return_value = {"Events": [{
            "Date": dt.datetime(2024, 1, 1),
            "Message": "DB instance restarted",
            "EventCategories": ["availability"],
        }]}
        response = client.get("/aws/rds/events?db_instance_id=prod-db")
    assert response.status_code == 200
    events = response.json()["rds_events"]["events"]
    assert "restarted" in events[0]["message"]


def test_aws_cloudtrail():
    with patch("app.integrations.aws_ops.boto3.client") as mock_boto:
        mock_boto.return_value.lookup_events.return_value = {"Events": [{
            "EventTime": dt.datetime(2024, 1, 1),
            "EventName": "StopInstances",
            "Username": "admin",
            "CloudTrailEvent": "{}",
            "Resources": [{"ResourceName": "i-123"}],
        }]}
        response = client.get("/aws/cloudtrail/events?hours=1&resource_name=i-123")
    assert response.status_code == 200
    events = response.json()["cloudtrail_events"]["events"]
    assert events[0]["event_name"] == "StopInstances"
    assert events[0]["user"] == "admin"


def test_aws_diagnose():
    mock_obs = {
        "resource_type": "ec2", "resource_id": "i-123", "region": "us-east-1", "hours": 1,
        "active_alarms": {"success": True, "alarms": [], "count": 0},
        "cloudtrail_events": {"success": True, "events": [], "count": 0},
    }
    mock_diagnosis = {
        "summary": "EC2 instance i-123 has high CPU utilization",
        "root_cause": "CPU spike due to runaway process",
        "confidence": 0.85,
        "severity": "high",
        "findings": ["CPUUtilization spiked to 98%"],
        "recommended_actions": ["SSH in and check top", "Review CloudWatch logs"],
    }
    with patch("app.orchestrator.main.collect_diagnosis_context", return_value=mock_obs), \
         patch("app.orchestrator.main.diagnose_aws_resource", return_value=mock_diagnosis):
        response = client.post("/aws/diagnose", json={"resource_type": "ec2", "resource_id": "i-123"})
    assert response.status_code == 200
    d = response.json()["diagnosis"]
    assert d["severity"] == "high"
    assert d["confidence"] == 0.85


# ── Incident pipeline (/incident/run) ─────────────────────────

_MOCK_PIPELINE_REPORT = {
    "incident_id":       "INC-001",
    "started_at":        "2024-01-01T00:00:00",
    "completed_at":      "2024-01-01T00:00:05",
    "description":       "High error rate on API",
    "reported_severity": "high",
    "summary":           "Service is returning 500 errors due to OOM",
    "root_cause":        "Memory leak in user-service pod",
    "confidence":        0.9,
    "ai_severity":       "critical",
    "findings":          ["Pod OOM killed 3 times in 2 hours"],
    "observability": {
        "aws_collected":    False,
        "k8s_collected":    True,
        "github_collected": True,
    },
    "auto_remediate":    False,
    "actions_taken": [
        {
            "type":   "github_pr",
            "reason": "Fix memory leak",
            "result": {"success": True, "pr_number": 42, "url": "https://github.com/test/pr/42"},
        }
    ],
    "raw_context": {"aws": {}, "k8s": {}, "github": {}},
}


def test_incident_run_basic():
    """Pipeline endpoint returns full incident report when pipeline is mocked."""
    with patch("app.orchestrator.main.run_incident_pipeline", return_value=_MOCK_PIPELINE_REPORT):
        response = client.post("/incident/run", json={
            "incident_id": "INC-001",
            "description": "High error rate on API",
            "severity":    "high",
        })
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == "INC-001"
    assert data["root_cause"] == "Memory leak in user-service pod"
    assert data["confidence"] == 0.9
    assert data["ai_severity"] == "critical"
    assert len(data["actions_taken"]) == 1
    assert data["actions_taken"][0]["type"] == "github_pr"


def test_incident_run_with_aws_k8s():
    """Pipeline endpoint forwards aws/k8s config to the pipeline correctly."""
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return _MOCK_PIPELINE_REPORT

    with patch("app.orchestrator.main.run_incident_pipeline", side_effect=fake_pipeline):
        response = client.post("/incident/run", json={
            "incident_id":    "INC-002",
            "description":    "ECS tasks crashing",
            "severity":       "critical",
            "aws":            {"resource_type": "ecs", "resource_id": "my-cluster"},
            "k8s":            {"namespace": "production"},
            "auto_remediate": True,
            "hours":          4,
        })
    assert response.status_code == 200
    assert captured["incident_id"] == "INC-002"
    assert captured["severity"] == "critical"
    assert captured["aws_cfg"]["resource_type"] == "ecs"
    assert captured["k8s_cfg"]["namespace"] == "production"
    assert captured["auto_remediate"] is True
    assert captured["hours"] == 4


def test_incident_run_missing_fields():
    """Pipeline endpoint rejects requests missing required fields."""
    response = client.post("/incident/run", json={"severity": "high"})
    assert response.status_code == 422  # Pydantic validation error


def test_incident_run_auto_remediate_false():
    """auto_remediate=False is forwarded and shown in report."""
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        report = dict(_MOCK_PIPELINE_REPORT)
        report["auto_remediate"] = False
        report["actions_taken"] = [{"type": "k8s_restart", "skipped": True,
                                     "reason": "auto_remediate=false — manual approval required"}]
        return report

    with patch("app.orchestrator.main.run_incident_pipeline", side_effect=fake_pipeline):
        response = client.post("/incident/run", json={
            "incident_id": "INC-003",
            "description": "Pod crashlooping",
            "auto_remediate": False,
        })
    assert response.status_code == 200
    assert captured["auto_remediate"] is False
    data = response.json()
    assert data["actions_taken"][0]["skipped"] is True


# ── Pipeline internals ─────────────────────────────────────────

def test_pipeline_aws_unavailable_uses_sentinel():
    """When AWS is unavailable, _collect_aws returns _data_available=False sentinel."""
    from app.agents.incident_pipeline import _collect_aws
    with patch("app.agents.incident_pipeline.aws_ops.collect_diagnosis_context",
               side_effect=Exception("No credentials")):
        result = _collect_aws({"resource_type": "ecs", "resource_id": "my-cluster"}, hours=2)
    assert result["_data_available"] is False
    assert "No credentials" in result["_reason"]


def test_pipeline_k8s_unavailable_uses_sentinel():
    """When K8s returns an error status, _collect_k8s returns _data_available=False sentinel."""
    from app.agents.incident_pipeline import _collect_k8s
    with patch("app.agents.incident_pipeline.k8s_checker.check_k8s_cluster",
               return_value={"status": "error", "details": "kubeconfig not found"}):
        result = _collect_k8s({"namespace": "default"})
    assert result["_data_available"] is False
    assert "kubeconfig not found" in result["_reason"]


def test_pipeline_k8s_exception_uses_sentinel():
    """When K8s raises an exception, _collect_k8s returns _data_available=False sentinel."""
    from app.agents.incident_pipeline import _collect_k8s
    with patch("app.agents.incident_pipeline.k8s_checker.check_k8s_cluster",
               side_effect=Exception("connection refused")):
        result = _collect_k8s({"namespace": "default"})
    assert result["_data_available"] is False
    assert "connection refused" in result["_reason"]


def test_pipeline_no_aws_config_sentinel():
    """When no AWS config is passed, _collect_aws returns sentinel without calling boto3."""
    from app.agents.incident_pipeline import _collect_aws
    result = _collect_aws({}, hours=2)
    assert result["_data_available"] is False


def test_pipeline_collect_all_parallel():
    """_collect_all returns aws/k8s/github keys even when all sources fail."""
    from app.agents.incident_pipeline import _collect_all
    with patch("app.agents.incident_pipeline._collect_aws",
               return_value={"_data_available": False, "_reason": "no creds"}), \
         patch("app.agents.incident_pipeline._collect_k8s",
               return_value={"_data_available": False, "_reason": "no kubeconfig"}), \
         patch("app.agents.incident_pipeline._collect_github",
               return_value={"_data_available": False, "_reason": "no token"}):
        obs = _collect_all({}, {}, hours=1)
    assert "aws" in obs
    assert "k8s" in obs
    assert "github" in obs
    assert obs["aws"]["_data_available"] is False
    assert obs["k8s"]["_data_available"] is False


# ── Claude JSON parsing ────────────────────────────────────────

def test_claude_extract_json_fenced():
    """_extract_json correctly strips markdown fences."""
    from app.llm.claude import _extract_json
    raw = '```json\n{"key": "value"}\n```'
    assert _extract_json(raw) == '{"key": "value"}'


def test_claude_extract_json_bare():
    """_extract_json handles raw JSON with no fencing."""
    from app.llm.claude import _extract_json
    raw = '{"key": "value"}'
    assert _extract_json(raw) == '{"key": "value"}'


def test_claude_extract_json_prose_around():
    """_extract_json extracts JSON even with surrounding prose."""
    from app.llm.claude import _extract_json
    raw = 'Here is the analysis:\n{"root_cause": "OOM"}\nLet me know if you need more.'
    result = _extract_json(raw)
    assert '"root_cause"' in result


def test_claude_extract_json_fenced_no_lang():
    """_extract_json handles fences without 'json' language tag."""
    from app.llm.claude import _extract_json
    raw = '```\n{"severity": "high"}\n```'
    result = _extract_json(raw)
    assert '"severity"' in result


# ── GitHub duplicate branch ────────────────────────────────────

def test_github_create_incident_pr_duplicate_branch():
    """create_incident_pr reuses existing branch instead of crashing."""
    from app.integrations import github as gh
    from github import GithubException

    mock_repo = MagicMock()
    mock_branch = MagicMock()
    mock_branch.commit.sha = "abc123"
    mock_repo.get_branch.return_value = mock_branch  # branch exists
    mock_repo.get_contents.side_effect = GithubException(404, "not found")
    mock_repo.create_file.return_value = None
    mock_pr = MagicMock()
    mock_pr.number = 99
    mock_pr.html_url = "https://github.com/test/pr/99"
    mock_repo.create_pull.return_value = mock_pr

    with patch.object(gh, "_repo", return_value=mock_repo):
        result = gh.create_incident_pr("INC-DUP", "Fix", "body")

    # Should succeed — branch already existed but was reused
    assert result["success"] is True
    assert result["pr_number"] == 99
    # create_git_ref should NOT have been called (branch already existed)
    mock_repo.create_git_ref.assert_not_called()
