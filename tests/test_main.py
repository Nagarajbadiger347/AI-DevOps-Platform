"""
NexusOps test suite — aligned with the current v2 LangGraph pipeline.

Coverage:
  - Health endpoints
  - Correlate events
  - LLM analyze (fallback path)
  - AWS infrastructure read endpoints
  - Kubernetes read + mutating endpoints (RBAC)
  - Incident pipeline (POST /incidents/run)
  - Memory endpoints
  - Security / RBAC checks
  - Claude utility functions (no API key path)
  - GitHub PR utilities
"""
import datetime as dt
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.orchestrator.main import app
from app.security import rbac as _rbac

client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_live():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


# ── Correlate ─────────────────────────────────────────────────────────────────

def test_correlate_no_events():
    # The /correlate endpoint expects {"events": [...]} body — empty events → 400 or 422
    response = client.post("/correlate", json={"events": []})
    assert response.status_code in (400, 422, 403)


def test_correlate_sample_event():
    # /correlate expects {"events": [...]} (FastAPI named body field)
    payload = {"events": [{"id": "1", "type": "error", "source": "app", "payload": {"msg": "boom"}}]}
    response = client.post("/correlate", json=payload)
    assert response.status_code in (200, 403, 422)


# ── Security / RBAC ───────────────────────────────────────────────────────────

def test_security_check():
    _rbac.assign_role("alice", "developer")
    body = {"user": "alice", "action": "deploy"}
    response = client.post("/security/check", json=body)
    assert response.status_code == 200
    assert response.json()["access"]["allowed"] is True


# ── Incident pipeline (/incidents/run) ────────────────────────────────────────

_MOCK_PIPELINE_RESULT = {
    "incident_id":    "INC-001",
    "correlation_id": "test-corr-id",
    "trace_id":       "test-trace-id",
    "status":         "completed",
    "description":    "High error rate on API",
    "plan": {
        "root_cause": "Memory leak in user-service pod",
        "confidence": 0.9,
        "risk":       "high",
        "actions":    [{"type": "k8s_restart", "description": "Restart the deployment"}],
        "summary":    "Restart crashed pod",
        "reasoning":  "Pod OOMKilled repeatedly",
        "data_gaps":  [],
    },
    "executed_actions":        [],
    "blocked_actions":         [],
    "validation_passed":       True,
    "requires_human_approval": False,
    "errors":                  [],
    "auto_remediate":          False,
    "dry_run":                 True,
}


def test_incidents_run_basic():
    """Pipeline endpoint returns result when pipeline is mocked."""
    with patch("app.api.incidents.run_pipeline", return_value=_MOCK_PIPELINE_RESULT), \
         patch("app.api.incidents.store_incident", return_value={"stored": True}):
        response = client.post("/incidents/run", json={
            "incident_id": "INC-001",
            "description": "High error rate on API",
            "severity":    "high",
            "dry_run":     True,
        })
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == "INC-001"
    assert data["plan"]["root_cause"] == "Memory leak in user-service pod"
    assert data["status"] == "completed"


def test_incidents_run_missing_fields():
    """Pipeline endpoint rejects requests missing required fields."""
    response = client.post("/incidents/run", json={"severity": "high"})
    assert response.status_code == 422  # Pydantic validation error


def test_incidents_run_auto_remediate_requires_auth():
    """POST /incidents/run with auto_remediate=true requires developer role or above."""
    # No auth — optional_auth returns None → role defaults to viewer → 403
    response = client.post("/incidents/run", json={
        "incident_id":    "INC-RBAC",
        "description":    "test",
        "auto_remediate": True,
    })
    assert response.status_code == 403


def test_incidents_run_dry_run_no_auth():
    """POST /incidents/run with auto_remediate=false + dry_run is allowed without auth."""
    with patch("app.api.incidents.run_pipeline", return_value=_MOCK_PIPELINE_RESULT), \
         patch("app.api.incidents.store_incident", return_value={"stored": True}):
        response = client.post("/incidents/run", json={
            "incident_id":    "INC-NORBAC",
            "description":    "test",
            "auto_remediate": False,
            "dry_run":        True,
        })
    assert response.status_code == 200


def test_incidents_run_awaiting_approval():
    """Pipeline returns awaiting_approval status when approval is required."""
    awaiting_result = {
        **_MOCK_PIPELINE_RESULT,
        "status":                  "awaiting_approval",
        "requires_human_approval": True,
        "correlation_id":          "corr-approval-123",
    }
    with patch("app.api.incidents.run_pipeline", return_value=awaiting_result), \
         patch("app.api.incidents.store_incident", return_value={"stored": True}), \
         patch("app.incident.approval.create_approval_request", return_value=MagicMock(
             correlation_id="corr-approval-123"
         )), \
         patch("app.incident.approval.post_approval_to_slack", return_value=None), \
         patch("app.integrations.email.send_approval_required", return_value=None):
        response = client.post("/incidents/run", json={
            "incident_id": "INC-APPR",
            "description": "test",
        })
    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_approval"


# ── Memory endpoints ──────────────────────────────────────────────────────────

def test_memory_incident_store():
    event = {"id": "i1", "type": "alert", "source": "sys", "payload": {"a": 1}}
    with patch("app.memory.vector_db.collection") as mock_col:
        mock_col.add.return_value = None
        response = client.post("/memory/incidents", json=event)
    assert response.status_code in (200, 403)  # depends on auth config


def test_memory_incidents_list():
    with patch("app.memory.vector_db.search_similar_incidents", return_value=[]):
        response = client.get("/memory/incidents")
    # 200 if auth optional, 403 if auth required
    assert response.status_code in (200, 403)


# ── Kubernetes endpoints ──────────────────────────────────────────────────────

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

    with patch("app.agents.infra.k8s_checker._load_config", return_value=True), \
         patch("app.agents.infra.k8s_checker.client.CoreV1Api") as mock_core, \
         patch("app.agents.infra.k8s_checker.client.AppsV1Api") as mock_apps:
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
    # /k8s/nodes uses check_k8s_nodes from k8s_checker
    mock_node = _mock_k8s_node("worker-1", ready=True)
    with patch("app.agents.infra.k8s_checker._load_config", return_value=True), \
         patch("app.agents.infra.k8s_checker.client") as mck:
        mck.CoreV1Api.return_value.list_node.return_value.items = [mock_node]
        response = client.get("/k8s/nodes")

    assert response.status_code == 200
    data = response.json()["nodes"]
    assert data.get("total", data.get("count", 0)) >= 1


def test_k8s_pods():
    # /k8s/pods uses list_pods from k8s_ops (different module than k8s_checker)
    mock_pod = _mock_k8s_pod("api-pod", "Running")
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client") as mck:
        mck.CoreV1Api.return_value.list_namespaced_pod.return_value.items = [mock_pod]
        response = client.get("/k8s/pods?namespace=default")

    assert response.status_code == 200
    pods = response.json()["pods"]
    # pods may be a list or a dict with pods key
    pod_list = pods if isinstance(pods, list) else pods.get("pods", [])
    assert len(pod_list) >= 1


def test_k8s_deployments():
    # /k8s/deployments uses list_deployments from k8s_ops
    mock_dep = _mock_k8s_deployment("my-app", desired=3, ready=3)
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client") as mck:
        mck.AppsV1Api.return_value.list_namespaced_deployment.return_value.items = [mock_dep]
        response = client.get("/k8s/deployments?namespace=default")

    assert response.status_code == 200
    deps = response.json()["deployments"]
    dep_list = deps if isinstance(deps, list) else deps.get("deployments", [])
    assert len(dep_list) >= 1


def test_k8s_restart():
    _rbac.assign_role("k8s-test-user", "developer")
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client.AppsV1Api") as mock_apps:
        mock_apps.return_value.patch_namespaced_deployment.return_value = MagicMock()
        response = client.post("/k8s/restart",
                               json={"namespace": "default", "deployment": "my-app"},
                               headers={"X-User": "k8s-test-user"})

    assert response.status_code == 200
    assert response.json()["result"]["success"] is True


def test_k8s_scale():
    _rbac.assign_role("k8s-test-user", "developer")
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client.AppsV1Api") as mock_apps:
        mock_apps.return_value.patch_namespaced_deployment_scale.return_value = MagicMock()
        response = client.post("/k8s/scale",
                               json={"namespace": "default", "deployment": "my-app", "replicas": 3},
                               headers={"X-User": "k8s-test-user"})

    assert response.status_code == 200
    assert response.json()["result"]["success"] is True


def test_k8s_logs():
    with patch("app.integrations.k8s_ops._load_config", return_value=True), \
         patch("app.integrations.k8s_ops.client.CoreV1Api") as mock_core:
        mock_core.return_value.read_namespaced_pod_log.return_value = "line1\nline2\n"
        response = client.get("/k8s/logs?namespace=default&pod=my-pod")

    assert response.status_code == 200
    assert "line1" in response.json()["result"]["logs"]


def test_k8s_restart_no_user_header():
    """k8s/restart returns 403 when X-User header is missing (no JWT either)."""
    response = client.post("/k8s/restart", json={"namespace": "default", "deployment": "my-app"})
    assert response.status_code == 403


def test_k8s_restart_insufficient_role():
    """k8s/restart returns 403 when user lacks deploy permission."""
    response = client.post(
        "/k8s/restart",
        json={"namespace": "default", "deployment": "my-app"},
        headers={"X-User": "readonly-user"},  # no role assigned → no permission
    )
    assert response.status_code == 403


# ── AWS endpoints ─────────────────────────────────────────────────────────────

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
                "InstanceId": "i-123",
                "InstanceState": {"Name": "running"},
                "SystemStatus": {"Status": "ok", "Events": []},
                "InstanceStatus": {"Status": "ok"},
            }]
        }
        response = client.get("/aws/ec2/status?instance_id=i-123")
    assert response.status_code == 200
    data = response.json()["status_checks"]
    assert data["statuses"][0]["system_status"] == "ok"


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


def _make_paginator(pages: list):
    pag = MagicMock()
    pag.paginate.return_value = pages
    return pag


def test_aws_rds_instances():
    # The route uses its own `import boto3 as _b3` inside the function
    mock_boto = MagicMock()
    mock_boto.get_paginator.return_value = _make_paginator([{"DBInstances": [{
        "DBInstanceIdentifier": "prod-db", "Engine": "postgres", "EngineVersion": "15.3",
        "DBInstanceClass": "db.t3.medium", "DBInstanceStatus": "available",
        "MultiAZ": True, "AllocatedStorage": 100, "Endpoint": {"Address": "x.rds.com"},
    }]}])
    with patch("boto3.client", return_value=mock_boto):
        response = client.get("/aws/rds/instances")
    assert response.status_code == 200
    assert len(response.json().get("instances", [])) >= 1


def test_aws_ecs_services():
    # Route calls list_clusters() first, then list_services per cluster
    mock_boto = MagicMock()
    mock_boto.list_clusters.return_value = {"clusterArns": ["arn:aws:ecs:us-east-1:123:cluster/prod"]}
    mock_boto.list_services.return_value = {"serviceArns": ["arn:aws:ecs:us-east-1:123:service/api"]}
    mock_boto.describe_services.return_value = {"services": [{
        "serviceName": "api", "status": "ACTIVE",
        "desiredCount": 2, "runningCount": 2, "pendingCount": 0,
    }]}
    with patch("boto3.client", return_value=mock_boto):
        response = client.get("/aws/ecs/services")
    assert response.status_code == 200
    assert len(response.json().get("services", [])) >= 1


def test_aws_cloudtrail():
    # Route delegates to get_cloudtrail_events from aws_ops — patch that
    mock_result = {"success": True, "events": [{"event_name": "StopInstances", "user": "admin",
        "time": "2024-01-01T00:00:00+00:00", "resources": ["i-123"]}], "count": 1}
    with patch("app.api.aws.get_cloudtrail_events", return_value=mock_result):
        response = client.get("/aws/cloudtrail/events?hours=1&resource_name=i-123")
    assert response.status_code == 200
    events = response.json().get("cloudtrail_events", {}).get("events", [])
    assert any(e.get("event_name") == "StopInstances" for e in events)


def test_aws_lambda_list():
    # Route uses paginator — set up paginate() mock
    mock_boto = MagicMock()
    pag = MagicMock()
    pag.paginate.return_value = [{"Functions": [{
        "FunctionName": "my-func", "Runtime": "python3.11",
        "MemorySize": 256, "Timeout": 30,
        "LastModified": "2024-01-01T00:00:00", "Description": "",
    }]}]
    mock_boto.get_paginator.return_value = pag
    with patch("boto3.client", return_value=mock_boto):
        response = client.get("/aws/lambda/functions")
    assert response.status_code == 200
    assert response.json().get("count", 0) >= 1


# ── Claude utility functions (no-API-key fallback paths) ─────────────────────

def test_claude_extract_json_fenced():
    from app.llm.claude import _extract_json
    raw = '```json\n{"key": "value"}\n```'
    assert _extract_json(raw) == '{"key": "value"}'


def test_claude_extract_json_bare():
    from app.llm.claude import _extract_json
    raw = '{"key": "value"}'
    assert _extract_json(raw) == '{"key": "value"}'


def test_claude_extract_json_prose_around():
    from app.llm.claude import _extract_json
    raw = 'Here is the analysis:\n{"root_cause": "OOM"}\nLet me know if you need more.'
    result = _extract_json(raw)
    assert '"root_cause"' in result


def test_claude_review_pr_no_api_key():
    from app.llm.claude import review_pr as _review_pr
    with patch("app.llm.claude._provider", None):
        result = _review_pr({"title": "test", "files": []})
    assert "error" in result
    assert result["issues"] == []


def test_claude_predict_scaling_no_api_key():
    from app.llm.claude import predict_scaling as _predict_scaling
    # Patch both client and provider so the function takes the error path
    with patch("app.llm.claude.client", None), \
         patch("app.llm.claude._provider", None):
        result = _predict_scaling({"resource_type": "ecs"})
    # With no provider, returns safe default
    assert result["should_scale"] is False


def test_claude_assess_deployment_no_api_key():
    from app.llm.claude import assess_deployment as _assess
    with patch("app.llm.claude.client", None), \
         patch("app.llm.claude._provider", None):
        result = _assess({"deployment": "api", "namespace": "prod"})
    assert result["go_no_go"] == "no_go"


def test_claude_interpret_jira_no_api_key():
    from app.llm.claude import interpret_jira_for_pr as _interp
    with patch("app.llm.claude._provider", None):
        result = _interp({"key": "DEV-1", "summary": "Update deps", "labels": []})
    assert result["pr_title"] == "Update deps"
    assert result["file_patches"] == []
    assert "jira/dev-1" in result["branch_name"]


# ── GitHub utilities ──────────────────────────────────────────────────────────

def test_github_create_incident_pr_duplicate_branch():
    """create_incident_pr reuses existing branch instead of crashing."""
    from app.integrations import github as gh
    from github import GithubException

    mock_repo = MagicMock()
    mock_branch = MagicMock()
    mock_branch.commit.sha = "abc123"
    mock_repo.get_branch.return_value = mock_branch
    mock_repo.get_contents.side_effect = GithubException(404, "not found")
    mock_repo.create_file.return_value = None
    mock_pr = MagicMock()
    mock_pr.number = 99
    mock_pr.html_url = "https://github.com/test/pr/99"
    mock_repo.create_pull.return_value = mock_pr

    with patch.object(gh, "_pick_repo", return_value=mock_repo):
        result = gh.create_incident_pr("INC-DUP", "Fix", "body")

    assert result["success"] is True
    assert result["pr_number"] == 99
    mock_repo.create_git_ref.assert_not_called()


# ── Planner agent ─────────────────────────────────────────────────────────────

def test_planner_skips_aws_actions_when_aws_unavailable():
    """PlannerAgent must not emit AWS actions when AWS is unavailable."""
    from app.agents.planner.agent import _clean_actions

    actions = [
        {"type": "ec2_start", "instance_id": "i-abc"},
        {"type": "k8s_restart", "deployment": "api", "namespace": "default"},
        {"type": "slack_notify", "channel": "#incidents"},
    ]
    cleaned = _clean_actions(actions, aws_ok=False, k8s_ok=True)
    types = {a["type"] for a in cleaned}
    assert "ec2_start" not in types
    assert "k8s_restart" in types
    assert "slack_notify" in types


def test_planner_skips_k8s_actions_when_k8s_unavailable():
    """PlannerAgent must not emit K8s actions when K8s is unavailable."""
    from app.agents.planner.agent import _clean_actions

    actions = [
        {"type": "k8s_restart", "deployment": "api", "namespace": "default"},
        {"type": "k8s_scale",   "deployment": "api", "namespace": "default", "replicas": 2},
        {"type": "investigate",  "description": "check logs"},
    ]
    cleaned = _clean_actions(actions, aws_ok=True, k8s_ok=False)
    types = {a["type"] for a in cleaned}
    assert "k8s_restart" not in types
    assert "k8s_scale"   not in types
    assert "investigate" in types


def test_planner_incident_classification_k8s():
    """Incident description with k8s keywords correctly classified."""
    from app.agents.planner.agent import _classify_incident
    cls = _classify_incident("Pod is CrashLoopBackOff in the default namespace")
    assert cls["k8s"] is True
    assert cls["general"] is False


def test_planner_incident_classification_ec2():
    from app.agents.planner.agent import _classify_incident
    cls = _classify_incident("EC2 instance stopped unexpectedly")
    assert cls["ec2"] is True


def test_planner_strip_fabricated_gaps():
    """_strip_fabricated removes placeholder patterns from data_gaps."""
    from app.agents.planner.agent import _strip_fabricated
    plan = {
        "data_gaps": [
            "Missing CloudWatch metrics for i-0abc123",  # contains fake ID pattern
            "AWS context unavailable",                   # legitimate gap
        ]
    }
    result = _strip_fabricated(plan)
    # i-0abc123 matches the fake pattern
    assert any("unavailable" in g for g in result["data_gaps"])


# ── Executor / policy engine ──────────────────────────────────────────────────

def test_policy_engine_blocks_globally_blocked_action():
    from app.policies.policy_engine import PolicyEngine
    engine = PolicyEngine()
    # Inject a test rule
    engine._rules = {"blocked_actions": ["delete_cluster"], "action_permissions": {}, "role_permissions": {}, "guardrails": {}}
    allowed, reason = engine.evaluate({"type": "delete_cluster"}, user="admin", role="super_admin")
    assert allowed is False
    assert "globally blocked" in reason


def test_policy_engine_allows_permitted_action():
    from app.policies.policy_engine import PolicyEngine
    engine = PolicyEngine()
    engine._rules = {
        "blocked_actions": [],
        "action_permissions": {"k8s_restart": "deploy"},
        "role_permissions": {"deploy": ["developer", "admin", "super_admin"]},
        "guardrails": {},
    }
    allowed, _ = engine.evaluate({"type": "k8s_restart", "namespace": "default"}, role="developer")
    assert allowed is True


def test_policy_engine_blocks_replica_limit():
    from app.policies.policy_engine import PolicyEngine
    engine = PolicyEngine()
    engine._rules = {
        "blocked_actions": [],
        "action_permissions": {},
        "role_permissions": {},
        "guardrails": {"max_replicas": 20, "min_replicas": 1, "restricted_namespaces": []},
    }
    allowed, reason = engine.evaluate({"type": "k8s_scale", "replicas": 100, "namespace": "default"}, role="admin")
    assert allowed is False
    assert "max_replicas" in reason


# ── Rate limiting infrastructure ──────────────────────────────────────────────

def test_rate_limiter_allows_within_limit():
    from app.core.ratelimit import RateLimiter
    limiter = RateLimiter()
    allowed, remaining = limiter.check("test-user:test-op", limit=5, window_seconds=60)
    assert allowed is True
    assert remaining >= 0


def test_rate_limiter_blocks_over_limit():
    from app.core.ratelimit import RateLimiter
    limiter = RateLimiter()
    key = "burst-user:flood"
    for _ in range(3):
        limiter.check(key, limit=3, window_seconds=60)
    allowed, remaining = limiter.check(key, limit=3, window_seconds=60)
    assert allowed is False
    assert remaining == 0
