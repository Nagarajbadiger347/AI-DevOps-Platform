# AI-Powered DevOps Intelligence Platform

A FastAPI-based platform for autonomous DevOps management using multi-agent AI orchestration, a plugin-based architecture, and integrations with common DevOps tools.

## Features

- **FastAPI** REST API + WebSocket server
- **Claude LLM** integration for AI-driven root cause analysis (via `anthropic` SDK)
- **Correlation engine** — pattern-based event correlation
- **Plugin system** — AWS (EC2, S3, CloudWatch) and Linux health checks
- **Integrations** — Slack, Jira, OpsGenie, GitHub, VS Code, Kubernetes
- **ChromaDB** vector memory for incident storage and similarity search
- **RBAC** — role-based access control with runtime user management

---

## Directory Structure

```
ai-devops-platform/
├── app/
│   ├── orchestrator/main.py      # FastAPI app, all REST & WebSocket endpoints
│   ├── agents/
│   │   └── incident_pipeline.py  # Autonomous end-to-end incident response pipeline
│   ├── llm/claude.py             # Claude AI integration (RCA + AWS diagnosis + synthesis)
│   ├── correlation/engine.py     # Event correlation logic
│   ├── plugins/
│   │   ├── aws_checker.py        # AWS infrastructure health check
│   │   ├── linux_checker.py      # Linux node health check
│   │   └── k8s_checker.py        # Kubernetes cluster health check
│   ├── integrations/
│   │   ├── slack.py              # Slack war-room automation
│   │   ├── jira.py               # Jira incident creation
│   │   ├── opsgenie.py           # OpsGenie on-call notification
│   │   ├── github.py             # GitHub issue / PR / commit observability
│   │   ├── aws_ops.py            # AWS observability (read-only: logs, metrics, events)
│   │   ├── vscode.py             # VS Code action stub
│   │   └── k8s_ops.py            # Kubernetes operations (restart, scale, logs)
│   ├── memory/vector_db.py       # ChromaDB incident storage
│   └── security/rbac.py          # Role-based access control
├── tests/test_main.py            # Pytest test suite
├── test_websocket.py             # Manual WebSocket test script
├── requirements.txt
├── Dockerfile                    # Python 3.11-slim
├── docker-compose.yml
├── .env.example                  # Environment variable template
└── .env                          # Your local credentials (not committed)
```

---

## Setup

### Requirements

- Python 3.11+
- Docker & Docker Compose (optional)

### Local Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and fill in your credentials (see Environment Variables below)

# 4. Start the server
uvicorn app.orchestrator.main:app --reload --host 127.0.0.1 --port 8000
```

### Docker Setup

```bash
# Copy and fill in credentials
cp .env.example .env

# Build and start with Docker Compose
docker compose up --build

# Or manually
docker build -t ai-devops-platform .
docker run -p 8000:8000 --env-file .env ai-devops-platform
```

Code changes are reflected immediately due to volume mounting and `uvicorn --reload`.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key for LLM analysis |
| `SLACK_BOT_TOKEN` | Slack bot token for war-room automation |
| `SLACK_CHANNEL` | Slack channel (default: `#general`) |
| `JIRA_URL` | Jira instance URL (e.g. `https://yourorg.atlassian.net`) |
| `JIRA_USER` | Jira user email |
| `JIRA_TOKEN` | Jira API token |
| `JIRA_PROJECT` | Jira project key (e.g. `DEVOPS`) |
| `OPSGENIE_API_KEY` | OpsGenie API key |
| `GITHUB_TOKEN` | GitHub personal access token |
| `GITHUB_REPO` | GitHub repo in `owner/repo` format |
| `AWS_REGION` | AWS region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | AWS access key (if not using IAM role) |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (if not using IAM role) |
| `RBAC_CONFIG_PATH` | Path to a JSON file with user→role mappings (optional) |
| `K8S_IN_CLUSTER` | Set to `true` when running inside a K8s pod |
| `KUBECONFIG` | Path to kubeconfig file (defaults to `~/.kube/config`) |

Features that depend on missing credentials degrade gracefully — they return an error response rather than crashing.

---

## API Endpoints

### General

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root / sanity check |
| `GET` | `/health` | Health status |

### AI & Correlation

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/correlate` | `[{id, type, source, payload}]` | Correlate a list of events |
| `POST` | `/llm/analyze` | `{incident_id, details}` | Claude root cause analysis |

### Infrastructure Checks

| Method | Path | Description |
|---|---|---|
| `GET` | `/check/aws` | AWS EC2 / S3 / CloudWatch health |
| `GET` | `/check/linux` | Linux node health |

### Incident Management

| Method | Path | Query Params | Description |
|---|---|---|---|
| `POST` | `/incident/war-room` | — | Create a Slack war-room |
| `POST` | `/incident/jira` | — | Create a Jira incident |
| `POST` | `/incident/opsgenie` | — | Notify OpsGenie on-call |
| `POST` | `/incident/github/issue` | — | Create a GitHub issue |
| `POST` | `/incident/github/pr` | `head`, `base` (default `main`) | Create a GitHub pull request |

### VS Code Integration

| Method | Path | Query Params | Description |
|---|---|---|---|
| `POST` | `/vscode/action` | `action` (`analyze`\|`lint`\|`format`\|`test`), `file_path` | Trigger a VS Code action |
| `POST` | `/vscode/open` | `file_path` | Open a file in VS Code |

### Memory

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/memory/incidents` | `{id, type, source, payload}` | Store an incident in ChromaDB |

### Security / RBAC

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/security/check` | `{user, action}` | Check whether a user is allowed to perform an action |
| `POST` | `/security/roles` | `{user, role}` | Assign a role to a user |
| `DELETE` | `/security/roles/{user}` | — | Revoke a user's role |

Available roles: `admin`, `developer`, `viewer`.

#### Role Permissions

| Role | Allowed Actions |
|---|---|
| `admin` | `deploy`, `rollback`, `read`, `write`, `delete`, `manage_users` |
| `developer` | `deploy`, `read`, `write` |
| `viewer` | `read` |

#### Loading users from a file

Create a JSON file mapping usernames to roles:

```json
{
  "alice": "developer",
  "bob": "viewer",
  "charlie": "admin"
}
```

Then set `RBAC_CONFIG_PATH=/path/to/roles.json` before starting the server.

### AWS Observability & AI Diagnosis

All AWS endpoints are **read-only** — designed to collect diagnostic data for AI root cause analysis.

#### EC2

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/ec2/instances` | `state` (optional) | List instances with health state |
| `GET` | `/aws/ec2/status` | `instance_id` | System & instance status checks (detects hardware failures) |
| `GET` | `/aws/ec2/console` | `instance_id` | Serial console output — kernel panics, boot errors |

#### CloudWatch Logs

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/logs/groups` | `prefix`, `limit` | List available log groups |
| `GET` | `/aws/logs/recent` | `log_group`, `minutes`, `limit` | Fetch recent log events |
| `GET` | `/aws/logs/search` | `log_group`, `pattern`, `hours`, `limit` | Search logs by keyword/pattern |

#### CloudWatch Metrics & Alarms

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/cloudwatch/alarms` | `state` (`OK`/`ALARM`/`INSUFFICIENT_DATA`) | List CloudWatch alarms |
| `POST` | `/aws/cloudwatch/metrics` | `{namespace, metric_name, dimensions, hours, period, stat}` | Fetch any metric series |

#### ECS

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/ecs/services` | `cluster` | Service running vs desired count |
| `GET` | `/aws/ecs/stopped-tasks` | `cluster`, `limit` | Stopped task reasons & container exit codes |

#### Lambda

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/lambda/functions` | — | List functions with runtime info |
| `GET` | `/aws/lambda/errors` | `function_name`, `hours` | Error, throttle & duration metrics |

#### RDS

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/rds/instances` | — | List DB instances with status |
| `GET` | `/aws/rds/events` | `db_instance_id`, `hours` | DB events — failovers, restarts, errors |

#### ELB / CloudTrail

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/elb/target-health` | `target_group_arn` | ALB target group health |
| `GET` | `/aws/cloudtrail/events` | `hours`, `resource_name` | Recent API changes — who did what before the incident |

#### AI Diagnosis

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/aws/diagnose` | `{resource_type, resource_id, log_group, hours}` | Collect all observability data for a resource and run Claude AI root cause analysis |

`resource_type` can be: `ec2`, `ecs`, `lambda`, `rds`, `alb`

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/aws/diagnose \
  -H "Content-Type: application/json" \
  -d '{"resource_type": "ec2", "resource_id": "i-0abc123", "log_group": "/app/api", "hours": 2}'
```
Returns: `summary`, `root_cause`, `confidence`, `severity`, `findings[]`, `recommended_actions[]`

### Kubernetes

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/check/k8s` | — | Cluster summary (nodes, pods, deployments) |
| `GET` | `/check/k8s/nodes` | — | Per-node ready status |
| `GET` | `/check/k8s/pods` | `namespace` (default `default`) | Pod status by namespace |
| `GET` | `/check/k8s/deployments` | `namespace` (default `default`) | Deployment rollout status |
| `POST` | `/k8s/restart` | `{namespace, deployment}` | Rolling restart a deployment |
| `POST` | `/k8s/scale` | `{namespace, deployment, replicas}` | Scale deployment replica count |
| `GET` | `/k8s/logs` | `namespace`, `pod`, `container`, `tail_lines` | Fetch pod logs |

**Authentication:** uses `~/.kube/config` by default. Set `K8S_IN_CLUSTER=true` when running inside a pod.

---

## Autonomous Incident Response Pipeline

`POST /incident/run` is the flagship endpoint — one call triggers the full end-to-end loop:

```
Trigger → Collect (AWS + K8s + GitHub) → AI Synthesis → Execute Actions → Report
```

| Step | What happens |
|---|---|
| **1. Collect** | Parallel data collection from AWS (CloudWatch, ECS, Lambda, RDS, etc.), Kubernetes (pods, deployments, events), and GitHub (recent commits, merged PRs, diffs) |
| **2. Synthesise** | Claude AI analyses all data, determines root cause, severity, confidence, and produces a typed action plan |
| **3. Act** | Executes recommended actions: K8s restart/scale, GitHub PR with fix report, Jira ticket, Slack war room, OpsGenie alert |
| **4. Store** | Incident stored in ChromaDB for future similarity search |
| **5. Report** | Full structured report returned with root cause, findings, confidence score, and results of every action |

### Request body

```json
{
  "incident_id":    "INC-001",
  "description":   "High 5xx error rate on the API service",
  "severity":      "critical",
  "aws": {
    "resource_type": "ecs",
    "resource_id":   "my-cluster",
    "log_group":     "/ecs/api-service"
  },
  "k8s": {
    "namespace": "production"
  },
  "auto_remediate": true,
  "hours": 2
}
```

| Field | Default | Description |
|---|---|---|
| `incident_id` | required | Unique identifier for this incident |
| `description` | required | Human-readable description of what went wrong |
| `severity` | `"high"` | Reported severity: `critical`, `high`, `medium`, `low` |
| `aws` | omit to skip | AWS resource to investigate (`resource_type`, `resource_id`, `log_group`) |
| `k8s` | omit to skip | Kubernetes config (`namespace`) |
| `auto_remediate` | `false` | When `true`, execute K8s restart/scale automatically; when `false`, those actions are listed but skipped pending manual approval |
| `hours` | `2` | Lookback window for all observability data |

### Response

```json
{
  "incident_id": "INC-001",
  "root_cause":  "Memory leak in user-service pod causing OOM kills",
  "confidence":  0.92,
  "ai_severity": "critical",
  "findings":    ["Pod restarted 4 times in 2 hours", "Heap usage at 98%"],
  "actions_taken": [
    { "type": "github_pr",     "result": { "success": true, "pr_number": 42 } },
    { "type": "jira_ticket",   "result": { "success": true, "issue_key": "DEVOPS-17" } },
    { "type": "slack_warroom", "result": { "success": true } }
  ],
  "observability": { "aws_collected": true, "k8s_collected": true, "github_collected": true },
  "started_at":  "...",
  "completed_at": "..."
}
```

### WebSocket

```
WS /realtime/events
```

Send a single event or a list of events as JSON. Receive correlation and AI analysis in real time:

```json
// Input
{"id": "1", "type": "error", "source": "app", "payload": {"msg": "boom"}}

// Output
{"correlation": {...}, "analysis": {...}}
```

---

## Running Tests

```bash
pytest -q
```

To run only the API tests:

```bash
pytest -q tests/test_main.py
```

To run the manual WebSocket test (requires a running server on port 8000):

```bash
python test_websocket.py
```

---

## Interactive API Docs

Once the server is running, visit:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
