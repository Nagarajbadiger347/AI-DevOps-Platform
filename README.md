# AI DevOps Intelligence Platform

Autonomous DevOps management powered by multi-agent AI — built by **Nagaraj**.

One platform to detect incidents, analyse root cause, execute remediation, assess deployments, and close the loop back to Jira and GitHub — automatically.

---

## What it does

| Capability | Description |
|---|---|
| 🤖 **Autonomous Incident Pipeline** | One API call collects AWS + K8s + GitHub data, runs Claude AI RCA, executes fixes (K8s restart, GitHub PR, Jira, Slack, OpsGenie) |
| 🔍 **Pre-deployment Assessment** | Before any deploy, Claude assesses current cluster state, active alarms, and past incidents → go / no-go decision with checklist |
| 🎫 **Jira → Auto PR** | When a Jira change-request ticket is created, Claude interprets it and automatically opens a GitHub PR with file patches |
| ☁ **AWS Observability** | Read-only data collection across EC2, ECS, Lambda, RDS, ALB, CloudWatch Logs/Metrics/Alarms, CloudTrail |
| ☸ **Kubernetes Operations** | Health checks + rolling restarts + scale deployments + fetch pod logs |
| 📈 **Predictive Scaling** | Analyse CloudWatch metric trends and predict if scaling is needed before a breach occurs |
| 🔎 **AI PR Review** | Claude reviews GitHub PRs for security issues, infra concerns, and code quality — posts comment directly on the PR |
| 🔐 **RBAC** | Dynamic role-based access control; sensitive endpoints enforce `X-User` header |
| 🧠 **ChromaDB Memory** | All incidents stored in vector DB; past similar incidents feed into future assessments |

---

## Directory Structure

```
ai-devops-platform/
├── app/
│   ├── orchestrator/main.py        # FastAPI app — all REST & WebSocket endpoints
│   ├── agents/
│   │   └── incident_pipeline.py    # Autonomous end-to-end incident response pipeline
│   ├── llm/claude.py               # Claude AI functions (RCA, synthesis, review, assess, predict)
│   ├── correlation/engine.py       # Event correlation logic
│   ├── plugins/
│   │   ├── aws_checker.py          # AWS infrastructure health check
│   │   ├── linux_checker.py        # Linux node health check
│   │   └── k8s_checker.py          # Kubernetes cluster health check
│   ├── integrations/
│   │   ├── aws_ops.py              # AWS observability + predictive scaling metrics
│   │   ├── github.py               # Commits, PRs, diffs, PR review, incident PRs
│   │   ├── jira.py                 # Jira incident creation, comments, issue fetch
│   │   ├── slack.py                # Slack war-room automation
│   │   ├── opsgenie.py             # OpsGenie on-call notification
│   │   ├── k8s_ops.py              # K8s restart / scale / logs
│   │   └── vscode.py               # VS Code action stub
│   ├── memory/vector_db.py         # ChromaDB incident storage + similarity search
│   └── security/rbac.py            # Role-based access control
├── tests/test_main.py              # 68 pytest tests
├── test_websocket.py               # Manual WebSocket test
├── requirements.txt
├── Dockerfile                      # Python 3.11-slim
├── docker-compose.yml
├── .env.example
└── .env                            # Your credentials (not committed)
```

---

## Setup

### Requirements

- Python 3.11+  (3.9+ works for local dev)
- Docker & Docker Compose (optional)

### Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials
uvicorn app.orchestrator.main:app --reload --host 127.0.0.1 --port 8000
```

### Docker

```bash
cp .env.example .env
docker compose up --build
```

Open: http://127.0.0.1:8000

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key — all AI features |
| `GITHUB_TOKEN` | For GitHub features | Personal access token |
| `GITHUB_REPO` | For GitHub features | `owner/repo` format |
| `SLACK_BOT_TOKEN` | For Slack | Bot token |
| `SLACK_CHANNEL` | For Slack | Channel name (default: `#general`) |
| `JIRA_URL` | For Jira | e.g. `https://yourorg.atlassian.net` |
| `JIRA_USER` | For Jira | User email |
| `JIRA_TOKEN` | For Jira | API token |
| `JIRA_PROJECT` | For Jira | Project key (default: `DEVOPS`) |
| `OPSGENIE_API_KEY` | For OpsGenie | API key |
| `AWS_REGION` | For AWS | Region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | For AWS | Access key (or use IAM role) |
| `AWS_SECRET_ACCESS_KEY` | For AWS | Secret key (or use IAM role) |
| `K8S_IN_CLUSTER` | For K8s in pod | Set `true` when running inside a pod |
| `KUBECONFIG` | For K8s local | Path to kubeconfig (default: `~/.kube/config`) |
| `RBAC_CONFIG_PATH` | Optional | Path to JSON file with user→role mappings |

Features degrade gracefully when credentials are missing — they return an error response rather than crashing.

---

## API Reference

### General

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/health` | Health status |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc reference |

### AI & Correlation

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/correlate` | `[{id, type, source, payload}]` | Correlate events, find patterns |
| `POST` | `/llm/analyze` | `{incident_id, details}` | Claude root cause analysis |

### Infrastructure Checks

| Method | Path | Description |
|---|---|---|
| `GET` | `/check/aws` | AWS EC2 / CloudWatch health |
| `GET` | `/check/linux` | Linux node health |

### Kubernetes

> ⚠️ `/k8s/restart` and `/k8s/scale` require `X-User` header with `deploy` permission.

| Method | Path | Params / Body | Description |
|---|---|---|---|
| `GET` | `/check/k8s` | — | Cluster summary |
| `GET` | `/check/k8s/nodes` | — | Per-node ready status |
| `GET` | `/check/k8s/pods` | `namespace` | Pod status |
| `GET` | `/check/k8s/deployments` | `namespace` | Deployment rollout status |
| `POST` | `/k8s/restart` | `{namespace, deployment}` | Rolling restart |
| `POST` | `/k8s/scale` | `{namespace, deployment, replicas}` | Scale replicas |
| `GET` | `/k8s/logs` | `namespace, pod, container, tail_lines` | Fetch pod logs |

### AWS Observability

All AWS endpoints are **read-only**.

| Method | Path | Params | Description |
|---|---|---|---|
| `GET` | `/aws/ec2/instances` | `state` | List instances |
| `GET` | `/aws/ec2/status` | `instance_id` | Status checks |
| `GET` | `/aws/ec2/console` | `instance_id` | Serial console output |
| `GET` | `/aws/logs/groups` | `prefix, limit` | List log groups |
| `GET` | `/aws/logs/recent` | `log_group, minutes, limit` | Recent log events |
| `GET` | `/aws/logs/search` | `log_group, pattern, hours` | Search logs by pattern |
| `GET` | `/aws/cloudwatch/alarms` | `state` | CloudWatch alarms |
| `POST` | `/aws/cloudwatch/metrics` | `{namespace, metric_name, dimensions, hours}` | Fetch metric series |
| `GET` | `/aws/ecs/services` | `cluster` | ECS service counts |
| `GET` | `/aws/ecs/stopped-tasks` | `cluster, limit` | Stopped task reasons |
| `GET` | `/aws/lambda/errors` | `function_name, hours` | Lambda error metrics |
| `GET` | `/aws/rds/events` | `db_instance_id, hours` | RDS events |
| `GET` | `/aws/elb/target-health` | `target_group_arn` | ALB target health |
| `GET` | `/aws/cloudtrail/events` | `hours, resource_name` | Recent API changes |
| `POST` | `/aws/diagnose` | `{resource_type, resource_id, log_group, hours}` | AI root cause analysis from live AWS data |
| `POST` | `/aws/predict-scaling` | `{resource_type, resource_id, hours}` | Predict if scaling needed from metric trends |

### Incident Management

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/incident/war-room` | `{topic}` | Create Slack war room |
| `POST` | `/incident/jira` | `{summary, description}` | Create Jira incident |
| `POST` | `/incident/opsgenie` | `{message}` | Notify OpsGenie on-call |
| `POST` | `/incident/github/issue` | `{title, body}` | Create GitHub issue |
| `POST` | `/incident/github/pr` | `{head, base, title, body}` | Create GitHub PR |

### Security / RBAC

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/security/check` | `{user, action}` | Check if user can perform action |
| `POST` | `/security/roles` | `{user, role}` | Assign role to user |
| `DELETE` | `/security/roles/{user}` | — | Revoke user role |

**Roles:** `admin` · `developer` · `viewer`

| Role | Permissions |
|---|---|
| `admin` | deploy, rollback, read, write, delete, manage_users |
| `developer` | deploy, read, write |
| `viewer` | read |

### Memory

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/memory/incidents` | `{id, type, source, payload}` | Store incident in ChromaDB |

### WebSocket

```
WS /realtime/events
```

Send events as JSON, receive correlation + AI analysis in real time:

```json
// Send
{"id": "1", "type": "error", "source": "api", "payload": {"msg": "500 spike"}}

// Receive
{"correlation": {...}, "analysis": {...}}
```

---

## Autonomous Incident Pipeline

**`POST /incident/run`** — the flagship endpoint.

> ⚠️ Requires `X-User` header with `deploy` permission when `auto_remediate: true`.

```
Trigger → Collect → Synthesise → Remediate → Report
```

| Step | What happens |
|---|---|
| **1. Collect** | AWS, K8s, and GitHub data collected in parallel threads |
| **2. Synthesise** | Claude AI determines root cause, severity, confidence, and action plan |
| **3. Remediate** | K8s restart/scale (if `auto_remediate: true`), GitHub PR with fix, Jira ticket, Slack war room, OpsGenie alert |
| **4. Store** | Incident saved to ChromaDB for future similarity search |
| **5. Report** | Full structured report with every action's result |

```bash
curl -X POST http://127.0.0.1:8000/incident/run \
  -H "Content-Type: application/json" \
  -H "X-User: alice" \
  -d '{
    "incident_id":    "INC-001",
    "description":   "High 5xx rate on API",
    "severity":      "critical",
    "aws":           {"resource_type": "ecs", "resource_id": "my-cluster", "log_group": "/ecs/api"},
    "k8s":           {"namespace": "production"},
    "auto_remediate": true,
    "hours":          2
  }'
```

---

## Pre-deployment Assessment

**`POST /deploy/assess`** — get a go/no-go before deploying.

> Requires `X-User` header with `deploy` permission.

Collects current K8s state, active AWS alarms, recent GitHub commits, and past similar incidents from ChromaDB — Claude returns a risk assessment with checklist.

```bash
curl -X POST http://127.0.0.1:8000/deploy/assess \
  -H "Content-Type: application/json" \
  -H "X-User: alice" \
  -d '{
    "deployment":  "api-server",
    "namespace":   "production",
    "new_image":   "myapp:v2.1.0",
    "description": "Add new payment endpoint"
  }'
```

**Response includes:** `go_no_go` (`go` / `go_with_caution` / `no_go`), `risk_score`, `concerns[]`, `checklist[]`, `safe_window`

---

## Jira Webhook → Auto PR

**`POST /jira/webhook`** — register this URL in Jira to auto-create GitHub PRs.

**Jira setup:** Project Settings → Webhooks → URL: `https://your-platform/jira/webhook` → Event: Issue Created

Triggers when:
- Issue type is **Change Request** (or Task / Story), **or**
- Issue has label **`auto-pr`**

**Flow:**
1. Claude reads the Jira ticket description
2. Generates a PR plan with title, body, and best-effort file patches
3. Creates a GitHub branch `jira/<ticket-key>-<slug>` and opens a PR
4. Posts the PR link as a comment back on the Jira ticket

---

## AI PR Review

**`POST /github/review-pr`**

```bash
curl -X POST http://127.0.0.1:8000/github/review-pr \
  -H "Content-Type: application/json" \
  -d '{"pr_number": 42, "post_comment": true}'
```

Fetches the PR diff from GitHub, runs Claude analysis for security issues, infra changes, and code quality. Set `post_comment: true` to automatically post the review on the PR.

---

## RBAC Usage

Assign a role before calling protected endpoints:

```bash
# Assign developer role
curl -X POST http://127.0.0.1:8000/security/roles \
  -H "Content-Type: application/json" \
  -d '{"user": "alice", "role": "developer"}'

# Use protected endpoint
curl -X POST http://127.0.0.1:8000/k8s/restart \
  -H "Content-Type: application/json" \
  -H "X-User: alice" \
  -d '{"namespace": "default", "deployment": "api-server"}'
```

**Protected endpoints** (require `X-User` with `deploy` permission):
- `POST /k8s/restart`
- `POST /k8s/scale`
- `POST /deploy/assess`
- `POST /incident/run` (only when `auto_remediate: true`)

Load users from file — set `RBAC_CONFIG_PATH=/path/to/roles.json`:

```json
{"alice": "developer", "bob": "viewer", "charlie": "admin"}
```

---

## Running Tests

```bash
pytest -q                         # all 68 tests
pytest -q tests/test_main.py     # API tests only
python test_websocket.py          # manual WebSocket test (needs running server)
```
