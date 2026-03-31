# AI DevOps Intelligence Platform

Autonomous DevOps management powered by a **multi-agent AI system** — built by **Nagaraj**.

One platform to detect incidents, analyse root cause, plan and safely execute remediation, assess deployments, and close the loop back to Jira and GitHub — automatically.

---

## What it does

| Capability | Description |
|---|---|
| 🤖 **Multi-Agent Incident Pipeline** | LangGraph-orchestrated agents collect context, plan remediation, score risk, execute safely (with policy guardrails), validate outcome, and store to memory |
| 🔍 **Pre-deployment Assessment** | Before any deploy, Claude assesses cluster state, active alarms, and past incidents → go / no-go decision with checklist |
| 🎫 **Jira → Auto PR** | When a Jira change-request ticket is created, Claude interprets it and opens a GitHub PR with file patches |
| ☁️ **AWS Observability** | Read-only collection across EC2, ECS, Lambda, RDS, ALB, CloudWatch Logs/Metrics/Alarms, CloudTrail |
| ☸️ **Kubernetes Operations** | Health checks + rolling restarts + scale deployments + fetch pod logs |
| 📈 **Predictive Scaling** | Analyse CloudWatch metric trends and predict if scaling is needed before a breach occurs |
| 🔎 **AI PR Review** | Claude reviews GitHub PRs for security issues, infra concerns, and code quality |
| 🔐 **RBAC + Policy Engine** | Role-based access control + declarative policy guardrails enforced before every action |
| 🧠 **ChromaDB Memory** | All incidents stored in vector DB; similar past incidents feed into future planning decisions |
| 🔁 **Continuous Monitoring** | Background loop polls K8s/AWS for anomalies and auto-triggers the pipeline |
| 🔀 **Multi-LLM Support** | Claude (primary) → OpenAI (fallback) → Groq/Llama → Ollama (local) — automatic fallback chain |

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
  API Request            │         LangGraph Orchestrator           │
  Webhook         ──────▶│                                         │
  Monitor Loop           │  collect_context                        │
                         │       │  (AWS + K8s + GitHub agents     │
                         │       │   + ChromaDB similar incidents) │
                         │       ▼                                 │
                         │  PlannerAgent  ──── LLMFactory          │
                         │       │         (Claude/OpenAI/Groq)    │
                         │       ▼                                 │
                         │  DecisionAgent (risk score + approval)  │
                         │       │                                 │
                         │       ├── auto_remediate=true ──▶ Executor
                         │       └── high risk / low confidence ──▶ awaiting_approval (END)
                         │                          │              │
                         │                     PolicyEngine        │
                         │                     ActionRegistry      │
                         │                          │              │
                         │                       Validator         │
                         │                     (re-check health)   │
                         │                       /       \         │
                         │               passed           failed   │
                         │                 │              retry/escalate
                         │                 ▼                       │
                         │           MemoryAgent (ChromaDB)        │
                         └─────────────────────────────────────────┘
```

### Core design principles

| Layer | Responsibility |
|---|---|
| **Agents** | Decision / data collection units — no direct infra calls |
| **LangGraph Graph** | Controls workflow, branching, retry logic, error propagation |
| **LLM** | Reasoning only — PlannerAgent and analysis functions |
| **Executor** | Performs all actions safely via ActionRegistry |
| **PolicyEngine** | Enforces guardrails before every action (role + parameter limits) |
| **Memory** | ChromaDB stores outcomes and informs future planning |

---

## Directory Structure

```
app/
├── orchestrator/
│   ├── main.py           # FastAPI server — all REST & WebSocket endpoints (v1 + v2)
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── state.py          # PipelineState TypedDict — shared across all nodes
│   └── runner.py         # run_pipeline() — public entry point
│
├── agents/
│   ├── base.py                    # BaseAgent ABC
│   ├── planner/agent.py           # PlannerAgent → structured JSON plan via LLM
│   ├── decision/agent.py          # DecisionAgent → risk score + approval gate
│   ├── infra/aws_agent.py         # AWS context collector (read-only)
│   ├── infra/k8s_agent.py         # K8s context collector (read-only)
│   ├── scm/github_agent.py        # GitHub commits/PRs collector
│   ├── memory/agent.py            # ChromaDB read (retrieve) + write (store)
│   └── incident_pipeline.py       # v1 pipeline (kept for backwards compatibility)
│
├── llm/
│   ├── base.py           # BaseLLM ABC + LLMResponse dataclass
│   ├── claude.py         # ClaudeProvider + all existing AI functions
│   ├── openai.py         # OpenAIProvider (GPT-4o fallback)
│   └── factory.py        # LLMFactory — automatic provider selection + fallback
│
├── execution/
│   ├── executor.py        # Policy-gated action execution
│   ├── validator.py       # Post-execution health verification
│   └── action_registry.py # Action type → integration function mapping
│
├── policies/
│   ├── policy_engine.py   # Evaluates actions against rules before execution
│   └── rules.json         # Declarative rules: blocked actions, RBAC, guardrails
│
├── monitoring/
│   └── loop.py            # Background anomaly detection loop
│
├── integrations/          # External service connectors (unchanged)
│   ├── aws_ops.py         # AWS observability + predictive scaling metrics
│   ├── github.py          # Commits, PRs, diffs, PR review, incident PRs
│   ├── jira.py            # Jira incident creation, comments, issue fetch
│   ├── slack.py           # Slack war-room automation
│   ├── opsgenie.py        # OpsGenie on-call notification
│   ├── k8s_ops.py         # K8s restart / scale / logs
│   ├── gitlab_ops.py      # GitLab pipelines/deployments
│   ├── grafana.py         # Grafana alert queries
│   └── universal_collector.py  # Multi-integration parallel aggregator
│
├── plugins/               # Local health checkers (unchanged)
│   ├── aws_checker.py
│   ├── k8s_checker.py
│   └── linux_checker.py
│
├── memory/
│   └── vector_db.py       # ChromaDB incident storage + similarity search
│
├── security/
│   └── rbac.py            # Role-based access control with file persistence
│
├── core/
│   ├── config.py          # Centralised pydantic-settings configuration
│   └── logging.py         # Structured JSON logger + correlation IDs
│
└── correlation/
    └── engine.py          # Event correlation logic

tests/test_main.py         # 68 pytest tests
test_websocket.py          # Manual WebSocket test
requirements.txt
Dockerfile                 # Python 3.11-slim
docker-compose.yml
.env.example
```

---

## Setup

### Requirements

- Python 3.9+ (3.11 recommended)
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

### LLM Providers

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `claude` | Preferred provider: `claude` \| `openai` \| `groq` |
| `ANTHROPIC_API_KEY` | — | Claude API key — primary provider |
| `OPENAI_API_KEY` | — | OpenAI API key — automatic fallback |
| `GROQ_API_KEY` | — | Groq API key — secondary fallback (Llama 3.3-70B) |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama — final fallback, no key needed |

### Multi-Agent Pipeline

| Variable | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE_THRESHOLD` | `0.6` | Plans below this confidence always require approval |
| `AUTO_EXECUTE_RISK_LEVELS` | `low,medium` | Risk levels that auto-execute without human approval |

### Monitoring Loop

| Variable | Default | Description |
|---|---|---|
| `ENABLE_MONITOR_LOOP` | `false` | Enable background anomaly detection |
| `MONITOR_INTERVAL_SECONDS` | `60` | Polling interval |
| `AUTO_REMEDIATE_ON_MONITOR` | `false` | Auto-fix detected anomalies (alert-only when false) |

### Integrations

| Variable | Required for | Description |
|---|---|---|
| `GITHUB_TOKEN` | GitHub features | Personal access token |
| `GITHUB_REPO` | GitHub features | `owner/repo` format |
| `SLACK_BOT_TOKEN` | Slack | Bot token |
| `SLACK_CHANNEL` | Slack | Default channel (default: `#incidents`) |
| `JIRA_URL` | Jira | e.g. `https://yourorg.atlassian.net` |
| `JIRA_USER` | Jira | User email |
| `JIRA_TOKEN` | Jira | API token |
| `JIRA_PROJECT` | Jira | Project key |
| `OPSGENIE_API_KEY` | OpsGenie | API key |
| `AWS_REGION` | AWS | Region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | AWS | Access key (or use IAM role) |
| `AWS_SECRET_ACCESS_KEY` | AWS | Secret key (or use IAM role) |
| `K8S_IN_CLUSTER` | K8s (in-pod) | Set `true` when running inside a pod |
| `KUBECONFIG` | K8s (local) | Path to kubeconfig (default: `~/.kube/config`) |
| `RBAC_CONFIG_PATH` | Optional | Path to JSON file with user→role mappings |
| `CORS_ORIGINS` | Optional | Comma-separated allowed CORS origins |

All integrations degrade gracefully — missing credentials return a structured error rather than crashing.

---

## API Reference

### General

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/health` | Health status |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc reference |

### Incident Pipelines

| Method | Path | Description |
|---|---|---|
| `POST` | `/incident/run` | **v1** — monolithic pipeline (original, backwards-compatible) |
| `POST` | `/v2/incident/run` | **v2** — LangGraph multi-agent pipeline with policy engine |

Both accept the same core fields. v2 adds `user`, `role`, `aws_cfg`, `k8s_cfg`, `slack_channel`.

> ⚠️ Requires `X-User` header with `deploy` permission when `auto_remediate: true`.

**v2 request body:**

```json
{
  "incident_id":    "INC-001",
  "description":   "API pods crash-looping in prod",
  "auto_remediate": false,
  "user":           "alice",
  "role":           "developer",
  "aws_cfg":        {"resource_type": "ecs", "resource_id": "prod-cluster", "log_group": "/ecs/api"},
  "k8s_cfg":        {"namespace": "production"},
  "hours":          2,
  "slack_channel":  "#incidents"
}
```

**v2 response includes:**
- `plan` — structured JSON plan from PlannerAgent (`actions`, `confidence`, `risk`, `root_cause`)
- `executed_actions` — each action's result
- `blocked_actions` — actions blocked by policy (with reason)
- `validation_passed` — post-execution health check result
- `risk_score` — numeric risk score
- `requires_human_approval` — whether approval gate was triggered
- `status` — `completed` \| `escalated` \| `awaiting_approval` \| `failed`
- `correlation_id` — for request tracing

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

> ⚠️ `/k8s/restart` and `/k8s/scale` require `X-User` with `deploy` permission.

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

| Method | Path | Description |
|---|---|---|
| `GET` | `/aws/ec2/instances` | List EC2 instances |
| `GET` | `/aws/ec2/status` | EC2 status checks |
| `GET` | `/aws/ec2/console` | Serial console output |
| `GET` | `/aws/logs/groups` | List CloudWatch log groups |
| `GET` | `/aws/logs/recent` | Recent log events |
| `GET` | `/aws/logs/search` | Search logs by pattern |
| `GET` | `/aws/cloudwatch/alarms` | CloudWatch alarms |
| `POST` | `/aws/cloudwatch/metrics` | Fetch metric series |
| `GET` | `/aws/ecs/services` | ECS service counts |
| `GET` | `/aws/ecs/stopped-tasks` | Stopped ECS task reasons |
| `GET` | `/aws/lambda/functions` | List Lambda functions |
| `GET` | `/aws/lambda/errors` | Lambda error metrics |
| `GET` | `/aws/rds/instances` | RDS instance list |
| `GET` | `/aws/rds/events` | RDS events |
| `GET` | `/aws/elb/target-health` | ALB target health |
| `GET` | `/aws/cloudtrail/events` | Recent CloudTrail API events |
| `GET` | `/aws/s3/buckets` | S3 bucket list |
| `GET` | `/aws/sqs/queues` | SQS queue list |
| `GET` | `/aws/dynamodb/tables` | DynamoDB table list |
| `GET` | `/aws/route53/healthchecks` | Route53 health checks |
| `GET` | `/aws/sns/topics` | SNS topic list |
| `POST` | `/aws/diagnose` | AI root cause analysis from live AWS data |
| `POST` | `/aws/predict-scaling` | Predict if scaling needed from metric trends |

### Incident Management

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/incident/war-room` | `{topic}` | Create Slack war room |
| `POST` | `/incident/jira` | `{summary, description}` | Create Jira incident |
| `POST` | `/incident/opsgenie` | `{message}` | Notify OpsGenie on-call |
| `POST` | `/incident/github/issue` | `{title, body}` | Create GitHub issue |
| `POST` | `/incident/github/pr` | `{head, base, title, body}` | Create GitHub PR |

### Deployment & Code Review

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/deploy/assess` | `{deployment, namespace, new_image, description}` | Pre-deploy go/no-go assessment |
| `POST` | `/github/review-pr` | `{pr_number, post_comment}` | AI code review of a GitHub PR |

### Security / RBAC

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/security/check` | `{user, action}` | Check if user can perform action |
| `POST` | `/security/roles` | `{user, role}` | Assign role to user |
| `DELETE` | `/security/roles/{user}` | — | Revoke user role |
| `GET` | `/security/roles` | — | List all user roles |

**Roles:** `admin` · `developer` · `viewer`

| Role | Permissions |
|---|---|
| `admin` | deploy, rollback, read, write, delete, manage_users, manage_secrets |
| `developer` | deploy, read, write |
| `viewer` | read |

### Memory

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/memory/incidents` | `{id, type, source, payload}` | Store incident in ChromaDB |
| `GET` | `/memory/incidents` | `query, n` | Search similar past incidents |

### Jira Webhook

| Method | Path | Description |
|---|---|---|
| `POST` | `/jira/webhook` | Receives Jira issue-created events → auto-creates GitHub PR |

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

## Multi-Agent Pipeline (v2)

**`POST /v2/incident/run`** — the flagship endpoint.

```
Input → collect_context → PlannerAgent → DecisionAgent
      → Executor (policy-gated) → Validator
      → MemoryAgent → Final Response
```

| Step | Agent / Node | What happens |
|---|---|---|
| **1. Context** | `AWSAgent` `K8sAgent` `GitHubAgent` | Parallel data collection + ChromaDB similar incident retrieval |
| **2. Plan** | `PlannerAgent` + LLM | Structured JSON plan: actions, confidence, risk, root_cause |
| **3. Decide** | `DecisionAgent` | Risk score + approval gate (no LLM call) |
| **4. Execute** | `Executor` + `PolicyEngine` | Each action checked against rules.json before running |
| **5. Validate** | `Validator` | Re-checks K8s health; triggers retry (up to 3×) or escalates |
| **6. Memory** | `MemoryAgent` | Stores outcome + actions to ChromaDB |

### Policy Engine

Actions are blocked before execution by `app/policies/rules.json`:

```json
{
  "blocked_actions": ["delete_cluster", "drop_database", "terminate_all_instances"],
  "guardrails": {
    "max_replicas": 20,
    "restricted_namespaces": ["kube-system", "kube-public"]
  }
}
```

Add new rules to `rules.json` without changing any Python code.

### Conditional branching

- **`requires_human_approval=true`** (high risk / low confidence / `auto_remediate=false`) → pipeline ends at `awaiting_approval`, no actions executed
- **Validation failed + retries < 3** → re-runs `execute` node
- **Validation failed + retries exhausted** → `escalate` node notifies Slack + OpsGenie

---

## Original Incident Pipeline (v1)

**`POST /incident/run`** — backwards-compatible, unchanged.

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

**Response:** `go_no_go` (`go` / `go_with_caution` / `no_go`), `risk_score`, `concerns[]`, `checklist[]`, `safe_window`

---

## Jira Webhook → Auto PR

**`POST /jira/webhook`** — register this URL in Jira to auto-create GitHub PRs.

**Jira setup:** Project Settings → Webhooks → URL: `https://your-platform/jira/webhook` → Event: Issue Created

Triggers when: Issue type is **Change Request**, **Task**, or **Story** — or issue has label **`auto-pr`**

**Flow:**
1. Claude reads the Jira ticket description
2. Generates PR plan with title, body, and best-effort file patches
3. Creates branch `jira/<ticket-key>-<slug>` and opens a PR
4. Posts the PR link as a comment on the Jira ticket

---

## AI PR Review

**`POST /github/review-pr`**

```bash
curl -X POST http://127.0.0.1:8000/github/review-pr \
  -H "Content-Type: application/json" \
  -d '{"pr_number": 42, "post_comment": true}'
```

Claude analyses the PR diff for security vulnerabilities, infra changes, and code quality. Set `post_comment: true` to post the review directly on the PR.

---

## Multi-LLM Support

The platform automatically selects the best available LLM:

```
Claude (ANTHROPIC_API_KEY) → OpenAI (OPENAI_API_KEY) → Groq (GROQ_API_KEY) → Ollama (local)
```

Override per-request by setting `LLM_PROVIDER` in `.env`. The factory is in `app/llm/factory.py` — add new providers by implementing `BaseLLM` in `app/llm/base.py`.

---

## Continuous Monitoring

Enable background anomaly detection:

```env
ENABLE_MONITOR_LOOP=true
MONITOR_INTERVAL_SECONDS=60
AUTO_REMEDIATE_ON_MONITOR=false   # alert-only until you're confident
```

The monitor (`app/monitoring/loop.py`) polls K8s for crash-looping pods and unhealthy states. When anomalies are found it triggers the v2 pipeline with `auto_remediate=AUTO_REMEDIATE_ON_MONITOR`.

---

## RBAC Usage

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

**Protected endpoints** require `X-User` header:

| Endpoint | Required permission |
|---|---|
| `POST /k8s/restart` | `deploy` |
| `POST /k8s/scale` | `deploy` |
| `POST /deploy/assess` | `deploy` |
| `POST /incident/run` (auto_remediate=true) | `deploy` |
| `POST /v2/incident/run` (auto_remediate=true) | `deploy` |

Persist users via file — set `RBAC_CONFIG_PATH=/path/to/roles.json`:

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

---

## Docker

```bash
docker compose up --build
```

The `docker-compose.yml` mounts the project directory, loads `.env`, and restarts on failure.
