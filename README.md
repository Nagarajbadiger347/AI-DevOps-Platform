# AI DevOps Intelligence Platform

Autonomous DevOps management powered by a **multi-agent AI system** — built by **Nagaraj**.

One platform to detect incidents, analyse root cause, plan and safely execute remediation, assess deployments, run interactive Slack war rooms, and close the loop back to Jira and GitHub — automatically.

---

## What it does

| Capability | Description |
|---|---|
| 🤖 **Multi-Agent Incident Pipeline** | LangGraph-orchestrated agents collect context, plan remediation, score risk, execute safely (with policy guardrails), validate outcome, and store to memory |
| 🔍 **Pre-deployment Assessment** | Before any deploy, Claude assesses cluster state, active alarms, and past incidents → go / no-go decision with checklist |
| 🎫 **Jira → Auto PR** | When a Jira change-request ticket is created, Claude interprets it and opens a GitHub PR with file patches |
| ☁️ **AWS Observability** | Read-only collection across EC2, ECS, Lambda, RDS, ALB, CloudWatch Logs/Metrics/Alarms, CloudTrail, S3, SQS, DynamoDB, Route53, SNS |
| ☸️ **Kubernetes Operations** | Health checks + rolling restarts + scale deployments + fetch pod logs + unhealthy pod detection |
| 📈 **Predictive Scaling** | Analyse CloudWatch metric trends and predict if scaling is needed before a breach occurs |
| 🔎 **AI PR Review** | Claude reviews GitHub PRs for security issues, infra concerns, and code quality |
| 🔐 **JWT Auth + RBAC** | JWT-based authentication with role-based access control (admin / developer / viewer) enforced on every endpoint |
| 🧠 **ChromaDB Memory** | All incidents stored in vector DB; similar past incidents feed into future planning decisions |
| 🔁 **Continuous Monitoring** | Background loop polls K8s/AWS for anomalies and auto-triggers the pipeline |
| 🔀 **Multi-LLM Support** | Claude (primary) → OpenAI (fallback) → Groq/Llama → Ollama (local) — automatic fallback chain |
| 💬 **Slack War Room Bot** | Dedicated incident channel auto-created with full AI analysis; bot answers live questions (PRs, logs, K8s state, AWS alarms) in thread |
| 🌐 **Nginx + TLS** | Production-grade reverse proxy with rate limiting, HTTPS redirect, security headers, WebSocket support |

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

  Browser / API Client
        │
        ▼
  Nginx (TLS + Rate Limiting)
        │
        ▼
  FastAPI App (4 uvicorn workers)
        │
        ├── JWT Auth middleware (Bearer token / X-User dev mode)
        ├── Redis (rate limiting + response cache)
        └── ChromaDB (incident memory)
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
│   ├── main.py           # FastAPI server — all REST & WebSocket endpoints
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
│   └── incident_pipeline.py       # v1 pipeline (backwards-compatible)
│
├── llm/
│   ├── base.py           # BaseLLM ABC + LLMResponse dataclass
│   ├── claude.py         # ClaudeProvider + all AI analysis functions
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
├── core/
│   ├── auth.py            # JWT authentication + require_admin/developer/viewer deps
│   ├── config.py          # Centralised pydantic-settings configuration
│   ├── ratelimit.py       # Rate limiter (Redis-backed with in-memory fallback)
│   ├── audit.py           # Audit log for all mutating actions
│   └── logging.py         # Structured JSON logger + correlation IDs
│
├── integrations/
│   ├── aws_ops.py              # AWS observability + operations
│   ├── github.py               # Commits, PRs, diffs, reviews, incident PRs
│   ├── jira.py                 # Jira incident creation, comments, issue fetch
│   ├── slack.py                # Slack messaging, war room channels, rich Block Kit summaries
│   ├── slack_bot.py            # War room bot — answers questions in Slack threads
│   ├── opsgenie.py             # OpsGenie on-call notification
│   ├── k8s_ops.py              # K8s restart / scale / logs / node ops
│   ├── grafana.py              # Grafana alerts, annotations, datasources
│   ├── gitlab_ops.py           # GitLab pipelines/deployments
│   └── universal_collector.py  # Multi-integration parallel aggregator
│
├── plugins/
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
└── correlation/
    └── engine.py          # Event correlation logic

nginx/
└── nginx.conf             # Reverse proxy: TLS, rate limiting, security headers

Dockerfile                 # Multi-stage build — builder + production (non-root user)
docker-compose.yml         # App + Nginx + Redis with health checks and resource limits
tests/test_main.py
requirements.txt
```

---

## Setup

### Requirements

- Python 3.9+ (3.11 recommended)
- Docker & Docker Compose

### Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials
uvicorn app.orchestrator.main:app --reload --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000

### Production (Docker)

```bash
cp .env.example .env          # fill in all credentials + set AUTH_ENABLED=true + JWT_SECRET_KEY
docker compose up --build -d
```

The stack starts three services:
- **nginx** — TLS termination, rate limiting, security headers on port 443 (HTTP redirects to HTTPS)
- **app** — FastAPI with 4 uvicorn workers, resource limits (2 CPU / 2 GB RAM)
- **redis** — Rate limiting + response cache, 256 MB LRU eviction

> **TLS certs:** Place `server.crt` and `server.key` in `nginx/certs/`. For production use Let's Encrypt or ACM. For local testing: `openssl req -x509 -newkey rsa:4096 -keyout nginx/certs/server.key -out nginx/certs/server.crt -days 365 -nodes -subj "/CN=localhost"`

---

## Authentication

The platform uses JWT-based authentication.

### Get a token

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=nagaraj&password=your-password"
# → {"access_token": "eyJ...", "token_type": "bearer"}
```

### Use the token

```bash
curl http://localhost:8000/aws/ec2/instances \
  -H "Authorization: Bearer eyJ..."
```

### Dev mode (no token required)

Set `AUTH_ENABLED=false` in `.env`. The platform trusts the `X-User` header and defaults to the `developer` role for unknown users.

```bash
curl http://localhost:8000/aws/ec2/instances -H "X-User: nagaraj"
```

### Roles

| Role | Permissions | Default user |
|---|---|---|
| `admin` | All permissions — deploy, manage users, manage secrets | `nagaraj` |
| `developer` | deploy, read, write | Any unregistered user (dev mode) |
| `viewer` | read only | — |

Assign roles via API:
```bash
curl -X POST http://localhost:8000/security/roles/assign \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"user": "alice", "role": "developer"}'
```

Persist roles via file — set `RBAC_CONFIG_PATH=/path/to/roles.json`:
```json
{"alice": "developer", "bob": "viewer", "charlie": "admin"}
```

---

## Environment Variables

### Auth

| Variable | Default | Description |
|---|---|---|
| `AUTH_ENABLED` | `true` | Set `false` to use dev mode (X-User header, no JWT) |
| `JWT_SECRET_KEY` | `change-me-...` | Random 32-byte hex: `openssl rand -hex 32` |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINS` | `480` | Token lifetime in minutes (8 hours) |

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
| `GITHUB_TOKEN` | GitHub features | Personal access token (repo + PR scope) |
| `GITHUB_REPO` | GitHub features | `owner/repo` format |
| `GITHUB_WEBHOOK_SECRET` | GitHub webhooks | HMAC secret for signature verification |
| `SLACK_BOT_TOKEN` | Slack | Bot token (`xoxb-...`) — channels:write, chat:write |
| `SLACK_CHANNEL` | Slack | Default channel (default: `#general`) |
| `SLACK_SIGNING_SECRET` | Slack war room bot | From Slack app → Basic Information |
| `SLACK_BOT_USER_ID` | Slack war room bot | Bot's Slack user ID (prevents self-replies) |
| `JIRA_URL` | Jira | e.g. `https://yourorg.atlassian.net` |
| `JIRA_USER` | Jira | User email |
| `JIRA_TOKEN` | Jira | API token |
| `JIRA_PROJECT` | Jira | Project key |
| `OPSGENIE_API_KEY` | OpsGenie | API key |
| `GRAFANA_URL` | Grafana | e.g. `http://grafana:3000` |
| `GRAFANA_TOKEN` | Grafana | Service account token |
| `AWS_REGION` | AWS | Region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | AWS | Access key (or use IAM role) |
| `AWS_SECRET_ACCESS_KEY` | AWS | Secret key (or use IAM role) |
| `K8S_IN_CLUSTER` | K8s (in-pod) | Set `true` when running inside a pod |
| `KUBECONFIG` | K8s (local) | Path to kubeconfig (default: `~/.kube/config`) |
| `REDIS_URL` | Rate limiting | Redis connection URL (default: `redis://localhost:6379/0`) |
| `RBAC_CONFIG_PATH` | Optional | Path to JSON file with user→role mappings |
| `CORS_ORIGINS` | Optional | Comma-separated allowed CORS origins |

All integrations degrade gracefully — missing credentials return a structured error rather than crashing.

---

## Slack War Room Bot

When a war room is created (`POST /warroom/create`), the platform:

1. Creates a dedicated `#inc-<incident-id>` Slack channel
2. Posts a rich summary with: severity, AI confidence score, root cause, findings, infrastructure snapshot (firing alarms, unhealthy pods, last commit), PR links, and action plan
3. Activates a **bot that answers questions from engineers in the thread**

### Bot capabilities

Ask anything in the channel and the bot replies in the thread:

| Question | What it does |
|---|---|
| "which PR raised this?" | Fetches recent GitHub PRs and commits with direct links |
| "check last 30 min grafana alerts" | Queries firing Grafana alerts and annotations |
| "why are pods crashing?" | Shows unhealthy pods + deployment replica status |
| "any AWS alarms firing?" | Lists CloudWatch alarms in ALARM state + unhealthy EC2 |
| "what should we do next?" | AI synthesises current state into actionable next steps |

### Enabling the bot

1. Add to `.env`:
   ```env
   SLACK_SIGNING_SECRET=your-slack-signing-secret
   SLACK_BOT_USER_ID=U0XXXXXXXXX
   ```

2. In Slack app settings:
   - **Event Subscriptions** → Request URL: `https://your-domain/webhooks/slack`
   - Subscribe to bot events: `message.channels`, `message.groups`, `app_mention`
   - **OAuth Scopes**: `channels:read`, `channels:write`, `chat:write`, `groups:write`, `groups:read`

---

## API Reference

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/token` | Get JWT — form body: `username` + `password` |
| `GET` | `/auth/me` | Current user identity and role |

### General

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | None | Dashboard UI |
| `GET` | `/health` | None | Health status |
| `GET` | `/health/full` | None | Full integration health check |
| `GET` | `/health/integrations` | viewer | Which integrations are configured |
| `GET` | `/docs` | None | Swagger UI |

### Chat

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | None | Conversational DevOps AI with action confirmation flow |
| `GET` | `/chat/action_count` | viewer | Number of actions executed this session |

### Incident Pipelines

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/incident/run` | developer | v1 monolithic pipeline (backwards-compatible) |
| `POST` | `/incidents/run` | developer | Alias for `/incident/run` |
| `POST` | `/incidents/run/async` | developer | Fire-and-forget — returns job ID immediately |
| `POST` | `/v2/incident/run` | developer | v2 LangGraph multi-agent pipeline with policy engine |

**Request body (both versions):**
```json
{
  "incident_id":    "INC-001",
  "description":   "API pods crash-looping in prod",
  "severity":      "critical",
  "auto_remediate": false,
  "aws":           {"resource_type": "ecs", "resource_id": "prod-cluster"},
  "k8s":           {"namespace": "production"},
  "hours":          2
}
```

### War Room

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/warroom/create` | developer | Create Slack war room with full AI analysis + activate bot |

**Request body:**
```json
{
  "incident_id":   "INC-001",
  "description":   "API latency spike in production",
  "severity":      "high",
  "post_to_slack": true
}
```

**What gets posted to Slack:** severity badge, AI confidence %, root cause, findings list, infrastructure snapshot (firing alarms, unhealthy pods, last commit), PR links, full action plan.

### Kubernetes

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/k8s/health` | viewer | Cluster summary |
| `GET` | `/k8s/nodes` | viewer | Per-node ready status |
| `GET` | `/k8s/pods` | viewer | Pod status (`?namespace=`) |
| `GET` | `/k8s/deployments` | viewer | Deployment rollout status |
| `POST` | `/k8s/diagnose` | developer | AI-powered K8s diagnosis |
| `POST` | `/k8s/restart` | developer | Rolling restart |
| `POST` | `/k8s/scale` | developer | Scale replicas |
| `GET` | `/k8s/logs` | viewer | Fetch pod logs |

### AWS Observability (all read-only)

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/check/aws` | viewer | AWS health summary |
| `GET` | `/aws/ec2/instances` | viewer | List EC2 instances |
| `GET` | `/aws/ec2/status` | viewer | EC2 status checks |
| `GET` | `/aws/ec2/console` | viewer | Serial console output |
| `GET` | `/aws/logs/groups` | viewer | List CloudWatch log groups |
| `GET` | `/aws/logs/recent` | viewer | Recent log events |
| `GET` | `/aws/logs/search` | viewer | Search logs by pattern |
| `GET` | `/aws/cloudwatch/alarms` | viewer | CloudWatch alarms |
| `GET` | `/aws/cloudwatch/logs` | viewer | Recent logs (simplified) |
| `POST` | `/aws/cloudwatch/metrics` | viewer | Fetch metric series |
| `GET` | `/aws/ecs/services` | viewer | ECS service counts |
| `GET` | `/aws/ecs/stopped-tasks` | viewer | Stopped ECS task reasons |
| `GET` | `/aws/lambda/functions` | viewer | List Lambda functions |
| `GET` | `/aws/lambda/errors` | viewer | Lambda error metrics |
| `GET` | `/aws/rds/instances` | viewer | RDS instance list |
| `GET` | `/aws/rds/events` | viewer | RDS events |
| `GET` | `/aws/elb/target-health` | viewer | ALB target health |
| `GET` | `/aws/cloudtrail/events` | viewer | Recent CloudTrail API events |
| `GET` | `/aws/s3/buckets` | viewer | S3 bucket list |
| `GET` | `/aws/sqs/queues` | viewer | SQS queue list |
| `GET` | `/aws/dynamodb/tables` | viewer | DynamoDB table list |
| `GET` | `/aws/route53/healthchecks` | viewer | Route53 health checks |
| `GET` | `/aws/sns/topics` | viewer | SNS topic list |
| `GET` | `/aws/context` | viewer | Full AWS observability snapshot |
| `GET` | `/aws/cost/summary` | viewer | Resource inventory cost summary |
| `POST` | `/aws/diagnose` | developer | AI root cause from live AWS data |
| `POST` | `/aws/predict-scaling` | viewer | Predict if scaling needed |
| `POST` | `/aws/synthesize` | developer | AI synthesis of full AWS state |
| `POST` | `/aws/assess-deployment` | developer | Pre-deploy assessment alias |

### GitHub

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/github/repos` | viewer | List repositories |
| `GET` | `/github/profile` | viewer | Account summary — repos, stars, languages |
| `GET` | `/github/commits` | viewer | Recent commits (`?hours=24&repo=`) |
| `GET` | `/github/prs` | viewer | Recent merged PRs (`?hours=48&repo=`) |
| `GET` | `/github/pr/{pr_number}/review` | developer | AI review of a specific PR |
| `POST` | `/github/issue` | developer | Create a GitHub issue |
| `POST` | `/github/review-pr` | developer | AI code review with optional PR comment |

### Grafana

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/grafana/alerts` | viewer | Firing Grafana alerts |
| `GET` | `/grafana/dashboards` | viewer | Grafana datasources |

### Deployment & Code Review

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/deploy/assess` | developer | Pre-deploy go/no-go assessment |
| `POST` | `/deploy/jira-to-pr` | developer | Convert Jira issue to GitHub PR |

### Incident Management

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/incident/war-room` | developer | Basic Slack war room announcement |
| `POST` | `/incident/jira` | developer | Create Jira incident |
| `POST` | `/incident/opsgenie` | developer | Notify OpsGenie on-call |
| `POST` | `/incident/github/issue` | developer | Create GitHub issue |
| `POST` | `/incident/github/pr` | developer | Create GitHub PR |
| `POST` | `/jira/incident` | developer | Create Jira incident (alias) |

### Secrets & Configuration

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/secrets/status` | admin | Which env vars are configured (boolean only, no values) |
| `POST` | `/secrets` | admin | Write secrets to `.env` file |

### Security / RBAC

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/security/check` | viewer | Check if user can perform action |
| `POST` | `/security/roles` | developer | Assign role (legacy endpoint) |
| `DELETE` | `/security/roles/{user}` | admin | Revoke user role |
| `GET` | `/security/roles` | admin | List all user roles and permissions |
| `POST` | `/security/roles/assign` | admin | Assign role to user |

### Memory

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/memory/incidents` | developer | Store incident in ChromaDB |
| `GET` | `/memory/incidents` | viewer | Search similar past incidents |

### Audit & Rate Limiting

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/audit/log` | viewer | Recent audit log entries |
| `GET` | `/rate-limit/status` | viewer | Current rate limit usage |

### Webhooks (no auth — verified by signature)

| Method | Path | Description |
|---|---|---|
| `POST` | `/webhooks/github` | GitHub push/PR events → auto-trigger pipeline |
| `POST` | `/webhooks/slack` | Slack Events API — war room bot messages |
| `POST` | `/webhooks/pagerduty` | PagerDuty incident trigger → auto-pipeline |
| `POST` | `/jira/webhook` | Jira issue-created → auto-create GitHub PR |

### WebSocket

```
WS /ws
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

**v2 response includes:**
- `plan` — structured JSON plan (`actions`, `confidence`, `risk`, `root_cause`)
- `executed_actions` — each action's result
- `blocked_actions` — actions blocked by policy (with reason)
- `validation_passed` — post-execution health check result
- `requires_human_approval` — whether approval gate was triggered
- `status` — `completed` | `escalated` | `awaiting_approval` | `failed`

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

## Jira Webhook → Auto PR

Register `https://your-platform/jira/webhook` in Jira:
**Project Settings → Webhooks → Event: Issue Created**

Triggers when issue type is **Change Request**, **Task**, or **Story** — or has label `auto-pr`.

**Flow:**
1. Claude reads the Jira ticket description
2. Generates PR plan with title, body, and best-effort file patches
3. Creates branch `jira/<ticket-key>-<slug>` and opens a PR
4. Posts the PR link as a comment on the Jira ticket

---

## GitHub Webhook

Register `https://your-platform/webhooks/github` in GitHub:
**Settings → Webhooks → Content type: application/json**

Set `GITHUB_WEBHOOK_SECRET` in `.env` and use the same value in GitHub. The platform verifies HMAC-SHA256 signatures on every request.

Triggers on: `push` to `main`/`master`, PR `opened`/`synchronize` events.

---

## Nginx Rate Limits

| Zone | Limit | Applied to |
|---|---|---|
| `api` | 60 req/min | All endpoints |
| `chat` | 20 req/min | `/chat` |
| `auth` | 10 req/min | `/auth/*` |
| `webhooks` | 30 req/min | `/webhooks/*` |

---

## Running Tests

```bash
pytest -q                         # all tests
pytest -q tests/test_main.py     # API tests only
python test_websocket.py          # manual WebSocket test (needs running server)
```

---

## Multi-LLM Support

The platform automatically selects the best available LLM:

```
Claude (ANTHROPIC_API_KEY) → OpenAI (OPENAI_API_KEY) → Groq (GROQ_API_KEY) → Ollama (local)
```

Override per-request with `provider` field in chat payload. The factory is in `app/llm/factory.py` — add new providers by implementing `BaseLLM` in `app/llm/base.py`.

---

## Continuous Monitoring

Enable background anomaly detection:

```env
ENABLE_MONITOR_LOOP=true
MONITOR_INTERVAL_SECONDS=60
AUTO_REMEDIATE_ON_MONITOR=false   # alert-only until you're confident
```

The monitor (`app/monitoring/loop.py`) polls K8s for crash-looping pods and unhealthy states. When anomalies are found it triggers the v2 pipeline with `auto_remediate=AUTO_REMEDIATE_ON_MONITOR`.
