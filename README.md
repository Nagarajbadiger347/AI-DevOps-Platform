# NexusOps — AI DevOps Intelligence Platform

Autonomous DevOps management powered by a **multi-agent AI system**.

One platform to detect incidents, analyse root cause, plan and safely execute remediation, assess deployments, run interactive Slack war rooms, analyse costs, and close the loop back to Jira and GitHub — automatically.

---

## What It Does

| Capability | Description |
|---|---|
| 🤖 **Multi-Agent Incident Pipeline** | LangGraph-orchestrated agents collect context, plan remediation, score risk, execute safely with policy guardrails, validate outcome, and store to memory |
| 🔍 **Pre-Deployment Assessment** | Before any deploy, the AI assesses cluster state, active alarms, and past incidents → go / no-go decision |
| 🎫 **Jira → Auto PR** | When a Jira change-request ticket is created, the AI reads it and opens a GitHub PR with file patches |
| ☁️ **AWS Observability** | Read-only collection across EC2, ECS, Lambda, RDS, ALB, CloudWatch, CloudTrail, S3, SQS, DynamoDB, Route53, SNS |
| ☸️ **Kubernetes Operations** | Health checks, rolling restarts, scale deployments, pod logs, unhealthy pod detection |
| 💰 **Smart Cost Analysis** | Live AWS spend (Cost Explorer), resource-level cost breakdown, multi-account AWS Organizations view, on-demand price estimation, Terraform plan cost analysis |
| 📈 **Predictive Scaling** | Analyse CloudWatch metric trends and predict scaling needs before a breach occurs |
| 🔎 **AI PR Review** | Reviews GitHub PRs for security issues, infra concerns, and code quality |
| 🔐 **JWT Auth + RBAC** | JWT-based authentication with role-based access control (admin / developer / viewer) enforced on every endpoint |
| 🧠 **ChromaDB Memory** | All incidents stored in vector DB; similar past incidents feed future planning decisions |
| 🔁 **Continuous Monitoring** | Background loop polls K8s/AWS for anomalies and auto-triggers the pipeline |
| 🔀 **Multi-LLM Support** | Claude → Groq → Ollama — automatic provider detection and fallback chain with live status indicator |
| 💬 **Slack War Room Bot** | Dedicated incident channel auto-created with full AI analysis; bot answers live questions in thread |
| 🏥 **Post-Mortem Reports** | AI-generated company-grade post-mortem documents enriched from vector memory |
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
                         │       │         (Claude / Groq / Ollama)│
                         │       ▼                                 │
                         │  DecisionAgent (risk score + approval)  │
                         │       │                                 │
                         │       ├── auto_remediate=true ──▶ Executor
                         │       └── high risk / low confidence ──▶ awaiting_approval
                         │                          │              │
                         │                     PolicyEngine        │
                         │                     ActionRegistry      │
                         │                          │              │
                         │                       Validator         │
                         │                     (re-check health)   │
                         │                       /       \         │
                         │               passed           failed   │
                         │                 │              retry / escalate
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
        ├── JWT Auth middleware
        ├── Redis (rate limiting + response cache)
        └── ChromaDB (incident memory)
```

### Core Design Principles

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
│   ├── main.py           # FastAPI server — all REST, WebSocket endpoints + full dashboard UI
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── state.py          # PipelineState TypedDict shared across all nodes
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
├── chat/
│   ├── intelligence.py    # AI chat with tool use — AWS, GitHub, K8s, Jira, Slack routing
│   └── memory.py          # Per-session conversation memory
│
├── llm/
│   ├── base.py            # BaseLLM ABC + LLMResponse dataclass
│   ├── claude.py          # ClaudeProvider (primary)
│   ├── openai.py          # OpenAIProvider (GPT-4o fallback)
│   └── factory.py         # LLMFactory — automatic provider selection + fallback chain
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
├── cost/
│   ├── analyzer.py        # AWS Cost Explorer integration — dashboard, accounts, trends
│   └── pricing.py         # On-demand price estimation + Terraform plan cost analysis
│
├── incident/
│   ├── approval.py               # Human-in-the-loop approval workflow
│   ├── post_mortem.py            # AI post-mortem report generation
│   └── war_room_intelligence.py  # LLM war room with session memory + JSON persistence
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
│   ├── slack.py                # Slack messaging, war room channels, Block Kit summaries
│   ├── slack_bot.py            # War room bot — answers questions in Slack threads
│   ├── opsgenie.py             # OpsGenie on-call notification
│   ├── k8s_ops.py              # K8s restart / scale / logs / node ops
│   ├── grafana.py              # Grafana alerts, annotations, datasources
│   ├── gitlab_ops.py           # GitLab pipelines/deployments
│   ├── webhooks.py             # Inbound webhook handlers (GitHub, Slack, PagerDuty, Jira)
│   └── universal_collector.py  # Multi-integration parallel aggregator
│
├── memory/
│   └── vector_db.py       # ChromaDB incident storage + similarity search
│
├── security/
│   ├── rbac.py            # Role-based access control with JSON file persistence
│   ├── users.py           # User store — bcrypt password hashing, bootstrap admin
│   └── invite.py          # Invite token management
│
├── plugins/
│   ├── aws_checker.py
│   ├── k8s_checker.py
│   └── linux_checker.py
│
└── correlation/
    └── engine.py          # Event correlation logic

nginx/
└── nginx.conf             # Reverse proxy: TLS, rate limiting, security headers

Dockerfile                 # Multi-stage build — builder + production (non-root user)
docker-compose.yml         # App + Nginx + Redis with health checks and resource limits
requirements.txt
```

---

## Setup

### Requirements

- Python 3.9+ (3.11 recommended)
- Docker & Docker Compose (for production)

### Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials
uvicorn app.orchestrator.main:app --reload --host 127.0.0.1 --port 8000
```

Open: **http://127.0.0.1:8000**

On first run, if no users exist, a temporary admin account is created and the credentials are printed to stdout. Change the password immediately, or set `ADMIN_PASSWORD` in `.env`.

### Production (Docker)

```bash
cp .env.example .env          # fill in all credentials
# Set AUTH_ENABLED=true and a strong JWT_SECRET_KEY
docker compose up --build -d
```

Three services start:

| Service | Description |
|---|---|
| **nginx** | TLS termination, rate limiting, security headers on port 443 (HTTP redirects to HTTPS) |
| **app** | FastAPI with 4 uvicorn workers, resource limits (2 CPU / 2 GB RAM) |
| **redis** | Rate limiting + response cache, 256 MB LRU eviction |

> **TLS certificates:** Place `server.crt` and `server.key` in `nginx/certs/`. For local testing:
> ```bash
> openssl req -x509 -newkey rsa:4096 -keyout nginx/certs/server.key \
>   -out nginx/certs/server.crt -days 365 -nodes -subj "/CN=localhost"
> ```

---

## Authentication

The platform uses JWT-based authentication with role-based access control.

### Get a Token

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=your-password"
# → {"access_token": "eyJ...", "token_type": "bearer"}
```

### Use the Token

```bash
curl http://localhost:8000/aws/ec2/instances \
  -H "Authorization: Bearer eyJ..."
```

### Dev Mode (no token required)

Set `AUTH_ENABLED=false` in `.env`. The platform trusts the `X-User` header and defaults to `developer` role for unknown users.

```bash
curl http://localhost:8000/aws/ec2/instances -H "X-User: alice"
```

### Roles

| Role | Permissions |
|---|---|
| `admin` | Full access — deploy, manage users, manage secrets, assign roles |
| `developer` | Deploy, read, write — all operational endpoints |
| `viewer` | Read-only — all observability endpoints |

**Assign a role:**
```bash
curl -X POST http://localhost:8000/security/roles/assign \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"user": "alice", "role": "developer"}'
```

---

## Environment Variables

### Auth & Security

| Variable | Default | Description |
|---|---|---|
| `AUTH_ENABLED` | `true` | Set `false` to use dev mode (X-User header, no JWT) |
| `JWT_SECRET_KEY` | `change-me-...` | Token signing key — `openssl rand -hex 32` |
| `APP_SECRET_KEY` | — | Password hashing key — separate from JWT key so rotating JWT doesn't invalidate passwords |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `JWT_EXPIRE_MINS` | `480` | Token lifetime in minutes (8 hours) |
| `ADMIN_USERNAME` | `admin` | Bootstrap admin username |
| `ADMIN_PASSWORD` | — | Bootstrap admin password (generated and printed if not set) |

### LLM Providers

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `claude` | Preferred provider: `claude` \| `openai` \| `groq` \| `ollama` |
| `ANTHROPIC_API_KEY` | — | Claude API key — primary provider |
| `OPENAI_API_KEY` | — | OpenAI API key — fallback |
| `GROQ_API_KEY` | — | Groq API key — secondary fallback (Llama 3.3-70B, 100k tokens/day free tier) |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama — final fallback, no key required |

> **Note:** The dashboard shows a warning banner when only Groq is active, since the free tier has a 100k tokens/day limit. Add an `ANTHROPIC_API_KEY` to switch to Claude automatically.

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

### AWS

| Variable | Required for | Description |
|---|---|---|
| `AWS_REGION` | AWS | Region (default: `us-east-1`) |
| `AWS_ACCESS_KEY_ID` | AWS | Access key (or use IAM instance role) |
| `AWS_SECRET_ACCESS_KEY` | AWS | Secret key |
| `AWS_SESSION_TOKEN` | AWS (STS/assumed role) | Session token for temporary credentials |

> **Cost Explorer & Organizations:** For the multi-account Organizations view, the credentials above must belong to the management account and have `ce:GetCostAndUsage` + `organizations:ListAccounts` IAM permissions. Cost Explorer must be enabled at: AWS Console → Billing → Cost Explorer → Enable.

### Integrations

| Variable | Required for | Description |
|---|---|---|
| `GITHUB_TOKEN` | GitHub | Personal access token (`repo` + `pull_requests` scope) |
| `GITHUB_REPO` | GitHub | Default repository in `owner/repo` format |
| `GITHUB_WEBHOOK_SECRET` | GitHub webhooks | HMAC secret for signature verification |
| `SLACK_BOT_TOKEN` | Slack | Bot token (`xoxb-...`) |
| `SLACK_CHANNEL` | Slack | Default channel (default: `#general`) |
| `SLACK_SIGNING_SECRET` | Slack war room bot | From Slack app → Basic Information |
| `SLACK_BOT_USER_ID` | Slack war room bot | Bot's user ID (prevents self-replies) |
| `JIRA_URL` | Jira | e.g. `https://yourorg.atlassian.net` |
| `JIRA_USER` | Jira | User email |
| `JIRA_TOKEN` | Jira | API token |
| `JIRA_PROJECT` | Jira | Project key (default: `DEVOPS`) |
| `OPSGENIE_API_KEY` | OpsGenie | API key |
| `GRAFANA_URL` | Grafana | e.g. `http://grafana:3000` |
| `GRAFANA_TOKEN` | Grafana | Service account token |
| `K8S_IN_CLUSTER` | K8s (in-pod) | Set `true` when running inside a Kubernetes pod |
| `KUBECONFIG` | K8s (local) | Path to kubeconfig (default: `~/.kube/config`) |
| `REDIS_URL` | Rate limiting | Redis connection URL (default: `redis://localhost:6379/0`) |
| `CORS_ORIGINS` | Optional | Comma-separated allowed CORS origins |

All integrations degrade gracefully — missing credentials return a structured error rather than crashing the platform.

---

## Cost Analysis

The Cost Analysis tab has five views:

| Tab | Description |
|---|---|
| **Overview** | MTD spend, last month total, forecast, MoM trend, top services bar chart, 6-month trend |
| **By Resource** | Searchable table of every EC2/RDS/Lambda/ECS resource with estimated monthly and annual cost |
| **By Account** | Cost Explorer breakdown by linked AWS account with spend bars and % of total |
| **Organizations** | Per-account spend list + proportional bar chart with AWS Organizations account names |
| **Estimate** | Quick Estimate (plain English), Action Impact analysis, Terraform Plan cost estimation |

Data sources:
- **Live spend** — AWS Cost Explorer API (`ce:GetCostAndUsage`)
- **Resource inventory** — direct EC2/RDS/Lambda/ECS API calls with on-demand pricing estimates
- **Account names** — AWS Organizations `list_accounts` (management account only)
- **Price estimation** — AWS Pricing API + fallback reference rates

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
| **4. Execute** | `Executor` + `PolicyEngine` | Each action checked against `rules.json` before running |
| **5. Validate** | `Validator` | Re-checks K8s health; triggers retry (up to 3×) or escalates |
| **6. Memory** | `MemoryAgent` | Stores outcome + actions to ChromaDB |

**Response fields:**

| Field | Description |
|---|---|
| `plan` | Structured JSON plan with actions, confidence, risk, root_cause |
| `executed_actions` | Each action's result |
| `blocked_actions` | Actions blocked by policy with reason |
| `validation_passed` | Post-execution health check result |
| `requires_human_approval` | Whether the approval gate was triggered |
| `status` | `completed` \| `escalated` \| `awaiting_approval` \| `failed` |

**Request body:**
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

### Policy Engine

Actions are evaluated against `app/policies/rules.json` before execution. No Python changes needed — just edit the JSON:

```json
{
  "blocked_actions": ["delete_cluster", "drop_database", "terminate_all_instances"],
  "guardrails": {
    "max_replicas": 20,
    "restricted_namespaces": ["kube-system", "kube-public"]
  }
}
```

### Conditional Branching

- **High risk / low confidence / `auto_remediate=false`** → pipeline ends at `awaiting_approval`, no actions executed
- **Validation failed + retries < 3** → re-runs `execute` node
- **Validation failed + retries exhausted** → `escalate` node notifies Slack + OpsGenie

---

## Slack War Room Bot

When a war room is created (`POST /warroom/create`), the platform:

1. Creates a dedicated `#inc-<incident-id>` Slack channel
2. Posts a rich summary with severity, AI confidence, root cause, findings, infrastructure snapshot, PR links, and action plan
3. Activates a **bot that answers questions from engineers in the thread**
4. War room state is persisted to disk (`data/war_rooms.json`) and survives restarts

### Bot Capabilities

| Question | What it does |
|---|---|
| "which PR raised this?" | Fetches recent GitHub PRs and commits with direct links |
| "check last 30 min grafana alerts" | Queries firing Grafana alerts and annotations |
| "why are pods crashing?" | Shows unhealthy pods + deployment replica status |
| "any AWS alarms firing?" | Lists CloudWatch alarms in ALARM state |
| "what should we do next?" | AI synthesises current state into actionable next steps |

### Enabling the Bot

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

## Jira Webhook → Auto PR

Register `https://your-platform/jira/webhook` in Jira:
**Project Settings → Webhooks → Event: Issue Created**

Triggers when issue type is **Change Request**, **Task**, or **Story** — or has label `auto-pr`.

**Flow:**
1. AI reads the Jira ticket description
2. Generates a PR plan with title, body, and best-effort file patches
3. Creates branch `jira/<ticket-key>-<slug>` and opens a PR
4. Posts the PR link as a Jira comment

> **Jira issue types:** The platform auto-detects which issue types are available in your project (Incident → Bug → Task → Story → Issue) so it never fails on missing types.

---

## GitHub Webhook

Register `https://your-platform/webhooks/github` in GitHub:
**Settings → Webhooks → Content type: application/json**

Set `GITHUB_WEBHOOK_SECRET` in `.env` — the platform verifies HMAC-SHA256 signatures on every request.

Triggers on: `push` to `main`/`master`, PR `opened`/`synchronize` events.

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
| `GET` | `/llm/status` | viewer | Active LLM provider + fallback chain status |
| `GET` | `/docs` | None | Swagger UI |

### Chat (AI Assistant)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/chat` | viewer | Conversational DevOps AI with tool use (AWS, GitHub, K8s, Jira, Slack) |
| `GET` | `/chat/action_count` | viewer | Number of actions executed this session |

### Incident Pipelines

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/incident/run` | developer | v1 monolithic pipeline |
| `POST` | `/incidents/run` | developer | Alias for `/incident/run` |
| `POST` | `/incidents/run/async` | developer | Fire-and-forget — returns job ID immediately |
| `POST` | `/v2/incident/run` | developer | v2 LangGraph multi-agent pipeline with policy engine |

### War Room

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/warroom/create` | developer | Create Slack war room with full AI analysis and activate bot |
| `GET` | `/warroom/list` | viewer | List active war rooms |
| `POST` | `/warroom/{id}/ask` | developer | Ask a question in a war room context |

### Post-Mortem

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/incident/postmortem` | developer | Generate AI post-mortem report (enriched from vector memory) |

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
| `GET` | `/aws/ecs/services` | viewer | ECS service counts |
| `GET` | `/aws/ecs/stopped-tasks` | viewer | Stopped ECS task reasons |
| `GET` | `/aws/lambda/functions` | viewer | List Lambda functions |
| `GET` | `/aws/lambda/errors` | viewer | Lambda error metrics |
| `GET` | `/aws/rds/instances` | viewer | RDS instance list |
| `GET` | `/aws/rds/events` | viewer | RDS events |
| `GET` | `/aws/elb/target-health` | viewer | ALB target health |
| `GET` | `/aws/cloudwatch/alarms` | viewer | CloudWatch alarms |
| `POST` | `/aws/cloudwatch/metrics` | viewer | Fetch metric series |
| `GET` | `/aws/logs/groups` | viewer | List CloudWatch log groups |
| `GET` | `/aws/logs/recent` | viewer | Recent log events |
| `GET` | `/aws/logs/search` | viewer | Search logs by pattern |
| `GET` | `/aws/cloudtrail/events` | viewer | Recent CloudTrail API events |
| `GET` | `/aws/s3/buckets` | viewer | S3 bucket list |
| `GET` | `/aws/sqs/queues` | viewer | SQS queue list |
| `GET` | `/aws/dynamodb/tables` | viewer | DynamoDB table list |
| `GET` | `/aws/route53/healthchecks` | viewer | Route53 health checks |
| `GET` | `/aws/sns/topics` | viewer | SNS topic list |
| `GET` | `/aws/context` | viewer | Full AWS observability snapshot |
| `POST` | `/aws/diagnose` | developer | AI root cause analysis from live AWS data |
| `POST` | `/aws/predict-scaling` | viewer | Predict if scaling is needed |
| `POST` | `/aws/assess-deployment` | developer | Pre-deploy assessment |

### Cost Analysis

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/cost/dashboard` | viewer | Full cost dashboard — MTD, forecast, service breakdown, 6-month trend, linked accounts |
| `GET` | `/cost/resources` | viewer | Resource-level cost breakdown (EC2, RDS, Lambda, ECS) with on-demand estimates |
| `POST` | `/cost/estimate` | viewer | Estimate cost from plain English description |
| `POST` | `/cost/analyze` | viewer | Estimate cost impact of a set of actions |
| `POST` | `/cost/terraform` | viewer | Estimate cost from Terraform plan JSON |

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
| `POST` | `/incident/war-room` | developer | Slack war room announcement |
| `POST` | `/incident/jira` | developer | Create Jira incident (auto-detects available issue types) |
| `POST` | `/incident/opsgenie` | developer | Notify OpsGenie on-call |
| `POST` | `/incident/github/issue` | developer | Create GitHub issue |
| `POST` | `/incident/github/pr` | developer | Create GitHub PR |

### Security & User Management

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/security/roles` | admin | List all users with roles and permissions |
| `POST` | `/security/roles/assign` | admin | Assign role to user |
| `DELETE` | `/security/roles/{user}` | admin | Revoke user role |
| `POST` | `/security/check` | viewer | Check if a user can perform an action |
| `GET` | `/secrets/status` | admin | Which env vars are configured (boolean only, no values) |
| `POST` | `/secrets` | admin | Write secrets to `.env` file |

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

### Webhooks (verified by signature, no auth)

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
pytest -q                          # all tests
pytest -q tests/test_main.py      # API tests only
python test_websocket.py           # manual WebSocket test (needs running server)
```

---

## Multi-LLM Support

The platform automatically selects the best available provider at startup:

```
Claude (ANTHROPIC_API_KEY)
  └── Groq (GROQ_API_KEY)  ← 100k tokens/day free tier
        └── Ollama (local)  ← no key required
```

The `/llm/status` endpoint returns the active provider and availability of each. The dashboard shows a live indicator and an amber warning banner when only Groq is active.

To add a new provider: implement `BaseLLM` in `app/llm/base.py` and register it in `LLMFactory` (`app/llm/factory.py`).

---

## Continuous Monitoring

Enable background anomaly detection:

```env
ENABLE_MONITOR_LOOP=true
MONITOR_INTERVAL_SECONDS=60
AUTO_REMEDIATE_ON_MONITOR=false   # start with alert-only
```

The monitor (`app/monitoring/loop.py`) polls K8s for crash-looping pods and unhealthy states. When anomalies are found it triggers the v2 pipeline with `auto_remediate=AUTO_REMEDIATE_ON_MONITOR`.
