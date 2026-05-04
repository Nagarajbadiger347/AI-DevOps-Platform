# NexusOps — AI DevOps Platform

An AI-powered DevOps command center built for SaaS multi-tenancy. Connect AWS, Kubernetes, GitHub, Slack, PagerDuty, and OpsGenie — then let the AI debug incidents, review PRs, analyze costs, and manage infrastructure autonomously. Works from a terminal CLI, a Slack war room, or a web dashboard.

---

## Features

- **Multi-agent incident pipeline** — LangGraph 5-agent workflow (Observe → Plan → Decide → Execute → Report) with RBAC, dry-run, confidence gating, and pgvector memory
- **AI chat assistant** — Conversational interface with live infra context, tool routing, and streaming responses
- **Terminal CLI** — `nexusops` command for incident response, K8s ops, AWS queries, approvals, and chat — no browser needed
- **PagerDuty & OpsGenie auto-triage** — Inbound webhooks with HMAC/token verification; on alert, creates Slack war room, posts change timeline, fires AI pipeline — zero manual steps
- **Continuous monitoring** — Background anomaly detection across K8s, EC2, ECS, Lambda, RDS, SQS, CloudWatch, and Grafana with deduplication and auto-escalation
- **Change timeline correlation** — Every incident context includes a unified, newest-first timeline of GitHub commits, merged PRs, CloudTrail write events, and K8s deployment events from the last 2 hours
- **Blast radius estimation** — Approval messages include downstream service impact count before any destructive action runs
- **Degraded mode** — Platform degrades gracefully when Postgres, Redis, or LLM is unavailable; read-only observability always works; Slack notified automatically
- **K8s operations** — Health checks, rolling restarts, scale deployments, pod log analysis
- **AWS observability** — EC2, ECS, Lambda, RDS, ALB, CloudWatch, CloudTrail, S3, SQS, DynamoDB
- **Cost analysis** — Live AWS spend, per-resource breakdown, multi-account Organizations view, Terraform cost estimation
- **Slack war room** — Dedicated incident channel with AI bot that answers engineer questions
- **AI PR review** — Security, infra, and code quality review on GitHub PRs
- **Jira → Auto PR** — Creates GitHub PRs automatically from Jira change-request tickets
- **Post-mortem generation** — AI-generated reports enriched from past incident memory
- **JWT auth + RBAC** — Role-based access control enforced at route and execution layers
- **Multi-LLM support** — Claude → OpenAI → Groq → Ollama with automatic fallback and per-call timeout
- **Multi-tenancy** — Full tenant isolation: incidents, chat, approvals, users all scoped per tenant
- **pgvector memory** — Semantic incident search powered by PostgreSQL + pgvector

---

## Architecture

```
Browser / CLI ──▶ Nginx (TLS + rate limiting)
                     │
                     ▼
              FastAPI (4 workers)
                ├── app/api/           HTTP layer — no business logic
                ├── app/orchestrator/  LangGraph StateGraph — routing and backoff
                ├── app/agents/        Decision units — read state, return diffs
                ├── app/execution/     RBAC + policy gated; audit log per action
                ├── app/integrations/  Pure I/O adapters (AWS, K8s, GitHub, Slack…)
                ├── app/memory/        Short-term scratchpad, long-term pgvector
                ├── app/monitoring/    Background anomaly detection loop
                ├── app/security/      JWT auth, RBAC roles, audit trail
                ├── app/tenants/       Multi-tenant isolation middleware + store
                └── app/core/          Config, logging, trace middleware, DB pool,
                                       degraded-mode health manager
                     │
                     ▼
              PostgreSQL + pgvector   ← all data, all tenants, fully isolated
              Redis                   ← rate limiting, session cache
```

### Incident Pipeline

```
Input — description + severity
  │   (from: CLI / API / PagerDuty webhook / OpsGenie webhook / monitor loop)
  │
  ├── Degraded-mode guard (abort + Slack if DB or LLM unavailable)
  │
  ├──▶ Collect AWS       ─┐
  ├──▶ Collect K8s       ─┼─ parallel
  ├──▶ Collect GitHub    ─┘
  ├──▶ Build change timeline (commits + CloudTrail + K8s events, last 2h)
  │
  ▼
AI: root cause, findings, confidence-scored action plan
  │
  ▼
Decision: approve auto-execute OR require human approval
  ├── blast radius estimated (downstream services affected)
  ├── approval message includes: risk, confidence, blast_radius, cost impact
  │
  ▼
Execute actions (RBAC + dry-run + LLM timeout guard)
  ├── K8s restart / scale
  ├── Slack war room + enriched brief
  ├── Jira ticket
  ├── GitHub PR
  └── OpsGenie / PagerDuty alert
  │
  ▼
Store to pgvector → future recall (per tenant)
  │
  ▼
Post-mortem generated on completion
```

### Inbound Alert Flow (PagerDuty / OpsGenie)

```
PagerDuty fires alert
  │
  ▼
POST /v1/webhooks/pagerduty
  ├── Verify HMAC-SHA256 signature
  ├── Parse v2 (messages[]) or v3 (event.data) payload
  ├── Create Slack war room immediately
  ├── Post initial brief to war room
  ├── Collect change timeline (GitHub + CloudTrail, 8s timeout)
  ├── Post change timeline to war room
  └── Fire AI pipeline in background thread
          → on-call engineer has context before they open Slack
```

### Degraded Mode

```
Background thread probes Postgres + Redis + LLM every 30 s
  │
  ├── All OK        → mode: full           (pipeline runs normally)
  ├── LLM down      → mode: no-ai          (observability works, pipeline skipped)
  ├── DB down       → mode: no-persistence (pipeline skipped, Slack notified)
  └── Both down     → mode: telemetry-only (read-only metric endpoints only)

GET /health/degraded  → always returns 200 with current mode
```

---

## CLI

The `cli.py` provides a terminal-first interface — the tool you reach for at 3am.

```bash
# Install dependencies
pip install click requests

# Authenticate
python cli.py login --url http://localhost:8000

# Or set env vars (for scripts / CI)
export NEXUSOPS_URL=http://localhost:8000
export NEXUSOPS_TOKEN=eyJ...
```

### Incident Response

```bash
# Run AI pipeline on an incident description
python cli.py incident run "payments-api pods crash-looping in prod" --severity high

# Dry-run (plan only, no execution)
python cli.py incident run "high CPU on auth service" --dry-run

# Auto-remediate low-risk actions without approval
python cli.py incident run "SQS backlog growing" --auto-remediate

# List recent incidents
python cli.py incident list --limit 20

# Get full detail for one incident
python cli.py incident get INC-001 --json
```

### Kubernetes

```bash
python cli.py k8s pods --namespace production
python cli.py k8s restart payments-api --namespace production
python cli.py k8s scale payments-api 5 --namespace production
```

### AWS

```bash
python cli.py aws alarms --state ALARM
python cli.py aws ec2
python cli.py aws cost --days 30
```

### Approvals

```bash
python cli.py approvals list
python cli.py approvals approve <approval_id>
python cli.py approvals reject  <approval_id> --reason "too broad"
```

### Health & Debug

```bash
python cli.py health status        # current platform mode (full / degraded / …)
python cli.py health integrations  # which integrations are configured
python cli.py health full          # deep scan across AWS, K8s, Grafana
```

### AI Chat

```bash
python cli.py chat "why is checkout latency spiking?"
```

All commands support `--json` for scripting / piping to `jq`.

---

## Quick Start

### Requirements

- Python 3.9+ (3.11 recommended)
- PostgreSQL 16 + pgvector extension
- Docker + Docker Compose (for production)

### Local Development

```bash
# 1. Install PostgreSQL + pgvector (Mac)
brew install postgresql@16
brew services start postgresql@16

git clone https://github.com/pgvector/pgvector.git /tmp/pgvector
cd /tmp/pgvector
PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make && make install

# 2. Create database
psql postgres -c "CREATE USER nexusops WITH PASSWORD 'nexusops';"
psql postgres -c "CREATE DATABASE nexusops OWNER nexusops;"
psql nexusops -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Install app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials

# 4. Run database migrations
python manage.py migrate

# 5. Run app
uvicorn app.orchestrator.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

On first run, a temporary admin account is created and credentials are printed to stdout. Change the password immediately or set `ADMIN_PASSWORD` in `.env`.

### Production (Docker)

```bash
cp .env.example .env          # set AUTH_ENABLED=true and a strong JWT_SECRET_KEY
docker compose up --build -d
```

| Service | Description |
|---|---|
| **nginx** | TLS termination, rate limiting, security headers (port 443) |
| **postgres** | PostgreSQL 15 with pgvector extension |
| **redis** | Rate limiting + LLM response cache (256 MB LRU) |
| **prometheus** | Metrics collection |
| **grafana** | Dashboards (port 3000) |
| **app** | FastAPI with 4 uvicorn workers (2 CPU / 2 GB limit) |

**TLS setup:**
```bash
openssl req -x509 -newkey rsa:4096 -keyout nginx/certs/server.key \
  -out nginx/certs/server.crt -days 365 -nodes -subj "/CN=localhost"
```

---

## Webhook Integrations

### PagerDuty

1. In PagerDuty → **Integrations → Generic Webhooks (v3)** → add endpoint:
   ```
   https://your-domain/v1/webhooks/pagerduty
   ```
2. Copy the signing secret into `.env`:
   ```
   PAGERDUTY_WEBHOOK_SECRET=your_secret_here
   ```
3. Subscribe to: `incident.triggered`, `incident.acknowledged`

When an alert fires, NexusOps will:
- Verify the HMAC-SHA256 signature
- Create a Slack war room (`#inc-pd-<id>`)
- Post a brief with severity and on-call context
- Collect the last 2h change timeline from GitHub + CloudTrail
- Fire the AI triage pipeline in the background

### OpsGenie

1. In OpsGenie → **Integrations → Webhook** → add endpoint:
   ```
   https://your-domain/v1/webhooks/opsgenie
   ```
2. Set a custom header `X-OG-Token: <your_token>` in the integration config
3. Copy the token into `.env`:
   ```
   OPSGENIE_WEBHOOK_TOKEN=your_token_here
   ```
4. Trigger on: `Create`, `Acknowledge`

Behaviour is identical to PagerDuty — war room created, change timeline posted, AI pipeline fired.

---

## Continuous Monitoring

The background monitor loop is **enabled by default**. It polls every 60 seconds across:

| Source | What it detects |
|---|---|
| Kubernetes | Pod restarts (≥5), NotReady nodes, deployments with 0 available replicas |
| EC2 | Stopped / terminated instances |
| ECS | Crashed tasks, services with 0/N running, degraded services |
| Lambda | Error count > 0 in last hour |
| CloudWatch | Alarms in ALARM state |
| RDS | Failover / crash / OOM events |
| SQS | Queue backlog > 1000 messages |
| Grafana | Firing alerts |

**Alert deduplication:** same alert suppressed for 10 minutes; hard cap at 3 pipeline triggers per fingerprint.

**Auto-remediation** is off by default — the loop alerts only. To enable:
```
AUTO_REMEDIATE_ON_MONITOR=true
```

To disable the loop entirely:
```
ENABLE_MONITOR_LOOP=false
```

---

## Authentication

```bash
# Get a token
curl -X POST http://localhost:8000/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'
# → {"access_token": "eyJ...", "token_type": "bearer"}

# Use the token
curl http://localhost:8000/v1/aws/ec2/instances \
  -H "Authorization: Bearer eyJ..."
```

**Dev mode** (no token required): set `AUTH_ENABLED=false`. The platform reads `X-User` header and defaults to `developer` role.

### Roles

| Role | Permissions |
|---|---|
| `super_admin` | Full access + can assign admin/super_admin roles |
| `admin` | Deploy, manage users, manage secrets, assign roles |
| `developer` | Deploy, read, write |
| `viewer` | Read-only |

---

## Environment Variables

### Auth & Security

| Variable | Default | Description |
|---|---|---|
| `AUTH_ENABLED` | `true` | `false` = dev mode (X-User header, no JWT) |
| `JWT_SECRET_KEY` | — | Token signing key — `openssl rand -hex 32` |
| `APP_SECRET_KEY` | — | Password hashing key |
| `JWT_EXPIRE_MINS` | `480` | Token lifetime in minutes |
| `ADMIN_USERNAME` | `admin` | Bootstrap admin username |
| `ADMIN_PASSWORD` | — | Bootstrap admin password (auto-generated if not set) |

### LLM Providers

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | Preferred provider: `claude` \| `openai` \| `groq` \| `ollama` |
| `ANTHROPIC_API_KEY` | Claude (primary) |
| `OPENAI_API_KEY` | OpenAI GPT-4o (fallback) |
| `GROQ_API_KEY` | Groq Llama 3.3-70B (secondary fallback) |
| `OLLAMA_HOST` | Local Ollama — default `http://localhost:11434` |
| `LLM_TIMEOUT_SECONDS` | `10` | Hard per-attempt timeout for LLM calls |

### Pipeline Behaviour

| Variable | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE_THRESHOLD` | `0.6` | Plans below this require human approval |
| `AUTO_EXECUTE_RISK_LEVELS` | `low` | Risk levels that auto-execute without approval |
| `ENABLE_MONITOR_LOOP` | `true` | Background anomaly detection (on by default) |
| `MONITOR_INTERVAL_SECONDS` | `60` | Polling interval in seconds |
| `AUTO_REMEDIATE_ON_MONITOR` | `false` | Auto-execute remediation from monitor alerts |

### Database & Cache

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://nexusops:nexusops@localhost:5432/nexusops` | PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis |

### AWS

| Variable | Description |
|---|---|
| `AWS_REGION` | Default: `us-west-2` |
| `AWS_ACCESS_KEY_ID` | Access key (or use IAM instance role) |
| `AWS_SECRET_ACCESS_KEY` | Secret key |
| `AWS_SESSION_TOKEN` | Session token (STS / assumed role) |

### Integrations

| Variable | Required for |
|---|---|
| `GITHUB_TOKEN` | GitHub — `repo` + `pull_requests` scope |
| `GITHUB_REPO` | GitHub — default repo in `owner/repo` format |
| `SLACK_BOT_TOKEN` | Slack messaging + war rooms |
| `SLACK_CHANNEL` | Default Slack channel (default: `#incidents`) |
| `SLACK_SIGNING_SECRET` | Slack war room bot event verification |
| `JIRA_URL` | Jira — e.g. `https://yourorg.atlassian.net` |
| `JIRA_USER` | Jira user email |
| `JIRA_TOKEN` | Jira API token |
| `JIRA_PROJECT` | Jira project key |
| `OPSGENIE_API_KEY` | OpsGenie — outbound on-call alerts |
| `OPSGENIE_WEBHOOK_TOKEN` | OpsGenie — inbound webhook auth token |
| `PAGERDUTY_WEBHOOK_SECRET` | PagerDuty — inbound webhook HMAC secret |
| `GRAFANA_URL` | Grafana — e.g. `http://grafana:3000` |
| `GRAFANA_TOKEN` | Grafana service account token |
| `K8S_IN_CLUSTER` | `true` when running inside a K8s pod |
| `KUBECONFIG` | Path to kubeconfig (default: `~/.kube/config`) |

---

## API Reference

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/auth/login` | Get JWT |
| `GET`  | `/v1/auth/me` | Current user identity and role |
| `POST` | `/v1/incidents/run` | Run incident pipeline |
| `GET`  | `/v1/incidents/{id}` | Get incident detail |
| `POST` | `/v1/chat` | AI chat (non-streaming) |
| `GET`  | `/v1/chat/stream` | AI chat (SSE streaming) |
| `GET`  | `/v1/memory/incidents` | List stored incidents (tenant-scoped) |
| `GET`  | `/v1/memory/incidents/search?q=cpu` | Semantic search |
| `GET`  | `/v1/memory/incidents/trends` | Trend analysis |
| `GET`  | `/v1/approvals/pending` | Pending approvals |
| `POST` | `/v1/approvals/{id}/approve` | Approve an AI action plan |
| `POST` | `/v1/approvals/{id}/reject` | Reject an AI action plan |

### Health Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Basic status + incident count |
| `GET` | `/health/live` | Liveness probe (K8s) |
| `GET` | `/health/ready` | Readiness probe — checks DB + LLM key |
| `GET` | `/health/degraded` | Current platform mode + component status |
| `GET` | `/health/full` | Deep scan — AWS, K8s, Grafana |
| `GET` | `/health/integrations` | Which integrations are configured |
| `GET` | `/health/detectors` | Monitor detector health (stale check) |

### Webhook Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/webhooks/pagerduty` | PagerDuty v2/v3 alert → auto-triage |
| `POST` | `/v1/webhooks/opsgenie` | OpsGenie alert → auto-triage |
| `POST` | `/v1/webhooks/github` | GitHub push/PR → pipeline |
| `POST` | `/v1/webhooks/grafana` | Grafana alert → pipeline |
| `POST` | `/v1/webhooks/cloudwatch` | CloudWatch SNS → pipeline |

### Run an Incident Pipeline

```bash
curl -X POST http://localhost:8000/v1/incidents/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id":     "INC-001",
    "description":    "High CPU on payment service, API latency spiking",
    "severity":       "high",
    "auto_remediate": false,
    "dry_run":        true
  }'
```

### Response Fields

| Field | Description |
|---|---|
| `status` | `completed` / `approval_required` / `degraded` / `failed` |
| `root_cause` | 2–3 sentence AI explanation |
| `risk` | `critical` / `high` / `medium` / `low` |
| `confidence` | 0.0–1.0 AI confidence score |
| `actions` | Planned actions with type, description, params |
| `executed_actions` | Actions that ran successfully |
| `blast_radius` | Downstream services affected by planned actions |
| `approval_reason` | Why human approval was required (if applicable) |
| `change_timeline` | Unified list of recent changes (commits, CloudTrail, K8s) |
| `report` | Full Markdown incident report |

---

## Slack War Room Bot

When an incident pipeline runs or a PagerDuty/OpsGenie webhook fires:

1. Creates a `#inc-<incident-id>` Slack channel
2. Posts an enriched brief — severity, root cause, findings, action plan, blast radius
3. Posts the change timeline (last 2h of GitHub commits + CloudTrail + K8s events)
4. Activates a bot that answers engineer questions in the thread

**Example questions the bot handles:**
- "which PR raised this?" → fetches recent GitHub PRs and commits
- "check Grafana alerts" → queries firing alerts and annotations
- "why are pods crashing?" → shows unhealthy pods + replica status
- "what should we do next?" → AI synthesises current state into next steps

**Setup:**
1. Add `SLACK_SIGNING_SECRET` and `SLACK_BOT_USER_ID` to `.env`
2. In Slack app settings, set Event Subscriptions URL to `https://your-domain/v1/webhooks/slack`
3. Subscribe to: `message.channels`, `message.groups`, `app_mention`
4. OAuth scopes: `channels:read`, `channels:write`, `chat:write`, `groups:write`, `groups:read`

---

## Database Migrations

```bash
# Check status
python manage.py status

# Apply pending migrations
python manage.py migrate

# Rollback last migration
python manage.py rollback

# Docker
docker compose exec app python manage.py migrate
```

---

## Directory Structure

```
app/
├── orchestrator/           # FastAPI app entry point, LangGraph StateGraph, PipelineState
│   ├── main.py             #   App wiring, middleware, lifespan (monitor loop start)
│   ├── graph.py            #   LangGraph pipeline definition
│   ├── runner.py           #   Public run_pipeline() — degraded-mode guard lives here
│   └── state.py            #   PipelineState schema
│
├── api/                    # HTTP handlers only — no business logic
│   ├── incidents.py        #   Run pipeline, memory search, trend endpoints
│   ├── warroom.py          #   War room + Slack bot events
│   ├── chat.py             #   AI chat (streaming + non-streaming)
│   ├── webhooks.py         #   Inbound: PagerDuty, OpsGenie, GitHub, Grafana, CloudWatch
│   ├── approvals.py        #   Human approval workflow
│   ├── deploy.py           #   Pre-deploy assessment, Jira → auto-PR
│   ├── aws.py              #   EC2, ECS, Lambda, RDS, CloudWatch, cost routes
│   ├── k8s.py              #   Kubernetes resource operations
│   ├── auth.py             #   JWT login, SSO (Google/GitHub), refresh
│   ├── saas.py             #   Orgs, billing, usage metering (Stripe)
│   ├── health.py           #   Health probes + /health/degraded
│   ├── agentic.py          #   Agentic task routing
│   ├── cost.py             #   AWS cost summary + estimation
│   ├── github.py           #   GitHub PR review routes
│   ├── vscode.py           #   VS Code extension bridge
│   ├── security.py         #   User management, invites, RBAC admin
│   ├── tenants.py          #   Workspace management
│   ├── websocket_routes.py #   WebSocket — streaming AI responses
│   └── misc.py             #   Miscellaneous utility routes
│
├── agents/                 # Decision-only units — read state, return diffs
│   ├── base.py             #   BaseAgent: LLM call (with timeout), cache, retry, logging
│   ├── decision/agent.py   #   Risk scoring, approval gating, blast radius estimation
│   ├── planner/agent.py    #   Generates confidence-scored action plans
│   ├── memory/agent.py     #   Writes to pgvector, retrieves similar past incidents
│   ├── infra/              #   AWSAgent, K8sAgent — collect infrastructure context
│   ├── scm/                #   GitHubAgent — recent commits, PRs
│   ├── observer.py         #   Collects initial observability data
│   ├── debugger.py         #   Root cause analysis
│   ├── executor.py         #   Executes approved actions
│   └── reporter.py         #   Generates final incident report
│
├── execution/              # RBAC + policy gated executor; audit log per action
│   ├── executor.py         #   Runs actions with RBAC + dry-run + circuit breaker
│   ├── validator.py        #   Simulates action outcomes before execution
│   └── action_registry.py  #   Catalog of ~50 runnable actions
│
├── integrations/           # Pure I/O adapters — no business logic
│   ├── aws_ops.py          #   EC2, ECS, Lambda, RDS, CloudWatch, CloudTrail, S3, SQS…
│   ├── k8s_ops.py          #   Kubernetes client wrapper
│   ├── github.py           #   GitHub API + PR review
│   ├── gitlab_ops.py       #   GitLab pipelines and deployments
│   ├── slack.py            #   Slack messaging + war room channel creation
│   ├── slack_bot.py        #   Slack event handling + bot Q&A
│   ├── jira.py             #   Jira ticket creation and comments
│   ├── grafana.py          #   Grafana alerts + annotations
│   ├── grafana_checker.py  #   Grafana health probe
│   ├── opsgenie.py         #   Outbound OpsGenie alert creation
│   ├── email.py            #   SMTP notifications
│   ├── vscode.py           #   VS Code extension output bridge
│   ├── linux_checker.py    #   Linux node health probe
│   ├── webhooks.py         #   Outbound webhook dispatcher
│   └── universal_collector.py  # Parallel context fetch + change timeline builder
│
├── memory/                 # Long-term knowledge
│   ├── vector_db.py        #   pgvector store — semantic incident search
│   ├── long_term.py        #   Trend detection, root cause clustering
│   ├── trend_analysis.py   #   MTTR, frequency, severity distribution
│   ├── short_term.py       #   Session-scoped scratchpad
│   └── knowledge.py        #   Static knowledge base
│
├── workflows/              # High-level workflow orchestration
│   ├── incident_workflow.py
│   └── unified_workflow.py
│
├── services/               # Business logic (between API and pipeline)
│   └── incident_service.py
│
├── llm/                    # Multi-LLM abstraction
│   ├── factory.py          #   Provider selection: claude → openai → groq → ollama
│   ├── claude.py           #   Claude API wrapper
│   ├── openai.py           #   OpenAI wrapper
│   ├── ollama.py           #   Ollama local wrapper
│   ├── resilient.py        #   Fallback chain
│   └── base.py             #   LLM base interface
│
├── core/                   # Platform utilities
│   ├── config.py           #   All settings (pydantic-settings, reads .env)
│   ├── degraded.py         #   Degraded-mode health manager (NEW)
│   ├── auth.py             #   JWT encode/decode, token rotation
│   ├── logging.py          #   Structured logging, TraceMiddleware, trace_id
│   ├── metrics.py          #   Prometheus counters + histograms
│   ├── database.py         #   PostgreSQL connection pool
│   ├── llm_cache.py        #   SHA-256 keyed LLM response cache (5-min TTL)
│   ├── ratelimit.py        #   Per-user/per-endpoint rate limiter (Redis)
│   ├── audit.py            #   Append-only audit log
│   ├── usage.py            #   Token metering + plan quota enforcement
│   ├── context.py          #   Request-scoped context vars (tenant_id, user)
│   ├── pipeline_events.py  #   Pipeline lifecycle event hooks
│   ├── schema.py           #   Migration runner (apply_migrations)
│   └── exceptions.py       #   Typed API error hierarchy
│
├── tenants/                # Multi-tenant isolation
│   ├── middleware.py        #   Injects tenant_id from JWT / X-Tenant-ID header
│   ├── store.py             #   Tenant data layer (DB queries)
│   └── models.py            #   Workspace / Plan / User schemas
│
├── security/               # Auth + access control
│   ├── rbac.py             #   Role hierarchy + permission matrix
│   ├── users.py            #   User management (PostgreSQL)
│   └── invite.py           #   Workspace invite flow
│
├── incident/               # Incident-specific state
│   ├── approval.py         #   Human approval requests + 30-min expiry
│   ├── post_mortem.py      #   AI-generated post-mortem reports
│   └── war_room_store.py   #   War room channel state
│
├── monitoring/             # Continuous background monitoring (on by default)
│   └── loop.py             #   Detectors: K8s, EC2, ECS, Lambda, RDS, SQS, CW, Grafana
│
├── cost/                   # Cloud cost analysis
│   ├── analyzer.py         #   Multi-account spend analysis + action cost estimation
│   └── pricing.py          #   AWS pricing data
│
├── policies/               # Policy enforcement
│   └── policy_engine.py    #   Rule-based policy check before execution
│
├── correlation/            # Incident correlation
│   └── engine.py           #   Cross-incident root cause clustering
│
├── chat/                   # Chat intelligence
│   ├── intelligence.py     #   Tool routing, intent classification
│   └── memory.py           #   Per-session chat history (PostgreSQL)
│
├── plugins/                # Plugin extension point
├── migrations/             # Database schema migrations
│   ├── 0001_initial.py
│   └── 0002_saas_foundation.py
│
└── tools/                  # LangChain-style tool wrappers
    ├── aws.py
    ├── kubernetes.py
    └── gitlab.py

cli.py                      # Terminal CLI — nexusops command (NEW)
manage.py                   # Database migration CLI
gunicorn.conf.py            # Gunicorn worker config (production)
run-docker.sh               # Docker helper script
conftest.py                 # Pytest fixtures
test_websocket.py           # WebSocket integration test

docker-compose.yml          # Full stack: App + Nginx + Postgres + Redis + Prometheus + Grafana
Dockerfile                  # Multi-stage build (non-root production user)

monitoring/
├── prometheus.yml           # Prometheus scrape config
└── alert_rules.yml          # Prometheus alerting rules

nginx/
└── nginx.conf               # Reverse proxy: TLS, rate limiting, security headers

vscode-extension/
├── extension.js             # VS Code extension source
├── package.json
└── nsops-vscode-1.0.0.vsix  # Packaged extension (install in VS Code)

scripts/
└── export_training_data.py  # Export incident data for LLM fine-tuning

tests/
└── test_main.py             # API integration tests

post_mortems/               # Generated post-mortem documents (local storage)
data/                       # Persisted alert state (monitor loop dedup state)
```

---

## What's New (May 2026)

| Change | Details |
|---|---|
| **PagerDuty auto-triage** | v2 + v3 webhook support, HMAC verification, war room + change timeline on alert |
| **OpsGenie auto-triage** | Bearer-token verification, P1–P5 severity mapping, same war room flow |
| **Degraded mode** | `app/core/degraded.py` — platform survives DB/Redis/LLM outages gracefully |
| **Monitor loop on by default** | `ENABLE_MONITOR_LOOP=true`; lifespan now reads from `settings`, not raw `os.getenv` |
| **LLM timeout** | 10s hard ceiling per attempt via `ThreadPoolExecutor`; configurable via `LLM_TIMEOUT_SECONDS` |
| **Change timeline** | `universal_collector` builds unified GitHub + CloudTrail + K8s event list per context collection |
| **Blast radius** | `DecisionAgent` estimates downstream services affected; shown in approval messages |
| **CLI** | `cli.py` — full terminal interface for incidents, K8s, AWS, approvals, health, chat |

---

## License

MIT
