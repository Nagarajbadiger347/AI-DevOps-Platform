# NexusOps — AI DevOps Platform

An AI-powered DevOps command center. Connect AWS, Kubernetes, GitHub, and Slack — then ask the AI to debug incidents, review PRs, analyze costs, and manage infrastructure from one interface.

---

## Features

- **Multi-agent incident pipeline** — LangGraph 5-agent workflow (Planner → Gather → Debug → Execute → Report) with RBAC, dry-run, and ChromaDB memory
- **AI chat assistant** — Conversational interface with live infra context, tool routing, and streaming responses
- **K8s operations** — Health checks, rolling restarts, scale deployments, pod log analysis
- **AWS observability** — EC2, ECS, Lambda, RDS, ALB, CloudWatch, CloudTrail, S3, SQS, DynamoDB
- **Cost analysis** — Live AWS spend, per-resource breakdown, multi-account Organizations view, Terraform cost estimation
- **Pre-deployment assessment** — AI checks cluster state and active alarms before any deploy
- **Slack war room** — Dedicated incident channel with AI bot that answers engineer questions
- **AI PR review** — Security, infra, and code quality review on GitHub PRs
- **Jira → Auto PR** — Creates GitHub PRs automatically from Jira change-request tickets
- **Post-mortem generation** — AI-generated reports enriched from past incident memory
- **JWT auth + RBAC** — Role-based access control enforced at route and execution layers
- **Multi-LLM support** — Claude → OpenAI → Groq → Ollama with automatic fallback

---

## Architecture

```
Browser ──▶ Nginx (TLS + rate limiting)
               │
               ▼
        FastAPI (4 workers)
          ├── app/api/          HTTP layer — no business logic
          ├── app/services/     Business logic — orchestrates pipeline
          ├── app/orchestrator/ LangGraph StateGraph — routing and backoff
          ├── app/agents/       Decision units — read state, return diffs
          ├── app/execution/    RBAC + policy gated; audit log per action
          ├── app/integrations/ Pure I/O adapters (AWS, K8s, GitHub, Slack...)
          ├── app/memory/       Short-term scratchpad, long-term ChromaDB, knowledge base
          ├── app/security/     JWT auth, RBAC roles, audit trail
          └── app/core/         Config, logging, trace middleware
```

### Incident Pipeline

```
Input (incident description + severity)
  │
  ├──▶ Collect AWS    ─┐
  ├──▶ Collect K8s    ─┼─ parallel
  └──▶ Collect GitHub ─┘
             │
             ▼
     AI: root cause, findings, action plan
             │
             ▼
     Execute actions (if auto_remediate=true)
       ├── K8s restart / scale
       ├── Slack war room
       ├── Jira ticket
       ├── GitHub PR
       └── OpsGenie alert
             │
             ▼
     Store to ChromaDB → future recall
```

---

## Quick Start

### Requirements

- Python 3.9+ (3.11 recommended)
- Docker + Docker Compose (for production)

### Local Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in credentials
uvicorn app.orchestrator.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

On first run with no users, a temporary admin account is created and credentials are printed to stdout. Change the password immediately or set `ADMIN_PASSWORD` in `.env`.

### Production (Docker)

```bash
cp .env.example .env          # set AUTH_ENABLED=true and a strong JWT_SECRET_KEY
docker compose up --build -d
```

| Service | Description |
|---|---|
| **nginx** | TLS termination, rate limiting, security headers (port 443) |
| **app** | FastAPI with 4 uvicorn workers (2 CPU / 2 GB limit) |
| **redis** | Rate limiting + response cache (256 MB LRU) |

**TLS setup:**
```bash
# Place server.crt and server.key in nginx/certs/
# For local testing:
openssl req -x509 -newkey rsa:4096 -keyout nginx/certs/server.key \
  -out nginx/certs/server.crt -days 365 -nodes -subj "/CN=localhost"
```

---

## Authentication

```bash
# Get a token
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=your-password"
# → {"access_token": "eyJ...", "token_type": "bearer"}

# Use the token
curl http://localhost:8000/aws/ec2/instances \
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

```bash
# Assign a role (admin token required)
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
| `AUTH_ENABLED` | `true` | `false` = dev mode (X-User header, no JWT) |
| `JWT_SECRET_KEY` | — | Token signing key — `openssl rand -hex 32` |
| `APP_SECRET_KEY` | — | Password hashing key (separate from JWT key) |
| `JWT_EXPIRE_MINS` | `480` | Token lifetime in minutes |
| `ADMIN_USERNAME` | `admin` | Bootstrap admin username |
| `ADMIN_PASSWORD` | — | Bootstrap admin password (auto-generated if not set) |

### LLM Providers

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | Preferred provider: `claude` \| `openai` \| `groq` \| `ollama` |
| `ANTHROPIC_API_KEY` | Claude (primary) |
| `OPENAI_API_KEY` | OpenAI GPT-4o (fallback) |
| `GROQ_API_KEY` | Groq Llama 3.3-70B (secondary fallback, 100k tokens/day free) |
| `OLLAMA_HOST` | Local Ollama — default `http://localhost:11434` |

### AWS

| Variable | Description |
|---|---|
| `AWS_REGION` | Default: `us-east-1` |
| `AWS_ACCESS_KEY_ID` | Access key (or use IAM instance role) |
| `AWS_SECRET_ACCESS_KEY` | Secret key |
| `AWS_SESSION_TOKEN` | Session token (STS / assumed role) |

### Integrations

| Variable | Required for |
|---|---|
| `GITHUB_TOKEN` | GitHub — `repo` + `pull_requests` scope |
| `GITHUB_REPO` | GitHub — default repo in `owner/repo` format |
| `SLACK_BOT_TOKEN` | Slack messaging |
| `SLACK_CHANNEL` | Slack — default channel (default: `#general`) |
| `SLACK_SIGNING_SECRET` | Slack war room bot |
| `JIRA_URL` | Jira — e.g. `https://yourorg.atlassian.net` |
| `JIRA_USER` | Jira user email |
| `JIRA_TOKEN` | Jira API token |
| `JIRA_PROJECT` | Jira project key (default: `DEVOPS`) |
| `OPSGENIE_API_KEY` | OpsGenie on-call alerts |
| `GRAFANA_URL` | Grafana — e.g. `http://grafana:3000` |
| `GRAFANA_TOKEN` | Grafana service account token |
| `K8S_IN_CLUSTER` | `true` when running inside a K8s pod |
| `KUBECONFIG` | Path to kubeconfig (default: `~/.kube/config`) |
| `REDIS_URL` | Redis — default `redis://localhost:6379/0` |

### Email Notifications

| Variable | Description |
|---|---|
| `SMTP_HOST` | SMTP server |
| `SMTP_PORT` | 587 (TLS) or 465 (SSL) |
| `SMTP_USER` / `SMTP_PASSWORD` | SMTP credentials |
| `ALERT_EMAIL_TO` | Comma-separated recipient addresses |

### Pipeline Behaviour

| Variable | Default | Description |
|---|---|---|
| `MIN_CONFIDENCE_THRESHOLD` | `0.6` | Plans below this always require approval |
| `AUTO_EXECUTE_RISK_LEVELS` | `low,medium` | Risk levels that execute without approval |
| `ENABLE_MONITOR_LOOP` | `false` | Background anomaly detection |
| `MONITOR_INTERVAL_SECONDS` | `60` | Polling interval |

All integrations degrade gracefully — missing credentials return a structured error, not a crash.

---

## API Reference

### Core Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/token` | Get JWT — form body: `username` + `password` |
| `GET` | `/auth/me` | Current user identity and role |
| `POST` | `/debug-pod` | K8s pod debug (LangGraph 5-agent) |
| `POST` | `/incidents/run` | General incident pipeline |
| `POST` | `/chat` | AI chat (non-streaming) |
| `GET` | `/chat/stream` | AI chat (SSE streaming) |
| `GET` | `/health` | Platform health check |

### Debug a K8s Pod

```bash
curl -X POST http://localhost:8000/debug-pod \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "production",
    "pod_name":  "payment-api-abc123",
    "dry_run":   true,
    "auto_fix":  false
  }'
```

`dry_run=true` (default) — safe to run in production. Shows what the AI would do without executing.

### Run an Incident Pipeline

```bash
curl -X POST http://localhost:8000/incidents/run \
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
| `failure_type` | `CrashLoopBackOff` / `OOMKilled` / `HighCPU` / etc. |
| `severity_ai` | AI-assessed severity: `critical` / `high` / `medium` / `low` |
| `root_cause` | 2–3 sentence explanation |
| `fix_suggestion` | Specific actionable fix |
| `findings` | Observations from collected data |
| `actions_taken` | Each action's result: skipped / dry-run / success / failed |
| `report` | Full Markdown incident report |
| `elapsed_s` | Total wall-clock time |

---

## Slack War Room Bot

When a war room is created, the platform:
1. Creates a `#inc-<incident-id>` Slack channel
2. Posts a rich summary — severity, root cause, findings, action plan, PR links
3. Activates a bot that answers engineer questions in the thread

**Example questions the bot handles:**
- "which PR raised this?" → fetches recent GitHub PRs and commits
- "check Grafana alerts" → queries firing alerts and annotations
- "why are pods crashing?" → shows unhealthy pods + replica status
- "what should we do next?" → AI synthesises current state into next steps

**Setup:**
1. Add `SLACK_SIGNING_SECRET` and `SLACK_BOT_USER_ID` to `.env`
2. In Slack app settings, set Event Subscriptions URL to `https://your-domain/webhooks/slack`
3. Subscribe to: `message.channels`, `message.groups`, `app_mention`
4. OAuth scopes: `channels:read`, `channels:write`, `chat:write`, `groups:write`, `groups:read`

---

## Jira → Auto PR

Register `https://your-platform/jira/webhook` in Jira under **Project Settings → Webhooks → Event: Issue Created**.

Triggers on Change Request, Task, or Story issue types (or issues with label `auto-pr`).

**Flow:** AI reads Jira ticket → generates PR with file patches → creates branch `jira/<key>-<slug>` → opens PR → posts PR link as Jira comment.

---

## Directory Structure

```
app/
├── orchestrator/      # FastAPI app, LangGraph StateGraph, PipelineState
├── api/               # HTTP handlers only — no business logic
├── services/          # Business logic — pipeline, approval flow, notifications
├── agents/            # Planner, Decision, Memory, Executor, Infra agents
├── execution/         # RBAC + policy gated executor; audit log per action
├── policies/          # PolicyEngine + rules.json
├── memory/            # Short-term, long-term (ChromaDB), knowledge base
├── integrations/      # AWS, K8s, GitHub, Slack, Jira, OpsGenie, Grafana...
├── llm/               # LLMFactory — provider selection and fallback chain
├── core/              # Config, structured logging, TraceMiddleware, audit
├── security/          # RBAC roles, JWT auth, user store
├── incident/          # Approval workflow, post-mortem, war room state
├── cost/              # AWS Cost Explorer, price estimation
├── monitoring/        # Background anomaly detection loop
├── chat/              # AI chat intelligence, session memory
└── tools/             # LangChain-style K8s/AWS/GitLab tool wrappers

nginx/nginx.conf        # Reverse proxy: TLS, rate limiting, security headers
Dockerfile              # Multi-stage build (non-root production user)
docker-compose.yml      # App + Nginx + Redis with health checks
```

---

## License

MIT
