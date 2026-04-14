# NsOps — AI DevOps Platform

**Your AI-powered DevOps command center.** Connect your AWS, Kubernetes, GitHub, and Slack — then ask the AI anything, debug incidents automatically, and manage your entire infrastructure from one place.

---

## What can I do with it?

| I want to… | Where to go |
|---|---|
| Ask about my infrastructure in plain English | **AI Assistant** — type anything |
| Debug a crashing K8s pod automatically | **AI Agents** → Debug with AI (K8s Pod) |
| Run full AI analysis on an incident | **Incidents** → Run Pipeline or Debug with AI |
| See my AWS costs breakdown | **Cost Analysis** |
| Monitor live infra status | **Infrastructure** or **Dashboard** |
| Set up a Slack war room for an incident | **War Room** or the Incidents pipeline |
| Approve or reject AI-suggested actions | **Approvals** |
| Review a GitHub PR with AI | **GitHub** → PR Review |
| View past incidents and post-mortems | **Incidents** history |

---

## Quick Start (after login)

**Step 1 — Connect your tools**
Go to **Integrations** and add credentials for AWS, GitHub, K8s, Slack, Jira. All integrations are optional — the platform works with whatever you have.

**Step 2 — Try the AI Chat**
Go to **AI Assistant** and type:
```
check my infra
```
The AI will pull live data from all connected integrations and give you a full status report.

**Step 3 — Debug an incident**
Go to **AI Agents** → fill in a namespace and pod name → click **Run AI Debug**.
The 5-agent system collects logs, events, and AWS data, then tells you the root cause and fix.

**Step 4 — Run the incident pipeline**
Go to **Incidents** → describe what's wrong → click **Run Pipeline**.
The AI collects all context, identifies root cause, and can automatically create a Jira ticket, open a Slack war room, and page on-call.

**Step 5 — Check your AWS costs**
Go to **Cost Analysis** → see live spend by service, by account, and trend charts.
Or just ask the AI: *"how much am I spending on RDS this month?"*

---

## Example AI Chat prompts

```
check my infra
list my EC2 instances
are there any firing CloudWatch alarms?
restart the payment-service deployment in production
check github repos
how much am I spending on AWS this month?
why is my pod crashing in namespace production?
list failed GitLab pipelines
check kubernetes pod logs for api-server
```

---

---

## What It Does

| Capability | Description |
|---|---|
| 🤖 **Multi-Agent Incident Pipeline** | LangGraph-orchestrated 5-agent workflow: Planner → Gather Data → Debugger → Executor → Reporter — with RBAC, dry-run mode, and ChromaDB memory recall |
| 🐛 **Debug with AI** | One-click AI debugging from Incidents or the Agents page: K8s pod analysis via `/debug-pod` (LangGraph) or full general incident pipeline (AWS + K8s + GitHub) |
| 🎬 **Real-Time Pipeline Streaming** | Animated 5-stage pipeline UI — watch AI work step-by-step: Context → Analysis → Plan → Execute → Complete |
| 🔍 **Pre-Deployment Assessment** | Before any deploy, the AI assesses cluster state, active alarms, and past incidents → go / no-go decision |
| 🎫 **Jira → Auto PR** | When a Jira change-request ticket is created, the AI reads it and opens a GitHub PR with file patches |
| ☁️ **AWS Observability** | Read-only collection across EC2, ECS, Lambda, RDS, ALB, CloudWatch, CloudTrail, S3, SQS, DynamoDB, Route53, SNS |
| ☸️ **Kubernetes Operations** | Health checks, rolling restarts, scale deployments, pod logs, unhealthy pod detection |
| 💰 **Smart Cost Analysis** | Live AWS spend (Cost Explorer), interactive Chart.js bar and trend charts, multi-account AWS Organizations view, on-demand price estimation, Terraform plan cost analysis |
| 📈 **Predictive Scaling** | Analyse CloudWatch metric trends and predict scaling needs before a breach occurs |
| 🔎 **AI PR Review** | Reviews GitHub PRs for security issues, infra concerns, and code quality |
| 🔐 **JWT Auth + RBAC** | JWT-based authentication with role-based access control (admin / developer / viewer) enforced on every endpoint |
| 🧠 **ChromaDB Memory** | All incidents stored in vector DB; similar past incidents feed future planning decisions |
| 🔁 **Continuous Monitoring** | Background loop polls K8s/AWS for anomalies and auto-triggers the pipeline |
| 🔀 **Multi-LLM Support** | Claude → Groq → Ollama — automatic provider detection and fallback chain with live status indicator |
| 💬 **Slack War Room** | Dedicated incident channel auto-created with full AI analysis; redesigned 2-column command-center UI with quick prompts and incident sidebar |
| 🏥 **Post-Mortem Reports** | AI-generated company-grade post-mortem documents enriched from vector memory |
| 🌐 **Nginx + TLS** | Production-grade reverse proxy with rate limiting, HTTPS redirect, security headers, WebSocket support |
| 💻 **VS Code Integration** | Live incident feed, file open, line highlights, terminal commands, Problems panel — all streamed to your IDE |
| 📧 **Email Notifications** | HTML emails on approval requests and incident completion with risk colour-coding |
| 📊 **Grafana Plugin** | Read-only health snapshots — firing alerts, datasource status, recent annotations |
| 📱 **Mobile Responsive** | Full mobile support — hamburger menu, collapsible sidebar, optimised layouts for phone and tablet |
| 🧭 **Onboarding Wizard** | First-time setup wizard guides you through connecting AWS, GitHub, Slack, and Jira |

---

## Architecture

### 1. LangGraph K8s Debugging Workflow — `POST /debug-pod`

```
  Input: { namespace, pod_name, dry_run, auto_fix }
         │
         ▼
  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
  │   Planner   │──▶│  Gather Data │──▶│   Debugger  │
  │             │   │              │   │             │
  │ Recall mem  │   │ Pod details  │   │ LLM reasons │
  │ Validate    │   │ Logs (200)   │   │ failure_type│
  │ Plan steps  │   │ K8s events   │   │ root_cause  │
  └─────────────┘   │ Resources    │   │ severity    │
                    └──────────────┘   └──────┬──────┘
                                              │
                          ┌───────────────────┴──────────────────┐
                          │ auto_fix=true?                        │
                          ▼                                       ▼
                   ┌────────────┐                         ┌──────────────┐
                   │  Executor  │────────────────────────▶│   Reporter   │
                   │ dry_run    │                         │ Markdown     │
                   │ RBAC check │                         │ Memory store │
                   │ restart/   │                         │ Slack alert  │
                   │ scale pod  │                         │ (critical)   │
                   └────────────┘                         └──────────────┘
```

### 2. General Incident Pipeline — `POST /incidents/run`

```
  Input: { incident_id, description, severity, aws_cfg, k8s_cfg }
         │
         ├──▶ _collect_aws()    ─┐
         ├──▶ _collect_k8s()    ─┼─ parallel (ThreadPoolExecutor)
         └──▶ _collect_github() ─┘
                    │
                    ▼
         AI Synthesis (Groq/Claude)
           └── summary, root_cause, findings, actions_to_take
                    │
                    ▼
         Execute Actions
           ├── k8s_restart / k8s_scale  (requires auto_remediate=true)
           ├── github_pr  (AI-generated file patches)
           ├── jira_ticket
           ├── slack_warroom  (dedicated incident channel)
           └── opsgenie_alert
                    │
                    ▼
         ChromaDB Memory  →  future incident recall
```

### 3. Full Stack

```
  Browser ──▶ Nginx (TLS + rate limiting)
                │
                ▼
         FastAPI (4 workers)
           ├── /debug-pod          LangGraph 5-agent K8s debugger
           ├── /agent/*            Plan · Execute · Observe · Workflows
           ├── /incidents/*        General incident pipeline
           ├── /chat/stream        Groq-first AI chat (SSE)
           ├── /aws/*  /k8s/*      Infra read + ops
           ├── JWT + RBAC          viewer / developer / operator / admin
           ├── SQLite              Per-session chat history
           └── ChromaDB            Incident vector memory
```

### Core Design Principles

| Layer | Responsibility |
|---|---|
| **Agents** | Specialised units: Planner, Debugger, Executor, Observer, Reporter |
| **LangGraph** | `StateGraph` controls flow, branching, conditional edges |
| **LLM** | Reasoning only — Debugger analysis, Planner decomposition |
| **Tools** | `KubernetesTool`, `AWSTool`, `GitLabTool` wrap all infra calls |
| **RBAC** | `require_operator` gate on destructive actions; dry-run mode |
| **Memory** | ChromaDB stores past incidents; Planner recalls similar events |

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
│   ├── planner.py          # PlannerAgent — LLM task decomposition into steps
│   ├── debugger.py         # DebuggerAgent — failure type detection + root cause analysis
│   ├── executor.py         # ExecutorAgent — RBAC-gated action execution + dry-run mode
│   ├── observer.py         # ObserverAgent — routes k8s/gitlab/prometheus/manual events
│   ├── reporter.py         # ReporterAgent — Markdown report + Slack notification
│   └── incident_pipeline.py  # General incident pipeline (AWS + K8s + GitHub + actions)
│
├── tools/
│   ├── kubernetes.py       # KubernetesTool — pods, logs, events, restart, scale
│   ├── aws.py              # AWSTool — EC2, CloudWatch, Secrets Manager, Cost
│   └── gitlab.py           # GitLabTool — pipeline logs, retry, failed jobs
│
├── workflows/
│   └── incident_workflow.py  # LangGraph StateGraph: Planner→Gather→Debugger→Executor→Reporter
│
├── routes/
│   └── agentic.py          # POST /debug-pod, /agent/plan, /agent/execute, /agent/observe, /agent/workflows
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
│   ├── aws_checker.py       # EC2, ECS, RDS, Lambda, ALB, DynamoDB, Route53, SNS, SQS
│   ├── k8s_checker.py       # Pods, deployments, nodes, statefulsets, daemonsets
│   ├── linux_checker.py     # CPU, memory, swap, disk
│   └── grafana_checker.py   # Firing alerts, datasources, annotations
│
├── integrations/
│   ├── vscode.py            # VS Code IDE bridge — open files, highlight, notify, terminal
│   └── email.py             # HTML email notifications for approvals and completions
│
├── vscode-extension/
│   ├── extension.js         # VS Code extension — local HTTP server on port 6789
│   └── package.json         # Extension manifest
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

### VS Code

| Variable | Default | Description |
|---|---|---|
| `VSCODE_BRIDGE_URL` | `http://127.0.0.1:6789` | URL of the NsOps VS Code extension server |
| `VSCODE_BRIDGE_TIMEOUT` | `5` | HTTP timeout in seconds for VS Code calls |

### Email Notifications

| Variable | Required for | Description |
|---|---|---|
| `SMTP_HOST` | Email | SMTP server hostname |
| `SMTP_PORT` | Email | SMTP port (587 for TLS, 465 for SSL) |
| `SMTP_USER` | Email | SMTP login username |
| `SMTP_PASSWORD` | Email | SMTP password or app password |
| `ALERT_EMAIL_FROM` | Email | From address (defaults to `SMTP_USER`) |
| `ALERT_EMAIL_TO` | Email | Comma-separated recipient addresses |

All integrations degrade gracefully — missing credentials return a structured error rather than crashing the platform.

---

## How to Use

### 1. AI Chat Assistant
Go to **Intelligence → AI Assistant**. Type naturally — the AI reads your configured integrations and answers or takes action.

**Example prompts:**
```
check my infra                          → full AWS + K8s + GitHub status report
list my EC2 instances                   → shows all EC2 with state, type, region
restart the payment-service deployment  → asks for confirmation, then restarts
check github repos                      → lists repos + recent commits
how much am I spending on AWS this month → cost breakdown
```

The AI selects the right tool automatically. It uses **Groq (Llama 3.3-70B)** by default — switch to Claude or GPT-4 from the provider selector. Chat history persists across page refreshes.

---

### 2. Run an Incident Pipeline
Go to **Operations → Incidents**.

1. Fill in **Description** — describe what's wrong (e.g. "API pods crash-looping in prod")
2. Set **Severity** and optional **Lookback Hours**
3. Click **Run Pipeline** — the AI collects AWS + K8s + GitHub context in parallel, synthesises root cause, and executes recommended actions

Optional toggles:
- **Auto-Remediate** — let the AI execute K8s restarts/scaling automatically (admin only)
- **Dry Run** — see what the AI *would* do without executing anything
- **Create Slack Channel** — auto-create a war room channel

---

### 3. Debug with AI (LangGraph 5-Agent Workflow)

**From Incidents page:** Fill in the description → click **Debug with AI** (purple button). This navigates to the Agents page and runs a full AI debug session.

**From Intelligence → AI Agents directly:**

**K8s Pod Debugging:**
1. Select target: **Kubernetes Pod**
2. Enter namespace (e.g. `production`) and pod name (e.g. `payment-api-abc123`)
3. Check **Dry Run** to preview, or uncheck to allow real fixes
4. Click **Run AI Debug**

The 5-agent LangGraph workflow runs:
- **Planner** — checks memory for similar past incidents
- **Gather Data** — fetches pod details, last 200 log lines, K8s events
- **Debugger** — LLM identifies failure type (`CrashLoopBackOff`, `OOMKilled`, etc.), root cause, severity
- **Executor** — optionally restarts/scales the pod (respects dry-run + RBAC)
- **Reporter** — generates structured Markdown report, stores to memory, alerts Slack if critical

**General Incident Debugging:**
1. Select target: **General Incident (AWS + K8s + GitHub)**
2. Describe the issue, set severity
3. Click **Run AI Debug** — runs the full multi-source incident pipeline

---

### 4. Observer — Trigger Events
In **AI Agents → Observer**, send events to route through the agent system:

| Event Type | When to use |
|---|---|
| `manual_debug` | Manually trigger a debug session for any component |
| `k8s_alert` | Simulate a Kubernetes alert (e.g. from Alertmanager) |
| `prometheus_alert` | Simulate a Prometheus firing alert |
| `gitlab_pipeline` | Simulate a failed GitLab pipeline event |

**Example payload** for `k8s_alert`:
```json
{"namespace": "production", "pod_name": "api-server-abc123", "reason": "OOMKilled"}
```

---

### 5. API Usage

All endpoints require a JWT token (or `X-User` header in dev mode):

```bash
# Get token
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=yourpassword"

# Debug a K8s pod
curl -X POST http://localhost:8000/debug-pod \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"namespace":"default","pod_name":"my-app-abc123","dry_run":true,"auto_fix":false}'

# Run general incident pipeline
curl -X POST http://localhost:8000/incidents/run \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"incident_id":"INC-001","description":"High CPU on payment service","severity":"high","auto_remediate":false}'

# List available workflows
curl http://localhost:8000/agent/workflows \
  -H "Authorization: Bearer <token>"

# AI Chat
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message":"check my infra","session_id":"sess-123"}'
```

---

## Dashboard

The main dashboard gives a full-platform overview at a glance.

### Stat Cards (clickable)

| Card | Navigates to |
|---|---|
| **Active Incidents** | Incidents list |
| **Pending Approvals** | Approvals queue |
| **AWS Alarms** | AWS monitoring |
| **K8s Clusters** | Infrastructure → Kubernetes tab |

### Dashboard Sections

| Section | Description |
|---|---|
| **Integration Health** | Live grid showing connection status for all configured integrations (AWS, GitHub, Slack, Jira, Grafana, K8s) |
| **Recent Activity** | Last 10 platform events — incidents run, approvals, pipeline completions |
| **Recent Incidents** | Latest 5 incidents with status, risk level, and one-click open |
| **Cost Overview** | MTD spend, last month comparison, and a Chart.js trend chart |

---

## Cost Analysis

The Cost Analysis tab has five views:

| Tab | Description |
|---|---|
| **Overview** | MTD spend, last month total, forecast, MoM trend, interactive Chart.js horizontal bar chart (top services) + 6-month line trend chart |
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

## Unified LangGraph Workflow

Both `POST /debug-pod` and `POST /incidents/run` now use the **same** LangGraph 5-agent workflow (`app/workflows/unified_workflow.py`). The input determines what gets collected and executed.

### The 5 Agents

| Agent | Role |
|---|---|
| **Planner** | Validates input, searches ChromaDB for similar past incidents, plans debug steps |
| **Gather All** | Collects K8s + AWS + GitHub data **in parallel** (ThreadPoolExecutor, 8s timeout) |
| **Debugger** | LLM analyses all gathered data → `failure_type`, `root_cause`, `findings`, `actions_to_take` |
| **Executor** | Executes K8s actions (restart/scale) + notifications (Jira, Slack, OpsGenie, GitHub PR) |
| **Reporter** | Generates Markdown report, stores to ChromaDB memory, Slack alert for critical severity |

### What gets collected per source

| Source | What is fetched |
|---|---|
| **Kubernetes** | Pod describe, last 200 log lines, K8s events (filtered to pod), resource usage |
| **AWS** | EC2 instances + state, CloudWatch alarms, logs (if `aws_cfg` provided) |
| **GitHub** | Recent commits, open PRs |

### Request — K8s Pod Debug

```json
POST /debug-pod
{
  "namespace": "production",
  "pod_name":  "payment-api-abc123",
  "dry_run":   true,
  "auto_fix":  false
}
```

`dry_run=true` (default) — safe in production, shows what the AI would do without executing.
`auto_fix=true` — executor restarts/scales the pod. Requires operator or admin role.

### Request — General Incident

```json
POST /incidents/run
{
  "incident_id":    "INC-001",
  "description":   "High CPU on payment service, API latency spiking",
  "severity":      "high",
  "auto_remediate": false,
  "k8s":           {"namespace": "production"},
  "aws":           {"resource_type": "ecs", "resource_id": "prod-cluster"},
  "hours":          2
}
```

`auto_remediate=true` — executor runs Jira ticket, Slack war room, OpsGenie alert, GitHub PR.

### Response fields (both endpoints)

| Field | Description |
|---|---|
| `failure_type` | `CrashLoopBackOff` / `OOMKilled` / `HighCPU` / `General` / etc. |
| `severity_ai` | AI-assessed severity: `critical` / `high` / `medium` / `low` |
| `root_cause` | 2–3 sentence explanation |
| `fix_suggestion` | Specific actionable fix with commands |
| `findings` | List of specific observations from the data |
| `actions_taken` | Each action's result (skipped / dry-run / success / failed) |
| `report` | Full Markdown incident report |
| `steps_taken` | Every step each agent took |
| `elapsed_s` | Total wall-clock time |

---

## War Room Command Center

The War Room is a 2-column incident command interface:

```
┌────────────────────────────────┬──────────────────┐
│  Live Chat (AI + team)         │  Incident Info   │
│                                │  ─────────────   │
│  [ message bubbles ]           │  ID / Severity   │
│                                │  Status / Time   │
│  [ input box ]                 │  Participants    │
│                                │  ─────────────   │
│                                │  Quick Prompts   │
│                                │  ─────────────   │
│                                │  Resolve         │
└────────────────────────────────┴──────────────────┘
```

**Quick Prompts** (one-click questions sent to the AI):
- What's the root cause?
- Show me the timeline
- What should we do next?
- Check AWS alarms
- Any recent deployments?
- Page the on-call engineer

**Resolve** button closes the war room and triggers post-mortem generation.

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

## Onboarding Wizard

On first login, a step-by-step wizard guides new users through integration setup:

| Step | What it configures |
|---|---|
| **1. AWS** | Region, Access Key ID, Secret Key |
| **2. GitHub** | Personal access token, default repository |
| **3. Slack** | Bot token, default channel |
| **4. Jira** | Instance URL, user email, API token, project key |

Onboarding state is stored in `localStorage` (`nsops_onboarded`) and shown exactly once per browser. It can be re-triggered from the UI if needed. All credentials are written via the `/secrets` API endpoint.

---

## Mobile Support

The dashboard is fully responsive:

| Breakpoint | Behaviour |
|---|---|
| `> 768px` | Full desktop layout with sidebar visible |
| `≤ 768px` | Sidebar collapses off-screen; hamburger button appears top-left |
| `≤ 480px` | Single-column cards, full-width stat cards, compact navigation |

Tap the **☰** hamburger button to slide the sidebar in. Tapping the overlay outside the sidebar closes it.

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
| `POST` | `/chat/stream` | viewer | Streaming SSE chat — used by the UI (Groq-first, persists history) |
| `GET` | `/chat/history/{session_id}` | viewer | Load conversation history for a session |
| `GET` | `/chat/sessions` | viewer | List all chat sessions |
| `GET` | `/chat/action_count` | viewer | Number of actions executed this session |

### Unified LangGraph Workflow (AI Agents)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/debug-pod` | viewer | Unified 5-agent workflow — K8s pod debug or general incident |
| `POST` | `/incidents/run` | developer | Same unified workflow — AWS+K8s+GitHub+actions |
| `POST` | `/agent/plan` | viewer | Planner agent — decompose a task into steps |
| `POST` | `/agent/execute` | operator | Executor agent — run a single action with RBAC + dry-run |
| `POST` | `/agent/observe` | viewer | Observer — route an event to the right workflow |
| `GET` | `/agent/workflows` | viewer | List available workflows and agents |

### Incident Pipelines (legacy — still work)

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/incident/run` | developer | v1 monolithic pipeline |
| `POST` | `/incidents/run/async` | developer | Fire-and-forget — returns job ID immediately |

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

### VS Code

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/vscode/status` | viewer | Check if VS Code extension is reachable |
| `POST` | `/vscode/notify` | viewer | Show a popup notification in VS Code |
| `POST` | `/vscode/open` | developer | Open a file, optionally jump to a line |
| `POST` | `/vscode/highlight` | developer | Yellow-highlight lines in a file |
| `POST` | `/vscode/terminal` | developer | Run a shell command in VS Code terminal |
| `POST` | `/vscode/diff` | developer | Open a two-panel diff view |
| `POST` | `/vscode/problems` | developer | Inject entries into the Problems panel |
| `POST` | `/vscode/clear-highlights` | developer | Remove all NsOps line decorations |
| `POST` | `/vscode/output` | viewer | Write a message to the NsOps output channel |
| `POST` | `/vscode/incident/{id}` | developer | Surface an incident — notify + open file + highlight |

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

The monitor (`app/monitoring/loop.py`) polls K8s and AWS for anomalies. Detectors cover:

| Detector | What it catches |
|---|---|
| EC2 | Stopped / terminated instances with auto-resolve when instance returns to running |
| ECS | Services with running count < desired count |
| CloudWatch | Alarms in ALARM state, Lambda error spikes |
| RDS | Instance events |
| SQS | Queues with messages in flight |
| K8s Nodes | NotReady nodes |
| K8s Deployments | 0 available pods |
| Grafana | Firing alert rules |

When anomalies are found it triggers the v2 pipeline with `auto_remediate=AUTO_REMEDIATE_ON_MONITOR`. All alerts and pipeline results are streamed live to the VS Code NsOps output channel if the extension is running.

---

## VS Code Integration

The platform ships with a VS Code extension that creates a two-way bridge between the AI pipeline and your editor.

### Install

```bash
cd vscode-extension
npm install -g @vscode/vsce        # install packager once
vsce package                        # builds nsops-vscode-1.0.0.vsix
code --install-extension nsops-vscode-1.0.0.vsix
```

Or install the VS Code CLI directly if `code` is not in PATH:
```bash
/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code \
  --install-extension vscode-extension/nsops-vscode-1.0.0.vsix
```

The extension auto-starts when VS Code opens. You will see **`⚡ NsOps :6789`** in the status bar.

### What streams to your editor

| Event | What appears in VS Code |
|---|---|
| Alert detected | `🚨 ALERT [type] resource — description` + Output panel pops open |
| Pipeline starts | `⚙️ PIPELINE [incident-id] Starting AI pipeline for: ...` |
| Pipeline completed | `✅ PIPELINE DONE status=completed risk=medium actions=1 root_cause=...` |
| Awaiting approval | `⏳ PIPELINE DONE` + warning notification popup |
| Pipeline failed | `❌ PIPELINE DONE` + error notification popup |
| Alert resolved | `✅ RESOLVED [type] resource` |

View the live feed: `Cmd+Shift+U` → select **NsOps** from the Output panel dropdown.

### Dashboard VS Code section

Navigate to **VS Code** in the sidebar to:
- See live connection status (green / amber / red)
- Send notifications, open files, highlight lines, run terminal commands, inject Problems panel entries — all from the browser UI

### API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/vscode/status` | viewer | Check if VS Code extension is reachable |
| `POST` | `/vscode/notify` | viewer | Show a popup notification in VS Code |
| `POST` | `/vscode/open` | developer | Open a file, optionally jump to a line |
| `POST` | `/vscode/highlight` | developer | Yellow-highlight lines in a file |
| `POST` | `/vscode/terminal` | developer | Run a shell command in VS Code terminal |
| `POST` | `/vscode/diff` | developer | Open a two-panel diff view |
| `POST` | `/vscode/problems` | developer | Inject entries into the Problems panel |
| `POST` | `/vscode/clear-highlights` | developer | Remove all NsOps line decorations |
| `POST` | `/vscode/output` | viewer | Write a message to the NsOps output channel |
| `POST` | `/vscode/incident/{id}` | developer | Surface an incident — notify + open file + highlight |

### Use from Python

```python
from app.integrations.vscode import (
    notify, open_file, highlight_lines,
    run_in_terminal, inject_problems, open_incident_context
)

# Popup notification
notify("Deployment failed on prod", level="error")

# Open file and jump to a line
open_file("/app/workers/processor.py", line=88)

# Highlight problem lines with hover messages
highlight_lines("/app/workers/processor.py", [
    {"line": 88, "message": "OOMKilled here"}
])

# Run a command in the VS Code integrated terminal
run_in_terminal("kubectl rollout restart deployment/api -n prod")

# Surface a full incident (notify + open + highlight + output)
open_incident_context(
    incident_id="INC-001",
    root_cause="OOM crash in worker",
    file_path="/app/workers/processor.py",
    problem_line=88,
)
```

### Extension configuration

| Setting | Default | Description |
|---|---|---|
| `nsops.serverPort` | `6789` | Local HTTP server port |
| `nsops.autoStart` | `true` | Start server automatically when VS Code opens |

Override in VS Code settings (`Cmd+,` → search "nsops").

---

## Email Notifications

Set these in `.env` to receive HTML emails on incident events:

```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_TO=oncall@yourcompany.com,team@yourcompany.com
```

Two email types are sent automatically:

| Trigger | Email sent |
|---|---|
| Pipeline reaches `awaiting_approval` | **Approval Required** — incident ID, description, risk level, confidence, proposed actions, approval reason |
| Pipeline completes or fails | **Incident Completed** — status, root cause, summary, actions executed, validation result |

---

## Plugins

Read-only health snapshot plugins live in `app/plugins/`. Each returns a consistent contract:

```python
{
    "status": "healthy" | "degraded" | "unavailable" | "error",
    "details": { ... }
}
```

| Plugin | Function | What it checks |
|---|---|---|
| `aws_checker.py` | `check_aws_infrastructure()` | EC2, ECS, RDS, Lambda, ALB, CloudWatch, S3, SQS, DynamoDB, Route53, SNS |
| `k8s_checker.py` | `check_k8s_cluster()` | Pods, deployments, nodes, statefulsets, daemonsets — namespace or cluster-wide |
| `linux_checker.py` | `check_linux_health()` | CPU, memory, swap, disk — via psutil |
| `grafana_checker.py` | `check_grafana()` | Firing alerts, datasource count, recent annotations |
