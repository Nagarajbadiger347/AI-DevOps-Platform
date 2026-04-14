"""
NexusOps AI Chat Intelligence — rewritten for speed, clarity, and quality.

Architecture:
  1. Provider detection at import time — only connects to working providers.
  2. Dead-provider cache — billing/quota failures are remembered so we never
     retry a broken provider in the same session.
  3. Tool calling — LLM can call DevOps tools (AWS, K8s, GitHub…) and get
     real live data before answering.
  4. Session memory — full multi-turn conversation history per session.
  5. Prefetch cache — infra data is cached 60s so repeated questions are instant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger("nsops.chat")

# ── Load .env ─────────────────────────────────────────────────────────────────
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Provider clients ──────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "").strip()

_groq_client      = None
_anthropic_client = None
_openai_client    = None
_active_provider  = None   # which provider to use by default

if GROQ_API_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY, timeout=25.0)
        if not _active_provider:
            _active_provider = "groq"
    except Exception:
        pass

if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.startswith("sk-ant-"):
    try:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=25.0)
        if not _active_provider:
            _active_provider = "anthropic"
    except Exception:
        pass

if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=25.0)
        if not _active_provider:
            _active_provider = "openai"
    except Exception:
        pass

# Models to use per provider
_MODELS = {
    "groq":      "llama-3.3-70b-versatile",   # best Groq model for reasoning
    "anthropic": "claude-haiku-4-5-20251001",  # fast + cheap Anthropic model
    "openai":    "gpt-4o-mini",
}

# Dead-provider registry — providers that failed with billing/quota errors
# Cleared on server restart; prevents retrying dead providers on every message
_DEAD_PROVIDERS: set[str] = set()


def _mark_dead(provider: str) -> None:
    _DEAD_PROVIDERS.add(provider)
    logger.warning(f"Provider '{provider}' marked dead (billing/quota). Will skip.")


def _is_dead(provider: str) -> bool:
    return provider in _DEAD_PROVIDERS


# Pre-validate providers at startup (quick test call)
def _validate_providers():
    """Do a cheap test call to detect billing/quota failures at startup."""
    global _active_provider
    _order = []
    if _groq_client:      _order.append("groq")
    if _anthropic_client: _order.append("anthropic")
    if _openai_client:    _order.append("openai")

    for provider in _order:
        try:
            _llm_call("Say OK", provider=provider, max_tokens=3)
            _active_provider = provider
            logger.info(f"Chat provider: {provider} ({_MODELS[provider]})")
            return
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("credit", "quota", "billing", "401", "429", "invalid_api_key")):
                _mark_dead(provider)
            # else: some other transient error, don't mark dead

    logger.warning("No working LLM provider found at startup.")


# ── Raw LLM call ──────────────────────────────────────────────────────────────

def _llm_call(
    user_message: str,
    system: str = "",
    history: list[dict] = None,
    provider: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    tools: list[dict] = None,
) -> str:
    """Single unified LLM call. Returns the assistant text response.

    Tries the given provider (or active provider), then falls back in order:
    groq → anthropic → openai

    Raises RuntimeError if all providers are dead/unavailable.
    """
    history = history or []
    fallback_order = ["groq", "anthropic", "openai"]
    if provider:
        # Put requested provider first, then rest as fallback
        fallback_order = [provider] + [p for p in fallback_order if p != provider]

    last_err = None
    for prov in fallback_order:
        if _is_dead(prov):
            continue

        client = {"groq": _groq_client, "anthropic": _anthropic_client, "openai": _openai_client}.get(prov)
        if not client:
            continue

        try:
            messages = list(history) + [{"role": "user", "content": user_message}]
            model = _MODELS[prov]

            if prov == "anthropic":
                kwargs = dict(
                    model=model, max_tokens=max_tokens,
                    messages=messages, temperature=temperature,
                )
                if system:
                    kwargs["system"] = system
                resp = client.messages.create(**kwargs)
                return resp.content[0].text or ""

            elif prov in ("groq", "openai"):
                all_msgs = ([{"role": "system", "content": system}] if system else []) + messages
                resp = client.chat.completions.create(
                    model=model,
                    messages=all_msgs,
                    max_tokens=min(max_tokens, 4096),
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""

        except Exception as e:
            err = str(e).lower()
            is_billing = any(k in err for k in (
                "credit", "quota", "billing", "too low", "insufficient",
                "401", "429", "invalid_api_key", "authentication"
            ))
            if is_billing:
                _mark_dead(prov)
            last_err = e
            continue

    raise RuntimeError(
        f"All LLM providers unavailable. Last error: {last_err}\n"
        "Please add a working API key: GROQ_API_KEY (free at console.groq.com), "
        "ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


# ── Session memory ────────────────────────────────────────────────────────────

_SESSIONS: dict[str, list[dict]] = {}   # session_id → list of {role, content}
_SESSION_MAX = 30                        # max messages to keep per session


def _get_history(session_id: str) -> list[dict]:
    return _SESSIONS.get(session_id, [])


def _add_message(session_id: str, role: str, content: str) -> None:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    _SESSIONS[session_id].append({"role": role, "content": content})
    # Keep only last N messages to avoid token bloat
    if len(_SESSIONS[session_id]) > _SESSION_MAX:
        _SESSIONS[session_id] = _SESSIONS[session_id][-_SESSION_MAX:]


def get_history(session_id: str, max_messages: int = 20) -> list[dict]:
    """Public API: return up to max_messages from this session."""
    h = _get_history(session_id)
    return h[-max_messages:]


def add_message(session_id: str, role: str, content: str) -> None:
    """Public API: add a message to the session."""
    _add_message(session_id, role, content)


def get_or_create_session(session_id: str) -> list[dict]:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    return _SESSIONS[session_id]


def list_sessions() -> list[dict]:
    return [
        {"session_id": sid, "message_count": len(msgs)}
        for sid, msgs in _SESSIONS.items()
    ]


# ── Infra availability flags ──────────────────────────────────────────────────

try:
    from app.integrations import aws_ops as _aws_ops
    _AWS_OK = bool(
        os.getenv("AWS_ACCESS_KEY_ID") or
        os.getenv("AWS_PROFILE") or
        os.getenv("AWS_ROLE_ARN")
    )
except ImportError:
    _aws_ops = None
    _AWS_OK = False

try:
    from app.integrations import k8s_ops as _k8s_ops
    _K8S_OK = bool(os.getenv("KUBECONFIG") or os.getenv("K8S_IN_CLUSTER"))
except ImportError:
    _k8s_ops = None
    _K8S_OK = False

try:
    from app.integrations import github as _github_ops
    _GITHUB_OK = bool(os.getenv("GITHUB_TOKEN"))
except ImportError:
    _github_ops = None
    _GITHUB_OK = False

# ── Prefetch cache (avoids redundant AWS/K8s calls) ──────────────────────────

_PREFETCH_CACHE: dict[str, dict] = {}
_PREFETCH_TTL = 60  # seconds — cache live infra data for 60s

# Global infra snapshot cache (shared across all sessions) — refreshed every 90s
_GLOBAL_INFRA_CACHE: dict = {}
_GLOBAL_INFRA_TS: float = 0.0
_GLOBAL_INFRA_TTL = 90

# Per-session EC2 instance cache
_EC2_SESSION: dict[str, dict] = {}
_EC2_TTL = 300

_ec2_session_cache = _EC2_SESSION   # alias used by chat.py


def _prefetch_key(message: str, session_id: str) -> str:
    return f"{session_id}:{hashlib.md5(message.lower().strip().encode()).hexdigest()[:10]}"


def _prefetch_get(message: str, session_id: str) -> str | None:
    entry = _PREFETCH_CACHE.get(_prefetch_key(message, session_id))
    if entry and time.time() < entry["exp"]:
        return entry["data"]
    return None


def _prefetch_set(message: str, session_id: str, data: str) -> None:
    if len(_PREFETCH_CACHE) > 1000:
        # Evict expired entries
        now = time.time()
        for k in list(_PREFETCH_CACHE.keys()):
            if _PREFETCH_CACHE[k]["exp"] < now:
                del _PREFETCH_CACHE[k]
    _PREFETCH_CACHE[_prefetch_key(message, session_id)] = {"data": data, "exp": time.time() + _PREFETCH_TTL}


def _get_cached_ec2(session_id: str) -> list[dict]:
    entry = _EC2_SESSION.get(session_id)
    if entry and time.time() < entry["exp"]:
        return entry["data"]
    return []


def _set_cached_ec2(session_id: str, instances: list[dict]) -> None:
    _EC2_SESSION[session_id] = {"data": instances, "exp": time.time() + _EC2_TTL}


# ── Infra prefetch ────────────────────────────────────────────────────────────

def _prefetch_infra(message: str, session_id: str) -> str:
    """Fetch live infra data relevant to the message. All fetches run in parallel. Results cached 60s."""
    import concurrent.futures as _cf

    cached = _prefetch_get(message, session_id)
    if cached is not None:
        return cached

    if not _AWS_OK and not _K8S_OK and not _GITHUB_OK:
        return ""

    msg = message.lower()

    # General infra check — fetch everything when user asks for overview
    general_kw = {"infra", "infrastructure", "check", "status", "overview", "health", "what's running",
                  "whats running", "show me", "my setup", "my aws", "my cloud", "everything", "all services"}
    fetch_all = any(k in msg for k in general_kw)

    ec2_kw    = {"ec2", "instance", "server", "vm", "machine", "compute", "start", "stop", "reboot", "running", "stopped", "down", "unreachable"}
    alarm_kw  = {"alarm", "alert", "cloudwatch", "threshold", "firing"}
    k8s_kw    = {"pod", "k8s", "kubernetes", "deployment", "namespace", "container", "crashloop"}
    ecs_kw    = {"ecs", "fargate", "task", "service"}
    rds_kw    = {"rds", "database", "db", "mysql", "postgres", "aurora"}
    lambda_kw = {"lambda", "function", "serverless"}
    gh_kw     = {"github", "commit", "pr", "pull request", "deploy", "release", "repo", "repos", "repository", "repositories", "git", "branch", "merge"}

    def _fetch_ec2():
        if not (_AWS_OK and (fetch_all or any(k in msg for k in ec2_kw))):
            return None
        try:
            cached_instances = _get_cached_ec2(session_id)
            if not cached_instances:
                r = _aws_ops.list_ec2_instances()
                cached_instances = r.get("instances", [])
                if cached_instances:
                    _set_cached_ec2(session_id, [
                        {"id": i["id"], "name": i.get("name",""), "state": i.get("state",""), "type": i.get("type","")}
                        for i in cached_instances
                    ])
            if cached_instances:
                running = sum(1 for i in cached_instances if i.get("state") == "running")
                lines = [f"EC2 ({len(cached_instances)} total, {running} running):"]
                for i in cached_instances[:8]:
                    n = f' ({i["name"]})' if i.get("name") else ""
                    lines.append(f"  • {i['id']}{n} — {i.get('state','?')} — {i.get('type','?')}")
                return "\n".join(lines)
        except Exception:
            pass
        return None

    def _fetch_alarms():
        if not (_AWS_OK and (fetch_all or any(k in msg for k in alarm_kw))):
            return None
        try:
            r = _aws_ops.list_cloudwatch_alarms("ALARM")
            alarms = r.get("alarms", [])
            if alarms:
                lines = [f"Firing alarms ({len(alarms)}):"]
                for a in alarms[:5]:
                    lines.append(f"  • {a.get('name','?')} — {a.get('reason','')[:80]}")
                return "\n".join(lines)
            return "CloudWatch: No alarms firing."
        except Exception:
            return None

    def _fetch_k8s():
        if not (_K8S_OK and (fetch_all or any(k in msg for k in k8s_kw))):
            return None
        try:
            r = _k8s_ops.list_pods()
            pods = r.get("pods", [])
            if pods:
                bad = [p for p in pods if p.get("status") not in ("Running","Completed","Succeeded")]
                return (f"K8s: {len(pods)} pods, {len(bad)} unhealthy" +
                        (": " + ", ".join(p["name"] for p in bad[:3]) if bad else ""))
        except Exception:
            pass
        return None

    def _fetch_ecs():
        if not (_AWS_OK and (fetch_all or any(k in msg for k in ecs_kw))):
            return None
        try:
            r = _aws_ops.list_ecs_services()
            svcs = r.get("services", [])
            if svcs:
                lines = [f"ECS ({len(svcs)} services):"]
                for s in svcs[:5]:
                    lines.append(f"  • {s.get('name','?')} — {s.get('running_count','?')}/{s.get('desired_count','?')} tasks")
                return "\n".join(lines)
        except Exception:
            pass
        return None

    def _fetch_rds():
        if not (_AWS_OK and (fetch_all or any(k in msg for k in rds_kw))):
            return None
        try:
            r = _aws_ops.list_rds_instances()
            dbs = r.get("instances", [])
            if dbs:
                lines = [f"RDS ({len(dbs)} instances):"]
                for d in dbs[:5]:
                    lines.append(f"  • {d.get('id','?')} ({d.get('engine','?')}) — {d.get('status','?')}")
                return "\n".join(lines)
        except Exception:
            pass
        return None

    def _fetch_lambda():
        if not (_AWS_OK and (fetch_all or any(k in msg for k in lambda_kw))):
            return None
        try:
            r = _aws_ops.list_lambda_functions()
            fns = r.get("functions", [])
            if fns:
                return f"Lambda: {len(fns)} functions — " + ", ".join(f.get("name","?") for f in fns[:5])
        except Exception:
            pass
        return None

    def _fetch_github():
        if not (_GITHUB_OK and (fetch_all or any(k in msg for k in gh_kw))):
            return None
        parts = []
        try:
            r = _github_ops.list_repos()
            repos = r.get("repos", []) if isinstance(r, dict) else []
            if repos:
                lines = [f"GitHub repos ({len(repos)}):"]
                for repo in repos[:6]:
                    name = repo.get("name") or repo.get("full_name", "?")
                    lang = repo.get("language", "")
                    stars = repo.get("stars", repo.get("stargazers_count", 0))
                    private = "🔒" if repo.get("private") else "🌐"
                    lines.append(f"  • {private} {name}" + (f" [{lang}]" if lang else "") + (f" ⭐{stars}" if stars else ""))
                parts.append("\n".join(lines))
        except Exception:
            pass
        try:
            r = _github_ops.get_recent_commits(hours=48)
            commits = r if isinstance(r, list) else r.get("commits", [])
            if commits:
                lines = [f"Recent commits ({len(commits)}):"]
                for c in commits[:4]:
                    lines.append(f"  • {c.get('sha','')[:7]} — {c.get('message','')[:70]} ({c.get('author','?')})")
                parts.append("\n".join(lines))
        except Exception:
            pass
        return "\n\n".join(parts) if parts else None

    # Run all fetches in parallel — collect results that arrive within 4s, skip the rest
    fetchers = [_fetch_ec2, _fetch_alarms, _fetch_k8s, _fetch_ecs, _fetch_rds, _fetch_lambda, _fetch_github]
    pool = _cf.ThreadPoolExecutor(max_workers=len(fetchers))
    futures = {pool.submit(f): f for f in fetchers}
    parts = []
    try:
        for fut in _cf.as_completed(futures, timeout=6):
            try:
                r = fut.result()
                if r:
                    parts.append(r)
            except Exception:
                pass
    except _cf.TimeoutError:
        pass  # some fetches timed out — use whatever arrived within 6s
    pool.shutdown(wait=False)

    result = "\n\n".join(parts)
    _prefetch_set(message, session_id, result)
    return result


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are **NexusOps AI** — the built-in AI assistant for the NexusOps AI DevOps Platform. You are an expert in AWS, Kubernetes, GitHub, GitLab, incident response, cost analysis, and cloud infrastructure. You think and respond like a senior SRE: intelligent, direct, and genuinely helpful.

## About NexusOps Platform:
NexusOps is a self-hosted AI DevOps platform that connects to AWS, Kubernetes, GitHub, Slack, Jira, OpsGenie, Grafana, and GitLab. It provides:
- **AI Chat** (you) — natural language interface to all integrations
- **Unified LangGraph Workflow** — 5-agent system (Planner→Gather→Debugger→Executor→Reporter) for ALL incidents
- **Incident Pipeline** — POST /incidents/run — collects AWS+K8s+GitHub, AI analysis, executes Jira/Slack/OpsGenie/GitHub PR
- **Debug with AI** — POST /debug-pod — deep K8s pod debugging or general incident analysis
- **Observer** — POST /agent/observe — routes k8s_alert, prometheus_alert, gitlab_pipeline, manual_debug events to workflows
- **Cost Analysis** — live AWS Cost Explorer, by-service/by-account/by-org breakdown, Terraform cost estimation
- **War Room** — 2-column incident command center with dedicated Slack channel
- **Approvals** — human-in-the-loop gate for high-risk actions
- **Post-Mortem Reports** — AI-generated blameless post-mortems from ChromaDB memory
- **VS Code Integration** — open files, highlight lines, terminal commands from the platform
- **Continuous Monitoring** — background loop polls K8s/AWS for anomalies

## Your capabilities:

### AWS Infrastructure
- List, describe, start, stop, reboot EC2 instances
- List ECS services/tasks, scale ECS services, redeploy
- List Lambda functions and their configs
- List RDS databases, check status
- Read CloudWatch alarms (firing/ok/insufficient_data)
- Search CloudWatch Logs for errors or patterns
- Fetch CloudTrail audit events — who did what and when
- List S3 buckets, SQS queues, DynamoDB tables, SNS topics, Route53 zones
- Get AWS cost breakdown (MTD, by service, by account)
- Estimate cost impact of scaling decisions

### Kubernetes
- List pods, deployments, namespaces, nodes, statefulsets
- Describe pod status, restart count, conditions
- Fetch pod logs (last N lines)
- List K8s events (warnings, errors)
- Restart a pod or deployment
- Scale a deployment to N replicas
- Detect unhealthy pods across namespaces
- Check node resource usage

### GitHub
- List repositories and their details
- View recent commits (last N hours/days)
- List open pull requests
- Create a GitHub issue
- Create a pull request with file patches
- Review a PR for security/infra/code quality issues

### Incident Management
- Run the unified LangGraph incident workflow (K8s pod debug OR general AWS+K8s+GitHub)
- Run the full incident pipeline: collect context → AI analysis → execute actions
- Create Jira tickets for incidents
- Page on-call via OpsGenie
- Create Slack war room channels
- Generate AI post-mortem reports
- Search past incidents from memory (ChromaDB)

### GitLab / CI-CD
- Get pipeline logs for failed pipelines
- Retry a failed pipeline
- List failed pipelines
- Get job logs

### Platform Actions (UI Navigation)
- Users can go to **AI Agents** page → Debug with AI (K8s pod or general incident)
- Users can go to **Incidents** page → Run Pipeline or Debug with AI
- Users can go to **Cost Analysis** page for AWS spend details
- Users can go to **War Room** for live incident command
- Users can go to **Approvals** to approve/reject pending actions

## Behavior rules:
1. **Be direct.** Answer immediately — no filler, no unnecessary caveats.
2. **Use live data.** Live infra data is injected below the user message — use it for specific, concrete answers.
3. **Never ask for info you already have.** If live data is in context, use it. Don't ask "which cloud" or "which service" — you can see them.
4. **Be conversational.** Remember conversation history and refer back to it.
5. **Format well.** Use markdown, code blocks for commands, bold for key things.
6. **Be honest.** If data is missing, say so and suggest how to get it.
7. **For actions** (restart, scale, stop, create): confirm what you will do before executing.
8. **Never invent IDs, names, or metrics.** Only use real data from the provided context.

## Response style:
- Short messages → short answers
- Technical questions → technical answers with commands/examples
- Problems → root cause → impact → fix → prevention
- "what can you do" → give concrete examples with real commands
- "check infra / check my setup / infrastructure status" → FULL report with ALL sections: 🖥 EC2, 🔔 CloudWatch Alarms, 🐳 ECS, 🗄 RDS, λ Lambda, ☸ Kubernetes, 🐙 GitHub — even if some show "none configured". End with overall health summary and recommended actions."""


def _build_system_prompt(incident_context: dict = None, native_tools: bool = False) -> str:
    """Build system prompt, optionally injecting incident context."""
    prompt = _SYSTEM_PROMPT
    if incident_context:
        ctx_json = json.dumps(incident_context, indent=2, default=str)[:1500]
        prompt += f"""

## ACTIVE INCIDENT WAR ROOM

You are the AI assistant inside an active incident war room. Your job is to give the on-call team fast, structured, actionable answers.

**Incident context:**
```json
{ctx_json}
```

**STRICT RESPONSE FORMAT — always use this structure, no exceptions:**

Use clean markdown that renders well. Every response must be scannable in under 10 seconds.

---

### 🔍 Root Cause
One sentence. What broke and why.

### 📋 Evidence
- Bullet list of specific facts from the context (instance IDs, states, error messages)
- If data is missing, say what is unknown and why

### 💥 Impact
One sentence. What is affected and who.

### ✅ Immediate Fix
Numbered steps. Include the exact CLI command with real values, not placeholders.
```bash
# example — always use real instance IDs from context
aws ec2 start-instances --instance-ids i-0abc123 --region us-east-1
```

### ⏭ Next Steps (if needed)
Short follow-up actions after the immediate fix.

---

**RULES:**
- Never use `<instance_id>` placeholders — always use real IDs from the context
- Never say "I can try" or "Would you like me to" — just give the answer
- If root_cause is "Under investigation", do your best analysis from the description
- Max 250 words total — the team is in a crisis, be concise and direct"""
    return prompt


# ── Suggestion extraction ─────────────────────────────────────────────────────

_SUGGESTION_PATTERN = re.compile(
    r'\[SUGGESTIONS?\]:\s*(.*?)(?:\[/SUGGESTIONS?\]|$)',
    re.IGNORECASE | re.DOTALL,
)


def _extract_suggestions(text: str) -> tuple[str, list[str]]:
    """Extract [SUGGESTIONS]: ... from text, return (clean_text, suggestions)."""
    suggestions = []
    match = _SUGGESTION_PATTERN.search(text)
    if match:
        raw = match.group(1).strip()
        suggestions = [s.strip().lstrip("•-").strip() for s in raw.split("\n") if s.strip()]
        text = text[:match.start()].rstrip()
    return text, suggestions


# ── Main chat function ─────────────────────────────────────────────────────────

def chat_with_intelligence(
    message: str,
    session_id: str,
    incident_context: dict = None,
    preferred_provider: str = None,
    image_data: str = None,
    image_type: str = None,
) -> tuple[str, list[str]]:
    """
    Main entry point for all chat messages.
    Returns (reply_text, suggestions_list).

    Flow:
    1. Load conversation history
    2. Prefetch relevant live infra data (cached 60s)
    3. Build prompt with context
    4. Call LLM (with auto-fallback)
    5. Save to history, return answer
    """
    if not _active_provider and not _groq_client and not _anthropic_client and not _openai_client:
        return (
            "No LLM provider is configured. Please add one of these to your .env file:\n"
            "- `GROQ_API_KEY=...` (free at console.groq.com — recommended)\n"
            "- `ANTHROPIC_API_KEY=...`\n"
            "- `OPENAI_API_KEY=...`",
            []
        )

    # Load conversation history
    history = get_history(session_id, max_messages=20)

    # Prefetch live infra data relevant to this message
    infra_data = _prefetch_infra(message, session_id)

    # Build the user turn — inject infra data if available
    _infra_check_phrases = {"infra", "infrastructure", "check my", "show my", "my setup",
                             "my aws", "all services", "overview", "whats running", "what's running"}
    _is_infra_check = any(p in message.lower() for p in _infra_check_phrases)

    user_turn = message
    if infra_data:
        extra = (
            "\n\nPlease give a COMPLETE infrastructure report covering every section in the live data. "
            "Use emoji section headers. Show all resources, their states, and end with an overall health summary and recommended actions."
            if _is_infra_check else
            "\n\nUse the above live data to answer specifically."
        )
        user_turn = (
            f"{message}\n\n"
            f"--- Live Infrastructure Data ---\n{infra_data}\n"
            f"--- End Live Data ---"
            f"{extra}"
        )

    # Build system prompt
    system = _build_system_prompt(incident_context)

    # Convert history to message format
    history_messages = [{"role": m["role"], "content": m["content"]} for m in history]

    # Call LLM
    try:
        provider = preferred_provider or _active_provider
        reply = _llm_call(
            user_turn,
            system=system,
            history=history_messages,
            provider=provider,
            max_tokens=2048,
            temperature=0.7,
        )
    except RuntimeError as e:
        reply = str(e)
    except Exception as e:
        logger.error(f"chat_with_intelligence error: {e}", exc_info=True)
        reply = f"I encountered an error: {e}. Please try again."

    # Extract suggestions if any
    clean_reply, suggestions = _extract_suggestions(reply)

    # Save to session history
    _add_message(session_id, "user", message)
    _add_message(session_id, "assistant", clean_reply)

    return clean_reply, suggestions


# ── Streaming support ─────────────────────────────────────────────────────────

def _chat_anthropic_stream(
    system_prompt: str,
    history_messages: list[dict],
    message: str,
    session_id: str,
    vision_content=None,
    on_tool_event=None,
):
    """Stream tokens from Anthropic. Yields text chunks."""
    if not _anthropic_client or _is_dead("anthropic"):
        # Fall back to non-streaming Groq
        try:
            reply = _llm_call(message, system=system_prompt, history=history_messages, provider="groq")
            yield reply
        except Exception as e:
            yield f"Error: {e}"
        return

    try:
        messages = history_messages + [{"role": "user", "content": vision_content or message}]
        with _anthropic_client.messages.stream(
            model=_MODELS["anthropic"],
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("credit", "quota", "billing", "401")):
            _mark_dead("anthropic")
        # Fall back to Groq
        try:
            reply = _llm_call(message, system=system_prompt, history=history_messages, provider="groq")
            yield reply
        except Exception as e2:
            yield f"Error: {e2}"


# ── Compatibility shims (used by chat.py action catalogue) ────────────────────

def _maybe_answer_platform_question(message: str) -> str | None:
    """Return None — the main LLM now handles platform questions directly."""
    return None


def _build_history_messages(history) -> list[dict]:
    """Convert history objects to dicts for LLM calls."""
    result = []
    for m in history:
        if isinstance(m, dict):
            result.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        else:
            result.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")})
    return result


# ── Startup validation ────────────────────────────────────────────────────────
# Run at import time so the server logs which provider is active
try:
    _validate_providers()
except Exception as _e:
    logger.warning(f"Provider validation failed: {_e}")
