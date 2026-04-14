import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.llm.base import BaseLLM, LLMResponse

# ── Provider auto-detection ───────────────────────────────────
# Priority: Anthropic → Groq → Ollama (local, no key needed)
import urllib.request as _urllib

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "").strip()
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://localhost:11434")

_provider     = None
_ollama_model = None
client        = None
_provider_warning = None   # surfaced in chat responses when key is misconfigured

# Validate Anthropic key format before attempting to use it.
# Real keys start with "sk-ant-". A key with "#" gets truncated by dotenv.
_ANTHROPIC_KEY_VALID = ANTHROPIC_API_KEY.startswith("sk-ant-")
if ANTHROPIC_API_KEY and not _ANTHROPIC_KEY_VALID:
    _provider_warning = (
        "⚠️ ANTHROPIC_API_KEY looks malformed (valid keys start with 'sk-ant-'). "
        "If your key contains a '#' character in .env, wrap it in quotes: "
        "ANTHROPIC_API_KEY=\"sk-ant-...\". Falling back to next available provider."
    )

_anthropic_client = None
_groq_client      = None

if ANTHROPIC_API_KEY and _ANTHROPIC_KEY_VALID:
    from anthropic import Anthropic
    _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=30.0)
    client    = _anthropic_client
    _provider = "anthropic"

if GROQ_API_KEY:
    from groq import Groq
    _groq_client = Groq(api_key=GROQ_API_KEY, timeout=30.0)
    if not _provider:
        client    = _groq_client
        _provider = "groq"

# Track billing/auth failures to skip dead providers in same session
_PROVIDER_DEAD: dict[str, bool] = {}


def _mark_dead(provider: str) -> None:
    _PROVIDER_DEAD[provider] = True


def _is_dead(provider: str) -> bool:
    return _PROVIDER_DEAD.get(provider, False)

# Always probe Ollama regardless of other providers — needed for explicit selection
try:
    _r = _urllib.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
    _models = json.loads(_r.read()).get("models", [])
    if _models:
        _ollama_model = _models[0]["name"]
        if not _provider:
            _provider = "ollama"
except Exception:
    pass


def _llm(system: str, messages: list, max_tokens: int = 1024,
         force_provider: str = "", temperature: float = 0.7) -> str:
    """Unified LLM call — Anthropic / Groq / Ollama (local, no key).

    force_provider: override the auto-detected provider ("anthropic", "groq", "ollama").
    Falls back to the auto-detected provider if the requested one is not configured.
    """
    provider = force_provider if force_provider else _provider
    # When user explicitly picks a provider, check it's actually available
    if force_provider:
        if force_provider == "anthropic" and not (ANTHROPIC_API_KEY and _ANTHROPIC_KEY_VALID):
            raise RuntimeError(
                "Claude (Anthropic) is not configured. Add a valid ANTHROPIC_API_KEY "
                "(starting with sk-ant-) to your .env file."
            )
        if force_provider == "groq" and not GROQ_API_KEY:
            raise RuntimeError(
                "Groq is not configured. Add GROQ_API_KEY to your .env file."
            )
        if force_provider == "ollama" and not _ollama_model:
            raise RuntimeError(
                "Ollama is not running or has no models loaded. "
                f"Start Ollama and pull a model (e.g. `ollama pull llama3`), "
                f"then restart the server. Checked: {OLLAMA_HOST}"
            )

    if provider == "anthropic":
        if _is_dead("anthropic"):
            # Previously failed with billing/auth — skip directly to next provider
            if _groq_client:
                return _llm(system, messages, max_tokens, force_provider="groq", temperature=temperature)
            raise RuntimeError("Anthropic billing/auth error — add credits at console.anthropic.com or configure GROQ_API_KEY.")
        try:
            resp = _anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=temperature,
            )
            return resp.content[0].text or ""
        except Exception as _ant_exc:
            _err = str(_ant_exc)
            _billing = (
                "credit balance" in _err.lower() or "too low" in _err.lower() or
                "no credits" in _err.lower() or "401" in _err or
                "invalid_api_key" in _err.lower() or "authentication" in _err.lower()
            )
            if _billing:
                _mark_dead("anthropic")
                # Auto-fallback to Groq if available
                if _groq_client:
                    return _llm(system, messages, max_tokens, force_provider="groq", temperature=temperature)
                raise RuntimeError(
                    "Anthropic API error: " + _err[:200] +
                    "\nFix: Add credits at console.anthropic.com or set GROQ_API_KEY in .env"
                )
            raise

    elif provider == "groq":
        all_msgs = [{"role": "system", "content": system}] + messages
        # Use a fast model; cap tokens to stay under free-tier limits (6000 tok/min)
        groq_max = min(max_tokens, 1024)
        try:
            resp = _groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",   # fastest Groq model, lowest token cost
                messages=all_msgs,
                max_tokens=groq_max,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as groq_exc:
            err = str(groq_exc)
            if "rate_limit" in err.lower() or "429" in err:
                raise RuntimeError(
                    "Groq free tier rate limit hit. Wait 60 seconds and try again, "
                    "or upgrade at console.groq.com."
                )
            raise

    elif provider == "ollama":
        # Build prompt from system + messages
        parts = []
        if system:
            parts.append(f"System: {system}")
        for m in messages:
            role = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{role}: {m['content']}")
        prompt = "\n\n".join(parts)
        payload = json.dumps({
            "model": _ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = _urllib.Request(f"{OLLAMA_HOST}/api/generate", data=payload,
                              headers={"Content-Type": "application/json"})
        r = _urllib.urlopen(req, timeout=30)
        return json.loads(r.read())["response"]

    else:
        raise RuntimeError(
            "No LLM configured. Either:\n"
            "  • Install Ollama (free, local): https://ollama.com\n"
            "  • Add ANTHROPIC_API_KEY or GROQ_API_KEY to .env"
        )


def _extract_json(text: str) -> str:
    """Robustly extract JSON from Claude's response.

    Handles all real-world formats:
      - Raw JSON
      - ```json ... ```
      - ``` ... ```
      - Prose before/after the JSON block
    """
    text = text.strip()

    # 1. Try to extract from a fenced code block (```json ... ``` or ``` ... ```)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # 2. Try to find a bare JSON object (first { to last })
    first = text.find("{")
    last  = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first:last + 1]

    # 3. Return as-is and let json.loads raise a clear error
    return text


def analyze_context(context: dict) -> dict:
    """Analyze incident context using Claude AI."""
    if not _provider:
        return {"error": "ANTHROPIC_API_KEY not configured", "rca": "Sample root cause", "confidence": 0.5}

    incident_id = context.get("incident_id", "unknown")
    details = context.get("details", {})

    prompt = f"""
    Analyze this DevOps incident and provide root cause analysis:

    Incident ID: {incident_id}
    Details: {details}

    Provide:
    1. Root cause analysis
    2. Confidence score (0-1)
    3. Recommended actions
    """

    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=1000)
        rca = content.split("1.")[1].split("2.")[0].strip() if "1." in content else content
        confidence = 0.8
        actions = content.split("3.")[1].strip() if "3." in content else "Investigate further"

        return {
            "rca": rca,
            "confidence": confidence,
            "recommended_actions": actions
        }
    except Exception as e:
        return {"error": str(e), "rca": "Analysis failed", "confidence": 0.0}


def diagnose_aws_resource(obs_context: dict) -> dict:
    """Feed AWS observability data into Claude for structured root cause analysis."""
    if not _provider:
        return {
            "error": "ANTHROPIC_API_KEY not configured",
            "summary": "AI diagnosis unavailable — set ANTHROPIC_API_KEY",
            "root_cause": "Unknown",
            "confidence": 0.0,
            "findings": [],
            "recommended_actions": [],
        }

    resource_type = obs_context.get("resource_type", "unknown")
    resource_id   = obs_context.get("resource_id", "unknown")
    region        = obs_context.get("region", "unknown")

    prompt = f"""You are an expert AWS SRE. Analyze the following observability data collected from a {resource_type} resource ({resource_id}) in region {region} and diagnose any issues.

=== OBSERVABILITY DATA ===
{json.dumps(obs_context, indent=2, default=str)}
=== END DATA ===

Respond in the following JSON format only (no markdown, no extra text):
{{
  "summary": "<one sentence summary of the situation>",
  "root_cause": "<most likely root cause>",
  "confidence": <0.0-1.0>,
  "severity": "<critical|high|medium|low>",
  "findings": [
    "<specific finding 1>",
    "<specific finding 2>"
  ],
  "recommended_actions": [
    "<actionable step 1>",
    "<actionable step 2>"
  ]
}}"""

    content = ""
    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=1500)
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return {
            "summary": content,
            "root_cause": "Could not parse structured response",
            "confidence": 0.5,
            "severity": "unknown",
            "findings": [],
            "recommended_actions": ["Review raw response above"],
        }
    except Exception as e:
        return {"error": str(e), "root_cause": "Analysis failed", "confidence": 0.0}


def synthesize_incident(incident_data: dict) -> dict:
    """Synthesize all collected observability data into an action plan.

    incident_data keys:
        incident_id, description, severity,
        aws_context, k8s_context, github_context
    Returns structured JSON with root_cause, findings, and actions_to_take.
    """
    if not _provider:
        return {
            "error": "ANTHROPIC_API_KEY not configured",
            "summary": "AI synthesis unavailable — set ANTHROPIC_API_KEY",
            "root_cause": "Unknown",
            "confidence": 0.0,
            "severity": incident_data.get("severity", "unknown"),
            "findings": [],
            "actions_to_take": [],
        }

    prompt = f"""You are an expert SRE (Site Reliability Engineer) performing autonomous incident response.

An incident has been reported. You have been given observability data from AWS, Kubernetes, and GitHub.
Your job is to:
1. Determine the root cause
2. Identify specific findings from the data
3. Decide what actions to take (be specific and actionable)

=== INCIDENT ===
ID: {incident_data.get('incident_id')}
Description: {incident_data.get('description')}
Reported Severity: {incident_data.get('severity')}

=== AWS OBSERVABILITY ===
{json.dumps(incident_data.get('aws_context', {}), indent=2, default=str)}

=== KUBERNETES STATE ===
{json.dumps(incident_data.get('k8s_context', {}), indent=2, default=str)}

=== GITHUB RECENT ACTIVITY ===
{json.dumps(incident_data.get('github_context', {}), indent=2, default=str)}

=== INSTRUCTIONS ===
Based on the above data, produce a structured incident analysis and action plan.
For actions_to_take, use these types:
- "k8s_restart": restart a Kubernetes deployment (provide namespace + deployment)
- "k8s_scale": scale a deployment (provide namespace + deployment + replicas)
- "github_pr": create a GitHub PR with a code fix (provide title + body; if you can determine exact file changes, include file_patches as [{{"path": "<file>", "content": "<full corrected file content>"}}])
- "jira_ticket": create a Jira ticket (provide title + body)
- "slack_warroom": create a Slack war room (provide title + description)
- "opsgenie_alert": send OpsGenie alert (provide message + priority)
- "none": no automated action needed (explain why in notes)

Respond ONLY in this JSON format (no markdown, no extra text):
{{
  "summary": "<one-sentence summary>",
  "root_cause": "<most likely root cause>",
  "confidence": <0.0-1.0>,
  "severity": "<critical|high|medium|low>",
  "findings": [
    "<specific finding from the data>"
  ],
  "actions_to_take": [
    {{
      "type": "<action type from list above>",
      "params": {{}},
      "reason": "<why this action is needed>"
    }}
  ]
}}"""

    content = ""
    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=2000)
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return {
            "summary": content,
            "root_cause": "Could not parse structured response",
            "confidence": 0.5,
            "severity": "unknown",
            "findings": [],
            "actions_to_take": [],
        }
    except Exception as e:
        return {"error": str(e), "root_cause": "Synthesis failed", "confidence": 0.0}


def review_pr(pr_data: dict) -> dict:
    """AI code review of a GitHub PR — security, infra, logic issues.

    pr_data: output of github.get_pr_for_review()
    Returns structured review with issues, recommendations, and a summary comment.
    """
    if not _provider:
        return {
            "error":            "ANTHROPIC_API_KEY not configured",
            "summary":          "AI review unavailable",
            "issues":           [],
            "recommendations":  [],
            "security_concerns": [],
            "infra_changes":    [],
            "comment":          "",
        }

    files_text = ""
    for f in pr_data.get("files", []):
        files_text += f"\n### {f['filename']} ({f['status']}, +{f['additions']} -{f['deletions']})\n"
        if f.get("patch"):
            files_text += f"```diff\n{f['patch']}\n```\n"

    prompt = f"""You are a senior DevOps engineer performing an automated code review.

Review the following GitHub Pull Request for issues related to:
1. Security vulnerabilities (hardcoded secrets, insecure configs, overly permissive IAM/RBAC)
2. Infrastructure concerns (Dockerfile, K8s manifests, Terraform, CI/CD pipeline changes)
3. Code quality issues that could cause production incidents
4. Dependencies (outdated libraries, vulnerable packages)

=== PULL REQUEST ===
Title:  {pr_data.get('title')}
Author: {pr_data.get('author')}
Base:   {pr_data.get('base_branch')} ← {pr_data.get('head_branch')}
+{pr_data.get('additions')} lines  -{pr_data.get('deletions')} lines

Description:
{pr_data.get('body', 'No description provided.')}

=== CHANGED FILES ===
{files_text}

Respond ONLY in this JSON format (no markdown, no extra text):
{{
  "verdict": "approve|request_changes|comment",
  "summary": "<2-3 sentence overall assessment>",
  "issues": [
    {{"severity": "critical|high|medium|low", "file": "<filename>", "description": "<issue>"}}
  ],
  "security_concerns": ["<concern 1>", "<concern 2>"],
  "infra_changes": ["<infra concern 1>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>"],
  "comment": "<full markdown comment ready to post on the PR>"
}}"""

    content = ""
    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=2000)
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return {
            "verdict":          "comment",
            "summary":          content,
            "issues":           [],
            "security_concerns": [],
            "infra_changes":    [],
            "recommendations":  [],
            "comment":          content,
        }
    except Exception as e:
        return {"error": str(e), "verdict": "comment", "summary": "Review failed", "issues": [], "recommendations": [], "security_concerns": [], "infra_changes": [], "comment": ""}


def predict_scaling(metrics_data: dict) -> dict:
    """Analyse CloudWatch metric trends and predict if scaling is needed.

    metrics_data: output of aws_ops.get_scaling_metrics()
    Returns a prediction with confidence, direction, and recommended action.
    """
    if not _provider:
        return {
            "error":                 "ANTHROPIC_API_KEY not configured",
            "should_scale":          False,
            "direction":             "none",
            "confidence":            0.0,
            "reasoning":             "AI unavailable",
            "recommended_action":    "none",
            "urgency":               "low",
        }

    prompt = f"""You are an expert AWS cloud architect analysing resource utilisation trends.

Based on the following CloudWatch metric data, predict whether this resource needs to be scaled.
Analyse the trend direction, peak values, and growth rate over the observation window.

=== METRICS DATA ===
{json.dumps(metrics_data, indent=2, default=str)}

Respond ONLY in this JSON format (no markdown, no extra text):
{{
  "should_scale":       true|false,
  "direction":          "up|down|none",
  "confidence":         <0.0-1.0>,
  "urgency":            "immediate|soon|low",
  "current_utilization": "<e.g. CPU averaging 78% over last 6h>",
  "trend":              "increasing|decreasing|stable|spike",
  "reasoning":          "<explanation of why scaling is or isn't needed>",
  "recommended_action": "<concrete action, e.g. scale ECS service to 6 tasks, or no action needed>",
  "predicted_breach_in_minutes": <minutes until threshold breach, or null if no breach predicted>
}}"""

    content = ""
    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=1000)
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return {
            "should_scale":       False,
            "direction":          "none",
            "confidence":         0.5,
            "reasoning":          content,
            "recommended_action": "Review raw response",
            "urgency":            "low",
        }
    except Exception as e:
        return {"error": str(e), "should_scale": False, "confidence": 0.0}


def assess_deployment(context: dict) -> dict:
    """Pre-deployment risk assessment — go or no-go decision.

    context keys:
        deployment, namespace, new_image, description,
        k8s_state, recent_incidents, aws_alarms, recent_commits
    Returns structured JSON with risk_level, go_no_go, concerns, checklist.
    """
    if not _provider:
        return {
            "error":       "ANTHROPIC_API_KEY not configured",
            "go_no_go":    "no_go",
            "risk_level":  "unknown",
            "risk_score":  0.0,
            "concerns":    [],
            "checklist":   [],
            "recommendations": [],
            "summary":     "AI assessment unavailable — set ANTHROPIC_API_KEY",
        }

    prompt = f"""You are a senior SRE performing a pre-deployment risk assessment.

Analyse the following context and decide if it is SAFE to deploy right now.
Be conservative — when in doubt, recommend no-go with a clear explanation.

=== DEPLOYMENT ===
Deployment:  {context.get('deployment')}
Namespace:   {context.get('namespace')}
New Image:   {context.get('new_image')}
Description: {context.get('description')}

=== CURRENT KUBERNETES STATE ===
{json.dumps(context.get('k8s_state', {}), indent=2, default=str)}

=== RECENT PAST INCIDENTS (from memory) ===
{json.dumps(context.get('recent_incidents', []), indent=2, default=str)}

=== ACTIVE AWS ALARMS ===
{json.dumps(context.get('aws_alarms', {}), indent=2, default=str)}

=== RECENT CODE CHANGES (last 2h) ===
{json.dumps(context.get('recent_commits', {}), indent=2, default=str)}

Respond ONLY in this JSON format (no markdown, no extra text):
{{
  "go_no_go":    "go|no_go|go_with_caution",
  "risk_level":  "critical|high|medium|low",
  "risk_score":  <0.0-1.0, where 1.0 = maximum risk>,
  "summary":     "<one-sentence assessment>",
  "concerns":    ["<specific concern 1>", "<specific concern 2>"],
  "checklist": [
    {{"item": "<pre-deploy check>", "required": true|false}}
  ],
  "recommendations": ["<action to reduce risk before deploying>"],
  "safe_window": "<e.g. Deploy after 22:00 UTC when traffic is low, or Now is fine>"
}}"""

    content = ""
    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=1500)
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return {
            "go_no_go":    "no_go",
            "risk_level":  "unknown",
            "risk_score":  0.5,
            "summary":     content,
            "concerns":    [],
            "checklist":   [],
            "recommendations": ["Review raw AI response above"],
            "safe_window": "Unknown",
        }
    except Exception as e:
        return {"error": str(e), "go_no_go": "no_go", "risk_score": 1.0}


def interpret_jira_for_pr(jira_data: dict) -> dict:
    """Interpret a Jira change-request ticket and generate a GitHub PR plan.

    jira_data keys: key, summary, description, issue_type, labels, reporter
    Returns: branch_name, pr_title, pr_body, file_patches (best-effort), target_files
    """
    if not _provider:
        return {
            "error":        "ANTHROPIC_API_KEY not configured",
            "pr_title":     jira_data.get("summary", "Change request"),
            "pr_body":      jira_data.get("description", ""),
            "branch_name":  f"jira/{jira_data.get('key', 'ticket').lower()}",
            "file_patches": [],
            "target_files": [],
        }

    prompt = f"""You are a DevOps engineer reading a Jira change-request ticket.
Your job is to plan the GitHub Pull Request that implements this change.

=== JIRA TICKET ===
Key:         {jira_data.get('key')}
Type:        {jira_data.get('issue_type')}
Summary:     {jira_data.get('summary')}
Reporter:    {jira_data.get('reporter')}
Labels:      {', '.join(jira_data.get('labels', []))}

Description:
{jira_data.get('description', 'No description provided.')}

=== INSTRUCTIONS ===
Based on the ticket, produce a PR plan.
- branch_name must be lowercase, use hyphens, format: jira/<ticket-key>-<short-slug>
- If the description mentions specific files, config values, or code to change, include
  file_patches with the exact changes needed.
- If you cannot determine the exact file content, leave file_patches empty and list
  target_files with filenames that likely need editing.

Respond ONLY in this JSON format (no markdown, no extra text):
{{
  "pr_title":     "<concise PR title>",
  "pr_body":      "<full markdown PR description referencing the Jira ticket>",
  "branch_name":  "jira/{jira_data.get('key', 'ticket').lower()}-<slug>",
  "target_files": ["<file that needs changing>"],
  "file_patches": [
    {{"path": "<file path>", "content": "<full new file content>"}}
  ],
  "confidence":   <0.0-1.0 how confident you are about the changes>
}}"""

    content = ""
    try:
        content = _llm("", [{"role": "user", "content": prompt}], max_tokens=2000)
        return json.loads(_extract_json(content))
    except json.JSONDecodeError:
        return {
            "pr_title":     jira_data.get("summary", "Change request"),
            "pr_body":      content,
            "branch_name":  f"jira/{jira_data.get('key', 'ticket').lower()}",
            "file_patches": [],
            "target_files": [],
            "confidence":   0.0,
        }
    except Exception as e:
        return {"error": str(e), "pr_title": jira_data.get("summary", ""), "file_patches": [], "branch_name": f"jira/{jira_data.get('key', 'ticket').lower()}", "target_files": []}


def chat_devops(message: str, history: list, context: dict,
                force_provider: str = "") -> str:
    """Conversational DevOps assistant — answers any question, uses live context when relevant."""
    if not _provider:
        return (
            "No LLM configured. Add ANTHROPIC_API_KEY or GROQ_API_KEY to your .env file "
            "(use the Secrets panel in the sidebar), then restart the server."
        )

    configured = context.get("configured", [])
    has_context = bool(configured)
    sources_str = ", ".join(configured) if configured else "none"

    # Build the list of integrations that are NOT returning data
    all_integrations = {"aws", "grafana", "k8s", "github", "gitlab"}
    missing = sorted(all_integrations - set(configured))

    SYSTEM = f"""You are a DevOps AI assistant. Match your response length to the question — be intelligent about it.

RESPONSE LENGTH RULES (follow strictly):
- Greetings (hi, hey, hello, sup): reply naturally and conversationally — vary your response, be warm but brief. Do NOT say "I'm ready to help with your DevOps questions" or any robotic assistant phrase. Just say hi back like a person would. Examples: "Hey! What's up?", "Hey there 👋", "Hi! What can I do for you?".
- Follow-up small talk (no thanks, I'm good, cool, ok): acknowledge it naturally and stop — do not repeat your intro or offer help again. If they say "no thanks", just say something like "Sure, no problem!" and leave it.
- Simple yes/no or factual questions: 1–2 sentences max.
- Status checks, "is X running?", "how many pods?": direct answer from context, 1–3 lines.
- Troubleshooting, debugging, root cause analysis: structured answer with relevant detail — use bullet points, include error context, suggest a fix.
- How-to, explain a concept, write a command/config: as long as needed, use code blocks, be thorough.
- Incident response or "what should I do?": step-by-step with priorities, be thorough.

PLAIN LANGUAGE RULES:
- Always write so that a non-engineer can understand. Avoid jargon; when you must use a technical term, briefly explain it in plain words (e.g. "pods — the small containers your app runs in").
- If someone asks in simple or casual language, reply the same way. Don't over-engineer the answer.
- Lead with the key takeaway, then add supporting detail.

CONTENT RULES:
- NEVER fabricate infrastructure data. Only use resource names, IDs, pod names, alarm names that appear VERBATIM in the live context below.
- If an integration has no data or is not configured, say so in one sentence. Do not invent examples.
- For questions about running state (pods, instances, alarms): answer from live context only. If no context, say "No live data — configure the integration in Secrets."
- For general DevOps questions (concepts, how-to, code): answer from knowledge, be thorough when the question warrants it.
- Do NOT add disclaimers, unsolicited next-steps, or filler padding.

FORMATTING RULES (critical):
- NEVER quote or show raw JSON, dict keys, or field names from the context (e.g. do NOT write "aws.alarms_firing.count" or show success/count fields as JSON).
- Translate data into natural English. Examples:
  - alarms list empty → "No alarms are currently firing." (not a JSON dump)
  - count 0 → say "none" or "0" inline, not as a field
  - pods with phase Running → "All pods are healthy." or list their names naturally
  - success true → just confirm the state, don't mention the field
- Use markdown tables for lists of resources (pods, instances, alarms) when there are 3 or more items.
- Use **bold** for resource names, states, and key values.
- Emoji are allowed sparingly for status: ✅ healthy, ⚠️ warning, 🔴 critical, ℹ️ info.

Live integrations: {sources_str if has_context else "none configured"}
No data from: {", ".join(missing) if missing else "all configured"}"""

    messages = []
    for h in history[-12:]:
        role = h.get("role", "user")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": h["content"]})

    # Attach real context only if we have it; otherwise tell the model explicitly there's none
    if has_context:
        raw = json.dumps(context, default=str)
        if len(raw) > 8000:
            raw = raw[:8000] + "\n...(truncated)"
        ctx_block = f"\n\n[LIVE CONTEXT — use only this data, do not invent anything]\n{raw}\n[END CONTEXT]"
    else:
        ctx_block = "\n\n[NO LIVE CONTEXT — no integrations are connected. Do not fabricate any infrastructure data.]"

    messages.append({"role": "user", "content": message + ctx_block})

    return _llm(SYSTEM, messages, max_tokens=1500, force_provider=force_provider)


# ── BaseLLM-compatible provider class ────────────────────────────────────────


class ClaudeProvider(BaseLLM):
    """Wraps the module-level _llm() into the BaseLLM interface.

    Falls back through the same Anthropic → Groq → Ollama chain that
    the rest of this module uses — so no separate client setup needed.
    """

    def __init__(self, force_provider: str = ""):
        self._force_provider = force_provider

    def is_available(self) -> bool:
        if self._force_provider == "anthropic":
            return bool(ANTHROPIC_API_KEY and _ANTHROPIC_KEY_VALID)
        if self._force_provider == "groq":
            return bool(GROQ_API_KEY)
        if self._force_provider == "ollama":
            return bool(_ollama_model)
        return _provider is not None

    def complete(
        self,
        prompt: str,
        *,
        system: str = "You are an expert DevOps AI assistant.",
        max_tokens: int = 2048,
        messages: list | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        try:
            # Build structured message list — prior turns + final user turn
            msg_list = list(messages) if messages else []
            msg_list.append({"role": "user", "content": prompt})
            content = _llm(system, msg_list, max_tokens,
                           force_provider=self._force_provider,
                           temperature=temperature)
        except Exception as exc:
            err = str(exc)
            # Credit exhausted or billing error — mark provider permanently unavailable
            # for this process so the factory falls back to next provider
            if "credit balance" in err.lower() or "billing" in err.lower() or "payment" in err.lower():
                _prov = self._force_provider or _provider or "anthropic"
                from app.llm.factory import mark_rate_limited
                # Mark both "claude" and "anthropic" keys so factory skips it
                mark_rate_limited(_prov, "try again in 999m0s")
                mark_rate_limited("claude", "try again in 999m0s")
                mark_rate_limited("anthropic", "try again in 999m0s")
                raise RuntimeError(
                    f"Provider '{_prov}' has no credits. Falling back to next available provider."
                ) from exc
            raise
        used = self._force_provider or _provider or "unknown"
        model_name = (
            "claude-sonnet-4-6" if used == "anthropic"
            else "llama-3.3-70b-versatile" if used == "groq"
            else _ollama_model or "ollama"
        )
        return LLMResponse(
            content=content,
            model=model_name,
            provider=used,
        )
