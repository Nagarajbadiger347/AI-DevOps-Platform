import json
import os
import re
from anthropic import Anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


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
    if not client:
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
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
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
    if not client:
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
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
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
    if not client:
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
- "github_pr": create a GitHub PR with a code fix (provide title + body + suggested_files list)
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
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
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
