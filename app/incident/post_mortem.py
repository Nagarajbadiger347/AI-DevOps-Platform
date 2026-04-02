"""Post-mortem document generator.

Uses LLMFactory to produce a structured post-mortem from incident state data,
then formats and saves it as a Markdown file.

Usage:
    from app.incident.post_mortem import generate_post_mortem, format_as_markdown, save_post_mortem

    pm = generate_post_mortem(incident_state)
    print(format_as_markdown(pm))
    save_post_mortem(pm)
"""
from __future__ import annotations

import json
import os
import re
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from app.llm.factory import LLMFactory
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single event in the incident timeline."""
    timestamp: str
    event: str
    actor: str = "system"


@dataclass
class ActionItem:
    """A follow-up action item from the post-mortem."""
    title:    str
    owner:    str
    priority: str   # P1 | P2 | P3
    due_date: str   # ISO date string, e.g. "2026-04-15"


@dataclass
class PostMortem:
    """Structured post-mortem document."""
    incident_id:          str
    title:                str
    severity:             str
    duration_minutes:     float
    timeline:             list[TimelineEvent]
    root_cause:           str
    contributing_factors: list[str]
    impact:               str
    resolution:           str
    action_items:         list[ActionItem]
    lessons_learned:      list[str]
    prevention_steps:     list[str]
    generated_at:         str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer writing professional post-mortem documents.
Your post-mortems are:
- Blameless and factual
- Concise but comprehensive
- Action-oriented with specific, assignable follow-up items
- Focused on systemic issues rather than individual mistakes
Respond ONLY with a JSON object matching the schema described in the user message.
"""

_JSON_SCHEMA_EXAMPLE = {
    "title": "Brief, descriptive incident title",
    "severity": "SEV1|SEV2|SEV3|SEV4",
    "duration_minutes": 42.0,
    "timeline": [
        {"timestamp": "2026-04-01T10:00:00Z", "event": "Alerts fired for elevated error rate", "actor": "PagerDuty"},
        {"timestamp": "2026-04-01T10:05:00Z", "event": "On-call engineer acknowledged", "actor": "alice"},
    ],
    "root_cause": "Detailed root cause description.",
    "contributing_factors": [
        "Insufficient load testing of the new payment service",
        "Missing circuit breaker on downstream DB connection",
    ],
    "impact": "Quantified description of user / business impact.",
    "resolution": "Step-by-step description of how the incident was resolved.",
    "action_items": [
        {"title": "Add circuit breaker to payment-service DB client", "owner": "backend-team", "priority": "P1", "due_date": "2026-04-15"},
    ],
    "lessons_learned": [
        "Load testing must include failure injection scenarios.",
    ],
    "prevention_steps": [
        "Implement chaos engineering tests in staging.",
        "Add automated rollback on p99 latency breach.",
    ],
}




def _log(level_fn, msg: str, **kwargs) -> None:
    """Emit a structured log message compatible with stdlib logging."""
    if kwargs:
        level_fn(msg, extra=kwargs)
    else:
        level_fn(msg)

def _build_prompt(incident_state: dict) -> str:
    """Build the user-facing LLM prompt from incident state.

    Args:
        incident_state: Dict containing any subset of incident keys.

    Returns:
        String prompt ready to send to the LLM.
    """
    # Extract relevant fields with graceful defaults
    incident_id   = incident_state.get("incident_id", "UNKNOWN")
    description   = incident_state.get("description", incident_state.get("incident_description", "Not provided"))
    root_cause    = incident_state.get("root_cause", incident_state.get("ai_root_cause", "Not determined"))
    actions_taken = incident_state.get("actions_taken", incident_state.get("executed_actions", []))
    validation    = incident_state.get("validation_results", incident_state.get("validation", {}))
    errors        = incident_state.get("errors", incident_state.get("error_log", []))
    started_at    = incident_state.get("started_at", incident_state.get("created_at", ""))
    resolved_at   = incident_state.get("resolved_at", incident_state.get("completed_at", ""))
    severity      = incident_state.get("severity", "SEV2")
    metrics       = incident_state.get("metrics_at_incident", incident_state.get("observability_context", {}))
    plan_summary  = incident_state.get("plan_summary", incident_state.get("action_plan", ""))

    # Calculate duration if both timestamps available
    duration_note = ""
    if started_at and resolved_at:
        try:
            s = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            e = datetime.datetime.fromisoformat(resolved_at.replace("Z", "+00:00"))
            mins = (e - s).total_seconds() / 60
            duration_note = f"\nDuration: {mins:.1f} minutes (from {started_at} to {resolved_at})"
        except Exception:
            duration_note = f"\nStarted: {started_at}  Resolved: {resolved_at}"

    # Format actions taken
    if isinstance(actions_taken, list):
        actions_str = "\n".join(
            f"  - {a.get('action_type', a.get('type', str(a)))}: {a.get('description', '')}"
            for a in actions_taken
        ) or "  None recorded"
    else:
        actions_str = str(actions_taken)

    # Format errors
    if isinstance(errors, list):
        errors_str = "\n".join(f"  - {e}" for e in errors) or "  None"
    else:
        errors_str = str(errors) or "  None"

    prompt = f"""Generate a comprehensive post-mortem for the following incident.

=== INCIDENT DETAILS ===
Incident ID  : {incident_id}
Severity     : {severity}{duration_note}

=== DESCRIPTION ===
{description}

=== ROOT CAUSE IDENTIFIED ===
{root_cause}

=== ACTION PLAN EXECUTED ===
{plan_summary}

=== ACTIONS TAKEN ===
{actions_str}

=== VALIDATION RESULTS ===
{json.dumps(validation, indent=2, default=str) if isinstance(validation, dict) else str(validation)}

=== ERRORS ENCOUNTERED ===
{errors_str}

=== OBSERVABILITY CONTEXT (metrics at time of incident) ===
{json.dumps(metrics, indent=2, default=str) if isinstance(metrics, dict) else str(metrics)}

=== INSTRUCTIONS ===
Based on the above incident data, produce a thorough, blameless post-mortem.
Fill all fields accurately. If information is missing, make reasonable inferences
but do NOT fabricate specific numbers or names.

Respond with ONLY a valid JSON object matching this exact schema:
{json.dumps(_JSON_SCHEMA_EXAMPLE, indent=2)}

Important:
- "duration_minutes" must be a float number (not a string).
- "timeline" must list events chronologically with realistic timestamps.
- "action_items" must each have title, owner, priority (P1/P2/P3), and due_date (YYYY-MM-DD).
- "prevention_steps" should be concrete and technical, not vague platitudes.
"""
    return prompt


def _parse_llm_response(raw: str, incident_state: dict) -> PostMortem:
    """Parse the LLM JSON response into a PostMortem dataclass.

    Falls back to a minimal PostMortem if parsing fails.

    Args:
        raw:            Raw string returned by the LLM.
        incident_state: Original incident state for fallback values.

    Returns:
        PostMortem dataclass.
    """
    incident_id = incident_state.get("incident_id", "UNKNOWN")

    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        # Find the outermost JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        data = json.loads(cleaned)

        timeline = [
            TimelineEvent(
                timestamp=e.get("timestamp", ""),
                event=e.get("event", ""),
                actor=e.get("actor", "system"),
            )
            for e in data.get("timeline", [])
        ]

        action_items = [
            ActionItem(
                title=a.get("title", ""),
                owner=a.get("owner", "TBD"),
                priority=a.get("priority", "P2"),
                due_date=a.get("due_date", ""),
            )
            for a in data.get("action_items", [])
        ]

        return PostMortem(
            incident_id=incident_id,
            title=data.get("title", f"Incident {incident_id} Post-Mortem"),
            severity=data.get("severity", incident_state.get("severity", "SEV2")),
            duration_minutes=float(data.get("duration_minutes", 0)),
            timeline=timeline,
            root_cause=data.get("root_cause", "Not determined"),
            contributing_factors=data.get("contributing_factors", []),
            impact=data.get("impact", "Impact not quantified"),
            resolution=data.get("resolution", ""),
            action_items=action_items,
            lessons_learned=data.get("lessons_learned", []),
            prevention_steps=data.get("prevention_steps", []),
        )

    except Exception as exc:
        _log(logger.warning, "post_mortem_parse_failed", error=str(exc), raw_length=len(raw))
        # Return minimal post-mortem from raw text
        return PostMortem(
            incident_id=incident_id,
            title=f"Incident {incident_id} Post-Mortem (auto-generated)",
            severity=incident_state.get("severity", "SEV2"),
            duration_minutes=0.0,
            timeline=[],
            root_cause=incident_state.get("root_cause", "See raw LLM output below"),
            contributing_factors=[],
            impact="See incident description",
            resolution=raw[:500] if raw else "LLM response parsing failed",
            action_items=[],
            lessons_learned=[f"LLM parse error: {exc}"],
            prevention_steps=[],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_post_mortem(incident_state: dict) -> PostMortem:
    """Generate a structured post-mortem from incident state.

    Args:
        incident_state: Dict with any combination of: incident_id, description,
            root_cause, actions_taken, validation_results, errors, started_at,
            resolved_at, severity, observability_context, plan_summary.

    Returns:
        PostMortem dataclass.
    """
    if not _LLM_AVAILABLE:
        logger.error("llm_unavailable_for_post_mortem")
        return PostMortem(
            incident_id=incident_state.get("incident_id", "UNKNOWN"),
            title="Post-mortem unavailable (LLM not configured)",
            severity=incident_state.get("severity", "SEV2"),
            duration_minutes=0.0,
            timeline=[],
            root_cause="LLM not available",
            contributing_factors=[],
            impact="",
            resolution="",
            action_items=[],
            lessons_learned=["LLMFactory could not be imported."],
            prevention_steps=[],
        )

    prompt = _build_prompt(incident_state)

    try:
        llm      = LLMFactory.get()
        response = llm.complete(prompt, system=_SYSTEM_PROMPT, max_tokens=3000)
        raw      = response.content
        _log(logger.info, 
            "post_mortem_llm_response",
            incident_id=incident_state.get("incident_id"),
            tokens_in=response.input_tokens,
            tokens_out=response.output_tokens,
        )
    except Exception as exc:
        _log(logger.error, "post_mortem_llm_failed", error=str(exc))
        raw = ""

    return _parse_llm_response(raw, incident_state)


def format_as_markdown(pm: PostMortem) -> str:
    """Render a PostMortem as nicely formatted Markdown.

    Args:
        pm: PostMortem dataclass.

    Returns:
        Markdown string.
    """
    lines = [
        f"# Post-Mortem: {pm.title}",
        "",
        f"**Incident ID:** {pm.incident_id}  ",
        f"**Severity:** {pm.severity}  ",
        f"**Duration:** {pm.duration_minutes:.1f} minutes  ",
        f"**Generated:** {pm.generated_at}  ",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"{pm.impact}",
        "",
        "---",
        "",
        "## Timeline",
        "",
    ]

    if pm.timeline:
        for event in pm.timeline:
            actor = f" *(by {event.actor})*" if event.actor and event.actor != "system" else ""
            lines.append(f"- **{event.timestamp}** — {event.event}{actor}")
    else:
        lines.append("_No timeline data recorded._")

    lines += [
        "",
        "---",
        "",
        "## Root Cause",
        "",
        pm.root_cause,
        "",
        "## Contributing Factors",
        "",
    ]

    if pm.contributing_factors:
        for factor in pm.contributing_factors:
            lines.append(f"- {factor}")
    else:
        lines.append("_None identified._")

    lines += [
        "",
        "## Impact",
        "",
        pm.impact,
        "",
        "## Resolution",
        "",
        pm.resolution,
        "",
        "---",
        "",
        "## Action Items",
        "",
    ]

    if pm.action_items:
        lines.append("| # | Title | Owner | Priority | Due Date |")
        lines.append("|---|-------|-------|----------|----------|")
        for i, ai in enumerate(pm.action_items, 1):
            lines.append(f"| {i} | {ai.title} | {ai.owner} | {ai.priority} | {ai.due_date} |")
    else:
        lines.append("_No action items recorded._")

    lines += [
        "",
        "---",
        "",
        "## Lessons Learned",
        "",
    ]

    if pm.lessons_learned:
        for lesson in pm.lessons_learned:
            lines.append(f"- {lesson}")
    else:
        lines.append("_None recorded._")

    lines += [
        "",
        "## Prevention Steps",
        "",
    ]

    if pm.prevention_steps:
        for step in pm.prevention_steps:
            lines.append(f"- {step}")
    else:
        lines.append("_None recorded._")

    lines.append("")
    return "\n".join(lines)


def save_post_mortem(pm: PostMortem, path: str = "post_mortems/") -> str:
    """Save a PostMortem as a Markdown file.

    Args:
        pm:   PostMortem dataclass to save.
        path: Directory path (default: post_mortems/ relative to cwd).

    Returns:
        Absolute path to the saved file.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    safe_id   = re.sub(r"[^a-zA-Z0-9_-]", "_", pm.incident_id)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename  = f"post_mortem_{safe_id}_{timestamp}.md"
    filepath  = directory / filename

    content = format_as_markdown(pm)
    filepath.write_text(content, encoding="utf-8")

    _log(logger.info, "post_mortem_saved", path=str(filepath), incident_id=pm.incident_id)
    return str(filepath.resolve())
