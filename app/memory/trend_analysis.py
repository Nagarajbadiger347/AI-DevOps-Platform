"""
Incident trend analysis over ChromaDB stored incidents.

Provides:
  - MTTR by incident type
  - Recurring root causes
  - Frequency over time (daily/weekly bucketing)
  - Top affected services
  - Reliability score
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_resolution_seconds(value: Any) -> float | None:
    """Parse a resolution time value to seconds.

    Accepts: float/int seconds, or strings like "3h 45m", "2h15m", "45m", "3600".
    Returns None if unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).lower().strip()
    # Pattern: "3h 45m", "2h15m", "45m", "1h", "90s"
    h = m = sec = 0
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|m|min|mins|s|sec|secs)", s):
        n, unit = float(match.group(1)), match.group(2)
        if unit.startswith("h"):
            h = n
        elif unit.startswith("m"):
            m = n
        elif unit.startswith("s"):
            sec = n
    total = h * 3600 + m * 60 + sec
    if total > 0:
        return total
    # Try plain number string
    try:
        return float(s)
    except ValueError:
        return None


def _extract_root_cause_keywords(text: str) -> list[str]:
    """Extract 2-3 word phrases that look like root cause categories."""
    if not text:
        return []
    text = text.lower()
    # Known patterns
    patterns = [
        "network connectivity", "network issue", "dns failure", "dns resolution",
        "database query", "query optimization", "slow query", "db timeout",
        "resource allocation", "oom", "out of memory", "memory pressure",
        "cpu spike", "high cpu", "cpu throttling",
        "disk full", "disk pressure", "storage exhaustion",
        "config change", "misconfiguration", "bad config",
        "dependency failure", "upstream timeout", "third party",
        "deployment issue", "bad deploy", "rollback",
        "certificate expiry", "tls error", "ssl error",
        "rate limit", "quota exceeded",
        "crashloopbackoff", "crash loop", "container restart",
        "pod eviction", "node pressure",
        "load balancer", "alb error", "target health",
        "autoscaling", "scaling failure",
        "permission denied", "iam error", "auth failure",
    ]
    found = []
    for p in patterns:
        if p in text:
            found.append(p)
    return found[:3]


def analyze_trends(incidents: list[dict]) -> dict:
    """
    Compute trend metrics from a list of incident metadata dicts.

    Returns a dict with:
      - total_incidents
      - mttr_by_type: {type: avg_seconds}
      - top_root_causes: [(cause, count), ...]
      - incident_frequency: {YYYY-WW: count}  (weekly buckets)
      - top_services: [(service, count), ...]
      - severity_distribution: {severity: count}
      - reliability_score: 0-100
      - recurring_patterns: list of insight strings
      - worst_mttr_type: str
      - best_mttr_type: str
    """
    if not incidents:
        return {"total_incidents": 0}

    total = len(incidents)
    mttr_buckets: dict[str, list[float]] = defaultdict(list)
    root_cause_counter: Counter = Counter()
    weekly_freq: Counter = Counter()
    service_counter: Counter = Counter()
    severity_counter: Counter = Counter()

    for inc in incidents:
        inc_type = str(inc.get("type") or inc.get("incident_type") or "unknown").lower()
        root_cause = str(inc.get("root_cause") or inc.get("analysis") or inc.get("_document") or "")
        service = str(inc.get("service") or inc.get("source") or inc.get("resource") or "unknown").lower()
        severity = str(inc.get("severity") or inc.get("severity_ai") or "unknown").lower()
        resolution_raw = inc.get("resolution_time") or inc.get("elapsed_s") or inc.get("duration_s")
        ts_raw = inc.get("timestamp") or inc.get("created_at") or inc.get("ts")

        # MTTR
        res_secs = _parse_resolution_seconds(resolution_raw)
        if res_secs and res_secs > 0:
            mttr_buckets[inc_type].append(res_secs)

        # Root causes
        keywords = _extract_root_cause_keywords(root_cause)
        for kw in keywords:
            root_cause_counter[kw] += 1

        # Weekly frequency
        if ts_raw:
            try:
                ts = float(ts_raw)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                week_key = dt.strftime("%Y-W%W")
                weekly_freq[week_key] += 1
            except (TypeError, ValueError, OSError):
                pass

        # Top services
        if service and service != "unknown":
            service_counter[service] += 1

        # Severity distribution
        if severity and severity != "unknown":
            severity_counter[severity] += 1

    # Compute average MTTR per type (in seconds)
    mttr_by_type = {
        t: round(sum(v) / len(v)) for t, v in mttr_buckets.items() if v
    }

    # Reliability score: penalise high frequency and long MTTR
    # 100 = perfect, 0 = very bad
    score = 100
    if total > 0:
        critical = severity_counter.get("critical", 0)
        high = severity_counter.get("high", 0)
        score -= min(40, (critical * 10 + high * 4))
        avg_mttr_all = (
            sum(sum(v) for v in mttr_buckets.values()) /
            max(1, sum(len(v) for v in mttr_buckets.values()))
        ) if mttr_buckets else 0
        # Penalise if avg MTTR > 1h
        if avg_mttr_all > 3600:
            score -= min(30, int((avg_mttr_all - 3600) / 600))
        # Penalise recurring root causes (same issue >2 times)
        recurring = sum(1 for c in root_cause_counter.values() if c > 2)
        score -= min(20, recurring * 5)
        score = max(0, score)

    # Recurring patterns — actionable insight strings
    patterns: list[str] = []
    for cause, count in root_cause_counter.most_common(5):
        if count >= 2:
            patterns.append(f"'{cause.title()}' recurred {count} times — consider a permanent fix")

    worst = max(mttr_by_type, key=mttr_by_type.get) if mttr_by_type else None
    best = min(mttr_by_type, key=mttr_by_type.get) if mttr_by_type else None

    return {
        "total_incidents": total,
        "mttr_by_type": mttr_by_type,
        "top_root_causes": root_cause_counter.most_common(8),
        "incident_frequency": dict(sorted(weekly_freq.items())),
        "top_services": service_counter.most_common(5),
        "severity_distribution": dict(severity_counter),
        "reliability_score": score,
        "recurring_patterns": patterns,
        "worst_mttr_type": worst,
        "best_mttr_type": best,
    }


def format_trend_report(trends: dict) -> str:
    """Format trend analysis dict into a clean markdown report."""
    if not trends or trends.get("total_incidents", 0) == 0:
        return "No incident history found in memory. Run some incidents first to build up trend data."

    total = trends["total_incidents"]
    score = trends.get("reliability_score", 0)
    score_emoji = "🟢" if score >= 80 else "🟡" if score >= 50 else "🔴"

    lines = [
        f"## 📊 Incident Trend Analysis",
        f"",
        f"**{total} incidents analysed** · Reliability Score: {score_emoji} **{score}/100**",
        "",
    ]

    # MTTR by type
    mttr = trends.get("mttr_by_type", {})
    if mttr:
        lines.append("### ⏱ Mean Time to Resolve (MTTR) by Type")
        for inc_type, secs in sorted(mttr.items(), key=lambda x: x[1], reverse=True):
            h, rem = divmod(int(secs), 3600)
            m = rem // 60
            duration = f"{h}h {m}m" if h else f"{m}m"
            flag = " ⚠️" if secs > 7200 else ""
            lines.append(f"- **{inc_type}**: {duration}{flag}")
        worst = trends.get("worst_mttr_type")
        best = trends.get("best_mttr_type")
        if worst:
            lines.append(f"\n> Slowest to resolve: **{worst}** · Fastest: **{best or '—'}**")
        lines.append("")

    # Severity distribution
    sev = trends.get("severity_distribution", {})
    if sev:
        lines.append("### 🚨 Severity Distribution")
        order = ["critical", "high", "medium", "low"]
        emojis = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        for s in order:
            if s in sev:
                lines.append(f"- {emojis.get(s, '•')} **{s.capitalize()}**: {sev[s]}")
        lines.append("")

    # Top root causes
    causes = trends.get("top_root_causes", [])
    if causes:
        lines.append("### 🔁 Recurring Root Causes")
        for cause, count in causes:
            bar = "█" * min(count, 10)
            lines.append(f"- `{cause.title()}` — {count}x {bar}")
        lines.append("")

    # Top affected services
    services = trends.get("top_services", [])
    if services:
        lines.append("### 🏗 Most Affected Services")
        for svc, count in services:
            lines.append(f"- **{svc}**: {count} incidents")
        lines.append("")

    # Weekly frequency
    freq = trends.get("incident_frequency", {})
    if freq:
        lines.append("### 📅 Incident Frequency (Weekly)")
        weeks = sorted(freq.items())[-8:]  # last 8 weeks
        max_count = max(c for _, c in weeks) if weeks else 1
        for week, count in weeks:
            bar = "█" * round((count / max_count) * 10)
            lines.append(f"- `{week}`: {count} {'▓' * count}")
        lines.append("")

    # Recurring patterns / recommendations
    patterns = trends.get("recurring_patterns", [])
    if patterns:
        lines.append("### 💡 Recommendations")
        for p in patterns:
            lines.append(f"- {p}")
        lines.append("")

    return "\n".join(lines)
