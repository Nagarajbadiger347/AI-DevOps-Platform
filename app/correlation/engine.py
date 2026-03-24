from typing import List, Dict, Any
from collections import Counter


def correlate_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Correlate events to find root causes."""
    if not events:
        return {"root_cause": "No events provided", "confidence": 0.0}

    # Simple correlation logic
    sources = [e.get("source", "unknown") for e in events]
    types = [e.get("type", "unknown") for e in events]

    source_counts = Counter(sources)
    type_counts = Counter(types)

    # Determine root cause based on patterns
    if type_counts.get("error", 0) > len(events) * 0.5:
        root_cause = "Multiple errors detected"
        confidence = 0.8
    elif source_counts.get("database", 0) > 0:
        root_cause = "Database connectivity issue"
        confidence = 0.9
    elif source_counts.get("network", 0) > 0:
        root_cause = "Network connectivity issue"
        confidence = 0.85
    elif type_counts.get("timeout", 0) > 0:
        root_cause = "Service timeout"
        confidence = 0.75
    else:
        root_cause = "General system issue"
        confidence = 0.6

    return {
        "root_cause": root_cause,
        "confidence": confidence,
        "event_count": len(events),
        "top_sources": dict(source_counts.most_common(3)),
        "top_types": dict(type_counts.most_common(3))
    }
