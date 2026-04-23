"""
Long-term memory — persisted incident history in PostgreSQL + pgvector.
Used for semantic similarity retrieval during planning.
"""
from __future__ import annotations

import logging
from typing import Any

from app.memory.vector_db import (
    store_incident,
    search_similar_incidents,
    search_incidents_by_type,
    get_all_incidents,
)
from app.memory.trend_analysis import analyze_trends, format_trend_report

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.6
JUNK_PHRASES = frozenset({"planning failed", "test incident", "dummy", "rate_limit"})


def store(incident: dict, tenant_id: str = "default") -> bool:
    confidence  = incident.get("confidence", 0.0)
    description = incident.get("description", "")

    if confidence < MIN_CONFIDENCE:
        logger.debug("long_term_skip_low_confidence incident=%s conf=%.2f",
                     incident.get("incident_id"), confidence)
        return False

    if _is_junk(description):
        logger.debug("long_term_skip_junk incident=%s", incident.get("incident_id"))
        return False

    try:
        store_incident(incident, tenant_id=tenant_id)
        return True
    except Exception as exc:
        logger.warning("long_term_store_failed error=%s", exc)
        return False


def retrieve_similar(query: str, n_results: int = 5, tenant_id: str = "default") -> list[dict[str, Any]]:
    try:
        results = search_similar_incidents(query, n_results=n_results, tenant_id=tenant_id)
        if results and isinstance(results[0], list):
            results = results[0]
        return [r for r in results if isinstance(r, dict)]
    except Exception as exc:
        logger.warning("long_term_retrieve_failed error=%s", exc)
        return []


def retrieve_by_type(incident_type: str, n_results: int = 20, tenant_id: str = "default") -> list[dict]:
    try:
        results = search_incidents_by_type(incident_type, n_results=n_results, tenant_id=tenant_id)
        if results and isinstance(results[0], list):
            results = results[0]
        return [r for r in results if isinstance(r, dict) and "error" not in r]
    except Exception as exc:
        logger.warning("long_term_retrieve_by_type_failed error=%s", exc)
        return []


def get_trends(tenant_id: str = "default") -> dict:
    try:
        incidents = get_all_incidents(limit=200, tenant_id=tenant_id)
        return analyze_trends(incidents)
    except Exception as exc:
        logger.warning("long_term_trends_failed error=%s", exc)
        return {"total_incidents": 0}


def get_trend_report(tenant_id: str = "default") -> str:
    trends = get_trends(tenant_id)
    return format_trend_report(trends)


def _is_junk(description: str) -> bool:
    low = description.lower()
    return len(description) < 20 or any(p in low for p in JUNK_PHRASES)
