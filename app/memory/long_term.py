"""
Long-term memory — persisted incident history in ChromaDB.
Used for semantic similarity retrieval during planning.

Wraps vector_db.py with deduplication and quality control.
"""
from __future__ import annotations

import logging
import time
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
MIN_RELEVANCE_SCORE = 0.5
JUNK_PHRASES = frozenset({"planning failed", "test incident", "dummy", "rate_limit"})


def store(incident: dict) -> bool:
    """
    Store an incident to long-term memory.
    Returns False if quality check fails (low confidence, junk, duplicate).
    """
    confidence = incident.get("confidence", 0.0)
    description = incident.get("description", "")

    if confidence < MIN_CONFIDENCE:
        logger.debug("long_term_skip_low_confidence incident=%s conf=%.2f",
                     incident.get("incident_id"), confidence)
        return False

    if _is_junk(description):
        logger.debug("long_term_skip_junk incident=%s", incident.get("incident_id"))
        return False

    try:
        store_incident(incident)
        return True
    except Exception as exc:
        logger.warning("long_term_store_failed error=%s", exc)
        return False


def retrieve_similar(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Retrieve semantically similar incidents, filtered by relevance threshold."""
    try:
        results = search_similar_incidents(query, n_results=n_results)
        if results and isinstance(results[0], list):
            results = results[0]
        return [r for r in results if isinstance(r, dict)]
    except Exception as exc:
        logger.warning("long_term_retrieve_failed error=%s", exc)
        return []


def retrieve_by_type(incident_type: str, n_results: int = 20) -> list[dict]:
    """Retrieve incidents filtered by type for trend analysis."""
    try:
        results = search_incidents_by_type(incident_type, n_results=n_results)
        if results and isinstance(results[0], list):
            results = results[0]
        return [r for r in results if isinstance(r, dict) and "error" not in r]
    except Exception as exc:
        logger.warning("long_term_retrieve_by_type_failed error=%s", exc)
        return []


def get_trends() -> dict:
    """Compute trend analysis across all stored incidents."""
    try:
        incidents = get_all_incidents(limit=200)
        return analyze_trends(incidents)
    except Exception as exc:
        logger.warning("long_term_trends_failed error=%s", exc)
        return {"total_incidents": 0}


def get_trend_report() -> str:
    """Return a formatted markdown trend report for all stored incidents."""
    trends = get_trends()
    return format_trend_report(trends)


def _is_junk(description: str) -> bool:
    low = description.lower()
    return len(description) < 20 or any(p in low for p in JUNK_PHRASES)
