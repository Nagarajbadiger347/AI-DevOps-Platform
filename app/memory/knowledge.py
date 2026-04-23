"""
Knowledge base — reusable infra patterns seeded from post-mortems and runbooks.
Stored in PostgreSQL incidents table with type='knowledge_pattern'.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeBase:

    def __init__(self, tenant_id: str = "default") -> None:
        self.tenant_id = tenant_id
        try:
            from app.core.database import health_check
            self._available = health_check()
        except Exception as exc:
            logger.warning("knowledge_base_unavailable error=%s", exc)
            self._available = False

    def add_pattern(self, pattern_id: str, description: str, solution: str, tags: list[str]) -> bool:
        if not self._available:
            return False
        try:
            from app.memory.vector_db import store_incident
            store_incident({
                "id": f"pattern_{pattern_id}",
                "type": "knowledge_pattern",
                "source": "knowledge_base",
                "description": description,
                "resolution": solution,
                "metadata": {"tags": tags},
            }, tenant_id=self.tenant_id)
            return True
        except Exception as exc:
            logger.warning("knowledge_add_failed pattern=%s error=%s", pattern_id, exc)
            return False

    def search(self, query: str, n: int = 3) -> list[dict[str, Any]]:
        if not self._available:
            return []
        try:
            from app.memory.vector_db import search_similar_incidents
            results = search_similar_incidents(query, n_results=n, tenant_id=self.tenant_id)
            return [
                {
                    "description": r.get("description", ""),
                    "solution": r.get("resolution", ""),
                    "tags": r.get("metadata", {}).get("tags", []),
                }
                for r in results if r.get("incident_type") == "knowledge_pattern"
            ]
        except Exception as exc:
            logger.warning("knowledge_search_failed error=%s", exc)
            return []
