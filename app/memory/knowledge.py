"""
Knowledge base — reusable infra patterns seeded from post-mortems and runbooks.
Low write frequency, high read frequency. Never auto-generated.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeBase:
    COLLECTION = "infra_patterns"

    def __init__(self) -> None:
        try:
            import chromadb
            from app.core.config import settings
            client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self._col = client.get_or_create_collection(self.COLLECTION)
            self._available = True
        except Exception as exc:
            logger.warning("knowledge_base_unavailable error=%s", exc)
            self._available = False
            self._col = None

    def add_pattern(
        self,
        pattern_id: str,
        description: str,
        solution: str,
        tags: list[str],
    ) -> bool:
        if not self._available:
            return False
        try:
            self._col.add(
                documents=[description],
                metadatas=[{"solution": solution, "tags": ",".join(tags)}],
                ids=[pattern_id],
            )
            return True
        except Exception as exc:
            logger.warning("knowledge_add_failed pattern=%s error=%s", pattern_id, exc)
            return False

    def search(self, query: str, n: int = 3) -> list[dict[str, Any]]:
        if not self._available:
            return []
        try:
            results = self._col.query(
                query_texts=[query],
                n_results=n,
                include=["documents", "metadatas"],
            )
            return [
                {"description": d, **m}
                for d, m in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                )
            ]
        except Exception as exc:
            logger.warning("knowledge_search_failed error=%s", exc)
            return []
