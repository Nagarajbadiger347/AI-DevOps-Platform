"""Incident vector store — PostgreSQL + pgvector (replaces ChromaDB).

Each incident is stored with:
  - Full metadata (type, severity, source, root_cause, resolution)
  - tenant_id for complete isolation between customers
  - 1536-dim embedding vector for semantic similarity search

Usage:
    from app.memory.vector_db import store_incident, search_similar_incidents, get_all_incidents
"""
from __future__ import annotations

import json
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536  # OpenAI/Claude embedding size
_VECTOR_SUPPORTED: bool | None = None


def _vector_supports_pgvector() -> bool:
    global _VECTOR_SUPPORTED
    if _VECTOR_SUPPORTED is not None:
        return _VECTOR_SUPPORTED
    try:
        from app.core.database import execute_one
        row = execute_one("SELECT extname FROM pg_extension WHERE extname = %s", ("vector",))
        _VECTOR_SUPPORTED = bool(row)
    except Exception as exc:
        logger.debug("vector_extension_check_failed error=%s", exc)
        _VECTOR_SUPPORTED = False
    return _VECTOR_SUPPORTED


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _get_embedding(text: str) -> list[float] | None:
    """Generate embedding vector for text. Returns None if unavailable."""
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding
    except Exception as e:
        logger.debug("embedding_failed error=%s", e)
        return None


def _db():
    from app.core.database import execute, execute_one
    return execute, execute_one


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_incident(incident: dict, tenant_id: str = "default") -> dict:
    """Store an incident in PostgreSQL with optional vector embedding."""
    execute, execute_one = _db()
    try:
        doc_id      = str(incident.get("id", incident.get("incident_id", "unknown")))
        inc_type    = incident.get("type", incident.get("incident_type", "unknown"))
        source      = incident.get("source", "unknown")
        description = incident.get("description", incident.get("payload", str(incident)))
        root_cause  = incident.get("root_cause", incident.get("ai_root_cause", ""))
        resolution  = incident.get("resolution", "")
        severity    = incident.get("severity", "SEV3")
        actions     = incident.get("actions_taken", incident.get("executed_actions", []))

        # Generate embedding for semantic search
        embed_text = f"{inc_type} {source} {description} {root_cause}"
        embedding = _get_embedding(embed_text)
        embed_str = f"[{','.join(str(v) for v in embedding)}]" if embedding else None
        embedding_json = json.dumps(embedding) if embedding else None

        metadata = {k: v for k, v in incident.items()
                    if k not in ("id", "incident_id", "type", "source", "description",
                                 "root_cause", "resolution", "severity", "actions_taken")}

        if embed_str and _vector_supports_pgvector():
            execute(
                """
                INSERT INTO incidents
                    (incident_id, tenant_id, incident_type, source, description,
                     root_cause, resolution, severity, actions_taken, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (tenant_id, incident_id) DO UPDATE SET
                    root_cause = EXCLUDED.root_cause,
                    resolution = EXCLUDED.resolution,
                    embedding  = EXCLUDED.embedding,
                    metadata   = EXCLUDED.metadata
                """,
                (doc_id, tenant_id, inc_type, source, str(description),
                 root_cause, resolution, severity,
                 json.dumps(actions if isinstance(actions, list) else []),
                 json.dumps(metadata), embed_str)
            )
        else:
            execute(
                """
                INSERT INTO incidents
                    (incident_id, tenant_id, incident_type, source, description,
                     root_cause, resolution, severity, actions_taken, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, incident_id) DO UPDATE SET
                    root_cause = EXCLUDED.root_cause,
                    resolution = EXCLUDED.resolution,
                    metadata   = EXCLUDED.metadata,
                    embedding  = EXCLUDED.embedding
                """,
                (doc_id, tenant_id, inc_type, source, str(description),
                 root_cause, resolution, severity,
                 json.dumps(actions if isinstance(actions, list) else []),
                 json.dumps(metadata), embedding_json)
            )
        return {"stored": True, "id": doc_id}
    except Exception as e:
        logger.warning("store_incident_failed error=%s", e)
        return {"stored": False, "error": str(e)}


def search_similar_incidents(query: str, n_results: int = 5, tenant_id: str = "default") -> list[dict]:
    """Search for semantically similar past incidents using pgvector cosine similarity."""
    execute, _ = _db()
    try:
        embedding = _get_embedding(query)
        if embedding and _vector_supports_pgvector():
            embed_str = f"[{','.join(str(v) for v in embedding)}]"
            rows = execute(
                """
                SELECT *, 1 - (embedding <=> %s::vector) AS _similarity
                FROM incidents
                WHERE tenant_id = %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embed_str, tenant_id, embed_str, n_results)
            )
        else:
            # Fallback: keyword search on description
            rows = execute(
                """
                SELECT *, 0.5 AS _similarity
                FROM incidents
                WHERE tenant_id = %s AND description ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (tenant_id, f"%{query}%", n_results)
            )
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("search_similar_incidents_failed error=%s", e)
        return [{"error": str(e)}]


def search_incidents_by_type(incident_type: str, n_results: int = 20, tenant_id: str = "default") -> list[dict]:
    """Fetch incidents filtered by type."""
    execute, _ = _db()
    try:
        rows = execute(
            "SELECT * FROM incidents WHERE tenant_id = %s AND incident_type = %s ORDER BY created_at DESC LIMIT %s",
            (tenant_id, incident_type, n_results)
        )
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("search_incidents_by_type_failed error=%s", e)
        return search_similar_incidents(incident_type, n_results, tenant_id)


def get_all_incidents(limit: int = 200, tenant_id: str = "default") -> list[dict]:
    """Fetch all stored incidents for trend analysis."""
    execute, _ = _db()
    try:
        rows = execute(
            "SELECT * FROM incidents WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s",
            (tenant_id, limit)
        )
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        logger.warning("get_all_incidents_failed error=%s", e)
        return []


def delete_incident(incident_id: str, tenant_id: str = "default") -> dict:
    execute, _ = _db()
    try:
        rows = execute(
            "DELETE FROM incidents WHERE incident_id = %s AND tenant_id = %s RETURNING incident_id",
            (incident_id, tenant_id)
        )
        return {"deleted": len(rows) > 0, "id": incident_id}
    except Exception as e:
        return {"deleted": False, "error": str(e)}


def backup_chromadb() -> dict:
    """No-op — PostgreSQL handles backups via pg_dump or managed service."""
    return {"success": True, "message": "PostgreSQL does not require manual backups"}


def get_backup_list() -> list[dict]:
    return []


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _row_to_dict(row: dict) -> dict:
    result = dict(row)
    # Parse JSON fields
    for field in ("actions_taken", "metadata"):
        val = result.get(field)
        if isinstance(val, str):
            try:
                result[field] = json.loads(val)
            except Exception:
                pass
    # Convert timestamps to string
    for field in ("created_at", "resolved_at"):
        if result.get(field) is not None:
            result[field] = str(result[field])
    # Remove embedding from output (large vector, not needed in results)
    result.pop("embedding", None)
    return result
