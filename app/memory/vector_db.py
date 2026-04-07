import json
import os
import shutil
import datetime
import pathlib
import chromadb
from chromadb.config import Settings

_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_BACKUP_DIR  = pathlib.Path(os.getenv("CHROMA_BACKUP_DIR", "./chroma_backups"))

# Initialize Chroma client
client = chromadb.PersistentClient(path=_CHROMA_PATH)
collection = client.get_or_create_collection(name="incidents")


def _flatten_metadata(data: dict) -> dict:
    """Flatten a dict so all values are ChromaDB-compatible primitives."""
    flat = {}
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v
        else:
            flat[k] = json.dumps(v)
    return flat


def store_incident(incident: dict) -> dict:
    """Store incident in vector database."""
    try:
        doc_id = str(incident.get("id", "unknown"))
        content = f"Incident: {incident.get('type', 'unknown')} from {incident.get('source', 'unknown')} - {incident.get('payload', {})}"

        collection.add(
            documents=[content],
            metadatas=[_flatten_metadata(incident)],
            ids=[doc_id]
        )
        return {"stored": True, "id": doc_id}
    except Exception as e:
        return {"stored": False, "error": str(e)}


def backup_chromadb() -> dict:
    """Copy the ChromaDB directory to a timestamped backup folder.

    Keeps the last 7 daily backups and deletes older ones automatically.
    Returns {"success": True, "backup_path": "..."} or {"success": False, "error": "..."}.
    """
    try:
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest = _BACKUP_DIR / f"chroma_{timestamp}"
        shutil.copytree(_CHROMA_PATH, str(dest))

        # Prune: keep only the 7 most recent backups
        backups = sorted(_BACKUP_DIR.glob("chroma_*"), key=lambda p: p.name)
        for old in backups[:-7]:
            shutil.rmtree(old, ignore_errors=True)

        return {"success": True, "backup_path": str(dest), "timestamp": timestamp}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_backup_list() -> list[dict]:
    """Return metadata for all available ChromaDB backups."""
    try:
        if not _BACKUP_DIR.exists():
            return []
        backups = sorted(_BACKUP_DIR.glob("chroma_*"), key=lambda p: p.name, reverse=True)
        return [
            {
                "name": p.name,
                "path": str(p),
                "size_mb": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024, 2),
            }
            for p in backups
        ]
    except Exception:
        return []


def search_similar_incidents(query: str, n_results: int = 5) -> list:
    """Search for similar past incidents."""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results.get("metadatas", [])
    except Exception as e:
        return [{"error": str(e)}]
