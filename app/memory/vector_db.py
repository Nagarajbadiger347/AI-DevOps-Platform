import json
import chromadb
from chromadb.config import Settings

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db")
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
