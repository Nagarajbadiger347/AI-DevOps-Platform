"""
Export NexusOps incident data into JSONL fine-tuning format for Llama 3 / Mistral.

Sources:
  - ChromaDB (incident memory)
  - post_mortems/ (markdown post-mortems)
  - chat_history.db (war room conversations)

Output:
  - data/finetune_dataset.jsonl   — Alpaca-style prompt/response pairs
  - data/finetune_stats.json      — dataset stats

Usage:
    python scripts/export_training_data.py
    python scripts/export_training_data.py --output data/my_dataset.jsonl --min-quality 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH     = os.getenv("CHROMA_DB_PATH", "./chroma_db")
POST_MORTEM_DIR = Path("./post_mortems")
CHAT_DB_PATH    = Path("./app/chat/chat_history.db")
OUTPUT_PATH     = Path("./data/finetune_dataset.jsonl")
STATS_PATH      = Path("./data/finetune_stats.json")

SYSTEM_PROMPT = (
    "You are NexusOps, an expert AI DevOps assistant specialized in incident detection, "
    "root cause analysis, and automated remediation for Kubernetes, AWS, and Linux infrastructure. "
    "Provide concise, actionable responses with specific commands and steps."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def alpaca(instruction: str, response: str, input_ctx: str = "") -> dict:
    """Alpaca-style training record (works with Unsloth, axolotl, LLaMA-Factory)."""
    return {
        "instruction": instruction.strip(),
        "input": input_ctx.strip(),
        "output": response.strip(),
        "system": SYSTEM_PROMPT,
    }


def clean(text: str) -> str:
    """Remove excessive whitespace."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Source 1: ChromaDB incidents
# ---------------------------------------------------------------------------

def load_chromadb_incidents() -> list[dict]:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection("incidents")
        results = collection.get(limit=1000, include=["metadatas", "documents"])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        records = []
        for i, meta in enumerate(metadatas):
            entry = dict(meta)
            if i < len(documents):
                entry["_document"] = documents[i]
            records.append(entry)
        print(f"  [chromadb] loaded {len(records)} incidents")
        return records
    except Exception as e:
        print(f"  [chromadb] failed: {e}")
        return []


def incidents_to_training(incidents: list[dict]) -> list[dict]:
    """Convert raw incident records into prompt/response pairs."""
    pairs = []

    for inc in incidents:
        inc_type    = inc.get("type", inc.get("incident_type", "unknown"))
        source      = inc.get("source", "unknown")
        description = inc.get("description", inc.get("_document", ""))
        root_cause  = inc.get("root_cause", inc.get("ai_root_cause", ""))
        resolution  = inc.get("resolution", inc.get("remediation", ""))
        severity    = inc.get("severity", "SEV3")
        actions     = inc.get("actions_taken", inc.get("executed_actions", ""))

        # Skip low-quality records
        if not description or len(description) < 20:
            continue

        # Pair 1: Incident analysis
        if root_cause and len(root_cause) > 20:
            pairs.append(alpaca(
                instruction=f"Analyze this {inc_type} incident from {source} and identify the root cause.",
                input_ctx=f"Alert: {description}\nSeverity: {severity}",
                response=f"Root Cause: {root_cause}",
            ))

        # Pair 2: Remediation steps
        if resolution and len(resolution) > 20:
            pairs.append(alpaca(
                instruction=f"What remediation steps should be taken for this {inc_type} incident?",
                input_ctx=f"Incident: {description}\nRoot cause: {root_cause or 'under investigation'}",
                response=resolution,
            ))

        # Pair 3: Actions taken (what actually worked)
        if actions and len(str(actions)) > 20:
            if isinstance(actions, str):
                actions_str = actions
            elif isinstance(actions, list):
                actions_str = "\n".join(f"- {a}" for a in actions if a)
            else:
                actions_str = json.dumps(actions)

            pairs.append(alpaca(
                instruction=f"List the actions that were executed to resolve this {inc_type} incident.",
                input_ctx=f"Incident: {description}",
                response=actions_str,
            ))

    return pairs


# ---------------------------------------------------------------------------
# Source 2: Post-mortem markdown files
# ---------------------------------------------------------------------------

def load_post_mortems() -> list[dict]:
    if not POST_MORTEM_DIR.exists():
        print(f"  [post_mortems] directory not found: {POST_MORTEM_DIR}")
        return []

    records = []
    for path in POST_MORTEM_DIR.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        records.append({"file": path.name, "content": text})

    print(f"  [post_mortems] loaded {len(records)} files")
    return records


def _extract_section(text: str, heading: str) -> str:
    """Extract content under a markdown heading."""
    pattern = rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def post_mortems_to_training(records: list[dict]) -> list[dict]:
    pairs = []

    for rec in records:
        text = rec["content"]

        root_cause   = _extract_section(text, "Root Cause")
        resolution   = _extract_section(text, "Resolution")
        lessons      = _extract_section(text, "Lessons Learned")
        prevention   = _extract_section(text, "Prevention Steps")
        timeline     = _extract_section(text, "Timeline")
        action_items = _extract_section(text, "Action Items")
        impact       = _extract_section(text, "Impact")
        summary      = _extract_section(text, "Executive Summary")

        # Extract incident ID and severity from header
        inc_id   = re.search(r"\*\*Incident ID:\*\*\s*(.+)", text)
        severity = re.search(r"\*\*Severity:\*\*\s*(.+)", text)
        inc_id   = inc_id.group(1).strip() if inc_id else "UNKNOWN"
        severity = severity.group(1).strip() if severity else "SEV2"

        context = f"Incident: {inc_id}, Severity: {severity}\nImpact: {impact or summary}"

        if root_cause and len(root_cause) > 20 and "LLM" not in root_cause:
            pairs.append(alpaca(
                instruction="What was the root cause of this incident?",
                input_ctx=context,
                response=root_cause,
            ))

        if resolution and len(resolution) > 20 and "parsing failed" not in resolution:
            pairs.append(alpaca(
                instruction="How was this incident resolved?",
                input_ctx=context,
                response=resolution,
            ))

        if lessons and "_None recorded_" not in lessons:
            pairs.append(alpaca(
                instruction="What lessons were learned from this incident?",
                input_ctx=context,
                response=lessons,
            ))

        if prevention and "_None recorded_" not in prevention:
            pairs.append(alpaca(
                instruction="What prevention steps should be implemented after this incident?",
                input_ctx=context,
                response=prevention,
            ))

        if timeline and "_No timeline_" not in timeline:
            pairs.append(alpaca(
                instruction="Describe the incident timeline.",
                input_ctx=context,
                response=timeline,
            ))

        if action_items and "_No action items_" not in action_items:
            pairs.append(alpaca(
                instruction="What action items were created from this incident?",
                input_ctx=context,
                response=action_items,
            ))

    return pairs


# ---------------------------------------------------------------------------
# Source 3: Chat history (war room conversations)
# ---------------------------------------------------------------------------

def load_chat_history() -> list[dict]:
    if not CHAT_DB_PATH.exists():
        print(f"  [chat_db] not found: {CHAT_DB_PATH}")
        return []

    try:
        conn = sqlite3.connect(str(CHAT_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Get sessions with their messages
        cur.execute("SELECT session_id FROM sessions ORDER BY last_active DESC LIMIT 200")
        sessions = [r["session_id"] for r in cur.fetchall()]

        records = []
        for sid in sessions:
            cur.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY timestamp ASC",
                (sid,)
            )
            messages = [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]
            if messages:
                records.append({"session_id": sid, "messages": messages})

        conn.close()
        print(f"  [chat_db] loaded {len(records)} sessions")
        return records
    except Exception as e:
        print(f"  [chat_db] failed: {e}")
        return []


def chat_to_training(records: list[dict]) -> list[dict]:
    """Convert chat sessions into instruction/response pairs."""
    pairs = []

    for rec in records:
        messages = rec["messages"]
        # Slide over user→assistant pairs
        for i in range(len(messages) - 1):
            user_msg = messages[i]
            asst_msg = messages[i + 1]
            if user_msg["role"] == "user" and asst_msg["role"] == "assistant":
                question = user_msg["content"].strip()
                answer   = asst_msg["content"].strip()
                # Skip very short or generic exchanges
                if len(question) < 15 or len(answer) < 30:
                    continue
                pairs.append(alpaca(
                    instruction=question,
                    response=answer,
                ))

    return pairs


# ---------------------------------------------------------------------------
# Dedup + quality filter
# ---------------------------------------------------------------------------

def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    out  = []
    for p in pairs:
        key = (p["instruction"][:100], p["output"][:100])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def quality_filter(pairs: list[dict], min_output_len: int = 30) -> list[dict]:
    bad_phrases = {"not determined", "lllm", "parsing failed", "none recorded", "see incident description"}
    out = []
    for p in pairs:
        output = p["output"].lower()
        if len(p["output"]) < min_output_len:
            continue
        if any(bp in output for bp in bad_phrases):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export NexusOps data to fine-tuning JSONL")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--min-output-len", type=int, default=30)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== NexusOps Fine-Tuning Data Exporter ===\n")

    all_pairs = []

    print("[1/3] Loading ChromaDB incidents...")
    incidents = load_chromadb_incidents()
    chroma_pairs = incidents_to_training(incidents)
    print(f"      → {len(chroma_pairs)} training pairs")
    all_pairs.extend(chroma_pairs)

    print("[2/3] Loading post-mortems...")
    post_mortems = load_post_mortems()
    pm_pairs = post_mortems_to_training(post_mortems)
    print(f"      → {len(pm_pairs)} training pairs")
    all_pairs.extend(pm_pairs)

    print("[3/3] Loading chat history...")
    chats = load_chat_history()
    chat_pairs = chat_to_training(chats)
    print(f"      → {len(chat_pairs)} training pairs")
    all_pairs.extend(chat_pairs)

    print("\n[filtering] Deduplicating and quality filtering...")
    all_pairs = deduplicate(all_pairs)
    all_pairs = quality_filter(all_pairs, min_output_len=args.min_output_len)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Write stats
    stats = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_pairs": len(all_pairs),
        "sources": {
            "chromadb_incidents": len(chroma_pairs),
            "post_mortems": len(pm_pairs),
            "chat_history": len(chat_pairs),
        },
        "output_file": str(output_path.resolve()),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Exported {len(all_pairs)} training pairs → {output_path}")
    print(f"✓ Stats → {STATS_PATH}")
    print("\nNext steps:")
    print("  1. Review data/finetune_dataset.jsonl")
    print("  2. Upload to RunPod + fine-tune with Unsloth (~$20, 2-4 hrs)")
    print("  3. Run: python scripts/setup_ollama_provider.py")
    print()

    if len(all_pairs) < 50:
        print("⚠  WARNING: Less than 50 pairs. Use the platform more to generate richer data.")
        print("   Tip: trigger test incidents, resolve them, let war room AI run — all of that becomes training data.")


if __name__ == "__main__":
    main()
