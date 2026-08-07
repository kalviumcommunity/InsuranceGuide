"""Batch-embeds chunks with retry/backoff, skip-on-rerun, and cost tracking.

Scales the single-chunk-per-request approach in create_embeddings.py into a
pipeline that is efficient (one API call per batch instead of per chunk),
resilient (retries transient failures with exponential backoff), safe to
re-run (skips chunks that already have a stored embedding), and auditable
(a run summary with totals, failures, and an approximate cost).
"""

import json
import os
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from google import genai

from chunk_metadata import tag_chunks
from document_loader import load_documents
from text_cleaning import clean

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not embedding_model:
    raise ValueError("EMBEDDING_MODEL not found in .env")

client = genai.Client(api_key=api_key)
encoding = tiktoken.get_encoding("cl100k_base")

DATA_FOLDER = "data"
STORE_FILE = Path(__file__).resolve().parent.parent / "outputs" / "embeddings_store.json"
SUMMARY_FILE = Path(__file__).resolve().parent.parent / "outputs" / "batch_embedding_summary.txt"

BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
MAX_ATTEMPTS = 5

# Approximate — replace with the real per-1K-token price for embedding_model.
PRICE_PER_1K_TOKENS = 0.00002


def chunk_id(metadata):
    """A stable identifier for a chunk, used to detect already-embedded chunks."""
    return f"{metadata['source']}::{metadata['chunk_index']}"


def load_all_chunks():
    documents = load_documents(DATA_FOLDER)
    all_chunks = []
    for document in documents:
        cleaned_document = {"source": document["source"], "text": clean(document["text"])}
        all_chunks.extend(tag_chunks(cleaned_document))
    return all_chunks


def load_store():
    if STORE_FILE.exists():
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    return []


def save_store(records):
    STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STORE_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def estimate_tokens(texts):
    return sum(len(encoding.encode(text)) for text in texts)


def embed_with_retry(texts, max_attempts=MAX_ATTEMPTS):
    for attempt in range(max_attempts):
        try:
            response = client.models.embed_content(model=embedding_model, contents=texts)
            return [embedding.values for embedding in response.embeddings]
        except Exception as error:
            if attempt == max_attempts - 1:
                raise
            wait_seconds = 2 ** attempt
            print(f"  retrying after error: {error} | wait={wait_seconds}s")
            time.sleep(wait_seconds)


def main(batch_size=BATCH_SIZE):
    print("=" * 70)
    print("BATCH EMBEDDING RUN")
    print("=" * 70)

    all_chunks = load_all_chunks()
    for chunk in all_chunks:
        chunk["id"] = chunk_id(chunk["metadata"])

    existing_records = load_store()
    existing_ids = {record["id"] for record in existing_records}

    pending_chunks = [chunk for chunk in all_chunks if chunk["id"] not in existing_ids]

    summary = {
        "total_chunks": len(all_chunks),
        "skipped_existing": len(all_chunks) - len(pending_chunks),
        "embedded": 0,
        "failed": 0,
        "failed_batches": [],
        "input_tokens": 0,
    }

    new_records = []

    for batch_number, batch in enumerate(batches(pending_chunks, batch_size), start=1):
        texts = [chunk["text"] for chunk in batch]
        summary["input_tokens"] += estimate_tokens(texts)

        print(f"\nBatch {batch_number}: embedding {len(batch)} chunk(s)")

        try:
            vectors = embed_with_retry(texts)
        except Exception as error:
            print(f"  batch {batch_number} failed permanently: {error}")
            summary["failed"] += len(batch)
            summary["failed_batches"].append({"batch": batch_number, "error": str(error)})
            continue

        for chunk, vector in zip(batch, vectors):
            new_records.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": vector,
            })
            summary["embedded"] += 1

    all_records = existing_records + new_records
    save_store(all_records)

    estimated_cost = summary["input_tokens"] / 1000 * PRICE_PER_1K_TOKENS

    lines = ["BATCH EMBEDDING RUN SUMMARY", "=" * 70]
    lines.append(f"Batch size            : {batch_size}")
    lines.append(f"Total chunks          : {summary['total_chunks']}")
    lines.append(f"Skipped (existing)    : {summary['skipped_existing']}")
    lines.append(f"Embedded (this run)   : {summary['embedded']}")
    lines.append(f"Failed                : {summary['failed']}")
    if summary["failed_batches"]:
        lines.append("Failed batches:")
        for failure in summary["failed_batches"]:
            lines.append(f"  - batch {failure['batch']}: {failure['error']}")
    lines.append(f"Input tokens (approx) : {summary['input_tokens']}")
    lines.append(f"Estimated cost (USD)  : {estimated_cost:.6f}")
    lines.append(f"\nTotal stored embeddings: {len(all_records)}")

    report = "\n".join(lines)
    print("\n" + report)

    SUMMARY_FILE.write_text(report, encoding="utf-8")
    print(f"\nSummary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
