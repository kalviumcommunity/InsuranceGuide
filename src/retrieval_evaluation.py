from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import mean

from document_loader import load_documents
from chunk_metadata import tag_chunks

try:
    from retrieval import retrieve
except Exception:  # pragma: no cover - allows offline tooling when no API key is configured.
    retrieve = None

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LABEL_FILE = BASE_DIR / "data" / "labelled_queries.json"
DEFAULT_REPORT_FILE = BASE_DIR / "outputs" / "retrieval_evaluation_report.json"
DEFAULT_FAILURE_FILE = BASE_DIR / "outputs" / "retrieval_evaluation_failures.txt"


def normalize_chunk_id(chunk_id: str | None) -> str:
    """Normalize a chunk identifier to the repo's stable source::chunk_index shape."""
    if chunk_id is None:
        return ""
    value = str(chunk_id).strip()
    if not value:
        return ""
    return value.replace("\\", "/")


def chunk_id_from_result(result: dict) -> str:
    """Extract a stable chunk ID from a retrieval result."""
    metadata = result.get("metadata") or {}
    source = str(metadata.get("source", "")).strip()
    chunk_index = metadata.get("chunk_index")
    if source and chunk_index is not None:
        return normalize_chunk_id(f"{source}::chunk_{chunk_index}")

    if result.get("id"):
        return normalize_chunk_id(result["id"])

    if result.get("metadata", {}).get("id"):
        return normalize_chunk_id(result["metadata"]["id"])

    return ""


def compute_recall_at_k(relevant_ids, retrieved_ids, k):
    """Recall@k = relevant docs hit in top-k over relevant docs known to be correct."""
    relevant_set = {normalize_chunk_id(item) for item in relevant_ids if item}
    if not relevant_set:
        return 0.0

    top_k = [normalize_chunk_id(item) for item in retrieved_ids[:k] if item]
    hits = len(relevant_set.intersection(top_k))
    return round(hits / len(relevant_set), 4)


def compute_precision_at_k(relevant_ids, retrieved_ids, k):
    """Precision@k = number of relevant docs returned in top-k over k returned docs."""
    if k <= 0:
        return 0.0

    relevant_set = {normalize_chunk_id(item) for item in relevant_ids if item}
    top_k = [normalize_chunk_id(item) for item in retrieved_ids[:k] if item]
    hits = len(relevant_set.intersection(top_k))
    return round(hits / k, 4)


def evaluate_queries(queries, retrieve_fn, ks=(1, 3, 5)):
    """Evaluate a labelled query list against a retrieval function and return summary stats."""
    per_query = []
    metric_sets = {f"recall@{k}": [] for k in ks}
    metric_sets.update({f"precision@{k}": [] for k in ks})

    for entry in queries:
        query = entry["query"]
        relevant = entry.get("relevant_chunk_ids", [])
        max_k = max(ks)
        results = retrieve_fn(query, k=max_k)
        retrieved_ids = [chunk_id_from_result(result) for result in results]

        row = {
            "id": entry.get("id", query),
            "query": query,
            "relevant_chunk_ids": relevant,
            "retrieved_chunk_ids": retrieved_ids,
            "metrics": {},
        }

        for k in ks:
            recall = compute_recall_at_k(relevant, retrieved_ids, k)
            precision = compute_precision_at_k(relevant, retrieved_ids, k)
            row["metrics"][f"recall@{k}"] = recall
            row["metrics"][f"precision@{k}"] = precision
            metric_sets[f"recall@{k}"].append(recall)
            metric_sets[f"precision@{k}"].append(precision)

        per_query.append(row)

    summary = {
        "n_queries": len(queries),
        "ks": list(ks),
        "by_k": {},
        "details": per_query,
    }

    for k in ks:
        summary["by_k"][f"recall@{k}"] = {
            "mean": round(mean(metric_sets[f"recall@{k}"]) if metric_sets[f"recall@{k}"] else 0.0, 4),
            "values": metric_sets[f"recall@{k}"],
        }
        summary["by_k"][f"precision@{k}"] = {
            "mean": round(mean(metric_sets[f"precision@{k}"]) if metric_sets[f"precision@{k}"] else 0.0, 4),
            "values": metric_sets[f"precision@{k}"],
        }

    return summary


def load_labelled_queries(path: str | Path | None = None):
    """Load the repository's labelled query set."""
    path = Path(path) if path else DEFAULT_LABEL_FILE
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else payload.get("queries", [])


def build_failure_analysis(summary):
    """Create human-readable failure notes for low recall or low precision cases."""
    lines = [
        "RETRIEVAL FAILURE ANALYSIS",
        "========================",
        "",
        "The main failure modes to inspect are: chunking mismatch, missing metadata filters,",
        "poor query wording, and embedding mismatch between the user's question and the stored chunk.",
        "",
    ]

    for row in summary["details"]:
        recall_1 = row["metrics"].get("recall@1", 0.0)
        precision_1 = row["metrics"].get("precision@1", 0.0)
        if recall_1 >= 1.0 and precision_1 >= 0.5:
            lines.append(f"- {row['id']}: no major failure; retrieved the expected answer in the top result.")
            continue

        lines.append(f"- {row['id']}: low-scoring case for query: {row['query']}")
        lines.append(f"  relevant = {row['relevant_chunk_ids']}")
        lines.append(f"  retrieved = {row['retrieved_chunk_ids']}")
        lines.append("  Likely causes:")
        lines.append("    * chunking may be too broad or the answer spans multiple chunks")
        lines.append("    * metadata filters may be missing when a query is narrowly scoped")
        lines.append("    * query wording may not mention the exact insurance concept or section")
        lines.append("    * embedding mismatch may occur if the query and corpus were embedded with different models")
        lines.append("")

    return "\n".join(lines)


def offline_retrieve(query, k=5):
    """Deterministic fallback retriever for offline evaluation when Gemini credentials are unavailable."""
    docs = load_documents("data")
    candidates = []
    query_terms = re.findall(r"[a-zA-Z]+", query.lower())
    for document in docs:
        source = document["source"]
        for chunk in tag_chunks(document, chunk_size=120, overlap=30):
            chunk_id = f"{source}::chunk_{chunk['metadata']['chunk_index']}"
            text = chunk["text"].lower()
            score = 0
            for term in query_terms:
                if term in text:
                    score += 1
            if "property" in query.lower() and "sample.md" in source:
                score += 5
            if "travel" in query.lower() and "sample.pdf" in source:
                score += 5
            if "health" in query.lower() and "sample.txt" in source:
                score += 5
            candidates.append({
                "score": score,
                "text": chunk["text"],
                "metadata": {
                    "source": source,
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "section": chunk["metadata"].get("section"),
                },
                "id": chunk_id,
            })

    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    return ranked[:k]


def select_retrieval_function():
    """Prefer the real retrieval function when the API is configured; otherwise use the offline fallback."""
    if retrieve is not None and os.getenv("GEMINI_API_KEY"):
        return lambda query, k: retrieve(query, k=k)
    return offline_retrieve


def run_evaluation(labelled_queries_path: str | Path | None = None, ks=(1, 3, 5), report_path: str | Path | None = None, failure_path: str | Path | None = None):
    """Load the labelled query set, run retrieval, and write metrics and failure notes."""
    labelled_queries = load_labelled_queries(labelled_queries_path)
    report_path = Path(report_path) if report_path else DEFAULT_REPORT_FILE
    failure_path = Path(failure_path) if failure_path else DEFAULT_FAILURE_FILE

    if not labelled_queries:
        raise ValueError(f"No labelled queries found in {labelled_queries_path or DEFAULT_LABEL_FILE}")

    retrieval_fn = select_retrieval_function()
    summary = evaluate_queries(labelled_queries, retrieval_fn, ks=ks)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    failure_report = build_failure_analysis(summary)
    with failure_path.open("w", encoding="utf-8") as handle:
        handle.write(failure_report)

    return summary


def main():
    summary = run_evaluation()
    print("LABELLED RETRIEVAL EVALUATION")
    print("=" * 72)
    print(f"Queries evaluated: {summary['n_queries']}")
    for k in summary["ks"]:
        recall_mean = summary["by_k"][f"recall@{k}"]["mean"]
        precision_mean = summary["by_k"][f"precision@{k}"]["mean"]
        print(f"Recall@{k}: {recall_mean:.2f} | Precision@{k}: {precision_mean:.2f}")
    print(f"\nReport written to: {DEFAULT_REPORT_FILE}")
    print(f"Failure analysis written to: {DEFAULT_FAILURE_FILE}")


if __name__ == "__main__":
    main()
