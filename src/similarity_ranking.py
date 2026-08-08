"""Similarity-based ranking for embedding retrieval demos.

This module provides a small, dependency-light ranking pipeline that can be
used with embeddings produced by the project or with synthetic vectors for
local testing and demonstrations.
"""

import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT_DIR / "outputs" / "similarity_ranking_output.json"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity between two vectors.

    Cosine similarity measures the angle between vectors, which makes it a
    strong choice for embeddings because it captures semantic direction rather
    than raw magnitude. Two texts with similar meaning tend to point in a
    similar direction in embedding space even if their lengths differ.
    """
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)

    if a_array.size == 0 or b_array.size == 0:
        return 0.0

    denom = np.linalg.norm(a_array) * np.linalg.norm(b_array)
    if denom == 0:
        return 0.0

    return float(np.dot(a_array, b_array) / denom)


def rank_chunks(query_embedding: Sequence[float], chunks: Iterable[dict]) -> List[dict]:
    """Rank chunks by cosine similarity against a query embedding."""
    ranked = []
    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk.get("embedding", []))
        ranked.append(
            {
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "embedding": chunk.get("embedding", []),
                "score": score,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def build_demo_report(query_text: str, query_embedding: Sequence[float], chunks: Iterable[dict]) -> str:
    """Create a text report with the top and bottom ranking chunks."""
    ranked = rank_chunks(query_embedding, chunks)
    top = ranked[:3]
    bottom = ranked[-3:]

    lines = []
    lines.append("SIMILARITY RANKING DEMO")
    lines.append("=" * 70)
    lines.append(f"Query: {query_text}")
    lines.append(f"Query embedding dimension: {len(query_embedding)}")
    lines.append(f"Ranked chunk count: {len(ranked)}")
    lines.append("")
    lines.append("Top matches")
    for item in top:
        metadata = item["metadata"]
        lines.append(
            f"- score={item['score']:.4f} | source={metadata.get('source')} | "
            f"chunk_index={metadata.get('chunk_index')} | section={metadata.get('section')} | "
            f"text={item['text'][:140]}"
        )

    lines.append("")
    lines.append("Bottom matches")
    for item in bottom:
        metadata = item["metadata"]
        lines.append(
            f"- score={item['score']:.4f} | source={metadata.get('source')} | "
            f"chunk_index={metadata.get('chunk_index')} | section={metadata.get('section')} | "
            f"text={item['text'][:140]}"
        )

    return "\n".join(lines)


def save_report(query_text: str, query_embedding: Sequence[float], chunks: Iterable[dict]) -> Path:
    """Write a JSON report of ranked chunks and a text summary."""
    ranked = rank_chunks(query_embedding, chunks)
    payload = {
        "query": query_text,
        "metric": "cosine_similarity",
        "justification": (
            "Cosine similarity is appropriate because embedding vectors are compared "
            "by direction in vector space, and this metric captures semantic overlap "
            "without being dominated by vector magnitude."
        ),
        "ranked_chunks": ranked,
    }

    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return OUTPUT_FILE


def main():
    sample_chunks = [
        {
            "text": "Property insurance protects homes and buildings against fire and storm damage.",
            "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
            "embedding": [0.95, 0.10, 0.20],
        },
        {
            "text": "Health insurance covers medical expenses for treatment and hospitalization.",
            "metadata": {"source": "sample.txt", "chunk_index": 1, "section": None},
            "embedding": [0.20, 0.90, 0.10],
        },
        {
            "text": "Motor insurance protects vehicles against accidents and theft.",
            "metadata": {"source": "sample.txt", "chunk_index": 2, "section": None},
            "embedding": [0.25, 0.15, 0.95],
        },
        {
            "text": "Claims are settled after the insurer approves evidence and completes assessment.",
            "metadata": {"source": "claims_guideline.txt", "chunk_index": 3, "section": "Settlement"},
            "embedding": [0.05, 0.10, 0.85],
        },
        {
            "text": "Travel insurance covers emergency trips and overseas medical support.",
            "metadata": {"source": "sample.txt", "chunk_index": 4, "section": None},
            "embedding": [-0.70, 0.05, 0.10],
        },
    ]

    query_text = "Which policy covers damage to my home caused by fire?"
    query_embedding = [0.90, 0.10, 0.15]

    output_path = save_report(query_text, query_embedding, sample_chunks)
    report = build_demo_report(query_text, query_embedding, sample_chunks)

    print(report)
    print(f"\nRanked JSON written to {output_path}")


if __name__ == "__main__":
    main()
