from __future__ import annotations

from typing import Iterable


def _extract_score(chunk: dict) -> float:
    """Extract the most relevant score field from a retrieved chunk."""
    if not chunk:
        return 0.0
    # Prefer rerank_score if present (higher-level signal), otherwise use score
    for key in ("rerank_score", "score", "hybrid_score"):
        if key in chunk and isinstance(chunk[key], (int, float)):
            return float(chunk[key])
    # Last-resort: check nested metadata idiosyncrasies
    try:
        return float(chunk.get("metadata", {}).get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def check_retrieval_quality(
    chunks: Iterable[dict],
    min_score: float = 0.25,
    min_chunks_above: int = 1,
    min_mean_score: float | None = None,
) -> dict:
    """Assess retrieval quality using simple, transparent signals.

    Args:
        chunks: iterable of retrieval result dicts containing numeric scores.
        min_score: per-chunk score threshold considered "relevant".
        min_chunks_above: minimum number of chunks with score >= min_score.
        min_mean_score: optional mean-score floor across all returned chunks.

    Returns a dict with keys: passed (bool), top_score, mean_score,
    n_chunks, n_above, and a short reason message.
    """
    scores = [ _extract_score(c) for c in (chunks or []) ]
    n_chunks = len(scores)
    if n_chunks == 0:
        return {"passed": False, "reason": "no_chunks", "n_chunks": 0, "n_above": 0, "top_score": 0.0, "mean_score": 0.0}

    n_above = sum(1 for s in scores if s >= min_score)
    top_score = max(scores) if scores else 0.0
    mean_score = sum(scores) / n_chunks if scores else 0.0

    if n_above < min_chunks_above:
        return {"passed": False, "reason": "too_few_high_score_chunks", "n_chunks": n_chunks, "n_above": n_above, "top_score": top_score, "mean_score": mean_score}

    if min_mean_score is not None and mean_score < min_mean_score:
        return {"passed": False, "reason": "mean_score_too_low", "n_chunks": n_chunks, "n_above": n_above, "top_score": top_score, "mean_score": mean_score}

    return {"passed": True, "reason": "ok", "n_chunks": n_chunks, "n_above": n_above, "top_score": top_score, "mean_score": mean_score}
