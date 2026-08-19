"""
Simple in-memory cache for repeated RAG queries.
"""

import hashlib
import json
from typing import Any


_QUERY_CACHE: dict[str, dict[str, Any]] = {}


def make_cache_key(
    question: str,
    k: int,
    model: str,
) -> str:
    """Create a stable cache key from relevant query settings."""

    payload = {
        "question": question.strip(),
        "k": k,
        "model": model,
    }

    raw_key = json.dumps(
        payload,
        sort_keys=True,
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


def get_cached_result(cache_key: str) -> dict[str, Any] | None:
    """Return a cached result if available."""

    return _QUERY_CACHE.get(cache_key)


def set_cached_result(
    cache_key: str,
    result: dict[str, Any],
) -> None:
    """Store a RAG result in the cache."""

    _QUERY_CACHE[cache_key] = result


def clear_cache() -> None:
    """Clear all cached queries."""

    _QUERY_CACHE.clear()


def cache_size() -> int:
    """Return the number of cached queries."""

    return len(_QUERY_CACHE)