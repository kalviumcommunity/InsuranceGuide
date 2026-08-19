"""
Structured logging and usage tracking for the RAG application.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "rag_requests.jsonl"


def log_request(
    *,
    question: str,
    answer: str = "",
    sources: list[dict[str, Any]] | None = None,
    cache_hit: bool = False,
    latency_ms: float = 0.0,
    token_usage: dict[str, Any] | None = None,
    estimated_cost: float = 0.0,
    error: str | None = None,
) -> None:
    """
    Write one structured JSON log entry for a RAG request.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer_preview": answer[:200] if answer else "",
        "sources": sources or [],
        "cache_hit": cache_hit,
        "latency_ms": round(latency_ms, 2),
        "token_usage": token_usage or {},
        "estimated_cost": round(estimated_cost, 8),
        "error": error,
    }

    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                entry,
                ensure_ascii=False,
            )
            + "\n"
        )


def timer() -> float:
    """Return a high-resolution start time."""
    return time.perf_counter()


def elapsed_ms(start_time: float) -> float:
    """Return elapsed time in milliseconds."""
    return (time.perf_counter() - start_time) * 1000