import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from similarity_ranking import cosine_similarity, rank_chunks


def test_cosine_similarity_is_high_for_similar_vectors():
    a = [1.0, 0.0, 1.0]
    b = [1.0, 0.0, 0.5]
    assert cosine_similarity(a, b) > 0.8


def test_rank_chunks_returns_descending_scores_with_metadata():
    query_embedding = [1.0, 0.0, 0.0]
    chunks = [
        {
            "text": "property insurance covers fire damage",
            "metadata": {"source": "sample.md", "chunk_index": 0},
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "office lunch menu is available at noon",
            "metadata": {"source": "sample.txt", "chunk_index": 1},
            "embedding": [0.0, 1.0, 0.0],
        },
    ]

    ranked = rank_chunks(query_embedding, chunks)

    assert ranked[0]["text"] == "property insurance covers fire damage"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["metadata"]["source"] == "sample.md"
    assert ranked[1]["metadata"]["source"] == "sample.txt"
