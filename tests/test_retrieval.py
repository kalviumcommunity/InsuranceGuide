import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_store import create_collection, insert_test_record

import retrieval


def _sample_collection(tmp_path):
    """A tiny, real Chroma collection with three known chunks/vectors,
    mirroring the shape of the real insurance_chunks collection but
    with small 3-dim vectors so the test is fast and deterministic."""
    collection = create_collection(
        db_path=str(tmp_path / "chroma"),
        collection_name="test_retrieval_chunks",
        dimension=3,
    )

    insert_test_record(
        collection,
        record_id="sample.md::chunk_0",
        text="Property insurance protects homes from fire and storm damage.",
        metadata={"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
        vector=[1.0, 0.0, 0.0],
    )
    insert_test_record(
        collection,
        record_id="sample.pdf::chunk_0",
        text="Travel insurance covers trip cancellations and lost baggage.",
        metadata={"source": "sample.pdf", "chunk_index": 0, "section": ""},
        vector=[0.0, 1.0, 0.0],
    )
    insert_test_record(
        collection,
        record_id="sample.txt::chunk_0",
        text="Motor insurance protects vehicles against accidents.",
        metadata={"source": "sample.txt", "chunk_index": 0, "section": ""},
        vector=[0.0, 0.0, 1.0],
    )

    return collection


def test_retrieve_returns_top_match_with_score_and_metadata(tmp_path, monkeypatch):
    collection = _sample_collection(tmp_path)

    # Bypass the real embedding API call: pretend the query embeds close
    # to the property-insurance direction in vector space.
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [0.9, 0.05, 0.05])

    results = retrieval.retrieve(
        "What does property insurance cover?", k=1, collection=collection
    )

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "sample.md"
    assert 0.0 <= results[0]["score"] <= 1.0
    assert "Property insurance" in results[0]["text"]


def test_increasing_k_returns_more_chunks_without_reordering(tmp_path, monkeypatch):
    collection = _sample_collection(tmp_path)
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [0.9, 0.05, 0.05])

    top_1 = retrieval.retrieve("query", k=1, collection=collection)
    top_3 = retrieval.retrieve("query", k=3, collection=collection)

    assert len(top_1) == 1
    assert len(top_3) == 3

    # The single best match for k=1 must still rank first for k=3.
    assert top_3[0]["metadata"]["source"] == top_1[0]["metadata"]["source"]

    # Scores must be sorted, most similar first.
    scores = [item["score"] for item in top_3]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_corpus_is_capped_at_available_chunks(tmp_path, monkeypatch):
    collection = _sample_collection(tmp_path)
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [0.9, 0.05, 0.05])

    results = retrieval.retrieve("query", k=10, collection=collection)

    assert len(results) == 3


def test_retrieve_accepts_a_metadata_where_filter(tmp_path, monkeypatch):
    collection = _sample_collection(tmp_path)
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [0.9, 0.05, 0.05])

    results = retrieval.retrieve(
        "What does property insurance cover?",
        k=3,
        collection=collection,
        where={"source": "sample.md"},
    )

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "sample.md"
    assert "Property insurance" in results[0]["text"]


def test_retrieve_supports_keyword_and_hybrid_boosting(tmp_path, monkeypatch):
    collection = _sample_collection(tmp_path)
    monkeypatch.setattr(retrieval, "embed_query", lambda query: [0.9, 0.05, 0.05])

    results = retrieval.retrieve(
        "What does property insurance cover?",
        k=3,
        collection=collection,
        keyword_terms=["property", "insurance"],
        hybrid=True,
    )

    assert len(results) == 3
    assert results[0]["metadata"]["source"] == "sample.md"
    assert results[0]["keyword_hits"] >= 1



