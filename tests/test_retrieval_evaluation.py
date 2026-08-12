import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval_evaluation import compute_precision_at_k, compute_recall_at_k, evaluate_queries


def test_compute_recall_at_k_detects_known_relevant_chunk():
    relevant_ids = {"sample.md::chunk_0"}
    retrieved_ids = ["sample.txt::chunk_0", "sample.md::chunk_0"]

    assert compute_recall_at_k(relevant_ids, retrieved_ids, k=2) == 1.0


def test_compute_precision_at_k_counts_only_relevant_items():
    relevant_ids = {"sample.md::chunk_0"}
    retrieved_ids = ["sample.md::chunk_0", "sample.txt::chunk_0"]

    assert compute_precision_at_k(relevant_ids, retrieved_ids, k=2) == 0.5


def test_evaluate_queries_aggregates_metric_scores():
    queries = [
        {
            "id": "q1",
            "query": "What does property insurance cover?",
            "relevant_chunk_ids": ["sample.md::chunk_0"],
        },
        {
            "id": "q2",
            "query": "What does health insurance cover?",
            "relevant_chunk_ids": ["sample.txt::chunk_0"],
        },
    ]

    def fake_retrieve(query, k):
        if "property" in query.lower():
            return [
                {"metadata": {"source": "sample.md", "chunk_index": 0}},
                {"metadata": {"source": "sample.txt", "chunk_index": 0}},
            ][:k]
        return [
            {"metadata": {"source": "sample.txt", "chunk_index": 0}},
            {"metadata": {"source": "sample.md", "chunk_index": 0}},
        ][:k]

    summary = evaluate_queries(queries, fake_retrieve, ks=(1, 2))

    assert summary["n_queries"] == 2
    assert "recall@1" in summary["by_k"]
    assert "precision@1" in summary["by_k"]
    assert summary["by_k"]["recall@1"]["mean"] >= 0.0
