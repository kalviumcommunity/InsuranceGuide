import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rerank import rerank, custom_rerank_score


SAMPLE_QUERY = "What does my policy cover if my house is damaged by fire or a storm?"

# Same three real chunks as the retrieval task, with a deliberately
# "wrong" vector-score order: the motor-insurance chunk scores highest
# from the (mock) embedding step even though it has nothing to do with
# fire/storm damage to a house, while the property-insurance chunk that
# actually answers the query scores lower.
SAMPLE_CANDIDATES = [
    {
        "score": 0.42,
        "text": "Insurance policies protect individuals against financial losses. "
        "Health insurance covers medical expenses. Motor insurance protects "
        "vehicles against accidents.",
        "metadata": {"source": "sample.txt", "chunk_index": 0, "section": ""},
    },
    {
        "score": 0.40,
        "text": "Property insurance protects homes and buildings. It also covers "
        "losses caused by fire, storms, and theft.",
        "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
    },
    {
        "score": 0.38,
        "text": "Travel insurance protects travelers from medical emergencies, trip "
        "cancellations, and lost baggage.",
        "metadata": {"source": "sample.pdf", "chunk_index": 0, "section": ""},
    },
]


def test_custom_rerank_score_prefers_direct_keyword_overlap():
    property_chunk = SAMPLE_CANDIDATES[1]
    travel_chunk = SAMPLE_CANDIDATES[2]

    assert custom_rerank_score(SAMPLE_QUERY, property_chunk) > custom_rerank_score(
        SAMPLE_QUERY, travel_chunk
    )


def test_rerank_can_promote_a_lower_vector_score_candidate():
    # sample.txt has the highest vector score, but sample.md is the
    # chunk that actually answers a fire/storm question. A working
    # re-ranker should move sample.md to rank 1.
    result = rerank(SAMPLE_QUERY, SAMPLE_CANDIDATES, final_k=2, scorer=custom_rerank_score)

    assert len(result) == 2
    assert result[0]["metadata"]["source"] == "sample.md"

    scores = [item["rerank_score"] for item in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_respects_final_k_smaller_than_candidate_set():
    # An identity scorer (just reuse the vector score) isolates final_k
    # slicing behaviour from the scoring logic itself.
    result = rerank(
        SAMPLE_QUERY, SAMPLE_CANDIDATES, final_k=1, scorer=lambda q, c: c["score"]
    )

    assert len(result) == 1
    assert result[0]["metadata"]["source"] == "sample.txt"


def test_rerank_attaches_rerank_score_to_every_result():
    result = rerank(SAMPLE_QUERY, SAMPLE_CANDIDATES, final_k=3, scorer=custom_rerank_score)

    assert len(result) == 3
    for item in result:
        assert "rerank_score" in item
        assert "score" in item  # original vector score is preserved
