import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grounded_generation import compare_with_and_without_retrieval, generate_grounded_answer


SAMPLE_CONTEXT = [
    {
        "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
        "text": "Property insurance protects homes and buildings. It also covers losses caused by fire, storms, and theft.",
    }
]


def test_generate_grounded_answer_uses_context_only():
    answer = generate_grounded_answer(
        "What does property insurance cover?",
        SAMPLE_CONTEXT,
    )

    assert "property insurance" in answer.lower()
    assert "source" in answer.lower() or "sample.md" in answer.lower()
    assert "not enough information" not in answer.lower()


def test_generate_grounded_answer_falls_back_when_context_is_missing():
    answer = generate_grounded_answer("What is the average premium for a luxury yacht?", [])

    assert "not enough information" in answer.lower()


def test_compare_with_and_without_retrieval_shows_grounding_change():
    comparison = compare_with_and_without_retrieval(
        "What does property insurance cover?",
        SAMPLE_CONTEXT,
    )

    assert "without_retrieval" in comparison
    assert "with_retrieval" in comparison
    assert comparison["with_retrieval"]["grounded"] is True
