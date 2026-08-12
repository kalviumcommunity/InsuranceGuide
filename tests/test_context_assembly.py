import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_assembly import (
    assemble_context,
    build_augmented_prompt,
    format_chunk,
    source_label,
)


def word_count_tokens(text):
    """A tiny, deterministic stand-in for count_tokens so budget tests
    don't depend on tiktoken / network access being available."""
    return len(text.split())


SAMPLE_CHUNKS = [
    {
        "score": 0.9,
        "text": "Property insurance covers fire and storm damage to a dwelling.",
        "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
    },
    {
        "score": 0.5,
        "text": "Travel insurance covers trip cancellations and lost baggage during travel.",
        "metadata": {"source": "sample.pdf", "chunk_index": 0, "section": ""},
    },
    {
        "score": 0.4,
        "text": "Motor insurance protects vehicles against accidents and theft on the road.",
        "metadata": {"source": "sample.txt", "chunk_index": 0, "section": ""},
    },
]


def test_source_label_includes_chunk_index():
    assert source_label({"source": "sample.md", "chunk_index": 0}) == "sample.md#0"
    assert source_label({"source": "sample.md"}) == "sample.md"


def test_format_chunk_includes_numbered_marker_and_source():
    block = format_chunk(1, SAMPLE_CHUNKS[0])
    assert block.startswith("[1] Source: sample.md#0")
    assert "fire and storm damage" in block


def test_assemble_context_includes_everything_within_a_generous_budget():
    result = assemble_context(SAMPLE_CHUNKS, budget_tokens=1000, count_fn=word_count_tokens)

    assert len(result["included"]) == 3
    assert len(result["dropped"]) == 0
    assert "[1]" in result["context_text"]
    assert "[3]" in result["context_text"]


def test_assemble_context_drops_lowest_ranked_chunks_under_a_tight_budget():
    # Each formatted block is well over 13 word-tokens, so a budget of
    # 15 fits exactly the highest-ranked chunk and drops the rest,
    # never truncating a chunk mid-way.
    result = assemble_context(SAMPLE_CHUNKS, budget_tokens=15, count_fn=word_count_tokens)

    assert len(result["included"]) == 1
    assert result["included"][0]["metadata"]["source"] == "sample.md"
    assert len(result["dropped"]) == 2
    assert result["tokens_used"] <= 15


def test_assemble_context_never_includes_a_partial_chunk():
    result = assemble_context(SAMPLE_CHUNKS, budget_tokens=15, count_fn=word_count_tokens)

    for chunk in result["included"]:
        block = format_chunk(1, chunk)
        # the full chunk text must appear verbatim, never cut short
        assert chunk["text"] in block


def test_build_augmented_prompt_has_grounding_instructions_and_question():
    result = build_augmented_prompt(
        "What does my policy cover if my house is damaged by fire?",
        SAMPLE_CHUNKS,
        budget_tokens=1000,
        count_fn=word_count_tokens,
    )

    prompt = result["prompt"]
    assert "Answer ONLY using the numbered context chunks" in prompt
    assert "I don't have enough information" in prompt
    assert "What does my policy cover" in prompt
    assert "[1] Source: sample.md#0" in prompt


def test_build_augmented_prompt_reports_dropped_chunks_when_budget_is_tight():
    result = build_augmented_prompt(
        "query",
        SAMPLE_CHUNKS,
        budget_tokens=15,
        count_fn=word_count_tokens,
    )

    assert len(result["dropped"]) > 0
    assert len(result["included"]) < len(SAMPLE_CHUNKS)
