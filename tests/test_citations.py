import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import citations
from citations import (
    build_citation_map,
    find_cited_markers,
    find_fabricated_markers,
    verify_citation,
    answer_with_citations,
    NO_CONTEXT_FALLBACK,
)


SAMPLE_CHUNKS = [
    {
        "score": 0.42,
        "text": "# Property Insurance\n\nProperty insurance protects homes and buildings. "
        "It also covers losses caused by fire, storms, and theft.",
        "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
    },
    {
        "score": 0.31,
        "text": "Travel Insurance\n\nTravel insurance protects travelers from medical "
        "emergencies, trip cancellations, and lost baggage.",
        "metadata": {"source": "sample.pdf", "chunk_index": 0, "section": ""},
    },
]


def test_build_citation_map_maps_markers_to_real_metadata():
    citation_map = build_citation_map(SAMPLE_CHUNKS)

    assert set(citation_map.keys()) == {"[1]", "[2]"}
    assert citation_map["[1]"]["source"] == "sample.md"
    assert citation_map["[1]"]["chunk_index"] == 0
    assert citation_map["[1]"]["section"] == "Property Insurance"
    assert "Property insurance protects homes" in citation_map["[1]"]["text"]


def test_find_cited_markers_extracts_all_distinct_marker_numbers_in_order():
    answer = "Fire damage is covered [1]. Travel claims are separate [2][1]."
    assert find_cited_markers(answer) == ["1", "2"]


def test_find_fabricated_markers_flags_markers_not_in_the_map():
    citation_map = build_citation_map(SAMPLE_CHUNKS)  # only [1] and [2] exist
    answer = "Your fishing vessel is covered up to $50,000 [3]."

    assert find_fabricated_markers(answer, citation_map) == ["3"]


def test_find_fabricated_markers_is_empty_when_all_markers_are_real():
    citation_map = build_citation_map(SAMPLE_CHUNKS)
    answer = "Property damage from fire and storms is covered [1]."

    assert find_fabricated_markers(answer, citation_map) == []


def test_verify_citation_returns_the_real_source_text():
    citation_map = build_citation_map(SAMPLE_CHUNKS)
    result = verify_citation(citation_map, "[1]")

    assert result["found"] is True
    assert result["source"] == "sample.md"
    assert "fire, storms, and theft" in result["text"]


def test_verify_citation_reports_a_missing_marker():
    citation_map = build_citation_map(SAMPLE_CHUNKS)
    result = verify_citation(citation_map, "[9]")

    assert result["found"] is False


def test_answer_with_citations_falls_back_when_there_are_no_chunks():
    result = answer_with_citations("What is the fishing vessel coverage limit?", chunks=[])

    assert result["used_fallback"] is True
    assert result["answer"] == NO_CONTEXT_FALLBACK
    assert result["citations"] == {}
    assert result["chunks_considered"] == 0


def test_answer_with_citations_never_calls_the_model_when_there_are_no_chunks(monkeypatch):
    calls = []
    monkeypatch.setattr(citations, "call_llm", lambda prompt: calls.append(prompt) or "should not happen")

    answer_with_citations("no context question", chunks=[])

    assert calls == []


def test_answer_with_citations_builds_a_real_citation_map_from_supplied_chunks(monkeypatch):
    monkeypatch.setattr(
        citations,
        "call_llm",
        lambda prompt: "Property insurance covers fire and storm damage [1].",
    )

    result = answer_with_citations("What does property insurance cover?", chunks=SAMPLE_CHUNKS)

    assert result["used_fallback"] is False
    assert "[1]" in result["citations"]
    assert result["citations"]["[1]"]["source"] == "sample.md"
    assert result["fabricated_markers"] == []


def test_answer_with_citations_flags_a_fabricated_marker(monkeypatch):
    monkeypatch.setattr(
        citations,
        "call_llm",
        lambda prompt: "Fishing vessels are covered up to $250,000 [4].",
    )

    result = answer_with_citations("What does property insurance cover?", chunks=SAMPLE_CHUNKS)

    assert result["fabricated_markers"] == ["4"]
