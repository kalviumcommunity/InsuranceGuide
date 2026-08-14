import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversational_rag import (
    ConversationHistory,
    needs_rewrite,
    custom_rewrite_query,
    rewrite_query,
    ask,
)


SAMPLE_CHUNKS = [
    {
        "score": 0.9,
        "text": "# Property Insurance\n\nProperty insurance protects homes and buildings. "
        "It also covers losses caused by fire, storms, and theft.",
        "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
    },
]


def test_history_tracks_turns_in_order():
    history = ConversationHistory()
    assert history.is_empty()

    history.add_turn("q1", "a1")
    history.add_turn("q2", "a2")

    assert not history.is_empty()
    assert history.last_question() == "q2"
    assert "User: q1" in history.as_text()
    assert "Assistant: a2" in history.as_text()


def test_needs_rewrite_is_false_with_no_history():
    history = ConversationHistory()
    assert needs_rewrite("Does it cover theft?", history) is False


def test_needs_rewrite_is_true_for_pronoun_followup_with_history():
    history = ConversationHistory()
    history.add_turn("What does property insurance cover?", "It covers fire and storm damage.")
    assert needs_rewrite("Does it exclude anything?", history) is True


def test_needs_rewrite_is_false_for_a_long_standalone_followup():
    history = ConversationHistory()
    history.add_turn("What does property insurance cover?", "It covers fire and storm damage.")
    assert (
        needs_rewrite(
            "What is the maximum payout for a totally separate travel insurance claim?",
            history,
        )
        is False
    )


def test_custom_rewrite_query_folds_in_previous_question_keywords():
    history = ConversationHistory()
    history.add_turn(
        "What does my policy cover if my house is damaged by fire or a storm?",
        "Fire and storm damage to your house is covered.",
    )

    rewritten = custom_rewrite_query("Does it exclude anything?", history)

    assert rewritten != "Does it exclude anything?"
    assert "fire" in rewritten.lower()
    assert "storm" in rewritten.lower()


def test_custom_rewrite_query_looks_back_more_than_one_turn():
    history = ConversationHistory()
    history.add_turn(
        "What does my policy cover if my house is damaged by fire or a storm?",
        "Fire and storm damage is covered.",
    )
    history.add_turn("Does it exclude anything?", "Not detailed in the context.")

    rewritten = custom_rewrite_query("What about that?", history)

    # keywords from turn 1 (two turns back) should still show up
    assert any(word in rewritten.lower() for word in ("fire", "storm", "house"))


def test_rewrite_query_uses_custom_fallback_without_an_api_key():
    history = ConversationHistory()
    history.add_turn("What does property insurance cover?", "Fire and storm damage.")

    rewritten, used_llm, was_rewritten = rewrite_query("Does it exclude anything?", history)

    assert was_rewritten is True
    assert used_llm is False  # no GEMINI_API_KEY in the test environment
    assert rewritten != "Does it exclude anything?"


def test_rewrite_query_leaves_a_standalone_first_question_untouched():
    history = ConversationHistory()
    rewritten, used_llm, was_rewritten = rewrite_query(
        "What does property insurance cover?", history
    )

    assert was_rewritten is False
    assert rewritten == "What does property insurance cover?"


def test_ask_rewrites_retrieves_and_updates_history():
    history = ConversationHistory()

    result_1 = ask(
        "What does my policy cover if my house is damaged by fire or a storm?",
        history,
        chunks=SAMPLE_CHUNKS,
        llm_fn=lambda prompt: "Fire and storm damage to your dwelling is covered [1].",
    )

    assert result_1["was_rewritten"] is False
    assert len(history.turns) == 1

    result_2 = ask(
        "Does it exclude anything?",
        history,
        chunks=SAMPLE_CHUNKS,
        llm_fn=lambda prompt: "The context does not detail exclusions [1].",
    )

    assert result_2["was_rewritten"] is True
    assert any(word in result_2["rewritten_query"].lower() for word in ("fire", "storm"))
    assert len(history.turns) == 2
    assert history.turns[-1]["question"] == "Does it exclude anything?"
    assert "[1]" in result_2["citations"]
