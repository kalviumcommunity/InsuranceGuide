"""Conversational RAG: multi-turn history + follow-up query rewriting.

A single-turn RAG pipeline answers one question at a time. Real users ask
follow-ups that only make sense in light of earlier turns.

This module adds:

1. Conversation history tracking.
2. Query rewriting for follow-up questions.
3. A dependency-free custom fallback rewriter.
4. Optional Gemini LLM rewriting.
5. Retrieval using the rewritten query.
6. A complete multi-turn demo.

LLM rewriting is opt-in through:

    ENABLE_LLM_REWRITE=true

This prevents automated tests from accidentally calling Gemini simply
because GEMINI_API_KEY exists in the local .env file.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv


# ------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

# LLM rewriting is deliberately opt-in.
#
# This is important for tests because a developer may have
# GEMINI_API_KEY in .env, but the test suite should still use the
# deterministic fallback unless LLM rewriting is explicitly enabled.
ENABLE_LLM_REWRITE = (
    os.getenv("ENABLE_LLM_REWRITE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "conversational_rag_sample_dialogue.txt"
)

_client = None


# ------------------------------------------------------------------
# Gemini client
# ------------------------------------------------------------------

def get_client():
    """Create the Gemini client lazily.

    The client is only required when LLM rewriting or LLM answer
    generation is explicitly enabled.
    """

    global _client

    if _client is None:
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")

        from google import genai

        _client = genai.Client(api_key=API_KEY)

    return _client


def call_llm(prompt):
    """Call Gemini and return the generated text."""

    client = get_client()

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )

    return response.text.strip()


# ------------------------------------------------------------------
# Task 1: Conversation history
# ------------------------------------------------------------------

class ConversationHistory:
    """Track user questions and assistant answers across turns."""

    def __init__(self):
        self.turns = []

    def add_turn(self, question, answer):
        """Add one user/assistant turn."""

        self.turns.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    def last_question(self):
        """Return the most recent user question."""

        if not self.turns:
            return None

        return self.turns[-1]["question"]

    def is_empty(self):
        """Return True when no conversation turns exist."""

        return not self.turns

    def as_text(self, max_turns=3):
        """Render recent conversation turns as plain dialogue."""

        recent = self.turns[-max_turns:]

        lines = []

        for turn in recent:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Task 2: Query rewriting
# ------------------------------------------------------------------

PRONOUN_RE = re.compile(
    r"\b(it|this|that|these|those|they|there)\b",
    re.IGNORECASE,
)


STOPWORDS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "we",
    "us",
    "our",
    "ours",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "to",
    "of",
    "for",
    "on",
    "in",
    "at",
    "by",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "from",
    "up",
    "down",
    "out",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "also",
    "very",
    "and",
    "or",
    "but",
    "if",
    "what",
    "which",
    "who",
    "whom",
    "anything",
    "else",
    "specifically",
    "regarding",
}


def _content_keywords(text):
    """Extract meaningful words while removing generic stopwords."""

    words = re.findall(r"[a-zA-Z']+", text.lower())

    return [
        word
        for word in words
        if word not in STOPWORDS
    ]


def needs_rewrite(question, history):
    """Determine whether a question needs conversation-aware rewriting.

    A rewrite is needed when:

    - There is previous conversation history, and
    - The question contains a pronoun/reference such as 'it', 'this',
      'that', etc., OR
    - The question is short enough that it probably depends on context.
    """

    if history.is_empty():
        return False

    return (
        bool(PRONOUN_RE.search(question))
        or len(question.split()) <= 6
    )


def custom_rewrite_query(question, history, lookback=2):
    """Rewrite a follow-up using recent conversation keywords.

    This is the deterministic fallback used when LLM rewriting is
    disabled or unavailable.
    """

    previous_questions = [
        turn["question"]
        for turn in history.turns[-lookback:]
    ]

    keywords = []
    seen = set()

    for previous_question in previous_questions:
        for word in _content_keywords(previous_question):

            if word not in seen:
                seen.add(word)
                keywords.append(word)

    context_phrase = " ".join(keywords[:10])

    cleaned_question = question.strip().rstrip("?")

    if not context_phrase:
        return question

    return (
        f"{cleaned_question}, specifically regarding "
        f"{context_phrase}?"
    )


def llm_rewrite_query(question, history):
    """Rewrite a follow-up using Gemini."""

    prompt = f"""Given the conversation history below, rewrite the
follow-up question into a single, fully standalone question that could
be understood and searched for without seeing the history.

Preserve the user's intent exactly.
Do not answer the question.
Return ONLY the rewritten question.

Conversation history:
{history.as_text()}

Follow-up question:
{question}

Standalone question:"""

    return call_llm(prompt).strip()


def rewrite_query(question, history):
    """Rewrite a follow-up question.

    Returns:

        (
            rewritten_query,
            used_llm,
            was_rewritten
        )

    LLM rewriting is only attempted when ENABLE_LLM_REWRITE=true and
    GEMINI_API_KEY is available.

    Otherwise the deterministic custom rewriter is used.
    """

    # Standalone questions do not need rewriting.
    if not needs_rewrite(question, history):
        return question, False, False

    # --------------------------------------------------------------
    # Optional LLM path
    # --------------------------------------------------------------

    if ENABLE_LLM_REWRITE and API_KEY:
        try:
            rewritten = llm_rewrite_query(
                question,
                history,
            )

            return rewritten, True, True

        except Exception as error:
            print(
                "LLM query rewrite failed "
                f"({error}); falling back to custom_rewrite_query."
            )

    # --------------------------------------------------------------
    # Deterministic fallback
    # --------------------------------------------------------------

    rewritten = custom_rewrite_query(
        question,
        history,
    )

    return rewritten, False, True


# ------------------------------------------------------------------
# Task 3: Retrieval using rewritten query
# ------------------------------------------------------------------

def get_ranked_chunks(query, k=4):
    """Run retrieve -> rerank using the rewritten query."""

    from retrieval import retrieve, get_collection
    from rerank import rerank

    collection = get_collection()

    candidates = retrieve(
        query,
        k=max(k, 5),
        collection=collection,
    )

    return rerank(
        query,
        candidates,
        final_k=k,
    )


def ask(
    question,
    history,
    k=4,
    llm_fn=None,
    chunks=None,
):
    """Process one conversational RAG turn.

    Steps:

    1. Rewrite the follow-up.
    2. Retrieve using the rewritten query.
    3. Generate an answer.
    4. Add the turn to conversation history.
    """

    from citations import answer_with_citations

    (
        rewritten_query,
        used_llm_rewrite,
        was_rewritten,
    ) = rewrite_query(
        question,
        history,
    )

    if chunks is None:
        chunks = get_ranked_chunks(
            rewritten_query,
            k=k,
        )

    result = answer_with_citations(
        rewritten_query,
        chunks=chunks,
        llm_fn=llm_fn,
    )

    result["original_question"] = question
    result["rewritten_query"] = rewritten_query
    result["was_rewritten"] = was_rewritten
    result["used_llm_rewrite"] = used_llm_rewrite

    history.add_turn(
        question,
        result["answer"],
    )

    return result


# ------------------------------------------------------------------
# Task 4/5: Multi-turn demo
# ------------------------------------------------------------------

PLACEHOLDER_CHUNKS = [
    {
        "score": 0.4198,
        "rerank_score": 1.4286,
        "text": (
            "# Property Insurance\n\n"
            "Property insurance protects homes and buildings.\n\n"
            "It also covers losses caused by fire, storms, and theft."
        ),
        "metadata": {
            "source": "sample.md",
            "chunk_index": 0,
            "section": "Property Insurance",
        },
    },
    {
        "score": 0.4235,
        "rerank_score": 0.0,
        "text": (
            "Insurance policies protect individuals against financial "
            "losses.\n\n"
            "Health insurance covers medical expenses.\n\n"
            "Motor insurance protects vehicles against accidents."
        ),
        "metadata": {
            "source": "sample.txt",
            "chunk_index": 0,
            "section": "",
        },
    },
    {
        "score": 0.3054,
        "rerank_score": 0.0,
        "text": (
            "Travel Insurance\n\n"
            "Travel insurance protects travelers from medical "
            "emergencies, trip cancellations, and lost baggage."
        ),
        "metadata": {
            "source": "sample.pdf",
            "chunk_index": 0,
            "section": "",
        },
    },
]


def demo_llm_fn(question_for_answer):
    """Deterministic answer-generation stand-in for the demo."""

    def _fn(prompt):

        question_lower = question_for_answer.lower()

        if (
            "storm" in question_lower
            or "fire" in question_lower
        ):
            return (
                "Your policy covers damage to your dwelling "
                "from fire and storms [1]."
            )

        return (
            "Based on the property insurance chunk, exclusions "
            "beyond fire, storm, and theft are not detailed "
            "in the provided context [1]."
        )

    return _fn


def content_overlap_score(query, chunk_text):
    """Calculate simple content-word overlap."""

    query_words = set(
        _content_keywords(query)
    )

    chunk_words = set(
        _content_keywords(chunk_text)
    )

    if not query_words or not chunk_words:
        return 0

    return len(
        query_words & chunk_words
    )


def rank_placeholder_chunks(
    query,
    chunks=PLACEHOLDER_CHUNKS,
):
    """Rank placeholder chunks using word overlap."""

    scored = [
        {
            "source": chunk["metadata"]["source"],
            "content_overlap_score": content_overlap_score(
                query,
                chunk["text"],
            ),
        }
        for chunk in chunks
    ]

    scored.sort(
        key=lambda item: item["content_overlap_score"],
        reverse=True,
    )

    return scored


def get_demo_chunks(query):
    """Use the real retrieval pipeline when available.

    Falls back to deterministic placeholder chunks otherwise.
    """

    try:
        return get_ranked_chunks(
            query,
            k=4,
        ), False

    except Exception as error:

        print(
            f"Falling back to placeholder chunks ({error})."
        )

        return PLACEHOLDER_CHUNKS, True


def run_demo():
    """Run the complete multi-turn demonstration."""

    history = ConversationHistory()

    used_placeholder = False

    report = [
        "CONVERSATIONAL RAG - SAMPLE MULTI-TURN DIALOGUE",
        "=" * 70,
        "",
        f"LLM rewrite enabled : {ENABLE_LLM_REWRITE}",
        f"Gemini API key      : {'configured' if API_KEY else 'not configured'}",
        "",
    ]

    # --------------------------------------------------------------
    # Turn 1
    # --------------------------------------------------------------

    question_1 = (
        "What does my policy cover if my house is damaged "
        "by fire or a storm?"
    )

    chunks_1, used_placeholder = get_demo_chunks(
        question_1
    )

    # For answer generation, only use Gemini when explicitly enabled.
    if ENABLE_LLM_REWRITE and API_KEY:
        llm_fn_1 = call_llm
    else:
        llm_fn_1 = demo_llm_fn(question_1)

    result_1 = ask(
        question_1,
        history,
        chunks=chunks_1,
        llm_fn=llm_fn_1,
    )

    report.append(
        "Chunk source : "
        + (
            "PLACEHOLDER "
            "(no live vector store / API in this environment)"
            if used_placeholder
            else "live retrieve() -> rerank() pipeline"
        )
    )

    report.append(
        "Answer source: "
        + (
            "live Gemini call"
            if ENABLE_LLM_REWRITE and API_KEY
            else "demo_llm_fn "
            "(hand-written stand-in, NOT a live model call)"
        )
    )

    report.append("")
    report.append("TURN 1")
    report.append("-" * 70)

    report.append(
        f"User question       : "
        f"{result_1['original_question']}"
    )

    report.append(
        "Was rewritten       : "
        f"{result_1['was_rewritten']} "
        "(no prior history yet, so used as-is)"
    )

    report.append(
        f"Query used for retrieval : "
        f"{result_1['rewritten_query']}"
    )

    report.append(
        f"Answer               : "
        f"{result_1['answer']}"
    )

    report.append(
        f"Citations            : "
        f"{list(result_1['citations'].keys())}"
    )

    report.append("")

    # --------------------------------------------------------------
    # Turn 2
    # --------------------------------------------------------------

    question_2 = "Does it exclude anything?"

    # BEFORE rewriting
    before_scores = rank_placeholder_chunks(
        question_2
    )

    (
        rewritten_2,
        used_llm_rewrite_2,
        was_rewritten_2,
    ) = rewrite_query(
        question_2,
        history,
    )

    # AFTER rewriting
    after_scores = rank_placeholder_chunks(
        rewritten_2
    )

    chunks_2, _ = get_demo_chunks(
        rewritten_2
    )

    if ENABLE_LLM_REWRITE and API_KEY:
        llm_fn_2 = call_llm
    else:
        llm_fn_2 = demo_llm_fn(rewritten_2)

    result_2 = ask(
        question_2,
        history,
        chunks=chunks_2,
        llm_fn=llm_fn_2,
    )

    report.append("TURN 2 (follow-up)")
    report.append("-" * 70)

    report.append(
        f"User question (raw)      : "
        f"{result_2['original_question']}"
    )

    report.append(
        f"Needed rewriting         : "
        f"{was_rewritten_2}"
    )

    report.append(
        "Rewriter used            : "
        + (
            "LLM"
            if used_llm_rewrite_2
            else "custom_rewrite_query"
        )
    )

    report.append(
        f"Rewritten standalone query: "
        f"{rewritten_2}"
    )

    report.append("")

    report.append(
        "Content-word overlap BEFORE rewriting "
        "(raw follow-up vs each chunk):"
    )

    for item in before_scores:
        report.append(
            f"  {item['source']:<12} "
            f"shared_content_words="
            f"{item['content_overlap_score']}"
        )

    report.append(
        "Content-word overlap AFTER rewriting "
        "(rewritten query vs each chunk):"
    )

    for item in after_scores:
        report.append(
            f"  {item['source']:<12} "
            f"shared_content_words="
            f"{item['content_overlap_score']}"
        )

    report.append("")

    report.append(
        f"Query actually used for retrieval : "
        f"{result_2['rewritten_query']}"
    )

    report.append(
        f"Answer                            : "
        f"{result_2['answer']}"
    )

    report.append(
        f"Citations                         : "
        f"{list(result_2['citations'].keys())}"
    )

    report.append(
        f"Fabricated markers                : "
        f"{result_2['fabricated_markers']}"
    )

    report.append("")

    # --------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------

    report.append("OBSERVATION")
    report.append("-" * 70)

    top_before = before_scores[0]
    top_after = after_scores[0]

    report.append(
        f"Before rewriting, the raw follow-up "
        f"'{question_2}' shares "
        f"{top_before['content_overlap_score']} "
        f"content word(s) with its best-matching chunk "
        f"({top_before['source']})."
    )

    report.append(
        f"After rewriting, the rewritten query shares "
        f"{top_after['content_overlap_score']} "
        f"content word(s) with "
        f"{top_after['source']}."
    )

    report.append(
        "This demonstrates how conversation history gives "
        "a vague follow-up enough context to become a "
        "standalone retrieval query."
    )

    report.append(
        "The overlap score is a deliberately simple exact-word "
        "proxy. It does not perform stemming, synonym matching, "
        "or semantic similarity."
    )

    report.append(
        "The real retrieve()/rerank() pipeline uses embeddings "
        "and therefore provides stronger semantic matching."
    )

    report.append(
        "The custom rewriter is deterministic and requires no "
        "API key. It may carry old topic keywords into a genuine "
        "topic change, which is why the Gemini rewriter can be "
        "enabled for production use."
    )

    text_report = "\n".join(report)

    print(text_report)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        text_report,
        encoding="utf-8",
    )

    print(
        f"\nSample dialogue saved to {OUTPUT_FILE}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    run_demo()


if __name__ == "__main__":
    main()