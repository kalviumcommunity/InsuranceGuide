"""Conversational RAG: multi-turn history + follow-up query rewriting.

A single-turn RAG pipeline (retrieval.py -> rerank.py -> context_assembly.py
-> citations.py) answers one question at a time using only that question's
own wording. Real users ask follow-ups that only make sense in light of
earlier turns - "does it exclude anything?" means nothing to a retriever
unless it also knows the previous turn was about fire/storm damage to a
house. This module adds:

  1. a small conversation history tracker                    (Task 1)
  2. a query rewriter that turns a follow-up into a standalone
     retrieval query using that history - an LLM rewriter by
     default, with a transparent, dependency-free fallback so the
     whole pipeline still works with no API key                (Task 2)
  3. retrieval using the REWRITTEN query, not the raw follow-up (Task 3)
  4. a full multi-turn demo showing the difference rewriting makes
     to what gets retrieved                                     (Task 4)

History + follow-up -> rewritten standalone query -> retrieve -> cited answer -> updated history.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "conversational_rag_sample_dialogue.txt"

_client = None


def get_client():
    """Create the Gemini client lazily. Only needed by llm_rewrite_query
    and call_llm, so history tracking and the custom rewrite fallback
    never require an API key or google-genai to be installed."""
    global _client
    if _client is None:
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")
        from google import genai
        _client = genai.Client(api_key=API_KEY)
    return _client


def call_llm(prompt):
    client = get_client()
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text.strip()


# ---------------------------------------------------------
# Task 1: conversation history
# ---------------------------------------------------------

class ConversationHistory:
    """Tracks user questions and assistant answers across turns, in the
    order they happened - the minimum context needed to understand a
    follow-up question."""

    def __init__(self):
        self.turns = []  # list of {"question": str, "answer": str}

    def add_turn(self, question, answer):
        self.turns.append({"question": question, "answer": answer})

    def last_question(self):
        return self.turns[-1]["question"] if self.turns else None

    def is_empty(self):
        return not self.turns

    def as_text(self, max_turns=3):
        """Render the most recent turns as plain dialogue text, for
        feeding into a rewrite or answer prompt."""
        recent = self.turns[-max_turns:]
        lines = []
        for turn in recent:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)


# ---------------------------------------------------------
# Task 2: rewrite a follow-up into a standalone query
# ---------------------------------------------------------

PRONOUN_RE = re.compile(r"\b(it|this|that|these|those|they|there)\b", re.IGNORECASE)

# A generic English stopword list - not tailored to this corpus - used
# to pull out the content words that actually carry meaning.
STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "we", "us", "our", "ours",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "to", "of", "for", "on", "in", "at", "by", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "up", "down", "out",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "also",
    "very", "and", "or", "but", "if", "what", "which", "who", "whom", "anything", "else",
    "specifically", "regarding",
}


def _content_keywords(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def needs_rewrite(question, history):
    """A follow-up needs rewriting if there IS prior history and the
    question either uses an anaphoric reference (it/that/this/...) or
    is short enough that it is unlikely to stand on its own without
    that history."""
    if history.is_empty():
        return False
    return bool(PRONOUN_RE.search(question)) or len(question.split()) <= 6


def custom_rewrite_query(question, history, lookback=2):
    """A transparent, dependency-free rewriter: fold recent turns' key
    content words into the follow-up so it can be embedded and
    retrieved on its own, without needing an LLM call.

    Looking back over more than just the immediately previous question
    matters: if turn 2 was itself vague ("does it exclude anything?"),
    only turn 1 actually carries the real topic words. This is still a
    coarser rewrite than an LLM would produce, and it has a known
    failure mode - if a follow-up actually changes topic entirely, this
    heuristic will still (wrongly) drag in the old topic's keywords,
    since it has no way to tell "continuing the same topic" apart from
    "starting a new one" without genuine language understanding. That
    trade-off is exactly why an LLM rewriter is the preferred
    production path (see llm_rewrite_query) and this is only the
    no-API-key fallback.
    """
    previous_questions = [turn["question"] for turn in history.turns[-lookback:]]

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

    return f"{cleaned_question}, specifically regarding {context_phrase}?"


def llm_rewrite_query(question, history):
    """Ask the chat model to rewrite the follow-up into a standalone
    query using the conversation so far. This is the production path:
    it produces a fluent, genuinely context-aware rewrite rather than a
    keyword-stitched one, and it can correctly recognize a real topic
    change instead of always carrying the old topic forward."""
    prompt = f"""Given the conversation history below, rewrite the follow-up question into a single, fully standalone question that could be understood and searched for without seeing the history. Preserve the user's intent exactly - do not answer it, just rewrite it. Return ONLY the rewritten question, nothing else.

Conversation history:
{history.as_text()}

Follow-up question: {question}

Standalone question:"""
    return call_llm(prompt).strip()


def rewrite_query(question, history):
    """Use the LLM rewriter when a Gemini API key is configured,
    otherwise fall back to the transparent custom rewriter. Returns
    (rewritten_query, used_llm, was_rewritten)."""
    if not needs_rewrite(question, history):
        return question, False, False

    if API_KEY:
        try:
            return llm_rewrite_query(question, history), True, True
        except Exception as error:
            print(f"LLM query rewrite failed ({error}); falling back to custom_rewrite_query.")

    return custom_rewrite_query(question, history), False, True


# ---------------------------------------------------------
# Task 3: retrieve using the rewritten query
# ---------------------------------------------------------

def get_ranked_chunks(query, k=4):
    """Real retrieve() -> rerank() pipeline for the (rewritten) query."""
    from retrieval import retrieve, get_collection
    from rerank import rerank

    collection = get_collection()
    candidates = retrieve(query, k=max(k, 5), collection=collection)
    return rerank(query, candidates, final_k=k)


def ask(question, history, k=4, llm_fn=None, chunks=None):
    """Answer one turn of a conversation: rewrite the question into a
    standalone query using prior history, retrieve with the REWRITTEN
    query (never the raw follow-up), generate a cited answer, and
    append the turn to history.

    chunks can be pre-supplied for testing/demo purposes; otherwise
    this calls get_ranked_chunks(rewritten_query) itself.
    """
    from citations import answer_with_citations

    rewritten_query, used_llm_rewrite, was_rewritten = rewrite_query(question, history)

    if chunks is None:
        chunks = get_ranked_chunks(rewritten_query, k=k)

    result = answer_with_citations(rewritten_query, chunks=chunks, llm_fn=llm_fn)
    result["original_question"] = question
    result["rewritten_query"] = rewritten_query
    result["was_rewritten"] = was_rewritten
    result["used_llm_rewrite"] = used_llm_rewrite

    history.add_turn(question, result["answer"])
    return result


# ---------------------------------------------------------
# Task 4/5: multi-turn demo + sample dialogue
# ---------------------------------------------------------

PLACEHOLDER_CHUNKS = [
    {
        "score": 0.4198,
        "rerank_score": 1.4286,
        "text": "# Property Insurance\n\nProperty insurance protects homes and buildings.\n\n"
        "It also covers losses caused by fire, storms, and theft.",
        "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
    },
    {
        "score": 0.4235,
        "rerank_score": 0.0,
        "text": "Insurance policies protect individuals against financial losses.\n\n"
        "Health insurance covers medical expenses.\n\n"
        "Motor insurance protects vehicles against accidents.",
        "metadata": {"source": "sample.txt", "chunk_index": 0, "section": ""},
    },
    {
        "score": 0.3054,
        "rerank_score": 0.0,
        "text": "Travel Insurance\n\nTravel insurance protects travelers from medical "
        "emergencies, trip cancellations, and lost baggage.",
        "metadata": {"source": "sample.pdf", "chunk_index": 0, "section": ""},
    },
]


def demo_llm_fn(question_for_answer):
    """A deterministic stand-in for call_llm's answer-generation use,
    keyed loosely to which question is being answered. NOT a real model
    call - used only when no GEMINI_API_KEY is configured, so the
    sample dialogue still shows complete, plausible cited answers."""

    def _fn(prompt):
        if "storm" in question_for_answer.lower() or "fire" in question_for_answer.lower():
            return (
                "Your policy covers damage to your dwelling from fire and storms [1]."
            )
        return (
            "Based on the property insurance chunk, exclusions beyond fire, storm, "
            "and theft are not detailed in the provided context [1]."
        )

    return _fn


def content_overlap_score(query, chunk_text):
    """A deliberately simple, dependency-free stand-in for "how well
    would this query retrieve this chunk" when there is no live
    embedding/Chroma search available: the count of stopword-filtered
    content words shared between the query and the chunk text.

    This is a raw count, not a ratio, so it does not get diluted when
    the rewritten query is longer than the raw follow-up (a ratio like
    rerank.py's custom_rerank_score would unfairly punish a longer, more
    descriptive rewrite). It also intentionally does not do any
    stemming or synonym matching (e.g. "storm" vs "storms", "house" vs
    "homes" count as different words) - a real embedding model captures
    that kind of semantic similarity, so this proxy score should be
    read as a conservative, likely-understated stand-in for what live
    retrieval would actually show.
    """
    query_words = set(_content_keywords(query))
    chunk_words = set(_content_keywords(chunk_text))
    if not query_words or not chunk_words:
        return 0
    return len(query_words & chunk_words)


def rank_placeholder_chunks(query, chunks=PLACEHOLDER_CHUNKS):
    """Score PLACEHOLDER_CHUNKS against a raw query string with
    content_overlap_score, purely to illustrate - deterministically and
    without any live vector store - the retrieval-relevance difference
    a rewritten query makes."""
    scored = [
        {
            "source": chunk["metadata"]["source"],
            "content_overlap_score": content_overlap_score(query, chunk["text"]),
        }
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item["content_overlap_score"], reverse=True)
    return scored


def get_demo_chunks(query):
    """Try the real retrieve()/rerank() pipeline; fall back to
    PLACEHOLDER_CHUNKS (built from the real indexed corpus text) if no
    vector store / API is available."""
    try:
        return get_ranked_chunks(query, k=4), False
    except Exception as error:
        print(f"Falling back to placeholder chunks ({error}).")
        return PLACEHOLDER_CHUNKS, True


def run_demo():
    history = ConversationHistory()
    used_placeholder = False

    report = ["CONVERSATIONAL RAG - SAMPLE MULTI-TURN DIALOGUE", "=" * 70, ""]

    # ---------------- Turn 1: standalone question ----------------
    question_1 = "What does my policy cover if my house is damaged by fire or a storm?"

    chunks_1, used_placeholder = get_demo_chunks(question_1)
    llm_fn_1 = call_llm if API_KEY else demo_llm_fn(question_1)

    result_1 = ask(question_1, history, chunks=chunks_1, llm_fn=llm_fn_1)

    report.append(f"Chunk source : {'PLACEHOLDER (no live vector store / API in this environment)' if used_placeholder else 'live retrieve() -> rerank() pipeline'}")
    report.append(f"Answer source: {'live Gemini call' if API_KEY else 'demo_llm_fn (hand-written stand-in, NOT a live model call)'}")
    report.append("")
    report.append("TURN 1")
    report.append("-" * 70)
    report.append(f"User question       : {result_1['original_question']}")
    report.append(f"Was rewritten        : {result_1['was_rewritten']} (no prior history yet, so used as-is)")
    report.append(f"Query used for retrieval : {result_1['rewritten_query']}")
    report.append(f"Answer               : {result_1['answer']}")
    report.append(f"Citations            : {list(result_1['citations'].keys())}")
    report.append("")

    # ---------------- Turn 2: vague follow-up ----------------
    question_2 = "Does it exclude anything?"

    # Show retrieval quality BEFORE rewriting: score the raw follow-up
    # against the same three chunks. With no shared vocabulary at all,
    # the raw follow-up cannot distinguish between them.
    before_scores = rank_placeholder_chunks(question_2)

    rewritten_2, used_llm_rewrite_2, was_rewritten_2 = rewrite_query(question_2, history)

    # Show retrieval quality AFTER rewriting: the rewritten query has
    # folded in turn 1's topic words, so it should now clearly favor
    # the property-insurance chunk.
    after_scores = rank_placeholder_chunks(rewritten_2)

    chunks_2, _ = get_demo_chunks(rewritten_2)
    llm_fn_2 = call_llm if API_KEY else demo_llm_fn(rewritten_2)
    result_2 = ask(question_2, history, chunks=chunks_2, llm_fn=llm_fn_2)

    report.append("TURN 2 (follow-up)")
    report.append("-" * 70)
    report.append(f"User question (raw)      : {result_2['original_question']}")
    report.append(f"Needed rewriting         : {was_rewritten_2}")
    report.append(f"Rewriter used            : {'LLM' if used_llm_rewrite_2 else 'custom_rewrite_query (no API key configured)'}")
    report.append(f"Rewritten standalone query: {rewritten_2}")
    report.append("")
    report.append("Content-word overlap BEFORE rewriting (raw follow-up vs each chunk):")
    for item in before_scores:
        report.append(f"  {item['source']:<12} shared_content_words={item['content_overlap_score']}")
    report.append("Content-word overlap AFTER rewriting (rewritten query vs each chunk):")
    for item in after_scores:
        report.append(f"  {item['source']:<12} shared_content_words={item['content_overlap_score']}")
    report.append("")
    report.append(f"Query actually used for retrieval : {result_2['rewritten_query']}")
    report.append(f"Answer                            : {result_2['answer']}")
    report.append(f"Citations                         : {list(result_2['citations'].keys())}")
    report.append(f"Fabricated markers                : {result_2['fabricated_markers']}")
    report.append("")

    report.append("OBSERVATION")
    report.append("-" * 70)
    top_before = before_scores[0]
    top_after = after_scores[0]
    report.append(
        f"Before rewriting, the raw follow-up '{question_2}' shares "
        f"{top_before['content_overlap_score']} content word(s) with even its best-matching "
        f"chunk ({top_before['source']}) - every chunk ties at that score, so this query "
        f"cannot distinguish between them at all. After folding in turn 1's key terms, the "
        f"rewritten query shares {top_after['content_overlap_score']} content word(s) with "
        f"{top_after['source']} specifically, while the other two chunks still share none - "
        f"exactly the chunk turn 1's answer came from, proving the follow-up was resolved "
        f"using conversation history rather than guessed from its own words."
    )
    report.append(
        "This overlap score is a deliberately simple, exact-word proxy (no stemming or "
        "synonym matching), so it likely UNDERSTATES the improvement: it cannot see that "
        "the rewrite's 'storm'/'house' and the chunk's 'storms'/'homes' mean the same "
        "thing, which a real embedding model would capture. The real retrieve()/rerank() "
        "pipeline (used automatically once a vector store and API key are available) "
        "would be expected to show an even clearer improvement than shown here."
    )
    report.append(
        "Known limitation of the custom (no-API-key) rewriter: it always folds in "
        "recent topic words, so if a follow-up genuinely changed subject entirely "
        "instead of continuing the same one, this heuristic could wrongly drag the "
        "old topic's vocabulary into the new query. An LLM rewriter (llm_rewrite_query) "
        "does not have this problem because it actually reasons about whether the "
        "follow-up continues or changes the topic."
    )

    text_report = "\n".join(report)
    print(text_report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text_report, encoding="utf-8")
    print(f"\nSample dialogue saved to {OUTPUT_FILE}")


def main():
    run_demo()


if __name__ == "__main__":
    main()
