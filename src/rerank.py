"""Chunk re-ranking for precision.

Initial vector retrieval (see retrieval.py) is fast and good at finding
likely candidates, but ordering by a single embedding-distance score is
not always the best final ordering for the language model. Re-ranking
adds a second, more careful scoring pass over a larger candidate set so
the chunks that are actually most relevant to the query move to the top
before the final top-k is sent to the model as grounding context.

Retrieval -> larger candidate set -> re-rank -> final top-k.

Scoring options (task allows any of these):
  - a re-ranker / cross-encoder model
  - an LLM scoring step               -> llm_rerank_score()
  - a clear custom scoring method     -> custom_rerank_score() (default)

rerank_score() uses the transparent custom scorer by default so the
whole pipeline works without any API key or network access, and
automatically upgrades to the LLM scorer when GEMINI_API_KEY is
configured.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "rerank_output.txt"

SAMPLE_QUERY = "What does my policy cover if my house is damaged by fire or a storm?"
CANDIDATE_K = 10
FINAL_K = 3

WORD_RE = re.compile(r"[a-z]+")

_client = None


# ---------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------

def get_client():
    """Create the Gemini client lazily. Only needed by llm_rerank_score,
    so importing/using the custom scorer never requires an API key or
    the google-genai package to be installed."""
    global _client
    if _client is None:
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")
        from google import genai
        _client = genai.Client(api_key=API_KEY)
    return _client


def llm_rerank_score(query, chunk):
    """Ask the chat model to score chunk relevance to the query from 0
    to 10. This is the production path re-ranking normally uses: one
    extra model call per candidate, which is exactly the added latency
    and cost that makes re-ranking a trade-off rather than a free win.
    """
    client = get_client()

    prompt = (
        "Score how relevant this chunk is to the query, from 0 to 10.\n"
        "Return only the number, nothing else.\n\n"
        f"Query: {query}\n"
        f"Chunk: {chunk['text']}\n"
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return float(response.text.strip())


def custom_rerank_score(query, chunk):
    """A transparent, dependency-free scoring function that looks at the
    query and the full chunk text together more closely than the vector
    search's single distance value did.

    It combines:
      - keyword overlap ratio (how much of the query's vocabulary shows
        up in the chunk)
      - a small repeat bonus (a term that appears more than once in the
        chunk is a stronger relevance signal than a single passing
        mention)

    Returns a score from 0 to 10, matching the scale an LLM scorer would
    use, so the two scorers are interchangeable.
    """
    query_words = set(WORD_RE.findall(query.lower()))
    chunk_words = set(WORD_RE.findall(chunk["text"].lower()))

    if not query_words or not chunk_words:
        return 0.0

    overlap = query_words & chunk_words
    overlap_ratio = len(overlap) / len(query_words)

    chunk_text_lower = chunk["text"].lower()
    repeat_bonus = sum(chunk_text_lower.count(word) - 1 for word in overlap)
    repeat_bonus = min(repeat_bonus, 3) * 0.1

    score = (overlap_ratio * 10) + repeat_bonus
    return round(min(score, 10.0), 4)


def rerank_score(query, chunk):
    """Use the LLM scorer when a Gemini API key is configured, otherwise
    fall back to the transparent custom scorer."""
    if API_KEY:
        try:
            return llm_rerank_score(query, chunk)
        except Exception as error:
            print(f"LLM re-rank scoring failed ({error}); falling back to custom_rerank_score.")
    return custom_rerank_score(query, chunk)


# ---------------------------------------------------------
# Re-ranking pipeline
# ---------------------------------------------------------

def rerank(query, candidates, final_k=FINAL_K, scorer=rerank_score):
    """Re-score every candidate against the query and return the
    final_k most relevant, most-relevant-first.

    candidates: list of {"score", "text", "metadata"} from retrieve()
    scorer: a (query, chunk) -> float function; defaults to
            rerank_score (custom scorer, or LLM scorer if a key is set)
    """
    reranked = []
    for chunk in candidates:
        reranked.append({**chunk, "rerank_score": scorer(query, chunk)})

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:final_k]


# ---------------------------------------------------------
# Demo: before-and-after comparison
# ---------------------------------------------------------

def format_block(rank, item, show_rerank):
    metadata = item.get("metadata", {})
    lines = [
        f"  rank         : {rank}",
        f"  vector_score : {item.get('score')}",
    ]
    if show_rerank:
        lines.append(f"  rerank_score : {item.get('rerank_score')}")
    lines += [
        f"  source       : {metadata.get('source')}",
        f"  chunk_index  : {metadata.get('chunk_index')}",
        f"  section      : {metadata.get('section') or None}",
        f"  text         : {item.get('text')}",
        "",
    ]
    return "\n".join(lines)


def run_demo(query=SAMPLE_QUERY, candidate_k=CANDIDATE_K, final_k=FINAL_K):
    """Retrieve a candidate set larger than final_k, re-rank it, and
    write a before-and-after report to outputs/rerank_output.txt."""

    # Imported here (not at module top) so this module and its custom
    # scorer stay importable/testable even without chromadb installed -
    # only running the live demo needs the vector store.
    from retrieval import retrieve, get_collection

    collection = get_collection()

    candidates = retrieve(query, k=candidate_k, collection=collection)
    before = candidates[:final_k]
    after = rerank(query, candidates, final_k=final_k)

    report = [
        "CHUNK RE-RANKING FOR PRECISION - SAMPLE RUN",
        "=" * 70,
        f"Query                    : {query}",
        f"Candidate set size (k)   : {candidate_k}",
        f"Candidates retrieved     : {len(candidates)}",
        f"Final top-k after rerank : {final_k}",
        f"Scorer used              : {'LLM (chat model)' if API_KEY else 'custom_rerank_score (no API key configured)'}",
        "",
        "CANDIDATE SET (initial vector-similarity order)",
        "-" * 70,
    ]
    for rank, item in enumerate(candidates, start=1):
        report.append(format_block(rank, item, show_rerank=False))

    report.append(f"BEFORE RE-RANKING (top {final_k} by vector score)")
    report.append("-" * 70)
    for rank, item in enumerate(before, start=1):
        report.append(format_block(rank, item, show_rerank=False))

    report.append(f"AFTER RE-RANKING (top {final_k} by rerank score)")
    report.append("-" * 70)
    for rank, item in enumerate(after, start=1):
        report.append(format_block(rank, item, show_rerank=True))

    order_before = [item["metadata"].get("source") for item in before]
    order_after = [item["metadata"].get("source") for item in after]

    report.append("COMPARISON")
    report.append("-" * 70)
    report.append(f"Order before re-ranking : {order_before}")
    report.append(f"Order after re-ranking  : {order_after}")
    report.append(f"Order changed           : {order_before != order_after}")
    report.append("")
    report.append(
        "Re-ranking scores the query against the full chunk text a second "
        "time, instead of relying only on the single embedding distance "
        "from the first pass. When the two signals disagree, the "
        "re-ranked order is expected to line up better with what a human "
        "would judge as directly relevant to the query. The cost is one "
        "extra scoring call per candidate - cheap for a local custom "
        "scorer, but real latency and API cost for an LLM or cross-"
        "encoder scorer - so re-rank a modest candidate set (10-20), not "
        "the whole corpus, and reserve it for cases where precision in "
        "the final context matters more than raw speed."
    )

    text_report = "\n".join(report)
    print(text_report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text_report, encoding="utf-8")
    print(f"\nSample re-ranking output saved to {OUTPUT_FILE}")


def main():
    run_demo()


if __name__ == "__main__":
    main()
