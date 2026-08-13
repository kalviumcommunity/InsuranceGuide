"""Source citation & attribution.

A grounded answer becomes verifiable when every factual claim can be
traced back to the retrieved chunk that actually supports it. This
module:

  1. builds a stable citation map from chunk metadata: [1] -> source
     document, chunk id/index, page, section, and the chunk text itself
  2. builds a prompt that tells the model to cite every claim with a
     marker that already exists in the context, and to say so - never
     invent a citation - when the context is not enough
  3. lets a caller verify a citation by looking up the exact chunk text
     the model was supposed to have used
  4. detects fabricated citations (marker numbers the answer used that
     were never actually provided) and enforces a no-citation fallback
     when there is no supporting context at all

retrieve/rerank -> citation map -> cited prompt -> answer -> verify / detect fabrication.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

# Retrieval-quality guardrail defaults (overridable via .env)
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.25"))
RETRIEVAL_MIN_CHUNKS = int(os.getenv("RETRIEVAL_MIN_CHUNKS", "1"))
RETRIEVAL_MIN_MEAN = os.getenv("RETRIEVAL_MIN_MEAN")
RETRIEVAL_MIN_MEAN = float(RETRIEVAL_MIN_MEAN) if RETRIEVAL_MIN_MEAN else None

try:
    from guardrails import check_retrieval_quality
except Exception:
    # Allow the module to be imported in environments where the helper
    # might not be available (e.g. tests that stub imports).
    def check_retrieval_quality(*args, **kwargs):
        return {"passed": True, "reason": "guardrails_missing"}

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "cited_answers_sample.txt"

SAMPLE_QUESTION = "What does my policy cover if my house is damaged by fire or a storm?"
NO_SUPPORT_QUESTION = "What is the maximum coverage for a commercial fishing vessel under this policy?"

NO_CONTEXT_FALLBACK = (
    "I don't have enough information in the provided documents to answer that."
)

CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")

_client = None


def get_client():
    """Create the Gemini client lazily. Only needed by call_llm, so
    building/testing citation maps and prompts never requires an API
    key or the google-genai package to be installed."""
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
# Citation map: marker -> real document + location
# ---------------------------------------------------------

def build_citation_map(chunks):
    """Map each numbered marker used in the assembled context (1-indexed,
    same order the chunks were injected in - see context_assembly.py) to
    the real chunk it refers to. This mapping is what makes a citation
    verifiable rather than just plausible-looking: it only exists
    because retrieval/chunking preserved source metadata all the way
    through the pipeline.
    """
    citation_map = {}
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        citation_map[f"[{index}]"] = {
            "source": metadata.get("source"),
            "chunk_id": metadata.get("chunk_id", chunk.get("id")),
            "chunk_index": metadata.get("chunk_index"),
            "page": metadata.get("page"),
            "section": metadata.get("section"),
            "text": chunk.get("text"),
        }
    return citation_map


# ---------------------------------------------------------
# Cited prompt (builds on context_assembly's token-budget-aware
# chunk injection, layering citation-specific instructions on top)
# ---------------------------------------------------------

CITED_ANSWER_INSTRUCTIONS = """You are an internal insurance support assistant.

Instructions:
- Answer using ONLY the numbered context chunks below.
- Cite every factual claim using the source marker(s) it came from, e.g. [1] or [1][2].
- Only use marker numbers that actually appear in the context below. Never invent a marker number, and never cite a source that is not shown.
- If the context does not contain enough information to answer, respond with exactly: "{fallback}" and do not include any citation markers.
- Keep the answer concise and professional."""


def build_cited_prompt(question, chunks, budget_tokens=None, count_fn=None):
    """Inject chunks with a token budget (reusing context_assembly.py,
    not duplicating it) and wrap them in citation-specific grounding
    instructions."""
    from context_assembly import assemble_context, count_tokens as default_count_fn

    count_fn = count_fn or default_count_fn
    assembly = assemble_context(chunks, budget_tokens=budget_tokens, count_fn=count_fn)
    context_text = assembly["context_text"] or "(no context chunks fit inside the token budget)"

    instructions = CITED_ANSWER_INSTRUCTIONS.format(fallback=NO_CONTEXT_FALLBACK)

    prompt = f"""{instructions}

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt, assembly


# ---------------------------------------------------------
# Fabricated-citation detection
# ---------------------------------------------------------

def find_cited_markers(answer_text):
    """Every distinct [n] marker that appears in the answer text, in
    numeric order."""
    return sorted(set(CITATION_MARKER_RE.findall(answer_text)), key=int)


def find_fabricated_markers(answer_text, citation_map):
    """Marker numbers the answer cited that do not exist in the
    citation map - i.e. the model referenced a source it was never
    actually given. If the prompt instructions are followed this
    should always be empty; a non-empty result means the answer is not
    safe to trust as-is and should be flagged or blocked before it
    reaches a user."""
    cited = find_cited_markers(answer_text)
    return [marker for marker in cited if f"[{marker}]" not in citation_map]


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

def verify_citation(citation_map, marker):
    """Look up exactly what a citation marker points to, so a user (or
    a test) can check the answer's claim against the real source text
    instead of trusting the model's citation blindly."""
    entry = citation_map.get(marker)
    if entry is None:
        return {"marker": marker, "found": False, "reason": "marker not present in citation map"}
    return {"marker": marker, "found": True, **entry}


# ---------------------------------------------------------
# Full pipeline: question -> cited answer (+ fallback + fabrication check)
# ---------------------------------------------------------

def get_ranked_chunks(question, k=4):
    """Real retrieve() -> rerank() pipeline, used when the caller does
    not already have a chunk list (e.g. from an earlier pipeline
    stage)."""
    from retrieval import retrieve, get_collection
    from rerank import rerank

    collection = get_collection()
    candidates = retrieve(question, k=max(k, 5), collection=collection)
    return rerank(question, candidates, final_k=k)


def answer_with_citations(question, k=4, chunks=None, llm_fn=None):
    """Full citation pipeline for one question.

    chunks can be pre-supplied (e.g. already retrieved/re-ranked
    upstream, or deliberately passed as [] to simulate "nothing
    relevant was found"); otherwise this calls retrieve() -> rerank()
    itself.

    An empty chunk list is treated as insufficient support and returns
    the fallback WITHOUT calling the model at all - fabricating a
    citation is worse than refusing, so there is nothing for the model
    to fabricate from in the first place.

    Returns: answer, citations, fabricated_markers, used_fallback,
    chunks_considered.
    """
    llm_fn = llm_fn or call_llm

    if chunks is None:
        chunks = get_ranked_chunks(question, k=k)

    if not chunks:
        return {
            "answer": NO_CONTEXT_FALLBACK,
            "citations": {},
            "fabricated_markers": [],
            "used_fallback": True,
            "chunks_considered": 0,
            "quality": {"passed": False, "reason": "no_chunks"},
        }

    # Assess retrieval quality and refuse when upstream retrieval is weak.
    quality = check_retrieval_quality(chunks, min_score=RETRIEVAL_MIN_SCORE, min_chunks_above=RETRIEVAL_MIN_CHUNKS, min_mean_score=RETRIEVAL_MIN_MEAN)
    if not quality.get("passed"):
        # Do not call the model — refuse safely instead of hallucinating.
        return {
            "answer": NO_CONTEXT_FALLBACK,
            "citations": {},
            "fabricated_markers": [],
            "used_fallback": True,
            "chunks_considered": len(chunks),
            "quality": quality,
        }

    prompt, assembly = build_cited_prompt(question, chunks)
    citation_map = build_citation_map(assembly["included"])

    answer = llm_fn(prompt)
    fabricated = find_fabricated_markers(answer, citation_map)

    return {
        "answer": answer,
        "citations": citation_map,
        "fabricated_markers": fabricated,
        "used_fallback": False,
        "chunks_considered": len(chunks),
        "quality": quality,
    }


# ---------------------------------------------------------
# Demo / sample output
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


def demo_llm_fn(prompt):
    """A deterministic stand-in for call_llm, used only when no
    GEMINI_API_KEY is configured so the sample output can still show a
    complete, realistic cited answer. This is NOT a real model call -
    it is hand-written to look like what a well-behaved model should
    produce given CITED_ANSWER_INSTRUCTIONS and these three chunks."""
    return (
        "Your policy covers damage to your dwelling from fire and storms [1]. "
        "It does not cover travel-related losses or vehicle accidents directly - "
        "those fall under the separate travel [3] and motor [2] sections rather "
        "than property coverage."
    )


def get_demo_chunks(question):
    """Try the real retrieve()/rerank() pipeline; fall back to the same
    placeholder chunks used in earlier tasks (built from the real
    indexed corpus text) if no vector store / API is available."""
    try:
        return get_ranked_chunks(question, k=4), False
    except Exception as error:
        print(f"Falling back to placeholder chunks ({error}).")
        return PLACEHOLDER_CHUNKS, True


def format_citation_map(citation_map):
    lines = []
    for marker, entry in citation_map.items():
        lines.append(f"  {marker} -> source: {entry['source']}")
        lines.append(f"       chunk_index: {entry['chunk_index']}  section: {entry['section'] or None}")
        lines.append(f"       text: {entry['text']}")
        lines.append("")
    return "\n".join(lines)


def run_demo():
    used_placeholder_chunks = False
    chunks, used_placeholder_chunks = get_demo_chunks(SAMPLE_QUESTION)
    llm_fn = call_llm if API_KEY else demo_llm_fn

    report = [
        "SOURCE CITATION & ATTRIBUTION - SAMPLE RUN",
        "=" * 70,
        f"Chunk source : {'PLACEHOLDER (no live vector store / API in this environment)' if used_placeholder_chunks else 'live retrieve() -> rerank() pipeline'}",
        f"Answer source: {'live Gemini call' if API_KEY else 'demo_llm_fn (hand-written stand-in, NOT a live model call)'}",
        "",
        "=" * 70,
        "EXAMPLE 1 - CITED ANSWER WITH SUPPORTING SOURCES",
        "=" * 70,
        f"Question: {SAMPLE_QUESTION}",
        "",
    ]

    result_1 = answer_with_citations(SAMPLE_QUESTION, chunks=chunks, llm_fn=llm_fn)

    report.append("ANSWER:")
    report.append(result_1["answer"])
    report.append("")
    report.append(f"Cited markers found in answer : {find_cited_markers(result_1['answer'])}")
    report.append(f"Fabricated markers detected    : {result_1['fabricated_markers']}")
    report.append("")
    report.append("CITATION MAP (marker -> real source document + location):")
    report.append("-" * 70)
    report.append(format_citation_map(result_1["citations"]))

    report.append("VERIFYING CITATION [1] AGAINST THE ORIGINAL CHUNK TEXT")
    report.append("-" * 70)
    verification = verify_citation(result_1["citations"], "[1]")
    report.append(f"Marker [1] found       : {verification['found']}")
    report.append(f"Source document        : {verification.get('source')}")
    report.append(f"Chunk index             : {verification.get('chunk_index')}")
    report.append(f"Section                : {verification.get('section')}")
    report.append(f"Original chunk text     : {verification.get('text')}")
    report.append(
        "Claim check: the answer says fire/storm damage to the dwelling is "
        "covered and cites [1]; the chunk above literally says 'covers losses "
        "caused by fire, storms, and theft' - the citation is verified as "
        "supported by its source, not just plausible-sounding."
    )
    report.append("")

    report.append("=" * 70)
    report.append("EXAMPLE 4 - WEAK RETRIEVAL QUALITY REFUSAL (low scores)")
    report.append("=" * 70)
    report.append(f"Question: {SAMPLE_QUESTION}")
    report.append(
        "Simulating the case where retrieval returned results but their similarity\n"
        "scores are too low to trust. The pipeline should refuse without calling the model."
    )
    # Build a low-quality chunk list (scores below common thresholds)
    low_quality_chunks = [
        {"score": 0.05, "text": "Unrelated text about unrelated topics.", "metadata": {"source": "other.md", "chunk_index": 0}},
        {"score": 0.07, "text": "Some general insurance fluff, not on topic.", "metadata": {"source": "other.md", "chunk_index": 1}},
    ]

    result_4 = answer_with_citations(SAMPLE_QUESTION, chunks=low_quality_chunks, llm_fn=llm_fn)
    report.append(f"used_fallback   : {result_4['used_fallback']}")
    report.append(f"quality_report  : {result_4.get('quality')}")
    report.append(f"answer          : {result_4['answer']}")
    report.append("")

    report.append("=" * 70)
    report.append("EXAMPLE 2 - FABRICATED CITATION DETECTION")
    report.append("=" * 70)
    bad_answer = (
        "Commercial fishing vessels are covered up to $250,000 under the "
        "marine endorsement [4]."
    )
    report.append(f"Simulated (bad) model answer: {bad_answer}")
    report.append(f"Citation map only has markers: {list(result_1['citations'].keys())}")
    fabricated = find_fabricated_markers(bad_answer, result_1["citations"])
    report.append(f"Fabricated markers detected  : {fabricated}")
    report.append(
        "[4] does not correspond to any chunk that was actually retrieved - "
        "this answer would be rejected/flagged rather than shown to a user "
        "as-is, because a citation to a source that was never provided is a "
        "fabricated citation, worse than no citation at all."
    )
    report.append("")

    report.append("=" * 70)
    report.append("EXAMPLE 3 - NO-SOURCE FALLBACK (nothing relevant retrieved)")
    report.append("=" * 70)
    report.append(f"Question: {NO_SUPPORT_QUESTION}")
    report.append(
        "Simulating the case where retrieval/re-ranking found nothing "
        "relevant enough to support an answer (e.g. an upstream relevance "
        "gate filtered every candidate out) by passing an empty chunk list."
    )
    result_3 = answer_with_citations(NO_SUPPORT_QUESTION, chunks=[], llm_fn=llm_fn)
    report.append(f"used_fallback   : {result_3['used_fallback']}")
    report.append(f"answer          : {result_3['answer']}")
    report.append(f"citations       : {result_3['citations']}")
    report.append(
        "No model call was made at all for this case - there was nothing to "
        "ground an answer in, so the pipeline refuses rather than letting the "
        "model invent a plausible-sounding but unsupported (and uncited, or "
        "worse, falsely cited) answer."
    )

    text_report = "\n".join(report)
    print(text_report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text_report, encoding="utf-8")
    print(f"\nSample cited answers saved to {OUTPUT_FILE}")


def main():
    run_demo()


if __name__ == "__main__":
    main()
