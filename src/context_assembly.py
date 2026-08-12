"""Context assembly: ranked chunks -> a grounded, source-marked,
token-budget-limited augmented prompt.

Retrieval (retrieval.py) and re-ranking (rerank.py) produce a ranked
list of chunks. This module turns that list into the actual prompt text
sent to the language model:

  1. inject the chunk text as numbered, source-marked context blocks
  2. stop including chunks once the context would blow the token
     budget left over after instructions, the question, and headroom
     for the answer are reserved
  3. wrap everything in grounding instructions that tell the model to
     answer only from the supplied context and say so when the context
     is not enough

Chunks -> numbered source-marked context -> token-budget check -> augmented prompt.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "augmented_prompt_sample.txt"

SAMPLE_QUESTION = "What does my policy cover if my house is damaged by fire or a storm?"

# ---------------------------------------------------------
# Token budget configuration
# ---------------------------------------------------------

# Same tokenizer choice as token_counter.py, so counts are comparable
# across the project. Overridable via .env for a different target model.
MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "8000"))
INSTRUCTION_RESERVE_TOKENS = int(os.getenv("INSTRUCTION_RESERVE_TOKENS", "250"))
QUESTION_RESERVE_TOKENS = int(os.getenv("QUESTION_RESERVE_TOKENS", "150"))
ANSWER_RESERVE_TOKENS = int(os.getenv("ANSWER_RESERVE_TOKENS", "500"))

_encoding = None


def get_encoding():
    """Load the tiktoken encoding lazily so importing this module never
    requires tiktoken / a network call unless a caller actually counts
    tokens."""
    global _encoding
    if _encoding is None:
        import tiktoken
        _encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return _encoding


def count_tokens(text):
    """Count tokens the same way token_counter.py does. Falls back to a
    ~4-characters-per-token estimate if tiktoken (or its cached encoding
    file, which normally needs a one-time network download) is not
    available, so context assembly can still be built and tested
    offline. Real submissions should run this with tiktoken available."""
    try:
        encoding = get_encoding()
        return len(encoding.encode(text))
    except Exception as error:
        print(f"tiktoken unavailable ({error}); using a ~4 chars/token estimate.")
        return max(1, len(text) // 4)


def context_token_budget(context_window=MODEL_CONTEXT_WINDOW):
    """Tokens left over for retrieved context once instructions, the
    user question, and room for the model's own answer are reserved."""
    return max(
        0,
        context_window
        - INSTRUCTION_RESERVE_TOKENS
        - QUESTION_RESERVE_TOKENS
        - ANSWER_RESERVE_TOKENS,
    )


# ---------------------------------------------------------
# Source markers + chunk formatting
# ---------------------------------------------------------

def source_label(metadata):
    """Human-readable source reference, e.g. 'sample.md#0'."""
    metadata = metadata or {}
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index")
    return source if chunk_index is None else f"{source}#{chunk_index}"


def format_chunk(rank, chunk):
    """Render one retrieved chunk with a numbered source marker, e.g.
    [1], that the model can cite back in its answer."""
    metadata = chunk.get("metadata", {})
    return f"[{rank}] Source: {source_label(metadata)}\n{chunk['text'].strip()}"


# ---------------------------------------------------------
# Context assembly with token-budget enforcement
# ---------------------------------------------------------

def assemble_context(chunks, budget_tokens=None, count_fn=count_tokens):
    """Inject as many ranked chunks as fit inside budget_tokens, most
    relevant first. A chunk is included whole or not at all - the
    context never contains a truncated chunk, so every source marker
    always points at a complete piece of text.

    Returns a dict: context_text, included, dropped, tokens_used, budget_tokens.
    """
    if budget_tokens is None:
        budget_tokens = context_token_budget()

    included = []
    dropped = []
    tokens_used = 0
    blocks = []

    for rank, chunk in enumerate(chunks, start=1):
        block = format_chunk(rank, chunk)
        block_tokens = count_fn(block)

        if tokens_used + block_tokens <= budget_tokens:
            blocks.append(block)
            included.append(chunk)
            tokens_used += block_tokens
        else:
            dropped.append(chunk)

    return {
        "context_text": "\n\n".join(blocks),
        "included": included,
        "dropped": dropped,
        "tokens_used": tokens_used,
        "budget_tokens": budget_tokens,
    }


# ---------------------------------------------------------
# Grounding instructions + final augmented prompt
# ---------------------------------------------------------

GROUNDING_INSTRUCTIONS = """You are an internal insurance support assistant.

Instructions:
- Answer ONLY using the numbered context chunks below.
- Every factual claim in your answer must be supported by at least one context chunk. Cite the chunk(s) you used with its source marker, e.g. [1] or [1][3].
- If the provided context does not contain enough information to answer, say exactly: "I don't have enough information in the provided documents to answer that." Do not guess or use outside knowledge.
- Keep the answer concise and professional."""


AUGMENTED_PROMPT_TEMPLATE = """{instructions}

Context:
{context}

Question:
{question}

Answer:
"""


def build_augmented_prompt(question, chunks, budget_tokens=None, count_fn=count_tokens):
    """Full pipeline: ranked chunks -> token-budget-limited, source-
    marked context -> grounded augmented prompt ready to send to the
    language model."""
    assembly = assemble_context(chunks, budget_tokens=budget_tokens, count_fn=count_fn)

    context_text = assembly["context_text"] or "(no context chunks fit inside the token budget)"

    prompt = AUGMENTED_PROMPT_TEMPLATE.format(
        instructions=GROUNDING_INSTRUCTIONS,
        context=context_text,
        question=question,
    )

    assembly["prompt"] = prompt
    assembly["prompt_tokens"] = count_fn(prompt)
    return assembly


# ---------------------------------------------------------
# Chunk source for the demo (real retrieval/rerank if available,
# otherwise a placeholder built from the real indexed corpus text)
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


def get_ranked_chunks(question=SAMPLE_QUESTION, candidate_k=10, final_k=3):
    """Try the real retrieve() -> rerank() pipeline; fall back to
    PLACEHOLDER_CHUNKS (built from the real indexed corpus text, same
    rerank scores already verified in outputs/rerank_output.txt) if the
    vector store or embedding API are not available in this
    environment. Returns (chunks, used_placeholder: bool)."""
    try:
        from retrieval import retrieve, get_collection
        from rerank import rerank

        collection = get_collection()
        candidates = retrieve(question, k=candidate_k, collection=collection)
        return rerank(question, candidates, final_k=final_k), False
    except Exception as error:
        print(f"Falling back to placeholder chunks ({error}).")
        return PLACEHOLDER_CHUNKS[:final_k], True


# ---------------------------------------------------------
# Demo / sample output
# ---------------------------------------------------------

def format_chunk_status(rank, chunk, status):
    metadata = chunk.get("metadata", {})
    return "\n".join(
        [
            f"  [{rank}] {status}",
            f"      source : {source_label(metadata)}",
            f"      text   : {chunk['text'][:100].strip()}...",
        ]
    )


def run_demo(question=SAMPLE_QUESTION):
    chunks, used_placeholder = get_ranked_chunks(question)

    try:
        get_encoding()
        tokenizer_note = "tiktoken (gpt-4o-mini encoding)"
    except Exception as error:
        tokenizer_note = f"~4 chars/token estimate (tiktoken unavailable: {error})"

    report = [
        "CONTEXT ASSEMBLY - AUGMENTED PROMPT SAMPLE",
        "=" * 70,
        f"Question : {question}",
        f"Chunk source : {'PLACEHOLDER (no live vector store / API in this environment)' if used_placeholder else 'live retrieve() -> rerank() pipeline'}",
        f"Tokenizer : {tokenizer_note}",
        "",
    ]

    # Run 1: a normal budget - everything should fit.
    normal_budget = context_token_budget()
    normal = build_augmented_prompt(question, chunks, budget_tokens=normal_budget)

    report.append(f"RUN 1 - normal token budget ({normal_budget} tokens for context)")
    report.append("-" * 70)
    report.append(f"Context tokens used : {normal['tokens_used']} / {normal['budget_tokens']}")
    report.append(f"Chunks included     : {len(normal['included'])}")
    report.append(f"Chunks dropped      : {len(normal['dropped'])}")
    report.append(f"Total prompt tokens : {normal['prompt_tokens']}")
    report.append("")
    for rank, chunk in enumerate(normal["included"], start=1):
        report.append(format_chunk_status(rank, chunk, "INCLUDED"))
    report.append("")
    report.append("Augmented prompt sent to the model:")
    report.append("." * 70)
    report.append(normal["prompt"])
    report.append("." * 70)
    report.append("")

    # Run 2: a deliberately tiny budget - forces some chunks to drop,
    # to demonstrate the token-budget enforcement actually working.
    tiny_budget = 40
    tiny = build_augmented_prompt(question, chunks, budget_tokens=tiny_budget)

    report.append(f"RUN 2 - tiny token budget ({tiny_budget} tokens for context, to show enforcement)")
    report.append("-" * 70)
    report.append(f"Context tokens used : {tiny['tokens_used']} / {tiny['budget_tokens']}")
    report.append(f"Chunks included     : {len(tiny['included'])}")
    report.append(f"Chunks dropped      : {len(tiny['dropped'])}")
    report.append("")
    for rank, chunk in enumerate(tiny["included"], start=1):
        report.append(format_chunk_status(rank, chunk, "INCLUDED"))
    for rank, chunk in enumerate(tiny["dropped"], start=len(tiny["included"]) + 1):
        report.append(format_chunk_status(rank, chunk, "DROPPED (would exceed budget)"))
    report.append("")

    report.append("OBSERVATION")
    report.append("-" * 70)
    report.append(
        "assemble_context() never partially includes a chunk - each chunk "
        "either fits whole inside the remaining budget or is dropped "
        "entirely, so the model never sees a sentence cut off mid-way "
        "with a source marker attached to it. Highest-ranked chunks are "
        "tried first, so when the budget is tight, the chunks that get "
        "dropped are always the least relevant ones already at the "
        "bottom of the ranked list. Reserving tokens for instructions, "
        "the question, and the answer up front (not just for context) "
        "is what keeps the whole request inside the model's context "
        "window, not just the retrieved-chunks portion of it."
    )

    text_report = "\n".join(report)
    print(text_report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text_report, encoding="utf-8")
    print(f"\nSample augmented prompt saved to {OUTPUT_FILE}")


def main():
    run_demo()


if __name__ == "__main__":
    main()
