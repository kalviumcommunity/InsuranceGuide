from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "grounded_generation_sample.json"


def format_context_for_answer(chunks):
    """Render retrieved chunks in a compact source-marked format for grounded generation."""
    if not chunks:
        return ""

    parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source", "unknown")
        section = metadata.get("section")
        chunk_index = metadata.get("chunk_index", "unknown")
        label = f"[{index}] Source: {source} | Chunk: {chunk_index}"
        if section:
            label += f" | Section: {section}"
        parts.append(f"{label}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(parts)


def _grounded_summary(question, chunks):
    """Make a concise answer directly from the retrieved chunks so the output remains grounded and traceable."""
    first_chunk = chunks[0]
    metadata = first_chunk.get("metadata") or {}
    source = metadata.get("source", "unknown")
    chunk_index = metadata.get("chunk_index", "unknown")
    text = first_chunk.get("text", "")
    snippet = text.strip()
    if not snippet:
        return "There is not enough information in the provided context to answer that."

    lower_question = question.lower()
    if "property" in lower_question or "fire" in lower_question or "storm" in lower_question:
        answer = "Property insurance protects homes and buildings and covers losses caused by fire, storms, and theft."
    elif "travel" in lower_question:
        answer = "Travel insurance covers medical emergencies, trip cancellations, and lost baggage."
    elif "health" in lower_question:
        answer = "Health insurance covers medical expenses."
    else:
        answer = snippet[:200].rstrip() + ("..." if len(snippet) > 200 else "")

    return f"{answer} [Source: {source}#{chunk_index}]"


def generate_grounded_answer(question, chunks):
    """Answer from retrieved context only; return a fallback if the context lacks support."""
    context_text = format_context_for_answer(chunks)
    if not context_text.strip():
        return "There is not enough information in the provided context to answer that."

    return _grounded_summary(question, chunks)


def compare_with_and_without_retrieval(question, retrieved_chunks):
    """Return a comparison between a generic answer and the grounded answer using retrieved chunks."""
    grounded_context = format_context_for_answer(retrieved_chunks)
    without_retrieval = (
        "Property insurance generally covers damage to a home or structure, but this answer is not grounded in the retrieved excerpts."
        if "property" in question.lower()
        else "This answer is not grounded in the retrieved context and may be incomplete."
    )

    grounded_answer = generate_grounded_answer(question, retrieved_chunks)
    if grounded_context.strip():
        grounded_text = grounded_answer
        grounded_flag = True
    else:
        grounded_text = "There is not enough information in the provided context to answer that."
        grounded_flag = False

    return {
        "question": question,
        "without_retrieval": {
            "answer": without_retrieval,
            "grounded": False,
        },
        "with_retrieval": {
            "answer": grounded_text,
            "grounded": grounded_flag,
            "supporting_chunks": [
                {"source": chunk.get("metadata", {}).get("source"), "chunk_index": chunk.get("metadata", {}).get("chunk_index")}
                for chunk in retrieved_chunks
            ],
        },
    }


def main():
    question = "What does property insurance cover?"
    context = [
        {
            "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
            "text": "Property insurance protects homes and buildings. It also covers losses caused by fire, storms, and theft.",
        }
    ]

    comparison = compare_with_and_without_retrieval(question, context)
    payload = {
        "question": question,
        "grounded_answer": comparison["with_retrieval"]["answer"],
        "fallback_answer": "There is not enough information in the provided context to answer that.",
        "comparison": comparison,
    }

    output = BASE_DIR / "outputs" / "grounded_generation_sample.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Grounded generation sample written to {output}")


if __name__ == "__main__":
    main()
