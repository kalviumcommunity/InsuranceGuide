"""Runs the metadata-tagging pipeline over the sample corpus, prints a
sample of tagged chunks, and demonstrates tracing a retrieved chunk back
to its exact source using only its metadata.
"""

from pathlib import Path

from chunk_metadata import tag_corpus
from document_loader import load_documents

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"
REPORT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "chunk_metadata_sample.txt"


def format_chunk(chunk):
    meta = chunk["metadata"]
    return (
        f"source={meta['source']} chunk_index={meta['chunk_index']} "
        f"char_start={meta['char_start']} page={meta['page']} section={meta['section']}\n"
        f"text: {chunk['text'][:120]!r}"
    )


def trace_chunk_to_source(chunk, documents_by_source):
    """Proves the chunk's text matches the original document at char_start."""
    meta = chunk["metadata"]
    original_text = documents_by_source[meta["source"]]

    if meta["page"] is not None:
        original_text = original_text.split("\f")[meta["page"] - 1]

    start = meta["char_start"]
    end = start + len(chunk["text"])
    original_slice = original_text[start:end]

    return original_slice == chunk["text"], original_slice


def main():
    documents = load_documents(str(DATA_FOLDER))
    documents_by_source = {doc["source"]: doc["text"] for doc in documents}

    # Small chunk_size so the tiny sample documents still produce several
    # chunks each, to demonstrate chunk_index/char_start progressing
    # consistently across a document.
    all_chunks = tag_corpus(documents, chunk_size=80, overlap=20)

    lines = [f"Tagged {len(all_chunks)} chunks from {len(documents)} documents.\n"]

    # Show a handful of sample chunks with their metadata.
    for chunk in all_chunks[:6]:
        lines.append(format_chunk(chunk))
        lines.append("")

    # Task 4: trace one retrieved chunk back to its exact source.
    retrieved = all_chunks[3] if len(all_chunks) > 3 else all_chunks[0]
    matches, original_slice = trace_chunk_to_source(retrieved, documents_by_source)
    meta = retrieved["metadata"]

    lines.append("=== Traceback demo ===")
    lines.append(
        f"Retrieved chunk claims to come from: {meta['source']} "
        f"(chunk {meta['chunk_index']}, page {meta['page']}, section {meta['section']})"
    )
    lines.append(f"Text at that exact location in the source document: {original_slice[:120]!r}")
    lines.append(f"Matches retrieved chunk text: {matches}")

    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nSample chunks + traceback demo written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
