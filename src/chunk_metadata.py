"""Chunks documents and tags every chunk with source-tracking metadata.

Every chunk gets the same metadata keys regardless of source format
(source, chunk_index, char_start, page, section) so downstream retrieval
and citation code can rely on one consistent shape. Fields that don't
apply to a given format (e.g. "page" for a .txt file) are simply None.
"""

import re

HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def simple_chunk(text, chunk_size=300, overlap=50):
    """Splits text into overlapping (chunk_text, char_start) windows."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append((text[start:end], start))
        if end == length:
            break
        start = end - overlap

    return chunks


def nearest_markdown_section(text, char_start):
    """Returns the closest heading at or before char_start, if any."""
    heading = None
    for match in HEADING_RE.finditer(text):
        if match.start() > char_start:
            break
        heading = match.group(1).strip()
    return heading


def tag_chunks(document, chunk_size=300, overlap=50):
    """Chunks one document and attaches consistent metadata to every chunk.

    document: {"source": str, "text": str} (as produced by document_loader)
    Returns: list of {"text": str, "metadata": {source, chunk_index,
             char_start, page, section}}
    """
    source = document["source"]
    extension = source.rsplit(".", 1)[-1].lower() if "." in source else ""
    tagged = []
    chunk_index = 0

    if extension == "pdf":
        pages_text = document["text"].split("\f")
        for page_number, page_text in enumerate(pages_text, start=1):
            for chunk_text, char_start in simple_chunk(page_text, chunk_size, overlap):
                if not chunk_text.strip():
                    continue
                tagged.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_index": chunk_index,
                        "char_start": char_start,
                        "page": page_number,
                        "section": None,
                    },
                })
                chunk_index += 1
        return tagged

    full_text = document["text"]
    for chunk_text, char_start in simple_chunk(full_text, chunk_size, overlap):
        if not chunk_text.strip():
            continue
        section = (
            nearest_markdown_section(full_text, char_start)
            if extension == "md"
            else None
        )
        tagged.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "chunk_index": chunk_index,
                "char_start": char_start,
                "page": None,
                "section": section,
            },
        })
        chunk_index += 1

    return tagged


def tag_corpus(documents, chunk_size=300, overlap=50):
    """Runs tag_chunks across every document in the corpus."""
    all_chunks = []
    for document in documents:
        all_chunks.extend(tag_chunks(document, chunk_size, overlap))
    return all_chunks
