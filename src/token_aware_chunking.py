from __future__ import annotations

from pathlib import Path
from statistics import mean

import tiktoken

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned" / "claims_guideline.txt"
REPORT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "token_aware_chunking_report.txt"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "token_aware_sample_chunks.txt"
TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")
TARGET_TOKENS = 120
OVERLAP_TOKENS = 25


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def chunk_tokens(tokens: list[int], chunk_size: int, overlap: int = 0) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    step = chunk_size - overlap
    chunks: list[list[int]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(window)
        if start + chunk_size >= len(tokens):
            break
    return chunks


def decode_chunk(token_ids: list[int]) -> str:
    return TOKENIZER.decode(token_ids).strip()


def summarize(chunks: list[list[int]]) -> tuple[int, float]:
    if not chunks:
        return 0, 0.0
    return len(chunks), mean(len(chunk) for chunk in chunks)


def overlap_demo(no_overlap: list[str], overlap_chunks: list[str]) -> str:
    boundary_phrase = "CHAPTER 2: DAMAGE ASSESSMENT"
    no_overlap_hit = next((chunk for chunk in no_overlap if boundary_phrase in chunk), None)
    overlap_hit = next((chunk for chunk in overlap_chunks if boundary_phrase in chunk), None)
    return (
        "Boundary phrase without overlap: "
        f"{no_overlap_hit or 'not observed'}\n"
        "Boundary phrase with overlap: "
        f"{overlap_hit or 'not observed'}"
    )


def main() -> None:
    text = load_text(DOC_PATH)
    all_tokens = TOKENIZER.encode(text)

    no_overlap_chunks = chunk_tokens(all_tokens, chunk_size=TARGET_TOKENS, overlap=0)
    overlap_chunks = chunk_tokens(all_tokens, chunk_size=TARGET_TOKENS, overlap=OVERLAP_TOKENS)

    no_overlap_text = [decode_chunk(chunk) for chunk in no_overlap_chunks]
    overlap_text = [decode_chunk(chunk) for chunk in overlap_chunks]

    no_count, no_avg = summarize(no_overlap_chunks)
    ov_count, ov_avg = summarize(overlap_chunks)

    report_lines = [
        "Token-aware chunking report",
        "=" * 60,
        f"Tokenizer: gpt-4o-mini",
        f"Chunk size: {TARGET_TOKENS} tokens",
        f"Overlap: {OVERLAP_TOKENS} tokens",
        "",
        f"No-overlap chunks: count={no_count}, average_tokens={no_avg:.1f}",
        f"Overlap chunks: count={ov_count}, average_tokens={ov_avg:.1f}",
        "",
        "Justification:",
        "- 120 tokens is small enough to stay well below the typical gpt-4o-mini context window while still preserving a complete claims-policy idea.",
        "- 25-token overlap preserves the prior boundary context so cross-chunk meaning remains intact without massively raising cost.",
        "",
        "Overlap demonstration:",
        overlap_demo(no_overlap_text, overlap_text),
        "",
        "Sample no-overlap chunk:",
        no_overlap_text[0],
        "",
        "Sample overlap chunk:",
        overlap_text[0],
    ]

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    SAMPLE_PATH.write_text(
        "\n\n".join(
            [
                "=== No-overlap sample ===",
                no_overlap_text[0],
                "=== Overlap sample ===",
                overlap_text[0],
                "=== Overlap boundary demo ===",
                overlap_demo(no_overlap_text, overlap_text),
            ]
        ),
        encoding="utf-8",
    )

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
