from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned" / "claims_guideline.txt"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "chunking_comparison_output.txt"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "outputs" / "sample_chunks.txt"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def split_by_paragraphs(text: str) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    return paragraphs


def split_fixed_overlap(text: str, chunk_size: int = 45, overlap: int = 10) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


def chunk_stats(chunks: list[str]) -> tuple[int, float]:
    if not chunks:
        return 0, 0.0
    return len(chunks), mean(len(chunk) for chunk in chunks)


def main() -> None:
    text = load_text(DOC_PATH)
    paragraph_chunks = split_by_paragraphs(text)
    fixed_chunks = split_fixed_overlap(text)

    paragraph_count, paragraph_avg = chunk_stats(paragraph_chunks)
    fixed_count, fixed_avg = chunk_stats(fixed_chunks)

    report_lines = [
        "Chunking strategy comparison on claims_guideline.txt",
        "=" * 60,
        f"Paragraph strategy: chunks={paragraph_count}, avg_size={paragraph_avg:.1f} chars",
        f"Fixed-overlap strategy: chunks={fixed_count}, avg_size={fixed_avg:.1f} chars",
        "",
        "Recommended choice: paragraph chunking",
        "Reason: the corpus is structured as chapter headings plus short topical paragraphs.",
        "Paragraph boundaries preserve the policy narrative and avoid cutting across legal/claims logic.",
        "",
        "Sample paragraph chunks:",
    ]

    report_lines.extend(f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(paragraph_chunks))
    report_lines.extend(["", "Sample fixed-overlap chunks:"])
    report_lines.extend(f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(fixed_chunks[:3]))

    OUTPUT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    SAMPLE_PATH.write_text(
        "\n\n".join(
            [
                "=== Paragraph sample ===",
                *paragraph_chunks[:2],
                "=== Fixed-overlap sample ===",
                *fixed_chunks[:2],
            ]
        ),
        encoding="utf-8",
    )

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
