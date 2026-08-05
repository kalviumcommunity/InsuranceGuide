"""Applies the cleaning pipeline to every raw document and reports before/after."""

from pathlib import Path

from text_cleaning import clean

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CLEANED_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"
REPORT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "cleaning_before_after.txt"


def load_corpus():
    return [
        {"source": p.name, "text": p.read_text(encoding="utf-8")}
        for p in sorted(RAW_DIR.glob("*.txt"))
    ]


def main():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_corpus()

    report_lines = []
    for doc in docs:
        before = doc["text"]
        after = clean(before)
        doc["text"] = after

        (CLEANED_DIR / doc["source"]).write_text(after, encoding="utf-8")

        report_lines.append(f"=== {doc['source']} ===")
        report_lines.append(f"chars: {len(before)} -> {len(after)}")
        report_lines.append("--- BEFORE (first 200 chars) ---")
        report_lines.append(before[:200])
        report_lines.append("--- AFTER (first 200 chars) ---")
        report_lines.append(after[:200])
        report_lines.append("")

        print(f"{doc['source']}: {len(before)} -> {len(after)} chars")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nFull before/after report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
