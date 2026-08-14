from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

# Ensure local imports work when running from repo root
import sys
sys.path.insert(0, str(ROOT / "src"))

from citations import answer_with_citations, find_cited_markers, verify_citation, demo_llm_fn, get_demo_chunks


TEST_SET = ROOT / "data" / "test_set.json"
OUTPUT_JSON = ROOT / "outputs" / "evaluation_results.json"
OUTPUT_SUMMARY = ROOT / "outputs" / "evaluation_summary.txt"
OUTPUT_FAILURES = ROOT / "outputs" / "evaluation_failures.txt"


def extract_keywords(text: str):
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stop = {"what", "does", "cover", "my", "the", "and", "for", "policy", "insurance"}
    return [w for w in words if w not in stop]


def evaluate_entry(entry: dict):
    question = entry.get("query")
    expected_source = entry.get("expected_source")

    # Use demo chunks + demo LLM when available to avoid external API calls
    chunks, used_placeholder = get_demo_chunks(question)
    result = answer_with_citations(question, k=4, chunks=chunks, llm_fn=demo_llm_fn)

    answer = result.get("answer", "")
    fabricated = result.get("fabricated_markers", [])
    used_fallback = result.get("used_fallback", False)
    citation_map = result.get("citations", {})

    cited_markers = find_cited_markers(answer)

    # Correctness heuristic: expected source appears among cited sources
    cited_sources = {v.get("source") for v in citation_map.values()}
    correctness = expected_source in cited_sources if expected_source else False

    # Grounding heuristic: no fallback, no fabricated markers
    grounded = (not used_fallback) and (len(fabricated) == 0)

    keywords = extract_keywords(question)

    citation_checks = []

    for marker in cited_markers:
        m = f"[{marker}]"
        ver = verify_citation(citation_map, m)
        found = ver.get("found", False)
        text = ver.get("text", "") or ""
        text_l = text.lower()
        supports = any(k in text_l for k in keywords) if keywords else found
        citation_checks.append({"marker": m, "found": found, "source": ver.get("source"), "supports": supports})

    citation_accuracy = round(mean([1.0 if c["supports"] else 0.0 for c in citation_checks]) if citation_checks else 0.0, 4)

    return {
        "id": entry.get("id"),
        "query": question,
        "answer": answer,
        "used_fallback": used_fallback,
        "fabricated_markers": fabricated,
        "cited_markers": [f"[{m}]" for m in cited_markers],
        "citation_map_summary": {k: {"source": v.get("source"), "chunk_index": v.get("chunk_index")} for k, v in citation_map.items()},
        "correctness": correctness,
        "grounded": grounded,
        "citation_checks": citation_checks,
        "citation_accuracy": citation_accuracy,
    }


def run_evaluation(test_set_path: Path | str | None = None):
    path = Path(test_set_path) if test_set_path else TEST_SET
    with path.open("r", encoding="utf-8") as fh:
        queries = json.load(fh)

    results = []

    for entry in queries:
        row = evaluate_entry(entry)
        results.append(row)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Summarize metrics
    correctness_vals = [1.0 if r["correctness"] else 0.0 for r in results]
    grounding_vals = [1.0 if r["grounded"] else 0.0 for r in results]
    citation_accs = [r["citation_accuracy"] for r in results]

    summary_lines = []
    summary_lines.append("RAG EVALUATION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Queries evaluated: {len(results)}")
    summary_lines.append(f"Correctness (expected source present): {mean(correctness_vals):.3f}")
    summary_lines.append(f"Grounding (no fallback, no fabricated markers): {mean(grounding_vals):.3f}")
    summary_lines.append(f"Mean citation accuracy (keyword support heuristic): {mean(citation_accs):.3f}")
    summary_lines.append("")
    summary_lines.append("Per-query details:")

    failures = []
    for r in results:
        summary_lines.append(f"- {r['id']}: correctness={r['correctness']} grounded={r['grounded']} citation_accuracy={r['citation_accuracy']}")
        if not r['correctness'] or not r['grounded'] or r['citation_accuracy'] < 1.0:
            failures.append(r)

    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))

    # Write a human-readable failures file
    with OUTPUT_FAILURES.open("w", encoding="utf-8") as fh:
        fh.write("FAILURES\n")
        fh.write("=" * 60 + "\n\n")
        for f in failures:
            fh.write(json.dumps(f, indent=2))
            fh.write("\n\n")

    return results


def main():
    print("Running RAG evaluation on test set...")
    results = run_evaluation()
    print(f"Wrote results to: {OUTPUT_JSON}")
    print(f"Summary written to: {OUTPUT_SUMMARY}")
    print(f"Failures written to: {OUTPUT_FAILURES}")


if __name__ == "__main__":
    main()
