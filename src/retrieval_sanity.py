"""
Retrieval sanity tests.

Checks whether known relevant chunks rank above unrelated chunks
using the same embedding model used during document embedding.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from numpy import array, dot
from numpy.linalg import norm

from document_loader import load_documents
from text_cleaning import clean
from chunk_metadata import tag_chunks


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"
OUTPUT_FOLDER = BASE_DIR / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

REPORT_FILE = OUTPUT_FOLDER / "retrieval_sanity_report.txt"


# ---------------------------------------------------------
# Embedding and similarity functions
# ---------------------------------------------------------

def generate_embedding(text):
    """Generate an embedding vector for the supplied text."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""
    a = array(vector_a)
    b = array(vector_b)

    denominator = norm(a) * norm(b)

    if denominator == 0:
        return 0.0

    return float(dot(a, b) / denominator)


# ---------------------------------------------------------
# Load and prepare corpus
# ---------------------------------------------------------

def prepare_chunks():
    """Load, clean and chunk the corpus."""
    documents = load_documents(str(DATA_FOLDER))

    chunks = []

    for document in documents:
        cleaned_document = {
            "source": document["source"],
            "text": clean(document["text"]),
        }

        document_chunks = tag_chunks(cleaned_document)

        chunks.extend(document_chunks)

    return chunks


# ---------------------------------------------------------
# Known test cases
# ---------------------------------------------------------

TEST_CASES = [
    {
        "name": "Property insurance",
        "query": "What does property insurance protect?",
        "expected_source": "sample.md",
        "unrelated_source": "sample.txt",
    },
    {
        "name": "Travel insurance",
        "query": "What does travel insurance cover?",
        "expected_source": "sample.pdf",
        "unrelated_source": "sample.md",
    },
    {
        "name": "Health insurance",
        "query": "What expenses does health insurance cover?",
        "expected_source": "sample.txt",
        "unrelated_source": "sample.pdf",
    },
    {
        "name": "Ambiguous insurance query",
        "query": "What insurance covers expenses?",
        "expected_source": "sample.txt",
        "unrelated_source": "sample.pdf",
    },
]


# ---------------------------------------------------------
# Run tests
# ---------------------------------------------------------

def run_tests(chunks):
    """Run known relevance tests and return report information."""

    print("\n" + "=" * 70)
    print("RETRIEVAL SANITY TESTS")
    print("=" * 70)

    print("\nLoading and preparing corpus...")

    print(f"Chunks available for testing : {len(chunks)}")

    # Generate embeddings for every corpus chunk.
    for chunk in chunks:
        chunk["embedding"] = generate_embedding(chunk["text"])

    results = []

    for test_number, test in enumerate(TEST_CASES, start=1):

        query = test["query"]
        expected_source = test["expected_source"]
        unrelated_source = test["unrelated_source"]

        query_vector = generate_embedding(query)

        ranked_results = []

        for chunk in chunks:
            score = cosine_similarity(
                query_vector,
                chunk["embedding"],
            )

            ranked_results.append(
                {
                    "source": chunk["metadata"]["source"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "score": score,
                }
            )

        ranked_results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        expected_result = next(
            item
            for item in ranked_results
            if item["source"] == expected_source
        )

        unrelated_result = next(
            item
            for item in ranked_results
            if item["source"] == unrelated_source
        )

        # The first three tests are strict ranking tests.
        # The fourth test intentionally checks an ambiguous query.
        if test_number <= 3:
            passed = (
                ranked_results[0]["source"] == expected_source
                and expected_result["score"] > unrelated_result["score"]
            )

            status = "PASS" if passed else "FAIL"

            note = (
                "Related source ranked above unrelated source."
                if passed
                else "Related source did not rank above unrelated source."
            )

        else:
            # This is intentionally treated as a borderline/surprising case.
            # The query is vague, so even a reasonable ranking is worth
            # recording as a retrieval-quality warning.
            passed = ranked_results[0]["source"] == expected_source
            status = "BORDERLINE"

            note = (
                "The query is ambiguous because it does not specify "
                "the insurance type or expense. This shows that vague "
                "queries can reduce retrieval confidence."
            )

        print("\n" + "-" * 70)
        print(f"Test {test_number} - {test['name']}")
        print(f"Query : {query}")
        print(f"Expected source : {expected_source}")
        print(f"Result : {status}")

        print("\nTop-ranked results:")

        for position, result in enumerate(ranked_results, start=1):
            print(
                f"{position}. "
                f"{result['source']} "
                f"(score={result['score']:.4f})"
            )

        print(
            f"\nExpected source score : "
            f"{expected_result['score']:.4f}"
        )

        print(
            f"Unrelated source score : "
            f"{unrelated_result['score']:.4f}"
        )

        print(f"Note : {note}")

        results.append(
            {
                "test_number": test_number,
                "name": test["name"],
                "query": query,
                "expected_source": expected_source,
                "unrelated_source": unrelated_source,
                "status": status,
                "ranked_results": ranked_results,
                "expected_score": expected_result["score"],
                "unrelated_score": unrelated_result["score"],
                "note": note,
            }
        )

    return results


# ---------------------------------------------------------
# Create report
# ---------------------------------------------------------

def create_report(results, chunk_count):
    """Create and save the retrieval sanity report."""

    strict_tests = [
        result
        for result in results
        if result["status"] in ("PASS", "FAIL")
    ]

    passes = sum(
        result["status"] == "PASS"
        for result in strict_tests
    )

    failures = sum(
        result["status"] == "FAIL"
        for result in strict_tests
    )

    borderline = sum(
        result["status"] == "BORDERLINE"
        for result in results
    )

    lines = [
        "RETRIEVAL SANITY TEST REPORT",
        "=" * 70,
        "",
        f"Embedding model : {EMBEDDING_MODEL}",
        f"Total chunks tested : {chunk_count}",
        "",
    ]

    for result in results:

        lines.append(
            f"TEST {result['test_number']} - {result['name']}"
        )
        lines.append("")
        lines.append(f"Query : {result['query']}")
        lines.append(
            f"Expected source : {result['expected_source']}"
        )
        lines.append(
            f"Unrelated source : {result['unrelated_source']}"
        )
        lines.append(f"Result : {result['status']}")
        lines.append("")
        lines.append("Top-ranked sources:")

        for position, ranked in enumerate(
            result["ranked_results"],
            start=1
        ):
            lines.append(
                f"{position}. "
                f"{ranked['source']} | "
                f"score={ranked['score']:.4f} | "
                f"chunk={ranked['chunk_index']}"
            )

        lines.append(
            f"Expected source score : "
            f"{result['expected_score']:.4f}"
        )

        lines.append(
            f"Unrelated source score : "
            f"{result['unrelated_score']:.4f}"
        )

        lines.append(
            f"Note : {result['note']}"
        )

        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    lines.extend(
        [
            f"Test count : {len(results)}",
            f"Passes : {passes}",
            f"Failures : {failures}",
            f"Borderline / surprising cases : {borderline}",
            "",
            "Overall observation:",
            (
                f"{passes} known relevance tests passed. "
                f"{borderline} ambiguous/borderline case was recorded "
                "to show how vague queries can affect retrieval."
            ),
            "",
            "Why this matters:",
            (
                "Sanity testing helps verify that semantically related "
                "chunks rank above unrelated chunks before retrieval is "
                "trusted in the RAG system."
            ),
        ]
    )

    report = "\n".join(lines)

    print("\n" + "=" * 70)
    print(report)

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"\nSanity report saved to {REPORT_FILE}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    chunks = prepare_chunks()

    if not chunks:
        raise RuntimeError(
            "No chunks were created from the corpus."
        )

    results = run_tests(chunks)

    create_report(
        results,
        len(chunks)
    )


if __name__ == "__main__":
    main()