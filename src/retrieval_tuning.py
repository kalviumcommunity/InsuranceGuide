import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL not found in .env")

client = genai.Client(api_key=API_KEY)

DB_PATH = "vector_store"
COLLECTION_NAME = "insurance_chunks"

OUTPUT_FILE = Path(
    "outputs/retrieval_tuning_report.txt"
)

TEST_QUERIES = [
    {
        "query": "What does property insurance protect?",
        "expected_source": "sample.md",
    },
    {
        "query": "What does travel insurance cover?",
        "expected_source": "sample.pdf",
    },
    {
        "query": "What expenses does health insurance cover?",
        "expected_source": "sample.txt",
    },
]


def generate_embedding(text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    return response.embeddings[0].values


def search_collection(collection, query, k):
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ranked_results = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        # Chroma returns distance.
        # Convert it to a simple similarity-style score.
        score = 1 - distance

        ranked_results.append(
            {
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "text": document,
                "score": score,
            }
        )

    return ranked_results


def run_experiment(collection, k):
    results = []

    for test in TEST_QUERIES:
        ranked = search_collection(
            collection,
            test["query"],
            k,
        )

        top_source = ranked[0]["source"]

        top_1_hit = (
            top_source == test["expected_source"]
        )

        top_k_hit = any(
            result["source"] == test["expected_source"]
            for result in ranked
        )

        results.append(
            {
                "query": test["query"],
                "expected_source": test["expected_source"],
                "ranked": ranked,
                "top_1_hit": top_1_hit,
                "top_k_hit": top_k_hit,
            }
        )

    return results


def calculate_rate(results, key):
    if not results:
        return 0.0

    hits = sum(
        1
        for result in results
        if result[key]
    )

    return hits / len(results) * 100


def main():
    print("=" * 70)
    print("RETRIEVAL TUNING EXPERIMENT")
    print("=" * 70)

    print("\nConnecting to ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection : {COLLECTION_NAME}"
    )

    print(
        f"Indexed chunks : {collection.count()}"
    )

    print("\nRunning retrieval experiments...")

    settings = [
        {
            "name": "SETTING A",
            "k": 1,
        },
        {
            "name": "SETTING B",
            "k": 3,
        },
    ]

    experiment_results = []

    for setting in settings:
        print(
            f"\n{setting['name']} - k={setting['k']}"
        )

        results = run_experiment(
            collection,
            setting["k"],
        )

        top_1_rate = calculate_rate(
            results,
            "top_1_hit",
        )

        top_k_rate = calculate_rate(
            results,
            "top_k_hit",
        )

        experiment_results.append(
            {
                "name": setting["name"],
                "k": setting["k"],
                "results": results,
                "top_1_rate": top_1_rate,
                "top_k_rate": top_k_rate,
            }
        )

        print(
            f"Top-1 Hit Rate : {top_1_rate:.1f}%"
        )

        print(
            f"Top-{setting['k']} Hit Rate : "
            f"{top_k_rate:.1f}%"
        )

    # Choose the setting with the best Top-1 result.
    best_setting = max(
        experiment_results,
        key=lambda item: (
            item["top_1_rate"],
            item["top_k_rate"],
        ),
    )

    report = []

    report.append(
        "RETRIEVAL TUNING EXPERIMENT"
    )
    report.append("=" * 70)

    report.append(
        f"Embedding model : {EMBEDDING_MODEL}"
    )

    report.append(
        f"Collection : {COLLECTION_NAME}"
    )

    report.append(
        f"Total test queries : {len(TEST_QUERIES)}"
    )

    for experiment in experiment_results:

        report.append("\n" + "-" * 70)

        report.append(
            f"{experiment['name']} - "
            f"k={experiment['k']}"
        )

        report.append(
            f"Top-1 Hit Rate : "
            f"{experiment['top_1_rate']:.1f}%"
        )

        report.append(
            f"Top-{experiment['k']} Hit Rate : "
            f"{experiment['top_k_rate']:.1f}%"
        )

        for index, result in enumerate(
            experiment["results"],
            start=1,
        ):
            report.append(
                f"\nQuery {index}: "
                f"{result['query']}"
            )

            report.append(
                f"Expected source : "
                f"{result['expected_source']}"
            )

            report.append(
                "Ranked results:"
            )

            for rank, item in enumerate(
                result["ranked"],
                start=1,
            ):
                report.append(
                    f"{rank}. "
                    f"{item['source']} "
                    f"(score={item['score']:.4f})"
                )

            report.append(
                f"Top-1 hit : "
                f"{result['top_1_hit']}"
            )

            report.append(
                f"Top-{experiment['k']} hit : "
                f"{result['top_k_hit']}"
            )

    report.append("\n" + "=" * 70)

    report.append(
        "\nBEST SETTING"
    )

    report.append(
        f"Setting : {best_setting['name']}"
    )

    report.append(
        f"k : {best_setting['k']}"
    )

    report.append(
        f"Top-1 Hit Rate : "
        f"{best_setting['top_1_rate']:.1f}%"
    )

    report.append(
        f"Top-{best_setting['k']} Hit Rate : "
        f"{best_setting['top_k_rate']:.1f}%"
    )

    report.append(
        "\nJUSTIFICATION"
    )

    report.append(
        "The selected setting achieved the strongest "
        "retrieval relevance on the known test queries. "
        "The experiment shows how changing k affects "
        "the number of relevant chunks returned."
    )

    report.append(
        "\nConclusion"
    )

    report.append(
        "Retrieval settings should be evaluated using "
        "known queries and expected sources rather than "
        "assuming that the first configuration is optimal."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        f"\nReport saved to {OUTPUT_FILE}"
    )

    print("\nBest setting:")
    print(
        f"k={best_setting['k']}, "
        f"Top-1 Hit Rate="
        f"{best_setting['top_1_rate']:.1f}%"
    )


if __name__ == "__main__":
    main()