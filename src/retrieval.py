"""Similarity search & top-k retrieval.

Given a user query, embed it with the SAME embedding model that was used
to embed the document chunks, search the persisted vector database, and
return the top-k most similar chunks together with their similarity
score and source-tracking metadata. These chunks become the grounding
context that the RAG answer-generation step (next module) will pass to
the language model.

Query -> embed -> search -> ranked chunks with score + text + metadata.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from vector_store import create_collection, COLLECTION_NAME

# ---------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "similarity_search_topk_output.txt"

# Same query, run at different k values, so the assignment can show how
# the retrieved context changes as k grows.
SAMPLE_QUERY = "What does my policy cover if my house is damaged by fire or a storm?"
SAMPLE_K_VALUES = [1, 3, 5]

_client = None


# ---------------------------------------------------------
# Embedding
# ---------------------------------------------------------

def get_client():
    """Create the Gemini client lazily so importing this module (e.g.
    from tests) does not require GEMINI_API_KEY to be configured."""
    global _client
    if _client is None:
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")
        _client = genai.Client(api_key=API_KEY)
    return _client


def embed_query(query):
    """Embed a user query with the same embedding model used for the
    document chunks.

    If the query and the corpus were embedded with different models,
    the vector database can still return results, but the two sets of
    vectors would not live in the same semantic space, so the
    similarity ranking could not be trusted.
    """
    client = get_client()

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )

    return response.embeddings[0].values


# ---------------------------------------------------------
# Vector database
# ---------------------------------------------------------

def get_collection():
    """Open the persisted Chroma collection created during indexing
    (see index_embeddings.py / vector_store.py)."""
    return create_collection()


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve(query, k=3, collection=None):
    """Embed `query`, run a top-k similarity search against the vector
    database, and return the k most similar chunks.

    Each returned item has:
        score    - similarity score, higher means more similar. The
                   collection is configured with hnsw:space="cosine",
                   so Chroma returns a cosine *distance*; we convert it
                   to a similarity with score = 1 - distance.
        text     - the retrieved chunk text
        metadata - source-tracking metadata attached at chunking time
                   (source, chunk_index, char_start, page, section)
    """
    if collection is None:
        collection = get_collection()

    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "score": round(1 - distance, 4),
                "text": text,
                "metadata": metadata,
            }
        )

    return retrieved


# ---------------------------------------------------------
# Demo / sample-results report
# ---------------------------------------------------------

def format_result_block(rank, result):
    metadata = result.get("metadata", {})
    return "\n".join(
        [
            f"  rank         : {rank}",
            f"  score        : {result['score']}",
            f"  source       : {metadata.get('source')}",
            f"  chunk_index  : {metadata.get('chunk_index')}",
            f"  section      : {metadata.get('section') or None}",
            f"  text         : {result['text']}",
            "",
        ]
    )


def run_demo(query=SAMPLE_QUERY, k_values=SAMPLE_K_VALUES):
    """Run the same query at several k values and write the results to
    outputs/similarity_search_topk_output.txt as the assignment's
    committed sample query results."""

    collection = get_collection()

    report = [
        "SIMILARITY SEARCH & TOP-K RETRIEVAL - SAMPLE RUN",
        "=" * 70,
        f"Embedding model : {EMBEDDING_MODEL}",
        f"Collection      : {COLLECTION_NAME}",
        f"Query           : {query}",
        "",
    ]

    for k in k_values:
        results = retrieve(query, k=k, collection=collection)
        report.append(f"k = {k}  ->  {len(results)} chunk(s) returned")
        report.append("-" * 70)
        for rank, result in enumerate(results, start=1):
            report.append(format_result_block(rank, result))
        report.append("")

    report.append("OBSERVATION")
    report.append("-" * 70)
    report.append(
        "Increasing k returns more chunks from the same ranked list; it "
        "does not change the order of chunks already returned by a "
        "smaller k. Once k reaches the total number of indexed chunks, "
        "asking for a larger k stops returning new results and simply "
        "returns everything in the collection."
    )

    text_report = "\n".join(report)
    print(text_report)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(text_report, encoding="utf-8")
    print(f"\nSample results saved to {OUTPUT_FILE}")


def main():
    run_demo()


if __name__ == "__main__":
    main()
