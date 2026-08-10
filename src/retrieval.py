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

def retrieve(query, k=3, collection=None, where=None, keyword_terms=None, hybrid=False):
    """Embed `query`, run a top-k similarity search against the vector
    database, and return the k most similar chunks.

    Optional parameters:
        where - Chroma metadata filter predicate. Restricts the vector
                search to a subset of the corpus before ranking.
        keyword_terms - exact terms that can be matched against the
                         returned chunk text for precision boosting.
        hybrid - if True, the raw vector scores are re-ranked using the
                 keyword match count once results have been returned.

    Each returned item has:
        score    - similarity score, higher means more similar. The
                   collection is configured with hnsw:space="cosine",
                   so Chroma returns a cosine *distance*; we convert it
                   to a similarity with score = 1 - distance.
        text     - the retrieved chunk text
        metadata - source-tracking metadata attached at chunking time
                   (source, chunk_index, char_start, page, section)
        keyword_hits - number of explicit keyword terms found in the
                       chunk text, if keyword_terms are supplied.
        hybrid_score - optional re-ranked score if hybrid=True.
    """
    if collection is None:
        collection = get_collection()

    if keyword_terms is None:
        keyword_terms = []
    normalized_terms = [term.lower() for term in keyword_terms if isinstance(term, str) and term.strip()]

    query_vector = embed_query(query)

    search_args = {
        "query_embeddings": [query_vector],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        search_args["where"] = where

    results = collection.query(**search_args)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        score = round(1 - distance, 4)
        keyword_hits = 0
        if normalized_terms:
            normalized_text = text.lower()
            for term in normalized_terms:
                if term in normalized_text:
                    keyword_hits += 1

        hybrid_score = score
        if hybrid and normalized_terms:
            hybrid_score = round(score + min(keyword_hits * 0.05, 0.25), 4)

        item = {
            "score": score,
            "text": text,
            "metadata": metadata,
            "keyword_hits": keyword_hits,
            "hybrid_score": hybrid_score,
        }
        if hybrid:
            item["score"] = hybrid_score
        retrieved.append(item)

    if hybrid and normalized_terms:
        retrieved.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return retrieved


# ---------------------------------------------------------
# Demo / sample-results report
# ---------------------------------------------------------

def format_result_block(rank, result):
    metadata = result.get("metadata", {})
    return "\n".join(
        [
            f"  rank         : {rank}",
            f"  score        : {result.get('score')}",
            f"  hybrid_score : {result.get('hybrid_score')}",
            f"  keyword_hits : {result.get('keyword_hits', 0)}",
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


FILTERED_OUTPUT_FILE = BASE_DIR / "outputs" / "filtered_retrieval_demo_output.txt"


def run_filtered_demo(
    query=SAMPLE_QUERY,
    filter_meta={"section": "Property Insurance"},
    keyword_terms=None,
    k=3,
):
    """Show the same top-k query both unfiltered and filtered.

    The report is committed under outputs/ so the repository carries a
    sample result showing where the metadata restriction changes the
    retrieved context and how hybrid scoring adds exact-term precision.
    """
    if keyword_terms is None:
        keyword_terms = ["fire", "storm", "home", "policy"]

    collection = get_collection()

    unfiltered = retrieve(query, k=k, collection=collection, keyword_terms=keyword_terms, hybrid=True)
    filtered = retrieve(query, k=k, collection=collection, where=filter_meta, keyword_terms=keyword_terms, hybrid=True)

    lines = [
        "FILTERED RETRIEVAL DEMO",
        "=" * 84,
        f"Query            : {query}",
        f"Unfiltered k     : {k}",
        f"Filter predicate : {filter_meta}",
        f"Keyword terms    : {', '.join(keyword_terms)}",
        "",
        "UNFILTERED VECTOR SEARCH",
        "-" * 84,
    ]

    for rank, result in enumerate(unfiltered, start=1):
        lines.extend(
            [
                f"rank      : {rank}",
                f"score     : {result.get('score')}",
                f"hybrid    : {result.get('hybrid_score')}",
                f"keyword   : {result.get('keyword_hits', 0)}",
                f"source    : {result.get('metadata', {}).get('source')}",
                f"section   : {result.get('metadata', {}).get('section')}",
                f"text      : {result.get('text')}",
                "",
            ]
        )

    lines.extend([
        "FILTERED VECTOR SEARCH",
        "-" * 84,
    ])

    for rank, result in enumerate(filtered, start=1):
        lines.extend(
            [
                f"rank      : {rank}",
                f"score     : {result.get('score')}",
                f"hybrid    : {result.get('hybrid_score')}",
                f"keyword   : {result.get('keyword_hits', 0)}",
                f"source    : {result.get('metadata', {}).get('source')}",
                f"section   : {result.get('metadata', {}).get('section')}",
                f"text      : {result.get('text')}",
                "",
            ]
        )

    lines.extend([
        "OBSERVATION",
        "-" * 84,
        "The filtered call scopes retrieval to the Property Insurance section and should remove broad policy or unrelated product chunks from the ranking list.",
        "That is expected to improve precision, not just recall, because the filter and the keyword/phrase terms narrow the context stringently.",
        "The hybrid score adds a small explicit-term boost to vector results that contain the exact wording the question asks about.",
    ])

    text_report = "\n".join(lines)
    print(text_report)

    FILTERED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FILTERED_OUTPUT_FILE.write_text(text_report, encoding="utf-8")
    print(f"\nFiltered search sample results saved to {FILTERED_OUTPUT_FILE}")


def main():
    run_demo()
    run_filtered_demo()


if __name__ == "__main__":
    main()


