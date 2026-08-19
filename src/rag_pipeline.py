"""
End-to-end RAG Pipeline

Flow:
User Query
    -> Cache Check
    -> Query Embedding
    -> ChromaDB Retrieval
    -> Context Assembly
    -> Gemini Generation
    -> Answer + Sources
    -> Cache Result
"""

import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieval import embed_query, get_collection

from observability import (
    elapsed_ms,
    log_request,
    timer,
)

from usage_tracker import calculate_usage

from query_cache import (
    get_cached_result,
    make_cache_key,
    set_cached_result,
)


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "rag_pipeline_output.txt"


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

def get_chat_model():
    """Read the configured chat model lazily from the environment."""

    model = os.getenv(
        "CHAT_MODEL",
        "gemini-2.0-flash",
    )

    if not model:
        raise ValueError(
            "CHAT_MODEL not found in environment."
        )

    return model


def get_gemini_client():
    """Create the Gemini client only when needed."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment."
        )

    return genai.Client(
        api_key=api_key
    )


# ---------------------------------------------------------
# 1. EMBED QUERY
# ---------------------------------------------------------

def embed_query_stage(query):
    """Convert the user query into an embedding vector."""

    return embed_query(query)


# ---------------------------------------------------------
# 2. RETRIEVE CONTEXT
# ---------------------------------------------------------

def retrieve_context(
    query_vector,
    k=3,
):
    """
    Retrieve the most relevant chunks from ChromaDB.
    """

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    chunks = []

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        chunks.append(
            {
                "text": text,
                "metadata": metadata or {},
                "score": round(
                    1 - distance,
                    4,
                ),
            }
        )

    return chunks


# ---------------------------------------------------------
# 3. ASSEMBLE CONTEXT
# ---------------------------------------------------------

def assemble_context(chunks):
    """
    Combine retrieved chunks into grounded context
    with source information.
    """

    if not chunks:
        return ""

    parts = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        chunk_index = metadata.get(
            "chunk_index",
            "unknown",
        )

        section = metadata.get(
            "section",
        )

        source_info = (
            f"[{index}] Source: {source} | "
            f"Chunk: {chunk_index}"
        )

        if section:
            source_info += (
                f" | Section: {section}"
            )

        parts.append(
            f"{source_info}\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------
# 4. GENERATE ANSWER
# ---------------------------------------------------------

def generate_answer(
    query,
    context,
):
    """
    Generate a grounded answer using only
    the retrieved context.
    """

    if not context:
        return (
            "I could not find relevant context "
            "for this question."
        )

    client = get_gemini_client()
    model_name = get_chat_model()

    prompt = f"""
Answer the user's question using ONLY the context provided below.

If the context does not contain enough information to answer,
say that the available documents do not contain enough information.

Do not invent facts.

For each factual claim, include the marker [n]
of the supporting context chunk.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )

    return response.text.strip()


# ---------------------------------------------------------
# 5. STREAMING RAG
# ---------------------------------------------------------

def stream_answer_query(
    query,
    k=3,
) -> Iterator[dict]:
    """
    Yield source metadata and answer text
    as it is generated.

    Streaming is kept separate from the normal
    cached query flow.
    """

    if query is None or not str(query).strip():
        raise ValueError(
            "Question is required."
        )

    query = str(query).strip()

    query_vector = embed_query_stage(
        query
    )

    chunks = retrieve_context(
        query_vector,
        k=k,
    )

    sources = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        )

        sources.append(
            {
                "marker": f"[{index}]",
                "source": metadata.get(
                    "source"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "section": metadata.get(
                    "section"
                ),
                "score": chunk.get(
                    "score"
                ),
                "text": chunk.get(
                    "text",
                    "",
                ),
            }
        )

    yield {
        "type": "sources",
        "sources": sources,
        "metadata": {
            "top_k": k,
            "cache_hit": False,
        },
    }

    if not chunks:
        yield {
            "type": "token",
            "text": (
                "I could not find relevant context "
                "for this question."
            ),
        }

        yield {
            "type": "done"
        }

        return

    context = assemble_context(
        chunks
    )

    prompt = f"""
Answer the user's question using ONLY the context provided below.

If the context does not contain enough information, say so.

Do not invent facts.

For each factual claim, include the marker [n]
of the supporting context chunk.

Context:
{context}

Question:
{query}
"""

    response = (
        get_gemini_client()
        .models
        .generate_content_stream(
            model=get_chat_model(),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )
    )

    for item in response:
        text = getattr(
            item,
            "text",
            None,
        )

        if text:
            yield {
                "type": "token",
                "text": text,
            }

    yield {
        "type": "done"
    }


# ---------------------------------------------------------
# 6. COMPLETE RAG PIPELINE WITH CACHE
# ---------------------------------------------------------

def answer_query(
    query,
    k=3,
):
    """
    Run the complete RAG pipeline with:

    - Query caching
    - Structured logging
    - Latency tracking
    - Approximate token tracking
    - Approximate cost tracking
    """

    if query is None or not str(query).strip():
        raise ValueError(
            "Question is required."
        )

    query = str(query).strip()

    start_time = timer()

    model = get_chat_model()

    # -----------------------------------------------------
    # CREATE CACHE KEY
    # -----------------------------------------------------

    cache_key = make_cache_key(
        question=query,
        k=k,
        model=model,
    )

    # -----------------------------------------------------
    # CHECK CACHE
    # -----------------------------------------------------

    cached_result = get_cached_result(
        cache_key
    )

    if cached_result is not None:

        result = dict(
            cached_result
        )

        result["cache_hit"] = True

        log_request(
            question=query,
            answer=result.get(
                "answer",
                "",
            ),
            sources=result.get(
                "sources",
                [],
            ),
            cache_hit=True,
            latency_ms=elapsed_ms(
                start_time
            ),
            token_usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            estimated_cost=0.0,
        )

        return result

    # -----------------------------------------------------
    # CACHE MISS
    # -----------------------------------------------------

    try:

        # -------------------------------------------------
        # STAGE 1: EMBED
        # -------------------------------------------------

        query_vector = embed_query_stage(
            query
        )

        # -------------------------------------------------
        # STAGE 2: RETRIEVE
        # -------------------------------------------------

        chunks = retrieve_context(
            query_vector,
            k=k,
        )

        # -------------------------------------------------
        # EMPTY RETRIEVAL
        # -------------------------------------------------

        if not chunks:

            answer = (
                "I could not find relevant context "
                "for this question."
            )

            usage = calculate_usage(
                query,
                answer,
            )

            result = {
                "answer": answer,
                "sources": [],
                "chunks": [],
                "context": "",
                "cache_hit": False,
                "usage": usage,
            }

            set_cached_result(
                cache_key,
                result,
            )

            log_request(
                question=query,
                answer=answer,
                sources=[],
                cache_hit=False,
                latency_ms=elapsed_ms(
                    start_time
                ),
                token_usage={
                    "input_tokens": usage.get(
                        "input_tokens",
                        0,
                    ),
                    "output_tokens": usage.get(
                        "output_tokens",
                        0,
                    ),
                    "total_tokens": usage.get(
                        "total_tokens",
                        0,
                    ),
                },
                estimated_cost=usage.get(
                    "estimated_cost",
                    0.0,
                ),
            )

            return result

        # -------------------------------------------------
        # STAGE 3: ASSEMBLE CONTEXT
        # -------------------------------------------------

        context = assemble_context(
            chunks
        )

        # -------------------------------------------------
        # STAGE 4: GENERATE ANSWER
        # -------------------------------------------------

        answer = generate_answer(
            query,
            context,
        )

        # -------------------------------------------------
        # PREPARE SOURCES
        # -------------------------------------------------

        sources = []

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            sources.append(
                {
                    "source": metadata.get(
                        "source"
                    ),
                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),
                    "section": metadata.get(
                        "section"
                    ),
                    "score": chunk.get(
                        "score"
                    ),
                }
            )

        # -------------------------------------------------
        # CALCULATE USAGE
        # -------------------------------------------------

        usage = calculate_usage(
            context + query,
            answer,
        )

        # -------------------------------------------------
        # BUILD RESULT
        # -------------------------------------------------

        result = {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "context": context,
            "cache_hit": False,
            "usage": usage,
        }

        # -------------------------------------------------
        # SAVE RESULT TO CACHE
        # -------------------------------------------------

        set_cached_result(
            cache_key,
            result,
        )

        # -------------------------------------------------
        # STRUCTURED REQUEST LOG
        # -------------------------------------------------

        log_request(
            question=query,
            answer=answer,
            sources=sources,
            cache_hit=False,
            latency_ms=elapsed_ms(
                start_time
            ),
            token_usage={
                "input_tokens": usage.get(
                    "input_tokens",
                    0,
                ),
                "output_tokens": usage.get(
                    "output_tokens",
                    0,
                ),
                "total_tokens": usage.get(
                    "total_tokens",
                    0,
                ),
            },
            estimated_cost=usage.get(
                "estimated_cost",
                0.0,
            ),
        )

        return result

    except Exception as exc:

        # -------------------------------------------------
        # LOG FAILURE
        # -------------------------------------------------

        log_request(
            question=query,
            answer="",
            sources=[],
            cache_hit=False,
            latency_ms=elapsed_ms(
                start_time
            ),
            token_usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            estimated_cost=0.0,
            error=str(exc),
        )

        raise


# ---------------------------------------------------------
# 7. SAMPLE END-TO-END RUN
# ---------------------------------------------------------

def main():

    query = (
        "What does property insurance protect?"
    )

    print("=" * 70)
    print("RAG PIPELINE")
    print("=" * 70)

    print(
        f"\nQuery:\n{query}"
    )

    print(
        "\n[1] Running RAG query..."
    )

    result = answer_query(
        query,
        k=3,
    )

    print(
        f"\nCache hit: "
        f"{result.get('cache_hit', False)}"
    )

    print(
        "\nGenerated Answer:"
    )

    print(
        result.get(
            "answer",
            "",
        )
    )

    print(
        "\nSources:"
    )

    for source in result.get(
        "sources",
        [],
    ):
        print(
            f"- {source.get('source')} "
            f"(chunk={source.get('chunk_index')}, "
            f"score={source.get('score')})"
        )

    usage = result.get(
        "usage",
        {},
    )

    report = [
        "RAG PIPELINE OUTPUT",
        "=" * 70,
        f"Embedding model : "
        f"{os.getenv('EMBEDDING_MODEL')}",
        f"Chat model      : "
        f"{get_chat_model()}",
        f"Query           : {query}",
        f"Cache hit       : "
        f"{result.get('cache_hit', False)}",
        f"Input tokens    : "
        f"{usage.get('input_tokens', 0)}",
        f"Output tokens   : "
        f"{usage.get('output_tokens', 0)}",
        f"Total tokens    : "
        f"{usage.get('total_tokens', 0)}",
        f"Estimated cost  : "
        f"{usage.get('estimated_cost', 0.0)}",
        "",
        "RETRIEVED SOURCES",
        "-" * 70,
    ]

    for rank, source in enumerate(
        result.get(
            "sources",
            [],
        ),
        start=1,
    ):
        report.append(
            f"{rank}. "
            f"{source.get('source')} | "
            f"chunk={source.get('chunk_index')} | "
            f"score={source.get('score')}"
        )

    report.extend(
        [
            "",
            "ASSEMBLED CONTEXT",
            "-" * 70,
            result.get(
                "context",
                "",
            ),
            "",
            "GENERATED ANSWER",
            "-" * 70,
            result.get(
                "answer",
                "",
            ),
            "",
            "PIPELINE FLOW",
            "-" * 70,
            "User Query",
            "    -> Cache Check",
            "    -> Query Embedding",
            "    -> ChromaDB Retrieval",
            "    -> Context Assembly",
            "    -> Gemini Generation",
            "    -> Answer + Sources",
            "    -> Cache Result",
        ]
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
        f"\nOutput saved to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()