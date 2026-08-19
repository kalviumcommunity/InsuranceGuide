from __future__ import annotations

import os
from pathlib import Path

from chromadb import PersistentClient

BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = os.getenv(
    "VECTOR_DB_PATH",
    str(BASE_DIR / "outputs" / "chroma_local")
)

COLLECTION_NAME = os.getenv(
    "VECTOR_COLLECTION",
    "insurance_chunks"
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "3072")
)


def create_collection(
    db_path: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    dimension: int = EMBEDDING_DIMENSION,
):
    """
    Create or open the ChromaDB collection.

    The collection is configured for cosine similarity and the
    embedding dimension used by the Gemini embedding model.
    """

    client = PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={
            "hnsw:space": "cosine",
            "dimension": str(dimension),
        },
    )

    return collection


def insert_test_record(
    collection,
    record_id: str,
    text: str,
    metadata: dict,
    vector: list[float],
):
    """
    Insert one vector record into ChromaDB.

    This helper is used by tests, so the vector dimension is validated
    against the actual collection dimension rather than the production
    Gemini embedding dimension.
    """

    if not vector:
        raise ValueError("Vector cannot be empty.")

    collection_dimension = None

    try:
        collection_metadata = collection.metadata or {}
        configured_dimension = collection_metadata.get("dimension")

        if configured_dimension is not None:
            collection_dimension = int(configured_dimension)
    except (TypeError, ValueError):
        collection_dimension = None

    if (
        collection_dimension is not None
        and len(vector) != collection_dimension
    ):
        raise ValueError(
            f"Invalid vector dimension: expected "
            f"{collection_dimension}, got {len(vector)}"
        )

    collection.upsert(
        ids=[record_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[metadata],
    )

    return {
        "id": record_id,
        "text": text,
        "metadata": metadata,
        "dimension": len(vector),
    }


def read_test_record(collection, record_id: str):
    """
    Read back one stored record for verification.
    """

    return collection.get(
        ids=[record_id],
        include=[
            "embeddings",
            "documents",
            "metadatas",
        ],
    )


def main():
    collection = create_collection()

    print("Vector DB collection")
    print("=" * 60)
    print(f"Collection name : {COLLECTION_NAME}")
    print(f"Database path   : {DB_PATH}")
    print(f"Expected vector : {EMBEDDING_DIMENSION}")
    print(f"Stored records  : {collection.count()}")


if __name__ == "__main__":
    main()