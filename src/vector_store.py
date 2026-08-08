from __future__ import annotations

import os
from pathlib import Path

from chromadb import PersistentClient

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "outputs" / "chroma_local"))
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION", "insurance_chunks")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))


def create_collection(db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME, dimension: int = EMBEDDING_DIMENSION):
    """Create or open a Chroma persistent collection with a stable embedding dimension."""
    client = PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "dimension": str(dimension)},
    )


def insert_test_record(collection, record_id: str, text: str, metadata: dict, vector: list[float]):
    """Insert one demonstration record and return the payload written to Chroma."""
    collection.add(
        ids=[record_id],
        embeddings=[vector],
        documents=[text],
        metadatas=[metadata],
    )
    return {"id": record_id, "text": text, "metadata": metadata, "dimension": len(vector)}


def read_test_record(collection, record_id: str):
    """Read back a single stored record by ID for verification."""
    return collection.get(ids=[record_id], include=["embeddings", "documents", "metadatas"])


def main():
    collection = create_collection()

    test_vector = [0.1, 0.2, 0.3]
    result = insert_test_record(
        collection,
        record_id="test-record-1",
        text="Property insurance protects a home from fire and storm damage.",
        metadata={
            "source": "sample.md",
            "chunk_index": 0,
            "section": "Property Insurance",
        },
        vector=test_vector,
    )

    readback = read_test_record(collection, record_id="test-record-1")

    print("Vector DB collection smoke test")
    print("=" * 60)
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Dimension: {EMBEDDING_DIMENSION}")
    print(f"Inserted id: {result['id']}")
    print(f"Vector length: {len(test_vector)}")
    print(f"Document text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
    print("Readback IDs:", readback.get("ids"))
    print("Readback embeddings length:", len(readback.get("embeddings", [[]])[0]))
    print("Readback documents:", readback.get("documents"))
    print("Readback metadata:", readback.get("metadatas"))


if __name__ == "__main__":
    main()
