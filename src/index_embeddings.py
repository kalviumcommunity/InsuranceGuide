"""
Index corpus embeddings into ChromaDB.

Tasks covered:
1. Load and prepare corpus chunks.
2. Generate embeddings for every chunk.
3. Store vector + text + metadata in ChromaDB.
4. Validate indexed count against chunk count.
5. Spot-check a stored record against its source chunk.
6. Write an indexing summary report.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from document_loader import load_documents
from text_cleaning import clean
from chunk_metadata import tag_chunks
from vector_store import create_collection, COLLECTION_NAME


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001",
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FOLDER = BASE_DIR / "data"

OUTPUT_FILE = (
    BASE_DIR
    / "outputs"
    / "indexing_summary.txt"
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "3072")
)


# ============================================================
# VALIDATION
# ============================================================

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=API_KEY)


# ============================================================
# EMBEDDING
# ============================================================

def generate_embedding(text):
    """
    Generate one embedding vector using Gemini.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )

    vector = response.embeddings[0].values

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension. "
            f"Expected {EMBEDDING_DIMENSION}, "
            f"got {len(vector)}"
        )

    return vector


# ============================================================
# PREPARE CORPUS
# ============================================================

def prepare_chunks():
    """
    Load documents, clean text, and create tagged chunks.
    """

    print("\nLoading and preparing corpus...")

    documents = load_documents(str(DATA_FOLDER))

    all_chunks = []

    for document in documents:

        cleaned_text = clean(document["text"])

        cleaned_document = {
            "source": document["source"],
            "text": cleaned_text,
        }

        chunks = tag_chunks(cleaned_document)

        all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# CREATE VECTOR RECORD
# ============================================================

def create_vector_record(chunk):
    """
    Convert one chunk into a ChromaDB record.

    Each record contains:
    - stable ID
    - embedding vector
    - source text
    - metadata
    """

    metadata = chunk.get("metadata", {})

    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", 0)

    record_id = chunk.get(
        "id",
        f"{source}::chunk_{chunk_index}",
    )

    vector = generate_embedding(chunk["text"])

    return {
        "id": record_id,
        "embedding": vector,
        "text": chunk["text"],
        "metadata": {
            "source": source,
            "chunk_index": chunk_index,
            "page": metadata.get("page", -1),
            "section": metadata.get("section", ""),
        },
    }


# ============================================================
# SPOT CHECK
# ============================================================

def spot_check(collection, source_chunk, stored_record_id):
    """
    Read one stored record and compare it with the
    original source chunk.
    """

    stored = collection.get(
        ids=[stored_record_id],
        include=[
            "embeddings",
            "documents",
            "metadatas",
        ],
    )

    if not stored["ids"]:
        return {
            "passed": False,
            "reason": "Stored record was not found.",
        }

    stored_id = stored["ids"][0]
    stored_text = stored["documents"][0]
    stored_metadata = stored["metadatas"][0]
    stored_vector = stored["embeddings"][0]

    source_metadata = source_chunk.get("metadata", {})

    text_matches = (
        stored_text == source_chunk["text"]
    )

    source_matches = (
        stored_metadata.get("source")
        == source_metadata.get("source")
    )

    chunk_index_matches = (
        stored_metadata.get("chunk_index")
        == source_metadata.get("chunk_index")
    )

    vector_length_matches = (
        len(stored_vector)
        == EMBEDDING_DIMENSION
    )

    passed = (
        text_matches
        and source_matches
        and chunk_index_matches
        and vector_length_matches
    )

    print("\n# Spot-check Stored Record")
    print()

    if passed:
        print("PASS - Stored record matches source chunk.")
    else:
        print("FAIL - Stored record does not match source chunk.")

    print(f"ID             : {stored_id}")
    print(
        f"Source         : "
        f"{stored_metadata.get('source')}"
    )
    print(
        f"Chunk Index    : "
        f"{stored_metadata.get('chunk_index')}"
    )
    print(
        f"Page           : "
        f"{stored_metadata.get('page')}"
    )
    print(
        f"Section        : "
        f"{stored_metadata.get('section')}"
    )
    print(
        f"Vector Length  : "
        f"{len(stored_vector)}"
    )

    print(
        f"Text           : "
        f"{stored_text[:150]}"
    )

    print(
        f"Vector Sample  : "
        f"{stored_vector[:8]}"
    )

    print(
        f"\nText matches source chunk : "
        f"{text_matches}"
    )

    print(
        f"Source matches            : "
        f"{source_matches}"
    )

    print(
        f"Chunk index matches       : "
        f"{chunk_index_matches}"
    )

    print(
        f"Vector length matches     : "
        f"{vector_length_matches}"
    )

    return {
        "passed": passed,
        "stored_id": stored_id,
        "text_matches": text_matches,
        "source_matches": source_matches,
        "chunk_index_matches": chunk_index_matches,
        "vector_length_matches": vector_length_matches,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INDEXING EMBEDDINGS INTO CHROMADB")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Prepare corpus
    # --------------------------------------------------------

    chunks = prepare_chunks()

    expected_count = len(chunks)

    print(
        f"Chunks produced : "
        f"{expected_count}"
    )

    # --------------------------------------------------------
    # Step 2: Connect to ChromaDB
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    collection = create_collection()

    existing_count = collection.count()

    print(
        f"Existing records found : "
        f"{existing_count}"
    )

    # --------------------------------------------------------
    # Step 3: Clear existing records
    # --------------------------------------------------------

    if existing_count > 0:

        print(
            "Clearing collection before re-indexing..."
        )

        # IMPORTANT:
        # Do NOT use collection.delete(where={})
        # ChromaDB rejects an empty where filter.

        existing_records = collection.get(
            include=[]
        )

        existing_ids = existing_records.get(
            "ids",
            []
        )

        if existing_ids:
            collection.delete(
                ids=existing_ids
            )

        print("Collection cleared.")

    # --------------------------------------------------------
    # Step 4: Generate embeddings
    # --------------------------------------------------------

    print(
        "\nGenerating embeddings and "
        "preparing records..."
    )

    records = []
    failures = []

    for index, chunk in enumerate(chunks):

        try:

            record = create_vector_record(chunk)

            records.append(record)

        except Exception as error:

            failures.append(
                {
                    "chunk_index": index,
                    "error": str(error),
                }
            )

            print(
                f"FAILED chunk {index}: "
                f"{error}"
            )

    print(
        f"Embeddings generated : "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Step 5: Insert records
    # --------------------------------------------------------

    print(
        "\nInserting records into ChromaDB..."
    )

    inserted = 0

    for record in records:

        try:

            collection.upsert(
                ids=[record["id"]],
                embeddings=[record["embedding"]],
                documents=[record["text"]],
                metadatas=[record["metadata"]],
            )

            inserted += 1

        except Exception as error:

            failures.append(
                {
                    "id": record["id"],
                    "error": str(error),
                }
            )

            print(
                f"FAILED record "
                f"{record['id']}: "
                f"{error}"
            )

    print(
        f"Records indexed : "
        f"{inserted}"
    )

    # --------------------------------------------------------
    # Step 6: Count validation
    # --------------------------------------------------------

    indexed_count = collection.count()

    count_passed = (
        indexed_count == expected_count
        and len(failures) == 0
    )

    print("\nCount Validation")

    if count_passed:
        print(
            "PASS - Indexed count matches "
            "the number of chunks."
        )
    else:
        print(
            "FAIL - Indexed count does not "
            "match the number of chunks."
        )

    # --------------------------------------------------------
    # Step 7: Spot check
    # --------------------------------------------------------

    spot_result = None

    if records and not failures:

        first_record = records[0]

        spot_result = spot_check(
            collection,
            chunks[0],
            first_record["id"],
        )

    else:

        print(
            "\nSpot-check skipped because "
            "indexing had failures."
        )

        spot_result = {
            "passed": False,
            "reason": "Indexing failures occurred.",
        }

    # --------------------------------------------------------
    # Step 8: Overall result
    # --------------------------------------------------------

    overall_passed = (
        count_passed
        and spot_result["passed"]
    )

    print("\n" + "=" * 70)

    print(
        f"Chunks produced : "
        f"{expected_count}"
    )

    print(
        f"Records indexed : "
        f"{indexed_count}"
    )

    print(
        f"Failures        : "
        f"{len(failures)}"
    )

    print(
        f"Count validation: "
        f"{'PASSED' if count_passed else 'FAILED'}"
    )

    print(
        f"Spot-check      : "
        f"{'PASSED' if spot_result['passed'] else 'FAILED'}"
    )

    print(
        f"Overall result  : "
        f"{'PASSED' if overall_passed else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Step 9: Write report
    # --------------------------------------------------------

    report = []

    report.append(
        "INDEXING SUMMARY"
    )

    report.append("=" * 70)

    report.append(
        f"Embedding model       : "
        f"{EMBEDDING_MODEL}"
    )

    report.append(
        f"Collection             : "
        f"{COLLECTION_NAME}"
    )

    report.append(
        "Vector database        : ChromaDB"
    )

    report.append(
        f"Embedding dimension    : "
        f"{EMBEDDING_DIMENSION}"
    )

    report.append("")

    report.append(
        "## INDEXING COUNTS"
    )

    report.append("")

    report.append(
        f"Chunks produced        : "
        f"{expected_count}"
    )

    report.append(
        f"Embeddings generated   : "
        f"{len(records)}"
    )

    report.append(
        f"Records indexed        : "
        f"{indexed_count}"
    )

    report.append(
        f"Failed records         : "
        f"{len(failures)}"
    )

    report.append("")

    report.append(
        "## COUNT VALIDATION"
    )

    report.append("")

    report.append(
        f"Expected records       : "
        f"{expected_count}"
    )

    report.append(
        f"Actual indexed records : "
        f"{indexed_count}"
    )

    report.append(
        f"Validation             : "
        f"{'PASSED' if count_passed else 'FAILED'}"
    )

    report.append("")

    report.append(
        "## SPOT-CHECK"
    )

    report.append("")

    report.append(
        f"Validation             : "
        f"{'PASSED' if spot_result['passed'] else 'FAILED'}"
    )

    if records:

        report.append(
            f"ID                     : "
            f"{records[0]['id']}"
        )

        report.append(
            f"Source                 : "
            f"{records[0]['metadata'].get('source')}"
        )

        report.append(
            f"Chunk Index            : "
            f"{records[0]['metadata'].get('chunk_index')}"
        )

        report.append(
            f"Vector Length          : "
            f"{len(records[0]['embedding'])}"
        )

    report.append("")

    report.append(
        "## FAILURES"
    )

    report.append("")

    if failures:

        for failure in failures:

            report.append(
                str(failure)
            )

    else:

        report.append("None")

    report.append("")

    report.append(
        "## OVERALL RESULT"
    )

    report.append("")

    report.append(
        f"Indexing result : "
        f"{'PASSED' if overall_passed else 'FAILED'}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(
        f"\nSummary saved to "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()