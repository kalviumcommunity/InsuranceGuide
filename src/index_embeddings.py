import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai

from document_loader import load_documents
from text_cleaning import clean
from chunk_metadata import tag_chunks


# ------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not embedding_model:
    raise ValueError("EMBEDDING_MODEL not found in .env")


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_FOLDER = "data"
VECTOR_DB_PATH = "vector_store"
COLLECTION_NAME = "insurance_chunks"
OUTPUT_FILE = "outputs/indexing_summary.txt"

client = genai.Client(api_key=api_key)


# ------------------------------------------------------------
# Generate embedding
# ------------------------------------------------------------

def generate_embedding(text):
    response = client.models.embed_content(
        model=embedding_model,
        contents=text,
    )

    return response.embeddings[0].values


# ------------------------------------------------------------
# Prepare all corpus chunks
# ------------------------------------------------------------

def prepare_chunks():
    documents = load_documents(DATA_FOLDER)

    all_chunks = []

    for document in documents:
        try:
            cleaned_text = clean(document["text"])

            cleaned_document = {
                "source": document["source"],
                "text": cleaned_text,
            }

            chunks = tag_chunks(cleaned_document)
            all_chunks.extend(chunks)

        except Exception as error:
            print(
                f"Failed to process {document['source']}: {error}"
            )

    return all_chunks


# ------------------------------------------------------------
# Convert metadata into Chroma-compatible values
# ------------------------------------------------------------

def prepare_metadata(metadata):
    return {
        "source": str(metadata.get("source", "")),
        "chunk_index": int(metadata.get("chunk_index", 0)),
        "char_start": int(metadata.get("char_start", 0)),
        "page": (
            int(metadata["page"])
            if metadata.get("page") is not None
            else -1
        ),
        "section": str(metadata.get("section") or ""),
    }


# ------------------------------------------------------------
# Main indexing process
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("INDEXING EMBEDDINGS INTO CHROMADB")
    print("=" * 70)

    print("\nLoading and preparing corpus...")

    chunks = prepare_chunks()

    print(f"Chunks produced : {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were produced from the corpus.")

    # --------------------------------------------------------
    # Create persistent ChromaDB client
    # --------------------------------------------------------

    print("\nConnecting to ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Clear the collection before indexing.
    # This makes the indexing run reproducible and prevents
    # duplicate records when the script is run again.
    existing_count = collection.count()

    if existing_count > 0:
        print(
            f"Existing records found : {existing_count}"
        )
        print("Clearing collection before re-indexing...")
        collection.delete(
            where={}
        )

    # --------------------------------------------------------
    # Generate embeddings and prepare records
    # --------------------------------------------------------

    print("\nGenerating embeddings and preparing records...")

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    failures = []

    for index, chunk in enumerate(chunks):

        metadata = chunk["metadata"]

        source = metadata["source"]
        chunk_index = metadata["chunk_index"]

        record_id = f"{source}::chunk_{chunk_index}"

        try:
            vector = generate_embedding(chunk["text"])

            ids.append(record_id)
            embeddings.append(vector)
            documents.append(chunk["text"])
            metadatas.append(
                prepare_metadata(metadata)
            )

        except Exception as error:

            print(
                f"Failed to embed {record_id}: {error}"
            )

            failures.append(
                {
                    "id": record_id,
                    "source": source,
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # Insert records into ChromaDB
    # --------------------------------------------------------

    print("\nInserting records into ChromaDB...")

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    indexed_count = collection.count()

    print(f"Records indexed : {indexed_count}")

    # --------------------------------------------------------
    # Validate indexed count
    # --------------------------------------------------------

    print("\nCount Validation")

    expected_count = len(chunks)
    successful_count = len(ids)

    if (
        indexed_count == expected_count
        and successful_count == expected_count
        and len(failures) == 0
    ):
        count_validation = "PASSED"
        print(
            "PASS - Indexed count matches the number of chunks."
        )
    else:
        count_validation = "FAILED"
        print(
            "FAIL - Indexed count does not match the corpus."
        )

    # --------------------------------------------------------
    # Spot-check stored record
    # --------------------------------------------------------

    print("\nSpot-check Stored Record")
    print("=" * 70)

    spot_check_status = "FAILED"
    spot_check_output = []

    if indexed_count > 0:

        result = collection.get(
            ids=[ids[0]],
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )

        stored_id = result["ids"][0]
        stored_text = result["documents"][0]
        stored_metadata = result["metadatas"][0]
        stored_vector = result["embeddings"][0]

        source_chunk = chunks[0]

        text_matches = (
            stored_text == source_chunk["text"]
        )

        source_matches = (
            stored_metadata["source"]
            == source_chunk["metadata"]["source"]
        )

        chunk_index_matches = (
            stored_metadata["chunk_index"]
            == source_chunk["metadata"]["chunk_index"]
        )

        vector_length = len(stored_vector)

        vector_length_matches = (
            vector_length == len(embeddings[0])
        )

        spot_check_passed = (
            text_matches
            and source_matches
            and chunk_index_matches
            and vector_length_matches
        )

        if spot_check_passed:
            spot_check_status = "PASSED"
            print("PASS - Stored record matches source chunk.")
        else:
            spot_check_status = "FAILED"
            print("FAIL - Stored record does not fully match.")

        print(f"ID             : {stored_id}")
        print(f"Source         : {stored_metadata['source']}")
        print(
            f"Chunk Index    : "
            f"{stored_metadata['chunk_index']}"
        )
        print(f"Page           : {stored_metadata['page']}")
        print(
            f"Section        : "
            f"{stored_metadata['section']}"
        )
        print(f"Vector Length  : {vector_length}")
        print(
            f"Text           : "
            f"{stored_text[:120]}"
        )
        print(
            f"Vector Sample  : "
            f"{stored_vector[:8]}"
        )

        spot_check_output = [
            f"ID             : {stored_id}",
            f"Source         : {stored_metadata['source']}",
            (
                f"Chunk Index    : "
                f"{stored_metadata['chunk_index']}"
            ),
            f"Page           : {stored_metadata['page']}",
            (
                f"Section        : "
                f"{stored_metadata['section']}"
            ),
            f"Vector Length  : {vector_length}",
            (
                f"Text           : "
                f"{stored_text[:120]}"
            ),
            (
                f"Vector Sample  : "
                f"{stored_vector[:8]}"
            ),
            "",
            f"Text matches source chunk : {text_matches}",
            f"Source matches            : {source_matches}",
            (
                f"Chunk index matches       : "
                f"{chunk_index_matches}"
            ),
            (
                f"Vector length matches     : "
                f"{vector_length_matches}"
            ),
        ]

    else:
        print("No records available for spot-check.")

    # --------------------------------------------------------
    # Write indexing summary
    # --------------------------------------------------------

    report = []

    report.append("VECTOR DATABASE INDEXING SUMMARY")
    report.append("=" * 70)

    report.append(
        f"Embedding model       : {embedding_model}"
    )
    report.append(
        f"Collection             : {COLLECTION_NAME}"
    )
    report.append(
        f"Vector database        : ChromaDB"
    )
    report.append(
        f"Database path          : {VECTOR_DB_PATH}"
    )

    report.append("")
    report.append("INDEXING COUNTS")
    report.append("-" * 70)

    report.append(
        f"Chunks produced        : {expected_count}"
    )
    report.append(
        f"Embeddings generated   : {successful_count}"
    )
    report.append(
        f"Records indexed        : {indexed_count}"
    )
    report.append(
        f"Failed records         : {len(failures)}"
    )

    report.append("")
    report.append("COUNT VALIDATION")
    report.append("-" * 70)

    report.append(
        f"Expected records       : {expected_count}"
    )
    report.append(
        f"Actual indexed records : {indexed_count}"
    )
    report.append(
        f"Validation             : {count_validation}"
    )

    report.append("")
    report.append("SPOT-CHECK")
    report.append("-" * 70)

    report.append(
        f"Validation             : {spot_check_status}"
    )

    report.extend(spot_check_output)

    report.append("")
    report.append("FAILURES")
    report.append("-" * 70)

    if failures:
        for failure in failures:
            report.append(
                f"ID     : {failure['id']}"
            )
            report.append(
                f"Source : {failure['source']}"
            )
            report.append(
                f"Error  : {failure['error']}"
            )
            report.append("")
    else:
        report.append("None")

    report.append("")
    report.append("OVERALL RESULT")
    report.append("-" * 70)

    if (
        count_validation == "PASSED"
        and spot_check_status == "PASSED"
        and len(failures) == 0
    ):
        overall_result = "PASSED"
    else:
        overall_result = "FAILED"

    report.append(
        f"Indexing result : {overall_result}"
    )

    Path(OUTPUT_FILE).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(OUTPUT_FILE).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    print(
        f"Chunks produced : {expected_count}"
    )
    print(
        f"Records indexed : {indexed_count}"
    )
    print(
        f"Failures        : {len(failures)}"
    )
    print(
        f"Count validation: {count_validation}"
    )
    print(
        f"Spot-check      : {spot_check_status}"
    )
    print(
        f"Overall result  : {overall_result}"
    )

    print(
        f"\nSummary saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()