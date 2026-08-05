from pathlib import Path

from document_loader import load_documents
from text_cleaning import clean
from chunk_metadata import tag_chunks

DATA_FOLDER = "data"
OUTPUT_FILE = "outputs/ingestion_summary.txt"


def main():
    print("=" * 70)
    print("RUNNING FULL INGESTION PIPELINE")
    print("=" * 70)

    documents = load_documents(DATA_FOLDER)

    # Count only supported source documents
    supported_extensions = {".txt", ".md", ".pdf"}

    total_source_documents = sum(
        1
        for file in Path(DATA_FOLDER).iterdir()
        if file.is_file() and file.suffix.lower() in supported_extensions
    )

    loaded_documents = []
    failed_documents = []
    all_chunks = []

    print("\nLoading, cleaning and chunking documents...\n")

    for document in documents:
        try:
            cleaned_text = clean(document["text"])

            cleaned_document = {
                "source": document["source"],
                "text": cleaned_text,
            }

            loaded_documents.append(cleaned_document)

            chunks = tag_chunks(cleaned_document)
            all_chunks.extend(chunks)

        except Exception as error:
            print(f"Failed to process {document['source']}: {error}")
            failed_documents.append(document["source"])

    successful = len(loaded_documents)
    failures = len(failed_documents)

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    print(f"Total source documents      : {total_source_documents}")
    print(f"Successfully ingested       : {successful}")
    print(f"Failed documents            : {failures}")
    print(f"Total chunks created        : {len(all_chunks)}")

    print("\nValidation")

    if total_source_documents == successful + failures:
        validation = "PASSED"
        print("PASS - No documents were silently dropped.")
    else:
        validation = "FAILED"
        print("FAIL - Document counts do not match.")

    print("\nSample Chunks")
    print("=" * 70)

    sample_output = []

    for chunk in all_chunks[:3]:
        metadata = chunk["metadata"]
        text = chunk["text"][:120].replace("\n", " ")

        print(f"Source       : {metadata['source']}")
        print(f"Chunk Index  : {metadata['chunk_index']}")
        print(f"Page         : {metadata['page']}")
        print(f"Section      : {metadata['section']}")
        print(f"Text         : {text}")
        print("-" * 70)

        sample_output.append(
            f"""
Source: {metadata['source']}
Chunk Index: {metadata['chunk_index']}
Page: {metadata['page']}
Section: {metadata['section']}
Text:
{text}
----------------------------------------------------------------------
"""
        )

    summary = f"""
FULL INGESTION SUMMARY

Total source documents : {total_source_documents}
Successfully ingested  : {successful}
Failed documents       : {failures}
Total chunks created   : {len(all_chunks)}

Validation : {validation}

Sample Chunks

{''.join(sample_output)}
"""

    Path(OUTPUT_FILE).write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()