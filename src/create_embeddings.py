import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from document_loader import load_documents
from text_cleaning import clean
from chunk_metadata import tag_chunks

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

if not embedding_model:
    raise ValueError("EMBEDDING_MODEL not found in .env")

client = genai.Client(api_key=api_key)

DATA_FOLDER = "data"
OUTPUT_FILE = "outputs/embedding_output.txt"


def generate_embedding(text):
    response = client.models.embed_content(
        model=embedding_model,
        contents=text,
    )
    return response.embeddings[0].values


def main():
    print("=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    documents = load_documents(DATA_FOLDER)

    stored_embeddings = []

    for document in documents:
        cleaned_text = clean(document["text"])

        cleaned_document = {
            "source": document["source"],
            "text": cleaned_text,
        }

        chunks = tag_chunks(cleaned_document)

        for chunk in chunks:
            vector = generate_embedding(chunk["text"])

            stored_embeddings.append(
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "embedding": vector,
                }
            )

    print(f"\nChunks Embedded : {len(stored_embeddings)}")

    if stored_embeddings:
        vector = stored_embeddings[0]["embedding"]

        print(f"Vector Length  : {len(vector)}")
        print(f"Sample Values  : {vector[:8]}")

    report = []

    report.append("EMBEDDING OUTPUT")
    report.append("=" * 70)
    report.append(f"Chunks Embedded : {len(stored_embeddings)}")

    if stored_embeddings:
        report.append(f"Vector Length : {len(stored_embeddings[0]['embedding'])}")

    report.append("\nSample Stored Embeddings\n")

    for item in stored_embeddings[:3]:

        report.append("-" * 70)
        report.append(f"Source : {item['metadata']['source']}")
        report.append(f"Chunk Index : {item['metadata']['chunk_index']}")
        report.append(f"Page : {item['metadata']['page']}")
        report.append(f"Section : {item['metadata']['section']}")
        report.append(f"Text : {item['text'][:120]}")
        report.append(f"Vector Length : {len(item['embedding'])}")
        report.append(f"Vector Sample : {item['embedding'][:8]}")

    Path(OUTPUT_FILE).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"\nOutput saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()