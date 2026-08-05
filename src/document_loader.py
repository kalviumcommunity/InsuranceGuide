import os
from pathlib import Path
from PyPDF2 import PdfReader

DATA_FOLDER = "data"


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def load_pdf(file_path):
    reader = PdfReader(file_path)

    # Pages are joined with a form-feed ("\f"), the standard page-break
    # marker, so page numbers can still be recovered later for metadata
    # (see src/chunk_metadata.py) without changing the return type.
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\f".join(pages)


def load_documents(folder):
    documents = []

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        # Skip directories (e.g., raw/, cleaned/)
        if os.path.isdir(file_path):
            continue

        try:
            extension = Path(file_name).suffix.lower()

            if extension == ".txt":
                text = load_txt(file_path)

            elif extension == ".md":
                text = load_md(file_path)

            elif extension == ".pdf":
                text = load_pdf(file_path)

            else:
                print(f"Skipping unsupported file: {file_name}")
                continue

            documents.append(
                {
                    "source": file_name,
                    "text": text,
                }
            )

        except Exception as error:
            print(f"Could not load {file_name}: {error}")

    return documents


def preview_documents(documents):
    print("\nDOCUMENT SUMMARY\n")

    for doc in documents:
        sample = doc["text"][:100].replace("\n", " ")

        print("=" * 60)
        print(f"Source : {doc['source']}")
        print(f"Length : {len(doc['text'])} characters")
        print(f"Sample : {sample}")
        print("=" * 60)
        print()  # Blank line between documents


if __name__ == "__main__":
    docs = load_documents(DATA_FOLDER)
    preview_documents(docs)