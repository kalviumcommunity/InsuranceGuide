from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from rag_pipeline import answer_query
from document_loader import load_documents
from text_cleaning import clean
from chunk_metadata import tag_chunks
from index_embeddings import generate_embedding
from vector_store import create_collection

load_dotenv()

APP_TITLE = os.getenv("API_TITLE", "InsuranceGuide RAG API")
APP_VERSION = os.getenv("API_VERSION", "1.0.0")
APP_HOST = os.getenv("API_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("API_PORT", "8000"))

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_TITLE,
        "version": APP_VERSION,
    }


# ============================================================
# QUERY
# ============================================================

@app.post("/api/query")
async def query_endpoint(request: Request) -> dict[str, Any]:
    """Run the existing RAG query pipeline."""

    try:
        payload = await request.json()

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Request body must be valid JSON.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object.",
        )

    question = payload.get("question")

    if question is None or not str(question).strip():
        raise HTTPException(
            status_code=400,
            detail="Question is required.",
        )

    question = str(question).strip()

    k = payload.get("k", 3)

    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > 10:
        raise HTTPException(
            status_code=400,
            detail="k must be an integer between 1 and 10.",
        )

    try:
        result = answer_query(question, k=k)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Server error while processing the question.",
        ) from exc

    if not result or not result.get("answer"):
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline did not return an answer.",
        )

    sources = result.get("sources", []) or []
    context = result.get("context", "") or ""

    return {
        "status": "success",
        "answer": result["answer"],
        "sources": sources,
        "metadata": {
            "question": question,
            "top_k": k,
            "retrieved_chunks": len(sources),
            "context_length": len(context),
        },
    }


# ============================================================
# RUNTIME DOCUMENT UPLOAD
# ============================================================

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...)
) -> dict[str, Any]:
    """
    Upload a document, ingest it, generate embeddings,
    and index its chunks into ChromaDB.

    Supported formats:
    .txt, .md, .pdf
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file format: {extension or 'unknown'}. "
                f"Supported formats are: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # --------------------------------------------------------
    # Prepare upload directory
    # --------------------------------------------------------

    UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = UPLOAD_FOLDER / filename

    # --------------------------------------------------------
    # Save file safely and enforce size limit
    # --------------------------------------------------------

    total_size = 0

    try:
        with destination.open("wb") as buffer:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    destination.unlink(missing_ok=True)

                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum size is 5 MB.",
                    )

                buffer.write(chunk)

    except HTTPException:
        raise

    except Exception as exc:

        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc

    # --------------------------------------------------------
    # Reject empty files
    # --------------------------------------------------------

    if total_size == 0:

        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Load uploaded document
    # --------------------------------------------------------

    try:

        documents = load_documents(
            str(UPLOAD_FOLDER)
        )

        uploaded_documents = [
            document
            for document in documents
            if document["source"] == filename
        ]

        if not uploaded_documents:

            raise ValueError(
                "Uploaded document could not be loaded."
            )

        document = uploaded_documents[0]

        # ----------------------------------------------------
        # Clean document
        # ----------------------------------------------------

        cleaned_text = clean(
            document["text"]
        )

        if not cleaned_text.strip():

            raise ValueError(
                "Uploaded document contains no usable text."
            )

        cleaned_document = {
            "source": filename,
            "text": cleaned_text,
        }

        # ----------------------------------------------------
        # Chunk document
        # ----------------------------------------------------

        chunks = tag_chunks(
            cleaned_document
        )

        if not chunks:

            raise ValueError(
                "No chunks were created from the uploaded document."
            )

        # ----------------------------------------------------
        # Connect to vector database
        # ----------------------------------------------------

        collection = create_collection()

        indexed_chunks = 0

        # ----------------------------------------------------
        # Embed + index every chunk
        # ----------------------------------------------------

        for chunk in chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            source = metadata.get(
                "source",
                filename,
            )

            chunk_index = metadata.get(
                "chunk_index",
                0,
            )

            record_id = (
                f"{source}::chunk_{chunk_index}"
            )

            vector = generate_embedding(
                chunk["text"]
            )

            collection.upsert(
                ids=[record_id],
                embeddings=[vector],
                documents=[chunk["text"]],
                metadatas=[
                    {
                        "source": source,
                        "chunk_index": chunk_index,
                        "page": metadata.get(
                            "page",
                            -1,
                        ),
                        "section": metadata.get(
                            "section",
                            "",
                        ),
                    }
                ],
            )

            indexed_chunks += 1

        # ----------------------------------------------------
        # Return indexing result
        # ----------------------------------------------------

        return {
            "status": "success",
            "message": "Document uploaded and indexed successfully.",
            "file": filename,
            "size_bytes": total_size,
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_chunks,
            "searchable_without_restart": True,
        }

    except ValueError as exc:

        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Document processing or indexing failed.",
        ) from exc


# ============================================================
# ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def exception_handler(
    request: Request,
    exc: Exception,
):

    if isinstance(exc, HTTPException):

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status": "error",
                "detail": exc.detail,
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "Internal server error.",
        },
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
    )