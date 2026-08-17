from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from rag_pipeline import answer_query

load_dotenv()

APP_TITLE = os.getenv("API_TITLE", "InsuranceGuide RAG API")
APP_VERSION = os.getenv("API_VERSION", "1.0.0")
APP_HOST = os.getenv("API_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("API_PORT", "8000"))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check for load balancers and frontend uptime probes."""
    return {
        "status": "ok",
        "service": APP_TITLE,
        "version": APP_VERSION,
    }


@app.post("/api/query")
async def query_endpoint(request: Request) -> dict[str, Any]:
    """Accept a user question, run the RAG pipeline, and return grounded JSON."""
    try:
        payload = await request.json()
    except Exception as exc:  # pragma: no cover - defensive guard for malformed JSON
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    question = payload.get("question")
    if question is None or not str(question).strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    question = str(question).strip()
    k = payload.get("k", 3)

    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or k > 10:
        raise HTTPException(status_code=400, detail="k must be an integer between 1 and 10.")

    try:
        result = answer_query(question, k=k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for runtime failures
        raise HTTPException(status_code=500, detail="Server error while processing the question.") from exc

    if not result or not result.get("answer"):
        raise HTTPException(status_code=500, detail="The RAG pipeline did not return an answer.")

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


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "detail": exc.detail},
        )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=APP_HOST, port=APP_PORT, reload=False)
