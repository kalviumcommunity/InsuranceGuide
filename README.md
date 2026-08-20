# InsuranceGuide

InsuranceGuide is a grounded insurance-policy assistant. It cleans uploaded documents, splits them into source-traceable chunks, stores Gemini embeddings in a local Chroma database, retrieves relevant passages, and streams an answer with citations.

## Architecture

```text
PDF upload -> PyPDF2 extraction -> text cleaning -> page-aware chunks
					 -> Gemini embeddings -> ChromaDB

Question -> Gemini query embedding -> Chroma retrieval -> grounded Gemini answer
					-> FastAPI JSON/NDJSON -> Streamlit UI with citations
```

## Requirements

- Python 3.12+
- A Gemini API key with access to the configured embedding and chat models
- Windows PowerShell, Git Bash, macOS, or Linux

The vector database is local and requires no separate database server. It is created under `outputs/chroma_local` by default.

## Setup

```bash
git clone <repository-url>
cd InsuranceGuide
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Git Bash
source .venv/Scripts/activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and set the secret locally. Do not commit `.env`.

```text
GEMINI_API_KEY=replace-with-your-key
CHAT_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=gemini-embedding-001
API_BASE_URL=http://127.0.0.1:8000
API_HOST=127.0.0.1
API_PORT=8000
VECTOR_DB_PATH=outputs/chroma_local
VECTOR_COLLECTION=insurance_chunks
EMBEDDING_DIMENSION=3072
MAX_FILE_SIZE_BYTES=52428800
```

Important variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Gemini authentication; required for embedding and answers | none |
| `CHAT_MODEL` | Grounded answer model | `gemini-2.0-flash` |
| `EMBEDDING_MODEL` | Query/document embedding model | `gemini-embedding-001` |
| `API_BASE_URL` | Streamlit URL for FastAPI | `http://localhost:8000` |
| `VECTOR_DB_PATH` | Persistent local Chroma path | `outputs/chroma_local` |
| `VECTOR_COLLECTION` | Chroma collection name | `insurance_chunks` |
| `MAX_FILE_SIZE_BYTES` | Upload limit | `52428800` |

The API health response reports model names and whether a key is configured, but never exposes the key itself.

## Run the application

Start the backend from the repository root:

```bash
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same environment and start the frontend:

```bash
streamlit run src/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>. The API route guide is available at <http://127.0.0.1:8000/api>, and health is available at <http://127.0.0.1:8000/health>.

## Use the main features

1. Open the Streamlit URL.
2. Upload a text-readable PDF in the sidebar and select **Process policy**.
3. The backend extracts text, cleans repeated boilerplate and encoding artifacts, creates page-aware chunks, generates embeddings, and indexes them without a restart.
4. Enter a policy question and select **Ask the policy**.
5. The answer streams progressively. Citation markers such as `[1]` point to the source panels, where the document name, page/chunk metadata, relevance score, and retrieved text can be inspected.

The ingestion path also accepts `.txt` and `.md` through the API. Password-protected, image-only, or scanned PDFs need OCR/text extraction before upload; PyPDF2 cannot read text that is only present as pixels.

## API examples

Upload a policy:

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
	-F "file=@data/sample.md"
```

Ask for a complete JSON answer:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
	-H "Content-Type: application/json" \
	-d '{"question":"What is covered after a fire?","k":3}'
```

Ask for streaming newline-delimited JSON events:

```bash
curl -N -X POST http://127.0.0.1:8000/api/query/stream \
	-H "Content-Type: application/json" \
	-d '{"question":"What is covered after a fire?","k":3}'
```

The stream emits `sources`, `token`, `done`, or `error` events. A missing key or unavailable index is returned as an actionable error event.

## Verification

Run the focused API and full test suite:

```bash
python -m pytest -q tests/test_api.py tests/test_streaming_api.py
python -m pytest -q
```

The committed [final end-to-end verification](outputs/final_e2e_verification.md) records the upload, grounded answer, citation, and local checks. A real upload/query run requires `GEMINI_API_KEY` to be present in the server environment.

## Troubleshooting

- `detail: Not Found`: use the documented paths and methods. Queries are `POST /api/query` or `POST /api/query/stream`; `GET /api` lists routes.
- `GEMINI_API_KEY is not configured`: set it in `.env` or the deployment secret store, then restart Uvicorn.
- `The policy index is unavailable`: process a policy PDF first, or point `VECTOR_DB_PATH` to the indexed Chroma directory.
- Empty PDF content: the PDF is likely scanned/image-only. OCR it first, then upload the text-readable PDF.
- Frontend cannot connect: confirm the API is running and that `API_BASE_URL` matches its host and port.

## Repository safety

Secrets belong in `.env` or deployment settings. Generated vector databases, uploads, caches, and runtime artifacts should remain outside commits unless they are intentionally documented samples.
\# InsuranceGuide



InsuranceGuide is a Retrieval-Augmented Generation (RAG) based AI assistant for answering insurance policy-related questions using grounded document retrieval.



\## Project Setup



\### 1. Clone the repository



```bash

git clone <repository-url>

cd InsuranceGuide

