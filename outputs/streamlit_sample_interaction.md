# Streamlit sample interaction

Run the UI with:

```bash
streamlit run src/streamlit_app.py
```

Environment:

```text
API_BASE_URL=http://localhost:8000
```

Question entered:

> What does property insurance cover?

Grounded answer shown:

> Property insurance covers damage to your home from fire or storm events.

Sources shown:

| Source | Chunk | Relevance |
| --- | ---: | ---: |
| sample.md | 0 | 0.98 |

Loading state shown while waiting:

> Retrieving grounded answer...

Error state shown when the backend is unavailable:

> Could not reach the RAG API: connection refused
