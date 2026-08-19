# Streaming and citation sample interaction

Run the UI with:

```bash
streamlit run src/streamlit_app.py
```

Environment:

```text
API_BASE_URL=http://localhost:8000
```

The UI calls `/api/query/stream` and receives newline-delimited events.

Question entered:

> What does property insurance cover?

Progressive answer display:

```text
Property insurance
Property insurance [1] covers fire damage.
Property insurance [1] covers fire damage and storm events.
```

Grounded answer shown after the final `done` event:

> Property insurance covers damage to your home from fire or storm events.

Sources shown:

| Source | Chunk | Relevance |
| --- | ---: | ---: |
| sample.md | 0 | 0.98 |

The `[1] sample.md | chunk 0` panel can be expanded to inspect:

```text
Property insurance covers damage to your home from fire and storms.
```

Loading state shown while waiting:

> Retrieving grounded answer...

Error state shown when the backend is unavailable:

> Could not reach the RAG API: connection refused

Error state shown after an interrupted stream:

> The stream stopped early. The partial answer remains visible above.
