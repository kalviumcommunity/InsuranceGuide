# Final end-to-end verification

Release branch: `release/insuranceguide-v1`

## Reproducible flow

With `GEMINI_API_KEY` configured and the backend running:

```bash
curl -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@data/sample.md"

curl -N -X POST http://127.0.0.1:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What does property insurance protect?","k":3}'
```

Expected flow:

```text
upload -> clean -> chunk -> embed -> index
query  -> retrieve -> grounded answer tokens -> [1] citation -> done
```

## Recorded sample result

The existing committed sample demonstrates the same flow after indexing `new_policy.md`:

- Upload: 1 chunk created and 1 chunk indexed.
- Question: `What does the new policy provide protection for?`
- Answer: `The new policy provides flood protection for insured residential properties.`
- Citation: `[1] new_policy.md`, chunk `0`, section `New Policy Coverage`.
- Searchability: `searchable_without_restart: true`.

See `outputs/upload_sample_response.json` and `outputs/runtime_query_result.json`.

## Local checks completed for this release

```text
python -m pytest -q tests/test_api.py tests/test_streaming_api.py
4 passed

python -m py_compile src/api.py src/streamlit_app.py
passed

git diff --check
passed
```

A live Gemini upload/query cannot be executed in an environment without `GEMINI_API_KEY`; the API reports that condition explicitly through `/health` and stream `error` events instead of claiming success.
