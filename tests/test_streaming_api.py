import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

import api


def test_streaming_query_sends_sources_tokens_and_done(monkeypatch):
    monkeypatch.setattr(
        api,
        "stream_answer_query",
        lambda question, k=3: iter([
            {"type": "sources", "sources": [{"marker": "[1]", "source": "sample.md", "chunk_index": 0, "text": "Fire damage."}]},
            {"type": "token", "text": "Property insurance [1]"},
            {"type": "token", "text": " covers fire damage."},
            {"type": "done"},
        ]),
    )

    response = TestClient(api.app).post(
        "/api/query/stream",
        json={"question": "What does property insurance cover?"},
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["marker"] == "[1]"
    assert "[1]" in "".join(event.get("text", "") for event in events)
    assert events[-1]["type"] == "done"


def test_streaming_query_reports_interruption(monkeypatch):
    def interrupted_stream(question, k=3):
        yield {"type": "sources", "sources": []}
        raise RuntimeError("provider disconnected")

    monkeypatch.setattr(api, "stream_answer_query", interrupted_stream)
    response = TestClient(api.app).post("/api/query/stream", json={"question": "query"})

    assert response.status_code == 200
    assert json.loads(response.text.splitlines()[-1])["type"] == "error"
