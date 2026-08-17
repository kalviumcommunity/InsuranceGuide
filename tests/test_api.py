import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

import api


def test_query_endpoint_returns_structured_json(monkeypatch):
    def fake_answer_query(question, k=3):
        assert question == "What does property insurance cover?"
        assert k == 3
        return {
            "answer": "Property insurance covers damage to your home from fire or storm events.",
            "sources": [
                {
                    "source": "sample.md",
                    "chunk_index": 0,
                    "section": "Property Insurance",
                    "score": 0.98,
                }
            ],
            "chunks": [{"text": "Property insurance covers damage from fire and storms.", "metadata": {"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"}, "score": 0.98}],
            "context": "Property insurance covers damage from fire and storms.",
        }

    monkeypatch.setattr(api, "answer_query", fake_answer_query)
    client = TestClient(api.app)

    response = client.post("/api/query", json={"question": "What does property insurance cover?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["answer"]
    assert payload["sources"][0]["source"] == "sample.md"
    assert payload["metadata"]["retrieved_chunks"] == 1


def test_query_endpoint_rejects_missing_question():
    client = TestClient(api.app)
    response = client.post("/api/query", json={})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required."
