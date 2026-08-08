import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vector_store import create_collection, insert_test_record, read_test_record


def test_insert_and_readback_record(tmp_path):
    collection = create_collection(
        db_path=str(tmp_path / "chroma"),
        collection_name="test_insurance_chunks",
        dimension=3,
    )

    record = insert_test_record(
        collection,
        record_id="test-record-1",
        text="Property insurance protects a home from fire and storm damage.",
        metadata={"source": "sample.md", "chunk_index": 0, "section": "Property Insurance"},
        vector=[0.1, 0.2, 0.3],
    )

    readback = read_test_record(collection, record_id="test-record-1")

    assert record["id"] == "test-record-1"
    assert readback is not None
    assert readback["ids"] == ["test-record-1"]
    assert len(readback["embeddings"][0]) == 3
    assert readback["documents"][0].startswith("Property insurance")
