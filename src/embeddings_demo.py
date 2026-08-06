"""Generates embeddings for sample texts, reports their vector dimension,
and compares a similar pair against a dissimilar pair using cosine
similarity, to demonstrate that embedding vectors capture meaning.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from numpy import array, dot
from numpy.linalg import norm

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("EMBEDDING_MODEL") or "text-embedding-004"

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "embeddings_demo_output.txt"

# Index 0 and 1 describe the same situation in different words (similar pair).
# Index 2 is on an unrelated topic (dissimilar pair, compared against index 0).
TEXTS = [
    "What does my policy cover if my house is damaged by fire?",
    "Am I protected against damage to my home caused by flames?",
    "The office cafeteria serves lunch at noon.",
]

EXPLANATION = (
    "An embedding vector is a numeric representation of a text's meaning, not a "
    "random ID or a count of keywords. A model maps each text to a point in vector "
    "space such that texts with similar meaning land near each other and texts on "
    "unrelated topics land far apart, even when they share no words. Similarity "
    "between two vectors is therefore a proxy for similarity of meaning, which is "
    "what lets retrieval match a question to a chunk that answers it without "
    "requiring identical wording."
)


def embed(texts):
    response = client.models.embed_content(model=model_name, contents=texts)
    return [embedding.values for embedding in response.embeddings]


def cosine(a, b):
    a, b = array(a), array(b)
    return float(dot(a, b) / (norm(a) * norm(b)))


def main():
    embeddings = embed(TEXTS)

    dimensions = [len(vector) for vector in embeddings]
    assert len(set(dimensions)) == 1, "Embeddings do not all share the same dimension"
    dimension = dimensions[0]

    similar_score = cosine(embeddings[0], embeddings[1])
    dissimilar_score = cosine(embeddings[0], embeddings[2])

    lines = ["EMBEDDINGS FUNDAMENTALS DEMO", "=" * 70]

    for text, vector in zip(TEXTS, embeddings):
        lines.append(f"\nText: {text}")
        lines.append(f"Dimension: {len(vector)}")
        lines.append(f"First 8 values: {vector[:8]}")

    lines.append("\n" + "=" * 70)
    lines.append(f"All {len(embeddings)} embeddings share dimension: {dimension}")

    lines.append("\nSimilarity comparison (cosine)")
    lines.append(f"Similar pair    (fire damage vs. flame damage): {similar_score:.4f}")
    lines.append(f"Dissimilar pair (fire damage vs. cafeteria):    {dissimilar_score:.4f}")
    lines.append(f"Similar pair scores higher: {similar_score > dissimilar_score}")

    lines.append("\nWhat these vectors represent")
    lines.append(EXPLANATION)

    report = "\n".join(lines)
    print(report)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nFull report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
