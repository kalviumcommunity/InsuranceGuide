# What embedding vectors represent

An embedding vector is a numeric representation of a text's **meaning**, not a random ID and not a count of keywords.

A model maps each text to a point in a high-dimensional vector space such that:

- Texts with similar meaning land **near** each other, even if they share no words (e.g. "reset my password" and "recover my login access").
- Texts on unrelated topics land **far apart**.

Because of this, the similarity between two vectors is a proxy for the similarity of their meaning. That's what lets a retrieval system match a user's question to the document chunk that actually answers it, instead of requiring the question to use the document's exact wording.

See `src/embeddings_demo.py` and `outputs/embeddings_demo_output.txt` for a concrete comparison: a similar pair (fire damage vs. flame damage) scores higher on cosine similarity than a dissimilar pair (fire damage vs. cafeteria menu).
