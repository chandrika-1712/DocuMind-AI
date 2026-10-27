
"""
Short Description:
Retrieve relevant document passages using semantic embeddings.

Approach:
1. Split extracted text into overlapping chunks.
2. Convert each chunk into an embedding.
3. Convert the user's question into an embedding.
4. Rank chunks using cosine similarity.
5. Return the most relevant passages as evidence.
"""

import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_into_chunks(
    text: str,
    chunk_size: int = 120,
    overlap: int = 25,
) -> list[str]:
    """Split text into word-based chunks with overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    words = re.findall(r"\S+", text)

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


class DocumentRetriever:
    """Find document passages semantically related to a question."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = []
        self.embeddings = None

    def index_document(self, text: str):
        """Create searchable embeddings for a document."""

        self.chunks = split_into_chunks(text)

        if not self.chunks:
            raise ValueError("The document contains no searchable text.")

        self.embeddings = self.model.encode(
            self.chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, question: str, top_k: int = 3):
        """Return the most relevant passages and similarity scores."""

        if self.embeddings is None:
            raise ValueError("Index a document before searching.")

        if not question.strip():
            raise ValueError("Please enter a question.")

        question_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = cosine_similarity(
            question_embedding,
            self.embeddings,
        )[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "passage": self.chunks[index],
                "similarity": float(scores[index]),
            }
            for index in ranked_indices
        ]
