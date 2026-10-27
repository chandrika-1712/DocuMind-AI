
"""
Short Description:
Retrieve relevant document passages using Sentence Transformers and FAISS.

Approach:
1. Split document text into overlapping chunks.
2. Convert chunks into normalized sentence embeddings.
3. Store embeddings in a FAISS vector index.
4. Convert the user's question into an embedding.
5. Retrieve the most similar passages using FAISS.
6. Return passages with similarity scores for RAG.
"""

import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_into_chunks(
    text: str,
    chunk_size: int = 120,
    overlap: int = 25,
) -> list[str]:
    """Split text into overlapping word-based chunks."""

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
        chunk = words[start:start + chunk_size]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return chunks


class DocumentRetriever:
    """Index document passages and retrieve them with FAISS."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = []
        self.index = None

    def index_document(self, text: str):
        """Create a FAISS index for a document's passages."""

        self.chunks = split_into_chunks(text)

        if not self.chunks:
            self.index = None
            raise ValueError(
                "The document contains no searchable text."
            )

        embeddings = self.model.encode(
            self.chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # FAISS expects a contiguous float32 matrix.
        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        # Normalized vectors + inner product = cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        """Return the most similar passages and their scores."""

        if self.index is None:
            raise ValueError("Index a document before searching.")

        if not question.strip():
            raise ValueError("Please enter a question.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        question_embedding = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        question_embedding = np.ascontiguousarray(
            question_embedding,
            dtype=np.float32,
        )

        k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            question_embedding,
            k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append(
                {
                    "passage": self.chunks[int(index)],
                    "similarity": float(score),
                }
            )

        return results
