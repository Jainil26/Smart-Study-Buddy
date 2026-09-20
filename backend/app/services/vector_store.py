import math
from typing import Any, Dict, List, Optional


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes the cosine similarity between two numerical vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0 (0.0 if invalid or zero vector).
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


class VectorStore:
    """
    Lightweight, in-memory local vector store for document chunks and embeddings.
    """

    def __init__(self):
        self._chunks: List[Dict[str, Any]] = []

    def clear(self):
        """Clears all stored chunks."""
        self._chunks.clear()

    def add_chunk(
        self,
        chunk_id: int,
        page_number: int,
        text: str,
        embedding: List[float]
    ):
        """
        Stores a single document chunk and its embedding vector.
        """
        self._chunks.append({
            "chunk_id": chunk_id,
            "page_number": page_number,
            "text": text,
            "embedding": embedding,
        })

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Stores multiple chunks and their corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        for chunk, embedding in zip(chunks, embeddings):
            self.add_chunk(
                chunk_id=chunk["chunk_id"],
                page_number=chunk["page_number"],
                text=chunk["text"],
                embedding=embedding,
            )

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Returns all stored chunks."""
        return self._chunks

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search between query_embedding and stored chunks.

        Args:
            query_embedding: Vector embedding of the query string.
            top_k: Number of top results to return.

        Returns:
            List of top-k chunk dicts sorted by similarity score in descending order:
                - chunk_id: int
                - page_number: int
                - score: float (rounded to 4 decimals)
                - text: str
        """
        if not self._chunks:
            return []

        if top_k <= 0:
            return []

        scored_results = []
        for item in self._chunks:
            chunk_vector = item["embedding"]
            score = cosine_similarity(query_embedding, chunk_vector)
            scored_results.append({
                "chunk_id": item["chunk_id"],
                "page_number": item["page_number"],
                "score": round(score, 4),
                "text": item["text"],
            })

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return scored_results[:top_k]
