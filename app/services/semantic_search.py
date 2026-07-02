from dataclasses import dataclass

import numpy as np

from app.services.embedding_service import (
    embed_documents,
    embed_query,
)
from app.services.text_chunker import TextChunk


@dataclass(frozen=True)
class SearchResult:
    """A retrieved document chunk and its similarity score."""

    chunk: TextChunk
    score: float


def search_chunks(
    query: str,
    chunks: list[TextChunk],
    top_k: int = 3,
) -> list[SearchResult]:
    """
    Find document chunks that are semantically related to a question.

    Higher scores indicate greater semantic similarity.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    document_embeddings = embed_documents(
        [chunk.text for chunk in chunks]
    )
    query_embedding = embed_query(query)

    scores = document_embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = ranked_indices[: min(top_k, len(chunks))]

    return [
        SearchResult(
            chunk=chunks[index],
            score=float(scores[index]),
        )
        for index in selected_indices
    ]