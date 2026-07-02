from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

EmbeddingVector = NDArray[np.float32]
EmbeddingMatrix = NDArray[np.float32]


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""

    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str]) -> EmbeddingMatrix:
    """Convert document passages into normalized embedding vectors."""

    if not texts:
        raise ValueError("At least one document is required")

    cleaned_texts = [text.strip() for text in texts]

    if any(not text for text in cleaned_texts):
        raise ValueError("Documents cannot contain empty text")

    model = get_embedding_model()

    embeddings = model.encode_document(
        cleaned_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str) -> EmbeddingVector:
    """Convert a search question into a normalized embedding vector."""

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty")

    model = get_embedding_model()

    embedding = model.encode_query(
        cleaned_query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(embedding, dtype=np.float32)