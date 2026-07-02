import numpy as np
import pytest

from app.services import embedding_service


class FakeEmbeddingModel:
    """A small fake model used to avoid network access during tests."""

    def encode_document(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, 0.0] for _ in texts],
            dtype=np.float32,
        )

    def encode_query(
        self,
        query: str,
        **kwargs: object,
    ) -> np.ndarray:
        return np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )


def test_embed_documents_returns_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        embedding_service,
        "get_embedding_model",
        lambda: fake_model,
    )

    result = embedding_service.embed_documents(
        ["First document", "Second document"]
    )

    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_embed_query_returns_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        embedding_service,
        "get_embedding_model",
        lambda: fake_model,
    )

    result = embedding_service.embed_query(
        "What does this document discuss?"
    )

    assert result.shape == (3,)
    assert result.dtype == np.float32


def test_empty_document_list_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="At least one document",
    ):
        embedding_service.embed_documents([])


def test_blank_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        embedding_service.embed_query("   ")