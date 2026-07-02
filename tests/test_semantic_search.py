import numpy as np
import pytest

from app.services import semantic_search
from app.services.text_chunker import TextChunk


def create_sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="reading.pdf-page-1-chunk-1",
            text="Machine learning identifies patterns in data.",
            source="reading.pdf",
            page_number=1,
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="reading.pdf-page-2-chunk-1",
            text="Cooking pasta requires boiling water.",
            source="reading.pdf",
            page_number=2,
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="reading.pdf-page-3-chunk-1",
            text="Artificial intelligence can analyze information.",
            source="reading.pdf",
            page_number=3,
            chunk_index=1,
        ),
    ]


def test_search_returns_most_similar_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = create_sample_chunks()

    monkeypatch.setattr(
        semantic_search,
        "embed_documents",
        lambda texts: np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    monkeypatch.setattr(
        semantic_search,
        "embed_query",
        lambda query: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    results = semantic_search.search_chunks(
        query="How can computers learn from data?",
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.page_number == 1
    assert results[1].chunk.page_number == 3
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)


def test_search_preserves_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = create_sample_chunks()

    monkeypatch.setattr(
        semantic_search,
        "embed_documents",
        lambda texts: np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )

    monkeypatch.setattr(
        semantic_search,
        "embed_query",
        lambda query: np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    result = semantic_search.search_chunks(
        query="machine learning",
        chunks=chunks,
        top_k=1,
    )[0]

    assert result.chunk.source == "reading.pdf"
    assert result.chunk.page_number == 1
    assert result.chunk.chunk_id == "reading.pdf-page-1-chunk-1"


def test_top_k_cannot_exceed_available_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = create_sample_chunks()

    monkeypatch.setattr(
        semantic_search,
        "embed_documents",
        lambda texts: np.eye(
            len(texts),
            dtype=np.float32,
        ),
    )

    monkeypatch.setattr(
        semantic_search,
        "embed_query",
        lambda query: np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    results = semantic_search.search_chunks(
        query="test question",
        chunks=chunks,
        top_k=10,
    )

    assert len(results) == 3


def test_empty_chunk_list_returns_no_results() -> None:
    results = semantic_search.search_chunks(
        query="A valid question",
        chunks=[],
    )

    assert results == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        semantic_search.search_chunks(
            query="A valid question",
            chunks=[],
            top_k=top_k,
        )


def test_blank_search_query_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        semantic_search.search_chunks(
            query="   ",
            chunks=[],
        )