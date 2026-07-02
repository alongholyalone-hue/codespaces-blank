from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.services import retrieval_pipeline
from app.services.semantic_search import SearchResult


def create_sample_pdf(file_path: Path) -> None:
    """Create a two-page text PDF for testing."""

    pdf = canvas.Canvas(str(file_path))

    pdf.drawString(
        72,
        720,
        "Machine learning systems identify patterns in training data.",
    )
    pdf.showPage()

    pdf.drawString(
        72,
        720,
        "Responsible artificial intelligence considers fairness.",
    )
    pdf.save()


def create_blank_pdf(file_path: Path) -> None:
    """Create a one-page PDF without extractable text."""

    pdf = canvas.Canvas(str(file_path))
    pdf.showPage()
    pdf.save()


def test_retrieve_from_pdf_connects_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_pdf = tmp_path / "sample.pdf"
    create_sample_pdf(sample_pdf)

    def fake_search_chunks(
        query: str,
        chunks: list,
        top_k: int,
    ) -> list[SearchResult]:
        assert query == "What is responsible AI?"
        assert len(chunks) == 2
        assert top_k == 1

        return [
            SearchResult(
                chunk=chunks[1],
                score=0.91,
            )
        ]

    monkeypatch.setattr(
        retrieval_pipeline,
        "search_chunks",
        fake_search_chunks,
    )

    response = retrieval_pipeline.retrieve_from_pdf(
        file_path=sample_pdf,
        query="What is responsible AI?",
        top_k=1,
        chunk_size=100,
        overlap=20,
    )

    assert response.source == "sample.pdf"
    assert response.page_count == 2
    assert response.chunk_count == 2
    assert len(response.results) == 1

    result = response.results[0]

    assert result.chunk.page_number == 2
    assert result.chunk.source == "sample.pdf"
    assert result.score == pytest.approx(0.91)


def test_blank_pdf_returns_no_results(tmp_path: Path) -> None:
    blank_pdf = tmp_path / "blank.pdf"
    create_blank_pdf(blank_pdf)

    response = retrieval_pipeline.retrieve_from_pdf(
        file_path=blank_pdf,
        query="What does this PDF discuss?",
    )

    assert response.page_count == 1
    assert response.chunk_count == 0
    assert response.results == []


def test_blank_query_is_rejected(tmp_path: Path) -> None:
    sample_pdf = tmp_path / "sample.pdf"
    create_sample_pdf(sample_pdf)

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        retrieval_pipeline.retrieve_from_pdf(
            file_path=sample_pdf,
            query="   ",
        )