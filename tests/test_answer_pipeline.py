from pathlib import Path

import pytest

from app.services import answer_pipeline
from app.services.answer_extractor import ExtractedAnswer
from app.services.answer_pipeline import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
)
from app.services.retrieval_pipeline import RetrievalResponse
from app.services.semantic_search import SearchResult
from app.services.text_chunker import TextChunk
from app.services.answer_pipeline import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    clean_answer_text,
)


def create_search_result(
    *,
    text: str,
    page_number: int,
    retrieval_score: float,
) -> SearchResult:
    chunk = TextChunk(
        chunk_id=f"lecture.pdf-page-{page_number}-chunk-1",
        text=text,
        source="lecture.pdf",
        page_number=page_number,
        chunk_index=1,
    )

    return SearchResult(
        chunk=chunk,
        score=retrieval_score,
    )


def test_answer_pipeline_selects_best_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [
        create_search_result(
            text="Responsible AI includes fairness.",
            page_number=1,
            retrieval_score=0.72,
        ),
        create_search_result(
            text="Transparency is a principle of responsible AI.",
            page_number=2,
            retrieval_score=0.65,
        ),
    ]

    retrieval = RetrievalResponse(
        query="What is a principle of responsible AI?",
        source="lecture.pdf",
        page_count=2,
        chunk_count=2,
        results=results,
    )

    monkeypatch.setattr(
        answer_pipeline,
        "retrieve_from_pdf",
        lambda **kwargs: retrieval,
    )

    def fake_extract_answer(
        question: str,
        context: str,
    ) -> ExtractedAnswer:
        if "fairness" in context:
            return ExtractedAnswer(
                text="fairness",
                confidence=0.30,
                start_character=32,
                end_character=40,
            )

        return ExtractedAnswer(
            text="Transparency",
            confidence=0.80,
            start_character=0,
            end_character=12,
        )

    monkeypatch.setattr(
        answer_pipeline,
        "extract_answer",
        fake_extract_answer,
    )

    response = answer_pipeline.answer_from_pdf(
        file_path=Path("lecture.pdf"),
        question="What is a principle of responsible AI?",
    )

    assert response.answered is True
    assert response.answer == "Transparency"
    assert response.answer_confidence == pytest.approx(0.80)
    assert response.retrieval_score == pytest.approx(0.65)

    assert response.citation is not None
    assert response.citation.page_number == 2
    assert response.citation.source == "lecture.pdf"


def test_low_retrieval_score_causes_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = RetrievalResponse(
        query="Who won the hockey game?",
        source="lecture.pdf",
        page_count=1,
        chunk_count=1,
        results=[
            create_search_result(
                text="This document discusses machine learning.",
                page_number=1,
                retrieval_score=0.08,
            )
        ],
    )

    monkeypatch.setattr(
        answer_pipeline,
        "retrieve_from_pdf",
        lambda **kwargs: retrieval,
    )

    response = answer_pipeline.answer_from_pdf(
        file_path=Path("lecture.pdf"),
        question="Who won the hockey game?",
    )

    assert response.answered is False
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert response.citation is None


def test_low_answer_confidence_causes_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval = RetrievalResponse(
        query="What is responsible AI?",
        source="lecture.pdf",
        page_count=1,
        chunk_count=1,
        results=[
            create_search_result(
                text="Responsible AI includes fairness.",
                page_number=1,
                retrieval_score=0.80,
            )
        ],
    )

    monkeypatch.setattr(
        answer_pipeline,
        "retrieve_from_pdf",
        lambda **kwargs: retrieval,
    )

    monkeypatch.setattr(
        answer_pipeline,
        "extract_answer",
        lambda **kwargs: ExtractedAnswer(
            text="fairness",
            confidence=0.02,
            start_character=32,
            end_character=40,
        ),
    )

    response = answer_pipeline.answer_from_pdf(
        file_path=Path("lecture.pdf"),
        question="What is responsible AI?",
    )

    assert response.answered is False
    assert response.citation is None


@pytest.mark.parametrize(
    (
        "minimum_retrieval_score",
        "minimum_answer_confidence",
    ),
    [
        (-0.1, 0.1),
        (1.1, 0.1),
        (0.1, -0.1),
        (0.1, 1.1),
    ],
)
def test_invalid_thresholds_are_rejected(
    minimum_retrieval_score: float,
    minimum_answer_confidence: float,
) -> None:
    with pytest.raises(ValueError):
        answer_pipeline.answer_from_pdf(
            file_path=Path("lecture.pdf"),
            question="A valid question?",
            minimum_retrieval_score=(
                minimum_retrieval_score
            ),
            minimum_answer_confidence=(
                minimum_answer_confidence
            ),
        )
def test_clean_answer_text_removes_question_and_label() -> None:
    result = clean_answer_text(
        question="What does 1 + 1 give us?",
        extracted_text="what does 1 + 1 give us? Ans: 2",
    )

    assert result == "2"


def test_clean_answer_text_preserves_normal_answer() -> None:
    result = clean_answer_text(
        question="What is responsible AI?",
        extracted_text="fairness and transparency",
    )

    assert result == "fairness and transparency"