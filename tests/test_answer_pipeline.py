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
    is_question_echo,
    build_explanatory_answer,
    is_uninformative_answer,
    is_explanatory_question,
    remove_leading_question_echo,
    rerank_sentence_candidates,
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


def test_leading_question_echo_is_removed() -> None:
    result = remove_leading_question_echo(
        question="How does a shark stay afloat?",
        context=(
            "How does a shark stay afloat? "
            "The answer appears later in this passage."
        ),
    )

    assert result == (
        "The answer appears later in this passage."
    )


def test_question_sentence_is_detected_as_echo() -> None:
    assert is_question_echo(
        question="How does a shark stay afloat?",
        candidate="How does a shark stay afloat?",
    ) is True


def test_question_word_is_rejected_as_answer() -> None:
    assert is_uninformative_answer(
        question="How does a shark stay afloat?",
        answer="How",
    ) is True


def test_meaningful_answer_is_not_rejected() -> None:
    assert is_uninformative_answer(
        question="How does a shark stay afloat?",
        answer="an oil-filled liver",
    ) is False


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
             retrieval_score=0.72,
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
    assert response.retrieval_score == pytest.approx(0.72)

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


def test_multi_sentence_passage_is_reranked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = create_search_result(
        text=(
            "The application uses Python and FastAPI. "
            "Its web interface uses Streamlit."
        ),
        page_number=4,
        retrieval_score=0.80,
    )

    def fake_search_chunks(
        query: str,
        chunks: list[TextChunk],
        top_k: int,
    ) -> list[SearchResult]:
        assert query == (
            "Which backend framework does the project use?"
        )

        assert len(chunks) == 2
        assert chunks[0].text == (
            "The application uses Python and FastAPI."
        )
        assert chunks[1].text == (
            "Its web interface uses Streamlit."
        )

        return [
            SearchResult(
                chunk=chunks[0],
                score=0.92,
            ),
            SearchResult(
                chunk=chunks[1],
                score=0.41,
            ),
        ]

    monkeypatch.setattr(
        answer_pipeline,
        "rerank_chunks",
        fake_search_chunks,
    )

    reranked = rerank_sentence_candidates(
        question=(
            "Which backend framework does the project use?"
        ),
        results=[result],
        top_k=3,
    )

    assert reranked[0].chunk.text == (
        "The application uses Python and FastAPI."
    )
    assert reranked[0].chunk.page_number == 4
    assert reranked[0].chunk.chunk_id.endswith(
        "-sentence-1"
    )
def test_retrieval_relevance_has_priority_over_qa_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fastapi_result = create_search_result(
        text="The backend framework is FastAPI.",
        page_number=4,
        retrieval_score=0.82,
    )

    streamlit_result = create_search_result(
        text="The web interface uses Streamlit.",
        page_number=4,
        retrieval_score=0.61,
    )

    retrieval = RetrievalResponse(
        query="Which backend framework does the project use?",
        source="evaluation.pdf",
        page_count=4,
        chunk_count=4,
        results=[
            fastapi_result,
            streamlit_result,
        ],
    )

    monkeypatch.setattr(
        answer_pipeline,
        "retrieve_from_pdf",
        lambda **kwargs: retrieval,
    )

    monkeypatch.setattr(
        answer_pipeline,
        "rerank_sentence_candidates",
        lambda **kwargs: [
            fastapi_result,
            streamlit_result,
        ],
    )

    def fake_extract_answer(
        question: str,
        context: str,
    ) -> ExtractedAnswer:
        if "FastAPI" in context:
            return ExtractedAnswer(
                text="FastAPI",
                confidence=0.30,
                start_character=25,
                end_character=32,
            )

        return ExtractedAnswer(
            text="Streamlit",
            confidence=0.95,
            start_character=23,
            end_character=32,
        )

    monkeypatch.setattr(
        answer_pipeline,
        "extract_answer",
        fake_extract_answer,
    )

    response = answer_pipeline.answer_from_pdf(
        file_path=Path("evaluation.pdf"),
        question=(
            "Which backend framework does the project use?"
        ),
    )

    assert response.answered is True
    assert response.answer == "FastAPI"
    assert response.retrieval_score == pytest.approx(0.82)


def test_low_cross_encoder_score_does_not_reject_valid_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_result = create_search_result(
        text="The backend framework is FastAPI.",
        page_number=4,
        retrieval_score=0.82,
    )

    retrieval = RetrievalResponse(
        query="Which backend framework does the project use?",
        source="evaluation.pdf",
        page_count=4,
        chunk_count=4,
        results=[original_result],
    )

    monkeypatch.setattr(
        answer_pipeline,
        "retrieve_from_pdf",
        lambda **kwargs: retrieval,
    )

    # CrossEncoder scores are not directly comparable to
    # embedding cosine-similarity scores.
    reranked_result = SearchResult(
        chunk=original_result.chunk,
        score=0.04,
    )

    monkeypatch.setattr(
        answer_pipeline,
        "rerank_sentence_candidates",
        lambda **kwargs: [reranked_result],
    )

    monkeypatch.setattr(
        answer_pipeline,
        "extract_answer",
        lambda **kwargs: ExtractedAnswer(
            text="FastAPI",
            confidence=0.90,
            start_character=25,
            end_character=32,
        ),
    )

    response = answer_pipeline.answer_from_pdf(
        file_path=Path("evaluation.pdf"),
        question="Which backend framework does the project use?",
    )

    assert response.answered is True
    assert response.answer == "FastAPI"
    assert response.citation is not None
    assert response.citation.page_number == 4


@pytest.mark.parametrize(
    "question",
    [
        "How does a shark stay afloat?",
        "Why do sharks need an oil-filled liver?",
        "Explain how shark buoyancy works.",
        "Describe the adaptations that help sharks float.",
    ],
)
def test_explanatory_questions_are_detected(
    question: str,
) -> None:
    assert is_explanatory_question(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What framework does the project use?",
        "When does the library open?",
        "How many vacation days are provided?",
        "How much oil is required?",
        "Who created the document?",
        "",
        "   ",
    ],
)
def test_factoid_questions_are_not_explanatory(
    question: str,
) -> None:
    assert is_explanatory_question(question) is False


def test_build_explanatory_answer_combines_evidence() -> None:
    results = [
        create_search_result(
            text=(
                "Sharks mainly rely on their large oil-filled "
                "liver to stay buoyant."
            ),
            page_number=1,
            retrieval_score=0.92,
        ),
        create_search_result(
            text=(
                "Their lightweight cartilage also helps "
                "them remain afloat."
            ),
            page_number=1,
            retrieval_score=0.81,
        ),
        create_search_result(
            text=(
                "Their fins and tail help maintain buoyancy "
                "while swimming."
            ),
            page_number=1,
            retrieval_score=0.75,
        ),
    ]

    answer = build_explanatory_answer(
        question="How does a shark stay afloat?",
        candidate_results=results,
        maximum_sentences=3,
    )

    assert "oil-filled liver" in answer
    assert "cartilage" in answer
    assert "fins and tail" in answer


def test_explanatory_answer_rejects_question_echo() -> None:
    results = [
        create_search_result(
            text="How does a shark stay afloat?",
            page_number=1,
            retrieval_score=0.99,
        ),
        create_search_result(
            text=(
                "Sharks mainly rely on their large oil-filled "
                "liver to stay buoyant."
            ),
            page_number=1,
            retrieval_score=0.90,
        ),
    ]

    answer = build_explanatory_answer(
        question="How does a shark stay afloat?",
        candidate_results=results,
    )

    assert answer == (
        "Sharks mainly rely on their large oil-filled "
        "liver to stay buoyant."
    )


def test_explanatory_answer_rejects_short_fragment() -> None:
    results = [
        create_search_result(
            text="in the water",
            page_number=1,
            retrieval_score=0.95,
        ),
        create_search_result(
            text=(
                "Oil in the shark's liver is lighter than "
                "water and supports buoyancy."
            ),
            page_number=1,
            retrieval_score=0.85,
        ),
    ]

    answer = build_explanatory_answer(
        question="How does a shark stay afloat?",
        candidate_results=results,
    )

    assert answer == (
        "Oil in the shark's liver is lighter than "
        "water and supports buoyancy."
    )


def test_explanatory_answer_removes_duplicates() -> None:
    results = [
        create_search_result(
            text="A shark's oil-filled liver supports buoyancy.",
            page_number=1,
            retrieval_score=0.90,
        ),
        create_search_result(
            text="A shark's oil-filled liver supports buoyancy.",
            page_number=1,
            retrieval_score=0.88,
        ),
    ]

    answer = build_explanatory_answer(
        question="How does a shark stay afloat?",
        candidate_results=results,
    )

    assert answer.count("oil-filled liver") == 1


@pytest.mark.parametrize(
    ("maximum_sentences", "minimum_words"),
    [
        (0, 5),
        (-1, 5),
        (3, 0),
        (3, -1),
    ],
)
def test_invalid_explanatory_answer_settings(
    maximum_sentences: int,
    minimum_words: int,
) -> None:
    with pytest.raises(ValueError):
        build_explanatory_answer(
            question="How does this work?",
            candidate_results=[],
            maximum_sentences=maximum_sentences,
            minimum_words=minimum_words,
        )    