import re

from dataclasses import dataclass
from pathlib import Path

from app.services.answer_extractor import extract_answer
from app.services.retrieval_pipeline import retrieve_from_pdf


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The uploaded document does not provide enough evidence "
    "to answer this question."
)


def clean_answer_text(
    question: str,
    extracted_text: str,
) -> str:
    """
    Remove a repeated question and common answer labels from
    an extracted answer.
    """

    cleaned = extracted_text.strip()
    question_pattern = re.escape(question.strip())

    cleaned = re.sub(
        rf"^{question_pattern}\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"^(?:answer|ans)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()


@dataclass(frozen=True)
class AnswerCitation:
    """Source information supporting an extracted answer."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    retrieval_score: float


@dataclass(frozen=True)
class DocumentAnswer:
    """A direct answer and its supporting evidence."""

    query: str
    source: str
    page_count: int
    chunk_count: int
    answered: bool
    answer: str
    answer_confidence: float
    retrieval_score: float
    citation: AnswerCitation | None


def answer_from_pdf(
    file_path: str | Path,
    question: str,
    top_k: int = 3,
    minimum_retrieval_score: float = 0.25,
    minimum_answer_confidence: float = 0.10,
) -> DocumentAnswer:
    """
    Retrieve relevant PDF passages and extract the best answer.

    A result is accepted only when both its retrieval score and
    answer-extraction confidence satisfy the configured thresholds.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty")

    if not 0.0 <= minimum_retrieval_score <= 1.0:
        raise ValueError(
            "minimum_retrieval_score must be between 0 and 1"
        )

    if not 0.0 <= minimum_answer_confidence <= 1.0:
        raise ValueError(
            "minimum_answer_confidence must be between 0 and 1"
        )

    retrieval = retrieve_from_pdf(
        file_path=file_path,
        query=cleaned_question,
        top_k=top_k,
    )

    best_result = None
    best_extracted_answer = None
    best_answer_text: str | None = None
    best_combined_score = -1.0

    for result in retrieval.results:
        # Do not run the larger QA model on clearly irrelevant chunks.
        if result.score < minimum_retrieval_score:
            continue

        extracted = extract_answer(
            question=cleaned_question,
            context=result.chunk.text,
        )

        if not extracted.text:
            continue

        cleaned_answer = clean_answer_text(
            question=cleaned_question,
            extracted_text=extracted.text,
        )

        if not cleaned_answer:
            continue

        if extracted.confidence < minimum_answer_confidence:
            continue

        combined_score = (
            max(result.score, 0.0) * extracted.confidence
        )

        if combined_score > best_combined_score:
            best_result = result
            best_extracted_answer = extracted
            best_answer_text = cleaned_answer
            best_combined_score = combined_score

    if (
        best_result is None
        or best_extracted_answer is None
        or best_answer_text is None
    ):
        return DocumentAnswer(
            query=retrieval.query,
            source=retrieval.source,
            page_count=retrieval.page_count,
            chunk_count=retrieval.chunk_count,
            answered=False,
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            answer_confidence=0.0,
            retrieval_score=0.0,
            citation=None,
        )

    citation = AnswerCitation(
        text=best_result.chunk.text,
        source=best_result.chunk.source,
        page_number=best_result.chunk.page_number,
        chunk_id=best_result.chunk.chunk_id,
        retrieval_score=best_result.score,
    )

    return DocumentAnswer(
        query=retrieval.query,
        source=retrieval.source,
        page_count=retrieval.page_count,
        chunk_count=retrieval.chunk_count,
        answered=True,
        answer=best_answer_text,
        answer_confidence=best_extracted_answer.confidence,
        retrieval_score=best_result.score,
        citation=citation,
    )