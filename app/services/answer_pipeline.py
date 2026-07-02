import re
from dataclasses import dataclass
from pathlib import Path

from app.services.answer_extractor import extract_answer
from app.services.retrieval_pipeline import retrieve_from_pdf
from app.services.reranker import rerank_chunks
from app.services.semantic_search import SearchResult
from app.services.text_chunker import TextChunk


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The uploaded document does not provide enough evidence "
    "to answer this question."
)


EXPLANATORY_PREFIXES = (
    "why ",
    "explain ",
    "describe ",
)

FACTOID_HOW_PREFIXES = (
    "how many ",
    "how much ",
    "how old ",
    "how long ",
    "how far ",
    "how often ",
)


def is_explanatory_question(question: str) -> bool:
    """
    Determine whether a question requires an explanation rather
    than a short factual answer.
    """

    cleaned_question = " ".join(
        question.lower().strip().split()
    )

    if not cleaned_question:
        return False

    if cleaned_question.startswith(FACTOID_HOW_PREFIXES):
        return False

    if cleaned_question.startswith("how "):
        return True

    return cleaned_question.startswith(
        EXPLANATORY_PREFIXES
    )


def build_explanatory_answer(
    question: str,
    candidate_results: list[SearchResult],
    maximum_sentences: int = 3,
    minimum_words: int = 5,
) -> str:
    """
    Combine several relevant evidence sentences into an explanation.

    Question echoes, duplicate sentences, questions, and very short
    fragments are excluded.
    """

    if maximum_sentences <= 0:
        raise ValueError(
            "maximum_sentences must be greater than zero"
        )

    if minimum_words <= 0:
        raise ValueError(
            "minimum_words must be greater than zero"
        )

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()

    for result in candidate_results:
        sentence = result.chunk.text.strip()

        if not sentence:
            continue

        if is_question_echo(
            question=question,
            candidate=sentence,
        ):
            continue

        # Exclude discussion questions contained in worksheets.
        if sentence.endswith("?"):
            continue

        words = normalize_tokens(sentence)

        # Reject vague fragments such as "in the water".
        if len(words) < minimum_words:
            continue

        normalized_sentence = " ".join(words)

        if normalized_sentence in seen_sentences:
            continue

        selected_sentences.append(sentence)
        seen_sentences.add(normalized_sentence)

        if len(selected_sentences) >= maximum_sentences:
            break

    return " ".join(selected_sentences)


QUESTION_WORDS = {
    "what",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "which",
}


def normalize_tokens(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""

    return re.findall(r"[a-z0-9]+", text.lower())


def remove_leading_question_echo(
    question: str,
    context: str,
) -> str:
    """
    Remove a repeated question from the beginning of a passage.

    This handles PDFs that use the user's question as a heading
    before presenting the actual answer.
    """

    cleaned_context = context.strip()
    question_stem = question.strip().rstrip("?!.-:").strip()

    if not question_stem:
        return cleaned_context

    return re.sub(
        rf"^\s*{re.escape(question_stem)}\s*[?!.\-:]*\s*",
        "",
        cleaned_context,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def is_question_echo(
    question: str,
    candidate: str,
) -> bool:
    """Detect a sentence that merely repeats the question."""

    question_tokens = normalize_tokens(question)
    candidate_tokens = normalize_tokens(candidate)

    if not question_tokens or not candidate_tokens:
        return False

    normalized_question = " ".join(question_tokens)
    normalized_candidate = " ".join(candidate_tokens)

    if normalized_question == normalized_candidate:
        return True

    overlap = len(
        set(question_tokens).intersection(candidate_tokens)
    ) / len(set(question_tokens))

    return (
        candidate.strip().endswith("?")
        and overlap >= 0.80
    )


def is_uninformative_answer(
    question: str,
    answer: str,
) -> bool:
    """Reject answers that only repeat question words."""

    answer_tokens = normalize_tokens(answer)

    if not answer_tokens:
        return True

    if (
        len(answer_tokens) == 1
        and answer_tokens[0] in QUESTION_WORDS
    ):
        return True

    question_tokens = set(normalize_tokens(question))

    if (
        len(answer_tokens) <= 2
        and set(answer_tokens).issubset(question_tokens)
    ):
        return True

    return False


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


def create_sentence_chunks(
    results: list[SearchResult],
) -> list[TextChunk]:
    """Split retrieved passages into sentence-level candidates."""

    sentence_chunks: list[TextChunk] = []

    for result in results:
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                result.chunk.text,
            )
            if sentence.strip()
        ]

        for sentence_index, sentence in enumerate(
            sentences,
            start=1,
        ):
            sentence_chunks.append(
                TextChunk(
                    chunk_id=(
                        f"{result.chunk.chunk_id}"
                        f"-sentence-{sentence_index}"
                    ),
                    text=sentence,
                    source=result.chunk.source,
                    page_number=result.chunk.page_number,
                    chunk_index=sentence_index,
                )
            )

    return sentence_chunks


def rerank_sentence_candidates(
    question: str,
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """
    Re-rank individual sentences when retrieved passages contain
    multiple sentences.
    """

    all_sentence_chunks = create_sentence_chunks(results)

    sentence_chunks = [
        chunk
        for chunk in all_sentence_chunks
        if not is_question_echo(
            question=question,
            candidate=chunk.text,
        )
    ]

    # No additional ranking is needed when every result already
    # contains only one sentence.
    if not sentence_chunks:
        return []

    if len(all_sentence_chunks) <= len(results):
        return [
            result
            for result in results
            if not is_question_echo(
                question=question,
                candidate=result.chunk.text,
            )
    ]

    candidate_count = min(
        max(top_k * 2, 3),
        len(sentence_chunks),
    )

    return rerank_chunks(
        query=question,
        chunks=sentence_chunks,
        top_k=candidate_count,
    )


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

    # Apply the semantic-retrieval threshold before reranking.
    # CrossEncoder scores use a different scale and should not
    # be compared directly with cosine-similarity thresholds.
    eligible_results = [
        result
        for result in retrieval.results
        if result.score >= minimum_retrieval_score
    ]

    if not eligible_results:
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

    candidate_results = rerank_sentence_candidates(
        question=cleaned_question,
        results=eligible_results,
        top_k=top_k,
    )

    best_result = None
    best_extracted_answer = None
    best_answer_text: str | None = None
    best_selection_score = (-1.0, -1.0)

    for result in candidate_results:
        qa_context = remove_leading_question_echo(
            question=cleaned_question,
            context=result.chunk.text,
        )

        if not qa_context:
            continue

        extracted = extract_answer(
            question=cleaned_question,
            context=qa_context,
    )

        if not extracted.text:
            continue

        cleaned_answer = clean_answer_text(
            question=cleaned_question,
            extracted_text=extracted.text,
        )

        if not cleaned_answer:
            continue

        if is_uninformative_answer(
            question=cleaned_question,
            answer=cleaned_answer,
        ):
            continue

        if extracted.confidence < minimum_answer_confidence:
            continue

        selection_score = (
            max(result.score, 0.0),
            extracted.confidence,
        )

        if selection_score > best_selection_score:
            best_result = result
            best_extracted_answer = extracted
            best_answer_text = cleaned_answer
            best_selection_score = selection_score

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