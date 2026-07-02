from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.answer_pipeline import answer_from_pdf
from app.services.pdf_extractor import PDFExtractionError
from app.services.retrieval_pipeline import retrieve_from_pdf

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class RetrievedPassage(BaseModel):
    """One relevant passage retrieved from an uploaded PDF."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    score: float


class AnswerCitationResponse(BaseModel):
    """Evidence supporting a direct document answer."""

    text: str
    source: str
    page_number: int
    chunk_id: str
    retrieval_score: float


class DocumentAnswerResponse(BaseModel):
    """Direct answer extracted from an uploaded PDF."""

    query: str
    source: str
    page_count: int
    chunk_count: int
    answered: bool
    answer: str
    answer_confidence: float | None
    retrieval_score: float
    citation: AnswerCitationResponse | None


class DocumentSearchResponse(BaseModel):
    """API response for a document-search request."""

    query: str
    source: str
    page_count: int
    chunk_count: int
    results: list[RetrievedPassage]


@router.post(
    "/search",
    response_model=DocumentSearchResponse,
)
async def search_document(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(3),
) -> DocumentSearchResponse:
    """
    Upload a PDF and retrieve passages related to a question.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty",
        )

    if top_k < 1 or top_k > 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10",
        )

    original_filename = file.filename or ""

    # Remove any directory information supplied in the filename.
    safe_filename = Path(
        original_filename.replace("\\", "/")
    ).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    file_contents = await file.read(
        MAX_FILE_SIZE_BYTES + 1
    )

    await file.close()

    if len(file_contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF must be 10 MB or smaller",
        )

    if not file_contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty",
        )

    try:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = (
                Path(temporary_directory) / safe_filename
            )

            temporary_path.write_bytes(file_contents)

            retrieval = retrieve_from_pdf(
                file_path=temporary_path,
                query=cleaned_query,
                top_k=top_k,
            )

    except (
        PDFExtractionError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


    return DocumentSearchResponse(
        query=retrieval.query,
        source=retrieval.source,
        page_count=retrieval.page_count,
        chunk_count=retrieval.chunk_count,
        results=[
            RetrievedPassage(
                text=result.chunk.text,
                source=result.chunk.source,
                page_number=result.chunk.page_number,
                chunk_id=result.chunk.chunk_id,
                score=result.score,
            )
            for result in retrieval.results
        ],
    )


@router.post(
    "/answer",
    response_model=DocumentAnswerResponse,
)
async def answer_document(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(3),
) -> DocumentAnswerResponse:
    """
    Upload a PDF and extract a direct answer with a citation.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty",
        )

    if top_k < 1 or top_k > 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10",
        )

    original_filename = file.filename or ""

    safe_filename = Path(
        original_filename.replace("\\", "/")
    ).name

    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    file_contents = await file.read(
        MAX_FILE_SIZE_BYTES + 1
    )

    await file.close()

    if len(file_contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF must be 10 MB or smaller",
        )

    if not file_contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty",
        )

    try:
        with TemporaryDirectory() as temporary_directory:
            temporary_path = (
                Path(temporary_directory) / safe_filename
            )

            temporary_path.write_bytes(file_contents)

            result = answer_from_pdf(
                file_path=temporary_path,
                question=cleaned_query,
                top_k=top_k,
            )

    except (
        PDFExtractionError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    citation = None

    if result.citation is not None:
        citation = AnswerCitationResponse(
            text=result.citation.text,
            source=result.citation.source,
            page_number=result.citation.page_number,
            chunk_id=result.citation.chunk_id,
            retrieval_score=result.citation.retrieval_score,
        )

    return DocumentAnswerResponse(
        query=result.query,
        source=result.source,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        answered=result.answered,
        answer=result.answer,
        answer_confidence=result.answer_confidence,
        retrieval_score=result.retrieval_score,
        citation=citation,
    )