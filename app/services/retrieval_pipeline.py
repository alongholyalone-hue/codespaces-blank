from dataclasses import dataclass
from pathlib import Path

from app.services.pdf_extractor import extract_pdf_pages
from app.services.semantic_search import SearchResult, search_chunks
from app.services.text_chunker import chunk_pages


@dataclass(frozen=True)
class RetrievalResponse:
    """Results and processing information for one PDF question."""

    query: str
    source: str
    page_count: int
    chunk_count: int
    results: list[SearchResult]


def retrieve_from_pdf(
    file_path: str | Path,
    query: str,
    top_k: int = 3,
    chunk_size: int = 120,
    overlap: int = 30,
) -> RetrievalResponse:
    """
    Extract, chunk, and search a PDF for passages related to a question.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty")

    path = Path(file_path)

    pages = extract_pdf_pages(path)

    chunks = chunk_pages(
        pages=pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    results = (
        search_chunks(
            query=cleaned_query,
            chunks=chunks,
            top_k=top_k,
        )
        if chunks
        else []
    )

    return RetrievalResponse(
        query=cleaned_query,
        source=path.name,
        page_count=len(pages),
        chunk_count=len(chunks),
        results=results,
    )