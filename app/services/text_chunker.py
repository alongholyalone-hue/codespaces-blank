from dataclasses import dataclass

from app.services.pdf_extractor import ExtractedPage

import re

@dataclass(frozen=True)
class TextChunk:
    """A searchable section of text with citation metadata."""

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int


def split_paragraphs(text: str) -> list[str]:
    """
    Split extracted page text into separate PDF text blocks.

    PyMuPDF blocks are separated by blank lines. Internal line
    breaks inside each block are converted into spaces.
    """

    raw_paragraphs = re.split(
        r"\n\s*\n+",
        text.strip(),
    )

    paragraphs: list[str] = []

    for paragraph in raw_paragraphs:
        cleaned = " ".join(paragraph.split())

        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


def chunk_paragraph(
    paragraph: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Divide one paragraph into overlapping word-based chunks.

    Overlap is applied only within the paragraph, never between
    unrelated PDF blocks.
    """

    words = paragraph.split()

    if not words:
        return []

    paragraph_chunks: list[str] = []
    start_index = 0

    while start_index < len(words):
        end_index = min(
            start_index + chunk_size,
            len(words),
        )

        paragraph_chunks.append(
            " ".join(words[start_index:end_index])
        )

        if end_index >= len(words):
            break

        start_index = end_index - overlap

    return paragraph_chunks


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 120,
    overlap: int = 30,
) -> list[TextChunk]:
    """
    Convert extracted PDF pages into searchable text chunks.

    PDF block and paragraph boundaries are preserved so unrelated
    captions, headings, columns, and body text are not merged.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks: list[TextChunk] = []

    for page in pages:
        paragraphs = split_paragraphs(page.text)
        chunk_index = 1

        for paragraph in paragraphs:
            paragraph_chunks = chunk_paragraph(
                paragraph=paragraph,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk_text in paragraph_chunks:
                chunks.append(
                    TextChunk(
                        chunk_id=(
                            f"{page.source}"
                            f"-page-{page.page_number}"
                            f"-chunk-{chunk_index}"
                        ),
                        text=chunk_text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

    return chunks
