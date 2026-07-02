import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class ExtractedPage:
    """Text and citation information extracted from one PDF page."""

    page_number: int
    text: str
    source: str


class PDFExtractionError(Exception):
    """Raised when a PDF exists but cannot be processed."""


def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace with single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_pages(file_path: str | Path) -> list[ExtractedPage]:
    """
    Extract text from every page of a PDF.

    Page numbers begin at 1 so that they match the page references
    normally shown to users.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.suffix}")

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise PDFExtractionError(
            f"Unable to read PDF file: {path.name}"
        ) from exc

    extracted_pages: list[ExtractedPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = normalize_whitespace(raw_text)

        extracted_pages.append(
            ExtractedPage(
                page_number=page_number,
                text=cleaned_text,
                source=path.name,
            )
        )

    return extracted_pages
