import re
import pymupdf

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedPage:
    """Text and citation information extracted from one PDF page."""

    page_number: int
    text: str
    source: str


class PDFExtractionError(Exception):
    """Raised when a PDF exists but cannot be processed."""


TextBlock = tuple[float, float, float, float, str]

SIDEBAR_MARKERS = (
    "program goals",
    "grades",
    "materials",
    "recommended assessment",
)


def select_main_text_blocks(
    blocks: list[TextBlock],
    page_width: float,
) -> list[str]:
    """
    Remove a labelled narrow left sidebar while keeping the
    main document text.

    This is a layout heuristic, not a universal PDF rule.
    """

    cleaned_blocks: list[TextBlock] = []

    for x0, y0, x1, y1, text in blocks:
        cleaned_text = normalize_whitespace(text)

        if not cleaned_text:
            continue

        cleaned_blocks.append(
            (x0, y0, x1, y1, cleaned_text)
        )

    def is_narrow_left_block(block: TextBlock) -> bool:
        x0, _, x1, _, _ = block

        return (
            x0 <= page_width * 0.15
            and x1 <= page_width * 0.33
        )

    sidebar_exists = any(
        is_narrow_left_block(block)
        and any(
            marker in block[4].lower()
            for marker in SIDEBAR_MARKERS
        )
        for block in cleaned_blocks
    )

    selected_blocks: list[str] = []

    for block in cleaned_blocks:
        if (
            sidebar_exists
            and is_narrow_left_block(block)
        ):
            continue

        selected_blocks.append(block[4])

    return selected_blocks


def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace with single spaces."""

    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_pages(
    file_path: str | Path,
) -> list[ExtractedPage]:
    """
    Extract text from every page of a PDF.

    Page numbers begin at 1 so that they match the page references
    normally shown to users.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {path.suffix}"
        )

    try:
        document = pymupdf.open(path)
    except (RuntimeError, OSError) as exc:
        raise PDFExtractionError(
            f"Unable to read PDF file: {path.name}"
        ) from exc

    extracted_pages: list[ExtractedPage] = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            raw_blocks: list[TextBlock] = []

            for block in page.get_text(
                "blocks",
                sort=True,
            ):
                # Text blocks use block type 0.
                # Image blocks use block type 1.
                if len(block) < 7 or int(block[6]) != 0:
                    continue

                raw_blocks.append(
                    (
                        float(block[0]),
                        float(block[1]),
                        float(block[2]),
                        float(block[3]),
                        str(block[4]),
                    )
                )

            main_blocks = select_main_text_blocks(
                blocks=raw_blocks,
                page_width=float(page.rect.width),
            )

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text="\n\n".join(main_blocks),
                    source=path.name,
                )
            )

    finally:
        document.close()

    return extracted_pages