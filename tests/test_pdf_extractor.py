from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.services.pdf_extractor import (
    ExtractedPage,
    extract_pdf_pages,
    normalize_whitespace,
    select_main_text_blocks,
)


def create_sample_pdf(file_path: Path) -> None:
    """Create a small two-page PDF for testing."""

    pdf = canvas.Canvas(str(file_path))

    pdf.drawString(
        72,
        720,
        "Machine learning uses data to identify patterns.",
    )
    pdf.showPage()

    pdf.drawString(
        72,
        720,
        "Responsible AI considers fairness and transparency.",
    )
    pdf.save()


def test_extract_pdf_pages(tmp_path: Path) -> None:
    sample_pdf = tmp_path / "sample_reading.pdf"
    create_sample_pdf(sample_pdf)

    pages = extract_pdf_pages(sample_pdf)

    assert len(pages) == 2
    assert all(isinstance(page, ExtractedPage) for page in pages)

    assert pages[0].page_number == 1
    assert pages[0].source == "sample_reading.pdf"
    assert "Machine learning" in pages[0].text

    assert pages[1].page_number == 2
    assert pages[1].source == "sample_reading.pdf"
    assert "Responsible AI" in pages[1].text


def test_normalize_whitespace() -> None:
    text = "Machine   learning\nuses\tdata."

    result = normalize_whitespace(text)

    assert result == "Machine learning uses data."


def test_missing_pdf_raises_error(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(missing_pdf)


def test_non_pdf_file_is_rejected(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("This is not a PDF.", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        extract_pdf_pages(text_file)


def test_labelled_left_sidebar_is_removed() -> None:
    blocks = [
        (
            10.0,
            100.0,
            170.0,
            700.0,
            (
                "PROGRAM GOALS\n"
                "GRADES\n"
                "MATERIALS\n"
                "Cooking oil\n"
                "Water"
            ),
        ),
        (
            190.0,
            100.0,
            590.0,
            700.0,
            (
                "Sharks mainly rely on their large "
                "oil-filled liver to stay buoyant."
            ),
        ),
    ]

    selected = select_main_text_blocks(
        blocks=blocks,
        page_width=600.0,
    )

    assert selected == [
        (
            "Sharks mainly rely on their large "
            "oil-filled liver to stay buoyant."
        )
    ]


def test_normal_left_text_is_kept_without_sidebar_labels() -> None:
    blocks = [
        (
            10.0,
            100.0,
            170.0,
            700.0,
            "This is ordinary document text.",
        ),
        (
            190.0,
            100.0,
            590.0,
            700.0,
            "This is another document section.",
        ),
    ]

    selected = select_main_text_blocks(
        blocks=blocks,
        page_width=600.0,
    )

    assert len(selected) == 2