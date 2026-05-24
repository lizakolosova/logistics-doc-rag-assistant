"""Tests for backend/app/ingestion/parser.py."""

from pathlib import Path
from uuid import UUID, uuid4

import fitz
import pytest
from docx import Document

from app.exceptions import DocumentParseError, DocumentTooLargeError, UnsupportedFormatError
from app.ingestion.parser import parse_document, parse_docx, parse_pdf
import app.ingestion.parser as parser_module

def test_parse_pdf_extracts_text(pdf_two_pages: Path, document_id: UUID) -> None:
    sections = parse_pdf(pdf_two_pages, document_id)

    assert len(sections) == 2
    combined = " ".join(s.text for s in sections)
    assert "page one" in combined.lower()
    assert "page two" in combined.lower()


def test_parse_pdf_preserves_page_numbers(pdf_two_pages: Path, document_id: UUID) -> None:
    sections = parse_pdf(pdf_two_pages, document_id)

    page_numbers = [s.page_number for s in sections]
    assert page_numbers == [1, 2]


def test_parse_pdf_image_only_raises(pdf_image_only: Path, document_id: UUID) -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse_pdf(pdf_image_only, document_id)

    assert "scanned image" in exc_info.value.reason
    assert "Page 1" in exc_info.value.reason
    assert exc_info.value.filename == pdf_image_only.name

def test_parse_docx_extracts_text(docx_with_headings: Path, document_id: UUID) -> None:
    sections = parse_docx(docx_with_headings, document_id)

    combined = " ".join(s.text for s in sections)
    assert "introduction" in combined.lower()
    assert "conclusion" in combined.lower()


def test_parse_docx_groups_by_heading(docx_with_headings: Path, document_id: UUID) -> None:
    sections = parse_docx(docx_with_headings, document_id)

    assert len(sections) == 2

    intro_section = next(s for s in sections if "Introduction" in s.text)
    assert "introduction paragraph" in intro_section.text.lower()

    conc_section = next(s for s in sections if "Conclusion" in s.text)
    assert "conclusion paragraph" in conc_section.text.lower()


def test_parse_docx_no_headings_uses_default(docx_no_headings: Path, document_id: UUID) -> None:
    sections = parse_docx(docx_no_headings, document_id)

    assert len(sections) == 1
    assert "First paragraph" in sections[0].text
    assert "Second paragraph" in sections[0].text
    assert sections[0].page_number is None

def test_parse_document_unsupported_format_raises(
    tmp_path: Path, document_id: UUID
) -> None:
    txt_file = tmp_path / "contract.txt"
    txt_file.write_text("Some legal text.")

    with pytest.raises(UnsupportedFormatError) as exc_info:
        parse_document(txt_file, document_id)

    assert exc_info.value.filename == "contract.txt"
    assert ".txt" in exc_info.value.detected_type


def test_parse_document_too_large_raises(
    tmp_path: Path,
    document_id: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "small.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((50, 72), "hello")
    doc.save(str(pdf_path))
    doc.close()

    monkeypatch.setattr(parser_module.settings, "max_file_size_mb", 0)
    with pytest.raises(DocumentTooLargeError) as exc_info:
        parse_document(pdf_path, document_id)

    assert exc_info.value.limit_type == "size"
    assert exc_info.value.filename == "small.pdf"

    monkeypatch.setattr(parser_module.settings, "max_file_size_mb", 50)
    monkeypatch.setattr(parser_module.settings, "max_pages", 0)

    with pytest.raises(DocumentTooLargeError) as exc_info:
        parse_document(pdf_path, document_id)

    assert exc_info.value.limit_type == "pages"


def test_parse_document_docx_too_many_sections_raises(
    tmp_path: Path,
    document_id: UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docx_path = tmp_path / "large.docx"
    doc = Document()
    for i in range(5):
        doc.add_paragraph(f"Paragraph number {i}.")
    doc.save(str(docx_path))

    monkeypatch.setattr(parser_module.settings, "max_pages", 3)

    with pytest.raises(DocumentTooLargeError) as exc_info:
        parse_document(docx_path, document_id)

    assert exc_info.value.limit_type == "sections"
    assert exc_info.value.filename == "large.docx"
    assert exc_info.value.actual == 5
    assert exc_info.value.limit == 3
