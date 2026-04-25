from uuid import UUID, uuid4

from backend.app.ingestion.chunker import chunk_sections
from backend.app.models.schemas import ParsedSection, TextChunk


def _section(
    text: str,
    *,
    document_id: UUID | None = None,
    source_file: str = "contract.pdf",
    page_number: int | None = 1,
    section_header: str | None = "Article 1",
    section_index: int = 0,
) -> ParsedSection:
    return ParsedSection(
        document_id=document_id or uuid4(),
        source_file=source_file,
        page_number=page_number,
        section_header=section_header,
        text=text,
        section_index=section_index,
    )


def test_chunk_preserves_metadata(parsed_sections: list[ParsedSection]) -> None:
    chunks = chunk_sections(parsed_sections)

    assert len(chunks) > 0
    first = parsed_sections[0]
    matching = [c for c in chunks if c.page_number == first.page_number]
    assert matching, "No chunk inherited page_number from the first section"
    chunk = matching[0]

    assert chunk.source_file == first.source_file
    assert chunk.page_number == first.page_number
    assert chunk.section_header == first.section_header
    assert chunk.document_id == first.document_id


def test_chunk_short_section_not_dropped() -> None:
    section = _section("Short.")
    chunks = chunk_sections([section], chunk_size=512, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].text == "Short."


def test_chunk_long_section_splits_correctly() -> None:
    long_text = "word " * 300
    section = _section(long_text)
    chunks = chunk_sections([section], chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_whitespace_section_skipped() -> None:
    doc_id = uuid4()
    sections = [
        _section("   \n\t  ", document_id=doc_id, section_index=0),
        _section("Valid text here.", document_id=doc_id, section_index=1),
    ]
    chunks = chunk_sections(sections)

    assert len(chunks) == 1
    assert "Valid text here." in chunks[0].text


def test_chunk_indices_are_sequential() -> None:
    doc_id = uuid4()
    sections = [
        _section("word " * 100, document_id=doc_id, section_index=0),
        _section("word " * 100, document_id=doc_id, section_index=1),
    ]
    chunks = chunk_sections(sections, chunk_size=100, chunk_overlap=10)

    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index values are not sequential"
    assert all(c.total_chunks == len(chunks) for c in chunks), "total_chunks is inconsistent"


def test_chunk_empty_sections_list_returns_empty() -> None:
    result = chunk_sections([])
    assert result == []
