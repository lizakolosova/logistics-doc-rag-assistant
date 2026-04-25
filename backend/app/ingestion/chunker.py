import logging
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models.schemas import ParsedSection, TextChunk

logger = logging.getLogger(__name__)


def chunk_sections(sections: list[ParsedSection], chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[TextChunk]:
    if not sections: return []

    start = time.monotonic()
    document_id = sections[0].document_id

    _chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    _chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
    )

    raw: list[tuple[str, ParsedSection]] = []
    for section in sections:
        if not section.text.strip():
            continue
        for piece in splitter.split_text(section.text):
            raw.append((piece, section))

    total = len(raw)

    chunks = [
        TextChunk( document_id=section.document_id, source_file=section.source_file, section_header=section.section_header,
            text=text, chunk_index=i, total_chunks=total, page_number=section.page_number,token_count=len(text.split()))
        for i, (text, section) in enumerate(raw)
    ]

    elapsed_ms = int((time.monotonic() - start) * 1000)
    avg_len = sum(len(c.text) for c in chunks) / total if total else 0.0

    logger.info( "Chunked document_id=%s: %d sections → %d chunks, avg_len=%.0f chars, elapsed=%dms",document_id,
        len(sections), total, avg_len, elapsed_ms)

    return chunks