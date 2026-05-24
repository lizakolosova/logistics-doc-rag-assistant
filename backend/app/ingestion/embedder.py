import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import chromadb
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import EmbeddingError
from app.ingestion.chunker import chunk_sections
from app.ingestion.parser import parse_document
from app.models.database import Chunk, Document
from app.models.schemas import DocumentStatus, EmbeddedChunk, TextChunk

logger = logging.getLogger(__name__)

_embed_model: Any = None


def _get_embed_model() -> Any:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


async def embed_chunks(chunks: list[TextChunk]) -> list[EmbeddedChunk]:
    """Embed text chunks using the local sentence-transformers model.

    Args:
        chunks: Text chunks to embed.

    Returns:
        Same chunks with a 384-dim embedding vector attached.

    Raises:
        EmbeddingError: On embedding failure.
    """
    if not chunks:
        return []

    start = time.monotonic()
    model = _get_embed_model()

    try:
        raw = await asyncio.to_thread(model.encode, [c.text for c in chunks], show_progress_bar=False)
        embeddings = raw.tolist()
    except Exception as exc:
        raise EmbeddingError(str(exc)) from exc

    embedded: list[EmbeddedChunk] = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                source_file=chunk.source_file,
                section_header=chunk.section_header,
                embedding=embedding,
            )
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info("Embedded %d chunks, elapsed=%dms", len(chunks), elapsed_ms)
    return embedded


async def store_in_chroma(embedded: list[EmbeddedChunk], collection_name: str | None = None) -> None:
    """Upsert embedded chunks into ChromaDB.

    Args:
        embedded: Chunks with embedding vectors to store.
        collection_name: Target collection; defaults to settings.chroma_collection.
    """
    if not embedded:
        return

    resolved_name = collection_name if collection_name is not None else settings.chroma_collection
    client = await chromadb.AsyncHttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collection = await client.get_or_create_collection(resolved_name)

    ids = [f"{chunk.document_id}_{chunk.chunk_index}" for chunk in embedded]
    await collection.upsert(
        ids=ids,
        embeddings=[chunk.embedding for chunk in embedded],
        documents=[chunk.text for chunk in embedded],
        metadatas=[
            {
                "source_file": chunk.source_file or "",
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
                "section_header": chunk.section_header or "",
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
            }
            for chunk in embedded
        ],
    )

    logger.info("Stored %d chunk(s) in ChromaDB collection '%s'", len(embedded), resolved_name)



async def ingest_document(file_path: Path, session: AsyncSession, original_filename: str | None = None) -> UUID:
    """Run the full ingestion pipeline for a single document.

    Parses, chunks, embeds, and stores the document atomically. On any failure
    the document is marked as failed in Postgres and the exception is re-raised.

    Args:
        file_path: Absolute path to the uploaded file (PDF or DOCX).
        session: Active async SQLAlchemy session.

    Returns:
        UUID of the successfully ingested document.

    Raises:
        DocumentParseError: If the file cannot be parsed.
        EmbeddingError: If the embedding call fails.
        DocumentTooLargeError: If the file exceeds configured size/page limits.
    """
    document_id = uuid4()
    filename = original_filename or file_path.name
    file_size = file_path.stat().st_size

    doc = Document(
        id=document_id,
        filename=filename,
        upload_time=datetime.now(UTC),
        file_size_bytes=file_size,
        status=DocumentStatus.processing,
    )
    session.add(doc)
    await session.commit()

    written_ids: list[str] = []

    try:
        sections = parse_document(file_path, document_id, source_filename=filename)
        chunks = chunk_sections(sections)
        embedded = await embed_chunks(chunks)

        written_ids = [f"{document_id}_{chunk.chunk_index}" for chunk in chunks]
        await store_in_chroma(embedded)

        for chunk in chunks:
            session.add(
                Chunk(
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    section_header=chunk.section_header,
                    chroma_id=f"{document_id}_{chunk.chunk_index}",
                )
            )

        doc.status = DocumentStatus.ready
        doc.num_chunks = len(chunks)
        await session.commit()

        logger.info("Ingested document_id=%s (%s), %d chunk(s)", document_id, filename, len(chunks))
        return document_id

    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", filename, exc)
        await session.rollback()

        if written_ids:
            try:
                client = await chromadb.AsyncHttpClient(host=settings.chroma_host, port=settings.chroma_port)
                collection = await client.get_collection(settings.chroma_collection)
                await collection.delete(ids=written_ids)
                logger.info("Rolled back %d ChromaDB vector(s) for document_id=%s", len(written_ids), document_id)
            except Exception as chroma_exc:
                logger.error(
                    "Failed to clean up ChromaDB vectors after rollback for document_id=%s: %s",
                    document_id,
                    chroma_exc,
                )

        doc.status = DocumentStatus.failed
        doc.error_message = str(exc)
        session.add(doc)
        await session.commit()
        raise
