import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import openai
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import EmbeddingError
from app.ingestion.chunker import chunk_sections
from app.ingestion.parser import parse_document
from app.models.database import Chunk, Document
from app.models.schemas import DocumentStatus, EmbeddedChunk, TextChunk


logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 100


async def embed_chunks(chunks: list[TextChunk]) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    start = time.monotonic()
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    batches = [chunks[i : i + _EMBED_BATCH_SIZE] for i in range(0, len(chunks), _EMBED_BATCH_SIZE)]
    logger.info("Embedding %d chunks in %d batch(es)", len(chunks), len(batches))

    embedded: list[EmbeddedChunk] = []
    for batch in batches:
        try:
            response = await client.embeddings.create( model=settings.embedding_model, input=[c.text for c in batch])
        except openai.OpenAIError as exc:
            raise EmbeddingError(str(exc)) from exc

        for chunk, item in zip(batch, response.data):
            embedded.append(EmbeddedChunk(chunk_id=chunk.chunk_id, document_id=chunk.document_id, text=chunk.text,
                    chunk_index=chunk.chunk_index, page_number=chunk.page_number, source_file=chunk.source_file,
                              section_header=chunk.section_header,embedding=item.embedding))

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info( "Embedded %d chunks in %d batch(es), elapsed=%dms",len(chunks), len(batches), elapsed_ms)
    return embedded


async def store_in_chroma(embedded: list[EmbeddedChunk], collection_name: str = "documents") -> None:

    if not embedded:
        return

    client = await chromadb.AsyncHttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collection = await client.get_or_create_collection(collection_name)

    ids = [f"{chunk.document_id}_{chunk.chunk_index}" for chunk in embedded]
    await collection.upsert(ids=ids, embeddings=[chunk.embedding for chunk in embedded],
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

    logger.info("Stored %d chunk(s) in ChromaDB collection '%s'", len(embedded), collection_name)


async def store_in_postgres(document_id: UUID, filename: str, chunks: list[TextChunk], session: AsyncSession) -> None:
    doc = Document(id=document_id, filename=filename, upload_time=datetime.now(UTC), num_chunks=len(chunks),
        status=DocumentStatus.ready, file_size_bytes=0)
    session.add(doc)

    for chunk in chunks:
        session.add(Chunk(document_id=document_id,chunk_index=chunk.chunk_index, text=chunk.text,
                page_number=chunk.page_number, section_header=chunk.section_header, chroma_id=f"{document_id}_{chunk.chunk_index}"))

    await session.commit()


async def ingest_document(file_path: Path, session: AsyncSession) -> UUID:
    document_id = uuid4()
    filename = file_path.name
    file_size = file_path.stat().st_size

    doc = Document( id=document_id, filename=filename, upload_time=datetime.now(UTC), file_size_bytes=file_size,
                    status=DocumentStatus.processing, )
    session.add(doc)
    await session.commit()

    try:
        sections = parse_document(file_path, document_id)
        chunks = chunk_sections(sections)
        embedded = await embed_chunks(chunks)
        await store_in_chroma(embedded)

        for chunk in chunks:
            session.add(Chunk(document_id=document_id, chunk_index=chunk.chunk_index, text=chunk.text,
                    page_number=chunk.page_number, section_header=chunk.section_header, chroma_id=f"{document_id}_{chunk.chunk_index}"))

        doc.status = DocumentStatus.ready
        doc.num_chunks = len(chunks)
        await session.commit()

        logger.info("Ingested document_id=%s (%s), %d chunk(s)", document_id, filename, len(chunks))
        return document_id

    except Exception as exc:
        logger.error("Ingestion failed for '%s': %s", filename, exc)
        await session.rollback()
        doc.status = DocumentStatus.failed
        doc.error_message = str(exc)
        session.add(doc)
        await session.commit()
        raise