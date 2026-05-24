import logging
import tempfile
from pathlib import Path
from uuid import UUID

import chromadb
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.dependencies import get_session
from app.ingestion.embedder import ingest_document
from app.models.database import Chunk, Document
from app.models.schemas import DocumentResponse, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    request: Request,
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    suffix = Path(file.filename or "upload").suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        document_id = await ingest_document(tmp_path, session, original_filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    request.app.state.bm25_dirty = True
    return IngestResponse(document_id=document_id, status="ready")


@router.get("", response_model=list[DocumentResponse])
async def list_documents(session: AsyncSession = Depends(get_session)) -> list[DocumentResponse]:
    result = await session.execute(select(Document).order_by(Document.upload_time.desc()))
    docs = result.scalars().all()
    return [
        DocumentResponse(
            document_id=doc.id,
            filename=doc.filename,
            status=doc.status,
            num_chunks=doc.num_chunks,
            upload_time=doc.upload_time,
            file_size_bytes=doc.file_size_bytes,
            error_message=doc.error_message,
        )
        for doc in docs
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    request: Request,
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    chroma_ids_result = await session.execute(select(Chunk.chroma_id).where(Chunk.document_id == document_id))
    chroma_ids = chroma_ids_result.scalars().all()

    if chroma_ids:
        try:
            client = await chromadb.AsyncHttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
            collection = await client.get_collection(settings.chroma_collection)
            await collection.delete(ids=list(chroma_ids))
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"ChromaDB unavailable; document not deleted: {exc}",
            ) from exc

    await session.delete(doc)
    await session.commit()
    request.app.state.bm25_dirty = True
