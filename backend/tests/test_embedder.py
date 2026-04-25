
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.app.exceptions import EmbeddingError
from backend.app.ingestion.embedder import embed_chunks, ingest_document, store_in_chroma
from backend.app.models.database import Document
from backend.app.models.schemas import DocumentStatus, TextChunk


def _make_chunks(n: int) -> list[TextChunk]:
    doc_id = uuid4()
    return [
        TextChunk(
            document_id=doc_id,
            source_file="contract.pdf",
            section_header="Clause 1",
            text=f"Chunk number {i}.",
            chunk_index=i,
            total_chunks=n,
            page_number=1,
            token_count=3,
        )
        for i in range(n)
    ]


def _fake_embedding_response(texts: list[str]) -> MagicMock:
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]
    return response

async def test_embed_chunks_batches_correctly() -> None:
    chunks = _make_chunks(250)

    mock_create = AsyncMock(
        side_effect=lambda model, input: _fake_embedding_response(input)
    )

    with patch("app.ingestion.embedder.openai.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.embeddings.create = mock_create
        mock_openai_cls.return_value = mock_client

        result = await embed_chunks(chunks)

    assert mock_create.call_count == 3
    assert len(result) == 250
    assert all(len(e.embedding) == 1536 for e in result)


async def test_embed_chunks_raises_on_api_failure() -> None:
    import openai as openai_module

    chunks = _make_chunks(5)

    mock_create = AsyncMock(side_effect=openai_module.APIError("server error", request=MagicMock(), body=None))

    with patch("app.ingestion.embedder.openai.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.embeddings.create = mock_create
        mock_openai_cls.return_value = mock_client

        with pytest.raises(EmbeddingError, match="server error"):
            await embed_chunks(chunks)


async def test_store_in_chroma_uses_idempotent_ids() -> None:
    import sys

    from app.models.schemas import EmbeddedChunk

    doc_id = uuid4()
    embedded = [
        EmbeddedChunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            text="sample text",
            chunk_index=i,
            page_number=1,
            source_file="contract.pdf",
            section_header="Clause 1",
            embedding=[0.0] * 1536,
        )
        for i in range(3)
    ]

    mock_collection = AsyncMock()
    mock_client = AsyncMock()
    mock_client.get_or_create_collection = AsyncMock(return_value=mock_collection)

    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(return_value=mock_client)

    with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
        await store_in_chroma(embedded)

    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args.kwargs
    expected_ids = [f"{doc_id}_{i}" for i in range(3)]
    assert call_kwargs["ids"] == expected_ids

def _make_mock_session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    return session


async def test_ingest_document_marks_failed_on_error(tmp_path: Path) -> None:
    from app.exceptions import DocumentParseError

    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"dummy content")

    added_objects: list = []
    mock_session = _make_mock_session()
    mock_session.add.side_effect = lambda obj: added_objects.append(obj)

    with patch(
        "app.ingestion.embedder.parse_document",
        side_effect=DocumentParseError("test.pdf", "no text found"),
    ):
        with pytest.raises(DocumentParseError):
            await ingest_document(file_path, mock_session)

    db_docs = [obj for obj in added_objects if isinstance(obj, Document)]
    assert db_docs, "Expected at least one Document to be added to the session"
    assert db_docs[-1].status == DocumentStatus.failed
    assert "no text found" in (db_docs[-1].error_message or "")


async def test_ingest_document_returns_uuid_on_success(tmp_path: Path) -> None:
    from uuid import UUID

    from app.models.schemas import EmbeddedChunk

    file_path = tmp_path / "contract.pdf"
    file_path.write_bytes(b"dummy content")

    doc_id_holder: list[Document] = []
    mock_session = _make_mock_session()
    mock_session.add.side_effect = lambda obj: (
        doc_id_holder.append(obj) if isinstance(obj, Document) else None
    )

    fake_sections = [
        MagicMock(
            document_id=uuid4(),
            source_file="contract.pdf",
            page_number=1,
            section_header=None,
            text="This is a test legal clause with enough text.",
            section_index=0,
        )
    ]
    fake_chunks = _make_chunks(2)
    fake_embedded = [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            text=c.text,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            source_file=c.source_file,
            section_header=c.section_header,
            embedding=[0.1] * 1536,
        )
        for c in fake_chunks
    ]

    with (
        patch("app.ingestion.embedder.parse_document", return_value=fake_sections),
        patch("app.ingestion.embedder.chunk_sections", return_value=fake_chunks),
        patch("app.ingestion.embedder.embed_chunks", return_value=fake_embedded),
        patch("app.ingestion.embedder.store_in_chroma", new_callable=AsyncMock),
    ):
        result = await ingest_document(file_path, mock_session)

    assert isinstance(result, UUID)
    assert doc_id_holder, "Expected Document to be added to the session"
    assert doc_id_holder[-1].status == DocumentStatus.ready
