from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

from app.exceptions import EmbeddingError
from app.ingestion.embedder import embed_chunks, ingest_document, store_in_chroma
from app.models.database import Document
from app.models.schemas import DocumentStatus, EmbeddedChunk, TextChunk


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


async def test_embed_chunks_batches_correctly() -> None:
    chunks = _make_chunks(250)

    mock_model = MagicMock()
    fake_embeddings = [[0.1] * 384 for _ in range(250)]
    mock_model.encode.return_value.tolist.return_value = fake_embeddings

    with patch("app.ingestion.embedder._get_embed_model", return_value=mock_model):
        result = await embed_chunks(chunks)

    mock_model.encode.assert_called_once()
    assert len(result) == 250
    assert all(len(e.embedding) == 384 for e in result)


async def test_embed_chunks_raises_on_api_failure() -> None:
    chunks = _make_chunks(5)

    mock_model = MagicMock()
    mock_model.encode.side_effect = RuntimeError("model error")

    with patch("app.ingestion.embedder._get_embed_model", return_value=mock_model):
        with pytest.raises(EmbeddingError, match="model error"):
            await embed_chunks(chunks)


async def test_store_in_chroma_uses_idempotent_ids() -> None:
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
            embedding=[0.0] * 384,
        )
        for i in range(3)
    ]

    mock_collection = AsyncMock()
    mock_client = AsyncMock()
    mock_client.get_or_create_collection = AsyncMock(return_value=mock_collection)

    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(return_value=mock_client)

    with patch("app.ingestion.embedder.chromadb", mock_chromadb):
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
            embedding=[0.1] * 384,
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


async def test_ingest_document_cleans_up_chroma_on_postgres_failure(tmp_path: Path) -> None:
    """Finding #5: if Postgres commit fails after ChromaDB write, cleanup must delete the vectors."""
    file_path = tmp_path / "contract.pdf"
    file_path.write_bytes(b"dummy content")

    fake_chunks = _make_chunks(2)
    doc_id = fake_chunks[0].document_id
    fake_embedded = [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            text=c.text,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            source_file=c.source_file,
            section_header=c.section_header,
            embedding=[0.1] * 384,
        )
        for c in fake_chunks
    ]

    commit_call = 0

    async def flaky_commit():
        nonlocal commit_call
        commit_call += 1
        if commit_call >= 2:
            raise RuntimeError("Postgres commit failed")

    mock_session = _make_mock_session()
    mock_session.commit.side_effect = flaky_commit

    mock_collection = AsyncMock()
    mock_client = AsyncMock()
    mock_client.get_collection = AsyncMock(return_value=mock_collection)
    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(return_value=mock_client)

    fake_section = MagicMock(
        document_id=doc_id,
        source_file="contract.pdf",
        page_number=1,
        section_header=None,
        text="some legal text",
        section_index=0,
    )

    with (
        patch("app.ingestion.embedder.parse_document", return_value=[fake_section]),
        patch("app.ingestion.embedder.chunk_sections", return_value=fake_chunks),
        patch("app.ingestion.embedder.embed_chunks", return_value=fake_embedded),
        patch("app.ingestion.embedder.store_in_chroma", new_callable=AsyncMock),
        patch("app.ingestion.embedder.chromadb", mock_chromadb),
    ):
        with pytest.raises(RuntimeError, match="Postgres commit failed"):
            await ingest_document(file_path, mock_session)

    mock_collection.delete.assert_called_once()
    deleted_ids = mock_collection.delete.call_args.kwargs["ids"]
    assert len(deleted_ids) == len(fake_chunks)


async def test_ingest_document_logs_error_when_chroma_cleanup_fails(tmp_path: Path) -> None:
    """Finding #5: if ChromaDB cleanup also fails, log error and re-raise original exception."""
    file_path = tmp_path / "contract.pdf"
    file_path.write_bytes(b"dummy content")

    fake_chunks = _make_chunks(1)
    fake_embedded = [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            text=c.text,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            source_file=c.source_file,
            section_header=c.section_header,
            embedding=[0.1] * 384,
        )
        for c in fake_chunks
    ]

    commit_call = 0

    async def flaky_commit():
        nonlocal commit_call
        commit_call += 1
        if commit_call >= 2:
            raise RuntimeError("Postgres commit failed")

    mock_session = _make_mock_session()
    mock_session.commit.side_effect = flaky_commit

    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(side_effect=RuntimeError("ChromaDB also down"))

    fake_section = MagicMock(
        document_id=fake_chunks[0].document_id,
        source_file="contract.pdf",
        page_number=1,
        section_header=None,
        text="some legal text",
        section_index=0,
    )

    with (
        patch("app.ingestion.embedder.parse_document", return_value=[fake_section]),
        patch("app.ingestion.embedder.chunk_sections", return_value=fake_chunks),
        patch("app.ingestion.embedder.embed_chunks", return_value=fake_embedded),
        patch("app.ingestion.embedder.store_in_chroma", new_callable=AsyncMock),
        patch("app.ingestion.embedder.chromadb", mock_chromadb),
    ):
        # The original Postgres exception must be re-raised, not the ChromaDB one
        with pytest.raises(RuntimeError, match="Postgres commit failed"):
            await ingest_document(file_path, mock_session)
