"""Tests for DELETE /documents/{id} behavior (Finding #4)."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from app.dependencies import get_session
from app.main import app
from app.models.database import Chunk, Document
from app.models.schemas import DocumentStatus


def _make_doc(document_id=None) -> Document:
    doc = MagicMock(spec=Document)
    doc.id = document_id or uuid4()
    doc.status = DocumentStatus.ready
    return doc


async def test_delete_document_returns_503_when_chroma_fails() -> None:
    """Finding #4: DELETE must return 503 and not remove Postgres record if ChromaDB delete fails."""
    document_id = uuid4()
    chroma_id = f"{document_id}_0"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=_make_doc(document_id))

    chroma_result = MagicMock()
    chroma_result.scalars.return_value.all.return_value = [chroma_id]
    mock_session.execute = AsyncMock(return_value=chroma_result)

    async def mock_session_dep():
        yield mock_session

    app.dependency_overrides[get_session] = mock_session_dep

    mock_collection = AsyncMock()
    mock_collection.delete = AsyncMock(side_effect=RuntimeError("ChromaDB unavailable"))
    mock_client = AsyncMock()
    mock_client.get_collection = AsyncMock(return_value=mock_collection)
    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(return_value=mock_client)

    try:
        with patch("app.api.routes_documents.chromadb", mock_chromadb):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.delete(f"/api/documents/{document_id}")

        assert response.status_code == 503
        assert "ChromaDB" in response.json()["detail"]

        # Postgres record must NOT have been deleted
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()
    finally:
        app.dependency_overrides.clear()


async def test_delete_document_succeeds_when_chroma_succeeds() -> None:
    """DELETE returns 204 and deletes Postgres record only after ChromaDB succeeds."""
    document_id = uuid4()
    chroma_id = f"{document_id}_0"

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=_make_doc(document_id))

    chroma_result = MagicMock()
    chroma_result.scalars.return_value.all.return_value = [chroma_id]
    mock_session.execute = AsyncMock(return_value=chroma_result)

    async def mock_session_dep():
        yield mock_session

    app.dependency_overrides[get_session] = mock_session_dep

    mock_collection = AsyncMock()
    mock_collection.delete = AsyncMock(return_value=None)
    mock_client = AsyncMock()
    mock_client.get_collection = AsyncMock(return_value=mock_collection)
    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(return_value=mock_client)

    try:
        with patch("app.api.routes_documents.chromadb", mock_chromadb):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.delete(f"/api/documents/{document_id}")

        assert response.status_code == 204

        # Postgres record must have been deleted after ChromaDB succeeded
        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()
    finally:
        app.dependency_overrides.clear()
