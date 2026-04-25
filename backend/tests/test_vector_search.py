import sys
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from  backend.app.config import settings
from  backend.app.exceptions import RetrievalError
from  backend.app.models.schemas import RetrievedChunk
from  backend.app.retrieval.vector_search import vector_search


def _make_openai_mock() -> tuple[MagicMock, AsyncMock]:
    mock_create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )
    mock_client = MagicMock()
    mock_client.embeddings.create = mock_create
    mock_openai_cls = MagicMock(return_value=mock_client)
    return mock_openai_cls, mock_create


def _make_chroma_mock(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> tuple[MagicMock, AsyncMock]:
    mock_collection = AsyncMock()
    mock_collection.query = AsyncMock(
        return_value={
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }
    )

    mock_chroma_client = AsyncMock()
    mock_chroma_client.get_or_create_collection = AsyncMock(return_value=mock_collection)

    mock_chromadb = MagicMock()
    mock_chromadb.AsyncHttpClient = AsyncMock(return_value=mock_chroma_client)

    return mock_chromadb, mock_collection


def _sample_metadatas(doc_id: str, count: int) -> list[dict]:
    return [
        {
            "document_id": doc_id,
            "source_file": "contract.pdf",
            "page_number": i + 1,
            "section_header": f"Section {i}",
        }
        for i in range(count)
    ]


async def test_vector_search_returns_ranked_results() -> None:
    doc_id = str(uuid4())
    mock_openai_cls, _ = _make_openai_mock()
    mock_chromadb, _ = _make_chroma_mock(
        ids=["doc_0", "doc_1", "doc_2"],
        documents=["text A", "text B", "text C"],
        metadatas=_sample_metadatas(doc_id, 3),
        distances=[0.1, 0.5, 0.9],
    )

    with patch("app.retrieval.vector_search.openai.AsyncOpenAI", mock_openai_cls):
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            results = await vector_search("what is the indemnity clause?", top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].score > results[1].score > results[2].score
    assert results[0].score == pytest.approx(0.9)
    assert results[0].chunk_id == "doc_0"


async def test_vector_search_filters_by_document_id() -> None:
    doc_id = str(uuid4())
    mock_openai_cls, _ = _make_openai_mock()
    mock_chromadb, mock_collection = _make_chroma_mock(
        ids=["doc_0"],
        documents=["relevant text"],
        metadatas=[
            {
                "document_id": doc_id,
                "source_file": "contract.pdf",
                "page_number": 1,
                "section_header": "",
            }
        ],
        distances=[0.2],
    )

    with patch("app.retrieval.vector_search.openai.AsyncOpenAI", mock_openai_cls):
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            results = await vector_search("indemnity", top_k=5, document_ids=[doc_id])

    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"document_id": {"$eq": doc_id}}
    assert len(results) == 1


async def test_vector_search_empty_collection_returns_empty() -> None:
    mock_openai_cls, _ = _make_openai_mock()
    mock_chromadb, _ = _make_chroma_mock(
        ids=[],
        documents=[],
        metadatas=[],
        distances=[],
    )

    with patch("app.retrieval.vector_search.openai.AsyncOpenAI", mock_openai_cls):
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            results = await vector_search("anything", top_k=10)

    assert results == []


async def test_vector_search_raises_on_chroma_failure() -> None:
    mock_openai_cls, _ = _make_openai_mock()
    mock_chromadb, mock_collection = _make_chroma_mock([], [], [], [])
    mock_collection.query.side_effect = Exception("connection refused")

    with patch("app.retrieval.vector_search.openai.AsyncOpenAI", mock_openai_cls):
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            with pytest.raises(RetrievalError, match="connection refused"):
                await vector_search("anything", top_k=5)


async def test_vector_search_defaults_to_settings_top_k() -> None:
    doc_id = str(uuid4())
    mock_openai_cls, _ = _make_openai_mock()
    mock_chromadb, mock_collection = _make_chroma_mock(
        ids=["doc_0"],
        documents=["text"],
        metadatas=[
            {
                "document_id": doc_id,
                "source_file": "file.pdf",
                "page_number": 1,
                "section_header": "",
            }
        ],
        distances=[0.3],
    )

    with patch("app.retrieval.vector_search.openai.AsyncOpenAI", mock_openai_cls):
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            await vector_search("test query")

    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["n_results"] == settings.retrieval_top_k
