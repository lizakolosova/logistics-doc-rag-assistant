import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.models.schemas import RetrievedChunk
from  backend.app.retrieval.bm25_search import bm25_search
from  backend.app.retrieval.hybrid import hybrid_search, reciprocal_rank_fusion

def _make_chunk(chroma_id: str, doc_id: str, page: int = 1) -> MagicMock:
    chunk = MagicMock()
    chunk.chroma_id = chroma_id
    chunk.document_id = uuid.UUID(doc_id)
    chunk.text = f"text for {chroma_id}"
    chunk.page_number = page
    chunk.section_header = None
    mock_doc = MagicMock()
    mock_doc.filename = "contract.pdf"
    chunk.document = mock_doc
    return chunk


def _make_retrieved(chunk_id: str, doc_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"text for {chunk_id}",
        score=score,
        source_file="contract.pdf",
        page_number=1,
        section_header=None,
    )


def _mock_bm25(scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.get_scores.return_value.tolist.return_value = list(scores)
    return mock

async def test_bm25_search_returns_ranked_results() -> None:
    doc_id = str(uuid.uuid4())
    chunks = [
        _make_chunk("c1", doc_id),
        _make_chunk("c2", doc_id),
        _make_chunk("c3", doc_id),
    ]
    bm25 = _mock_bm25([0.8, 0.0, 1.2])

    results = bm25_search("indemnity clause", chunks, bm25, top_k=3)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].chunk_id == "c3"
    assert results[1].chunk_id == "c1"
    assert results[0].score >= results[1].score


async def test_bm25_search_filters_by_document_id() -> None:
    doc1 = str(uuid.uuid4())
    doc2 = str(uuid.uuid4())
    chunks = [
        _make_chunk("c1", doc1),
        _make_chunk("c2", doc2),
        _make_chunk("c3", doc1),
    ]
    bm25 = _mock_bm25([0.8, 0.9, 0.7])

    results = bm25_search("indemnity", chunks, bm25, top_k=5, document_ids=[doc1])

    assert len(results) == 2
    assert all(r.document_id == doc1 for r in results)
    assert {r.chunk_id for r in results} == {"c1", "c3"}


async def test_bm25_search_normalizes_scores() -> None:
    doc_id = str(uuid.uuid4())
    chunks = [
        _make_chunk("c1", doc_id),
        _make_chunk("c2", doc_id),
    ]
    # c1 matches, c2 does not
    bm25 = _mock_bm25([0.6, 0.0])

    results = bm25_search("indemnity", chunks, bm25, top_k=2)

    assert len(results) == 1
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # top result is divided by itself → 1.0
    assert results[0].score == pytest.approx(1.0)


async def test_bm25_search_empty_index_returns_empty() -> None:
    results = bm25_search("anything", [], MagicMock(), top_k=5)

    assert results == []

def test_rrf_merges_and_scores_correctly() -> None:
    doc_id = str(uuid.uuid4())
    vector_results = [
        _make_retrieved("c1", doc_id, 0.9),
        _make_retrieved("c2", doc_id, 0.7),
    ]
    bm25_results = [
        _make_retrieved("c3", doc_id, 1.0),
        _make_retrieved("c1", doc_id, 0.8),
    ]

    merged = reciprocal_rank_fusion(vector_results, bm25_results)

    # c1: 1/61 + 1/62 is around 0.0325, c3: 1/61 is around 0.0164, c2: 1/62 is around 0.0161
    assert len(merged) == 3
    assert merged[0].chunk_id == "c1"
    assert {r.chunk_id for r in merged} == {"c1", "c2", "c3"}


def test_rrf_handles_disjoint_results() -> None:
    doc_id = str(uuid.uuid4())
    vector_results = [
        _make_retrieved("c1", doc_id, 0.9),
        _make_retrieved("c2", doc_id, 0.8),
    ]
    bm25_results = [
        _make_retrieved("c3", doc_id, 0.9),
        _make_retrieved("c4", doc_id, 0.8),
    ]

    merged = reciprocal_rank_fusion(vector_results, bm25_results)

    assert len(merged) == 4
    assert {r.chunk_id for r in merged} == {"c1", "c2", "c3", "c4"}

async def test_hybrid_search_returns_merged_results() -> None:
    doc_id = str(uuid.uuid4())
    vector_results = [
        _make_retrieved("c1", doc_id, 0.9),
        _make_retrieved("c2", doc_id, 0.8),
    ]
    bm25_results = [
        _make_retrieved("c2", doc_id, 1.0),
        _make_retrieved("c3", doc_id, 0.7),
    ]

    mock_session = AsyncMock()

    with patch("app.retrieval.hybrid.vector_search", AsyncMock(return_value=vector_results)):
        with patch("app.retrieval.hybrid.bm25_search", return_value=bm25_results):
            results = await hybrid_search(
                "indemnity",
                session=mock_session,
                top_k=5,
                bm25_index=(MagicMock(), []),
            )

    assert results[0].chunk_id == "c2"
    assert {r.chunk_id for r in results} == {"c1", "c2", "c3"}


async def test_hybrid_search_document_filter_propagates() -> None:
    doc_id = str(uuid.uuid4())
    mock_result = [_make_retrieved("c1", doc_id, 0.9)]
    target_doc_ids = [doc_id]

    mock_vector = AsyncMock(return_value=mock_result)
    mock_bm25 = MagicMock(return_value=mock_result)
    mock_session = AsyncMock()

    with patch("app.retrieval.hybrid.vector_search", mock_vector):
        with patch("app.retrieval.hybrid.bm25_search", mock_bm25):
            await hybrid_search(
                "indemnity",
                session=mock_session,
                top_k=5,
                document_ids=target_doc_ids,
                bm25_index=(MagicMock(), []),
            )

    mock_vector.assert_called_once()
    assert mock_vector.call_args.kwargs.get("document_ids") == target_doc_ids

    mock_bm25.assert_called_once()
    assert mock_bm25.call_args.kwargs.get("document_ids") == target_doc_ids