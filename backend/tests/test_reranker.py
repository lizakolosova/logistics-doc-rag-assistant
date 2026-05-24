import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.dependencies import get_bm25_index, get_session
from app.exceptions import RetrievalError
from app.models.schemas import RetrievedChunk
from app.retrieval.reranker import rerank
import httpx

from app.main import app


def _make_chunk(chunk_id: str, doc_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"text for {chunk_id}",
        score=score,
        source_file="contract.pdf",
        page_number=1,
        section_header=None,
    )


def _make_st_mock(scores: list[float]) -> tuple[MagicMock, MagicMock]:
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = scores
    mock_st = MagicMock()
    mock_st.CrossEncoder = MagicMock(return_value=mock_encoder)
    return mock_st, mock_encoder


async def test_rerank_returns_top_k() -> None:
    doc_id = "doc-1"
    chunks = [
        _make_chunk("c1", doc_id, 0.5),
        _make_chunk("c2", doc_id, 0.5),
        _make_chunk("c3", doc_id, 0.5),
    ]
    mock_st, _ = _make_st_mock([1.2, 3.5, 0.8])

    with patch("app.retrieval.reranker._cross_encoder", None):
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            result = await rerank("indemnity clause", chunks, top_k=2)

    assert len(result) == 2
    assert all(isinstance(r, RetrievedChunk) for r in result)
    assert result[0].chunk_id == "c2"  # highest reranker score 3.5
    assert result[1].chunk_id == "c1"  # second highest 1.2
    assert result[0].score > result[1].score


async def test_rerank_empty_input_returns_empty() -> None:
    result = await rerank("anything", [], top_k=5)

    assert result == []


async def test_rerank_scores_replace_original_scores() -> None:
    doc_id = "doc-1"
    chunks = [
        _make_chunk("c1", doc_id, 0.1),  # low original score
        _make_chunk("c2", doc_id, 0.9),  # high original score
    ]
    mock_st, _ = _make_st_mock([7.5, 2.3])  # reranker inverts the ranking

    with patch("app.retrieval.reranker._cross_encoder", None):
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            result = await rerank("clause", chunks, top_k=2)

    assert result[0].chunk_id == "c1"  # flipped: reranker preferred c1
    assert result[1].chunk_id == "c2"
    assert result[0].score == pytest.approx(7.5)
    assert result[1].score == pytest.approx(2.3)


async def test_rerank_raises_on_model_failure() -> None:
    doc_id = "doc-1"
    chunks = [_make_chunk("c1", doc_id)]

    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("CUDA out of memory")
    mock_st = MagicMock()
    mock_st.CrossEncoder = MagicMock(return_value=mock_encoder)

    with patch("app.retrieval.reranker._cross_encoder", None):
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            with pytest.raises(RetrievalError, match="CUDA out of memory"):
                await rerank("test query", chunks)


async def test_query_endpoint_full_pipeline() -> None:
    doc_id = "doc-1"
    hybrid_chunks = [_make_chunk(f"c{i}", doc_id, 0.9 - i * 0.1) for i in range(1, 5)]
    reranked = [_make_chunk("c2", doc_id, 9.1), _make_chunk("c1", doc_id, 7.2)]
    fake_bm25 = (MagicMock(), [])

    async def mock_session_dep():
        yield AsyncMock()

    async def mock_bm25_dep():
        return fake_bm25

    app.dependency_overrides[get_session] = mock_session_dep
    app.dependency_overrides[get_bm25_index] = mock_bm25_dep

    mock_hybrid = AsyncMock(return_value=hybrid_chunks)
    mock_rerank = AsyncMock(return_value=reranked)
    mock_generate = AsyncMock(return_value="The indemnity clause provides...")
    mock_build = MagicMock(return_value=[{"role": "user", "content": "..."}])
    mock_citations = MagicMock(return_value=[])

    try:
        with patch("app.api.routes_query.hybrid_search", mock_hybrid):
            with patch("app.api.routes_query.rerank", mock_rerank):
                with patch("app.api.routes_query.generate_answer", mock_generate):
                    with patch("app.api.routes_query.build_messages", mock_build):
                        with patch("app.api.routes_query.extract_citations", mock_citations):
                            async with httpx.AsyncClient(
                                transport=httpx.ASGITransport(app=app),
                                base_url="http://testserver",
                            ) as client:
                                response = await client.post(
                                    "/api/query",
                                    json={
                                        "question": "what is the indemnity clause?",
                                        "top_k": 2,
                                        "use_reranker": True,
                                    },
                                )

        assert response.status_code == 200
        data = response.json()
        assert data["retrieval_method"] == "hybrid+reranker"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "c2"
        assert "latency_ms" in data
        assert "answer" in data
        assert "citations" in data

        mock_hybrid.assert_called_once()
        assert mock_hybrid.call_args.kwargs.get("top_k") == 4  # top_k * 2

        mock_rerank.assert_called_once()
        assert mock_rerank.call_args.kwargs.get("top_k") == 2
    finally:
        app.dependency_overrides.clear()
