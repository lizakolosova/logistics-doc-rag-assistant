import asyncio
import logging
import time
from typing import Any

from app.config import settings
from app.exceptions import RetrievalError
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_cross_encoder: Any = None


def _get_cross_encoder() -> Any:
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(settings.reranker_model)
    return _cross_encoder


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Re-score and truncate chunks using a cross-encoder model.

    Args:
        query: The user's question.
        chunks: Candidate chunks from hybrid retrieval.
        top_k: Number of chunks to return; defaults to settings.rerank_top_k.

    Returns:
        Top-k chunks sorted by cross-encoder score descending.
    """
    if not chunks:
        return []

    k = top_k if top_k is not None else settings.rerank_top_k
    start = time.monotonic()

    try:
        cross_encoder = _get_cross_encoder()
        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = await asyncio.to_thread(cross_encoder.predict, pairs)
        scores = [float(s) for s in raw_scores]
    except Exception as exc:
        raise RetrievalError(str(exc)) from exc

    scored = [
        chunk.model_copy(update={"score": score})
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda c: c.score, reverse=True)
    result = scored[:k]

    elapsed_ms = int((time.monotonic() - start) * 1000)
    score_min = min(c.score for c in result) if result else 0.0
    score_max = max(c.score for c in result) if result else 0.0
    logger.info(
        "rerank query='%.50s' input=%d output=%d score_range=[%.4f, %.4f] elapsed=%dms",
        query,
        len(chunks),
        len(result),
        score_min,
        score_max,
        elapsed_ms,
    )

    return result
