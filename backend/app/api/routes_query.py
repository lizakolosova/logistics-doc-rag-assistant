import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_bm25_index, get_session
from app.generation.llm_client import generate_answer
from app.generation.prompt_builder import build_messages, extract_citations
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
    bm25_index: tuple = Depends(get_bm25_index),
) -> QueryResponse:
    """Run the full RAG pipeline and return a structured answer with citations."""
    start = time.monotonic()
    k = body.top_k

    results = await hybrid_search(
        body.question,
        session,
        top_k=k * 2,
        document_ids=body.document_ids,
        bm25_index=bm25_index,
    )

    if body.use_reranker:
        chunks = await rerank(body.question, results, top_k=k)
        retrieval_method = "hybrid+reranker"
    else:
        chunks = results[:k]
        retrieval_method = "hybrid"

    messages = build_messages(body.question, chunks)
    answer = await generate_answer(messages, stream=False)
    citations = extract_citations(chunks)

    latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "query question='%.50s' method=%s chunks=%d elapsed=%.1fms",
        body.question,
        retrieval_method,
        len(chunks),
        latency_ms,
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        chunks=chunks,
        retrieval_method=retrieval_method,
        latency_ms=latency_ms,
    )


@router.post("/stream")
async def stream_query_documents(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
    bm25_index: tuple = Depends(get_bm25_index),
) -> StreamingResponse:
    """Stream the LLM answer token-by-token as newline-delimited JSON, then emit citations."""
    k = body.top_k

    results = await hybrid_search(
        body.question,
        session,
        top_k=k * 2,
        document_ids=body.document_ids,
        bm25_index=bm25_index,
    )

    if body.use_reranker:
        chunks = await rerank(body.question, results, top_k=k)
    else:
        chunks = results[:k]

    messages = build_messages(body.question, chunks)
    citations = extract_citations(chunks)
    answer_stream = await generate_answer(messages, stream=True)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for delta in answer_stream:
            yield json.dumps({"delta": delta}) + "\n"
        yield json.dumps({"done": True, "citations": [c.model_dump() for c in citations]}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
