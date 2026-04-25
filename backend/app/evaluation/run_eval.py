import logging
import time
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.evaluation.eval_dataset import GoldenQA, load_golden_dataset
from app.evaluation.metrics import (
    EvalResult,
    compute_ragas_metrics,
    compute_retrieval_metrics,
)
from app.generation.llm_client import generate_answer
from app.generation.prompt_builder import build_messages
from app.models.database import EvaluationRun
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


async def _evaluate_single(  qa: GoldenQA,session: AsyncSession) -> EvalResult:
    start = time.monotonic()

    raw_results = await hybrid_search(qa.question,session, top_k=settings.retrieval_top_k)
    chunks = rerank(qa.question, raw_results, top_k=settings.rerank_top_k)

    messages = build_messages(qa.question, chunks)
    answer = await generate_answer(messages, stream=False)

    latency_ms = (time.monotonic() - start) * 1000

    retrieval_metrics = compute_retrieval_metrics(qa, chunks)
    ragas_metrics = compute_ragas_metrics(qa.question, answer, chunks)

    return EvalResult( question_id=qa.id, question=qa.question, expected_answer=qa.expected_answer,
        generated_answer=answer, retrieved_chunks=chunks, context_precision=retrieval_metrics["context_precision"],
        context_recall=retrieval_metrics["context_recall"], faithfulness=ragas_metrics["faithfulness"],
        answer_relevancy=ragas_metrics["answer_relevancy"], latency_ms=latency_ms)

async def run_evaluation(session: AsyncSession,golden_path: Path | None = None) -> list[EvalResult]:
    golden = load_golden_dataset(golden_path)
    logger.info("Starting evaluation run over %d questions", len(golden))

    results: list[EvalResult] = []
    for qa in golden:
        try:
            result = await _evaluate_single(qa, session)
            results.append(result)
            logger.info("eval q=%s precision=%.3f recall=%.3f latency=%.0fms",qa.id,
                result.context_precision, result.context_recall, result.latency_ms)
        except Exception as exc:
            logger.error("Evaluation failed for question '%s': %s", qa.id, exc)

    def _mean(values: list[float | None]) -> float | None:
        non_null = [v for v in values if v is not None]
        return sum(non_null) / len(non_null) if non_null else None

    summary: dict = {
        "context_precision": _mean([r.context_precision for r in results]),
        "context_recall": _mean([r.context_recall for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        "answer_relevancy": _mean([r.answer_relevancy for r in results]),
        "avg_latency_ms": _mean([r.latency_ms for r in results]),
        "num_questions": len(results),
    }

    config: dict = {
        "top_k": settings.rerank_top_k,
        "chunk_size": settings.chunk_size,
        "model": settings.openai_model,
    }

    eval_run = EvaluationRun(
        metrics={
            "summary": summary,
            "results": [r.model_dump() for r in results],
        },
        config=config,
    )
    session.add(eval_run)
    await session.commit()
    await session.refresh(eval_run)

    logger.info("Evaluation run %s stored: precision=%.3f recall=%.3f",eval_run.id,
        summary.get("context_precision") or 0.0,summary.get("context_recall") or 0.0)

    return results