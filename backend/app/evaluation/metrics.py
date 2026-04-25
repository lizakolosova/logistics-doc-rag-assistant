import logging
from typing import Any

from pydantic import BaseModel

from app.evaluation.eval_dataset import GoldenQA
from app.models.schemas import RetrievedChunk
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    retrieved_chunks: list[RetrievedChunk]
    context_precision: float
    context_recall: float
    faithfulness: float | None
    answer_relevancy: float | None
    latency_ms: float


def compute_retrieval_metrics(question: GoldenQA,retrieved_chunks: list[RetrievedChunk],) -> dict[str, float]:
    if not retrieved_chunks:
        return {"context_precision": 0.0, "context_recall": 0.0}

    relevant_files = {s["file"] for s in question.relevant_sources}

    matching_retrieved = sum(
        1 for chunk in retrieved_chunks if chunk.source_file in relevant_files
    )
    context_precision = matching_retrieved / len(retrieved_chunks)

    retrieved_files = {chunk.source_file for chunk in retrieved_chunks}
    covered_sources = sum(
        1 for s in question.relevant_sources if s["file"] in retrieved_files
    )
    context_recall = covered_sources / len(question.relevant_sources)

    return {"context_precision": context_precision, "context_recall": context_recall}


def compute_ragas_metrics(question: str,answer: str,chunks: list[RetrievedChunk],) -> dict[str, float | None]:
    try:
        contexts = [chunk.text for chunk in chunks]
        data: dict[str, list[Any]] = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        result_row = result.to_pandas().iloc[0].to_dict()

        return {
            "faithfulness": float(result_row.get("faithfulness", 0.0)),
            "answer_relevancy": float(result_row.get("answer_relevancy", 0.0)),
        }
    except Exception as exc:
        logger.warning("RAGAS evaluation failed, returning None scores: %s", exc)
        return {"faithfulness": None, "answer_relevancy": None}
