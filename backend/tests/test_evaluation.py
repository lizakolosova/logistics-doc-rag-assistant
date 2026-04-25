import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("POSTGRES_URL", "postgresql+asyncpg://test:test@localhost:5432/test")

import pytest

from backend.app.evaluation.eval_dataset import GoldenQA, _DEFAULT_PATH, load_golden_dataset
from backend.app.evaluation.metrics import compute_ragas_metrics, compute_retrieval_metrics
from backend.app.models.schemas import RetrievedChunk
from backend.app.evaluation.run_eval import run_evaluation

def _make_golden_qa(id: str = "q1",question: str = "What is the notice period?",expected_answer: str = "30 days.",
    relevant_sources: list[dict] | None = None) -> GoldenQA:
    return GoldenQA( id=id, question=question,expected_answer=expected_answer,
                     relevant_sources=relevant_sources or [{"file": "contract.pdf", "page": 4}])

def _make_chunk(source_file: str = "contract.pdf",page_number: int = 4,text: str = "The notice period is 30 days.",) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=str(uuid4()), document_id=str(uuid4()), text=text, score=0.9, source_file=source_file,
        page_number=page_number, section_header=None)

def test_load_golden_dataset_returns_correct_count() -> None:
    dataset = load_golden_dataset(_DEFAULT_PATH)
    assert len(dataset) == 5
    assert all(isinstance(q, GoldenQA) for q in dataset)


def test_load_golden_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Golden dataset not found"):
        load_golden_dataset(tmp_path / "nonexistent.json")

def test_compute_retrieval_metrics_perfect_retrieval() -> None:
    qa = _make_golden_qa(relevant_sources=[{"file": "contract.pdf", "page": 4}])
    chunks = [_make_chunk("contract.pdf"), _make_chunk("contract.pdf")]
    metrics = compute_retrieval_metrics(qa, chunks)
    assert metrics["context_precision"] == pytest.approx(1.0)
    assert metrics["context_recall"] == pytest.approx(1.0)


def test_compute_retrieval_metrics_no_overlap() -> None:
    qa = _make_golden_qa(relevant_sources=[{"file": "contract.pdf", "page": 4}])
    chunks = [_make_chunk("other_doc.pdf"), _make_chunk("another.pdf")]
    metrics = compute_retrieval_metrics(qa, chunks)
    assert metrics["context_precision"] == pytest.approx(0.0)
    assert metrics["context_recall"] == pytest.approx(0.0)


def test_compute_retrieval_metrics_partial_overlap() -> None:
    qa = _make_golden_qa(
        relevant_sources=[
            {"file": "contract.pdf", "page": 4},
            {"file": "annex.pdf", "page": 2},
        ]
    )
    chunks = [
        _make_chunk("contract.pdf"),
        _make_chunk("unrelated.pdf"),
    ]
    metrics = compute_retrieval_metrics(qa, chunks)
    assert metrics["context_precision"] == pytest.approx(0.5)
    assert metrics["context_recall"] == pytest.approx(0.5)


def test_ragas_metrics_returns_none_on_failure() -> None:
    with patch.dict(sys.modules, {"ragas": None, "ragas.metrics": None, "datasets": None}):
        result = compute_ragas_metrics(
            "What is the notice period?",
            "30 days.",
            [_make_chunk()],
        )
    assert result["faithfulness"] is None
    assert result["answer_relevancy"] is None

async def test_run_evaluation_stores_results(tmp_path: Path) -> None:
    golden_data = [
        {
            "id": "q1",
            "question": "What is the notice period?",
            "expected_answer": "30 days.",
            "relevant_sources": [{"file": "contract.pdf", "page": 4}],
        }
    ]
    golden_path = tmp_path / "test_golden.json"
    golden_path.write_text(json.dumps(golden_data), encoding="utf-8")

    mock_chunk = _make_chunk()
    mock_session = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    with (
        patch("app.evaluation.run_eval.hybrid_search", AsyncMock(return_value=[mock_chunk])),
        patch("app.evaluation.run_eval.rerank", return_value=[mock_chunk]),
        patch(
            "app.evaluation.run_eval.generate_answer",
            AsyncMock(return_value="The notice period is 30 days."),
        ),
        patch(
            "app.evaluation.run_eval.compute_ragas_metrics",
            return_value={"faithfulness": None, "answer_relevancy": None},
        ),
    ):
        results = await run_evaluation(mock_session, golden_path)

    assert len(results) == 1
    assert results[0].question_id == "q1"
    assert results[0].generated_answer == "The notice period is 30 days."
    assert results[0].context_precision == pytest.approx(1.0)
    assert results[0].context_recall == pytest.approx(1.0)
    assert results[0].faithfulness is None
    assert results[0].answer_relevancy is None
    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()