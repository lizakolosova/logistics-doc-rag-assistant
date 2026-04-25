import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "eval_data" / "legal_qa_golden.json"


class GoldenQA(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant_sources: list[dict]


def load_golden_dataset(path: Path | None = None) -> list[GoldenQA]:
    resolved = path if path is not None else _DEFAULT_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Golden dataset not found at '{resolved}'. "
            "Create eval_data/legal_qa_golden.json or pass an explicit path.")

    raw = json.loads(resolved.read_text(encoding="utf-8"))
    dataset = [GoldenQA(**item) for item in raw]
    logger.info("Loaded %d golden Q&A pairs from %s", len(dataset), resolved)
    return dataset