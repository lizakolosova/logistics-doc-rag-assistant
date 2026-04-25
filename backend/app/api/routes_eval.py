import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.metrics import EvalResult
from app.evaluation.run_eval import run_evaluation
from app.models.database import EvaluationRun, get_async_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


class EvalRunSummaryResponse(BaseModel):
    run_id: uuid.UUID
    created_at: datetime
    summary: dict
    config: dict


class EvalRunListResponse(BaseModel):
    runs: list[EvalRunSummaryResponse]


class EvalRunDetailResponse(BaseModel):
    run_id: uuid.UUID
    created_at: datetime
    summary: dict
    config: dict
    results: list[EvalResult]


async def _get_session() -> AsyncSession:
    session_factory = get_async_session()
    async with session_factory() as session:
        yield session


@router.post("/run", response_model=EvalRunSummaryResponse)
async def trigger_evaluation(
    session: AsyncSession = Depends(_get_session),
) -> EvalRunSummaryResponse:
    await run_evaluation(session)

    stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(1)
    latest_run = (await session.execute(stmt)).scalar_one_or_none()
    if latest_run is None:
        raise HTTPException(status_code=500, detail="Evaluation run was not stored.")

    metrics_blob = latest_run.metrics or {}
    return EvalRunSummaryResponse(
        run_id=latest_run.id,
        created_at=latest_run.created_at,
        summary=metrics_blob.get("summary", {}),
        config=latest_run.config or {},
    )


@router.get("/results", response_model=EvalRunListResponse)
async def list_evaluation_runs(
    session: AsyncSession = Depends(_get_session),
) -> EvalRunListResponse:
    stmt = select(EvaluationRun).order_by(EvaluationRun.created_at.desc())
    rows = (await session.execute(stmt)).scalars().all()

    runs = [
        EvalRunSummaryResponse(
            run_id=row.id,
            created_at=row.created_at,
            summary=(row.metrics or {}).get("summary", {}),
            config=row.config or {},
        )
        for row in rows
    ]
    return EvalRunListResponse(runs=runs)


@router.get("/results/{run_id}", response_model=EvalRunDetailResponse)
async def get_evaluation_run(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(_get_session),
) -> EvalRunDetailResponse:
    stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Evaluation run '{run_id}' not found.")

    metrics_blob = row.metrics or {}
    raw_results = metrics_blob.get("results", [])
    results = [EvalResult.model_validate(r) for r in raw_results]

    return EvalRunDetailResponse(
        run_id=row.id,
        created_at=row.created_at,
        summary=metrics_blob.get("summary", {}),
        config=row.config or {},
        results=results,
    )