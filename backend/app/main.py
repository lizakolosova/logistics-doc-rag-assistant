import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.exceptions import (
    AppError,
    DocumentParseError,
    DocumentTooLargeError,
    EmbeddingError,
    RetrievalError,
    UnsupportedFormatError,
)

from backend.app.models.database import Base, get_async_engine
from backend.app.api.routes_documents import router as documents_router
from backend.app.api.routes_eval import router as eval_router
from backend.app.api.routes_query import router as query_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")
    yield


app = FastAPI(
    title="Legal Doc RAG Assistant",
    description=(
        "Upload legal PDFs or DOCX files and ask questions. "
        "Answers include inline citations back to the source document."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

@app.exception_handler(DocumentTooLargeError)
async def document_too_large_handler(request: Request, exc: DocumentTooLargeError) -> JSONResponse:
    logger.warning("Document too large: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(UnsupportedFormatError)
async def unsupported_format_handler(request: Request, exc: UnsupportedFormatError) -> JSONResponse:
    logger.warning("Unsupported format: %s", exc)
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(DocumentParseError)
async def document_parse_handler(request: Request, exc: DocumentParseError) -> JSONResponse:
    logger.error("Parse error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    logger.error("Embedding error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError) -> JSONResponse:
    logger.error("Retrieval error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(AppError)
async def generic_app_error_handler( request: Request, exc: AppError) -> JSONResponse:
    logger.error("Unhandled application error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(eval_router)

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
