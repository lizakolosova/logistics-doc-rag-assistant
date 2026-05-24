from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_session(request: Request) -> AsyncSession:
    """Yield a SQLAlchemy async session from the shared session factory in app.state."""
    async with request.app.state.session_factory() as session:
        yield session


async def get_bm25_index(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> tuple:
    """Return the cached BM25 index, rebuilding lazily when the dirty flag is set."""
    from app.retrieval.bm25_search import build_bm25_index

    state = request.app.state
    if state.bm25_dirty or state.bm25_cache is None:
        state.bm25_cache = await build_bm25_index(session)
        state.bm25_dirty = False
    return state.bm25_cache
