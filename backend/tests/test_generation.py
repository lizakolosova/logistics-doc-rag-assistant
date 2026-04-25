from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.exceptions import GenerationError
from backend.app.generation.prompt_builder import build_messages, extract_citations
from backend.app.models.schemas import RetrievedChunk

from openai import RateLimitError
from backend.app.generation.llm_client import _RETRY_DELAYS, generate_answer


def _make_chunk(
    chunk_id: str = "c1",
    source_file: str = "contract.pdf",
    page_number: int = 1,
    section_header: str | None = None,
    text: str = "Sample chunk text.",
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=score,
        source_file=source_file,
        page_number=page_number,
        section_header=section_header,
    )


def _make_openai_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response

def test_build_messages_includes_all_chunks() -> None:
    chunks = [
        _make_chunk("c1", text="First clause text.", page_number=1),
        _make_chunk("c2", text="Second clause text.", page_number=2),
        _make_chunk("c3", text="Third clause text.", page_number=3),
    ]
    messages = build_messages("What are the obligations?", chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    assert "[1]" in user_content
    assert "[2]" in user_content
    assert "[3]" in user_content
    assert "First clause text." in user_content
    assert "Second clause text." in user_content
    assert "Third clause text." in user_content
    assert "What are the obligations?" in user_content


def test_build_messages_empty_chunks_returns_no_context_message() -> None:
    messages = build_messages("What is the penalty?", [])

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    combined = messages[0]["content"].lower() + messages[1]["content"].lower()
    assert "no context" in combined


def test_build_messages_truncates_long_chunks() -> None:
    long_text = "A" * 2000  # exceeds the 1500-char limit
    chunks = [_make_chunk("c1", text=long_text)]
    messages = build_messages("What does this say?", chunks)

    user_content = messages[1]["content"]
    assert "..." in user_content
    assert long_text not in user_content


def test_extract_citations_maps_correctly() -> None:
    chunks = [
        _make_chunk("c1", source_file="doc_a.pdf", page_number=3, section_header="Definitions"),
        _make_chunk("c2", source_file="doc_b.pdf", page_number=7, section_header=None),
    ]
    citations = extract_citations(chunks)

    assert len(citations) == 2

    assert citations[0].index == 1
    assert citations[0].source_file == "doc_a.pdf"
    assert citations[0].page_number == 3
    assert citations[0].section_header == "Definitions"
    assert citations[0].chunk_text == chunks[0].text

    assert citations[1].index == 2
    assert citations[1].source_file == "doc_b.pdf"
    assert citations[1].page_number == 7
    assert citations[1].section_header is None
    assert citations[1].chunk_text == chunks[1].text

async def test_generate_answer_returns_string() -> None:
    mock_response = _make_openai_response("The indemnity clause states that...")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("app.generation.llm_client._client", mock_client):
        result = await generate_answer([{"role": "user", "content": "What is the indemnity clause?"}])

    assert isinstance(result, str)
    assert result == "The indemnity clause states that..."
    mock_client.chat.completions.create.assert_called_once()


async def test_generate_answer_retries_on_rate_limit() -> None:

    mock_response = _make_openai_response("Answer after retry.")
    rate_limit_exc = RateLimitError("Rate limited", response=MagicMock(), body={})

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[rate_limit_exc, mock_response]
    )

    with patch("app.generation.llm_client._client", mock_client):
        with patch("app.generation.llm_client.asyncio.sleep", AsyncMock()):
            result = await generate_answer([{"role": "user", "content": "question"}])

    assert result == "Answer after retry."
    assert mock_client.chat.completions.create.call_count == 2


async def test_generate_answer_raises_after_max_retries() -> None:
    rate_limit_exc = RateLimitError("Rate limited", response=MagicMock(), body={})
    total_attempts = len(_RETRY_DELAYS) + 1

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[rate_limit_exc] * total_attempts
    )

    with patch("app.generation.llm_client._client", mock_client):
        with patch("app.generation.llm_client.asyncio.sleep", AsyncMock()):
            with pytest.raises(GenerationError):
                await generate_answer([{"role": "user", "content": "question"}])

    assert mock_client.chat.completions.create.call_count == total_attempts