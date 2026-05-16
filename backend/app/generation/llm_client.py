import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from app.config import settings
from app.exceptions import GenerationError

logger = logging.getLogger(__name__)

_configured = False
_RETRY_DELAYS = (1.0, 2.0, 4.0)


def _ensure_configured() -> None:
    global _configured
    if not _configured:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True


def _build_gemini_request(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split standard chat-format messages into a Gemini system instruction and content list."""
    system_instruction: str | None = None
    contents: list[dict] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [content]})
    return system_instruction, contents


async def generate_answer(
    messages: list[dict],
    stream: bool = False,
) -> str | AsyncGenerator[str, None]:
    """Call Gemini and return the answer as a string or async stream.

    Args:
        messages: Standard chat-format message list (system + user at minimum).
        stream: If True, returns an AsyncGenerator yielding token deltas.

    Returns:
        Complete answer string when stream=False; AsyncGenerator of deltas when stream=True.

    Raises:
        GenerationError: On unrecoverable API failure after all retries.
    """
    if stream:
        return _stream_answer(messages)
    return await _blocking_answer(messages)


async def _blocking_answer(messages: list[dict]) -> str:
    _ensure_configured()
    system_instruction, contents = _build_gemini_request(messages)
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_instruction,
    )
    max_attempts = len(_RETRY_DELAYS) + 1

    for attempt in range(max_attempts):
        start = time.monotonic()
        try:
            response = await model.generate_content_async(contents)
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.info("llm model=%s latency_ms=%d", settings.gemini_model, latency_ms)
            return response.text
        except ResourceExhausted as exc:
            is_last = attempt == max_attempts - 1
            if is_last:
                raise GenerationError(str(exc)) from exc
            delay = _RETRY_DELAYS[attempt]
            logger.warning(
                "llm attempt=%d/%d error=%s retrying_in=%.1fs",
                attempt + 1,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            raise GenerationError(str(exc)) from exc

    raise GenerationError("exceeded max retries")


async def _stream_answer(messages: list[dict]) -> AsyncGenerator[str, None]:
    _ensure_configured()
    system_instruction, contents = _build_gemini_request(messages)
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_instruction,
    )
    start = time.monotonic()
    try:
        stream = await model.generate_content_async(contents, stream=True)
        async for chunk in stream:
            if chunk.text:
                yield chunk.text
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info("llm stream model=%s latency_ms=%d", settings.gemini_model, latency_ms)
    except Exception as exc:
        raise GenerationError(str(exc)) from exc
