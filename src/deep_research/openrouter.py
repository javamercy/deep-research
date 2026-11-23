"""OpenRouter chat and structured-output model factories."""

from typing import TYPE_CHECKING, Any, cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

from deep_research.configuration import LLMModelConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


def init_openrouter_model(
    llm_config: LLMModelConfig, session_id: str | None = None
) -> ChatOpenRouter:
    """Create a `ChatOpenRouter` instance from `llm_config`.

    Configures `max_completion_tokens` from `max_output_tokens`, routes via
    `openrouter_provider` with `ignored_providers` excluded and
    `require_parameters=True`, sets `reasoning={"effort": ...}` only when
    `reasoning_effort` is not `None`, disables internal client retries
    (`max_retries=0`), and forwards `session_id` only when provided.

    Args:
        llm_config: Per-role model and provider configuration.
        session_id: Optional session identifier used to tag OpenRouter requests.

    Returns:
        Configured `ChatOpenRouter` chat model.
    """
    reasoning = (
        {"effort": llm_config.reasoning_effort} if llm_config.reasoning_effort is not None else None
    )

    session_kwargs: Mapping[str, Any] = {"session_id": session_id} if session_id is not None else {}
    return ChatOpenRouter(
        model=llm_config.model.value,
        temperature=llm_config.temperature,
        max_completion_tokens=llm_config.max_output_tokens,
        openrouter_provider={"ignore": llm_config.ignored_providers, "require_parameters": True},
        reasoning=reasoning,
        max_retries=0,
        **session_kwargs,
    )


def init_openrouter_structured_model[OutputT: BaseModel](
    llm_config: LLMModelConfig,
    *,
    session_id: str | None = None,
    output_schema: type[OutputT],
    max_retries: int = 3,
) -> Runnable[LanguageModelInput, OutputT]:
    """Create a structured-output OpenRouter runnable with retry handling.

    Builds a `ChatOpenRouter` model via `init_openrouter_model`, wraps it with
    `with_structured_output(output_schema, method="json_schema",
    include_raw=False, strict=True)`, and applies `.with_retry` up to
    `max_retries` total attempts.

    Args:
        llm_config: Per-role model and provider configuration.
        session_id: Optional session identifier used to tag OpenRouter requests.
        output_schema: Pydantic model class enforced as the strict JSON schema.
        max_retries: Maximum total attempts for `.with_retry`.

    Returns:
        Runnable returning validated `output_schema` instances.
    """
    structured_model = (
        init_openrouter_model(llm_config, session_id=session_id)
        .with_structured_output(
            output_schema,
            method="json_schema",
            include_raw=False,
            strict=True,
        )
        .with_retry(stop_after_attempt=max_retries)
    )

    return cast("Runnable[LanguageModelInput, OutputT]", structured_model)
