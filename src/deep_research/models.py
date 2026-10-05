from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel

from deep_research.configuration import LLMModelConfig


def init_openrouter_model(llm_config: LLMModelConfig) -> ChatOpenRouter:
    """Initialize a ChatOpenRouter model based on the provided LLMModelConfig."""

    reasoning = (
        {"effort": llm_config.reasoning_effort}
        if llm_config.reasoning_effort is not None
        else None
    )
    return ChatOpenRouter(
        model=llm_config.model.value,
        temperature=llm_config.temperature,
        max_completion_tokens=llm_config.max_output_tokens,
        openrouter_provider={"require_parameters": True},
        reasoning=reasoning,
        max_retries=0
    )


def init_openrouter_structured_model[OutputT: BaseModel](
        llm_config: LLMModelConfig,
        *,
        output_schema: type[OutputT],
        max_retries: int = 3,
) -> Runnable[LanguageModelInput, OutputT]:
    """Initialize a ChatOpenRouter model with structured output based on the provided Configuration and output schema."""

    structured_model = init_openrouter_model(llm_config).with_structured_output(
        output_schema,
        method="json_schema",
        include_raw=False,
        strict=True,
    ).with_retry(stop_after_attempt=max_retries)

    return cast(Runnable[LanguageModelInput, OutputT], structured_model)
