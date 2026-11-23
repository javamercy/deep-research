from __future__ import annotations

import os
from enum import StrEnum
from typing import Mapping, TYPE_CHECKING

from pydantic import model_validator, Field

from deep_research.schemas import FrozenBaseModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt import ToolRuntime
    from langgraph.runtime import Runtime


class LLMModel(StrEnum):
    """Enum representing different LLM models available for use."""

    DEEPSEEK_V4_FLASH = (
        "deepseek/deepseek-v4-flash-0731:floor"  # effort: none, low, high (default), max
    )
    DEEPSEEK_V4_1_FLASH = (
        "deepseek/deepseek-v4.1-flash-1017:floor"  # effort: none, low, high (default), max
    )
    GLM_5_3_FLASH = (
        "z-ai/glm-5.3-flash:floor"  # effort: low, high, max (default); reasoning mandatory
    )


class LLMModelConfig(FrozenBaseModel):
    """Configuration for a specific LLM model."""

    model: LLMModel
    max_output_tokens: int
    temperature: float | None = None
    reasoning_effort: str | None = None
    ignored_providers: list[str] = Field(default_factory=lambda: ["open-inference", "morph"])


def get_session_id(config: RunnableConfig) -> str:
    """Extract the thread-scoped session ID from a RunnableConfig.
    Args:
        config: The RunnableConfig containing `configurable.thread_id`.
    Returns:
        The stringified `thread_id` used to scope store namespaces and OpenRouter sessions.
    Raises:
        ValueError: If `thread_id` is missing from `config["configurable"]`.
    """

    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id is None:
        raise ValueError("thread_id is not set in the configuration")

    return str(thread_id)


class Configuration(FrozenBaseModel):
    """Base configuration for the Deep Research framework.

    Attributes:
        max_structured_output_retries (int): Maximum number of retries for structured output.
        allow_clarification (bool): Whether to allow clarification during research.
        allow_collaborative_planning (bool): Whether to allow collaborative planning.
        max_planning_iterations (int): Maximum number of planning iterations.
        max_concurrent_research_threads (int): Maximum number of concurrent research threads.
        max_supervisor_iterations (int): Maximum number of supervisor iterations.
        max_researcher_tool_calls (int): Maximum number of tool calls for researchers.
        max_content_length (int): Maximum length of webpage content to be processed.
        scoping_llm_config (LLMModelConfig): Configuration for the scoping LLM model.
        researcher_llm_config (LLMModelConfig): Configuration for the researcher LLM model.
        summarization_llm_config (LLMModelConfig): Configuration for the summarization LLM model.
        compression_llm_config (LLMModelConfig): Configuration for the compression LLM model.
        research_verifier_llm_config (LLMModelConfig): Configuration for the verifier LLM model.
        supervisor_llm_config (LLMModelConfig): Configuration for the supervisor LLM model.
        writer_llm_config (LLMModelConfig): Configuration for the writer LLM model.
    """

    max_structured_output_retries: int = 3
    allow_clarification: bool = True
    allow_collaborative_planning: bool = True
    max_planning_iterations: int = 3
    max_concurrent_research_threads: int = 5
    max_supervisor_iterations: int = 5
    max_researcher_tool_calls: int = 3
    max_report_reviewer_iterations: int = 3
    max_content_length: int = 20_000

    scoping_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=32_000,
    )
    researcher_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=32_000,
    )
    summarization_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=32_000,
    )
    compression_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=64_000,
    )
    research_verifier_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=64_000,
    )
    supervisor_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=32_000,
    )
    writer_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=64_000,
    )
    report_reviewer_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_1_FLASH,
        reasoning_effort="high",
        max_output_tokens=64_000,
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_env_defaults(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data

        values = dict(data)
        for field_name in cls.model_fields:
            if field_name not in values:
                env_value = os.getenv(field_name.upper())
                if env_value is not None:
                    values[field_name] = env_value

        return values

    @classmethod
    def _from_runnable_config(cls, config: RunnableConfig) -> Configuration:
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {})
        values = {
            field_name: configurable[field_name]
            for field_name in cls.model_fields
            if field_name in configurable and configurable[field_name] is not None
        }
        return cls(**values)

    @classmethod
    def from_runtime(
            cls,
            runtime: Runtime[Configuration] | ToolRuntime[Configuration] | None,
            config: RunnableConfig | None = None,
    ) -> Configuration:
        """Resolve Configuration from LangGraph `Runtime.context`, falling back to `RunnableConfig`."""
        if runtime is not None and isinstance(runtime.context, cls):
            return runtime.context

        return cls._from_runnable_config(config or {})
