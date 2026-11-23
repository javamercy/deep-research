"""Runtime configuration models, environment overrides, and session helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from deep_research.schemas import FrozenBaseModel

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt import ToolRuntime
    from langgraph.runtime import Runtime


class LLMModel(StrEnum):
    """Supported OpenRouter model identifiers."""

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
    """Model and provider parameters for a single agent role.

    Attributes:
        model: OpenRouter model identifier from `LLMModel`.
        max_output_tokens: Completion token limit passed as
            `max_completion_tokens`.
        temperature: Sampling temperature, or `None` to use the provider
            default.
        reasoning_effort: Reasoning effort level passed to OpenRouter when set,
            or `None` to omit explicit reasoning settings.
        ignored_providers: OpenRouter provider slugs to exclude during routing.
    """

    model: LLMModel
    max_output_tokens: int
    temperature: float | None = None
    reasoning_effort: str | None = None
    ignored_providers: list[str] = Field(default_factory=lambda: ["open-inference", "morph"])


def get_session_id(config: RunnableConfig) -> str:
    """Return the thread-scoped session ID from a `RunnableConfig`.

    Reads `configurable.thread_id` and converts it to `str` for scoping store
    namespaces and OpenRouter session tracking.

    Args:
        config: LangGraph runnable configuration containing
            `configurable.thread_id`.

    Raises:
        ValueError: If `thread_id` is missing or `None` in
            `config["configurable"]`.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id is None:
        raise ValueError("thread_id is not set in the configuration")

    return str(thread_id)


class Configuration(FrozenBaseModel):
    """Runtime configuration for the deep research graphs.

    When resolved via `from_runtime` or `_from_runnable_config`, values come
    from `runtime.context` (if already a `Configuration` instance), then
    `config["configurable"]`, then uppercased environment variables, and
    finally class defaults.

    Attributes:
        max_structured_output_retries: Retry attempts for structured LLM calls.
        allow_clarification: Whether scoping may pause to ask the user a
            clarifying question.
        allow_collaborative_planning: Whether scoping pauses for user plan
            review and revision.
        max_planning_iterations: Maximum planning revisions before proceeding.
        max_concurrent_research_threads: Concurrency limit advertised to the
            supervisor prompt.
        max_supervisor_iterations: Maximum supervisor delegation turns.
        max_researcher_tool_calls: Maximum tool calls allowed per researcher.
        max_report_reviewer_iterations: Maximum writer-reviewer revision cycles.
        max_content_length: Character limit for raw webpage content passed to
            summarization.
        scoping_llm_config: Model settings for clarification, brief, and plan
            nodes.
        researcher_llm_config: Model settings for `conduct_research`.
        summarization_llm_config: Model settings for webpage summarization in
            `tavily_search`.
        compression_llm_config: Model settings for `compress_research`.
        research_verifier_llm_config: Model settings for `verify_research`.
        supervisor_llm_config: Model settings for the `supervisor` node.
        writer_llm_config: Model settings for `write_report`.
        report_reviewer_llm_config: Model settings for `review_report`.
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
        """Populate omitted fields from uppercased environment variables."""
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
        """Build a `Configuration` from `config["configurable"]`, env vars, and defaults.

        Non-`None` entries in `config["configurable"]` take precedence over
        uppercased environment variables, which in turn override class defaults.
        """
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
        """Resolve `Configuration` from runtime context or `RunnableConfig`.

        Returns `runtime.context` directly when it is already a `Configuration`
        instance; otherwise falls back to `config["configurable"]`, uppercased
        environment variables, and class defaults.

        Args:
            runtime: LangGraph node or tool runtime carrying `context`.
            config: Optional runnable configuration used when `runtime.context`
                is absent or not a `Configuration`.
        """
        if runtime is not None and isinstance(runtime.context, cls):
            return runtime.context

        return cls._from_runnable_config(config or {})
