from __future__ import annotations

import os
from enum import StrEnum

from langchain_core.runnables import RunnableConfig

from deep_research.schemas import FrozenBaseModel


class LLMModel(StrEnum):
    """Enum representing different LLM models available for use."""

    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash-0731:floor"  # effort: none, low, high (default), max
    DEEPSEEK_V4_1_FLASH = "deepseek/deepseek-v4.1-flash:floor"  # effort: none, low, high (default), max
    GLM_5_3_FLASH = "z-ai/glm-5.3-flash:floor"  # effort: low, high, max (default); reasoning mandatory


class LLMModelConfig(FrozenBaseModel):
    """Configuration for a specific LLM model."""

    model: LLMModel
    max_output_tokens: int
    temperature: float | None = None
    reasoning_effort: str | None = None
    ignored_providers: list[str] = ["open-inference", "morph"]


class Configuration(FrozenBaseModel):
    """Base configuration for the Deep Research framework.

    Attributes:
        session_id (str | None): Unique identifier for the session.
        max_structured_output_retries (int): Maximum number of retries for structured output.
        allow_clarification (bool): Whether to allow clarification during research.
        allow_collaborative_planning (bool): Whether to allow collaborative planning.
        max_planning_iterations (int): Maximum number of planning iterations.
        max_concurrent_research_threads (int): Maximum number of concurrent research threads.
        max_supervisor_iterations (int): Maximum number of supervisor iterations.
        max_researcher_tool_calls (int): Maximum number of tool calls for researchers.
        max_content_length (int): Maximum length of webpage content to be processed.
        scoping_llm_config (LLMModelConfig): Configuration for the scoping LLM model.
        research_llm_config (LLMModelConfig): Configuration for the research LLM model.
        summarization_llm_config (LLMModelConfig): Configuration for the summarization LLM model.
        compression_llm_config (LLMModelConfig): Configuration for the compression LLM model.
        supervisor_llm_config (LLMModelConfig): Configuration for the supervisor LLM model.
        writer_llm_config (LLMModelConfig): Configuration for the writer LLM model.
    """

    session_id: str | None = None
    max_structured_output_retries: int = 3
    allow_clarification: bool = True
    allow_collaborative_planning: bool = True
    max_planning_iterations: int = 3
    max_concurrent_research_threads: int = 5
    max_supervisor_iterations: int = 1
    max_researcher_tool_calls: int = 2
    max_content_length: int = 20_000

    scoping_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.GLM_5_3_FLASH,
        reasoning_effort="high",
        max_output_tokens=32_000,
    )
    research_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=16_000,
    )
    summarization_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=16_000,
    )
    compression_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=16_000,
    )
    supervisor_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=16_000,
    )
    writer_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=64_000,
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> Configuration:
        """Create a Configuration instance from a RunnableConfig.

        Args:
            config (RunnableConfig | None): The RunnableConfig instance to extract configuration from.

        Returns:
            Configuration: An instance of the Configuration class populated with values from the RunnableConfig.
        """
        configurable = config.get("configurable", {}) if config else {}
        field_names = cls.model_fields.keys()
        values = {
            field_name: os.getenv(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }

        thread_id = configurable.get("thread_id")
        if thread_id is not None:
            values["session_id"] = str(thread_id)

        return cls(**{key: value for key, value in values.items() if value is not None})
