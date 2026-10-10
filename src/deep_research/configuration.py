from __future__ import annotations

import os
from enum import StrEnum

from langchain_core.runnables import RunnableConfig

from deep_research.schemas import FrozenBaseModel


class SearchAPI(StrEnum):
    TAVILY = "tavily"
    OPENROUTER = "openrouter"
    NONE = "none"


class LLMModel(StrEnum):
    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash-0731:floor"
    DEEPSEEK_V4_1_FLASH = "deepseek/deepseek-v4.1-flash:floor"
    NEMOTRON_3_SUPER_FREE = "nvidia/nemotron-3-super-120b-a12b:free"
    GLM_5_3_FLASH = "z-ai/glm-5.3-flash:floor"


class LLMModelConfig(FrozenBaseModel):
    model: LLMModel
    max_output_tokens: int
    temperature: float | None = None
    reasoning_effort: str | None = None
    ignored_providers: list[str] = ["open-inference"]


class Configuration(FrozenBaseModel):
    session_id: str | None = None

    max_structured_output_retries: int = 3

    allow_clarification: bool = True

    allow_collaborative_planning: bool = True

    max_planning_iterations: int = 3

    max_concurrent_research_threads: int = 5

    search_api: SearchAPI = SearchAPI.TAVILY

    max_researcher_iterations: int = 5

    max_research_tool_calls: int = 5

    max_content_length: int = 50000

    scoping_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    research_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    summarization_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    compression_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    supervisor_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=8192,
    )

    writer_llm_config: LLMModelConfig = LLMModelConfig(
        model=LLMModel.DEEPSEEK_V4_FLASH,
        max_output_tokens=64000,
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> Configuration:
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
