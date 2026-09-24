from enum import StrEnum
from typing import assert_never

from langchain_core.language_models import BaseChatModel
from langchain_openrouter import ChatOpenRouter

from deep_research.configuration import Configuration, LLMModelConfig


class ModelRole(StrEnum):
    """Model responsibilities within the research workflow."""

    SCOPING = "scoping"
    SUPERVISOR = "supervisor"
    RESEARCH = "research"
    COMPRESSION = "compression"
    REPORT = "report"


def init_model(role: ModelRole, configuration: Configuration) -> BaseChatModel:
    """Return the configured chat model for a workflow role."""

    model_config = _model_config(role, configuration)
    return ChatOpenRouter(
        model_name=model_config.model_name,
        max_completion_tokens=model_config.max_output_tokens,
        temperature=model_config.temperature
    )


def _model_config(role: ModelRole, configuration: Configuration) -> LLMModelConfig:
    """Return the model configuration for a given role."""

    match role:
        case ModelRole.SCOPING:
            return configuration.scoping_model_config
        case ModelRole.SUPERVISOR:
            return configuration.supervisor_model_config
        case ModelRole.RESEARCH:
            return configuration.research_model_config
        case ModelRole.COMPRESSION:
            return configuration.compression_model_config
        case ModelRole.REPORT:
            return configuration.report_model_config
        case _:
            assert_never(role)
