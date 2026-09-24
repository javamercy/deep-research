from pydantic import BaseModel, ConfigDict, Field


class LLMModelConfig(BaseModel):
    model_name: str
    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=1.0)


class Configuration(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    scoping_model_config: LLMModelConfig = LLMModelConfig(
        model_name="deepseek/deepseek-v4-flash-0731",
        max_output_tokens=8_192,
        temperature=0.2,
    )
    supervisor_model_config: LLMModelConfig = LLMModelConfig(
        model_name="deepseek/deepseek-v4-flash-0731",
        max_output_tokens=64_000,
        temperature=0.2,
    )
    research_model_config: LLMModelConfig = LLMModelConfig(
        model_name="deepseek/deepseek-v4-flash-0731",
        max_output_tokens=64_000,
        temperature=0.2,
    )
    compression_model_config: LLMModelConfig = LLMModelConfig(
        model_name="deepseek/deepseek-v4-flash-0731",
        max_output_tokens=64_000,
        temperature=0.2,
    )
    report_model_config: LLMModelConfig = LLMModelConfig(
        model_name="deepseek/deepseek-v4-flash-0731",
        max_output_tokens=64_000,
        temperature=0.2,
    )
