from enum import StrEnum


class LLMModel(StrEnum):
    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash-0731"
    NEMOTRON_3_SUPER_FREE = "nvidia/nemotron-3-super-120b-a12b:free"
    GLM_5_3_FLASH = "z-ai/glm-5.3-flash"
