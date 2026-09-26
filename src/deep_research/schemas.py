from pydantic import BaseModel, ConfigDict, Field


class FrozenBaseModel(BaseModel):
    """Base model with frozen configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ClarificationDecision(FrozenBaseModel):
    """Decision about whether research requires user clarification."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        default="",
        description=(
            "One concise clarification question. Empty when clarification is unnecessary."
        ),
    )
    verification: str = Field(
        default="",
        description=(
            "A concise acknowledgement before research begins. Empty when clarification is required."
        ),
    )


class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )
