from typing import Literal

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
        description=(
            "One concise clarification question. Empty when clarification is unnecessary."
        ),
    )
    verification: str = Field(
        description=(
            "A concise acknowledgement before research begins. Empty when clarification is required."
        ),
    )


class ClarificationAnswer(FrozenBaseModel):
    """Answer to a clarification question."""

    answer: str


class ResearchQuestion(FrozenBaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ContentSummary(FrozenBaseModel):
    """Schema for webpage content summarization."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")


class ResearchTask(BaseModel):
    """Schema for structured research task generation."""

    id: str = Field(
        description="A unique identifier for the research task. Should be related to the research objective.",
    )
    objective: str = Field(
        description="A detailed research objective that outlines the goal of the research task.",
    )
    success_criteria: str = Field(
        description="The criteria that will be used to determine if the research objective was met.",
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending",
        description="The current status of the research task. Can be 'pending', 'in_progress', or 'completed'.",
    )


class ResearchPlan(FrozenBaseModel):
    """Schema for structured research plan generation."""

    tasks: list[ResearchTask] = Field(
        description="A list of research tasks that will be executed to achieve the research.",
    )


class ResearchPlanReview(FrozenBaseModel):
    """Schema for structured research plan review."""

    approved: bool
    feedback: str = ""
