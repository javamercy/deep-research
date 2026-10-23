from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Base
# =============================================================================

class FrozenBaseModel(BaseModel):
    """Base model with frozen configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# =============================================================================
# Scoping
# =============================================================================

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

    answer: str = Field(
        description="The user's answer to the clarification question.",
    )


class ResearchQuestion(FrozenBaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ResearchTask(FrozenBaseModel):
    """Schema for structured research task generation."""

    id: str = Field(
        description="A unique identifier for the research task. Should be related to the research objective.",
    )
    objective: str = Field(
        description="A detailed research objective that outlines the goal of the research task.",
    )
    success_criteria: str = Field(
        description="The criteria that will be used to determine if the research objective was met. Should be verifiable and evidence-based, not necessarily quantitative.",
    )
    depends_on: list[str] = Field(
        description="A list of task IDs that this task depends on to be completed. Empty list if there are no dependencies.",
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

    approved: bool = Field(
        description="Whether the research plan is approved or not.",
    )
    feedback: str = Field(
        default="",
        description="Feedback on the research plan. Empty if the plan is approved.",
    )


# =============================================================================
# Search and Sources
# =============================================================================

class ContentSummary(FrozenBaseModel):
    """Schema for webpage content summarization."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")


class TavilySearchResult(FrozenBaseModel):
    """Schema for a single search result from Tavily API."""

    id: str
    url: str
    title: str
    content: str
    score: float
    raw_content: str | None = None


class ProcessedSearchResult(FrozenBaseModel):
    """Schema for processed search result with summary and key excerpts."""

    id: str
    url: str
    title: str
    summary: str
    key_excerpts: str | None = None


# =============================================================================
# Research
# =============================================================================


class ResearchFinding(FrozenBaseModel):
    """Schema for single factual research finding attributed to source IDs."""

    statement: str = Field(
        description=(
            "A comprehensive factual statement or finding, preserving exact numbers, "
            "dates, names, quotes, technical details, qualifications and disagreements. "
            "Do NOT include inline citation brackets like [1] or raw URLs in this text; "
            "citations are attached automatically from source_ids."
        )
    )

    source_ids: list[str] = Field(
        description="List of 8-character source IDs (from <source id='...'>) that directly substantiate this statement."
    )


class ResearchSection(FrozenBaseModel):
    """Schema for thematic section grouping related research findings."""

    heading: str = Field(
        description="Descriptive heading for this thematic group of research findings."
    )
    findings: list[ResearchFinding] = Field(
        description="Comprehensive list of evidence-backed findings under this heading."
    )


class DraftResearch(FrozenBaseModel):
    """Schema for structured research compression before citation and evidence verification."""

    queries_and_tool_calls: list[str] = Field(
        description="List of search queries and external retrieval tool calls executed during the research conversation."
    )
    sections: list[ResearchSection] = Field(
        description="Thematic sections organizing all relevant research findings and their source IDs."
    )


class VerifiedResearch(FrozenBaseModel):
    """Schema for verified and source-validated research findings."""

    sections: list[ResearchSection] = Field(
        description="Verified thematic sections containing only source-substantiated findings with validated source IDs."
    )
