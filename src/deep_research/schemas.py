from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

# =============================================================================
# Base
# =============================================================================


class FrozenBaseModel(BaseModel):
    """Immutable Pydantic base model that forbids extra fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# =============================================================================
# Scoping
# =============================================================================


class ClarificationDecision(FrozenBaseModel):
    """Decision on whether to ask the user a clarifying question or begin research.

    Set `need_clarification` to true with a concise `question` and an empty
    `verification` when essential details or ambiguous terms must be resolved,
    or false with an empty `question` and a brief `verification` when ready
    to proceed.
    """

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
    """Human or caller resume payload answering a scoping clarification interrupt.

    Passed as `response_schema` to `interrupt()` in `ask_clarification`.
    """

    answer: str = Field(
        description="The user's answer to the clarification question.",
    )


class ResearchQuestion(FrozenBaseModel):
    """Detailed, self-contained research question guiding downstream investigation.

    Synthesize the user's explicit requirements, constraints, and key
    dimensions from the conversation without inventing unstated preferences.
    """

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class ResearchTask(FrozenBaseModel):
    """Single investigative task within a research plan.

    Define a distinct, non-overlapping investigation goal with verifiable
    success criteria and only genuine prerequisite dependencies in `depends_on`.
    """

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
    """Structured set of investigative tasks decomposing the research question."""

    tasks: list[ResearchTask] = Field(
        description="A list of research tasks that will be executed to achieve the research.",
    )


class ResearchPlanReview(FrozenBaseModel):
    """Human or caller resume payload reviewing a proposed research plan.

    Passed as `response_schema` to `interrupt()` in `review_plan`. Setting
    `approved` to false routes back to `create_plan` with `feedback` unless
    `max_planning_iterations` has been reached.
    """

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
    """Structured summary and important verbatim excerpts from a retrieved webpage.

    Preserve key facts, statistics, dates, entities, and credible quotes
    needed by downstream research agents.
    """

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")


class TavilySearchResult(FrozenBaseModel):
    """Deduplicated search hit returned by the Tavily API before summarization.

    Attributes:
        id: Deterministic 8-character hex prefix of the SHA-256 digest of the
            normalized URL.
        url: Original result URL as returned by Tavily.
        title: Page title, or an empty string if omitted.
        content: Search snippet from Tavily, used as a fallback summary when
            `raw_content` is missing.
        score: Relevance score reported by Tavily.
        raw_content: Raw webpage content when available, or `None`.
    """

    id: str
    url: str
    title: str
    content: str
    score: float
    raw_content: str | None = None


class ProcessedSearchResult(FrozenBaseModel):
    """Summarized search source persisted in the session store and cited in reports.

    Attributes:
        id: Deterministic 8-character hex source ID carried over from
            `TavilySearchResult`.
        url: Original source URL displayed in formatted bibliographies.
        title: Page title displayed in formatted bibliographies.
        summary: Webpage summary produced by `ContentSummary`, falling back to
            the Tavily snippet when raw content is absent, or an empty string
            if summarization fails.
        key_excerpts: Important quotes and excerpts extracted during
            summarization, or `None` when raw content is absent or
            summarization fails.
    """

    id: str
    url: str
    title: str
    summary: str
    key_excerpts: str | None = None


# =============================================================================
# Research
# =============================================================================


class ResearchFinding(FrozenBaseModel):
    """Discrete, source-backed factual finding without inline citation markers."""

    id: SkipJsonSchema[str] = Field(
        default="",
        description="Unique identifier for the research finding, assigned automatically after compression.",
    )
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
    """Thematic section grouping related evidence-backed research findings."""

    heading: str = Field(
        description="Descriptive heading for this thematic group of research findings."
    )
    findings: list[ResearchFinding] = Field(
        description="Comprehensive list of evidence-backed findings under this heading."
    )


class DraftResearch(FrozenBaseModel):
    """Comprehensive, source-attributed draft organizing all gathered research findings.

    Include every executed search query and group all relevant facts, metrics,
    quotes, and source IDs from external search results into thematic sections.
    """

    queries_and_tool_calls: list[str] = Field(
        description="List of search queries and external retrieval tool calls executed during the research conversation."
    )
    sections: list[ResearchSection] = Field(
        description="Thematic sections organizing all relevant research findings and their source IDs."
    )


class ResearchFindingUpdate(FrozenBaseModel):
    """Correction or removal for a single finding in `<draft_research>`.

    Emit only when a finding requires factual or citation correction
    (`action: "update"`) or lacks support in `<sources>` (`action: "remove"`).
    """

    finding_id: str = Field(
        description="The exact 'id' attribute of the <finding id='...'> in <draft_research> to update or remove."
    )
    action: Literal["update", "remove"] = Field(
        description=(
            "Set to 'update' to modify the research finding's statement and/or source_ids. "
            "Set to 'remove' if the finding is not substantiated by any source in <sources>."
        )
    )
    statement: str | None = Field(
        description=(
            "Updated factual statement if a detail needed correction. "
            "Set to null if keeping the original statement unchanged, or if action is 'remove'."
        )
    )
    source_ids: list[str] | None = Field(
        description=(
            "Updated list of 8-character hex source IDs if citations needed correction. "
            "Set to null if keeping the original source_ids unchanged, or if action is 'remove'."
        )
    )


class VerifiedResearch(FrozenBaseModel):
    """Audit result listing only the finding updates or removals needed for `<draft_research>`."""

    updates: list[ResearchFindingUpdate] = Field(
        description="List of finding updates or removals. Leave empty if all findings in <draft_research> are already accurate and properly attributed."
    )


# =============================================================================
# Writer
# =============================================================================


class FinalReportIssue(FrozenBaseModel):
    """Single verification issue in `<final_report>` with a concrete fix grounded in `<findings>`."""

    section: str = Field(
        description="The heading or location in <final_report> where the issue was found."
    )
    description: str = Field(
        description="Specific explanation of what is missing, ungrounded, misattributed, contradictory, or redundant in the report."
    )
    fix_suggestion: str = Field(
        description="Concrete, actionable suggestion for how the writer can fix this issue using <findings>."
    )


class FinalReportReview(FrozenBaseModel):
    """Verification audit of `<final_report>` against `<research_question>` and `<findings>`.

    Approve with `issues` set to `null` when all audit dimensions pass, or
    reject with concrete, evidence-grounded issues requiring revision.
    """

    approved: bool = Field(
        description=(
            "True if the final report successfully synthesizes all research findings into a comprehensive, accurate, and well-structured report. "
            "False only if one or more issues were identified that require revision."
        )
    )
    issues: list[FinalReportIssue] | None = Field(
        description="List of all identified issues in the final report. Null when approved is True."
    )
