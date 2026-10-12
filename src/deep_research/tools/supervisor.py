from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deep_research.schemas import ResearchTask


@tool
class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""


@tool
class UpdateResearchPlan(BaseModel):
    """Tool for updating the research plan with new or modified tasks."""

    task_updates: list[ResearchTask] = Field(
        description="A list of updated research tasks. If existing tasks are being updated, they should be included in this list with their updated details. New tasks can also be added to this list.",
    )
    summary: str = Field(
        description="A summary of the updates made to the research plan. This should provide a concise overview of the changes, including any new tasks added or existing tasks modified.",
    )
