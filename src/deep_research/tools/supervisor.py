from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""


class UpdateResearchTask(BaseModel):
    """Model representing a research task to be updated in the research plan."""

    task_id: str = Field(
        description="The unique identifier of the research task to update."
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="The new status of the research task. Can be 'pending', 'in_progress', or 'completed'."
    )


@tool
class UpdateResearchPlan(BaseModel):
    """Tool for updating the research plan with modified tasks."""

    tasks: list[UpdateResearchTask] = Field(
        description="A list of research tasks that will be updated in the research plan.",
    )
