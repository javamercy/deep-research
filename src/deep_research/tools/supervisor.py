from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
class ConductResearch(BaseModel):
    """Delegate a standalone research assignment to a dedicated researcher.

    Provide complete, self-contained instructions for a single topic without
    acronyms or abbreviations, since each researcher runs independently and
    cannot see other researchers' work.
    """

    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


@tool
class ResearchComplete(BaseModel):
    """Declare that all research plan tasks and success criteria are satisfied.

    Call only when the gathered findings fully address the research plan and no
    further investigation is needed.
    """


class UpdateResearchTask(BaseModel):
    """Status update for a single task in the research plan."""

    task_id: str = Field(description="The unique identifier of the research task to update.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="The new status of the research task. Can be 'pending', 'in_progress', or 'completed'."
    )


@tool
class UpdateResearchPlan(BaseModel):
    """Update the status of existing tasks in the research plan by task ID.

    Only task `status` (`pending`, `in_progress`, or `completed`) may be
    changed. Do not treat an update as applied until confirmed by the tool
    response.
    """

    tasks: list[UpdateResearchTask] = Field(
        description="A list of research tasks that will be updated in the research plan.",
    )
