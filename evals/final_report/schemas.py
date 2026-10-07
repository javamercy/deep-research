from typing import Literal

from pydantic import Field

from deep_research.schemas import FrozenBaseModel


class RequirementEvaluation(FrozenBaseModel):
    """Represents the evaluation result for a single requirement in the final report."""

    requirement_id: str = Field(description="The unique identifier of the requirement being evaluated.")
    score: Literal[0, 1, 2] = Field(
        description="The score assigned to the requirement based on how well it is addressed in the final report. 0 = Not Addressed, 1 = Partially Addressed, 2 = Fully Addressed."
    )
    reasoning: str = Field(
        description="A detailed explanation of the evaluation, including specific examples or quotes from the final report that support the assigned score."
    )
