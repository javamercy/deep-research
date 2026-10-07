from pydantic import Field

from deep_research.schemas import FrozenBaseModel


class CriteriaEvaluation(FrozenBaseModel):
    """
        Individual success criterion evaluation result.

        This model represents a single evaluation criterion that should be present
        in the research brief, along with a detailed assessment of whether it was
        successfully captured and the reasoning behind that assessment.
    """

    criterion: str = Field(
        description="The specific success criterion being evaluated (e.g., 'Current age is 25', 'Monthly rent below 7k')"
    )
    reasoning: str = Field(
        description="Detailed explanation of why this criterion is or isn't captured in the research brief, including specific evidence from the brief"
    )
    captured: bool = Field(
        description="Whether this specific criterion is adequately captured in the research brief (True) or missing/inadequately addressed (False)"
    )


class ResearchBriefGroundednessEvaluation(FrozenBaseModel):
    """
        Overall research brief groundedness evaluation result.

        This model represents the overall assessment of whether the research brief
        is adequately grounded in the provided context, along with a detailed
        explanation of the reasoning behind that assessment.
    """

    reasoning: str = Field(
        description="Detailed explanation of whether the research brief is grounded in the provided context, including specific evidence from the brief"
    )
    passes: bool = Field(
        description="Whether the research brief is adequately grounded in the provided context (True) or not (False)"
    )
