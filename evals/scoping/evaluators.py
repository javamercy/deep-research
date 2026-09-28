from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langsmith import EvaluationResult

from deep_research.configuration import LLMModel
from evals.scoping.prompts import BRIEF_CRITERIA_SYSTEM_PROMPT, BRIEF_CRITERIA_USER_PROMPT, BRIEF_GROUNDEDNESS_SYSTEM_PROMPT, BRIEF_GROUNDEDNESS_USER_PROMPT
from evals.scoping.schemas import BriefGroundednessEvaluation, CriteriaEvaluation

judge = ChatOpenRouter(model=LLMModel.DEEPSEEK_V4_FLASH, temperature=0.1)
structured_judge = judge.with_structured_output(
    CriteriaEvaluation,
    method="json_schema",
    include_raw=False,
    strict=True)


async def brief_criteria_evaluator(
        outputs: dict[str, object],
        reference_outputs: dict[str, object]
) -> EvaluationResult:
    research_brief = outputs.get("research_brief")
    criteria = reference_outputs.get("criteria")

    if not isinstance(criteria, list):
        raise TypeError("Reference outputs must contain a list of criteria strings.")

    responses = cast(
        list[CriteriaEvaluation],
        await structured_judge.abatch([
            [
                SystemMessage(content=BRIEF_CRITERIA_SYSTEM_PROMPT),
                HumanMessage(content=BRIEF_CRITERIA_USER_PROMPT.format(
                    research_brief=research_brief,
                    criterion=criterion))
            ]
            for criterion in criteria]))

    individual_evaluations = [
        CriteriaEvaluation(
            reasoning=response.reasoning,
            criterion=criterion,
            captured=response.captured
        )
        for criterion, response in zip(criteria, responses, strict=True)
    ]

    captured_count = sum(1 for evaluation in individual_evaluations if evaluation.captured)
    total_count = len(individual_evaluations)

    return EvaluationResult(
        key="criteria_evaluation",
        score=captured_count / total_count if total_count > 0 else 0.0,
        extra={"individual_evaluations": [
            {
                "criterion": eval_result.criterion,
                "captured": eval_result.captured,
                "reasoning": eval_result.reasoning
            }
            for eval_result in individual_evaluations
        ]}
    )


async def brief_groundedness_evaluator(
        outputs: dict[str, object],
        reference_outputs: dict[str, object]
) -> EvaluationResult:
    research_brief = outputs.get("research_brief")
    criteria = reference_outputs.get("criteria")

    if not isinstance(criteria, list):
        raise TypeError("Reference outputs must contain a list of criteria strings.")

    response = cast(
        BriefGroundednessEvaluation,
        await structured_judge.ainvoke([
            SystemMessage(content=BRIEF_GROUNDEDNESS_SYSTEM_PROMPT),
            HumanMessage(
                content=BRIEF_GROUNDEDNESS_USER_PROMPT
                .format(research_brief=research_brief, criteria=criteria)
            )
        ])
    )

    return EvaluationResult(
        key="brief_groundedness_evaluation",
        score=1.0 if response.passes else 0.0,
        comment=response.reasoning,
        extra={"criteria": criteria}
    )
