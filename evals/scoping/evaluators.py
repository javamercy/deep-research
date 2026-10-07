from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import EvaluationResult
from langsmith.schemas import Run

from deep_research.configuration import LLMModelConfig
from deep_research.models import init_openrouter_structured_model
from evals.scoping.prompts import BRIEF_CRITERIA_SYSTEM_PROMPT, BRIEF_CRITERIA_USER_PROMPT, BRIEF_GROUNDEDNESS_SYSTEM_PROMPT, BRIEF_GROUNDEDNESS_USER_PROMPT
from evals.scoping.schemas import CriteriaEvaluation, ResearchBriefGroundednessEvaluation
from evals.types import EvaluatorFunc


def create_research_brief_evaluators(judge_config: LLMModelConfig) -> list[EvaluatorFunc]:
    criteria_judge = init_openrouter_structured_model(
        llm_config=judge_config,
        output_schema=CriteriaEvaluation
    )

    groundedness_judge = init_openrouter_structured_model(
        llm_config=judge_config,
        output_schema=ResearchBriefGroundednessEvaluation
    )

    async def research_brief_criteria_evaluator(
            outputs: dict,
            reference_outputs: dict,
            run: Run
    ) -> EvaluationResult:
        research_brief = outputs.get("research_brief")
        criteria = reference_outputs.get("criteria")

        if not isinstance(criteria, list):
            raise TypeError("Reference outputs must contain a list of criteria strings.")

        session_id = run.metadata.get("session_id", run.id)

        batch_messages: list[LanguageModelInput] = [
            [
                SystemMessage(content=BRIEF_CRITERIA_SYSTEM_PROMPT),
                HumanMessage(content=BRIEF_CRITERIA_USER_PROMPT.format(
                    research_brief=research_brief,
                    criterion=criterion))
            ]
            for criterion in criteria
        ]

        responses = await (
            criteria_judge
            .bind(session_id=session_id)
            .abatch(batch_messages)
        )

        individual_evaluations = [
            CriteriaEvaluation(
                reasoning=response.reasoning,
                criterion=criterion,
                captured=response.captured
            )
            for criterion, response in zip(criteria, responses, strict=True)
        ]

        captured_count = sum(evaluation.captured for evaluation in individual_evaluations)
        total_count = len(individual_evaluations)
        score = captured_count / total_count if total_count > 0 else 0.0

        return EvaluationResult(
            key="research_brief_criteria_evaluation",
            score=score,
            comment=f"Captured {captured_count}/{len(criteria)} criteria.",
            metadata={"individual_evaluations": [
                {
                    "criterion": eval_result.criterion,
                    "captured": eval_result.captured,
                    "reasoning": eval_result.reasoning
                }
                for eval_result in individual_evaluations
            ]}
        )

    async def research_brief_groundedness_evaluator(
            outputs: dict,
            reference_outputs: dict,
            run: Run
    ) -> EvaluationResult:
        research_brief = outputs.get("research_brief")
        criteria = reference_outputs.get("criteria")

        if not isinstance(criteria, list):
            raise TypeError("Reference outputs must contain a list of criteria strings.")

        session_id = str(run.metadata.get("thread_id", run.id))

        messages = [
            SystemMessage(content=BRIEF_GROUNDEDNESS_SYSTEM_PROMPT),
            HumanMessage(
                content=BRIEF_GROUNDEDNESS_USER_PROMPT
                .format(research_brief=research_brief, criteria=criteria)
            )
        ]

        response = await (
            groundedness_judge
            .bind(session_id=session_id)
            .ainvoke(messages)
        )

        return EvaluationResult(
            key="research_brief_groundedness_evaluation",
            score=float(response.passes),
            comment=response.reasoning,
        )

    return [research_brief_criteria_evaluator, research_brief_groundedness_evaluator]
