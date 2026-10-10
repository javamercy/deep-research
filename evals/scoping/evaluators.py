from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import EvaluationResult
from langsmith.schemas import Run

from deep_research.configuration import LLMModelConfig
from deep_research.models import init_openrouter_structured_model
from deep_research.schemas import ResearchPlan
from evals.scoping.prompts import BRIEF_CRITERIA_SYSTEM_PROMPT, BRIEF_CRITERIA_USER_PROMPT, BRIEF_GROUNDEDNESS_SYSTEM_PROMPT, BRIEF_GROUNDEDNESS_USER_PROMPT, PLAN_CRITERIA_SYSTEM_PROMPT, PLAN_CRITERIA_USER_PROMPT
from evals.scoping.schemas import BriefCriteriaEvaluation, BriefGroundednessEvaluation, PlanCriteriaEvaluation
from evals.types import EvaluatorFunc


def create_research_brief_evaluators(judge_config: LLMModelConfig) -> list[EvaluatorFunc]:
    criteria_judge = init_openrouter_structured_model(
        llm_config=judge_config,
        output_schema=BriefCriteriaEvaluation
    )

    groundedness_judge = init_openrouter_structured_model(
        llm_config=judge_config,
        output_schema=BriefGroundednessEvaluation
    )

    async def research_brief_criteria_evaluator(
            outputs: dict,
            reference_outputs: dict,
            run: Run
    ) -> EvaluationResult:
        research_brief = outputs.get("research_brief")
        criteria = reference_outputs.get("criteria")

        if not criteria or not isinstance(criteria, list):
            raise ValueError("Missing or invalid 'criteria' in reference outputs")

        if len(criteria) == 0:
            raise ValueError("Reference outputs must contain at least one criterion.")

        session_id = str(run.metadata.get("thread_id", run.id))

        batch_messages: list[LanguageModelInput] = [
            [
                SystemMessage(content=BRIEF_CRITERIA_SYSTEM_PROMPT),
                HumanMessage(
                    content=BRIEF_CRITERIA_USER_PROMPT.format(research_brief=research_brief)
                ),
                HumanMessage(
                    content=f"<criterion_to_evaluate>\n{criterion}\n</criterion_to_evaluate>"
                )
            ]
            for criterion in criteria
        ]

        first = await (
            criteria_judge
            .bind(session_id=session_id)
            .ainvoke(batch_messages[0])
        )

        remaining = await (
            criteria_judge
            .bind(session_id=session_id)
            .abatch(batch_messages[1:], config={"max_concurrency": 4})
        )

        responses = [first, *remaining]

        individual_evaluations = [
            BriefCriteriaEvaluation(
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

        if len(criteria) == 0:
            raise ValueError("Reference outputs must contain at least one criterion.")

        session_id = str(run.metadata.get("thread_id", run.id))

        messages = [
            SystemMessage(content=BRIEF_GROUNDEDNESS_SYSTEM_PROMPT),
            HumanMessage(
                content=BRIEF_GROUNDEDNESS_USER_PROMPT.format(research_brief=research_brief)
            ),
            HumanMessage(
                content=f"<criteria_to_evaluate>\n{criteria}\n</criteria_to_evaluate>"
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


def create_research_plan_evaluators(judge_config: LLMModelConfig) -> list[EvaluatorFunc]:
    criteria_judge = init_openrouter_structured_model(
        llm_config=judge_config,
        output_schema=PlanCriteriaEvaluation
    )

    async def research_plan_criteria_evaluator(
            outputs: dict,
            reference_outputs: dict,
            run: Run
    ) -> EvaluationResult:
        research_plan = outputs.get("research_plan")
        if not research_plan or not isinstance(research_plan, ResearchPlan):
            raise ValueError("Missing or invalid 'research_plan' in outputs")

        criteria = reference_outputs.get("criteria")
        if not criteria or not isinstance(criteria, list):
            raise ValueError("Missing or invalid 'criteria' in reference outputs")

        if len(criteria) == 0:
            raise ValueError("Reference outputs must contain at least one criterion.")

        session_id = str(run.metadata.get("thread_id", run.id))

        formatted_plan = format_research_plan(research_plan)
        batch_messages: list[LanguageModelInput] = [
            [
                SystemMessage(content=PLAN_CRITERIA_SYSTEM_PROMPT),
                HumanMessage(
                    content=PLAN_CRITERIA_USER_PROMPT.format(research_plan=formatted_plan)
                ),
                HumanMessage(
                    content=f"<criterion>\n{criterion}\n</criterion>"
                )
            ]
            for criterion in criteria
        ]

        first = await (
            criteria_judge
            .bind(session_id=session_id)
            .ainvoke(batch_messages[0])
        )

        remaining = await (
            criteria_judge
            .bind(session_id=session_id)
            .abatch(batch_messages[1:], config={"max_concurrency": 4})
        )

        responses = [first, *remaining]

        individual_evaluations = [
            PlanCriteriaEvaluation(
                reasoning=response.reasoning,
                criterion=criterion,
                captured=response.captured
            )
            for criterion, response in zip(criteria, responses, strict=True)
        ]

        captured_count = sum(evaluation.captured for evaluation in individual_evaluations)
        total_count = len(individual_evaluations)
        score = captured_count / total_count

        return EvaluationResult(
            key="research_plan_criteria_evaluation",
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

    return [research_plan_criteria_evaluator]


def format_research_plan(research_plan: ResearchPlan) -> str:
    """Format research tasks for an evaluation prompt."""

    if not research_plan.tasks:
        return "The research plan contains no tasks."

    return "\n\n".join(
        (
            f"Task {index}: {task.id}\n"
            f"Objective:\n{task.objective}\n"
            f"Success criteria:\n{task.success_criteria}"
        )
        for index, task in enumerate(research_plan.tasks, start=1)
    )
