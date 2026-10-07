from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import EvaluationResult
from langsmith.schemas import Run

from deep_research.configuration import LLMModelConfig
from deep_research.models import init_openrouter_structured_model
from deep_research.utils import get_today_str
from evals.final_report.prompts import REQUIREMENTS_COVERAGE_SYSTEM_PROMPT, REQUIREMENTS_COVERAGE_USER_PROMPT
from evals.final_report.schemas import RequirementEvaluation
from evals.types import EvaluatorFunc


def create_final_report_evaluators(judge_config: LLMModelConfig) -> list[EvaluatorFunc]:
    structured_judge = init_openrouter_structured_model(
        judge_config,
        output_schema=RequirementEvaluation,
    )

    async def final_report_requirement_coverage_evaluator(
            outputs: dict,
            reference_outputs: dict,
            run: Run
    ) -> EvaluationResult:
        final_report = outputs.get("final_report")
        if not final_report:
            raise ValueError("Missing 'final_report' in outputs")

        requirements = reference_outputs.get("requirements")
        if not requirements or not isinstance(requirements, list):
            raise ValueError("Missing or invalid 'requirements' in reference outputs")

        session_id = str(run.metadata.get("thread_id") or run.id)

        batch_messages: list[LanguageModelInput] = [
            [
                SystemMessage(content=REQUIREMENTS_COVERAGE_SYSTEM_PROMPT.format(
                    date=get_today_str()
                )),
                HumanMessage(content=REQUIREMENTS_COVERAGE_USER_PROMPT.format(
                    final_report=final_report,
                )),
                HumanMessage(content=f"<requirement>\n{requirement}\n</requirement>")
            ]
            for requirement in requirements
        ]

        first = await (
            structured_judge
            .bind(session_id=session_id)
            .ainvoke(batch_messages[0])
        )
        remaining = await (
            structured_judge
            .bind(session_id=session_id)
            .abatch(batch_messages[1:], config={"max_concurrency": 4})
        )
        responses = [first, *remaining]

        total_score = 0
        individual_evaluations = []

        for requirement, evaluation in zip(requirements, responses, strict=True):
            requirement_id = evaluation.requirement_id
            if requirement_id != requirement.get("id"):
                raise ValueError(f"Requirement ID mismatch: expected {requirement.get('id')}, got {requirement_id}")

            total_score += evaluation.score

            individual_evaluations.append({
                "requirement_id": requirement_id,
                "requirement_description": requirement.get("description"),
                "score": evaluation.score,
                "reasoning": evaluation.reasoning
            })

        average_score = total_score / (2 * len(responses))

        return EvaluationResult(
            key="final_report_requirement_coverage_evaluation",
            score=average_score,
            comment=f"Evaluated {len(responses)} requirements.",
            metadata={"individual_evaluations": individual_evaluations}
        )

    return [final_report_requirement_coverage_evaluator]
