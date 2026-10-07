from langsmith import EvaluationResult

from evals.types import EvaluatorFunc


def create_supervisor_evaluators() -> list[EvaluatorFunc]:
    async def supervisor_parallelization_evaluator(
            outputs: dict,
            reference_outputs: dict
    ) -> EvaluationResult:
        tool_calls = outputs["output"].update["supervisor_messages"].tool_calls
        score = len(tool_calls) == reference_outputs["num_expected_threads"]

        return EvaluationResult(
            key="supervisor_parallelization_evaluation",
            score=score
        )

    return [supervisor_parallelization_evaluator]
