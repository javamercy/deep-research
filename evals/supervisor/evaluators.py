from langsmith import EvaluationResult


async def supervisor_parallelization_evaluator(outputs: dict, reference_outputs: dict) -> EvaluationResult:
    tool_calls = outputs["output"].update["supervisor_messages"].tool_calls

    return EvaluationResult(
        key="supervisor_parallelization_evaluation",
        score=len(tool_calls) == reference_outputs["num_expected_threads"]
    )
