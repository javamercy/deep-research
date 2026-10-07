from langsmith import EvaluationResult

from evals.types import EvaluatorFunc


def create_research_evaluators() -> list[EvaluatorFunc]:
    async def next_step_evaluator(
            outputs: dict,
            reference_outputs: dict
    ) -> EvaluationResult:
        messages = outputs.get("researcher_messages")
        if not isinstance(messages, list):
            raise TypeError("Outputs must contain a list of researcher messages.")

        expected = reference_outputs.get("next_step")
        if expected not in ["continue", "stop"]:
            raise ValueError("Reference next_step must be 'continue' or 'stop'.")

        made_tool_call = len(messages[-1].tool_calls) > 0
        score = made_tool_call == (expected == "continue")

        return EvaluationResult(
            key="research_next_step_evaluation",
            score=score
        )

    return [next_step_evaluator]
