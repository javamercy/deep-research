from langsmith import EvaluationResult


async def next_step_evaluator(
        outputs: dict[str, object],
        reference_outputs: dict[str, object]
) -> EvaluationResult:
    messages = outputs.get("researcher_messages")
    if not isinstance(messages, list):
        raise TypeError("Outputs must contain a list of researcher messages.")

    made_tool_call = len(messages[-1].tool_calls) > 0
    score = made_tool_call == (reference_outputs.get("next_step") == "continue")
    
    return EvaluationResult(key="next_step_evaluation", score=score)
