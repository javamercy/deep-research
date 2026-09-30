from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith.client import Client

should_parallelize = [
    HumanMessage(content="Compare OpenAI vs Gemini deep research."),
    AIMessage(content="I need to analyze this request to determine if can should be parallelized.", tool_calls=[
        {
            "name": "think_tool",
            "args": {
                "reflection": "This is a comparison task involving two distinct AI products: OpenAI v Gemini Deep Research."},
            "id": "call_think_1"
        }
    ]),
    ToolMessage(
        content="Analysis complete: This is a comparison task involving two distinct AI products: OpenAI v Gemini Deep Research.",
        tool_call_id="call_think_1", name="think_tool")
]

should_not_parallelize = [
    HumanMessage(content="What are the top three Chinese restaurants in Chelsea, Manhattan"),
    AIMessage(content="Let me think about whether this task requires parallelization.", tool_calls=[
        {
            "name": "think_tool",
            "args": {
                "reflection": "This is a ranking/listing task for restaurants in a specific geographic area (Chelsea, Manhattan)."},
            "id": "call_think_2"
        }
    ]),
    ToolMessage(
        content="Analysis complete: This is a ranking/listing task for restaurants in a specific geographic area (Chelsea, Manhattan).",
        tool_call_id="call_think_2", name="think_tool")
]


def ensure_supervisor_dataset(langsmith_client: Client, dataset_name: str):
    if not langsmith_client.has_dataset(dataset_name=dataset_name):
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="A dataset that evaluates whether a supervisor can accurately decide when to parallelize research."
        )

        langsmith_client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"supervisor_messages": should_parallelize},
                    "outputs": {"num_expected_threads": 2}
                },
                {
                    "inputs": {"supervisor_messages": should_not_parallelize},
                    "outputs": {"num_expected_threads": 1}
                }
            ]
        )
