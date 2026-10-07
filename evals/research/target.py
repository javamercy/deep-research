from langchain_core.messages import BaseMessage, convert_to_messages

from deep_research.workflows.research import build_research_graph
from evals.config import create_eval_config


async def research_target(inputs: dict) -> dict[str, list[BaseMessage]]:
    raw_messages = inputs.get("messages")
    if not raw_messages or not isinstance(raw_messages, list):
        raise TypeError("Input 'messages' must be a list of message dictionaries.")

    messages = convert_to_messages(raw_messages)

    config = create_eval_config()
    graph = build_research_graph(checkpointer=None)
    result = await (
        graph.nodes["conduct_research"]
        .ainvoke({"researcher_messages": messages}, config=config)
    )

    return result
