from langchain_core.messages import convert_to_messages

from deep_research.workflows.scoping import build_scoping_graph
from evals.config import create_eval_config


async def research_brief_target(inputs: dict) -> dict[str, object]:
    raw_messages = inputs.get("messages")
    if not raw_messages or not isinstance(raw_messages, list):
        raise TypeError("Input 'messages' must be a list of message dictionaries.")

    messages = convert_to_messages(raw_messages)

    config = create_eval_config()
    graph = build_scoping_graph(checkpointer=None)
    result = await (
        graph.nodes["write_research_brief"]
        .ainvoke({"messages": messages}, config=config)
    )

    return {"research_brief": result["research_brief"]}
