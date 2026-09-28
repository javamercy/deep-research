from typing import cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.workflows.research import build_research_graph


async def research_target(inputs: dict[str, object]) -> dict[str, list[BaseMessage]]:
    messages = inputs["messages"]

    graph = build_research_graph(InMemorySaver())
    config = {"configurable": {"thread_id": uuid4()}}

    result = cast(
        dict[str, list[BaseMessage]],
        await graph.nodes["llm_call"].ainvoke({"researcher_messages": messages}, config=config)
    )

    return result
