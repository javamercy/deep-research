from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from deep_research.workflows.supervisor import build_supervisor_graph


async def supervisor_target(inputs: dict[str, object]) -> dict[str, object]:
    config = RunnableConfig(configurable={"thread_id": uuid4()})
    graph = build_supervisor_graph(InMemorySaver())

    return await graph.nodes["supervisor"].ainvoke(inputs, config=config)
