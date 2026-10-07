from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from deep_research.workflows.supervisor import build_supervisor_graph


async def supervisor_target(inputs: dict) -> dict[str, object]:
    if not inputs.get("supervisor_messages"):
        raise ValueError("supervisor_messages is required in inputs")

    config = RunnableConfig(configurable={"thread_id": uuid4()})
    graph = build_supervisor_graph(checkpointer=None)

    return await graph.nodes["supervisor"].ainvoke(inputs, config=config)
