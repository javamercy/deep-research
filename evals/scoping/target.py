from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from deep_research.workflows.scoping import build_scoping_graph


async def scoping_target(inputs: dict[str, object]) -> dict[str, object]:
    graph = build_scoping_graph(InMemorySaver())
    config = {"configurable": {"thread_id": uuid4()}}
    messages = inputs["messages"]

    result = await graph.ainvoke({"messages": messages}, config=config)

    return {"research_brief": result["research_brief"]}
