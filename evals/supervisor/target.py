from deep_research.workflows.supervisor import build_supervisor_graph
from evals.config import create_eval_config


async def supervisor_target(inputs: dict) -> dict[str, object]:
    if not inputs.get("supervisor_messages"):
        raise ValueError("supervisor_messages is required in inputs")

    config = create_eval_config()
    graph = build_supervisor_graph(checkpointer=None)

    return await graph.nodes["supervisor"].ainvoke(inputs, config=config)
