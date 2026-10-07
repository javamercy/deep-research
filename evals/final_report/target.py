from deep_research.workflows.deep_research import build_deep_research_graph
from evals.config import create_eval_config


async def final_report_target(inputs: dict) -> dict[str, object]:
    research_brief = inputs.get("research_brief")
    if not research_brief:
        raise ValueError("Missing 'research_brief' in inputs")

    notes = inputs.get("notes")
    if not notes or not isinstance(notes, list):
        raise ValueError("Missing or invalid 'notes' in inputs")

    config = create_eval_config()
    graph = build_deep_research_graph(checkpointer=None)

    return await graph.nodes["generate_final_report"].ainvoke(
        inputs,
        config=config
    )
