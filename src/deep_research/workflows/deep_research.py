from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deep_research.state import AgentInputState, AgentState
from deep_research.workflows.scoping import build_scoping_graph
from deep_research.workflows.supervisor import build_supervisor_graph
from deep_research.workflows.writer import writer


def build_deep_research_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Builds the deep research graph with the given checkpointer.

    Args:
        checkpointer (Checkpointer): The checkpointer to use for the graph.
    Returns:
        The compiled deep research graph.
    """

    # pyrefly: ignore [bad-argument-type, bad-specialization]
    builder = StateGraph(AgentState, input_schema=AgentInputState)

    builder.add_node("scoping_graph", build_scoping_graph(checkpointer))
    builder.add_node("supervisor_graph", build_supervisor_graph(checkpointer))
    builder.add_node("writer_graph", writer)

    builder.add_edge(START, "scoping_graph")
    builder.add_edge("scoping_graph", "supervisor_graph")
    builder.add_edge("supervisor_graph", "writer_graph")
    builder.add_edge("writer_graph", END)

    return builder.compile(checkpointer, store=store)
