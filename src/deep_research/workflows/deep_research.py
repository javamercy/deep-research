"""Top-level deep research workflow chaining scoping, supervision, and report writing."""

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deep_research.configuration import Configuration
from deep_research.state import AgentInputState, AgentState
from deep_research.workflows.scoping import build_scoping_graph
from deep_research.workflows.supervisor import build_supervisor_graph
from deep_research.workflows.writer import build_writer_graph


def build_deep_research_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Build and compile the end-to-end deep research workflow graph.

    Uses `AgentState` as internal and output state, `AgentInputState` as input schema, and
    `Configuration` as context schema. Chains `scoping_graph`, `supervisor_graph`, and
    `writer_graph` with fixed edges from `START` through `END`; all conditional routing
    and iteration occur inside those subgraphs.

    Args:
        checkpointer: Checkpointer passed to all subgraphs and the compiled graph, or
            `None` when supplied by the runtime. Required at execution time for
            `scoping_graph` interrupts.
        store: Store passed to `supervisor_graph`, `writer_graph`, and the compiled graph,
            or `None` when supplied by the runtime (such as LangGraph Studio). A
            `BaseStore` is required at execution time for source storage and citation
            verification.

    Returns:
        The compiled deep research workflow graph.
    """
    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        AgentState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=AgentInputState,
        context_schema=Configuration,
    )

    builder.add_node("scoping_graph", build_scoping_graph(checkpointer))
    builder.add_node("supervisor_graph", build_supervisor_graph(checkpointer, store))
    builder.add_node("writer_graph", build_writer_graph(checkpointer, store))

    builder.add_edge(START, "scoping_graph")
    builder.add_edge("scoping_graph", "supervisor_graph")
    builder.add_edge("supervisor_graph", "writer_graph")
    builder.add_edge("writer_graph", END)

    return builder.compile(checkpointer, store=store)
