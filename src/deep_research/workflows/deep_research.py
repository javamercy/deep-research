from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer

from deep_research.state import AgentInputState, AgentState
from deep_research.workflows.final_report import generate_final_report
from deep_research.workflows.scoping import build_scoping_graph
from deep_research.workflows.supervisor import build_supervisor_graph


def build_deep_research_graph(checkpointer: Checkpointer):
    # pyrefly: ignore [bad-argument-type, bad-specialization]
    builder = StateGraph(AgentState, input_schema=AgentInputState)

    builder.add_node("scoping_graph", build_scoping_graph(checkpointer))
    builder.add_node("supervisor_graph", build_supervisor_graph(checkpointer))
    builder.add_node("generate_final_report", generate_final_report)

    builder.add_edge(START, "scoping_graph")
    builder.add_edge("scoping_graph", "supervisor_graph")
    builder.add_edge("supervisor_graph", "generate_final_report")
    builder.add_edge("generate_final_report", END)

    return builder.compile(checkpointer)
