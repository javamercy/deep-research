from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer

from deep_research.state import AgentInputState, AgentState
from deep_research.workflows.final_report import generate_final_report
from deep_research.workflows.scoping import clarify_with_user, write_research_brief
from deep_research.workflows.supervisor import build_supervisor_graph


def build_deep_research_graph(checkpointer: Checkpointer):
    # pyrefly: ignore [bad-argument-type, bad-specialization]
    builder = StateGraph(AgentState, input_schema=AgentInputState)

    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)
    builder.add_node("supervisor_graph", build_supervisor_graph(checkpointer))
    builder.add_node("generate_final_report", generate_final_report)

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", "supervisor_graph")
    builder.add_edge("supervisor_graph", "generate_final_report")
    builder.add_edge("generate_final_report", END)

    return builder.compile(checkpointer)
