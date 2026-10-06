from deep_research.workflows.deep_research import build_deep_research_graph
from deep_research.workflows.research import build_research_graph
from deep_research.workflows.scoping import build_scoping_graph
from deep_research.workflows.supervisor import build_supervisor_graph

scoping_graph = build_scoping_graph(checkpointer=None)

research_graph = build_research_graph(checkpointer=None)

supervisor_graph = build_supervisor_graph(checkpointer=None)

deep_research_graph = build_deep_research_graph(checkpointer=None)
