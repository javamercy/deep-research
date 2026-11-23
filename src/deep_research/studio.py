"""Compiled graph entry points exported at import time for LangGraph Studio.

Each graph listed in `langgraph.json` is built and compiled at import time with
`checkpointer=None` and `store=None` (no LLM or network calls occur on import).
LangGraph Studio replaces these with managed persistence at runtime; invoking
graphs directly from this module without injecting a checkpointer (for
interrupts) or store (for source persistence) will fail at runtime.
"""

from deep_research.workflows.deep_research import build_deep_research_graph
from deep_research.workflows.research import build_researcher_graph
from deep_research.workflows.scoping import build_scoping_graph
from deep_research.workflows.supervisor import build_supervisor_graph
from deep_research.workflows.writer import build_writer_graph

scoping_graph = build_scoping_graph(checkpointer=None)

researcher_graph = build_researcher_graph(checkpointer=None)

supervisor_graph = build_supervisor_graph(checkpointer=None)

writer_graph = build_writer_graph(checkpointer=None)

deep_research_graph = build_deep_research_graph(checkpointer=None)
