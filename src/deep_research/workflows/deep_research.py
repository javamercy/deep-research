from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer

from deep_research.configuration import LLMModel
from deep_research.prompts.final_report import GENERATE_FINAL_REPORT_SYSTEM_PROMPT, GENERATE_FINAL_REPORT_USER_PROMPT
from deep_research.state import AgentInputState, AgentState
from deep_research.utils import get_today_str
from deep_research.workflows.scoping import clarify_with_user, write_research_brief
from deep_research.workflows.supervisor import build_supervisor_graph

writer_model = ChatOpenRouter(
    model=LLMModel.DEEPSEEK_V4_FLASH,
    temperature=0.1,
    reasoning={"effort": "high", "exclude": True},
)


async def generate_final_report(state: AgentState) -> dict:
    """Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state["notes"]
    findings = "\n".join(notes)

    messages = [
        SystemMessage(content=GENERATE_FINAL_REPORT_SYSTEM_PROMPT
                      .format(date=get_today_str())),
        HumanMessage(content=GENERATE_FINAL_REPORT_USER_PROMPT
                     .format(findings=findings, research_brief=state["research_brief"])),
    ]
    response = await writer_model.ainvoke(messages)

    return {
        "final_report": response.text,
        "messages": [AIMessage(f"Here is the final report:\n\n{response.text}")],
    }


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
