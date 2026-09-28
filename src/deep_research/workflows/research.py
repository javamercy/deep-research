from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_openrouter import ChatOpenRouter
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer

from deep_research.configuration import LLMModel
from deep_research.prompts.research import COMPRESS_SYSTEM_PROMPT, COMPRESS_USER_PROMPT, RESEARCH_SYSTEM_PROMPT
from deep_research.state import ResearcherOutputState, ResearcherState
from deep_research.tools.reflection import think_tool
from deep_research.tools.search import tavily_search
from deep_research.utils import get_today_str

MAX_TOOL_CALL_ITERATIONS = 5

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}
tool_node = ToolNode(
    name="tool_node",
    tools=tools,
    messages_key="researcher_messages",
)

model_with_tools = ChatOpenRouter(
    model=LLMModel.DEEPSEEK_V4_FLASH,
    temperature=0.1,
    reasoning={"effort": "medium"}
).bind_tools(tools)

compress_model = ChatOpenRouter(
    model=LLMModel.DEEPSEEK_V4_FLASH,
    temperature=0.1,
    reasoning={"effort": "high"},
    max_completion_tokens=64000
)


def increment_tool_call_iterations(state: ResearcherState) -> dict:
    """Increment the tool call iterations counter in the state."""

    return {"tool_call_iterations": state["tool_call_iterations"] + 1}


async def llm_call(state: ResearcherState) -> dict:
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """

    messages: list[BaseMessage] = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())),
        *state["researcher_messages"],
    ]
    response = await model_with_tools.ainvoke(messages)

    return {"researcher_messages": [response]}


async def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

   Takes all the research messages and tool outputs and creates
   a compressed summary suitable for the supervisor's decision-making.
   """

    messages: list[BaseMessage] = [
        SystemMessage(content=COMPRESS_SYSTEM_PROMPT.format(date=get_today_str())),
        *state["researcher_messages"],
        HumanMessage(
            content=COMPRESS_USER_PROMPT.format(
                research_topic=state["research_topic"]
            )
        ),
    ]
    response = await compress_model.ainvoke(messages)

    raw_notes = [
        str(message.content) for message in filter_messages(
            state["researcher_messages"],
            include_types=[ToolMessage, AIMessage]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """

    if state["tool_call_iterations"] == MAX_TOOL_CALL_ITERATIONS:
        return "compress_research"

    last_message = cast(AIMessage, state["researcher_messages"][-1])

    if (
            state["tool_call_iterations"] < MAX_TOOL_CALL_ITERATIONS
            and last_message.tool_calls
    ):
        return "tool_node"

    return "compress_research"


def build_research_graph(checkpointer: Checkpointer):
    """Build the research workflow graph."""

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        ResearcherState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=ResearcherOutputState)

    builder.add_node("llm_call", llm_call)
    builder.add_node("compress_research", compress_research)
    builder.add_node("tool_node", tool_node)
    builder.add_node("increment_tool_call_iterations", increment_tool_call_iterations)

    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", should_continue)
    builder.add_edge("tool_node", "increment_tool_call_iterations")
    builder.add_edge("increment_tool_call_iterations", "llm_call")

    builder.add_edge("compress_research", END)

    return builder.compile(checkpointer=checkpointer)
