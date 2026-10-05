from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_model
from deep_research.prompts.research import COMPRESS_SYSTEM_PROMPT, COMPRESS_USER_PROMPT, RESEARCH_SYSTEM_PROMPT
from deep_research.state import ResearcherOutputState, ResearcherState
from deep_research.tools.reflection import think_tool
from deep_research.tools.search import tavily_search
from deep_research.utils import get_today_str

_tools: list[BaseTool] = [tavily_search, think_tool]


def create_tool_node() -> ToolNode:
    return ToolNode(
        name="tool_node",
        tools=_tools,
        messages_key="researcher_messages",
    )


def increment_tool_call_iterations(state: ResearcherState) -> dict:
    """Increment the tool call iterations counter in the state."""

    return {"tool_call_iterations": state["tool_call_iterations"] + 1}


async def conduct_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """

    if not state.get("research_topic"):
        raise ValueError("Research input requires a nonempty research topic.")

    configuration = Configuration.from_runnable_config(config)
    model_with_tools = (
        init_openrouter_model(llm_config=configuration.research_llm_config)
        .bind_tools(_tools)
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())),
        *state["researcher_messages"],
    ]
    response = await model_with_tools.ainvoke(messages, config=config)

    return {"researcher_messages": [response]}


async def compress_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Compress research findings into a concise summary.

   Takes all the research messages and tool outputs and creates
   a compressed summary suitable for the supervisor's decision-making.
   """

    configuration = Configuration.from_runnable_config(config)
    compress_model = init_openrouter_model(llm_config=configuration.compression_llm_config)

    messages: list[BaseMessage] = [
        SystemMessage(content=COMPRESS_SYSTEM_PROMPT.format(date=get_today_str())),
        *state["researcher_messages"],
        HumanMessage(
            content=COMPRESS_USER_PROMPT.format(
                research_topic=state["research_topic"],
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


def should_continue(
        state: ResearcherState,
        config: RunnableConfig
) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """

    configuration = Configuration.from_runnable_config(config)
    max_tool_calls = configuration.max_research_tool_calls

    if state["tool_call_iterations"] == max_tool_calls:
        return "compress_research"

    last_message = cast(AIMessage, state["researcher_messages"][-1])

    if last_message.tool_calls:
        return "tool_node"

    return "compress_research"


def build_research_graph(checkpointer: Checkpointer):
    """Build the research workflow graph."""

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        ResearcherState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=ResearcherOutputState)

    builder.add_node("conduct_research", conduct_research)
    builder.add_node("compress_research", compress_research)
    builder.add_node("tool_node", create_tool_node())
    builder.add_node("increment_tool_call_iterations", increment_tool_call_iterations)

    builder.add_edge(START, "conduct_research")
    builder.add_conditional_edges("conduct_research", should_continue)
    builder.add_edge("tool_node", "increment_tool_call_iterations")
    builder.add_edge("increment_tool_call_iterations", "conduct_research")

    builder.add_edge("compress_research", END)

    return builder.compile(checkpointer=checkpointer)
