from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_model
from deep_research.prompts.research import COMPRESS_SYSTEM_PROMPT, COMPRESS_USER_PROMPT, RESEARCH_SYSTEM_PROMPT
from deep_research.state import ResearcherInputState, ResearcherOutputState, ResearcherState
from deep_research.tools.search import tavily_search
from deep_research.utils import get_today_str

TOOLS: list[BaseTool] = [tavily_search]

tool_node = ToolNode(
    name="tool_node",
    tools=TOOLS,
    messages_key="researcher_messages",
    handle_tool_errors=True
)


async def conduct_research(
        state: ResearcherState,
        config: RunnableConfig
) -> Command[Literal["tool_node", "compress_research"]]:
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """

    configuration = Configuration.from_runnable_config(config)

    if state.get("researcher_tool_calls", 0) >= configuration.max_researcher_tool_calls:
        return Command(goto="compress_research")

    model_with_tools = (
        init_openrouter_model(
            llm_config=configuration.research_llm_config,
            session_id=configuration.session_id
        )
        .bind_tools(TOOLS)
    )

    messages: list[BaseMessage] = [
        SystemMessage(
            content=RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        *state["researcher_messages"],
    ]
    response = await model_with_tools.ainvoke(messages, config=config)

    return Command(
        goto="tool_node" if response.tool_calls else "compress_research",
        update={
            "researcher_messages": [response]
        }
    )


async def researcher_tools(
        state: ResearcherState,
        config: RunnableConfig
) -> dict:
    last_message = state["researcher_messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("Expected an AIMessage before tool execution")

    result = await tool_node.ainvoke(state, config=config)

    return {
        **result,
        "researcher_tool_calls": (
                state.get("researcher_tool_calls", 0)
                + len(last_message.tool_calls)
        ),
    }


async def compress_research(
        state: ResearcherState,
        config: RunnableConfig
) -> dict:
    """Compress research findings into a concise summary.

   Takes all the research messages and tool outputs and creates
   a compressed summary suitable for the supervisor's decision-making.
   """

    configuration = Configuration.from_runnable_config(config)
    compress_model = init_openrouter_model(
        llm_config=configuration.compression_llm_config,
        session_id=configuration.session_id
    )

    messages: list[BaseMessage] = [
        SystemMessage(
            content=COMPRESS_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        *state["researcher_messages"],
        HumanMessage(
            content=COMPRESS_USER_PROMPT.format(research_topic=state["research_topic"])
        ),
    ]
    response = await compress_model.ainvoke(messages)

    # TODO: Consider using a more structured approach to extract raw notes,
    #  separating them by message type or source for better clarity and analysis.
    raw_notes = [
        message.text for message in filter_messages(
            state["researcher_messages"],
            include_types=[ToolMessage, AIMessage]
        )
    ]

    return {
        "compressed_research": response.text,
        # TODO: See todo above.
        "raw_notes": ["\n".join(raw_notes)]
    }


def build_researcher_graph(checkpointer: Checkpointer):
    """Build the research workflow graph."""

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        ResearcherState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=ResearcherInputState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=ResearcherOutputState)

    builder.add_node("conduct_research", conduct_research)
    builder.add_node("compress_research", compress_research)
    builder.add_node("researcher_tools", researcher_tools)

    builder.add_edge(START, "conduct_research")
    builder.add_edge("researcher_tools", "conduct_research")
    builder.add_edge("compress_research", END)

    return builder.compile(checkpointer=checkpointer)
