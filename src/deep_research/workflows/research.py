from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.openrouter import init_openrouter_model, init_openrouter_structured_model
from deep_research.prompts.research import COMPRESS_SYSTEM_PROMPT, COMPRESS_USER_PROMPT, RESEARCHER_SYSTEM_PROMPT
from deep_research.schemas import DraftResearch
from deep_research.state import ResearcherInputState, ResearcherOutputState, ResearcherState
from deep_research.tools.search import tavily_search
from deep_research.utils import get_today_str


async def conduct_research(
        state: ResearcherState,
        config: RunnableConfig,
        *,
        tools: list[BaseTool]
) -> Command[Literal["researcher_tools", "compress_research"]]:
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Invoke a tool for further research (goto "tool_node").
    2. Compress the research findings into a summary (goto "compress_research").

    Args:
        state (ResearcherState): The current state of the researcher, including messages and tool calls.
        config (RunnableConfig): The configuration for the runnable, including model settings.
        tools (list[BaseTool]): The list of tools available for the researcher to use.

    Returns:
        Command indicating the next node to go to and any state updates.
    """

    configuration = Configuration.from_runnable_config(config)

    if state.get("researcher_tool_calls", 0) >= configuration.max_researcher_tool_calls:
        return Command(goto="compress_research")

    model_with_tools = (
        init_openrouter_model(
            llm_config=configuration.researcher_llm_config,
            session_id=configuration.session_id
        )
        .bind_tools(tools)
    )

    messages: list[BaseMessage] = [
        SystemMessage(
            content=RESEARCHER_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        *state["researcher_messages"],
    ]
    response = await model_with_tools.ainvoke(messages, config=config)

    return Command(
        goto="researcher_tools" if response.tool_calls else "compress_research",
        update={
            "researcher_messages": [response]
        }
    )


async def researcher_tools(
        state: ResearcherState,
        config: RunnableConfig,
        *,
        tool_node: ToolNode
) -> dict:
    """Execute the tool calls made by the researcher.

    This node executes the tool calls made by the researcher in the previous step.
    It updates the state with the results of the tool executions and increments the count of tool calls made.

    Args:
        state (ResearcherState): The current state of the researcher, including messages and tool calls.
        config (RunnableConfig): The configuration for the runnable, including model settings.
        tool_node (ToolNode): The node responsible for executing the tool calls.

    Returns:
        Partial state update containing the results of the tool executions and the updated count of tool calls.
    """

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

   Args:
       state (ResearcherState): The current state of the researcher, including messages and tool calls.
       config (RunnableConfig): The configuration for the runnable, including model settings.

   Returns:
       Partial state update containing the compressed research summary and raw notes.
   """

    configuration = Configuration.from_runnable_config(config)
    compress_model = init_openrouter_structured_model(
        configuration.compression_llm_config,
        output_schema=DraftResearch,
        session_id=configuration.session_id,
        max_retries=configuration.max_structured_output_retries
    )

    messages = [
        SystemMessage(
            content=COMPRESS_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        *state["researcher_messages"],
        HumanMessage(
            content=COMPRESS_USER_PROMPT.format(research_topic=state["research_topic"])
        ),
    ]
    draft_research = await compress_model.ainvoke(messages)

    # TODO: Consider using a more structured approach to extract raw notes,
    #  separating them by message type or source for better clarity and analysis.
    raw_notes = [
        message.text for message in filter_messages(
            state["researcher_messages"],
            include_types=[ToolMessage, AIMessage]
        )
    ]

    return {
        "draft_research": draft_research,
        # TODO: See todo above.
        "raw_notes": ["\n".join(raw_notes)]
    }


async def verify_research(
        state: ResearcherState,
        config: RunnableConfig,
        *,
        store: BaseStore
) -> dict:
    raise NotImplementedError("Research verification is not implemented yet.")


def build_researcher_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Build the research workflow graph with the given checkpointer.

    Args:
        checkpointer (Checkpointer): The checkpointer to use for the graph.
        store (BaseStore): The store to use for the graph.
        
    Returns:
        The compiled research workflow graph.
    """

    tools: list[BaseTool] = [tavily_search]

    tool_node = ToolNode(
        name="tool_node",
        tools=tools,
        messages_key="researcher_messages",
        handle_tool_errors=True
    )

    async def conduct_research_node(
            state: ResearcherState,
            config: RunnableConfig
    ) -> Command[Literal["researcher_tools", "compress_research"]]:
        return await conduct_research(
            state,
            config,
            tools=tools
        )

    async def researcher_tools_node(
            state: ResearcherState,
            config: RunnableConfig
    ) -> dict:
        return await researcher_tools(
            state,
            config,
            tool_node=tool_node,
        )

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        ResearcherState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=ResearcherInputState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=ResearcherOutputState)

    builder.add_node("conduct_research", conduct_research_node)
    builder.add_node("researcher_tools", researcher_tools_node)
    builder.add_node("compress_research", compress_research)

    builder.add_edge(START, "conduct_research")
    builder.add_edge("researcher_tools", "conduct_research")
    builder.add_edge("compress_research", END)

    return builder.compile(checkpointer=checkpointer, store=store)
