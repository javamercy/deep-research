import asyncio
from collections.abc import Sequence
from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command

from deep_research.configuration import LLMModel
from deep_research.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from deep_research.state import ResearcherState, SupervisorState
from deep_research.tools.reflection import think_tool
from deep_research.tools.supervisor import ConductResearch, ResearchComplete
from deep_research.utils import get_today_str
from deep_research.workflows.research import build_research_graph

tools = [ConductResearch, ResearchComplete, think_tool]
model = ChatOpenRouter(
    model=LLMModel.DEEPSEEK_V4_FLASH,
    temperature=0.1,
    reasoning={"effort": "high"},
)
model_with_tools = model.bind_tools(tools=tools)

# Maximum number of tool call iterations for individual researcher agents
# This prevents infinite loops and controls research depth per topic
max_researcher_iterations = 6  # Calls to think_tool + ConductResearch

# Maximum number of concurrent research agents the supervisor can launch
# This is passed to the lead_researcher_prompt to limit parallel research tasks
max_concurrent_researchers = 3


def get_notes_from_tool_calls(messages: Sequence[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.

    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.

    Args:
        messages: List of messages from supervisor's conversation history

    Returns:
        List of research note strings extracted from ToolMessage objects
    """

    return [
        tool_message.text
        for tool_message in filter_messages(messages, include_types="tool")
    ]


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Coordinate research activities.

    Analyzes the research brief and current progress to decide:
    - What research topics need investigation
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress

    Returns:
        Command to proceed to supervisor_tools node with updated state
    """

    supervisor_messages = state.get("supervisor_messages", [])
    system_message = SUPERVISOR_SYSTEM_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
    )
    messages = [SystemMessage(content=system_message), *supervisor_messages]

    response = await model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": response,
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
        state: SupervisorState,
        config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
       state: Current supervisor state with messages and iteration count
       config: Runnable configuration for managing sub-agent execution

    Returns:
       Command to continue supervision, end process, or handle errors
    """

    supervisor_messages = state["supervisor_messages"]
    research_iterations = state["research_iterations"]
    last_message = cast(AIMessage, supervisor_messages[-1])
    tool_calls = last_message.tool_calls

    tool_messages: list[ToolMessage] = []
    all_raw_notes: list[str] = []
    next_step = "supervisor"
    should_end = False

    iterations_exceeded = research_iterations >= max_researcher_iterations
    research_complete = any(
        call["name"] == "ResearchComplete"
        for call in tool_calls
    )

    if iterations_exceeded or research_complete or not tool_calls:
        should_end = True
        next_step = END

    else:
        try:
            think_tool_calls = [
                tool_call for tool_call in tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            for tool_call in think_tool_calls:
                observation = await think_tool.ainvoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        tool_name="think_tool",
                        tool_call_id=tool_call["id"]
                    )
                )

            if conduct_research_calls:
                researcher_agent = build_research_graph(InMemorySaver())

                threads = []
                for tool_call in conduct_research_calls:
                    child_config = {
                        **config,
                        "configurable": {
                            **config.get("configurable", {}),
                            "thread_id": f"researcher-{tool_call['id']}",
                        }
                    }
                    researcher_state = ResearcherState(
                        researcher_messages=[
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        research_topic=tool_call["args"]["research_topic"],
                        tool_call_iterations=0,
                        compressed_research="",
                        raw_notes=[],
                    )

                    # pyrefly: ignore [no-matching-overload]
                    thread = researcher_agent.ainvoke(
                        researcher_state,
                        config=child_config
                    )
                    threads.append(thread)

                tool_results = await asyncio.gather(*threads)

                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error occurred in research agent"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                    for result, tool_call in zip(tool_results, conduct_research_calls, strict=True)
                ]

                tool_messages.extend(research_tool_messages)

                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]

        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "raw_notes": all_raw_notes,
            }
        )

    return Command(
        goto=next_step,
        update={
            "supervisor_messages": tool_messages,
            "raw_notes": all_raw_notes,
        }
    )


def build_supervisor_graph(checkpointer: Checkpointer):
    """Construct the supervisor agent graph."""

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("supervisor_tools", supervisor_tools)
    builder.add_edge(START, "supervisor")

    return builder.compile(checkpointer=checkpointer)
