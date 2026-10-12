import asyncio
import logging
from collections.abc import Sequence
from typing import Literal, cast
from xml.etree import ElementTree

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_model
from deep_research.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT
from deep_research.schemas import ResearchPlan, ResearchTask
from deep_research.state import ResearcherState, SupervisorState
from deep_research.tools.supervisor import ConductResearch, ResearchComplete, UpdateResearchPlan
from deep_research.utils import get_today_str
from deep_research.workflows.research import build_research_graph

logger = logging.getLogger(__name__)

TOOLS = [ConductResearch, UpdateResearchPlan, ResearchComplete]


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


async def supervisor(
        state: SupervisorState,
        config: RunnableConfig,
) -> dict:
    """Coordinate research activities.

    Analyzes the research plan to decide:
    - What tasks need investigation
    - What tasks need updates or revisions
    - Whether to conduct parallel research
    - When research is complete

    Args:
        state: Current supervisor state with messages and research progress
        config: Runnable configuration for managing sub-agent execution

    Returns:
        Updated state with supervisor messages and iteration count
    """

    configuration = Configuration.from_runnable_config(config)
    model_with_tools = (
        init_openrouter_model(
            llm_config=configuration.supervisor_llm_config,
            session_id=configuration.session_id
        )
        .bind_tools(TOOLS)
    )

    messages = [
        SystemMessage(
            content=SUPERVISOR_SYSTEM_PROMPT.format(
                date=get_today_str(),
                max_concurrent_research_threads=configuration.max_concurrent_research_threads,
                max_researcher_iterations=configuration.max_researcher_iterations,
            )
        ),
        HumanMessage(
            content=SUPERVISOR_USER_PROMPT.format(
                research_plan=research_plan_to_xml(state["research_plan"])
            )
        ),
        *state.get("supervisor_messages", [])
    ]

    response = await model_with_tools.ainvoke(messages)

    return {
        "supervisor_messages": response,
        "research_iterations": state.get("research_iterations", 0) + 1,
    }


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

    # TODO: Exception handling for tool execution failures, including logging and fallback strategies

    configuration = Configuration.from_runnable_config(config)
    max_researcher_iterations = configuration.max_researcher_iterations

    supervisor_messages = state["supervisor_messages"]
    research_iterations = state["research_iterations"]
    last_message = cast(AIMessage, supervisor_messages[-1])
    tool_calls = last_message.tool_calls

    research_plan = state["research_plan"]
    tool_messages: list[ToolMessage] = []
    all_raw_notes: list[str] = []

    next_step = "supervisor"
    should_end = False

    research_complete = any(
        call["name"] == "ResearchComplete"
        for call in tool_calls
    )
    iterations_exceeded = research_iterations >= max_researcher_iterations
    if iterations_exceeded or research_complete or not tool_calls:
        should_end = True
        next_step = END
    else:
        try:
            update_plan_calls, conduct_research_calls = (
                collect_tool_calls_from_supervisor_message(tool_calls)
            )

            updated_tasks = list(research_plan.tasks)
            for tool_call in update_plan_calls:
                for raw_update in tool_call["args"]["task_updates"]:
                    task_update = ResearchTask.model_validate(raw_update)

                    for i, task in enumerate(updated_tasks):
                        if task.id == task_update.id:
                            updated_tasks[i] = task_update
                            break

                tool_messages.append(
                    ToolMessage(
                        content=tool_call["args"]["summary"],
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            research_plan = ResearchPlan(tasks=updated_tasks)

            if conduct_research_calls:
                researcher_agent = build_research_graph(checkpointer=None)

                threads = []
                for tool_call in conduct_research_calls:
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
                    thread = researcher_agent.ainvoke(researcher_state, config=config)
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

        except Exception:
            logger.error("Error during supervisor tool execution")
            raise

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
            "research_plan": research_plan,
            "supervisor_messages": tool_messages,
            "raw_notes": all_raw_notes,
        }
    )


def research_plan_to_xml(plan: ResearchPlan) -> str:
    """Convert a ResearchPlan object to an XML string representation."""

    root = ElementTree.Element("research_plan")
    tasks_element = ElementTree.SubElement(root, "tasks")

    for i, task in enumerate(plan.tasks, start=1):
        task_element = ElementTree.SubElement(tasks_element, "task", attrib={"index": str(i)})
        ElementTree.SubElement(task_element, "id").text = task.id
        ElementTree.SubElement(task_element, "objective").text = task.objective
        ElementTree.SubElement(task_element, "success_criteria").text = task.success_criteria
        ElementTree.SubElement(task_element, "status").text = task.status

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def collect_tool_calls_from_supervisor_message(
        tool_calls: list[ToolCall]
) -> tuple[list[ToolCall], list[ToolCall]]:
    """Collect tool calls from the supervisor's last message."""

    update_plan_calls = [
        tool_call for tool_call in tool_calls
        if tool_call["name"] == "UpdateResearchPlan"
    ]

    conduct_research_calls = [
        tool_call for tool_call in tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    return update_plan_calls, conduct_research_calls


def build_supervisor_graph(checkpointer: Checkpointer):
    """Construct the supervisor agent graph."""

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("supervisor_tools", supervisor_tools)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "supervisor_tools")

    return builder.compile(checkpointer=checkpointer)
