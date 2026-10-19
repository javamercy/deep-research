import asyncio
import logging
from collections.abc import Sequence
from typing import Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.formatting import research_plan_to_xml
from deep_research.models import init_openrouter_model
from deep_research.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT
from deep_research.schemas import ResearchPlan
from deep_research.state import ResearcherInputState, SupervisorInputState, SupervisorOutputState, SupervisorState
from deep_research.tools.supervisor import ConductResearch, ResearchComplete, UpdateResearchPlan, UpdateResearchTask
from deep_research.utils import get_today_str
from deep_research.workflows.research import build_researcher_graph

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
        str(tool_message.content)
        for tool_message in filter_messages(
            messages,
            include_names=["ConductResearch"],
            include_types=[ToolMessage]
        )
    ]


async def supervisor(
        state: SupervisorState,
        config: RunnableConfig,
) -> Command[Literal["supervisor_tools", "__end__"]]:
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
        Command indicating the next node to go to and any state updates, including supervisor messages and iteration count.
    """

    configuration = Configuration.from_runnable_config(config)

    if state.get("supervisor_iterations", 0) >= configuration.max_supervisor_iterations:
        return Command(
            goto="__end__",
            update={
                "notes": get_notes_from_tool_calls(
                    state.get("supervisor_messages", [])
                ),
            },
        )

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
                max_researcher_iterations=configuration.max_supervisor_iterations,
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

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "supervisor_iterations": state.get("supervisor_iterations", 0) + 1,
        }
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
       Command to either return to the supervisor for further planning or end the research process.
    """

    # TODO: Exception handling for tool execution failures, including logging and fallback strategies

    supervisor_messages = state["supervisor_messages"]
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

    try:
        update_plan_calls, conduct_research_calls = (
            _collect_tool_calls_from_supervisor_message(tool_calls)
        )

        all_updates: list[UpdateResearchTask] = []
        for tool_call in update_plan_calls:
            all_updates.extend(
                UpdateResearchTask.model_validate(raw) for raw in tool_call["args"]["tasks"]
            )
            tool_messages.append(
                ToolMessage(
                    content=f"Updated {len(tool_call['args']['tasks'])} tasks in research plan",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )

        research_plan = _apply_task_updates(research_plan, all_updates)

        if research_complete:
            should_end = True
            next_step = "__end__"

        elif conduct_research_calls:
            researcher_agent = build_researcher_graph(checkpointer=None)

            threads = []
            for tool_call in conduct_research_calls:
                input_state = ResearcherInputState(
                    researcher_messages=[
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    research_topic=tool_call["args"]["research_topic"],
                    researcher_tool_calls=0,
                )

                # pyrefly: ignore [no-matching-overload]
                thread = researcher_agent.ainvoke(input_state, config=config)
                threads.append(thread)

            # TODO: Max concurrent research threads should be enforced here.
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


def _apply_task_updates(
        plan: ResearchPlan,
        task_updates: Sequence[UpdateResearchTask]
) -> ResearchPlan:
    status_by_id = {update.task_id: update.status for update in task_updates}
    unknown_ids = status_by_id.keys() - {task.id for task in plan.tasks}

    if unknown_ids:
        logger.warning(
            "UpdateResearchPlan referenced unknown task ids: %s", sorted(unknown_ids)
        )

    return plan.model_copy(
        update={
            "tasks": [
                task.model_copy(update={"status": status_by_id[task.id]})
                if task.id in status_by_id else task
                for task in plan.tasks
            ]
        }
    )


def _collect_tool_calls_from_supervisor_message(
        tool_calls: list[ToolCall]
) -> tuple[list[ToolCall], list[ToolCall]]:
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
    """Builds the supervisor graph with the given checkpointer.

    Args:
        checkpointer (Checkpointer): The checkpointer to use for the graph.

    Returns:
        The compiled supervisor graph.
    """

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        SupervisorState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=SupervisorInputState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=SupervisorOutputState
    )

    builder.add_node("supervisor", supervisor)
    builder.add_node("supervisor_tools", supervisor_tools)

    builder.add_edge(START, "supervisor")

    return builder.compile(checkpointer=checkpointer)
