"""Supervisor workflow for coordinating research delegation and plan progress."""

import logging
from collections.abc import Sequence
from typing import Literal, TypedDict, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command, Send
from pydantic import BaseModel

from deep_research.configuration import Configuration, get_session_id
from deep_research.formatting import research_plan_to_xml
from deep_research.openrouter import init_openrouter_model
from deep_research.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT
from deep_research.schemas import ResearchPlan
from deep_research.state import (
    ResearcherInputState,
    SupervisorInputState,
    SupervisorOutputState,
    SupervisorState,
)
from deep_research.tools.supervisor import (
    ConductResearch,
    ResearchComplete,
    UpdateResearchPlan,
    UpdateResearchTask,
)
from deep_research.utils import get_today_str
from deep_research.workflows.research import build_researcher_graph

logger = logging.getLogger(__name__)


class _ResearchTaskInput(TypedDict):
    """Per-task payload sent from `supervisor_tools` to `run_researcher` via `Send`."""

    research_topic: str
    tool_call_id: str
    tool_name: str


async def supervisor(
    state: SupervisorState,
    runtime: Runtime[Configuration],
    tools: Sequence[type[BaseModel]],
) -> Command[Literal["supervisor_tools", "__end__"]]:
    """Plan the next research delegations, plan status updates, or completion.

    Reads `supervisor_iterations`, `research_plan`, and `supervisor_messages` from state,
    and requires `configurable.thread_id` for session tracking. Immediately terminates to
    `__end__` when `supervisor_iterations` reaches `max_supervisor_iterations`. Otherwise
    invokes the `supervisor_llm_config` model with `tools` bound and appends the response
    via the `add_messages` reducer on `supervisor_messages`.

    Args:
        tools: Pydantic tool schemas bound to the supervisor model.

    Returns:
        A `Command` routing to `__end__` at the iteration limit, or to `supervisor_tools`
        with the appended `AIMessage` and incremented `supervisor_iterations`.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    if state.get("supervisor_iterations", 0) >= configuration.max_supervisor_iterations:
        return Command(goto="__end__")

    model_with_tools = init_openrouter_model(
        llm_config=configuration.supervisor_llm_config,
        session_id=session_id,
    ).bind_tools(tools)

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
                research_plan=research_plan_to_xml(state["research_plan"]),
            )
        ),
        *state.get("supervisor_messages", []),
    ]

    response = await model_with_tools.ainvoke(messages, config=config)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "supervisor_iterations": state.get("supervisor_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """Process the supervisor's plan updates, researcher delegations, and completion signal.

    Reads the trailing `AIMessage` from `supervisor_messages` and the current
    `research_plan` from state. Applies any `UpdateResearchPlan` task status updates first
    and records their `ToolMessage` acknowledgements. Unless `ResearchComplete` was also
    called, dispatches each `ConductResearch` call to `run_researcher` via `Send`
    (`max_concurrent_research_threads` is not enforced).

    Because `supervisor_messages` uses `add_messages`, only the new `ToolMessage` entries
    are returned, while `research_plan` replaces the prior plan.

    Returns:
        A `Command` routing to `__end__` with updated `research_plan` when
        `ResearchComplete` is present (skipping any `ConductResearch` calls), to a list of
        `Send("run_researcher", ...)` tasks with `research_plan` and `supervisor_messages`
        when `ConductResearch` calls are present, or back to `supervisor` with
        `research_plan` and `supervisor_messages`.

    Raises:
        Exception: Logs and re-raises any error during tool-call parsing or task-update
            validation.
    """
    # TODO: Exception handling for tool execution failures, including logging and fallback strategies

    supervisor_messages = state["supervisor_messages"]
    last_message = cast("AIMessage", supervisor_messages[-1])

    tool_calls = last_message.tool_calls
    research_complete = any(call["name"] == "ResearchComplete" for call in tool_calls)

    tool_messages: list[ToolMessage] = []

    try:
        update_plan_calls, conduct_research_calls = _extract_tool_calls_from_supervisor_message(
            tool_calls
        )

        all_updates: list[UpdateResearchTask] = []
        for tool_call in update_plan_calls:
            all_updates.extend(
                UpdateResearchTask.model_validate(raw) for raw in tool_call["args"]["tasks"]
            )
            tool_messages.append(
                ToolMessage(
                    content=f"Updated {len(tool_call['args']['tasks'])} tasks in research plan",
                    name=tool_call["name"],  # UpdateResearchPlan
                    tool_call_id=tool_call["id"],
                )
            )

        research_plan = _apply_task_updates(state["research_plan"], all_updates)

        if research_complete:
            return Command(
                goto="__end__",
                update={"research_plan": research_plan},
            )

        if conduct_research_calls:
            sends = [
                Send(
                    "run_researcher",
                    _ResearchTaskInput(
                        research_topic=tool_call["args"]["research_topic"],
                        tool_call_id=tool_call["id"] or "",
                        tool_name=tool_call["name"],
                    ),
                )
                for tool_call in conduct_research_calls
            ]
            return Command(
                goto=sends,
                update={
                    "research_plan": research_plan,
                    "supervisor_messages": tool_messages,
                },
            )

    except Exception as e:
        logger.error(f"Error during supervisor tool execution {e}")
        raise

    if research_complete:
        return Command(
            goto="__end__",
            update={"research_plan": research_plan},
        )

    return Command(
        goto="supervisor",
        update={
            "research_plan": research_plan,
            "supervisor_messages": tool_messages,
        },
    )


def _apply_task_updates(
    plan: ResearchPlan, task_updates: Sequence[UpdateResearchTask]
) -> ResearchPlan:
    """Return a copy of `plan` with task statuses updated from `task_updates`.

    If multiple updates target the same `task_id`, the last one wins. Unknown task IDs
    are logged as a warning and ignored; task ordering and all other fields are preserved.
    """
    status_by_id = {update.task_id: update.status for update in task_updates}
    unknown_ids = status_by_id.keys() - {task.id for task in plan.tasks}

    if unknown_ids:
        logger.warning("UpdateResearchPlan referenced unknown task ids: %s", sorted(unknown_ids))

    return plan.model_copy(
        update={
            "tasks": [
                task.model_copy(update={"status": status_by_id[task.id]})
                if task.id in status_by_id
                else task
                for task in plan.tasks
            ]
        }
    )


def _extract_tool_calls_from_supervisor_message(
    tool_calls: list[ToolCall],
) -> tuple[list[ToolCall], list[ToolCall]]:
    """Partition supervisor tool calls into plan updates and research delegations.

    Preserves relative order within each list and ignores other tool calls such as
    `ResearchComplete`, returning `(update_plan_calls, conduct_research_calls)`.
    """
    update_plan_calls = [
        tool_call for tool_call in tool_calls if tool_call["name"] == "UpdateResearchPlan"
    ]

    conduct_research_calls = [
        tool_call for tool_call in tool_calls if tool_call["name"] == "ConductResearch"
    ]

    return update_plan_calls, conduct_research_calls


def build_supervisor_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Build and compile the supervisor workflow graph.

    Uses `SupervisorState` internally with `SupervisorInputState` as input,
    `SupervisorOutputState` as output, and `Configuration` as context schema. Compiles
    `build_researcher_graph(checkpointer=None, store=store)` for the `run_researcher`
    node and binds `ConductResearch`, `UpdateResearchPlan`, and `ResearchComplete` to
    `supervisor`. Fixed edges connect `START` to `supervisor` and `run_researcher` back
    to `supervisor`; `supervisor` and `supervisor_tools` drive dynamic routing (including
    `Send` fan-out to `run_researcher`) and termination via `Command`.

    Args:
        checkpointer: Checkpointer for state persistence, or `None` when supplied by the
            runtime.
        store: Store passed to the compiled graph and researcher subgraph, or `None` when
            supplied by the runtime (such as LangGraph Studio). A `BaseStore` is required
            at execution time for researcher subgraphs.

    Returns:
        The compiled supervisor workflow graph.
    """
    tools: list[type[BaseModel]] = [
        ConductResearch,
        UpdateResearchPlan,
        ResearchComplete,
    ]
    researcher_graph = build_researcher_graph(checkpointer=None, store=store)

    async def supervisor_node(
        state: SupervisorState,
        runtime: Runtime[Configuration],
    ) -> Command[Literal["supervisor_tools", "__end__"]]:
        return await supervisor(state, runtime, tools)

    async def run_researcher_node(
        task_input: _ResearchTaskInput,
        config: RunnableConfig,
    ) -> dict:
        """Execute a single delegated research task via `researcher_graph`.

        Invoked per `Send` from `supervisor_tools` with `task_input`, forwarding `config`
        so runtime configuration and `BaseStore` propagate to the subgraph. Falls back to
        an error string if `compressed_research` is absent from the subgraph output.

        Because `compressed_researches` uses `operator.add` and `supervisor_messages` uses
        `add_messages`, only the single `(research_topic, compressed_research)` tuple and
        matching `ToolMessage` for `task_input["tool_call_id"]` are returned.

        Args:
            task_input: Per-task delegation payload from `supervisor_tools`.

        Returns:
            A partial state update appending the result to `compressed_researches` and
            its `ToolMessage` observation to `supervisor_messages`.
        """
        state = ResearcherInputState(
            researcher_messages=[HumanMessage(content=task_input["research_topic"])],
            research_topic=task_input["research_topic"],
        )
        output = await researcher_graph.ainvoke(state, config=config)
        compressed_research = output.get(
            "compressed_research", "Error occurred in researcher agent"
        )
        return {
            "compressed_researches": [(state["research_topic"], compressed_research)],
            "supervisor_messages": [
                ToolMessage(
                    content=compressed_research,
                    name=task_input["tool_name"],
                    tool_call_id=task_input["tool_call_id"],
                )
            ],
        }

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        SupervisorState,
        context_schema=Configuration,
        # pyrefly: ignore [bad-argument-type]
        input_schema=SupervisorInputState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=SupervisorOutputState,
    )

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("supervisor_tools", supervisor_tools)
    # LangGraph infers the Send payload schema from the node annotation at runtime.
    # Pyrefly checks the node against SupervisorState instead.
    builder.add_node("run_researcher", run_researcher_node)  # pyrefly: ignore [bad-argument-type]

    builder.add_edge(START, "supervisor")
    builder.add_edge("run_researcher", "supervisor")

    return builder.compile(checkpointer=checkpointer, store=store)
