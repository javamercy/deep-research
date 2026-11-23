"""Scoping workflow for request clarification, research brief synthesis, and planning."""

import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.config import get_config
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command, interrupt

from deep_research.configuration import Configuration, get_session_id
from deep_research.formatting import research_plan_to_xml
from deep_research.openrouter import init_openrouter_structured_model
from deep_research.prompts.scoping import (
    CLARIFICATION_SYSTEM_PROMPT,
    CLARIFICATION_USER_PROMPT,
    RESEARCH_PLANNING_SYSTEM_PROMPT,
    RESEARCH_PLANNING_USER_PROMPT,
    WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT,
    WRITE_RESEARCH_BRIEF_USER_PROMPT,
)
from deep_research.schemas import (
    ClarificationAnswer,
    ClarificationDecision,
    ResearchPlan,
    ResearchPlanReview,
    ResearchQuestion,
)
from deep_research.state import ScopingInputState, ScopingOutputState, ScopingState
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)


async def assess_clarification(
    state: ScopingState,
    runtime: Runtime[Configuration],
) -> Command[Literal["ask_clarification", "write_research_brief"]]:
    """Assess whether the user request needs clarification before research begins.

    Reads `messages` from state and resolves runtime settings (`allow_clarification`,
    `scoping_llm_config`, `max_structured_output_retries`) plus `configurable.thread_id`
    for session tracking. When `allow_clarification` is false, skips the model call and
    routes directly to `write_research_brief`. Because `messages` uses the `add_messages`
    reducer, the verification acknowledgement is returned as a single-item list to append
    to the conversation.

    Returns:
        A `Command` routing to `ask_clarification` with `clarification_question` set, or
        to `write_research_brief` with an appended verification `AIMessage` (or no update
        when clarification is disabled).
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    if not configuration.allow_clarification:
        return Command(goto="write_research_brief")

    clarification_model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        output_schema=ClarificationDecision,
        max_retries=configuration.max_structured_output_retries,
        session_id=session_id,
    )

    messages = [
        SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=CLARIFICATION_USER_PROMPT.format(
                messages=get_buffer_string(messages=state["messages"])
            )
        ),
    ]
    response = await clarification_model.ainvoke(messages, config=config)

    if response.need_clarification:
        return Command(
            goto="ask_clarification", update={"clarification_question": response.question}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def ask_clarification(state: ScopingState) -> dict:
    """Interrupt execution to collect the user's answer to a clarification question.

    Reads `clarification_question` from state and suspends the graph with payload
    `{"type": "clarification", "question": ...}` and `response_schema=ClarificationAnswer`.
    Requires a `checkpointer` at runtime and expects to be resumed with a
    `ClarificationAnswer` payload (`answer`). Because `messages` uses the `add_messages`
    reducer, only the new question `AIMessage` and answer `HumanMessage` are returned.

    Returns:
        A partial state update appending the question and user answer to `messages`.
    """
    question = state["clarification_question"]

    answer = interrupt(
        {
            "type": "clarification",
            "question": question,
        },
        response_schema=ClarificationAnswer,
    )

    return {
        "messages": [AIMessage(content=question), HumanMessage(content=answer.answer)],
    }


async def write_research_brief(state: ScopingState, runtime: Runtime[Configuration]) -> dict:
    """Synthesize the conversation history into a structured research brief.

    Reads `messages` from state and invokes the scoping model (`scoping_llm_config`,
    retried up to `max_structured_output_retries`) with `ResearchQuestion` structured
    output. Requires `configurable.thread_id` for session tracking.

    Returns:
        A partial state update setting `research_brief`.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    research_brief_model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        output_schema=ResearchQuestion,
        max_retries=configuration.max_structured_output_retries,
        session_id=session_id,
    )

    messages = [
        SystemMessage(content=WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=WRITE_RESEARCH_BRIEF_USER_PROMPT.format(
                messages=get_buffer_string(state["messages"])
            )
        ),
    ]
    response = await research_brief_model.ainvoke(messages, config=config)

    return {"research_brief": response.research_brief}


async def create_plan(
    state: ScopingState,
    runtime: Runtime[Configuration],
) -> Command[Literal["review_plan", "__end__"]]:
    """Generate a structured research plan or revise an existing plan from feedback.

    Reads `research_brief` and `planning_iterations` from state; when `research_plan`
    is already present, also reads `plan_feedback` and appends the serialized plan and
    feedback to the prompt for revision. Requires `configurable.thread_id` for session
    tracking.

    Returns:
        A `Command` updating `research_plan` and incrementing `planning_iterations`,
        routing to `review_plan` when `allow_collaborative_planning` is enabled or
        `__end__` otherwise.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    planning_model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        session_id=session_id,
        max_retries=configuration.max_structured_output_retries,
        output_schema=ResearchPlan,
    )

    messages = [
        SystemMessage(content=RESEARCH_PLANNING_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=RESEARCH_PLANNING_USER_PROMPT.format(research_question=state["research_brief"])
        ),
    ]

    research_plan = state.get("research_plan")
    if research_plan:
        messages.append(
            HumanMessage(
                content=(
                    f"{research_plan_to_xml(research_plan)}\n\n"
                    f"<user_feedback>\n{state['plan_feedback']}\n</user_feedback>"
                )
            )
        )

    research_plan = await planning_model.ainvoke(messages, config=config)

    return Command(
        goto="review_plan" if configuration.allow_collaborative_planning else "__end__",
        update={
            "research_plan": research_plan,
            "planning_iterations": state.get("planning_iterations", 0) + 1,
        },
    )


def review_plan(
    state: ScopingState, runtime: Runtime[Configuration]
) -> Command[Literal["create_plan", "__end__"]]:
    """Interrupt for human review of the proposed research plan and route next steps.

    Reads `research_plan` and `planning_iterations` from state, then suspends execution
    with `{"research_plan": <json dict>, "message": "Review the proposed research plan."}`
    and `response_schema=ResearchPlanReview`. Requires a `checkpointer` at runtime and
    resumes with a `ResearchPlanReview` payload (`approved`, `feedback`). Routes to
    `__end__` when approved or when `planning_iterations` reaches
    `max_planning_iterations` (logging a warning); otherwise routes back to `create_plan`.

    Returns:
        A `Command` updating `plan_approved` and `plan_feedback` and routing to either
        `create_plan` or `__end__`.
    """
    response = interrupt(
        {
            "research_plan": state["research_plan"].model_dump(mode="json"),
            "message": "Review the proposed research plan.",
        },
        response_schema=ResearchPlanReview,
    )

    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)

    should_end = False
    if response.approved:
        should_end = True
    elif state.get("planning_iterations", 0) >= configuration.max_planning_iterations:
        should_end = True
        logger.warning(
            "Maximum planning iterations reached. Proceeding with the current research plan."
        )

    return Command(
        goto="__end__" if should_end else "create_plan",
        update={"plan_approved": response.approved, "plan_feedback": response.feedback},
    )


def build_scoping_graph(checkpointer: Checkpointer):
    """Build and compile the scoping workflow graph.

    Uses `ScopingState` internally with `ScopingInputState` as input, `ScopingOutputState`
    as output, and `Configuration` as context schema. Declares fixed edges from `START`
    to `assess_clarification`, `ask_clarification` back to `assess_clarification`, and
    `write_research_brief` to `create_plan`; all branching and loop termination are
    driven by `Command` returns in `assess_clarification`, `create_plan`, and
    `review_plan`.

    Args:
        checkpointer: Checkpointer for state persistence, or `None` when supplied by the
            runtime (such as LangGraph Studio). A checkpointer is required at execution
            time for `ask_clarification` and `review_plan` interrupts.

    Returns:
        The compiled scoping workflow graph.
    """
    # TODO: Workaround for Pyrefly/LangGraph TypedDict compatibility.
    # See: https://github.com/facebook/pyrefly/issues/3745
    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        ScopingState,
        context_schema=Configuration,
        # pyrefly: ignore [bad-argument-type]
        input_schema=ScopingInputState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=ScopingOutputState,
    )

    builder.add_node("assess_clarification", assess_clarification)
    builder.add_node("ask_clarification", ask_clarification)
    builder.add_node("write_research_brief", write_research_brief)
    builder.add_node("create_plan", create_plan)
    builder.add_node("review_plan", review_plan)

    builder.add_edge(START, "assess_clarification")
    builder.add_edge("ask_clarification", "assess_clarification")
    builder.add_edge("write_research_brief", "create_plan")

    return builder.compile(checkpointer=checkpointer)
