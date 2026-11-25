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
from deep_research.formatting import landscape_context_to_xml
from deep_research.openrouter import init_openrouter_structured_model
from deep_research.prompts.scoping import (
    CLARIFICATION_SYSTEM_PROMPT,
    CLARIFICATION_USER_PROMPT,
    CREATE_PLAN_SYSTEM_PROMPT,
    CREATE_PLAN_USER_PROMPT,
    RECON_QUERIES_SYSTEM_PROMPT,
    RECON_QUERIES_USER_PROMPT,
    WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT,
    WRITE_RESEARCH_BRIEF_USER_PROMPT,
)
from deep_research.schemas import (
    ClarificationAnswer,
    ClarificationDecision,
    ReconQueries,
    ResearchPlan,
    ResearchQuestion,
)
from deep_research.search.tavily import search_tavily_multiple
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


async def create_recon_queries(state: ScopingState, runtime: Runtime[Configuration]) -> dict:
    """Generate 0 to 3 pre-planning reconnaissance queries from `research_brief`.

    Reads `research_brief` from state and invokes the structured model with
    `ReconQueries` (capped at 3 queries, retried up to
    `max_structured_output_retries`). Requires `configurable.thread_id` for
    session tracking.

    Returns:
        A partial state update setting `recon_queries`.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    model = init_openrouter_structured_model(
        configuration.recon_queries_llm_config,
        output_schema=ReconQueries,
        max_retries=configuration.max_structured_output_retries,
        session_id=session_id,
    )

    messages = [
        SystemMessage(content=RECON_QUERIES_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=RECON_QUERIES_USER_PROMPT.format(research_question=state["research_brief"])
        ),
    ]
    response = await model.ainvoke(messages, config=config)

    return {"recon_queries": response.queries}


async def recon_search(state: ScopingState) -> dict:
    """Run parallel snippet-only Tavily searches for `recon_queries` and format XML context.

    Reads `recon_queries` from state. Short-circuits with an empty
    `landscape_context` string when `recon_queries` is empty or when Tavily
    returns no results; otherwise queries Tavily with `search_depth="advanced"`
    and `include_raw_content=False` (without writing to `BaseStore`) and
    serializes the deduplicated hits via `landscape_context_to_xml`.

    Returns:
        A partial state update setting `landscape_context`.
    """
    queries = state["recon_queries"]
    if not queries:
        return {"landscape_context": ""}

    results = await search_tavily_multiple(
        queries,
        max_results=3,
        topic="general",
        include_raw_content=False,
        search_depth="advanced",
    )

    if not results:
        return {"landscape_context": ""}

    return {"landscape_context": landscape_context_to_xml(results)}


async def create_plan(
    state: ScopingState,
    runtime: Runtime[Configuration],
) -> dict:
    """Generate a structured research plan from `research_brief` and optional reconnaissance context.

    Reads `research_brief` and optional `landscape_context` from state, appending
    `landscape_context` as a second `HumanMessage` when non-empty. Invokes the
    `scoping_llm_config` structured model with `ResearchPlan` (retried up to
    `max_structured_output_retries`). Requires `configurable.thread_id` for
    session tracking.

    Returns:
        A partial state update setting `research_plan`.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        session_id=session_id,
        max_retries=configuration.max_structured_output_retries,
        output_schema=ResearchPlan,
    )

    messages = [
        SystemMessage(content=CREATE_PLAN_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=CREATE_PLAN_USER_PROMPT.format(research_question=state["research_brief"])
        ),
    ]

    if landscape_context := state.get("landscape_context"):
        messages.append(HumanMessage(content=landscape_context))

    research_plan = await model.ainvoke(messages, config=config)

    return {
        "research_plan": research_plan,
    }


def build_scoping_graph(checkpointer: Checkpointer):
    """Build and compile the scoping workflow graph.

    Uses `ScopingState` internally with `ScopingInputState` as input,
    `ScopingOutputState` as output, and `Configuration` as context schema.
    Declares fixed edges from `START` to `assess_clarification`, from
    `ask_clarification` back to `assess_clarification`, and sequentially from
    `write_research_brief` through `create_recon_queries`, `recon_search`, and
    `create_plan` to `__end__`; clarification branching is driven by `Command`
    returns in `assess_clarification`.

    Args:
        checkpointer: Checkpointer for state persistence, or `None` when supplied by the
            runtime (such as LangGraph Studio). A checkpointer is required at execution
            time for the `ask_clarification` interrupt.

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
    builder.add_node("create_recon_queries", create_recon_queries)
    builder.add_node("recon_search", recon_search)
    builder.add_node("create_plan", create_plan)

    builder.add_edge(START, "assess_clarification")
    builder.add_edge("ask_clarification", "assess_clarification")
    builder.add_edge("write_research_brief", "create_recon_queries")
    builder.add_edge("create_recon_queries", "recon_search")
    builder.add_edge("recon_search", "create_plan")
    builder.add_edge("create_plan", "__end__")

    return builder.compile(checkpointer=checkpointer)
