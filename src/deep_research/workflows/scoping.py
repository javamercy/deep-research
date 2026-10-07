from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_structured_model
from deep_research.prompts.scoping import CLARIFICATION_SYSTEM_PROMPT, CLARIFICATION_USER_PROMPT, WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT, WRITE_RESEARCH_BRIEF_USER_PROMPT
from deep_research.schemas import ClarificationDecision, ResearchQuestion
from deep_research.state import AgentInputState, AgentState
from deep_research.utils import get_today_str


async def clarify_with_user(
        state: AgentState,
        config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    configuration = Configuration.from_runnable_config(config)

    if not configuration.allow_clarification:
        return Command(goto="write_research_brief")

    clarification_model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        output_schema=ClarificationDecision,
        max_retries=configuration.max_structured_output_retries,
        session_id=configuration.session_id
    )

    messages = [
        SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT.format(
            date=get_today_str()
        )),
        HumanMessage(content=CLARIFICATION_USER_PROMPT.format(
            messages=get_buffer_string(messages=state["messages"])
        ))
    ]
    response = await clarification_model.ainvoke(messages, config=config)

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> dict:
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """

    configuration = Configuration.from_runnable_config(config)

    research_brief_model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        output_schema=ResearchQuestion,
        max_retries=configuration.max_structured_output_retries,
        session_id=configuration.session_id
    )

    messages = [
        SystemMessage(content=WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT.format(
            date=get_today_str()
        )),
        HumanMessage(content=WRITE_RESEARCH_BRIEF_USER_PROMPT.format(
            messages=get_buffer_string(state["messages"])
        ))
    ]
    response = await research_brief_model.ainvoke(messages, config=config)

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=response.research_brief)]
    }


def build_scoping_graph(checkpointer: Checkpointer):
    # TODO: Workaround for Pyrefly/LangGraph TypedDict compatibility.
    # See: https://github.com/facebook/pyrefly/issues/3745
    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        AgentState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=AgentInputState,
    )

    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", END)

    return builder.compile(checkpointer=checkpointer)
