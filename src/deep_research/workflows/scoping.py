from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.integrations.models import ModelRole, init_model
from deep_research.prompts.scoping import CLARIFICATION_SYSTEM_PROMPT, CLARIFICATION_USER_PROMPT, WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT, WRITE_RESEARCH_BRIEF_USER_PROMPT
from deep_research.schemas import ClarificationDecision, ResearchQuestion
from deep_research.state import AgentInputState, AgentState
from deep_research.utils import get_today_str


async def clarify_with_user(
        state: AgentState,
        runtime: Runtime[Configuration]
) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    model = init_model(ModelRole.SCOPING, runtime.context)
    structured_output_model = model.with_structured_output(
        ClarificationDecision,
        method="json_schema",
        include_raw=False,
        strict=True)

    messages = [
        SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(content=CLARIFICATION_USER_PROMPT.format(
            messages=get_buffer_string(messages=state.get("messages", []))
        ))
    ]
    response = await structured_output_model.ainvoke(messages)

    if not isinstance(response, ClarificationDecision):
        raise TypeError("Model response is not of type ClarificationDecision")

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


async def write_research_brief(
        state: AgentState,
        runtime: Runtime[Configuration]
) -> AgentState:
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """

    model = init_model(ModelRole.SCOPING, runtime.context)
    structured_output_model = model.with_structured_output(
        ResearchQuestion,
        include_raw=False,
        method="json_schema",
        strict=True
    )
    messages = [
        SystemMessage(content=WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(content=WRITE_RESEARCH_BRIEF_USER_PROMPT.format(
            messages=get_buffer_string(state.get("messages", [])),
        ))
    ]
    response = await structured_output_model.ainvoke(messages)

    if not isinstance(response, ResearchQuestion):
        raise TypeError("Model response is not of type ResearchQuestion")

    state["research_brief"] = response.research_brief
    state["supervisor_messages"] = [HumanMessage(content=response.research_brief)]
    return state


def build_scoping_graph(checkpointer: Checkpointer):
    # TODO: Workaround for Pyrefly/LangGraph TypedDict compatibility.
    # See: https://github.com/facebook/pyrefly/issues/3745
    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        AgentState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=AgentInputState,
        context_schema=Configuration,
    )

    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", END)

    return builder.compile(checkpointer=checkpointer)
