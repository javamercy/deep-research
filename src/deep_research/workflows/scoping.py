from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Checkpointer, Command, interrupt

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_structured_model
from deep_research.prompts.scoping import CLARIFICATION_SYSTEM_PROMPT, CLARIFICATION_USER_PROMPT, RESEARCH_PLANNING_SYSTEM_PROMPT, RESEARCH_PLANNING_USER_PROMPT, WRITE_RESEARCH_BRIEF_SYSTEM_PROMPT, WRITE_RESEARCH_BRIEF_USER_PROMPT
from deep_research.schemas import ClarificationDecision, ResearchPlan, ResearchPlanReview, ResearchQuestion
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


async def create_plan(
        state: AgentState,
        config: RunnableConfig
) -> dict:
    configuration = Configuration.from_runnable_config(config)

    planning_model = init_openrouter_structured_model(
        configuration.scoping_llm_config,
        session_id=configuration.session_id,
        max_retries=configuration.max_structured_output_retries,
        output_schema=ResearchPlan
    )

    # TODO: Write prompts for research planning.
    messages = [
        SystemMessage(
            content=RESEARCH_PLANNING_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        HumanMessage(
            content=RESEARCH_PLANNING_USER_PROMPT.format(research_question=state["research_brief"])
        )
    ]

    research_plan = state.get("research_plan")
    if research_plan:
        messages.append(
            HumanMessage(
                content=f"<research_plan>\n{research_plan.model_dump_json(indent=2)}\n</research_plan>\n\n"
                        f"<user_feedback>\n{state["plan_feedback"]}\n</user_feedback>"
            )
        )

    research_plan = await planning_model.ainvoke(messages, config=config)

    return {
        "research_plan": research_plan,
        "planning_iterations": state.get("planning_iterations", 0) + 1
    }


def review_plan(state: AgentState) -> dict:
    response = interrupt(
        {
            "research_plan": state["research_plan"].model_dump(mode="json"),
            "message": "Review the proposed research plan."
        },
        response_schema=ResearchPlanReview
    )

    return {
        "plan_approved": response.approved,
        "plan_feedback": response.feedback
    }


def route_review(
        state: AgentState,
        config: RunnableConfig
) -> Literal["create_plan", "__end__"]:
    configuration = Configuration.from_runnable_config(config)

    if (
            state["planning_iterations"] == configuration.max_planning_iterations
            or state["plan_approved"]
    ):
        return "__end__"

    return "create_plan"


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
    builder.add_node("create_plan", create_plan)
    builder.add_node("review_plan", review_plan)

    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", "create_plan")
    builder.add_edge("create_plan", "review_plan")

    builder.add_conditional_edges("review_plan", route_review)

    return builder.compile(checkpointer=checkpointer)
