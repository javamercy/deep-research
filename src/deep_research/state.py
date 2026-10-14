import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState, add_messages

from deep_research.schemas import ResearchPlan


class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass


class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """

    # Research brief generated from user conversation history
    research_brief: str

    # Research plan generated from the research brief and user input
    research_plan: ResearchPlan

    # Messages exchanged with the supervisor agent for coordination
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]

    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add]

    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add]

    # Final formatted research report
    final_report: str


class ScopingInputState(MessagesState):
    pass


class ScopingState(MessagesState):
    # Response from AI agent asking for clarification on the user's request
    clarification_question: str

    # Research brief generated from user conversation history
    research_brief: str

    # Research plan generated from the research brief and user input
    research_plan: ResearchPlan

    # Feedback from the user on the research plan
    plan_feedback: str

    # User decision on whether to approve or revise the research plan
    plan_approved: bool

    # Counter tracking the number of planning iterations performed
    planning_iterations: int


class ScopingOutputState(TypedDict):
    research_brief: str
    research_plan: ResearchPlan


class ResearcherInputState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topic: str
    researcher_tool_calls: int


class ResearcherState(TypedDict):
    """
    State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """

    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topic: str
    researcher_tool_calls: int
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]


class ResearcherOutputState(TypedDict):
    """
    Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """

    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add]
    compressed_research: str


class SupervisorInputState(TypedDict):
    research_plan: ResearchPlan


class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.

    Manages coordination between supervisor and research agents, tracking
    research progress and accumulating findings from multiple sub-agents.
    """

    # Messages exchanged with supervisor for coordination and decision-making
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]

    # Research plan generated from the research brief 
    research_plan: ResearchPlan

    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], operator.add]

    # Counter tracking the number of research iterations performed
    supervisor_iterations: int

    # Raw unprocessed research notes collected from sub-agent research
    raw_notes: Annotated[list[str], operator.add]


class SupervisorOutputState(TypedDict):
    research_plan: ResearchPlan
    notes: Annotated[list[str], operator.add]
