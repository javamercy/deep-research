import logging
from collections.abc import Mapping, Sequence
from typing import Literal
from xml.etree import ElementTree

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration
from deep_research.formatting import draft_research_to_xml, processed_search_results_to_xml
from deep_research.memory.sources import get_sources_by_ids
from deep_research.openrouter import init_openrouter_model, init_openrouter_structured_model
from deep_research.prompts.research import (
    COMPRESS_SYSTEM_PROMPT,
    COMPRESS_USER_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    VERIFY_RESEARCH_SYSTEM_PROMPT,
    VERIFY_RESEARCH_USER_PROMPT,
)
from deep_research.schemas import (
    DraftResearch,
    ProcessedSearchResult,
    ResearchFinding,
    ResearchFindingUpdate,
    ResearchSection,
    VerifiedResearch,
)
from deep_research.state import ResearcherInputState, ResearcherOutputState, ResearcherState
from deep_research.tools.search import tavily_search
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)


async def conduct_research(
        state: ResearcherState,
        config: RunnableConfig,
        *,
        tools: list[BaseTool]
) -> Command[Literal["researcher_tools", "compress_research"]]:
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Invoke a tool for further research (goto "researcher_tools").
    2. Compress the research findings into a structured draft (goto "compress_research").

    Args:
        state (ResearcherState): The current state of the researcher, including messages and tool calls.
        config (RunnableConfig): The configuration for the runnable, including model settings.
        tools (list[BaseTool]): The list of tools available for the researcher to use.

    Returns:
        Command indicating the next node to go to and any state updates.
    """

    configuration = Configuration.from_runnable_config(config)

    if state.get("researcher_tool_calls", 0) >= configuration.max_researcher_tool_calls:
        return Command(goto="compress_research")

    model_with_tools = (
        init_openrouter_model(
            llm_config=configuration.researcher_llm_config,
            session_id=configuration.session_id
        )
        .bind_tools(tools)
    )

    messages: list[BaseMessage] = [
        SystemMessage(
            content=RESEARCHER_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        *state["researcher_messages"],
    ]
    response = await model_with_tools.ainvoke(messages, config=config)

    return Command(
        goto="researcher_tools" if response.tool_calls else "compress_research",
        update={
            "researcher_messages": [response]
        }
    )


async def researcher_tools(
        state: ResearcherState,
        config: RunnableConfig,
        *,
        tool_node: ToolNode
) -> dict:
    """Execute the tool calls made by the researcher.

    This node executes the tool calls made by the researcher in the previous step.
    It updates the state with the results of the tool executions and increments the count of tool calls made.

    Args:
        state (ResearcherState): The current state of the researcher, including messages and tool calls.
        config (RunnableConfig): The configuration for the runnable, including model settings.
        tool_node (ToolNode): The node responsible for executing the tool calls.

    Returns:
        Partial state update containing the results of the tool executions and the updated count of tool calls.
    """

    last_message = state["researcher_messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("Expected an AIMessage before tool execution")

    result = await tool_node.ainvoke(state, config=config)

    return {
        **result,
        "researcher_tool_calls": (
                state.get("researcher_tool_calls", 0)
                + len(last_message.tool_calls)
        ),
    }


async def compress_research(
        state: ResearcherState,
        config: RunnableConfig
) -> dict:
    """Organize and compress research findings into a structured DraftResearch report.

    Takes all research messages and tool outputs and extracts structured, source-attributed
    findings for downstream verification and supervisor decision-making.

    Args:
        state (ResearcherState): The current state of the researcher, including messages and tool calls.
        config (RunnableConfig): The configuration for the runnable, including model settings.

    Returns:
        Partial state update containing the structured draft_research and raw_notes.
    """

    configuration = Configuration.from_runnable_config(config)
    compress_model = init_openrouter_structured_model(
        configuration.compression_llm_config,
        output_schema=DraftResearch,
        session_id=configuration.session_id,
        max_retries=configuration.max_structured_output_retries
    )

    messages = [
        SystemMessage(
            content=COMPRESS_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        *state["researcher_messages"],
        HumanMessage(
            content=COMPRESS_USER_PROMPT.format(research_topic=state["research_topic"])
        ),
    ]
    draft_research = _assign_finding_ids(
        await compress_model.ainvoke(messages)
    )

    # TODO: Consider using a more structured approach to extract raw notes,
    #  separating them by message type or source for better clarity and analysis.
    raw_notes = [
        message.text for message in filter_messages(
            state["researcher_messages"],
            include_types=[ToolMessage, AIMessage]
        )
    ]

    return {
        "draft_research": draft_research,
        # TODO: See todo above.
        "raw_notes": ["\n".join(raw_notes)]
    }


async def verify_research(
        state: ResearcherState,
        config: RunnableConfig,
        *,
        store: BaseStore
) -> dict:
    """Audit draft research findings against stored sources and format verified citations.

    Args:
        state (ResearcherState): The current state of the researcher, including draft_research and messages.
        config (RunnableConfig): The configuration for the runnable, including model settings.
        store (BaseStore): The store containing session-scoped ProcessedSearchResult sources.

    Returns:
        Partial state update containing the formatted compressed_research markdown with citations.
    """

    if store is None:
        raise ValueError("Runtime store is not available.")

    configuration = Configuration.from_runnable_config(config=config)
    draft_research = state["draft_research"]

    tool_messages = [
        tool_message.text
        for tool_message in filter_messages(
            state["researcher_messages"],
            include_names=["tavily_search"],
            include_types=[ToolMessage]
        )
    ]
    tool_source_ids = _extract_source_ids_from_tool_messages(tool_messages)

    draft_source_ids = [
        s_id
        for section in draft_research.sections
        for finding in section.findings
        for s_id in finding.source_ids
    ]

    sources_by_id = await get_sources_by_ids(
        store,
        session_id=configuration.session_id,
        source_ids=[*tool_source_ids, *draft_source_ids]
    )

    verified_sections: list[ResearchSection] = []
    if draft_research.sections and sources_by_id:
        verifier_model = init_openrouter_structured_model(
            configuration.verifier_llm_config,
            session_id=configuration.session_id,
            output_schema=VerifiedResearch,
            max_retries=configuration.max_structured_output_retries
        )

        messages = [
            SystemMessage(content=VERIFY_RESEARCH_SYSTEM_PROMPT.format(
                date=get_today_str()
            )),
            HumanMessage(content=VERIFY_RESEARCH_USER_PROMPT.format(
                draft_research=draft_research_to_xml(
                    draft_research
                ),
                sources=processed_search_results_to_xml(
                    list(sources_by_id.values())
                )
            ))
        ]

        verified_research = await verifier_model.ainvoke(messages)
        verified_sections = _apply_finding_updates(
            draft_research,
            verified_research.updates
        )

    compressed_research = _format_verified_research(
        queries_and_tool_calls=draft_research.queries_and_tool_calls,
        sections=verified_sections,
        sources_by_id=sources_by_id
    )

    return {
        "compressed_research": compressed_research
    }


def _assign_finding_ids(draft_research: DraftResearch) -> DraftResearch:
    """Assign deterministic '<section_idx>.<finding_idx>' IDs to all findings in DraftResearch.

    Args:
        draft_research: The compressed DraftResearch report with unassigned finding IDs.

    Returns:
        Updated DraftResearch with populated finding IDs.
    """

    return draft_research.model_copy(
        update={
            "sections": [
                section.model_copy(
                    update={
                        "findings": [
                            finding.model_copy(update={"id": f"{sec_idx}.{find_idx}"})
                            for find_idx, finding in enumerate(section.findings, start=1)
                        ]
                    }
                )
                for sec_idx, section in enumerate(draft_research.sections, start=1)
            ]
        }
    )


def _apply_finding_updates(
        draft_research: DraftResearch,
        finding_updates: Sequence[ResearchFindingUpdate],
) -> list[ResearchSection]:
    """Apply verification updates and removals to draft research sections.

    Args:
        draft_research: The DraftResearch containing sections and ID-tagged findings.
        finding_updates: Sequence of update or removal operations produced by the verifier.

    Returns:
        List of verified ResearchSection objects with empty sections omitted.
    """

    updates_by_id = {update.finding_id: update for update in finding_updates}
    known_ids = {
        finding.id
        for section in draft_research.sections
        for finding in section.findings
    }
    unknown_ids = updates_by_id.keys() - known_ids

    if unknown_ids:
        logger.warning(
            "VerifiedResearch referenced unknown finding ids: %s",
            sorted(unknown_ids),
        )

    verified_sections: list[ResearchSection] = []
    for section in draft_research.sections:

        verified_findings: list[ResearchFinding] = []
        for finding in section.findings:
            update = updates_by_id.get(finding.id)
            if update is None:
                verified_findings.append(finding)
                continue

            if update.action == "remove":
                continue

            verified_findings.append(
                finding.model_copy(
                    update={
                        "statement": (
                            update.statement
                            if update.statement is not None
                            else finding.statement
                        ),
                        "source_ids": (
                            update.source_ids
                            if update.source_ids is not None
                            else finding.source_ids
                        ),
                    }
                )
            )

        if verified_findings:
            verified_sections.append(
                section.model_copy(update={"findings": verified_findings})
            )

    return verified_sections


def _extract_source_ids_from_tool_messages(
        messages: Sequence[str]
) -> list[str]:
    """Parse XML `<sources>` tool messages and extract all `<source id="...">` attributes.

    Args:
        messages: Sequence of XML string payloads from `tavily_search` tool messages.

    Returns:
        List of extracted 8-character hex source IDs.
    """

    source_ids = []
    for message in messages:
        try:
            root = ElementTree.fromstring(message)

            source_ids.extend(
                [
                    source_id
                    for result in root.findall("source")
                    if (source_id := result.attrib.get("id"))
                ]
            )

        except ElementTree.ParseError as e:
            logger.exception(f"Error while parsing message: {e}")

    return source_ids


def _format_verified_research(
        *,
        queries_and_tool_calls: Sequence[str],
        sections: Sequence[ResearchSection],
        sources_by_id: Mapping[str, ProcessedSearchResult]
) -> str:
    """Render verified findings into markdown with source ID citations.

    Args:
        queries_and_tool_calls: Search queries executed during the research session.
        sections: Verified thematic sections and findings to render.
        sources_by_id: Mapping of valid source IDs to their ProcessedSearchResult metadata.

    Returns:
        Formatted markdown string containing queries, cited findings, and a Sources list.
    """

    rendered_sections: list[str] = []
    cited_source_ids: dict[str, None] = {}

    for section in sections:
        rendered_findings: list[str] = []
        for finding in section.findings:
            statement = finding.statement.strip()
            if not statement:
                continue

            valid_ids = list(
                dict.fromkeys(
                    s_id for s_id in finding.source_ids
                    if s_id in sources_by_id
                )
            )
            if not valid_ids:
                continue

            markers: list[str] = []
            for s_id in valid_ids:
                if s_id not in cited_source_ids:
                    cited_source_ids[s_id] = None

                markers.append(f"[{s_id}]")

            rendered_findings.append(f"- {statement} {"".join(markers)}")

        if rendered_findings:
            heading = section.heading.strip()
            rendered_sections.append(
                f"### {heading}\n\n" + "\n".join(rendered_findings)
            )

    queries_block = "\n".join(f"- {q}" for q in queries_and_tool_calls)
    findings_block = "\n\n".join(rendered_sections)
    sources_block = "\n".join(
        f"- [{s_id}] {sources_by_id[s_id].title}: {sources_by_id[s_id].url}"
        for s_id in cited_source_ids
    )

    parts = [
        f"## List of Queries and Tool Calls Made\n{queries_block}",
        f"## Fully Comprehensive Findings\n\n{findings_block}",
        f"### Sources\n{sources_block}"
    ]

    return "\n\n".join(parts)


def build_researcher_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Build the research workflow graph with the given checkpointer.

    Args:
        checkpointer (Checkpointer): The checkpointer to use for the graph.
        store (BaseStore): The store to use for the graph.
        
    Returns:
        The compiled research workflow graph.
    """

    tools: list[BaseTool] = [tavily_search]

    tool_node = ToolNode(
        name="tool_node",
        tools=tools,
        messages_key="researcher_messages",
        handle_tool_errors=True
    )

    async def conduct_research_node(
            state: ResearcherState,
            config: RunnableConfig
    ) -> Command[Literal["researcher_tools", "compress_research"]]:
        return await conduct_research(
            state,
            config,
            tools=tools
        )

    async def researcher_tools_node(
            state: ResearcherState,
            config: RunnableConfig
    ) -> dict:
        return await researcher_tools(
            state,
            config,
            tool_node=tool_node,
        )

    # pyrefly: ignore [bad-specialization]
    builder = StateGraph(
        ResearcherState,
        # pyrefly: ignore [bad-argument-type]
        input_schema=ResearcherInputState,
        # pyrefly: ignore [bad-argument-type]
        output_schema=ResearcherOutputState)

    builder.add_node("conduct_research", conduct_research_node)
    builder.add_node("researcher_tools", researcher_tools_node)
    builder.add_node("compress_research", compress_research)
    builder.add_node("verify_research", verify_research)

    builder.add_edge(START, "conduct_research")
    builder.add_edge("researcher_tools", "conduct_research")
    builder.add_edge("compress_research", "verify_research")
    builder.add_edge("verify_research", END)

    return builder.compile(checkpointer=checkpointer, store=store)
