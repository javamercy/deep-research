"""Researcher sub-agent workflow for iterative web search, compression, and verification."""

import logging
from collections.abc import Mapping, Sequence
from typing import Literal
from xml.etree import ElementTree

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_config
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration, get_session_id
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
    runtime: Runtime[Configuration],
    *,
    tools: list[BaseTool],
) -> Command[Literal["researcher_tools", "compress_research"]]:
    """Decide whether to call research tools or compress gathered findings.

    Reads `researcher_tool_calls` and `researcher_messages` from state, and
    requires `configurable.thread_id` in the runnable config to tag the
    OpenRouter session. Routes directly to `compress_research` without invoking
    the model once `researcher_tool_calls` reaches
    `max_researcher_tool_calls`.

    Otherwise invokes `researcher_llm_config` bound to `tools`, appends the
    returned `AIMessage` to `researcher_messages` (via the `add_messages`
    reducer), and routes to `researcher_tools` when tool calls are emitted or
    `compress_research` when the model stops searching.

    Args:
        tools: Tools bound to the researcher model for gathering evidence.

    Returns:
        `Command` targeting `"researcher_tools"` or `"compress_research"`,
        updating `researcher_messages` when the model is invoked.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    if state.get("researcher_tool_calls", 0) >= configuration.max_researcher_tool_calls:
        return Command(goto="compress_research")

    model_with_tools = init_openrouter_model(
        llm_config=configuration.researcher_llm_config,
        session_id=session_id,
    ).bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=RESEARCHER_SYSTEM_PROMPT.format(date=get_today_str())),
        *state["researcher_messages"],
    ]
    response = await model_with_tools.ainvoke(messages, config=config)

    return Command(
        goto="researcher_tools" if response.tool_calls else "compress_research",
        update={"researcher_messages": [response]},
    )


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig, *, tool_node: ToolNode
) -> dict:
    """Execute researcher tool calls and update the tool-call counter.

    Reads `researcher_messages` (requiring the latest message to be an
    `AIMessage`) and `researcher_tool_calls` from state. Executes the pending
    calls via `tool_node` and returns the resulting `ToolMessage` updates
    (appended via `add_messages`) along with the incremented total
    `researcher_tool_calls` count (plain `int`, no reducer).

    Args:
        tool_node: Preconfigured `ToolNode` that runs the researcher's tools.

    Returns:
        Partial state update containing `tool_node` message outputs and the
        updated `researcher_tool_calls` total.

    Raises:
        ValueError: If the last message in `researcher_messages` is not an
            `AIMessage`.
    """
    last_message = state["researcher_messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("Expected an AIMessage before tool execution")

    result = await tool_node.ainvoke(state, config=config)

    return {
        **result,
        "researcher_tool_calls": (
            state.get("researcher_tool_calls", 0) + len(last_message.tool_calls)
        ),
    }


async def compress_research(state: ResearcherState, runtime: Runtime[Configuration]) -> dict:
    """Compress the researcher conversation into a structured `DraftResearch`.

    Reads `researcher_messages` and `research_topic` from state, and requires
    `configurable.thread_id` in the runnable config for session tagging.
    Invokes `compression_llm_config` with `DraftResearch` structured output
    (retried up to `max_structured_output_retries`) and assigns deterministic
    finding IDs via `_assign_finding_ids`.

    Returns:
        Partial state update setting `draft_research` (validated via
        `ValidatedLastValue`).
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    compress_model = init_openrouter_structured_model(
        configuration.compression_llm_config,
        output_schema=DraftResearch,
        session_id=session_id,
        max_retries=configuration.max_structured_output_retries,
    )

    messages = [
        SystemMessage(content=COMPRESS_SYSTEM_PROMPT.format(date=get_today_str())),
        *state["researcher_messages"],
        HumanMessage(content=COMPRESS_USER_PROMPT.format(research_topic=state["research_topic"])),
    ]
    draft_research = _assign_finding_ids(await compress_model.ainvoke(messages, config=config))

    return {
        "draft_research": draft_research,
    }


async def verify_research(
    state: ResearcherState,
    runtime: Runtime[Configuration],
) -> dict:
    """Audit `draft_research` against stored sources and format cited markdown.

    Reads `draft_research` and `researcher_messages` from state, and requires
    `configurable.thread_id` in the runnable config to scope store lookups.
    Collects source IDs from `tavily_search` `ToolMessage` payloads and draft
    findings, then loads matching `ProcessedSearchResult` records via
    `get_sources_by_ids`.

    When both `draft_research.sections` and `sources_by_id` are non-empty,
    invokes `research_verifier_llm_config` with `VerifiedResearch` structured
    output and applies updates via `_apply_finding_updates`. Otherwise skips
    LLM verification and leaves `sections` empty so ungrounded findings are
    omitted. Formats the final markdown via `_format_verified_research`, keeping
    raw `[<source_id>]` hex tags for downstream synthesis.

    Returns:
        Partial state update setting `compressed_research`.

    Raises:
        ValueError: If `runtime.store` is `None` or `configurable.thread_id`
            is missing.
    """
    if runtime.store is None:
        raise ValueError("Runtime store is not available.")

    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    draft_research = state["draft_research"]

    tool_messages = [
        tool_message.text
        for tool_message in filter_messages(
            state["researcher_messages"],
            include_names=["tavily_search"],
            include_types=[ToolMessage],
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
        runtime.store,
        session_id=session_id,
        source_ids=[*tool_source_ids, *draft_source_ids],
    )

    verified_sections: list[ResearchSection] = []
    if draft_research.sections and sources_by_id:
        verifier_model = init_openrouter_structured_model(
            configuration.research_verifier_llm_config,
            session_id=session_id,
            output_schema=VerifiedResearch,
            max_retries=configuration.max_structured_output_retries,
        )

        messages = [
            SystemMessage(content=VERIFY_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())),
            HumanMessage(
                content=VERIFY_RESEARCH_USER_PROMPT.format(
                    draft_research=draft_research_to_xml(draft_research),
                    sources=processed_search_results_to_xml(list(sources_by_id.values())),
                )
            ),
        ]

        verified_research = await verifier_model.ainvoke(messages, config=config)
        verified_sections = _apply_finding_updates(draft_research, verified_research.updates)

    compressed_research = _format_verified_research(
        queries_and_tool_calls=draft_research.queries_and_tool_calls,
        sections=verified_sections,
        sources_by_id=sources_by_id,
    )

    return {"compressed_research": compressed_research}


def _assign_finding_ids(draft_research: DraftResearch) -> DraftResearch:
    """Assign deterministic 1-based `<section_idx>.<finding_idx>` IDs to findings.

    Returns a non-mutating copy of `draft_research` preserving section and
    finding order, populating each `ResearchFinding.id` before verification.
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
    """Apply verifier updates and removals to ID-tagged `draft_research` sections.

    Preserves original section and finding order without mutating
    `draft_research`. If duplicate updates target the same `finding_id`, the
    last update wins; updates targeting unknown IDs are logged and ignored.
    For `"update"` actions, `None` in `statement` or `source_ids` keeps the
    existing value. Findings with `action == "remove"` and sections left with
    no findings are omitted from the returned list.
    """
    updates_by_id = {update.finding_id: update for update in finding_updates}
    known_ids = {finding.id for section in draft_research.sections for finding in section.findings}
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
                            update.statement if update.statement is not None else finding.statement
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
            verified_sections.append(section.model_copy(update={"findings": verified_findings}))

    return verified_sections


def _extract_source_ids_from_tool_messages(messages: Sequence[str]) -> list[str]:
    """Extract `<source id="...">` attributes from XML tool-message payloads.

    Preserves encounter order without deduplicating and skips `<source>`
    elements with missing or empty `id` attributes. Malformed XML payloads (such
    as tool error messages) are logged and skipped without raising.
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
    sources_by_id: Mapping[str, ProcessedSearchResult],
) -> str:
    """Render verified findings and cited sources as a three-section Markdown report.

    Emits `## List of Queries and Tool Calls Made`, `## Fully Comprehensive
    Findings`, and `### Sources`. Strips whitespace from headings and
    statements, omits findings with blank statements or no `source_id` present
    in `sources_by_id`, and omits sections that have no remaining findings.
    Deduplicates each finding's valid IDs in original order, keeps raw
    `[<source_id>]` hex brackets inline, and orders `### Sources` entries by
    first citation appearance across rendered findings.
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
                dict.fromkeys(s_id for s_id in finding.source_ids if s_id in sources_by_id)
            )
            if not valid_ids:
                continue

            markers: list[str] = []
            for s_id in valid_ids:
                if s_id not in cited_source_ids:
                    cited_source_ids[s_id] = None

                markers.append(f"[{s_id}]")

            rendered_findings.append(f"- {statement} {''.join(markers)}")

        if rendered_findings:
            heading = section.heading.strip()
            rendered_sections.append(f"### {heading}\n\n" + "\n".join(rendered_findings))

    queries_block = "\n".join(f"- {q}" for q in queries_and_tool_calls)
    findings_block = "\n\n".join(rendered_sections)
    sources_block = "\n".join(
        f"- [{s_id}] {sources_by_id[s_id].title}: {sources_by_id[s_id].url}"
        for s_id in cited_source_ids
    )

    parts = [
        f"## List of Queries and Tool Calls Made\n{queries_block}",
        f"## Fully Comprehensive Findings\n\n{findings_block}",
        f"### Sources\n{sources_block}",
    ]

    return "\n\n".join(parts)


def build_researcher_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Compile the researcher `StateGraph` (`ResearcherInputState` -> `ResearcherOutputState`).

    Uses `ResearcherState` internally with `Configuration` as `context_schema`.
    Fixed edges connect `START -> conduct_research`, `researcher_tools ->
    conduct_research`, `compress_research -> verify_research`, and
    `verify_research -> END`, while `conduct_research` handles routing between
    `"researcher_tools"` (running `tavily_search` in a `ToolNode` with
    `handle_tool_errors=True`) and `"compress_research"`.

    `checkpointer` may be `None` when invoked as a sub-agent or in LangGraph
    Studio. Although `store` defaults to `None` for Studio/sub-agent wiring, a
    `BaseStore` must be provided at compile time or via the parent runtime so
    `tavily_search` and `verify_research` can access session sources.

    Args:
        checkpointer: Checkpointer for graph state persistence, or `None`.
        store: Optional `BaseStore` for session-scoped search source caching.

    Returns:
        Compiled researcher workflow graph.
    """
    tools: list[BaseTool] = [tavily_search]

    tool_node = ToolNode(
        name="tool_node", tools=tools, messages_key="researcher_messages", handle_tool_errors=True
    )

    async def conduct_research_node(
        state: ResearcherState,
        runtime: Runtime[Configuration],
    ) -> Command[Literal["researcher_tools", "compress_research"]]:
        return await conduct_research(state, runtime, tools=tools)

    async def researcher_tools_node(state: ResearcherState, config: RunnableConfig) -> dict:
        return await researcher_tools(
            state,
            config,
            tool_node=tool_node,
        )

    # Pyrefly does not recognize TypedDict schemas as satisfying LangGraph's StateLike bound.
    # pyrefly: ignore [bad-specialization]
    builder = StateGraph[
        ResearcherState, Configuration, ResearcherInputState, ResearcherOutputState
    ](
        ResearcherState,
        context_schema=Configuration,
        input_schema=ResearcherInputState,
        output_schema=ResearcherOutputState,
    )

    builder.add_node("conduct_research", conduct_research_node)
    builder.add_node("researcher_tools", researcher_tools_node)
    builder.add_node("compress_research", compress_research)
    builder.add_node("verify_research", verify_research)

    builder.add_edge(START, "conduct_research")
    builder.add_edge("researcher_tools", "conduct_research")
    builder.add_edge("compress_research", "verify_research")
    builder.add_edge("verify_research", END)

    return builder.compile(checkpointer=checkpointer, store=store)
