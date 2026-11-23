"""Final report writer and review workflow with citation numbering."""

import logging
import re
from typing import Literal
from xml.etree import ElementTree

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Command

from deep_research.configuration import Configuration, get_session_id
from deep_research.formatting import final_report_review_to_xml
from deep_research.memory.sources import get_sources_by_ids
from deep_research.openrouter import init_openrouter_model, init_openrouter_structured_model
from deep_research.prompts.writer import (
    GENERATE_FINAL_REPORT_SYSTEM_PROMPT,
    GENERATE_FINAL_REPORT_USER_PROMPT,
    REVIEW_FINAL_REPORT_SYSTEM_PROMPT,
    REVIEW_FINAL_REPORT_USER_PROMPT,
)
from deep_research.schemas import FinalReportIssue, FinalReportReview, ProcessedSearchResult
from deep_research.state import WriterInputState, WriterOutputState, WriterState
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)

_CITATION_GROUP_RE = re.compile(r"\[([0-9a-fA-F]{8}(?:\s*[,;]\s*[0-9a-fA-F]{8})*)\]")
_HEX_ID_RE = re.compile(r"[0-9a-fA-F]{8}")


async def write_report(
    state: WriterState,
    runtime: Runtime[Configuration],
) -> dict:
    """Synthesize or revise the Markdown final report from verified findings.

    Reads `research_brief`, `compressed_researches`, and optional `final_report`
    and `report_review` from `WriterState`, requiring `configurable.thread_id`
    in the runnable config for session tagging. Formats `compressed_researches`
    as XML and invokes `writer_llm_config`; when both `final_report` and
    `report_review` are present, appends them as a revision prompt.

    Returns:
        Partial state update setting `final_report` with `[source_id]` hex
        tags intact for downstream review and citation formatting.
    """
    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    writer_model = init_openrouter_model(
        llm_config=configuration.writer_llm_config,
        session_id=session_id,
    )

    messages = [
        SystemMessage(content=GENERATE_FINAL_REPORT_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=GENERATE_FINAL_REPORT_USER_PROMPT.format(
                research_question=state["research_brief"],
                findings=_compressed_researches_to_xml(state["compressed_researches"]),
            )
        ),
    ]

    report = state.get("final_report")
    review_xml = state.get("report_review")
    if report is not None and review_xml is not None:
        messages.append(
            HumanMessage(content=f"<final_report>\n{report}\n</final_report>\n\n{review_xml}")
        )

    response = await writer_model.ainvoke(messages, config=config)

    return {
        "final_report": response.text,
    }


def _compressed_researches_to_xml(compressed_researches: list[tuple[str, str]]) -> str:
    """Serialize `(topic, content)` research tuples into an indented `<findings>` XML string.

    Preserves tuple order and relies on `ElementTree` text escaping to isolate
    untrusted finding content inside `<topic>` and `<content>` elements, using
    explicit closing tags even when empty.
    """
    root = ElementTree.Element("findings")

    for finding in compressed_researches:
        topic, content = finding
        finding_element = ElementTree.SubElement(root, "finding")
        ElementTree.SubElement(finding_element, "topic").text = topic
        ElementTree.SubElement(finding_element, "content").text = content

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


async def review_report(
    state: WriterState,
    runtime: Runtime[Configuration],
) -> Command[Literal["write_report", "__end__"]]:
    """Audit `final_report` against findings and finalize citations when approved.

    Reads `final_report` and `writer_iterations` from `WriterState` (plus
    `research_brief` and `compressed_researches` when below the iteration
    limit), and requires `configurable.thread_id` in the runnable config.
    Extracts cited source IDs and fetches their metadata from `runtime.store`
    via `get_sources_by_ids`.

    When `writer_iterations` reaches `max_report_reviewer_iterations`, skips LLM
    review, logs any unknown source IDs, formats citations via
    `_format_report_citations`, and exits with the finalized `final_report` and
    an `AIMessage` appended to `messages` (via `add_messages`).

    Otherwise invokes `report_reviewer_llm_config` with `FinalReportReview`
    structured output. If the review approves yet unknown source IDs remain,
    appends a synthetic `"Sources"` issue. When no issues remain, formats
    citations and routes to `"__end__"`; otherwise serializes the failed review
    to `report_review` XML, increments `writer_iterations` (plain `int`, no
    reducer), and routes back to `"write_report"`.

    Returns:
        `Command` targeting `"__end__"` on finalization or `"write_report"`
        for revision, with the corresponding state updates.

    Raises:
        ValueError: If `runtime.store` is `None` or `configurable.thread_id`
            is missing.
    """
    if runtime.store is None:
        raise ValueError("Runtime store is not  available.")

    config = get_config()
    configuration = Configuration.from_runtime(runtime, config)
    session_id = get_session_id(config)

    report = state["final_report"]

    report_source_ids = _extract_source_ids_from_report(report)
    sources_by_id = await get_sources_by_ids(
        runtime.store,
        session_id=session_id,
        source_ids=report_source_ids,
    )
    unknown_source_ids = [s_id for s_id in report_source_ids if s_id not in sources_by_id]

    if state.get("writer_iterations", 0) >= configuration.max_report_reviewer_iterations:
        logger.warning(
            "Maximum writer iterations reached. Skipping verification and finalizing report."
        )

        if unknown_source_ids:
            logger.warning(f"Unknown source IDs found in the report: {unknown_source_ids}.")

        report = _format_report_citations(report, report_source_ids, sources_by_id)

        return Command(
            goto="__end__",
            update={
                "final_report": report,
                "messages": [AIMessage(content=report)],
            },
        )

    reviewer_model = init_openrouter_structured_model(
        configuration.report_reviewer_llm_config,
        session_id=session_id,
        output_schema=FinalReportReview,
        max_retries=configuration.max_structured_output_retries,
    )

    messages = [
        SystemMessage(content=REVIEW_FINAL_REPORT_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(
            content=REVIEW_FINAL_REPORT_USER_PROMPT.format(
                research_question=state["research_brief"],
                findings=_compressed_researches_to_xml(state["compressed_researches"]),
            )
        ),
        HumanMessage(content=f"<final_report>\n{report}\n</final_report>"),
    ]

    review = await reviewer_model.ainvoke(messages, config=config)

    issues = review.issues or []
    if review.approved and unknown_source_ids:
        issues.append(
            FinalReportIssue(
                section="Sources",
                description=f"Unknown source IDs found in the report: {unknown_source_ids}.",
                fix_suggestion="Remove or correct these source IDs in the report.",
            )
        )

    if not issues:
        formatted_report = _format_report_citations(report, report_source_ids, sources_by_id)
        return Command(
            goto="__end__",
            update={
                "final_report": formatted_report,
                "messages": [AIMessage(content=formatted_report)],
            },
        )

    updated_review = FinalReportReview(approved=False, issues=issues)
    return Command(
        goto="write_report",
        update={
            "report_review": final_report_review_to_xml(updated_review),
            "writer_iterations": state.get("writer_iterations", 0) + 1,
        },
    )


def _extract_source_ids_from_report(report: str) -> list[str]:
    """Extract unique lowercase 8-character hex source IDs from bracketed citations.

    Matches single `[xxxxxxxx]` and comma- or semicolon-delimited citation
    groups (for example `[a1b2c3d4, e5f60718]`), ignores numbered `[1]` or
    unbracketed hex strings, and preserves first-appearance order.
    """
    source_ids = (
        s_id.lower()
        for group in _CITATION_GROUP_RE.findall(report)
        for s_id in _HEX_ID_RE.findall(group)
    )
    return list(dict.fromkeys(source_ids))


def _format_report_citations(
    report: str, report_source_ids: list[str], sources_by_id: dict[str, ProcessedSearchResult]
) -> str:
    """Replace inline `[source_id]` tags with sequential `[n]` markers and append sources.

    Filters `report_source_ids` (expected as unique lowercase IDs from
    `_extract_source_ids_from_report`) to IDs present in `sources_by_id`,
    preserving order for 1-based numbering. Within each matched citation group,
    known IDs are deduplicated in group order and rendered as adjacent brackets
    (`[1][2]`), while groups containing only unknown IDs are removed and any
    resulting spaces before punctuation (`.,;:!?`) are collapsed. Appends a
    `### Sources` bibliography only when at least one valid source was cited.
    """
    valid_source_ids = [s_id for s_id in report_source_ids if s_id in sources_by_id]

    index_by_id = {source_id: index for index, source_id in enumerate(valid_source_ids, start=1)}

    def _replace_citation_group(match: re.Match[str]) -> str:
        raw_ids = (s_id.lower() for s_id in _HEX_ID_RE.findall(match.group(1)))
        valid_ids = list(dict.fromkeys(s_id for s_id in raw_ids if s_id in index_by_id))
        if not valid_ids:
            return ""

        return "".join(f"[{index_by_id[s_id]}]" for s_id in valid_ids)

    formatted_report = _CITATION_GROUP_RE.sub(_replace_citation_group, report)
    formatted_report = re.sub(r" +([.,;:!?])", r"\1", formatted_report)

    if not valid_source_ids:
        return formatted_report

    sources_lines = "\n".join(
        f"- [{index}] {sources_by_id[source_id].title}: {sources_by_id[source_id].url}"
        for index, source_id in enumerate(valid_source_ids, start=1)
    )

    return f"{formatted_report}\n\n### Sources\n\n{sources_lines}"


def build_writer_graph(checkpointer: Checkpointer, store: BaseStore | None = None):
    """Compile the writer `StateGraph` (`WriterInputState` -> `WriterOutputState`).

    Uses `WriterState` internally with `Configuration` as `context_schema`.
    Fixed edges connect `START -> write_report` and `write_report ->
    review_report`, while `review_report` handles routing either back to
    `"write_report"` for revisions or to `"__end__"` once approved or when
    `max_report_reviewer_iterations` is reached.

    `checkpointer` may be `None` when invoked as a sub-agent or in LangGraph
    Studio. Although `store` defaults to `None` for Studio/sub-agent wiring, a
    `BaseStore` must be provided at compile time or via the parent runtime
    because `review_report` requires `runtime.store` for source lookups.

    Args:
        checkpointer: Checkpointer for graph state persistence, or `None`.
        store: Optional `BaseStore` providing session-scoped search sources.

    Returns:
        Compiled writer workflow graph.
    """
    # pyrefly: ignore[bad-specialization]
    builder = StateGraph(
        WriterState,
        context_schema=Configuration,
        # pyrefly: ignore[bad-argument-type]
        input_schema=WriterInputState,
        # pyrefly: ignore[bad-argument-type]
        output_schema=WriterOutputState,
    )

    builder.add_node("write_report", write_report)
    builder.add_node("review_report", review_report)

    builder.add_edge(START, "write_report")
    builder.add_edge("write_report", "review_report")

    return builder.compile(checkpointer=checkpointer, store=store)
