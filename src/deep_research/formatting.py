"""XML serialization helpers for inserting structured data into LLM prompts.

All functions build elements with `xml.etree.ElementTree` and serialize with
`short_empty_elements=False`. Standard XML character escaping applies to element
text and attributes, without tag stripping or prompt-injection sanitization;
prompts rely on outer XML tags paired with `<data_boundaries>` instructions to
isolate untrusted content.
"""

from collections.abc import Sequence
from xml.etree import ElementTree

from deep_research.schemas import (
    DraftResearch,
    FinalReportReview,
    ProcessedSearchResult,
    ResearchPlan,
    TavilySearchResult,
)


def landscape_context_to_xml(results: Sequence[TavilySearchResult]) -> str:
    """Serialize reconnaissance search hits into an indented `<landscape_context>` XML string.

    Preserves input order and renders each hit as `<result>` with `<title>`,
    `<url>`, and `<snippet>` (from `TavilySearchResult.content`). Omits `id`,
    `score`, and `raw_content` so unverified scoping snippets stay distinct from
    citable `<sources>` persisted in the session store.

    Element text is XML-escaped by `ElementTree`, with no prompt-injection
    sanitization; callers rely on `<landscape_context>` tags paired with
    `<data_boundaries>` instructions to isolate untrusted web content.

    Args:
        results: Deduplicated reconnaissance search hits from Tavily.

    Returns:
        Indented XML string rooted at `<landscape_context>` using explicit
        closing tags.
    """
    root = ElementTree.Element("landscape_context")

    for result in results:
        result_element = ElementTree.SubElement(root, "result")
        ElementTree.SubElement(result_element, "title").text = result.title
        ElementTree.SubElement(result_element, "url").text = result.url
        ElementTree.SubElement(result_element, "snippet").text = result.content

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def research_plan_to_xml(plan: ResearchPlan) -> str:
    """Serialize a `ResearchPlan` into an indented `<research_plan>` XML string.

    Preserves task order with 1-based `index` attributes and preserves
    `depends_on` order inside `<depends_on>`. Empty containers use explicit
    opening and closing tags. Element text and attributes are XML-escaped by
    `ElementTree`.

    Args:
        plan: Structured research plan to render for scoping or supervisor
            prompts.

    Returns:
        Indented XML string rooted at `<research_plan>` containing `<tasks>`
        and `<task>` elements.
    """
    root = ElementTree.Element("research_plan")
    tasks_element = ElementTree.SubElement(root, "tasks")

    for i, task in enumerate(plan.tasks, start=1):
        task_element = ElementTree.SubElement(tasks_element, "task", attrib={"index": str(i)})
        ElementTree.SubElement(task_element, "id").text = task.id
        ElementTree.SubElement(task_element, "objective").text = task.objective
        ElementTree.SubElement(task_element, "success_criteria").text = task.success_criteria

        depends_on_element = ElementTree.SubElement(task_element, "depends_on")
        for depends_id in task.depends_on:
            ElementTree.SubElement(depends_on_element, "task_id").text = depends_id

        ElementTree.SubElement(task_element, "status").text = task.status

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def processed_search_results_to_xml(sources: list[ProcessedSearchResult]) -> str:
    """Serialize processed search results into an indented `<sources>` XML string.

    Preserves input order and renders each item as `<source id="...">` with
    `<url>`, `<title>`, `<summary>`, and optional `<key_excerpts>` (omitted
    only when `key_excerpts` is `None`).

    Element text and attributes are XML-escaped by `ElementTree`, with no
    prompt-injection sanitization; callers rely on `<sources>` tags to fence
    untrusted web content in prompts.

    Args:
        sources: Processed search results to format for tool output or
            verification prompts.

    Returns:
        Indented XML string rooted at `<sources>` using explicit closing tags.
    """
    root = ElementTree.Element("sources")

    for source in sources:
        result_element = ElementTree.SubElement(root, "source", attrib={"id": source.id})
        ElementTree.SubElement(result_element, "url").text = source.url
        ElementTree.SubElement(result_element, "title").text = source.title
        ElementTree.SubElement(result_element, "summary").text = source.summary

        if source.key_excerpts is not None:
            ElementTree.SubElement(result_element, "key_excerpts").text = source.key_excerpts

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def draft_research_to_xml(research: DraftResearch) -> str:
    """Serialize a `DraftResearch` into an indented `<draft_research>` XML string.

    Preserves section, finding, and `source_ids` order, rendering each finding
    as `<finding id="...">` with `<statement>` and `<source_ids>`. Omits
    `queries_and_tool_calls` from the output.

    Element text and attributes are XML-escaped by `ElementTree`. Because
    findings summarize untrusted web sources, callers use `<draft_research>`
    tags to isolate this content in verification prompts.

    Args:
        research: Compressed research draft with assigned finding IDs.

    Returns:
        Indented XML string rooted at `<draft_research>` using explicit
        closing tags.
    """
    root = ElementTree.Element("draft_research")
    sections_element = ElementTree.SubElement(root, "sections")

    for section in research.sections:
        section_element = ElementTree.SubElement(sections_element, "section")
        ElementTree.SubElement(section_element, "heading").text = section.heading

        findings_element = ElementTree.SubElement(section_element, "findings")
        for finding in section.findings:
            finding_element = ElementTree.SubElement(
                findings_element, "finding", attrib={"id": finding.id}
            )
            ElementTree.SubElement(finding_element, "statement").text = finding.statement

            source_ids_element = ElementTree.SubElement(finding_element, "source_ids")
            for s_id in finding.source_ids:
                ElementTree.SubElement(source_ids_element, "source_id").text = s_id

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def final_report_review_to_xml(review: FinalReportReview) -> str:
    """Serialize a `FinalReportReview` into a `<report_review>` XML string.

    Omits `approved` from the output. When `review.issues` is `None`, returns
    `"<report_review></report_review>"` with no `<issues>` element; otherwise
    preserves issue order inside an indented `<issues>` list of `<issue>`
    elements (`<section>`, `<description>`, `<fix_suggestion>`).

    Element text is XML-escaped by `ElementTree` for insertion into writer
    revision prompts.

    Args:
        review: Reviewer output containing any identified report issues.

    Returns:
        XML string rooted at `<report_review>` using explicit closing tags.
    """
    root = ElementTree.Element("report_review")
    if review.issues is None:
        return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)

    issues_element = ElementTree.SubElement(root, "issues")

    for issue in review.issues:
        issue_element = ElementTree.SubElement(issues_element, "issue")
        ElementTree.SubElement(issue_element, "section").text = issue.section
        ElementTree.SubElement(issue_element, "description").text = issue.description
        ElementTree.SubElement(issue_element, "fix_suggestion").text = issue.fix_suggestion

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)
