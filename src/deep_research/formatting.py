from collections.abc import Mapping, Sequence
from xml.etree import ElementTree

from deep_research.schemas import DraftResearch, ResearchPlan, ResearchSection
from deep_research.search.processing import ProcessedSearchResult


def research_plan_to_xml(plan: ResearchPlan) -> str:
    """Convert research plan to XML format.

    The root element is included, and each task is represented as a child element with its attributes.
    Empty elements are represented with opening and closing tags.

    Args:
        plan (ResearchPlan): The research plan to convert.

    Returns:
        The XML string representation of the research plan.
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


def processed_search_results_to_xml(processed_results: list[ProcessedSearchResult]) -> str:
    """Convert processed search results to XML format."""

    root = ElementTree.Element("search_results")

    for result in processed_results:
        result_element = ElementTree.SubElement(root, "result", attrib={"id": result.id})
        ElementTree.SubElement(result_element, "url").text = result.url
        ElementTree.SubElement(result_element, "title").text = result.title
        ElementTree.SubElement(result_element, "summary").text = result.summary

        if result.key_excerpts is not None:
            ElementTree.SubElement(result_element, "key_excerpts").text = result.key_excerpts

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def draft_research_to_xml(research: DraftResearch) -> str:
    root = ElementTree.Element("draft_research")
    sections_element = ElementTree.SubElement(root, "sections")

    for section in research.sections:
        section_element = ElementTree.SubElement(sections_element, "section")
        ElementTree.SubElement(section_element, "heading").text = section.heading

        findings_element = ElementTree.SubElement(section_element, "findings")
        for finding in section.findings:

            finding_element = ElementTree.SubElement(findings_element, "finding")
            ElementTree.SubElement(finding_element, "statement").text = finding.statement

            source_ids_element = ElementTree.SubElement(finding_element, "source_ids")
            for s_id in finding.source_ids:
                ElementTree.SubElement(source_ids_element, "source_id").text = s_id

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", short_empty_elements=False)


def format_verified_research(
        *,
        queries_and_tool_calls: Sequence[str],
        sections: Sequence[ResearchSection],
        sources_by_id: Mapping[str, ProcessedSearchResult]
) -> str:
    """Render verified findings with sequential ([1], [2], ...) citations."""

    citation_numbers: dict[str, int] = {}
    rendered_sections: list[str] = []

    for section in sections:
        rendered_findings: list[str] = []
        for finding in section.findings:
            statement = finding.statement.strip()
            if not statement:
                continue

            valid_ids = list(
                dict.fromkeys(
                    s_id for s_id in finding.source_ids if s_id in sources_by_id
                )
            )
            if not valid_ids:
                continue

            for s_id in valid_ids:
                if s_id not in citation_numbers:
                    citation_numbers[s_id] = len(citation_numbers) + 1

            markers = "".join(f"[{citation_numbers[s_id]}]" for s_id in valid_ids)
            rendered_findings.append(f"- {statement} {markers}")

        if rendered_findings:
            heading = section.heading.strip()
            rendered_sections.append(
                f"### {heading}\n" + "\n".join(rendered_findings)
            )

    queries_block = "\n".join(f"- {q}" for q in queries_and_tool_calls)
    findings_block = "\n\n".join(rendered_sections)
    sources_block = "\n".join(
        f"- [{num}] {sources_by_id[s_id].title}: {sources_by_id[s_id].url}"
        for s_id, num in citation_numbers.items()
    )

    parts = [
        f"## List of Queries and Tool Calls Made\n{queries_block}",
        f"## Fully Comprehensive Findings\n\n{findings_block}",
        f"### Sources\n{sources_block}"
    ]

    return "\n\n".join(parts)
