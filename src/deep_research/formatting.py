from xml.etree import ElementTree

from deep_research.schemas import DraftResearch, ProcessedSearchResult, ResearchPlan


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


def processed_search_results_to_xml(sources: list[ProcessedSearchResult]) -> str:
    """Convert processed search results to an XML `<sources>` string.

    Args:
        sources: List of ProcessedSearchResult objects to serialize.

    Returns:
        The XML string representation of the sources.
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
    """Convert structured draft research findings to an XML `<draft_research>` string.

    Args:
        research: The DraftResearch object to serialize.

    Returns:
        The XML string representation of the draft research sections and findings.
    """

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
