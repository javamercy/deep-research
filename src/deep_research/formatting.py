from xml.etree import ElementTree

from deep_research.schemas import ResearchPlan


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
