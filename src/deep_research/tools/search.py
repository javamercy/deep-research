from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool

from deep_research.search.tavily import deduplicate_search_results, format_search_results, process_search_results, tavily_search_multiple


@tool(parse_docstring=True)
async def tavily_search(
        query: str,
        max_results: Annotated[int, InjectedToolArg] = 5,
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Perform a search using the Tavily API and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        topic: Topic filter for search results (default: "general")

    Returns:
        Formatted string of search results with summaries
    """

    search_results = await tavily_search_multiple(
        search_queries=[query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    unique_results = deduplicate_search_results(search_results[0].get("results", []))
    processed_results = await process_search_results(unique_results)
    return format_search_results(processed_results)
