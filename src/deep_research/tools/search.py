from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolRuntime

from deep_research.configuration import Configuration
from deep_research.models import init_openrouter_structured_model
from deep_research.schemas import ContentSummary
from deep_research.search.processing import process_search_results
from deep_research.search.tavily import search_tavily


@tool(parse_docstring=True)
async def tavily_search(
        query: str,
        runtime: ToolRuntime,
        max_results: Annotated[int, InjectedToolArg] = 5,
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Perform a search using the Tavily API and return formatted results.

    Args:
        query: The search query string
        runtime: ToolRuntime object providing context for the tool execution
        max_results: Maximum number of results to return (default: 5)
        topic: Topic filter for search results (default: "general")

    Returns:
        Formatted string of search results with summaries
    """

    configuration = Configuration.from_runnable_config(runtime.config)

    summarization_model = init_openrouter_structured_model(
        configuration.summarization_llm_config,
        output_schema=ContentSummary,
        max_retries=configuration.max_structured_output_retries,
        session_id=configuration.session_id,
    )

    unique_results = await search_tavily(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    processed_results = await process_search_results(
        unique_results=unique_results,
        summarization_model=summarization_model,
        max_content_length=configuration.max_content_length,
    )
    return _format_search_results(processed_results)


def _format_search_results(processed_results: dict) -> str:
    if not processed_results:
        raise ValueError("No valid search results found. Please try different search queries.")

    formatted_output = "Search Results:"

    for i, (url, result) in enumerate(processed_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output
