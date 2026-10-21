import logging
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolRuntime

from deep_research.configuration import Configuration
from deep_research.formatting import processed_search_results_to_xml
from deep_research.memory.sources import StoredSearchResult, store_sources
from deep_research.openrouter import init_openrouter_structured_model
from deep_research.schemas import ContentSummary
from deep_research.search.processing import process_search_results
from deep_research.search.tavily import search_tavily

logger = logging.getLogger(__name__)


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

    store = runtime.store
    if store is None:
        raise ValueError("ToolRuntime store is not available.")

    configuration = Configuration.from_runnable_config(runtime.config)

    summarization_model = init_openrouter_structured_model(
        configuration.summarization_llm_config,
        output_schema=ContentSummary,
        max_retries=configuration.max_structured_output_retries,
        session_id=configuration.session_id,
    )

    search_results = await search_tavily(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    processed_results = await process_search_results(
        search_results=search_results,
        summarization_model=summarization_model,
        max_content_length=configuration.max_content_length,
    )

    stored_results: list[StoredSearchResult] = []
    for search_result in search_results:
        processed_result = processed_results[search_result.id]
        stored_results.append(
            StoredSearchResult(
                id=processed_result.id,
                url=processed_result.url,
                title=processed_result.title,
                raw_content=search_result.raw_content,
                summary=processed_result.summary,
                key_excerpts=processed_result.key_excerpts
            )
        )

    await store_sources(
        store,
        configuration.session_id,
        stored_results
    )

    return processed_search_results_to_xml(list(processed_results.values()))
