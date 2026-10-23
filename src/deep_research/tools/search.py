import asyncio
import logging
from typing import Annotated, Literal
from weakref import WeakValueDictionary

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore

from deep_research.configuration import Configuration
from deep_research.formatting import processed_search_results_to_xml
from deep_research.memory.sources import get_source_by_id, store_source
from deep_research.openrouter import init_openrouter_structured_model
from deep_research.schemas import ContentSummary, ProcessedSearchResult, TavilySearchResult
from deep_research.search.processing import process_search_result
from deep_research.search.tavily import search_tavily

logger = logging.getLogger(__name__)

_SOURCE_LOCKS: WeakValueDictionary[tuple[str, str], asyncio.Lock] = WeakValueDictionary()


def _get_source_lock(session_id: str, source_id: str) -> asyncio.Lock:
    """Get or create an in-memory asyncio.Lock scoped to a session and source ID.

    Args:
        session_id: Unique identifier for the current research session.
        source_id: The 8-character hex ID of the search result.

    Returns:
        The shared asyncio.Lock instance for the (session_id, source_id) pair.
    """

    key = (session_id, source_id)
    lock = _SOURCE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _SOURCE_LOCKS[key] = lock

    return lock


async def _get_or_process_source(
        store: BaseStore,
        result: TavilySearchResult,
        *,
        session_id: str,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
        max_content_length: int,
) -> ProcessedSearchResult:
    """Retrieve an existing processed source from the store or summarize and store it under a lock.

    Args:
        store: The LangGraph BaseStore instance.
        result: Raw TavilySearchResult to look up or process.
        session_id: Unique identifier for the current research session.
        summarization_model: Structured model runnable for summarizing raw webpage content.
        max_content_length: Maximum character length of raw content to pass to the summarizer.

    Returns:
        The cached or newly summarized ProcessedSearchResult.
    """

    lock = _get_source_lock(session_id, result.id)
    async with lock:
        existing = await get_source_by_id(store, session_id=session_id, source_id=result.id)
        if existing is not None:
            return existing

        processed_result = await process_search_result(
            result,
            summarization_model=summarization_model,
            max_content_length=max_content_length
        )
        await store_source(store, processed_result, session_id=session_id)
        return processed_result


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

    processed_results = await asyncio.gather(
        *(
            _get_or_process_source(
                store,
                search_result,
                session_id=configuration.session_id,
                summarization_model=summarization_model,
                max_content_length=configuration.max_content_length,
            )
            for search_result in search_results
        )
    )

    return processed_search_results_to_xml(list(processed_results))
