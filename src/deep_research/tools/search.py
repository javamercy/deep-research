import asyncio
import logging
from typing import Annotated, Literal
from weakref import WeakValueDictionary

from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore

from deep_research.configuration import Configuration, get_session_id
from deep_research.formatting import processed_search_results_to_xml
from deep_research.memory.sources import get_source_by_id, store_source
from deep_research.openrouter import init_openrouter_structured_model
from deep_research.schemas import ContentSummary, ProcessedSearchResult, TavilySearchResult
from deep_research.search.processing import process_search_result
from deep_research.search.tavily import search_tavily

logger = logging.getLogger(__name__)

_SOURCE_LOCKS: WeakValueDictionary[tuple[str, str], asyncio.Lock] = WeakValueDictionary()


def _get_source_lock(session_id: str, source_id: str) -> asyncio.Lock:
    """Return the shared `asyncio.Lock` for a `(session_id, source_id)` pair.

    Looks up or creates a lock in the module-level `WeakValueDictionary`, so
    unused entries are automatically reclaimed once all holders release them.

    Args:
        session_id: Research session identifier.
        source_id: Deterministic 8-character hex source identifier.

    Returns:
        The `asyncio.Lock` coordinating concurrent processing of that source.
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
    config: RunnableConfig | None = None,
) -> ProcessedSearchResult:
    """Return a cached processed source or summarize and persist it under a lock.

    Acquires the per-`(session_id, result.id)` lock so concurrent searches for
    the same URL do not summarize it multiple times. On a cache hit in `store`,
    returns the stored `ProcessedSearchResult`; otherwise runs
    `process_search_result`, persists it with `store_source`, and returns it.

    Args:
        store: LangGraph store holding session sources.
        result: Raw Tavily search result to fetch or process.
        session_id: Research session identifier scoping the store and lock.
        summarization_model: Structured model runnable returning `ContentSummary`.
        max_content_length: Maximum raw character count sent to the summarizer.
        config: Optional LangChain runnable configuration forwarded to the model.

    Returns:
        The cached or newly processed `ProcessedSearchResult`.
    """
    lock = _get_source_lock(session_id, result.id)
    async with lock:
        existing = await get_source_by_id(store, session_id=session_id, source_id=result.id)
        if existing is not None:
            return existing

        processed_result = await process_search_result(
            result,
            summarization_model=summarization_model,
            max_content_length=max_content_length,
            config=config,
        )
        await store_source(store, processed_result, session_id=session_id)
        return processed_result


@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    runtime: ToolRuntime[Configuration],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web and return summarized sources with citation IDs.

    Use one focused search query per call—start broad for initial exploration
    and follow up with narrower queries to fill specific gaps. Each result is
    returned as a `<source id="...">` entry with its URL, title, summary, and
    optional key excerpts.

    Args:
        query: A targeted web search query string.
        runtime: Injected execution context; hidden from the model.
        max_results: Injected limit on results returned per call; hidden from the model.
        topic: Injected Tavily topic category; hidden from the model.

    Returns:
        XML `<sources>` string containing deduplicated `ProcessedSearchResult` entries.

    Raises:
        ValueError: If `runtime.store` is `None`, `thread_id` is missing from config,
            or `TAVILY_API_KEY` is unset.
    """
    if runtime.store is None:
        raise ValueError("ToolRuntime store is not available.")

    configuration = Configuration.from_runtime(runtime, runtime.config)
    session_id = get_session_id(runtime.config)

    summarization_model = init_openrouter_structured_model(
        configuration.summarization_llm_config,
        output_schema=ContentSummary,
        max_retries=configuration.max_structured_output_retries,
        session_id=session_id,
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
                runtime.store,
                search_result,
                session_id=session_id,
                summarization_model=summarization_model,
                max_content_length=configuration.max_content_length,
                config=runtime.config,
            )
            for search_result in search_results
        )
    )

    return processed_search_results_to_xml(list(processed_results))
