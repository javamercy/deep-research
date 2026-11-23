import logging

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig

from deep_research.prompts.research import (
    SUMMARIZE_WEBPAGE_SYSTEM_PROMPT,
    SUMMARIZE_WEBPAGE_USER_PROMPT,
)
from deep_research.schemas import ContentSummary, ProcessedSearchResult, TavilySearchResult
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)


async def process_search_result(
    search_result: TavilySearchResult,
    *,
    summarization_model: Runnable[LanguageModelInput, ContentSummary],
    config: RunnableConfig | None = None,
    max_content_length: int = 20_000,
) -> ProcessedSearchResult:
    """Summarize a search result's raw webpage content or fall back to its snippet.

    When `search_result.raw_content` is non-empty, truncates it to
    `max_content_length` characters (appending `"..."` if truncated) and invokes
    `summarization_model` to produce a `ContentSummary`. If summarization fails,
    the exception is logged and swallowed, returning an empty `summary` and
    `key_excerpts=None`. When `raw_content` is missing or empty, falls back to
    `search_result.content` with `key_excerpts=None`.

    Args:
        search_result: Raw Tavily result to process.
        summarization_model: Structured model runnable returning `ContentSummary`.
        config: Optional LangChain runnable configuration forwarded to the model.
        max_content_length: Maximum raw character count sent to the summarizer.

    Returns:
        `ProcessedSearchResult` preserving `id`, `title`, and `url`.
    """
    summary = ""
    key_excerpts = None
    if search_result.raw_content:
        webpage_content = search_result.raw_content
        if len(webpage_content) > max_content_length:
            webpage_content = webpage_content[:max_content_length] + "..."

        try:
            content_summary = await _summarize_webpage_content(
                webpage_content, config=config, summarization_model=summarization_model
            )
            summary = content_summary.summary
            key_excerpts = content_summary.key_excerpts
        except Exception as e:
            logger.error(f"Error summarizing content for URL {search_result.url}: {e}")
    else:
        summary = search_result.content

    return ProcessedSearchResult(
        id=search_result.id,
        title=search_result.title,
        url=search_result.url,
        summary=summary,
        key_excerpts=key_excerpts,
    )


async def _summarize_webpage_content(
    webpage_content: str,
    *,
    summarization_model: Runnable[LanguageModelInput, ContentSummary],
    config: RunnableConfig | None = None,
) -> ContentSummary:
    """Invoke `summarization_model` with webpage summarization prompts.

    Args:
        webpage_content: Webpage text (already truncated by caller if needed).
        summarization_model: Structured model runnable returning `ContentSummary`.
        config: Optional LangChain runnable configuration forwarded to the model.

    Returns:
        Structured `ContentSummary` extracted from the webpage text.
    """
    messages = [
        SystemMessage(content=SUMMARIZE_WEBPAGE_SYSTEM_PROMPT.format(date=get_today_str())),
        HumanMessage(content=SUMMARIZE_WEBPAGE_USER_PROMPT.format(webpage_content=webpage_content)),
    ]
    return await summarization_model.ainvoke(messages, config=config)
