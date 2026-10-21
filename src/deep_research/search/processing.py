import logging

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from deep_research.prompts.research import SUMMARIZE_WEBPAGE_SYSTEM_PROMPT, SUMMARIZE_WEBPAGE_USER_PROMPT
from deep_research.schemas import ContentSummary, FrozenBaseModel
from deep_research.search.tavily import TavilySearchResult
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)


class ProcessedSearchResult(FrozenBaseModel):
    """Schema for processed search result with summary and key excerpts."""

    id: str
    url: str
    title: str
    summary: str
    key_excerpts: str | None = None


async def process_search_results(
        search_results: list[TavilySearchResult],
        *,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
        max_content_length: int = 20_000
) -> dict[str, ProcessedSearchResult]:
    """Process search results by summarizing content where available.

    Args:
        search_results: List of TavilySearchResult objects
        summarization_model: Runnable model for summarizing content
        max_content_length: Maximum length of content to summarize

    Returns:
        Dictionary of ProcessedSearchResult objects keyed by result ID
    """

    processed_results: dict[str, ProcessedSearchResult] = {}

    for result in search_results:
        summary = ""
        key_excerpts = None
        if result.raw_content:
            webpage_content = result.raw_content
            if len(webpage_content) > max_content_length:
                webpage_content = webpage_content[:max_content_length] + "..."

            try:
                content_summary = await _summarize_webpage_content(
                    webpage_content,
                    summarization_model=summarization_model
                )
                summary = content_summary.summary
                key_excerpts = content_summary.key_excerpts
            except Exception as e:
                logger.error(f"Error summarizing content for URL {result.url}: {e}")
        else:
            summary = result.content

        processed_results[result.id] = ProcessedSearchResult(
            id=result.id,
            title=result.title,
            url=result.url,
            summary=summary,
            key_excerpts=key_excerpts
        )

    return processed_results


async def _summarize_webpage_content(
        webpage_content: str,
        *,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
) -> ContentSummary:
    messages = [
        SystemMessage(
            content=SUMMARIZE_WEBPAGE_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        HumanMessage(
            content=SUMMARIZE_WEBPAGE_USER_PROMPT.format(webpage_content=webpage_content)
        )
    ]
    return await summarization_model.ainvoke(messages)
