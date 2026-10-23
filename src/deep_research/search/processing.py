import logging

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from deep_research.prompts.research import SUMMARIZE_WEBPAGE_SYSTEM_PROMPT, SUMMARIZE_WEBPAGE_USER_PROMPT
from deep_research.schemas import ContentSummary, ProcessedSearchResult, TavilySearchResult
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)


async def process_search_result(
        search_result: TavilySearchResult,
        *,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
        max_content_length: int = 20_000
) -> ProcessedSearchResult:
    """Process a single search result by summarizing its raw content or falling back to its snippet.

    Args:
        search_result: The TavilySearchResult to process.
        summarization_model: Structured model runnable for summarizing raw webpage content.
        max_content_length: Maximum character length of raw content to pass to the summarizer.

    Returns:
        ProcessedSearchResult containing the summary and optional key excerpts.
    """

    summary = ""
    key_excerpts = None
    if search_result.raw_content:
        webpage_content = search_result.raw_content
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
            logger.error(f"Error summarizing content for URL {search_result.url}: {e}")
    else:
        summary = search_result.content

    return ProcessedSearchResult(
        id=search_result.id,
        title=search_result.title,
        url=search_result.url,
        summary=summary,
        key_excerpts=key_excerpts
    )


async def _summarize_webpage_content(
        webpage_content: str,
        *,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
) -> ContentSummary:
    """Summarize raw webpage content into a structured summary and key excerpts.

    Args:
        webpage_content: Truncated raw text content of the webpage.
        summarization_model: Structured model runnable returning ContentSummary.

    Returns:
        Structured ContentSummary containing the summary and key excerpts.
    """

    messages = [
        SystemMessage(
            content=SUMMARIZE_WEBPAGE_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        HumanMessage(
            content=SUMMARIZE_WEBPAGE_USER_PROMPT.format(webpage_content=webpage_content)
        )
    ]
    return await summarization_model.ainvoke(messages)
