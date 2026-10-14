import logging

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from deep_research.prompts.research import SUMMARIZE_WEBPAGE_SYSTEM_PROMPT, SUMMARIZE_WEBPAGE_USER_PROMPT
from deep_research.schemas import ContentSummary
from deep_research.utils import get_today_str

logger = logging.getLogger(__name__)


async def process_search_results(
        unique_results: dict,
        *,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
        max_content_length: int = 10000
) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results
        summarization_model: Runnable model for summarizing content
        max_content_length: Maximum length of content to summarize

    Returns:
        Dictionary of processed results with summaries
    """

    processed_results = {}

    for url, result in unique_results.items():
        if not result.get("raw_content"):
            content = result["content"]
        else:
            webpage_content = result["raw_content"]
            if len(webpage_content) > max_content_length:
                webpage_content = webpage_content[:max_content_length] + "..."

            try:
                content = await _summarize_webpage_content(
                    webpage_content,
                    summarization_model=summarization_model
                )
            except Exception as e:
                logger.error(f"Error summarizing content for URL {url}: {e}")
                content = webpage_content

        processed_results[url] = {
            "title": result["title"],
            "content": content
        }

    return processed_results


async def _summarize_webpage_content(
        webpage_content: str,
        *,
        summarization_model: Runnable[LanguageModelInput, ContentSummary],
) -> str:
    messages = [
        SystemMessage(
            content=SUMMARIZE_WEBPAGE_SYSTEM_PROMPT.format(date=get_today_str())
        ),
        HumanMessage(
            content=SUMMARIZE_WEBPAGE_USER_PROMPT.format(webpage_content=webpage_content)
        )
    ]
    summary = await summarization_model.ainvoke(messages)
    formatted_summary = (
        f"<summary>\n{summary.summary}\n</summary>\n\n"
        f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
    )
    
    return formatted_summary
