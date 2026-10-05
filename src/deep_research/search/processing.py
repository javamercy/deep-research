from langchain_core.messages import HumanMessage

from deep_research.configuration import LLMModelConfig
from deep_research.models import init_openrouter_structured_model
from deep_research.prompts.research import SUMMARIZE_WEBPAGE_SYSTEM_PROMPT
from deep_research.schemas import ContentSummary
from deep_research.utils import get_today_str


async def process_search_results(
        unique_results: dict,
        llm_config: LLMModelConfig,
        max_retries: int = 3,
        max_content_length: int = 10000
) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results
        llm_config: Configuration for the language model used for summarization
        max_retries: Maximum number of retries for summarization
        max_content_length: Maximum length of content to summarize

    Returns:
        Dictionary of processed results with summaries
    """

    processed_results = {}

    for url, result in unique_results.items():
        if not result.get("raw_content"):
            content = result["content"]
        else:
            content = await _summarize_webpage_content(
                result["raw_content"],
                llm_config=llm_config,
                max_retries=max_retries,
                max_content_length=max_content_length
            )

        processed_results[url] = {
            "title": result["title"],
            "content": content
        }

    return processed_results


async def _summarize_webpage_content(
        webpage_content: str,
        *,
        llm_config: LLMModelConfig,
        max_retries: int,
        max_content_length: int
) -> str:
    try:
        messages = [
            HumanMessage(
                content=SUMMARIZE_WEBPAGE_SYSTEM_PROMPT.format(
                    webpage_content=webpage_content,
                    date=get_today_str(),
                )
            )
        ]

        summarization_model = init_openrouter_structured_model(
            llm_config,
            output_schema=ContentSummary,
            max_retries=max_retries
        )
        summary = await summarization_model.ainvoke(messages)

        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        return formatted_summary

    except Exception as e:
        print(f"Error summarizing webpage content: {e}")

        if len(webpage_content) > max_content_length:
            return webpage_content[:max_content_length] + "..."

        return webpage_content
