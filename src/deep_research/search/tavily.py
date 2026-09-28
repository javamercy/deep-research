from typing import Literal, cast

from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from tavily import AsyncTavilyClient

from deep_research.configuration import LLMModel
from deep_research.prompts.research import SUMMARIZE_WEBPAGE_SYSTEM_PROMPT
from deep_research.schemas import ContentSummary
from deep_research.utils import get_today_str

tavily_client = AsyncTavilyClient()

model = ChatOpenRouter(
    model=LLMModel.DEEPSEEK_V4_FLASH,
    temperature=0.1,
    reasoning={"effort": "medium"})

summarization_model = model.with_structured_output(
    ContentSummary,
    include_raw=False,
    method="json_schema",
    strict=True
)


async def tavily_search_multiple(
        search_queries: list[str],
        *,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True
) -> list[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """

    results = []
    for query in search_queries:
        result = await tavily_client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
            include_images=False,
        )
        results.append(result)

    return results


def deduplicate_search_results(search_results: list[dict]) -> dict:
    """Deduplicate search results based on their URLs.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary of unique search results keyed by URL
    """

    seen_urls = {}

    for result in search_results:
        url = result["url"]
        if url not in seen_urls:
            seen_urls[url] = result

    return seen_urls


async def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """

    processed_results = {}

    for url, result in unique_results.items():
        if not result.get("raw_content"):
            content = result["content"]
        else:
            content = await summarize_webpage_content(result["raw_content"])

        processed_results[url] = {
            "title": result["title"],
            "content": content
        }

    return processed_results


async def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary string containing the summary and key excerpts
    """

    try:

        messages = [
            HumanMessage(
                content=SUMMARIZE_WEBPAGE_SYSTEM_PROMPT.format(
                    webpage_content=webpage_content,
                    date=get_today_str(),
                )
            )
        ]
        summary = cast(
            ContentSummary,
            await summarization_model.ainvoke(messages)
        )

        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:
        print(f"Error summarizing webpage content: {e}")
        return webpage_content[:2000] + "..." if len(webpage_content) > 2000 else webpage_content


def format_search_results(processed_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        processed_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """

    if not processed_results:
        raise ValueError("No valid search results found. Please try different search queries.")

    formatted_output = "Search Results:"

    for i, (url, result) in enumerate(processed_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output
