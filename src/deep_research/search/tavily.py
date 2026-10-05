from typing import Literal

from tavily import AsyncTavilyClient

_client = AsyncTavilyClient()


async def tavily_search_multiple(
        search_queries: list[str],
        *,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True
) -> dict:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Dictionary of unique search results keyed by URL
    """

    results = []
    for query in search_queries:
        result = await _client.search(
            query=query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
            include_images=False,
        )
        results.append(result)

    return _deduplicate_search_results(results[0].get("results", []))


def _deduplicate_search_results(search_results: list[dict]) -> dict:
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
