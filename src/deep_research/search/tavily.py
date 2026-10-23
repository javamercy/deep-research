import os
from functools import lru_cache
from hashlib import sha256
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tavily import AsyncTavilyClient

from deep_research.schemas import TavilySearchResult

_TRACKING_PARAM_PREFIXES = ("utm_",)
_TRACKING_PARAMS = frozenset({"gclid", "fbclid", "msclkid", "igshid", "mc_cid", "mc_eid"})
_DEFAULT_PORTS = frozenset({80, 443})


@lru_cache(maxsize=1)
def _get_tavily_client() -> AsyncTavilyClient:
    """Lazily initialize and reuse a single AsyncTavilyClient instance."""
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    return AsyncTavilyClient(api_key=api_key)


async def search_tavily(
        query: str,
        *,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True
) -> list[TavilySearchResult]:
    """Perform search using Tavily API and return unique results.

    Args:
        query: Search query string
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Dictionary of unique search results keyed by URL
    """

    client = _get_tavily_client()
    response = await client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
        include_images=False,
    )

    return _deduplicate_and_map_search_results(response.get("results", []))


def _deduplicate_and_map_search_results(search_results: list[dict]) -> list[TavilySearchResult]:
    mapped_results: list[TavilySearchResult] = []

    seen_urls: set[str] = set()
    for result in search_results:
        url = result["url"]
        normalized_url = _normalize_url(url)
        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)

            url_digest = sha256(normalized_url.encode("utf-8")).hexdigest()
            mapped_results.append(TavilySearchResult(
                id=url_digest[:8],
                url=url,
                title=result.get("title", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                raw_content=result.get("raw_content", None)
            ))

    return mapped_results


def _normalize_url(url: str) -> str:
    """Return a canonical form of a URL, used only as an identity key.

    Two URLs that point to the same page should produce the same string.
    The result is not meant to be fetched or shown to the user.
    """

    parts = urlsplit(url.strip())
    host = (parts.hostname or "").removeprefix("www.")
    if parts.port is not None and parts.port not in _DEFAULT_PORTS:
        host = f"{host}:{parts.port}"

    path = parts.path.rstrip("/")

    query_params = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS and
        not key.lower().startswith(_TRACKING_PARAM_PREFIXES)
    )
    return urlunsplit(("", host, path, urlencode(query_params), ""))
