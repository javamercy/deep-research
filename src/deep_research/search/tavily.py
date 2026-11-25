import asyncio
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
    """Return a process-wide cached `AsyncTavilyClient` instance.

    Reads `TAVILY_API_KEY` from the environment on first call and caches the
    client via `lru_cache(maxsize=1)`.

    Raises:
        ValueError: If the `TAVILY_API_KEY` environment variable is unset or empty.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    return AsyncTavilyClient(api_key=api_key)


async def search_tavily_multiple(
    queries: list[str],
    *,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    search_depth: Literal["ultra-fast", "fast", "basic", "advanced"] = "basic",
) -> list[TavilySearchResult]:
    """Query the Tavily Search API concurrently across multiple queries.

    Executes `client.search` in parallel for each query with
    `include_images=False`, silently skipping queries that raise an exception
    or return `results=None`. Merges results in query order, deduplicates by
    normalized URL (keeping the first occurrence), and assigns each result a
    deterministic 8-character hex ID.

    Args:
        queries: Search query strings to run concurrently.
        max_results: Maximum number of results requested per query from Tavily.
        topic: Tavily search topic category.
        include_raw_content: Whether to request raw webpage content in results.
        search_depth: Tavily search depth preset.

    Returns:
        Deduplicated `TavilySearchResult` items merged across all queries.

    Raises:
        ValueError: If the `TAVILY_API_KEY` environment variable is unset or empty.
    """
    client = _get_tavily_client()
    responses: list[dict | BaseException] = await asyncio.gather(
        *[
            client.search(
                query=query,
                max_results=max_results,
                topic=topic,
                include_raw_content=include_raw_content,
                search_depth=search_depth,
                include_images=False,
            )
            for query in queries
        ],
        return_exceptions=True,
    )

    search_results: list[dict] = []
    for response in responses:
        if isinstance(response, BaseException):
            # Log the exception or handle it as needed
            continue

        results: list[dict] | None = response.get("results")
        if results is not None:
            search_results.extend(results)

    return _deduplicate_and_map_search_results(search_results)


async def search_tavily(
    query: str,
    *,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    search_depth: Literal["ultra-fast", "fast", "basic", "advanced"] = "basic",
) -> list[TavilySearchResult]:
    """Query the Tavily Search API and return URL-deduplicated results.

    Executes an external Tavily search with `include_images=False`, deduplicates
    results by normalized URL (keeping the first, highest-ranked occurrence),
    and assigns each result a deterministic 8-character hex ID.

    Args:
        query: Search query string.
        max_results: Maximum number of results requested from Tavily.
        topic: Tavily search topic category.
        include_raw_content: Whether to request raw webpage content in results.
        search_depth: Tavily search depth preset.

    Returns:
        Deduplicated `TavilySearchResult` items in Tavily's original rank order.

    Raises:
        ValueError: If the `TAVILY_API_KEY` environment variable is unset or empty.
    """
    client = _get_tavily_client()
    response = await client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
        search_depth=search_depth,
        include_images=False,
    )

    return _deduplicate_and_map_search_results(response.get("results", []))


def _deduplicate_and_map_search_results(search_results: list[dict]) -> list[TavilySearchResult]:
    """Deduplicate raw Tavily result dicts by normalized URL and map to models.

    Keeps the first occurrence of each normalized URL (preserving Tavily's
    relevance rank) and sets `id` to the first 8 hex characters of the
    SHA-256 digest of the normalized URL while preserving the original `url`.

    Args:
        search_results: Raw result dictionaries from a Tavily API response.

    Returns:
        Deduplicated `TavilySearchResult` models in original relative order.
    """
    mapped_results: list[TavilySearchResult] = []

    seen_urls: set[str] = set()
    for result in search_results:
        url = result["url"]
        normalized_url = _normalize_url(url)
        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)

            url_digest = sha256(normalized_url.encode("utf-8")).hexdigest()
            mapped_results.append(
                TavilySearchResult(
                    id=url_digest[:8],
                    url=url,
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    raw_content=result.get("raw_content", None),
                )
            )

    return mapped_results


def _normalize_url(url: str) -> str:
    """Canonicalize a URL into a deduplication key.

    Produces a scheme-less, fragment-less key suitable only for URL identity
    matching and hashing, not for fetching or display. Strips leading `www.`
    from the lowercased host, drops default ports (80, 443), strips trailing
    slashes from the path, removes common tracking query parameters (`utm_*`,
    `gclid`, `fbclid`, `msclkid`, `igshid`, `mc_cid`, `mc_eid`), and sorts
    remaining query parameters.

    Args:
        url: Raw URL string to canonicalize.

    Returns:
        Canonical identity string for hashing and deduplication.
    """
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").removeprefix("www.")
    if parts.port is not None and parts.port not in _DEFAULT_PORTS:
        host = f"{host}:{parts.port}"

    path = parts.path.rstrip("/")

    query_params = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
        and not key.lower().startswith(_TRACKING_PARAM_PREFIXES)
    )
    return urlunsplit(("", host, path, urlencode(query_params), ""))
