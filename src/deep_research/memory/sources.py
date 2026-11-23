import asyncio
import logging
from collections.abc import Iterable

from langgraph.store.base import BaseStore

from deep_research.schemas import ProcessedSearchResult

logger = logging.getLogger(__name__)


def sources_namespace(session_id: str) -> tuple[str, str, str]:
    """Return the session-scoped `BaseStore` namespace `("sessions", session_id, "sources")`."""
    return "sessions", session_id, "sources"


async def store_source(
    store: BaseStore,
    source: ProcessedSearchResult,
    *,
    session_id: str,
) -> None:
    """Persist a processed search result in the session store.

    Stores `source` as JSON under namespace `("sessions", session_id, "sources")`
    keyed by `source.id` with vector indexing disabled (`index=False`),
    overwriting any existing entry with the same ID.

    Args:
        store: Target LangGraph store.
        source: Processed search result to persist.
        session_id: Research session identifier scoping the namespace.
    """
    namespace = sources_namespace(session_id)
    await store.aput(
        namespace,
        source.id,
        source.model_dump(mode="json"),
        index=False,
    )


async def get_source_by_id(
    store: BaseStore, *, session_id: str, source_id: str
) -> ProcessedSearchResult | None:
    """Fetch and validate a stored source by ID within a session.

    Looks up `source_id` in namespace `("sessions", session_id, "sources")`.

    Args:
        store: LangGraph store to query.
        session_id: Research session identifier scoping the namespace.
        source_id: Source identifier (8-character hex digest).

    Returns:
        The validated `ProcessedSearchResult`, or `None` if `source_id` is missing.
    """
    namespace = sources_namespace(session_id)
    item = await store.aget(namespace, source_id)
    if item is None:
        return None

    return ProcessedSearchResult.model_validate(item.value)


async def get_sources_by_ids(
    store: BaseStore, *, session_id: str, source_ids: Iterable[str]
) -> dict[str, ProcessedSearchResult]:
    """Fetch multiple stored sources concurrently within a session.

    Deduplicates `source_ids` while preserving first-seen order and queries
    `("sessions", session_id, "sources")` in parallel. Missing IDs are omitted
    from the result without raising; returned entries preserve first-seen order.

    Args:
        store: LangGraph store to query.
        session_id: Research session identifier scoping the namespace.
        source_ids: Source identifiers to look up.

    Returns:
        Mapping from `source.id` to `ProcessedSearchResult` for all found IDs,
        or an empty dict if `source_ids` is empty or none are found.
    """
    unique_ids = list(dict.fromkeys(source_ids))
    if not unique_ids:
        return {}

    namespace = sources_namespace(session_id)
    items = await asyncio.gather(*(store.aget(namespace, source_id) for source_id in unique_ids))

    sources_by_ids = {}
    for item in items:
        if item is not None:
            source = ProcessedSearchResult.model_validate(item.value)
            sources_by_ids[source.id] = source

    return sources_by_ids
