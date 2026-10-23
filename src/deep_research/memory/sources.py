import asyncio
import logging
from collections.abc import Iterable

from langgraph.store.base import BaseStore

from deep_research.schemas import ProcessedSearchResult

logger = logging.getLogger(__name__)


def sources_namespace(session_id: str) -> tuple[str, str, str]:
    """Build the BaseStore namespace tuple for session-scoped sources.

    Args:
        session_id: Unique identifier for the current research session.

    Returns:
        Namespace tuple ("sessions", session_id, "sources").
    """

    return "sessions", session_id, "sources"


async def store_source(
        store: BaseStore,
        source: ProcessedSearchResult,
        *,
        session_id: str,
) -> None:
    """Persist a single processed search result into the session store keyed by its ID.

    Args:
        store: The LangGraph BaseStore instance.
        source: The ProcessedSearchResult to persist.
        session_id: Unique identifier for the current research session.
    """

    namespace = sources_namespace(session_id)
    await store.aput(
        namespace,
        source.id,
        source.model_dump(mode="json"),
        index=False,
    )


async def get_source_by_id(
        store: BaseStore,
        *,
        session_id: str,
        source_id: str
) -> ProcessedSearchResult | None:
    """Retrieve a single stored source by its ID from the session store.

    Args:
        store: The LangGraph BaseStore instance.
        session_id: Unique identifier for the current research session.
        source_id: The 8-character hex ID of the source to retrieve.

    Returns:
        The validated ProcessedSearchResult if found, or None if not present.
    """

    namespace = sources_namespace(session_id)
    item = await store.aget(namespace, source_id)
    if item is None:
        return None

    return ProcessedSearchResult.model_validate(item.value)


async def get_sources_by_ids(
        store: BaseStore,
        *,
        session_id: str,
        source_ids: Iterable[str]
) -> dict[str, ProcessedSearchResult]:
    """Retrieve multiple stored sources by their IDs in parallel, preserving order.

    Args:
        store: The LangGraph BaseStore instance.
        session_id: Unique identifier for the current research session.
        source_ids: Iterable of 8-character hex source IDs to look up.

    Returns:
        Dictionary mapping each found source ID to its ProcessedSearchResult.
    """
    unique_ids = list(dict.fromkeys(source_ids))
    if not unique_ids:
        return {}

    namespace = sources_namespace(session_id)
    items = await asyncio.gather(
        *(
            store.aget(namespace, source_id)
            for source_id in unique_ids
        )
    )

    sources_by_ids = {}
    for item in items:
        if item is not None:
            source = ProcessedSearchResult.model_validate(item.value)
            sources_by_ids[source.id] = source

    return sources_by_ids
