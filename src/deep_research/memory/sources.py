import asyncio
import logging
from collections.abc import Iterable, Sequence

from langgraph.store.base import BaseStore

from deep_research.schemas import ProcessedSearchResult

logger = logging.getLogger(__name__)


def sources_namespace(session_id: str) -> tuple[str, str, str]:
    return "sessions", session_id, "sources"


async def store_sources(
        store: BaseStore,
        *,
        session_id: str,
        sources: Sequence[ProcessedSearchResult]
) -> None:
    if not sources:
        return

    namespace = sources_namespace(session_id)
    await asyncio.gather(
        *(
            store.aput(
                namespace,
                source.id,
                source.model_dump(mode="json"),
                index=False,
            )
            for source in sources
        )
    )


async def get_source_by_id(
        store: BaseStore,
        *,
        session_id: str,
        source_id: str
) -> ProcessedSearchResult | None:
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
