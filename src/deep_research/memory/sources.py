import asyncio
import logging
from collections.abc import Iterable, Sequence

from langgraph.store.base import BaseStore, Item

from deep_research.schemas import FrozenBaseModel

logger = logging.getLogger(__name__)


class StoredSearchResult(FrozenBaseModel):
    id: str
    url: str
    title: str
    raw_content: str | None = None
    summary: str
    key_excerpts: str | None = None


def sources_namespace(session_id: str) -> tuple[str, str, str]:
    return "sessions", session_id, "sources"


async def store_sources(
        store: BaseStore,
        session_id: str,
        sources: Sequence[StoredSearchResult]
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


async def get_sources_by_ids(
        store: BaseStore,
        session_id: str,
        source_ids: Iterable[str]
) -> dict[str, StoredSearchResult]:
    unique_ids = list(set(source_ids))
    if not unique_ids:
        return {}

    namespace = sources_namespace(session_id)
    items: list[Item | None] = await asyncio.gather(
        *(
            store.aget(namespace, source_id)
            for source_id in unique_ids
        )
    )

    sources_by_ids = {}
    for item in items:
        if item is not None:
            source = StoredSearchResult.model_validate(item.value)
            sources_by_ids[source.id] = source

    return sources_by_ids
