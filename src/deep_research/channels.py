from functools import lru_cache
from typing import Any, Sequence, Self

from langgraph.channels import LastValue
from pydantic import TypeAdapter


@lru_cache(maxsize=None)
def _get_type_adapter(typ: Any) -> TypeAdapter[Any]:
    """Cache a Pydantic TypeAdapter per annotated type to avoid rebuilding per super-step."""
    return TypeAdapter(typ)


class ValidatedLastValue[T](LastValue[T]):
    """LastValue channel that validates and coerces values against its annotated type.
    Coerces JSON dicts arriving from HTTP/Studio inputs, state edits, or checkpoints
    into the declared Pydantic model before storing them in the channel.
    """
    __slots__ = ()

    def __eq__(self, value: object) -> bool:
        return isinstance(value, ValidatedLastValue) and self.typ == value.typ

    def _coerce(self, raw: Any) -> T:
        if isinstance(self.typ, type) and isinstance(raw, self.typ):
            return raw
        return _get_type_adapter(self.typ).validate_python(raw)

    def update(self, values: Sequence[T]) -> bool:
        if len(values) != 1:
            return super().update(values)
        coerced = self._coerce(values[0])
        return super().update((coerced,))

    def from_checkpoint(self, checkpoint: Any) -> Self:
        channel = super().from_checkpoint(checkpoint)
        if channel.is_available():
            channel.value = self._coerce(channel.value)
        return channel
