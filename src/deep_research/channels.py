"""Custom LangGraph state channels with Pydantic validation."""

from collections.abc import Sequence
from functools import cache
from typing import Any, Self

from langgraph.channels import LastValue
from pydantic import TypeAdapter


@cache
def _get_type_adapter(typ: Any) -> TypeAdapter[Any]:
    """Return a cached Pydantic `TypeAdapter` for `typ`.

    Caches adapters without bound so channels do not rebuild them on every
    super-step or checkpoint restore.
    """
    return TypeAdapter(typ)


class ValidatedLastValue[T](LastValue[T]):
    """`LastValue` channel that validates and coerces values via Pydantic.

    Coerces raw inputs (such as JSON dicts from HTTP/Studio inputs, state edits,
    or serialized checkpoints) into the channel's annotated type before storing
    them.
    """

    __slots__ = ()

    def __eq__(self, value: object) -> bool:
        """Return whether `value` is a `ValidatedLastValue` for the same type."""
        return isinstance(value, ValidatedLastValue) and self.typ == value.typ

    def _coerce(self, raw: Any) -> T:
        """Validate and coerce `raw` into `self.typ` unless already an instance."""
        if isinstance(self.typ, type) and isinstance(raw, self.typ):
            return raw
        return _get_type_adapter(self.typ).validate_python(raw)

    def update(self, values: Sequence[T]) -> bool:
        """Coerce a single-value update before delegating to `LastValue.update`.

        Non-single updates (empty sequences or concurrent multi-value writes) are
        forwarded unchanged so `LastValue.update` preserves its no-op and
        `InvalidUpdateError` behavior.
        """
        if len(values) != 1:
            return super().update(values)
        coerced = self._coerce(values[0])
        return super().update((coerced,))

    def from_checkpoint(self, checkpoint: Any) -> Self:
        """Restore the channel from a checkpoint and coerce any stored value."""
        channel = super().from_checkpoint(checkpoint)
        if channel.is_available():
            channel.value = self._coerce(channel.value)
        return channel
