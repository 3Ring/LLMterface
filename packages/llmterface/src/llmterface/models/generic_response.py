from dataclasses import dataclass, field
import typing as t

TOrig = t.TypeVar("TOrig")


@dataclass(frozen=True, slots=True)
class GenericResponse(t.Generic[TOrig]):

    original: TOrig
    text: str
    metadata: t.Mapping[str, t.Any] = field(default_factory=dict)

    def get_hash(self) -> int:
        return hash(self.text)

    __hash__ = get_hash
