import typing as t
from dataclasses import dataclass, field

TOrig = t.TypeVar("TOrig")


@dataclass(frozen=True, slots=True)
class GenericResponse(t.Generic[TOrig]):
    original: TOrig
    text: str
    metadata: t.Mapping[str, t.Any] = field(default_factory=dict)
