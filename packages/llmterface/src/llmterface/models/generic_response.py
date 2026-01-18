import typing as t
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GenericResponse[T]:
    original: T
    text: str
    metadata: t.Mapping[str, t.Any] = field(default_factory=dict)
