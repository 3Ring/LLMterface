import typing as t
from copy import deepcopy

from pydantic import BaseModel


def compile_values(base: t.Any, override: t.Any, merge: bool = True) -> t.Any:
    if override is None:
        return deepcopy(base) if merge else None

    if base is None:
        return deepcopy(override)

    if isinstance(base, BaseModel):
        base = base.model_dump()

    if isinstance(override, BaseModel):
        override = override.model_dump(
            exclude_computed_fields=True,
            exclude_none=merge,
            exclude_unset=merge,
        )

    if not isinstance(base, t.Mapping) or not isinstance(override, t.Mapping):
        return deepcopy(override)

    result: dict[t.Any, t.Any] = dict(base)

    if merge:
        for k, ov in override.items():
            if k in result:
                result[k] = compile_values(result[k], ov, merge=True)
            else:
                result[k] = deepcopy(ov)
        return result

    result.update(deepcopy(dict(override)))
    return result
