from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

if t.TYPE_CHECKING:
    from llmterface.models.generic_config import GenericConfig
from llmterface.providers.discovery import get_provider_config


class ProviderConfig(BaseModel, ABC):
    """Base class for provider configs.

    PROVIDER:
        Provider identifier for this config subclass.
    """

    PROVIDER: t.ClassVar[str]

    @classmethod
    def for_provider(cls, provider: str) -> type["ProviderConfig"]:

        try:
            return get_provider_config(provider)
        except KeyError as e:
            if not isinstance(provider, str):
                raise TypeError(f"provider must be a str, got {type(provider)}") from e
            raise ValueError(
                f"No ProviderConfig registered for provider='{provider}'"
            ) from e

    @classmethod
    @abstractmethod
    def from_generic_config(
        cls,
        config: GenericConfig | None,
    ) -> "ProviderConfig": ...
