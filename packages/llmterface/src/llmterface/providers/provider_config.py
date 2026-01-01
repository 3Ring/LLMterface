from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel

if t.TYPE_CHECKING:
    from llmterface.models.generic_config import GenericConfig


class ProviderConfig(BaseModel, ABC):
    """Base class for provider configs.

    PROVIDER:
        Provider identifier for this config subclass.
    """

    PROVIDER: t.ClassVar[str]

    @classmethod
    @abstractmethod
    def from_generic_config(
        cls,
        config: GenericConfig | None,
    ) -> "ProviderConfig": ...
