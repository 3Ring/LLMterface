import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, ConfigDict

from llmterface.models.question import Question
from llmterface.models.generic_response import GenericResponse
from llmterface.models.generic_config import GenericConfig
from llmterface.providers.provider_config import ProviderConfig


class ProviderChat(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid")

    PROVIDER: t.ClassVar[str] = NotImplemented
    id: str = Field(..., description="Unique identifier for the chat instance.")
    config: t.Optional[GenericConfig] = Field(
        default=None, description="Configuration for the chat instance."
    )

    @abstractmethod
    def ask(self, question: Question, provider_config: ProviderConfig) -> GenericResponse:
        """
        Ask a question to the AI chat provider.
        """
        ...

    def close(self) -> None:
        """
        Optional standard method to close the chat and perform any necessary cleanup.
        """
        pass
