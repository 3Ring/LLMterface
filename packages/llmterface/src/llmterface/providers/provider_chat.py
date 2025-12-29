import typing as t
from abc import ABC, abstractmethod

from llmterface.models.question import Question
from llmterface.models.generic_response import GenericResponse
from llmterface.models.generic_config import GenericConfig
from llmterface.providers.provider_config import ProviderConfig


class ProviderChat(ABC):

    def __init__(
        self,
        id: str,
        config: t.Optional[GenericConfig] = None,
    ):
        self.id = id
        self.config = config

    @abstractmethod
    def ask(self, question: Question, client_config: ProviderConfig) -> GenericResponse:
        """
        Ask a question to the AI chat provider.
        """
        ...

    def close(self) -> None:
        """
        Optional standard method to close the chat and perform any necessary cleanup.
        """
        pass
