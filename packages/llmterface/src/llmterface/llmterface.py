import logging
import typing as t
import uuid
from contextlib import contextmanager

from llmterface.models.generic_chat import GenericChat
from llmterface.models.generic_config import AllowedResponseTypes, GenericConfig
from llmterface.models.question import Question

logger = logging.getLogger("llmterface")


class LLMterface[TRes: AllowedResponseTypes]:
    def __init__(
        self,
        config: GenericConfig[TRes] | None = None,
        chats: dict[str, GenericChat] = None,
    ):
        if chats is None:
            chats = dict()
        self.chats = chats
        self.base_config = config

    @t.overload
    def ask(self, question: Question[None] | str, chat_id: None = None) -> TRes: ...
    @t.overload
    def ask(self, question: Question[None] | str, chat_id: str) -> AllowedResponseTypes: ...
    @t.overload
    def ask(self, question: str, chat_id: str) -> AllowedResponseTypes: ...
    @t.overload
    def ask[TReturn: AllowedResponseTypes](
        self, question: Question[TReturn], chat_id: str | None = None
    ) -> TReturn: ...
    def ask(
        self,
        question: Question | str,
        chat_id: str | None = None,
    ):
        if isinstance(question, str):
            question: Question[str] = Question(
                question=question,
            )
        if chat_id:
            chat = self.chats.get(chat_id)
            if not chat:
                raise KeyError(f"Chat with id '{chat_id}' not found.")
            question = question.with_prioritized_config([chat.config, self.base_config])
            return chat.ask(question)
        question = question.with_prioritized_config([self.base_config])
        with self.temp_chat(config=None, provider=question.config.provider) as temp:
            return temp.ask(question)

    @contextmanager
    def temp_chat(
        self, config: GenericConfig[TRes] | None = None, provider: str | None = None
    ) -> t.Generator[GenericChat[TRes]]:
        provider = (
            provider
            or (config.provider if config else None)
            or (self.base_config.provider if self.base_config else None)
        )
        if not provider:
            raise ValueError("Provider must be specified either in config or as an argument for temporary chat.")
        chat_id = f"temp-{uuid.uuid4().hex}"
        chat = GenericChat.create(
            provider=provider,
            chat_id=chat_id,
            config=config,
        )
        try:
            yield chat
        finally:
            chat.close()

    def close(self) -> None:
        """
        Close all chats and perform any necessary cleanup.
        """
        for chat in self.chats.values():
            chat.close()
        self.chats.clear()

    def create_chat[TChatRes: AllowedResponseTypes](
        self,
        provider: str,
        config: GenericConfig[TChatRes] | None = None,
        chat_id: str | None = None,
    ) -> GenericChat[TChatRes]:
        chat = GenericChat.create(provider, chat_id=chat_id or uuid.uuid4().hex, config=config)
        self.chats[chat.id] = chat
        return chat
