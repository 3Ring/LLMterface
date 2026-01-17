import typing as t
import logging
from contextlib import contextmanager
import uuid
import json

from pydantic import BaseModel

import llmterface.exceptions as ex
from llmterface.models.question import Question
from llmterface.models.generic_chat import GenericChat
from llmterface.models.generic_config import GenericConfig

logger = logging.getLogger("llmterface")

TAns = t.TypeVar("TAns", bound=BaseModel)


class LLMterface:

    def __init__(
        self,
        config: t.Optional[GenericConfig] = None,
        chats: t.Optional[dict[str, GenericChat]] = None,
    ):
        if chats is None:
            chats = dict()
        self.chats = chats
        self.base_config = config

    @contextmanager
    def temp_chat(
        self, config: t.Optional[GenericConfig] = None, provider: t.Optional[str] = None
    ) -> t.Generator[GenericChat, None, None]:
        provider = (
            provider
            or (config.provider if config else None)
            or (self.base_config.provider if self.base_config else None)
        )
        if not provider:
            raise ValueError(
                "Provider must be specified either in config or as an argument for temporary chat."
            )
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

    @t.overload
    def ask(
        self,
        question: Question[TAns],
        chat_id: t.Optional[str] = None,
    ) -> TAns: ...
    @t.overload
    def ask(
        self,
        question: Question[None],
        chat_id: t.Optional[str] = None,
    ) -> str: ...
    @t.overload
    def ask(
        self,
        question: str,
        chat_id: t.Optional[str] = None,
    ) -> str: ...
    def ask(
        self,
        question: Question[TAns] | str,
        chat_id: t.Optional[str] = None,
    ) -> TAns | str:

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


    def create_chat(
        self,
        provider: str,
        config: t.Optional[GenericConfig] = None,
        chat_id: t.Optional[str] = None,
    ) -> str:
        chat = GenericChat.create(
            provider, chat_id=chat_id or uuid.uuid4().hex, config=config
        )
        self.chats[chat.id] = chat
        return chat.id
