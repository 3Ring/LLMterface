import typing as t
import logging
from contextlib import contextmanager
import uuid
import json

from pydantic import BaseModel

import llmterface.exceptions as ex
from llmterface.models.question import Question
from llmterface.models.simple_answers import SimpleString
from llmterface.models.generic_chat import GenericChat
from llmterface.models.generic_config import GenericConfig

logger = logging.getLogger("ai_handler")

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
        self, config: GenericConfig | None
    ) -> t.Generator[GenericChat, None, None]:
        if not config and not self.base_config:
            raise RuntimeError("Either a config or base_config must be provided.")
        chat_id = f"temp-{uuid.uuid4().hex}"
        config = config or self.base_config
        chat = GenericChat.create(
            provider=config.provider,
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
    ) -> SimpleString: ...
    @t.overload
    def ask(
        self,
        question: str,
        chat_id: t.Optional[str] = None,
    ) -> SimpleString: ...
    def ask(
        self,
        question: Question[TAns] | str,
        chat_id: t.Optional[str] = None,
    ) -> TAns | SimpleString:

        if isinstance(question, str):
            question: Question[SimpleString] = Question(
                question=question,
            )
        if chat_id:
            chat = self.chats.get(chat_id)
            conf = chat.config
            if not chat:
                raise KeyError(f"Chat with id '{chat_id}' not found.")
            return self._ask(question, chat)
        with self.temp_chat(question.config) as temp:
            return self._ask(question, temp)

    def _ask(self, question: Question[TAns], chat: GenericChat) -> TAns:
        retries = 0
        res = None
        if not question.config:
            config = chat.config or self.base_config
            if not config:
                raise RuntimeError(
                    "No config found for question or chat, and no base_config set."
                )
            question = question.model_copy(update={"config": config})
        while True:
            try:
                res = chat.ask(question)
                json_res = json.loads(res.text)
                return question.config.response_model.model_validate(json_res)
            except ex.AiHandlerError:
                raise
            except Exception as e:
                if isinstance(e, (json.JSONDecodeError, ValueError)):
                    exc = ex.SchemaError(
                        f"Error parsing response: [{type(e)}]{e}", original_exception=e
                    )
                else:
                    exc = ex.ProviderError(
                        f"Error from provider: [{type(e)}]{e}", original_exception=e
                    )
                exc.__cause__ = e
                retry_question = question.on_retry(
                    question, response=res, e=exc, retries=retries
                )
                if not retry_question:
                    raise exc from e
                question = retry_question
                retries += 1
                continue

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
