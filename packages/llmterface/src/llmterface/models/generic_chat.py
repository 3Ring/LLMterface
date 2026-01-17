import json
import typing as t

import llmterface.exceptions as ex
from llmterface.models.generic_config import GenericConfig
from llmterface.models.question import Question
from llmterface.providers.discovery import get_provider_chat, get_provider_config
from llmterface.providers.provider_chat import ProviderChat
from llmterface.providers.provider_config import ProviderConfig
from pydantic import BaseModel

TChatCls = t.TypeVar("TChatCls", bound=ProviderChat)

TAns = t.TypeVar("TAns", bound=BaseModel)


class GenericChat(t.Generic[TChatCls]):
    def __init__(
        self,
        id: str,
        client_chat: TChatCls | None = None,
        config: GenericConfig | ProviderConfig | None = None,
    ):
        self.id = id
        self.client = client_chat
        if isinstance(config, ProviderConfig):
            config = GenericConfig(provider=config.PROVIDER, provider_overrides={config.PROVIDER: config})
        self.config = config

    @staticmethod
    def get_provider_config(
        config: GenericConfig,
    ) -> ProviderConfig:
        if not config.provider:
            raise ValueError("Provider must be specified in the GenericConfig.")
        if override := config.provider_overrides.get(config.provider):
            return override

        provider_config_cls = get_provider_config(config.provider)
        if not provider_config_cls:
            raise NotImplementedError(f"No config factory found for provider: {config.provider}")

        return provider_config_cls.from_generic_config(config)

    def ask(self, question: Question[TAns]) -> TAns:
        """
        Ask a question using the chat's AI client and store the response.
        """
        try:
            question = question.with_prioritized_config([self.config])
            provider_config = question.config.provider_overrides.get(self.client.PROVIDER) or self.get_provider_config(
                question.config
            )
            return self._ask(question, provider_config)
        except Exception as e:
            raise ex.ClientError(f"Error while asking question to AI client: [{type(e)}]{e}") from e

    def _ask(self, question: Question[TAns], provider_config: ProviderConfig) -> TAns:
        retries = 0
        res = None
        while True:
            try:
                res = self.client.ask(question, provider_config)
                json_res = json.loads(res.text)
                return question.config.validate_response(json_res)
            except ex.AiHandlerError:
                raise
            except Exception as e:
                if isinstance(e, (json.JSONDecodeError, ValueError)):
                    exc = ex.SchemaError(f"Error parsing response: [{type(e)}]{e}", original_exception=e)
                else:
                    exc = ex.ProviderError(f"Error from provider: [{type(e)}]{e}", original_exception=e)
                exc.__cause__ = e
                retry_question = question.on_retry(question, response=res, e=exc, retries=retries)
                if not retry_question:
                    raise exc from e
                question = retry_question
                retries += 1
                continue

    def close(self) -> None:
        """
        Close the chat and perform any necessary cleanup.
        """
        self.client.close()

    @classmethod
    def create(
        cls,
        provider: str,
        chat_id: str,
        config: GenericConfig | None = None,
    ) -> "GenericChat":
        """
        Factory method to create a GenericChat with the specified provider.
        """
        ProviderChatCls = get_provider_chat(provider)
        if not ProviderChatCls:
            raise NotImplementedError(f"No provider chat class found for provider: {provider}")
        client_chat = ProviderChatCls(id=chat_id, config=config)
        return cls(client_chat.id, client_chat=client_chat, config=config)
