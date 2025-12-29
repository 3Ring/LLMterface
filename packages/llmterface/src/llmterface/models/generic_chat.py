import typing as t

import llmterface.exceptions as ex
from llmterface.models.question import Question
from llmterface.models.generic_config import GenericConfig
from llmterface.models.generic_response import GenericResponse
from llmterface.providers.provider_chat import ProviderChat
from llmterface.providers.provider_config import ProviderConfig
from llmterface.providers.discovery import get_provider_config, get_provider_chat

TChatCls = t.TypeVar("TChatCls", bound=ProviderChat)


class GenericChat(t.Generic[TChatCls]):

    def __init__(
        self,
        id: str,
        client_chat: t.Optional[TChatCls] = None,
        config: t.Optional[GenericConfig | ProviderConfig] = None,
    ):
        self.id = id
        self.client = client_chat
        self.config = config

    @staticmethod
    def get_provider_client_config(
        config: GenericConfig,
    ) -> type[ProviderConfig]:
        if not config.provider:
            raise ValueError("Provider must be specified in the GenericConfig.")
        if override := config.provider_overrides.get(config.provider):
            return override

        provider_config_cls = get_provider_config(config.provider)
        if not provider_config_cls:
            raise NotImplementedError(
                f"No config factory found for provider: {config.provider}"
            )

        return provider_config_cls.from_generic_config(config)

    def ask(self, question: Question) -> GenericResponse:
        """
        Ask a question using the chat's AI client and store the response.
        """
        provider_config = question.config or self.client.config or self.config
        if isinstance(provider_config, GenericConfig) and (
            override := provider_config.provider_overrides.get(provider_config.provider)
        ):
            provider_config = override
        try:
            if not provider_config:
                raise RuntimeError(
                    "No configuration available for asking the question."
                )
            if not isinstance(provider_config, ProviderConfig):
                ProviderConfigCls = self.get_provider_client_config(provider_config)
                provider_config = ProviderConfigCls.from_generic_config(provider_config)
            return self.client.ask(question, provider_config)
        except Exception as e:
            raise ex.ClientError(
                f"Error while asking question to AI client: [{type(e)}]{e}"
            ) from e

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
        config: t.Optional[GenericConfig] = None,
    ) -> "GenericChat":
        """
        Factory method to create a GenericChat with the specified provider.
        """
        ProviderChatCls = get_provider_chat(provider)
        if not ProviderChatCls:
            raise NotImplementedError(
                f"No provider chat class found for provider: {provider}"
            )
        client_chat = ProviderChatCls(chat_id, config)
        return cls(client_chat.id, client_chat=client_chat, config=config)
