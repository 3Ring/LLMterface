import typing as t

from pydantic import PrivateAttr
from google.genai.types import GenerateContentResponse
from google.genai.client import Client as GenaiClient
from google.genai.chats import Chat as GenaiChat

from llmterface.models.question import Question
from llmterface.models.generic_response import GenericResponse
from llmterface.providers.provider_chat import ProviderChat
from llmterface_gemini.config import (
    GeminiConfig,
)


def convert_response_to_generic(
    response: GenerateContentResponse,
) -> GenericResponse[GenerateContentResponse]:
    return GenericResponse(
        original=response,
        text=response.text or "",
    )


class GeminiChat(ProviderChat):
    PROVIDER: t.ClassVar[str] = GeminiConfig.PROVIDER
    _client: t.Optional[GenaiClient] = PrivateAttr(default=None)
    _sdk_chat: t.Optional[GenaiChat] = PrivateAttr(default=None)

    def ask(self, question: Question, provider_config: t.Optional[GeminiConfig] = None) -> GenericResponse:
        provider_config = provider_config or self.config
        if provider_config is None:
            raise ValueError("GeminiConfig must be provided to ask a question.")
        if not self._sdk_chat:
            if self._client is None:
                self._client = GenaiClient(api_key=provider_config.api_key)
            self._sdk_chat = self._client.chats.create(model=provider_config.model.value)
        res = self._sdk_chat.send_message(question.prompt, config=provider_config.gen_content_config)
        return convert_response_to_generic(res)
