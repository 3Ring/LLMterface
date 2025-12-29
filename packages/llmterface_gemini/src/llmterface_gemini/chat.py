import typing as t

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

    def __init__(
        self,
        id: str,
        config: t.Optional[GeminiConfig] = None,
        client: t.Optional[GenaiClient] = None,
        sdk_chat: t.Optional[GenaiChat] = None,
    ):
        self.id = id
        self.config = config
        self.client = client
        self.sdk_chat = sdk_chat

    def ask(
        self, question: Question, client_config: t.Optional[GeminiConfig] = None
    ) -> GenericResponse:
        client_config = client_config or self.config
        if client_config is None:
            raise ValueError("GeminiConfig must be provided to ask a question.")
        if not self.sdk_chat:
            if self.client is None:
                self.client = GenaiClient(api_key=client_config.api_key)
            self.sdk_chat = self.client.chats.create(model=client_config.model.value)
        res = self.sdk_chat.send_message(
            question.prompt, config=client_config.gen_content_config
        )
        return convert_response_to_generic(res)
