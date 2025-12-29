from llmterface.providers.provider_spec import ProviderSpec
from llmterface_gemini.config import GeminiConfig
from llmterface_gemini.chat import GeminiChat

PROVIDER = ProviderSpec(
    provider=GeminiConfig.PROVIDER,
    config_cls=GeminiConfig,
    chat_cls=GeminiChat,
)
