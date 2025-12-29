from dataclasses import dataclass

from llmterface.providers.provider_config import ProviderConfig
from llmterface.providers.provider_chat import ProviderChat


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    config_cls: type[ProviderConfig]
    chat_cls: type[ProviderChat]
