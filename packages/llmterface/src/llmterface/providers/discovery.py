from __future__ import annotations

import typing as t
from importlib.metadata import entry_points

if t.TYPE_CHECKING:
    from llmterface.providers.provider_chat import ProviderChat
    from llmterface.providers.provider_config import ProviderConfig
    from llmterface.providers.provider_spec import ProviderSpec

ENTRYPOINT_GROUP = "llmterface.providers"

_loaded = False
_PROVIDER_SPECS: dict[str, ProviderSpec] = dict()


def load_provider_configs() -> None:
    from llmterface.providers.provider_spec import ProviderSpec

    eps = entry_points(group=ENTRYPOINT_GROUP)
    for ep in eps:
        obj = ep.load()
        if not isinstance(obj, ProviderSpec):
            raise ValueError(f"Entry point {ep.name} did not return a ProviderSpec instance")
        _PROVIDER_SPECS[obj.provider] = obj


def load_provider_configs_once() -> None:
    global _loaded
    if _loaded:
        return
    load_provider_configs()
    _loaded = True


def get_provider_config(provider: str) -> type[ProviderConfig]:
    load_provider_configs_once()
    if provider not in _PROVIDER_SPECS:
        if not isinstance(provider, str):
            raise TypeError(f"provider must be a str, got {type(provider)}")
        raise NotImplementedError(f"No provider spec found for provider: '{provider}'. Did you install it correctly?")
    return _PROVIDER_SPECS[provider].config_cls


def get_provider_chat(provider: str) -> type[ProviderChat]:
    load_provider_configs_once()
    if provider not in _PROVIDER_SPECS:
        if not isinstance(provider, str):
            raise TypeError(f"provider must be a str, got {type(provider)}")
        raise NotImplementedError(f"No provider spec found for provider: '{provider}'. Did you install it correctly?")
    return _PROVIDER_SPECS[provider].chat_cls
