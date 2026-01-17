from __future__ import annotations

import json
import typing as t

import llmterface as llm
from llmterface.providers.provider_chat import ProviderChat
from llmterface.providers.provider_spec import ProviderSpec


class FakeProviderConfig(llm.ProviderConfig):
    PROVIDER: t.ClassVar[str] = "mock"

    @classmethod
    def from_generic_config(cls, config: llm.GenericConfig | None) -> FakeProviderConfig:
        return cls()


class FakeChat(ProviderChat):
    PROVIDER: t.ClassVar[str] = FakeProviderConfig.PROVIDER

    def ask(self, question: llm.Question, provider_config: llm.ProviderConfig) -> llm.GenericResponse:
        text = dict()
        res = None
        if "What is the airspeed velocity of an unladen swallow?" in question.prompt:
            if issubclass(question.config.response_model, str):
                res = "An African or European swallow?"
            elif issubclass(question.config.response_model, float):
                res = 42.0
            elif issubclass(question.config.response_model, bool):
                res = True
            elif issubclass(question.config.response_model, int):
                res = 42
            else:
                raise NotImplementedError("Unsupported response format in FakeChat.")
        if "What is the current weather in Paris?" in question.prompt:
            text = {
                "temperature_c": 12.0,
                "condition": "Sunny with a chance of croissants",
            }
        elif issubclass(question.config.response_model, str):
            text["response"] = res or "mock response"
        elif issubclass(question.config.response_model, float):
            text["response"] = res or 3.14
        elif issubclass(question.config.response_model, bool):
            if res is not None:
                text["response"] = res
            else:
                text["response"] = True
        elif issubclass(question.config.response_model, int):
            text["response"] = res or 42
        else:
            raise NotImplementedError("Unsupported response format in FakeChat.")
        print(f"FakeChat returning text: {text}")
        return llm.GenericResponse(original={}, text=json.dumps(text))


def mock_all_prov() -> None:
    from llmterface.providers.discovery import (
        _PROVIDER_SPECS,
        load_provider_configs_once,
    )

    load_provider_configs_once()
    for key in ["openai", "gemini", "anthropic", "mock"]:
        if key in _PROVIDER_SPECS:
            continue
        _PROVIDER_SPECS[key] = ProviderSpec(
            provider=key,
            config_cls=FakeProviderConfig,
            chat_cls=FakeChat,
        )
    for k in _PROVIDER_SPECS:
        _PROVIDER_SPECS[k] = ProviderSpec(
            provider=k,
            config_cls=FakeProviderConfig,
            chat_cls=FakeChat,
        )
