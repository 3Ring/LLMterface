from __future__ import annotations

import typing as t
import json

import llmterface as llm
from llmterface.providers.provider_chat import ProviderChat
from llmterface.providers.provider_spec import ProviderSpec


class FakeConfig(llm.ProviderConfig):
    PROVIDER: t.ClassVar[str] = "mock"

    @classmethod
    def from_generic_config(cls, config: llm.GenericConfig | None) -> "FakeConfig":
        return cls()


class FakeChat(ProviderChat):
    PROVIDER: t.ClassVar[str] = FakeConfig.PROVIDER

    @classmethod
    def from_provider_config(
        cls,
        config: llm.ProviderConfig,
    ) -> "FakeChat":
        return cls()

    def ask(
        self, question: llm.Question, client_config: llm.ProviderConfig
    ) -> llm.GenericResponse:
        text = dict()
        res = None
        if "which provider are you" in question.prompt:
            res = question.config.provider
        if "What is the current weather in Paris?" in question.prompt:
            text = {
                "temperature_c": 12.0,
                "condition": "Sunny with a chance of croissants",
            }
        elif issubclass(
            question.config.response_model, llm.simple_answers.SimpleString
        ):
            text["response"] = res or "mock response"
        elif issubclass(question.config.response_model, llm.simple_answers.SimpleFloat):
            text["response"] = res or "3.14"
        elif issubclass(
            question.config.response_model, llm.simple_answers.SimpleInteger
        ):
            text["response"] = res or "42"
        else:
            raise NotImplementedError("Unsupported response format in FakeChat.")
        return llm.GenericResponse(original={}, text=json.dumps(text))


def mock_all_prov() -> None:
    from llmterface.providers.discovery import (
        _PROVIDER_SPECS,
        load_provider_configs_once,
    )

    load_provider_configs_once()
    for key in ["openai", "gemini", "anthropic"]:
        if key in _PROVIDER_SPECS:
            continue
        _PROVIDER_SPECS[key] = ProviderSpec(
            provider=key,
            config_cls=FakeConfig,
            chat_cls=FakeChat,
        )
    for k in _PROVIDER_SPECS:
        _PROVIDER_SPECS[k] = ProviderSpec(
            provider=k,
            config_cls=FakeConfig,
            chat_cls=FakeChat,
        )
