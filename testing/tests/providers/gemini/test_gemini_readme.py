import os

import dotenv
import llmterface as llm
import pytest
from llmterface.providers.discovery import load_provider_configs

dotenv.load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


@pytest.mark.integration()
def test_readme_basic():
    load_provider_configs()

    handler = llm.LLMterface(
        config=llm.GenericConfig(
            api_key=API_KEY,
            provider="gemini",
        )
    )
    res = handler.ask("how many LLMs does it take to screw in a lightbulb? Explain your reasoning.")
    assert res, "Response should not be empty."
    assert isinstance(res, str), "Response should be a string."


@pytest.mark.integration()
def test_readme_config_precedence():
    load_provider_configs()
    from functools import partial

    import llmterface_gemini as gemini

    gemini_config = partial(
        llm.GenericConfig,
        provider=gemini.GeminiConfig.PROVIDER,
        api_key=API_KEY,
    )
    handler_config = gemini_config(response_model=int)
    chat_config = gemini_config(response_model=float)
    handler = llm.LLMterface(config=handler_config)
    chat_id = handler.create_chat(chat_config.provider, config=chat_config)

    Q = "What is the airspeed velocity of an unladen swallow?"
    question = llm.Question(
        question=Q,
        config=gemini_config(),  # response_model defaults to str
    )
    int_res = handler.ask(Q)
    assert isinstance(int_res, int), "Expected int response from handler config"
    float_res = handler.ask(Q, chat_id=chat_id)
    assert isinstance(float_res, float), "Expected float response from chat config"
    str_res = handler.ask(question, chat_id=chat_id)
    assert isinstance(str_res, str), "Expected str response from question config"


@pytest.mark.integration()
def test_readme_vendor_override():
    load_provider_configs()
    import llmterface_gemini as gemini

    gemini_override = gemini.GeminiConfig.from_generic_config(
        llm.GenericConfig(
            api_key=API_KEY,
            provider=gemini.GeminiConfig.PROVIDER,
        )
    )

    config = llm.GenericConfig(
        provider=gemini.GeminiConfig.PROVIDER,
        provider_overrides={gemini.GeminiConfig.PROVIDER: gemini_override},
    )

    handler = llm.LLMterface(config=config)

    res = handler.ask("How many LLMs does it take to screw in a lightbulb?")
    assert res is not None
    assert isinstance(res, str)


@pytest.mark.integration()
def test_readme_strucutured_response():
    load_provider_configs()
    from pydantic import BaseModel, Field

    class WeatherResponse(BaseModel):
        temperature_c: float = Field(..., description="Temperature in Celsius")
        condition: str = Field(..., description="Weather described in a silly way")

    question = llm.Question(
        question="What is the current weather in Paris?",
        config=llm.GenericConfig(
            api_key=API_KEY,
            provider="gemini",
            response_model=WeatherResponse,
        ),
    )

    res = llm.LLMterface().ask(question)

    assert isinstance(res, WeatherResponse)
    assert res.temperature_c is not None
    assert res.condition is not None
    assert isinstance(res.temperature_c, float)
    assert isinstance(res.condition, str)
