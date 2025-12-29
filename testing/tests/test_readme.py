import llmterface as llm

from testing.helpers.fakes import mock_all_prov


def test_config():
    llm.GenericConfig()


def test_readme_basic():
    mock_all_prov()
    import llmterface as llm

    handler = llm.LLMterface(
        config=llm.GenericConfig(
            api_key="<YOUR API KEY>",
            model=llm.GenericModelType.text_lite,
            provider="gemini",
        )
    )
    question = llm.Question(
        question="how many LLMs does it take to screw in a lightbulb?"
        "\nExplain your reasoning.",
    )
    res = handler.ask(question)
    assert res.response is not None
    assert isinstance(res.response, str)


def test_readme_config_precedence():
    mock_all_prov()
    Q = "which provider are you, 'anthropic', 'openai', or 'gemini'?"

    handler = llm.LLMterface(
        config=llm.GenericConfig(provider="gemini", api_key="<YOUR GEMINI API KEY>")
    )
    chat_config = llm.GenericConfig(provider="openai", api_key="<YOUR OPENAI API KEY>")
    chat_id = handler.create_chat(chat_config.provider, config=chat_config)

    question = llm.Question(
        question=Q,
        config=llm.GenericConfig(
            provider="anthropic", api_key="<YOUR ANTHROPIC API KEY>"
        ),
    )

    assert handler.ask(Q).response == "gemini"
    assert handler.ask(Q, chat_id=chat_id).response == "openai"
    assert handler.ask(question, chat_id=chat_id).response == "anthropic"


def test_readme_vendor_override():
    mock_all_prov()
    import llmterface as llm
    import llmterface_gemini as gemini

    gemini_override = gemini.GeminiConfig(
        api_key="<YOUR GEMINI API KEY>",
        model=gemini.GeminiTextModelType.CHAT_2_0_FLASH_LITE,
    )

    config = llm.GenericConfig(
        provider=gemini.GeminiConfig.PROVIDER,
        provider_overrides={gemini.GeminiConfig.PROVIDER: gemini_override},
    )

    handler = llm.LLMterface(config=config)

    res = handler.ask("How many LLMs does it take to screw in a lightbulb?")
    assert res.response is not None
    assert isinstance(res.response, str)


def test_readme_strucutured_response():
    mock_all_prov()
    from pydantic import BaseModel, Field
    import llmterface as llm

    class WeatherResponse(BaseModel):
        temperature_c: float = Field(..., description="Temperature in Celsius")
        condition: str = Field(..., description="Weather described in a silly way")

    question = llm.Question(
        question="What is the current weather in Paris?",
        config=llm.GenericConfig(
            api_key="<YOUR API KEY>",
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
