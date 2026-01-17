import llmterface as llm

from testing.helpers.fakes import mock_all_prov


def test_config():
    llm.GenericConfig()


def test_readme_basic():
    mock_all_prov()
    import llmterface as llm

    handler = llm.LLMterface(
        config=llm.GenericConfig(
            api_key="<YOUR GEMINI API KEY>",
            provider="gemini",
        )
    )
    res = handler.ask("how many LLMs does it take to screw in a lightbulb? Explain your reasoning.")
    assert res, "Response should not be empty."
    assert isinstance(res, str), "Response should be a string."


def test_readme_config_precedence():
    mock_all_prov()
    import llmterface as llm
    import llmterface_gemini as gemini
    from functools import partial

    gemini_config = partial(
        llm.GenericConfig,
        provider=gemini.GeminiConfig.PROVIDER,
        api_key="<YOUR GEMINI API KEY>",
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
    assert res is not None
    assert isinstance(res, str)


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
