import pytest
import llmterface as llm

from testing.helpers.fakes import mock_all_prov, FakeProviderConfig


def test_instantiate_handler():
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
    )
    handler = llm.LLMterface(config=config)
    assert isinstance(handler, llm.LLMterface), "Handler should be an instance of LLMterface"


def test_ask_with_temp_chat():
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
        response_model=str,
    )
    handler = llm.LLMterface()
    question = llm.Question[str](question="What is the airspeed velocity of an unladen swallow?", config=config)
    with handler.temp_chat(provider=FakeProviderConfig.PROVIDER) as chat:
        response = chat.ask(question)
        assert isinstance(response, str), "Response should be a string"
        assert "African or European swallow" in response, "Response text should contain expected answer"


def test_create_chat():
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
        response_model=str,
    )
    handler = llm.LLMterface(config=config)
    chat_id = handler.create_chat(provider=FakeProviderConfig.PROVIDER, config=config)
    assert isinstance(chat_id, str), "Chat ID should be a string"
    chat = handler.chats[chat_id]
    assert isinstance(chat, llm.GenericChat), "Chat should be an instance of GenericChat"
    assert chat.id in handler.chats, "Chat ID should be in handler's chats"


def test_ask_with_chat_id():
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
        response_model=str,
    )
    handler = llm.LLMterface(config=config)
    chat_id = handler.create_chat(provider=FakeProviderConfig.PROVIDER, config=config)
    question = llm.Question(question="What is the airspeed velocity of an unladen swallow?")
    response = handler.ask(question, chat_id=chat_id)
    assert isinstance(response, str), "Response should be a string"
    assert "African or European swallow" in response, "Response text should contain expected answer"


@pytest.mark.parametrize("response_model", [str, int, float, bool])
def test_response_model_types(response_model):
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
        response_model=response_model,
    )
    handler = llm.LLMterface(config=config)
    res = handler.ask("return a value of the correct type")
    assert isinstance(res, response_model), f"Response should be of type {response_model}"
