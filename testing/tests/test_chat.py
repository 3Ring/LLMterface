import llmterface as llm
import pytest

from testing.helpers.fakes import FakeProviderConfig, mock_all_prov


def test_instantiate_chat():
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
    )
    chat = llm.GenericChat.create(
        provider=FakeProviderConfig.PROVIDER,
        chat_id="test_chat_id",
        config=config,
    )
    assert isinstance(chat, llm.GenericChat), "Chat should be an instance of GenericChat"


def test_ask_question():
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
        response_model=str,
    )
    chat = llm.GenericChat.create(
        provider=FakeProviderConfig.PROVIDER,
        chat_id="test_chat_id",
        config=config,
    )
    question = llm.Question[str](question="What is the capital of France?", config=config)
    response = chat.ask(question)
    assert isinstance(response, str), "Response should be a string"


@pytest.mark.parametrize("response_model", [str, int, float, bool])
def test_response_model_types(response_model):
    mock_all_prov()
    config = llm.GenericConfig(
        provider=FakeProviderConfig.PROVIDER,
        api_key="test_api_key",
        response_model=response_model,
    )
    handler = llm.LLMterface(config=config)
    chat = handler.create_chat(provider=FakeProviderConfig.PROVIDER, config=config)
    res = chat.ask(llm.Question(question="return a value of the correct type"))
    assert isinstance(res, response_model), f"Response should be of type {response_model}"
