import typing as t
import pytest


@pytest.fixture(autouse=True)
def clear_provider_registry() -> None:
    """
    Clear the provider registry before each test to ensure a clean state.
    """
    from llmterface.providers.discovery import _PROVIDER_SPECS, load_provider_configs

    _PROVIDER_SPECS.clear()
    load_provider_configs()


# class MockChat(ProviderChat):
#     """
#     Minimal fake chat used for testing AiHandler and GenericChat behavior
#     without touching real providers or SDKs.
#     """

#     def ask(self, question: Question) -> GenericResponse:  # type: ignore
#         return GenericResponse(
#             original=None,
#             text="Mock response",
#         )


# @pytest.fixture
# def basic_config() -> GenericConfig:
#     """
#     Minimal valid GenericConfig for tests.
#     """
#     return GenericConfig(
#         provider=,
#         api_key="test-key",
#     )


# @pytest.fixture
# def simple_question() -> Question[SimpleString]:
#     """
#     A basic Question expecting a simple string response.
#     """
#     return Question(
#         question="Hello world",
#         response_format=SimpleString,
#         max_retries=0,
#     )


# @pytest.fixture
# def fake_chat_success() -> MockChat:
#     """
#     Fake chat that returns valid JSON once.
#     """
#     return MockChat(
#         responses=['{"response": "ok"}'],
#     )


# @pytest.fixture
# def fake_chat_schema_fail_then_success() -> MockChat:
#     """
#     Fake chat that first returns invalid JSON, then valid JSON.
#     Useful for retry tests.
#     """
#     return MockChat(
#         responses=[
#             "not json",
#             '{"response": "recovered"}',
#         ]
#     )


# @pytest.fixture
# def fake_chat_always_invalid() -> MockChat:
#     """
#     Fake chat that always returns invalid JSON.
#     """
#     return MockChat(
#         responses=["not json", "still not json", "nope"],
#     )
