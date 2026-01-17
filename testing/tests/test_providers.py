import llmterface as llm
import pytest
from llmterface.providers.discovery import get_provider_config, load_provider_configs
from llmterface.providers.provider_spec import ProviderSpec

load_provider_configs()
from llmterface.providers.discovery import _PROVIDER_SPECS

pop_config = lambda: llm.GenericConfig(
    api_key="test_api_key",
    model=llm.GenericModelType.text_standard,
    temperature=0.7,
    max_output_tokens=512,
    max_input_tokens=1024,
    system_instruction="You are a helpful assistant.",
    response_model=str,
)


@pytest.mark.parametrize(
    "provider_spec",
    list(_PROVIDER_SPECS.values()),
)
def test_provider_spec(provider_spec: ProviderSpec):
    assert isinstance(provider_spec, ProviderSpec), "provider_spec must be a ProviderSpec instance"
    assert issubclass(provider_spec.config_cls, llm.ProviderConfig), "config_cls must be a ProviderConfig subclass"
    assert issubclass(provider_spec.chat_cls, llm.ProviderChat), "chat_cls must be a ProviderChat subclass"


@pytest.mark.parametrize(
    "provider_config_cls",
    list(spec.config_cls for spec in _PROVIDER_SPECS.values()),
)
def test_provider_config_registration(provider_config_cls: type[llm.ProviderConfig]):
    assert issubclass(provider_config_cls, llm.ProviderConfig), "expected_class must be a ProviderConfig subclass"
    assert isinstance(provider_config_cls.PROVIDER, str), "PROVIDER must be a string"
    config_cls = get_provider_config(provider_config_cls.PROVIDER)
    assert config_cls is provider_config_cls, (
        f"ProviderConfig.for_provider did not return the expected class for provider '{provider_config_cls.PROVIDER}'"
    )


@pytest.mark.parametrize(
    "provider_config_cls",
    list(spec.config_cls for spec in _PROVIDER_SPECS.values()),
)
def test_from_generic_config(provider_config_cls: type[llm.ProviderConfig]):
    generic_config = pop_config()
    provider_config = provider_config_cls.from_generic_config(generic_config)
    assert isinstance(provider_config, llm.ProviderConfig), (
        "from_generic_config did not return a ProviderConfig instance"
    )


@pytest.mark.parametrize(
    "provider_spec",
    list(_PROVIDER_SPECS.values()),
)
def test_provider_chat_instantiation(provider_spec: ProviderSpec):
    generic_config = pop_config()
    chat_id = "test_chat"
    ChatCls = provider_spec.chat_cls
    provider_chat_no_config = ChatCls(id=chat_id)
    assert isinstance(provider_chat_no_config, llm.ProviderChat), (
        "ProviderChat instantiation without config did not return a ProviderChat instance"
    )
    provider_chat_with_config = ChatCls(id=chat_id, config=generic_config)
    assert isinstance(provider_chat_with_config, llm.ProviderChat), (
        "ProviderChat instantiation with config did not return a ProviderChat instance"
    )
    for chat in [provider_chat_no_config, provider_chat_with_config]:
        assert hasattr(chat, "ask"), "ProviderChat instance does not have an 'ask' method"
        assert callable(getattr(chat, "ask", None)), "'ask' method of ProviderChat instance is not callable"
        assert hasattr(chat, "close"), "ProviderChat instance does not have a 'close' method"
        assert callable(getattr(chat, "close", None)), "'close' method of ProviderChat instance is not callable"
        assert hasattr(chat, "id"), "ProviderChat instance does not have an 'id' attribute"
        assert hasattr(chat, "config"), "ProviderChat instance does not have a 'config' attribute"
