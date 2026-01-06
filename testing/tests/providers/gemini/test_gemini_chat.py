import pytest

import llmterface as llm
import llmterface.exceptions as ex
import llmterface_gemini as gemini


class DummyClientChat:
    def __init__(self, chat_id: str, config=None):
        self.id = chat_id
        self.config = config
        self.ask_calls = []
        self.closed = False

    def ask(self, question, provider_config):
        self.ask_calls.append((question, provider_config))
        return "dummy-response"

    def close(self):
        self.closed = True


class DummyQuestion:
    def __init__(self, config=None):
        self.config = config


# -------------------------
# get_provider_client_config tests
# -------------------------


def test_get_provider_client_config_requires_provider():
    cfg = llm.GenericConfig(provider=None)

    with pytest.raises(
        ValueError, match="Provider must be specified in the GenericConfig"
    ):
        llm.GenericChat.get_provider_client_config(cfg)


def test_get_provider_client_config_uses_override_when_present():
    override = gemini.GeminiConfig(api_key="override-key")
    cfg = llm.GenericConfig(provider="gemini", provider_overrides={"gemini": override})

    got = llm.GenericChat.get_provider_client_config(cfg)
    assert got is override


def test_get_provider_client_config_no_factory_raises(monkeypatch):
    import llmterface.models.generic_chat as generic_chat_mod

    monkeypatch.setattr(generic_chat_mod, "get_provider_config", lambda provider: None)

    cfg = llm.GenericConfig(provider="nope")

    with pytest.raises(
        NotImplementedError, match="No config factory found for provider: nope"
    ):
        llm.GenericChat.get_provider_client_config(cfg)


def test_get_provider_client_config_uses_factory_from_generic_config(monkeypatch):
    import llmterface.models.generic_chat as generic_chat_mod

    created = gemini.GeminiConfig(api_key="override-key")

    class FactoryCls:
        @classmethod
        def from_generic_config(cls, config):
            return created

    monkeypatch.setattr(
        generic_chat_mod, "get_provider_config", lambda provider: FactoryCls
    )

    cfg = llm.GenericConfig(provider="gemini")
    got = llm.GenericChat.get_provider_client_config(cfg)
    assert got is created


# -------------------------
# ask tests
# -------------------------


def test_ask_prefers_question_config_over_client_and_self_config():
    client_cfg = gemini.GeminiConfig(api_key="override-key")
    self_cfg = gemini.GeminiConfig(api_key="override-key")
    q_cfg = gemini.GeminiConfig(api_key="override-key")

    client = DummyClientChat("chat1", config=client_cfg)
    chat = llm.GenericChat("chat1", client_chat=client, config=self_cfg)
    q = DummyQuestion(config=q_cfg)

    res = chat.ask(q)

    assert res == "dummy-response"
    assert len(client.ask_calls) == 1
    _, used_cfg = client.ask_calls[0]
    assert used_cfg is q_cfg


def test_ask_uses_client_config_when_question_config_missing():
    client_cfg = gemini.GeminiConfig(api_key="override-key")
    self_cfg = gemini.GeminiConfig(api_key="override-key")

    client = DummyClientChat("chat1", config=client_cfg)
    chat = llm.GenericChat("chat1", client_chat=client, config=self_cfg)
    q = DummyQuestion(config=None)

    res = chat.ask(q)

    assert res == "dummy-response"
    _, used_cfg = client.ask_calls[0]
    assert used_cfg is client_cfg


def test_ask_uses_self_config_when_question_and_client_config_missing():
    self_cfg = gemini.GeminiConfig(api_key="override-key")

    client = DummyClientChat("chat1", config=None)
    chat = llm.GenericChat("chat1", client_chat=client, config=self_cfg)
    q = DummyQuestion(config=None)

    res = chat.ask(q)

    assert res == "dummy-response"
    _, used_cfg = client.ask_calls[0]
    assert used_cfg is self_cfg


def test_ask_generic_config_with_override_uses_override():
    override_cfg = gemini.GeminiConfig(api_key="override-key")
    generic_cfg = llm.GenericConfig(
        provider="gemini", provider_overrides={"gemini": override_cfg}
    )

    client = DummyClientChat("chat1", config=None)
    chat = llm.GenericChat("chat1", client_chat=client, config=None)

    q = DummyQuestion(config=generic_cfg)
    res = chat.ask(q)

    assert res == "dummy-response"
    _, used_cfg = client.ask_calls[0]
    assert used_cfg is override_cfg


def test_ask_generic_config_without_override_calls_get_provider_client_config(
    monkeypatch,
):
    converted = gemini.GeminiConfig(api_key="override-key")
    generic_cfg = llm.GenericConfig(provider="gemini")

    called = {"n": 0}

    def fake_get_provider_client_config(cfg):
        called["n"] += 1
        assert cfg is generic_cfg
        return converted

    monkeypatch.setattr(
        llm.GenericChat,
        "get_provider_client_config",
        staticmethod(fake_get_provider_client_config),
    )

    client = DummyClientChat("chat1", config=None)
    chat = llm.GenericChat("chat1", client_chat=client, config=None)

    q = DummyQuestion(config=generic_cfg)
    res = chat.ask(q)

    assert res == "dummy-response"
    assert called["n"] == 1
    _, used_cfg = client.ask_calls[0]
    assert used_cfg is converted


def test_ask_no_config_raises_client_error():
    client = DummyClientChat("chat1", config=None)
    chat = llm.GenericChat("chat1", client_chat=client, config=None)
    q = DummyQuestion(config=None)

    with pytest.raises(
        ex.ClientError, match="No configuration available for asking the question"
    ):
        chat.ask(q)


def test_ask_wraps_client_exception_in_client_error():
    class ExplodingClientChat(DummyClientChat):
        def ask(self, question, provider_config):
            raise RuntimeError("boom")

    client = ExplodingClientChat(
        "chat1", config=gemini.GeminiConfig(api_key="override-key")
    )
    chat = llm.GenericChat("chat1", client_chat=client, config=None)
    q = DummyQuestion(config=None)

    with pytest.raises(
        ex.ClientError, match=r"Error while asking question to AI client:"
    ):
        chat.ask(q)


# -------------------------
# close tests
# -------------------------


def test_close_calls_client_close():
    client = DummyClientChat("chat1", config=None)
    chat = llm.GenericChat("chat1", client_chat=client, config=None)

    chat.close()
    assert client.closed is True


# -------------------------
# create tests
# -------------------------


def test_create_no_provider_chat_class_raises(monkeypatch):
    import llmterface.models.generic_chat as generic_chat_mod

    monkeypatch.setattr(generic_chat_mod, "get_provider_chat", lambda provider: None)

    with pytest.raises(
        NotImplementedError, match="No provider chat class found for provider: gemini"
    ):
        llm.GenericChat.create(provider="gemini", chat_id="c1", config=None)


def test_create_builds_client_and_returns_generic_chat(monkeypatch):
    import llmterface.models.generic_chat as generic_chat_mod

    class ProviderChatCls(DummyClientChat):
        pass

    monkeypatch.setattr(
        generic_chat_mod, "get_provider_chat", lambda provider: ProviderChatCls
    )

    cfg = llm.GenericConfig(provider="gemini")
    chat = llm.GenericChat.create(provider="gemini", chat_id="c1", config=cfg)

    assert isinstance(chat, llm.GenericChat)
    assert chat.id == "c1"
    assert chat.client.id == "c1"
    assert chat.config is cfg
