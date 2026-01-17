import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel
from google.genai.types import GenerateContentConfig

import llmterface as llm
from llmterface_gemini.models import GeminiTextModelType
from llmterface_gemini.config import GeminiConfig


# -------------------------
# Helpers / strategies
# -------------------------

allowed_models = list(GeminiTextModelType)
allowed_model_values = [m.value for m in allowed_models]

st_allowed_models = st.sampled_from(allowed_models)
st_generic_models = st.sampled_from(list(llm.GenericModelType))
st_valid_model_strings = st.sampled_from(allowed_model_values)

# Make strings that are NOT any allowed enum value
st_invalid_model_strings = st.text(min_size=1).filter(lambda s: s not in set(allowed_model_values))


# -------------------------
# validate_model tests
# -------------------------


def test_validate_model_none_returns_none():
    assert GeminiConfig.validate_model(None) is None


@given(st_allowed_models)
def test_validate_model_allowed_model_passthrough(model):
    assert GeminiConfig.validate_model(model) is model


def test_validate_model_generic_model_maps_correctly_for_known_mappings():
    val_lite = GeminiConfig.validate_model(llm.GenericModelType.text_lite)
    assert val_lite == GeminiConfig.GENERIC_MODEL_MAPPING[llm.GenericModelType.text_lite]
    assert val_lite in allowed_models

    val_standard = GeminiConfig.validate_model(llm.GenericModelType.text_standard)
    assert val_standard == GeminiConfig.GENERIC_MODEL_MAPPING[llm.GenericModelType.text_standard]
    assert val_standard in allowed_models

    val_heavy = GeminiConfig.validate_model(llm.GenericModelType.text_heavy)
    assert val_heavy == GeminiConfig.GENERIC_MODEL_MAPPING[llm.GenericModelType.text_heavy]
    assert val_heavy in allowed_models


def test_validate_model_generic_model_missing_mapping_raises(monkeypatch):
    # Remove one mapping to simulate "not implemented"
    for enum in (
        llm.GenericModelType.text_lite,
        llm.GenericModelType.text_standard,
        llm.GenericModelType.text_heavy,
    ):
        monkeypatch.delitem(GeminiConfig.GENERIC_MODEL_MAPPING, enum, raising=True)

        with pytest.raises(NotImplementedError, match=r"No mapping for generic model type:"):
            GeminiConfig.validate_model(enum)


@given(st_valid_model_strings)
def test_validate_model_valid_string_parses_to_enum(s):
    model = GeminiConfig.validate_model(s)
    assert isinstance(model, GeminiTextModelType)
    assert model.value == s


@given(st_invalid_model_strings)
def test_validate_model_invalid_string_raises_value_error(s):
    with pytest.raises(ValueError, match=r"Invalid Gemini model type:"):
        GeminiConfig.validate_model(s)


# -------------------------
# from_generic_config tests
# -------------------------


class _DummyResponseModel(BaseModel):
    @classmethod
    def model_json_schema(cls):
        return {
            "title": "Dummy",
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }


def test_from_generic_config_builds_generate_content_config():
    cfg = llm.GenericConfig(
        api_key="abc123",
        temperature=0.2,
        max_output_tokens=123,
        system_instruction="be nice",
        response_model=_DummyResponseModel,
    )

    gem_cfg = GeminiConfig.from_generic_config(cfg)

    assert isinstance(gem_cfg, GeminiConfig)
    assert gem_cfg.api_key == cfg.api_key
    assert gem_cfg.model not in list(llm.GenericModelType)
    assert isinstance(gem_cfg.gen_content_config, GenerateContentConfig)

    gcc = gem_cfg.gen_content_config
    assert gcc.temperature == cfg.temperature
    assert gcc.max_output_tokens == cfg.max_output_tokens
    assert gcc.system_instruction == cfg.system_instruction
    assert gcc.response_mime_type == "application/json"
    assert gcc.response_json_schema == cfg.get_response_schema()


def test_gemini_config_model_validation_runs_on_init_for_generic_model():
    gem_cfg = GeminiConfig(
        api_key="abc123",
        model=llm.GenericModelType.text_lite,  # should validate/convert
    )
    assert gem_cfg.model == GeminiConfig.GENERIC_MODEL_MAPPING[llm.GenericModelType.text_lite]


def test_gemini_config_model_validation_runs_on_init_for_string():
    gem_cfg = GeminiConfig(
        api_key="abc123",
        model=GeminiTextModelType.CHAT_2_0_FLASH.value,
    )
    assert gem_cfg.model == GeminiTextModelType.CHAT_2_0_FLASH


def test_gemini_config_model_invalid_string_raises_on_init():
    with pytest.raises(ValueError, match=r"Invalid Gemini model type:"):
        GeminiConfig(api_key="abc123", model="definitely-not-a-real-model")
