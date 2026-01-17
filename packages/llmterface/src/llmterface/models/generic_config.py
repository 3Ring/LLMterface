import typing as t

from pydantic import BaseModel, Field, field_validator, SerializeAsAny, ConfigDict

from llmterface.models.generic_model_types import GenericModelType
from llmterface.providers.provider_config import ProviderConfig
from llmterface.providers.discovery import get_provider_config
from llmterface.models.simple_answers import SIMPLE_MAP

TRes = t.TypeVar("TRes", bound=BaseModel | str | int | float | bool)


class GenericConfig(BaseModel, t.Generic[TRes]):
    """
    Generic configuration shared across all LLM providers.

    This model defines a common set of configuration fields that are mapped
    to provider-specific configurations internally. Fields that are not
    supported by a given provider are ignored.

    Provider-specific differences should be handled via `provider_overrides`
    rather than by branching on provider logic in application code.
    """

    model_config = ConfigDict(extra="forbid")
    provider: str | None = Field(
        default=None,
        description=(
            "The LLM provider to use. If not specified, the provider must be set via provider-specific overrides."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "API key used to authenticate with the provider. "
            "This value is typically supplied via provider-specific overrides. "
            "If set on the base config, it is used only when no provider-specific "
            "override is present."
        ),
    )
    provider_overrides: dict[str, SerializeAsAny[ProviderConfig]] = Field(
        default_factory=dict,
        description=(
            "Optional provider-specific configuration overrides. "
            "Overrides are applied on top of the base config when resolving "
            "settings for a specific provider."
        ),
    )
    model: GenericModelType = Field(
        default=GenericModelType.text_lite,
        description=(
            "Generic model tier to use. This value is mapped to a concrete "
            "provider model internally. "
            "In general: lite -> smaller/cheaper/faster models, "
            "standard -> default models, "
            "heavy -> larger/slower/more expensive models."
        ),
    )
    temperature: float = Field(
        default=0.2,
        description=(
            "Sampling temperature used by the model. "
            "Higher values increase randomness and creativity, "
            "lower values produce more deterministic output."
        ),
    )
    system_instruction: t.Optional[str] = Field(
        default=None,
        description=(
            "Optional system-level instruction used to guide model behavior. "
            "This is typically prepended or injected according to provider semantics."
        ),
    )
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens allowed in the input prompt.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens the model is allowed to generate.",
    )
    response_model: type[TRes] = Field(
        default=str,
        description=(
            "The expected response model type. "
            "This can be a Pydantic model or a simple type like str, int, or float. "
            "The model's output will be validated against this type."
        ),
    )

    @field_validator("provider_overrides", mode="before")
    @classmethod
    def validate_provider_overrides(cls, v: t.Any) -> dict[str, ProviderConfig]:
        if v is None:
            return {}

        if not isinstance(v, dict):
            raise ValueError("provider_overrides must be a dictionary")

        validated: dict[str, ProviderConfig] = dict()

        for key, value in v.items():
            cfg_cls = get_provider_config(key)
            if isinstance(value, ProviderConfig):
                if not isinstance(value, cfg_cls):
                    value = cfg_cls.model_validate(value.model_dump())
            elif isinstance(value, dict):
                value = cfg_cls.model_validate(value)
            else:
                raise ValueError(f"Invalid provider config for '{key}': {type(value)}")

            validated[key] = value

        return validated

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(
        cls,
        v: t.Any,
    ) -> GenericModelType:
        if isinstance(v, GenericModelType):
            return v
        try:
            return GenericModelType(v)
        except ValueError:
            raise ValueError(f"Invalid model enum value: {v}")

    def get_response_schema(self) -> dict[str, t.Any]:
        """
        Get the JSON schema for the expected response model.
        """
        if not isinstance(self.response_model, type):
            raise TypeError(f"response_model must be a type, got: {type(self.response_model)}")
        if issubclass(self.response_model, BaseModel):
            return self.response_model.model_json_schema()
        elif self.response_model in SIMPLE_MAP:
            return SIMPLE_MAP[self.response_model].model_json_schema()
        else:
            raise NotImplementedError(f"Response schema generation not implemented for type: {self.response_model}")

    def validate_response(
        self,
        response_data: t.Any,
    ) -> TRes:
        """
        Validate the response data against the expected response model.
        """
        if issubclass(self.response_model, BaseModel):
            return self.response_model.model_validate(response_data)
        if self.response_model not in SIMPLE_MAP:
            raise NotImplementedError(f"Response validation not implemented for type: {self.response_model}")
        return SIMPLE_MAP[self.response_model].model_validate(response_data).response

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(provider={self.provider}, model={self.model}, "
            f"temperature={self.temperature}, api_key={'***' if self.api_key else None}, "
            f"max_input_tokens={self.max_input_tokens}, "
            f"max_output_tokens={self.max_output_tokens}, "
            f"provider_overrides={list(k.value for k in self.provider_overrides.keys())})"
        )
