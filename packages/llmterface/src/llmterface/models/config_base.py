import typing as t

from llmterface.models.generic_model_types import GenericModelType
from pydantic import BaseModel, Field


class ConfigBase(BaseModel):
    api_key: str | None = Field(default=None)
    model: GenericModelType | str | None = Field(default=None)
    temperature: float | None = Field(default=None)
    sys_instruction: str | None = Field(default=None)
    max_input_tokens: int | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)
    max_task_tokens: int | None = Field(default=None)
    additional_settings: dict[str, t.Any] | None = Field(default=None)
