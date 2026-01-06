import typing as t

from pydantic import BaseModel
from abc import ABC


class SimpleAnswersBase(BaseModel, ABC):
    response: t.Any


class SimpleString(SimpleAnswersBase):
    response: str


class SimpleInteger(SimpleAnswersBase):
    response: int


class SimpleFloat(SimpleAnswersBase):
    response: float


class SimpleBoolean(SimpleAnswersBase):
    response: bool


SIMPLE_MAP: dict[type, type[SimpleAnswersBase]] = {
    str: SimpleString,
    int: SimpleInteger,
    float: SimpleFloat,
    bool: SimpleBoolean,
}
