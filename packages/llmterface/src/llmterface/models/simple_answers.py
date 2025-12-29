from pydantic import BaseModel


class SimpleString(BaseModel):
    """
    Enforce that the answer is a simple string.
    """

    response: str


class SimpleInteger(BaseModel):
    """
    Enforce that the answer is a simple integer.
    """

    response: int


class SimpleFloat(BaseModel):
    """
    Enforce that the answer is a simple float.
    """

    response: float
