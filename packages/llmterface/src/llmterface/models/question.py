import typing as t
from textwrap import dedent

import llmterface.exceptions as ex
from llmterface.models.generic_config import GenericConfig
from llmterface.models.generic_response import GenericResponse
from pydantic import BaseModel, ConfigDict, Field

TRes = t.TypeVar("TRes", bound=BaseModel | str | int | float | bool)


class Question(BaseModel, t.Generic[TRes]):
    model_config = ConfigDict(extra="forbid")
    config: GenericConfig[TRes] | None = Field(
        default=None,
        description="Optional configuration for this question.This will override chat and module level configurations.",
    )
    question: str = Field(default="", description="The question to ask the AI.")
    max_retries: int = Field(default=1, description="Maximum number of retries for this question.")

    def get_question(self) -> str:
        """
        called to get the question string to send to the AI provider.
        Subclasses can override this to get fancy
        """
        return dedent(self.question).strip()

    @staticmethod
    def on_retry(
        q: "Question",
        response: GenericResponse | None = None,
        e: Exception | None = None,
        retries: int = 0,
    ) -> t.Optional["Question"]:
        """
        Override this method to provide custom retry logic.
        This method should return a new Question instance to retry with
        or None to stop retrying.
        """
        if retries >= q.max_retries:
            return None
        fail_msg = "Please ensure your response strictly follows the required format."
        if isinstance(e, ex.ProviderError):
            return q
        data = q.model_dump()
        if isinstance(e, ex.SchemaError):
            try:
                i = q.question.index(fail_msg)
            except ValueError:
                msg = f"{q.question.strip()}\n{fail_msg}"
            else:
                msg = f"{q.question[:i].strip()}\n{fail_msg}"
            if response:
                msg += f"\n\nYour previous erroneous response was:\n{response.text}"
            else:
                msg += "\n\nNo response was received."
            data["question"] = msg
            return q.__class__.model_validate(data)
        return None

    def get_config(self) -> GenericConfig | None:
        """
        Returns a dictionary of configuration options for the question.
        Subclasses can override this method to get fancy
        """
        return self.config

    @property
    def prompt(self) -> str:
        return self.get_question()

    def with_prioritized_config(self, ordered_configs: t.Sequence[GenericConfig[TRes] | None]):
        if self.config is not None:
            return self
        for cfg in ordered_configs:
            if cfg is not None:
                return self.model_copy(update={"config": cfg})
        raise RuntimeError("No configuration available to prioritize.")
