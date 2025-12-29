class AiHandlerError(Exception):
    """Base class for all exceptions raised by the AI handler library."""

    pass


class ClientError(AiHandlerError):
    """Base class for client-related errors."""

    pass


class ProviderError(ClientError):
    """Raised when there is an error with the AI provider."""

    def __init__(self, message: str = "", original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception


class SchemaError(ClientError):
    """Raised when there is a schema validation error."""

    def __init__(self, message: str = "", original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception
