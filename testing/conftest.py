import pytest


@pytest.fixture(autouse=True)
def clear_provider_registry() -> None:
    """
    Clear the provider registry before each test to ensure a clean state.
    """
    from llmterface.providers.discovery import _PROVIDER_SPECS, load_provider_configs

    _PROVIDER_SPECS.clear()
    load_provider_configs()
