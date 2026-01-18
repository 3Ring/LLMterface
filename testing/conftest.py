import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        return

    skip_integration = pytest.mark.skip(reason="integration tests disabled (use --integration)")

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def clear_provider_registry() -> None:
    """
    Clear the provider registry before each test to ensure a clean state.
    """
    from llmterface.providers.discovery import _PROVIDER_SPECS, load_provider_configs

    _PROVIDER_SPECS.clear()
    load_provider_configs()
