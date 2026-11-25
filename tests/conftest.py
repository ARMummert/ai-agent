import pytest

CUSTOM_MODELS = [
    "llama3.2",
    "qwen3:4b",
    "qwen3:8b",
    "llama3.1:8b",
]

CUSTOM_BASE_URLS = [
    "http://127.0.0.1:11434",
    "http://localhost:11435",
    "http://127.0.0.1:8080",
]


@pytest.fixture
def expected_default_model() -> str:
    return "phi4-mini"


@pytest.fixture(params=CUSTOM_MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def expected_default_base_url() -> str:
    return "http://localhost:11434"


@pytest.fixture(params=CUSTOM_BASE_URLS)
def base_url(request: pytest.FixtureRequest) -> str:
    return request.param
