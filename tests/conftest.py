import pytest

CUSTOM_MODELS = [
    "llama3.2",
    "qwen3:4b",
    "qwen3:8b",
    "llama3.1:8b",
]


@pytest.fixture
def expected_default_model() -> str:
    return "phi4-mini"


@pytest.fixture(params=CUSTOM_MODELS)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param
