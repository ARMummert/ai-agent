from collections.abc import Iterator

import httpx
import pytest

from model import create_model

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

CUSTOM_TEMPERATURES = [
    0.0,
    0.2,
    0.5,
    0.7,
    1.0,
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


@pytest.fixture
def expected_default_temperature() -> float:
    return 0.0


@pytest.fixture(params=CUSTOM_TEMPERATURES)
def temperature(request: pytest.FixtureRequest) -> float:
    return request.param


# This fixture uses the actual server url returned by the production model.
@pytest.fixture
def ollama_client() -> Iterator[httpx.Client]:
    llm = create_model()

    assert llm.base_url is not None
    # Reusable httpx client
    with httpx.Client(base_url=llm.base_url, timeout=5.0, trust_env=False) as client:
        yield client
