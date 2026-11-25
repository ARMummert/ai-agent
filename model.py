from langchain_ollama import ChatOllama

DEFAULT_MODEL = "phi4-mini"
DEFAULT_BASE_URL = "http://localhost:11434"


def create_model(model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> ChatOllama:
    return ChatOllama(model=model_name, base_url=base_url)
