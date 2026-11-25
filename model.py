from langchain_ollama import ChatOllama


def create_model() -> ChatOllama:
    return ChatOllama(model="phi4-mini")
