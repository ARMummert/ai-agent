from langchain_ollama import ChatOllama

from model import create_model


class TestModel:
    def test_create_model_returns_chat_ollama(self):

        llm = create_model()

        assert isinstance(llm, ChatOllama)

    def test_create_model_uses_default_model(self, expected_default_model: str):

        llm = create_model()

        assert llm.model == expected_default_model

    def test_create_model_uses_custom_model(self, model_name: str):

        llm = create_model(model_name=model_name)

        assert llm.model == model_name
