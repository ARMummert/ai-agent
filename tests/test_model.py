from langchain_ollama import ChatOllama

from model import create_model


class TestModel:

    def test_create_model_returns_chat_ollama(self):

        model = create_model()

        assert isinstance(model, ChatOllama)
