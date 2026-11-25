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

    def test_create_model_uses_default_base_url(self, expected_default_base_url: str):

        llm = create_model()

        assert llm.base_url == expected_default_base_url

    def test_create_model_uses_custom_base_url(self, base_url: str):

        llm = create_model(base_url=base_url)

        assert llm.base_url == base_url

    def test_create_model_accepts_custom_configuration(
        self,
        model_name: str,
        base_url: str,
    ):

        llm = create_model(
            model_name=model_name,
            base_url=base_url,
        )

        assert llm.model == model_name
        assert llm.base_url == base_url
