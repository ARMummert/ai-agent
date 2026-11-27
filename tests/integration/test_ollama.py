import httpx
import pytest


@pytest.mark.integration
class TestOllama:
    def test_ollama_server_is_available(self, ollama_client: httpx.Client):
        response = ollama_client.get("/api/version")

        assert response.status_code == 200

        payload = response.json()

        assert isinstance(payload, dict)
        assert "version" in payload
        assert isinstance(payload["version"], str)
        assert payload["version"].strip()
