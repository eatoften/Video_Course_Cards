from app.llm_client import (
    LLMMessage,
    LocalLLMClient,
    _should_trust_environment,
)
from app.settings import LLMSettings


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": '{"cards": []}'}}]}


class RecordingHttpClient:
    instances: list["RecordingHttpClient"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.payload: dict[str, object] | None = None
        self.__class__.instances.append(self)

    def __enter__(self) -> "RecordingHttpClient":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def post(self, *args: object, **kwargs: object) -> FakeResponse:
        self.payload = kwargs["json"]  # type: ignore[assignment]
        return FakeResponse()


def test_local_ollama_request_disables_thinking_and_ignores_proxy(
    monkeypatch,
) -> None:
    RecordingHttpClient.instances.clear()
    monkeypatch.setattr("app.llm_client.httpx.Client", RecordingHttpClient)
    client = LocalLLMClient(
        LLMSettings(
            base_url="http://localhost:11434/v1",
            reasoning_effort="none",
        )
    )

    output = client.create_chat_completion(
        [LLMMessage(role="user", content="Return JSON.")]
    )

    recorded = RecordingHttpClient.instances[0]
    assert output == '{"cards": []}'
    assert recorded.kwargs["trust_env"] is False
    assert recorded.payload is not None
    assert recorded.payload["reasoning_effort"] == "none"


def test_remote_provider_can_use_environment_proxy() -> None:
    assert _should_trust_environment("https://example.com/v1") is True
    assert _should_trust_environment("http://127.0.0.1:11434/v1") is False
