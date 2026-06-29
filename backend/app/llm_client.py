from typing import Literal

import httpx
from pydantic import BaseModel

from .settings import LLMSettings, get_llm_settings


MessageRole = Literal["system", "user", "assistant"]


class LLMMessage(BaseModel):
    role: MessageRole
    content: str


class LLMStatus(BaseModel):
    provider: str
    base_url: str
    model: str
    available: bool
    error_message: str | None = None


class LLMModelList(BaseModel):
    provider: str
    base_url: str
    default_model: str
    models: list[str]
    available: bool
    error_message: str | None = None


class LLMClientError(Exception):
    pass


class LLMTimeoutError(LLMClientError):
    pass


class LocalLLMClient:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or get_llm_settings()

    def check_status(self) -> LLMStatus:
        model_list = self.list_models()

        return LLMStatus(
            provider=self.settings.provider,
            base_url=self.settings.base_url,
            model=self.settings.model,
            available=model_list.available,
            error_message=model_list.error_message,
        )

    def list_models(self) -> LLMModelList:
        try:
            with httpx.Client(
                timeout=self.settings.timeout_seconds,
            ) as client:
                response = client.get(
                    self._url("/models"),
                    headers=self._headers(),
                )
                response.raise_for_status()
                data = response.json()

        except (httpx.HTTPError, ValueError) as exc:
            return LLMModelList(
                provider=self.settings.provider,
                base_url=self.settings.base_url,
                default_model=self.settings.model,
                models=[],
                available=False,
                error_message=str(exc),
            )

        models = _extract_model_ids(data)

        return LLMModelList(
            provider=self.settings.provider,
            base_url=self.settings.base_url,
            default_model=self.settings.model,
            models=models,
            available=True,
        )

    def create_chat_completion(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, object] | None = None,
    ) -> str:
        payload = {
            "model": model or self.settings.model,
            "messages": [
                message.model_dump()
                for message in messages
            ],
            "temperature": (
                self.settings.temperature
                if temperature is None
                else temperature
            ),
            "max_tokens": (
                self.settings.max_tokens
                if max_tokens is None
                else max_tokens
            ),
            "stream": False,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        try:
            with httpx.Client(
                timeout=self.settings.timeout_seconds,
            ) as client:
                response = client.post(
                    self._url("/chat/completions"),
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                "Local LLM request timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(
                f"Local LLM request failed: {exc}"
            ) from exc
        except ValueError as exc:
            raise LLMClientError(
                "Local LLM returned invalid JSON."
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                "Local LLM response did not include message content."
            ) from exc

        if not isinstance(content, str):
            raise LLMClientError(
                "Local LLM message content was not text."
            )

        return content

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.settings.base_url.rstrip('/')}{path}"


def _extract_model_ids(data: object) -> list[str]:
    if not isinstance(data, dict):
        return []

    raw_models = data.get("data")

    if not isinstance(raw_models, list):
        return []

    model_ids: list[str] = []

    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            continue

        model_id = raw_model.get("id")

        if isinstance(model_id, str) and model_id:
            model_ids.append(model_id)

    return sorted(set(model_ids))
