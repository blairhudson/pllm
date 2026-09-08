from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx


PrivacyMode = Literal["strict_he", "he_preprocessed", "trusted_backend", "manifest_only"]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend: str
    responses: bool
    chat_completions: bool
    streaming: bool
    tools: bool
    stateful_responses: bool
    token_ids: bool
    logprobs: bool
    model_loading: bool
    strict_he: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "responses": self.responses,
            "chat_completions": self.chat_completions,
            "streaming": self.streaming,
            "tools": self.tools,
            "stateful_responses": self.stateful_responses,
            "token_ids": self.token_ids,
            "logprobs": self.logprobs,
            "model_loading": self.model_loading,
            "strict_he": self.strict_he,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class BackendModel:
    id: str
    owned_by: str
    backend: str
    privacy_mode: PrivacyMode
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_object(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "created": int(self.metadata.get("created", 0)),
            "owned_by": self.owned_by,
            "he": {
                "backend": self.backend,
                "privacy_mode": self.privacy_mode,
                **{k: v for k, v in self.metadata.items() if k != "created"},
            },
        }


class BackendAdapter(Protocol):
    name: str
    capabilities: BackendCapabilities

    async def list_models(self) -> list[BackendModel]: ...
    async def create_response(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def stream_response(self, body: dict[str, Any]): ...


class HTTPBackendAdapter:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "local",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._owned_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    async def close(self) -> None:
        if self._owned_client:
            await self.client.aclose()

    async def _get_json(self, path: str) -> Any:
        response = await self.client.get(f"{self.base_url}{path}", headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def _post_json(self, path: str, body: dict[str, Any]) -> Any:
        response = await self.client.post(f"{self.base_url}{path}", headers=self.headers, json=body)
        response.raise_for_status()
        return response.json()
