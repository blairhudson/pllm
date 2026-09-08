from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import BackendCapabilities, BackendModel, HTTPBackendAdapter
from .translation import chat_completion_to_response, responses_to_chat


async def _iter_sse(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    event_type: str | None = None
    data: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data:
                raw = "\n".join(data)
                if raw != "[DONE]":
                    value = json.loads(raw)
                    if event_type and "type" not in value:
                        value["type"] = event_type
                    yield value
            event_type = None
            data.clear()
        elif line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        raw = "\n".join(data)
        if raw != "[DONE]":
            value = json.loads(raw)
            if event_type and "type" not in value:
                value["type"] = event_type
            yield value


class NativeResponsesAdapter(HTTPBackendAdapter):
    strip_state = False

    async def _native_body(self, body: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        value = dict(body)
        value["stream"] = stream
        if self.strip_state:
            value.pop("previous_response_id", None)
            value.pop("conversation", None)
        return value

    async def create_response(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._post_json("/v1/responses", await self._native_body(body, stream=False))

    async def stream_response(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/responses",
            headers={**self.headers, "Accept": "text/event-stream"},
            json=await self._native_body(body, stream=True),
        ) as response:
            response.raise_for_status()
            async for event in _iter_sse(response):
                yield event


class VLLMAdapter(NativeResponsesAdapter):
    name = "vllm"
    capabilities = BackendCapabilities(
        backend=name,
        responses=True,
        chat_completions=True,
        streaming=True,
        tools=True,
        stateful_responses=True,
        token_ids=True,
        logprobs=True,
        model_loading=True,
        strict_he=False,
        notes=("Responses API is native", "strict HE requires a weight/runtime plugin, not the HTTP frontend"),
    )

    async def list_models(self) -> list[BackendModel]:
        value = await self._get_json("/v1/models")
        return [
            BackendModel(
                id=str(item["id"]),
                owned_by=str(item.get("owned_by", "vllm")),
                backend=self.name,
                privacy_mode="trusted_backend",
                metadata={"created": item.get("created", 0)},
            )
            for item in value.get("data", [])
        ]


class OllamaAdapter(NativeResponsesAdapter):
    name = "ollama"
    strip_state = True
    capabilities = BackendCapabilities(
        backend=name,
        responses=True,
        chat_completions=True,
        streaming=True,
        tools=True,
        stateful_responses=False,
        token_ids=False,
        logprobs=False,
        model_loading=True,
        strict_he=False,
        notes=("Responses API is non-stateful", "previous_response_id must be maintained by the HE client"),
    )

    async def list_models(self) -> list[BackendModel]:
        try:
            value = await self._get_json("/v1/models")
            return [
                BackendModel(str(item["id"]), "ollama", self.name, "trusted_backend")
                for item in value.get("data", [])
            ]
        except Exception:
            value = await self._get_json("/api/tags")
            return [
                BackendModel(
                    str(item.get("name") or item.get("model")),
                    "ollama",
                    self.name,
                    "trusted_backend",
                    {"size": item.get("size")},
                )
                for item in value.get("models", [])
            ]


class LlamaCppAdapter(NativeResponsesAdapter):
    name = "llama.cpp"
    capabilities = BackendCapabilities(
        backend=name,
        responses=True,
        chat_completions=True,
        streaming=True,
        tools=True,
        stateful_responses=False,
        token_ids=True,
        logprobs=True,
        model_loading=True,
        strict_he=False,
        notes=("Responses may be implemented through the server's Chat Completions shim",),
    )

    async def list_models(self) -> list[BackendModel]:
        value = await self._get_json("/v1/models")
        return [
            BackendModel(str(item["id"]), str(item.get("owned_by", "llama.cpp")), self.name, "trusted_backend")
            for item in value.get("data", [])
        ]

    async def create_response(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            return await super().create_response(body)
        except httpx.HTTPError:
            chat = await self._post_json("/v1/chat/completions", responses_to_chat(body))
            return chat_completion_to_response(chat, model=str(body.get("model", "local")))


class MLXLMAdapter(HTTPBackendAdapter):
    name = "mlx-lm"
    capabilities = BackendCapabilities(
        backend=name,
        responses=False,
        chat_completions=True,
        streaming=True,
        tools=False,
        stateful_responses=False,
        token_ids=False,
        logprobs=False,
        model_loading=True,
        strict_he=False,
        notes=("Responses is supplied by the HE gateway translation layer",),
    )

    async def list_models(self) -> list[BackendModel]:
        value = await self._get_json("/v1/models")
        return [
            BackendModel(str(item["id"]), str(item.get("owned_by", "mlx-lm")), self.name, "trusted_backend")
            for item in value.get("data", [])
        ]

    async def create_response(self, body: dict[str, Any]) -> dict[str, Any]:
        chat = await self._post_json("/v1/chat/completions", responses_to_chat(body))
        return chat_completion_to_response(chat, model=str(body.get("model", "local")))

    async def stream_response(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"
        seq = 0
        created = {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "status": "in_progress",
            "model": body.get("model"),
            "output": [],
            "usage": None,
        }
        yield {"type": "response.created", "sequence_number": seq, "response": created}
        seq += 1
        yield {"type": "response.in_progress", "sequence_number": seq, "response": created}
        seq += 1
        item = {"id": message_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
        yield {"type": "response.output_item.added", "sequence_number": seq, "output_index": 0, "item": item}
        seq += 1
        yield {
            "type": "response.content_part.added",
            "sequence_number": seq,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        }
        seq += 1
        chunks: list[str] = []
        chat_body = responses_to_chat({**body, "stream": True})
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            headers={**self.headers, "Accept": "text/event-stream"},
            json=chat_body,
        ) as response:
            response.raise_for_status()
            async for event in _iter_sse(response):
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    chunks.append(str(delta))
                    yield {
                        "type": "response.output_text.delta",
                        "sequence_number": seq,
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": str(delta),
                        "logprobs": [],
                    }
                    seq += 1
        text = "".join(chunks)
        final = chat_completion_to_response(
            {"id": response_id, "created": time.time(), "choices": [{"message": {"content": text}}]},
            model=str(body.get("model", "local")),
        )
        yield {
            "type": "response.output_text.done",
            "sequence_number": seq,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        }
        seq += 1
        yield {"type": "response.completed", "sequence_number": seq, "response": final}


class GenericOpenAIAdapter(VLLMAdapter):
    name = "openai-compatible"
    capabilities = BackendCapabilities(
        backend=name,
        responses=True,
        chat_completions=True,
        streaming=True,
        tools=True,
        stateful_responses=False,
        token_ids=False,
        logprobs=True,
        model_loading=False,
        strict_he=False,
        notes=("capabilities vary by upstream",),
    )
