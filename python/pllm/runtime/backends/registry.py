from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters import GenericOpenAIAdapter, LlamaCppAdapter, MLXLMAdapter, OllamaAdapter, VLLMAdapter
from .base import BackendAdapter, BackendModel


ADAPTERS = {
    "vllm": VLLMAdapter,
    "ollama": OllamaAdapter,
    "llama.cpp": LlamaCppAdapter,
    "llama_cpp": LlamaCppAdapter,
    "mlx-lm": MLXLMAdapter,
    "mlx_lm": MLXLMAdapter,
    "openai": GenericOpenAIAdapter,
    "openai-compatible": GenericOpenAIAdapter,
}


class BackendRegistry:
    def __init__(self) -> None:
        self.adapters: dict[str, BackendAdapter] = {}
        self.model_routes: dict[str, str] = {}

    def add(self, name: str, adapter: BackendAdapter) -> None:
        self.adapters[name] = adapter

    async def refresh(self) -> list[BackendModel]:
        models: list[BackendModel] = []
        self.model_routes.clear()
        for name, adapter in self.adapters.items():
            for model in await adapter.list_models():
                public_id = model.id if model.id not in self.model_routes else f"{name}/{model.id}"
                if public_id != model.id:
                    model = BackendModel(public_id, model.owned_by, model.backend, model.privacy_mode, model.metadata)
                self.model_routes[public_id] = name
                models.append(model)
        return models

    def adapter_for(self, model: str) -> BackendAdapter:
        return self.resolve(model)[0]

    def resolve(self, model: str) -> tuple[BackendAdapter, str]:
        """Return adapter and its upstream model ID for a public model name."""
        route = self.model_routes.get(model)
        upstream = model
        if route is None:
            if "/" in model and model.split("/", 1)[0] in self.adapters:
                route, upstream = model.split("/", 1)
            elif len(self.adapters) == 1:
                route = next(iter(self.adapters))
            else:
                raise KeyError(f"no backend route for model {model!r}")
        elif model.startswith(route + "/"):
            upstream = model.split("/", 1)[1]
        return self.adapters[route], upstream

    @classmethod
    def from_config(cls, config: list[dict[str, Any]]) -> "BackendRegistry":
        registry = cls()
        for item in config:
            kind = str(item["kind"])
            adapter_type = ADAPTERS.get(kind)
            if adapter_type is None:
                raise ValueError(f"unsupported backend kind: {kind}")
            name = str(item.get("name", kind))
            registry.add(
                name,
                adapter_type(
                    base_url=str(item["base_url"]),
                    api_key=str(item.get("api_key", "local")),
                    timeout=float(item.get("timeout", 30.0)),
                ),
            )
        return registry
