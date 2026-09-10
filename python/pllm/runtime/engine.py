from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import ModelManifest, StageSpec


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    name: str
    model_sources: tuple[str, ...]
    protocols: tuple[str, ...]
    online_fhe: bool
    he_preprocessed: bool
    continuous_batching: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_sources": list(self.model_sources),
            "protocols": list(self.protocols),
            "online_fhe": self.online_fhe,
            "he_preprocessed": self.he_preprocessed,
            "continuous_batching": self.continuous_batching,
            "notes": list(self.notes),
        }


class HEEngine(Protocol):
    capabilities: EngineCapabilities

    async def load(self, manifest: ModelManifest) -> None: ...
    async def unload(self, model_id: str) -> None: ...
    async def execute_stage(self, model_id: str, stage: StageSpec, payloads: list[bytes]) -> list[bytes]: ...
    def seeded_stage_ids(self, model_id: str) -> tuple[str, ...]: ...
    def seeded_profile(self, model_id: str, stage_id: str) -> Any: ...


class MaskedTransformerHEEngine(HEEngine, Protocol):
    async def stage_metadata(self, model_id: str, stage_id: str) -> Any: ...
    async def create_local_correlations(
        self, model_id: str, stage_id: str, *, rows: int = 1, count: int = 1, seed: int | None = None
    ) -> list[Any]: ...
    async def client_bundle_bytes(self, model_id: str) -> bytes: ...


class BackendModelImporter(Protocol):
    """Control-plane contract for vLLM/MLX/Ollama/llama.cpp integration.

    Normal OpenAI-compatible HTTP endpoints cannot execute encrypted activations.
    A strict-HE deployment imports the same source weights into an HEEngine via
    this interface; trusted passthrough remains a separate backend mode.
    """

    async def inspect(self, source: str) -> ModelManifest: ...
    async def import_into(self, source: str, engine: HEEngine) -> ModelManifest: ...


class EngineStageExecutor:
    """Adapt an :class:`HEEngine` stage to the continuous batching scheduler."""

    def __init__(self, engine: HEEngine, model_id: str, stage: StageSpec) -> None:
        self.engine = engine
        self.model_id = model_id
        self.stage = stage

    async def execute(self, payloads: list[bytes]) -> list[bytes]:
        return await self.engine.execute_stage(self.model_id, self.stage, payloads)
