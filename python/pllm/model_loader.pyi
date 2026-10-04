from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping

from pllm.configuration import Model
from pllm.runtime.loaders import ModelLoadError as ModelLoadError
from pllm.runtime.models import ModelManifest as ModelManifest

def expected_model_id(model: Model) -> str: ...

@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model: Model
    manifest: ModelManifest
    path: Path | None
    checkpoint_digest: str | None
    source_lock_digest: str | None

def resolve_model(
    value: Model | str | Mapping[str, Any], *, token: str | bool | None = None,
    cache_dir: str | Path | None = None, api_key: str | None = None,
) -> ResolvedModel: ...

def load_model(
    value: Model | str | Mapping[str, Any],
    *,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    api_key: str | None = None,
) -> ModelManifest: ...
