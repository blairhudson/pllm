from pathlib import Path
from typing import Any, Mapping

from pllm.configuration import Model
from pllm.runtime.loaders import ModelLoadError as ModelLoadError
from pllm.runtime.models import ModelManifest as ModelManifest

def load_model(
    value: Model | str | Mapping[str, Any],
    *,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    api_key: str | None = None,
) -> ModelManifest: ...
