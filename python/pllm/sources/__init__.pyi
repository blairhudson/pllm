from typing import Any, Protocol

from pllm.configuration import Model

class ModelSource(Protocol):
    source: str
    kind: str
    model_id: str | None
    def to_spec(self) -> dict[str, Any]: ...

class TinyModel(Model):
    def __init__(self, family: str = "qwen2", *, model_id: str | None = None) -> None: ...

class BundleModel(Model):
    def __init__(
        self,
        path: str,
        *,
        format: str = "huggingface",
        model_id: str | None = None,
    ) -> None: ...
