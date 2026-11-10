from typing import Any, Protocol

from pllm.configuration import Model
from pllm.sources._declarations import AuthenticatedSourceBindings as AuthenticatedSourceBindings
from pllm.sources._declarations import AuthenticatedOriginalProducer as AuthenticatedOriginalProducer
from pllm.sources._declarations import PublicCommittedQ18Source as PublicCommittedQ18Source
from pllm.sources._declarations import FSSSourceBinding as FSSSourceBinding
from pllm.sources._declarations import AuthenticatedSourceInputProducer as AuthenticatedSourceInputProducer
from pllm.sources._declarations import PinnedSemanticSourceBindings as PinnedSemanticSourceBindings

def _tiny_model_config() -> dict[str, Any]: ...

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
