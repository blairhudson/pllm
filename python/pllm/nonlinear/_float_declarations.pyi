from collections.abc import Mapping
from typing import Literal, Self

from pllm.configuration import ComponentDescriptor, ComponentRef

class F32Activation(ComponentRef):
    descriptor: ComponentDescriptor
    def __init__(self, *, function: Literal["source"] = "source",
                 execution_owner: Literal["client"] | None = None) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

__all__ = ["F32Activation"]
