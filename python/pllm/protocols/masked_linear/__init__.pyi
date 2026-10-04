from pllm.configuration import ComponentDescriptor
from pllm.components._planned import PendingComponent as PendingMethod
from pllm.protocols.base import ProtocolMethod

class MaskedLinear(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(self, *, output_encoding: str = "raw", prefill_chunk_rows: int = 0) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class SlalomTeeVerifiableInference(PendingMethod): ...
