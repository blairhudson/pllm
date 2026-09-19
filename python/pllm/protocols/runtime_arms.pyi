from pllm.configuration import ComponentDescriptor
from pllm.protocols.base import ProtocolMethod

class GuardedLinear(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(
        self,
        *,
        max_rows_per_request: int = 4096,
        max_rows_per_owner_stage: int = 16384,
        max_requests_per_minute: int = 4096,
        output_dither_bound: int = 0,
    ) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class BlindedLinear(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class SecureLinear(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class DirectFHE(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class CleartextLinear(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
