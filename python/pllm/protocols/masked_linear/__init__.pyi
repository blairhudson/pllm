from pllm.configuration import ComponentDescriptor
from pllm.protocols.base import ProtocolMethod

class MaskedLinear(ProtocolMethod):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
