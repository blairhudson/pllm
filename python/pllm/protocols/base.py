from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef


class ProtocolMethod(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...
