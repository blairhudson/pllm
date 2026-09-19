from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.runtime.preparation_server import create_preparation_app as create_preparation_app

class PreparationProvider(ComponentRef): ...

class ModelAwareCorrections(PreparationProvider):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class BFVCorrelations(PreparationProvider):
    descriptor: ComponentDescriptor
    def __init__(self, *, poly_modulus_degree: int = 4096) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class HEAuthenticatedPreprocessing(PreparationProvider):
    descriptor: ComponentDescriptor
    def __init__(
        self,
        *,
        poly_modulus_degree: int = 4096,
        threads: int = 1,
        enable_linear_correlations: bool = True,
    ) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
