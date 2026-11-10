from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.schedulers._declarations import HELevelErrorSchedule as HELevelErrorSchedule
from pllm.schedulers._declarations import HELevelSchemePlacement as HELevelSchemePlacement
from pllm.schedulers._declarations import SetupOnlineRetirementSchedule as SetupOnlineRetirementSchedule
from pllm.schedulers._declarations import FSSOnlineSchedule as FSSOnlineSchedule
from pllm.schedulers._declarations import BoundedConsumingProtectedDag as BoundedConsumingProtectedDag

class MlpPrefillTiling(ComponentRef):
    descriptor: ComponentDescriptor
    def __init__(self, *, tile_rows: int = 64) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class ProtectedScheduler(ComponentRef): ...

class BoundedIndependentElementsProtectedTensorSchedule(ProtectedScheduler):
    descriptor: ComponentDescriptor
    def __init__(self, *, max_elements: int = 128) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class ScalarProtectedTensorSchedule(ProtectedScheduler):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class IndependentLanesProtectedTensorSchedule(ProtectedScheduler):
    descriptor: ComponentDescriptor
    def __init__(self, *, max_elements: int = 4) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class ChunkedIndependentLanesProtectedTensorSchedule(ProtectedScheduler):
    descriptor: ComponentDescriptor
    def __init__(self, *, max_elements: int) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
