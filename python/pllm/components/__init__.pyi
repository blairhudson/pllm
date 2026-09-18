from pllm.configuration import BinaryTableGatedMultiplyQ7 as BinaryTableGatedMultiplyQ7
from pllm.configuration import ComponentDescriptor as ComponentDescriptor
from pllm.configuration import ComponentRef as ComponentRef
from pllm.configuration import (
    ChunkedIndependentLanesProtectedTensorSchedule as ChunkedIndependentLanesProtectedTensorSchedule,
)
from pllm.configuration import (
    IndependentLanesProtectedTensorSchedule as IndependentLanesProtectedTensorSchedule,
)
from pllm.configuration import KvCacheEviction as KvCacheEviction
from pllm.configuration import R03CrtGatedMultiplyQ7 as R03CrtGatedMultiplyQ7
from pllm.configuration import ScalarProtectedTensorSchedule as ScalarProtectedTensorSchedule

def list_components() -> tuple[ComponentDescriptor, ...]: ...
def get_component(identity: str) -> ComponentDescriptor: ...
