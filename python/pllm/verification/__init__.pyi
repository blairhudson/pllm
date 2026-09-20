from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.runtime.linear_integrity import LinearCheckKey as LinearCheckKey
from pllm.runtime.linear_integrity import LinearCheckResult as LinearCheckResult
from pllm.runtime.linear_integrity import LinearIntegrityError as LinearIntegrityError
from pllm.runtime.linear_integrity import check_linear_result as check_linear_result
from pllm.runtime.linear_integrity import create_linear_check_key as create_linear_check_key

class VerificationScheme(ComponentRef): ...

class FreivaldsVerify(VerificationScheme):
    descriptor: ComponentDescriptor
    target_failure_bits: int
    def __init__(self, target_failure_bits: int = 40) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class LinearIntegrity(VerificationScheme):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
