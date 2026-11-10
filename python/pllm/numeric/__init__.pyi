from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.numeric._declarations import ExactSourceFieldBoundaries as ExactSourceFieldBoundaries
from pllm.numeric._declarations import ReplicatedRing64Q18Edges as ReplicatedRing64Q18Edges
from pllm.numeric._declarations import Ring50Q12Edges as Ring50Q12Edges
from pllm.numeric._declarations import SourceCalibration as SourceCalibration
from pllm.numeric._declarations import NumericDomain as NumericDomain
from pllm.numeric._declarations import SourceNumericDomainCertificates as SourceNumericDomainCertificates

class Q11I16Ring64Boundary(ComponentRef):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class SourceFloat32(ComponentRef):
    NUMERIC_PROFILE: str
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
