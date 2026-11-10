from collections.abc import Mapping
from typing import Self

from pllm.configuration import ComponentDescriptor, ComponentRef

class LWEHighPrecisionComparison(ComponentRef):
    __slots__ = ()
    descriptor: ComponentDescriptor
    def __init__(self, *, key_graph: str, numeric: str) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class RLWEEncryptedAccumulatorSegmentedAffine(ComponentRef):
    __slots__ = ()
    descriptor: ComponentDescriptor
    def __init__(self, *, comparison: str, key_graph: str, numeric: str) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class HEEvalMod(ComponentRef):
    __slots__ = ()
    descriptor: ComponentDescriptor
    def __init__(self, *, numeric: str) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class HENormalization(ComponentRef):
    __slots__ = ()
    descriptor: ComponentDescriptor
    def __init__(
        self, *, square_reduce: str, inverse_sqrt: str, multiply: str,
        to_coefficients: str, ring_switch: str, extract: str, compare: str,
        evaluate: str, key_switch: str, modulus_raise: str, to_slots: str,
        evalmod: str, numeric: str,
    ) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class HESoftmaxSquareNormalize(ComponentRef):
    __slots__ = ()
    descriptor: ComponentDescriptor
    def __init__(
        self, *, square_reduce: str, reciprocal: str, multiply: str,
        to_coefficients: str, ring_switch: str, extract: str, compare: str,
        evaluate: str, key_switch: str, modulus_raise: str, to_slots: str,
        evalmod: str, numeric: str,
    ) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class HEActivationBridge(ComponentRef):
    __slots__ = ()
    descriptor: ComponentDescriptor
    def __init__(
        self, *, to_coefficients: str, ring_switch: str, extract: str,
        compare: str, evaluate: str, key_switch: str, modulus_raise: str,
        to_slots: str, evalmod: str, numeric: str,
    ) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, object]: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

__all__ = [
    "LWEHighPrecisionComparison",
    "RLWEEncryptedAccumulatorSegmentedAffine",
    "HEEvalMod",
    "HENormalization",
    "HESoftmaxSquareNormalize",
    "HEActivationBridge",
]
