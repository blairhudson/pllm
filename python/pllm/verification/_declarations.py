"""Verification declarations requiring compiler/runtime binding, not executors.

Construction performs no publication, challenge generation, or window checks.
These identities and descriptors are not implementation or security evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef


class TwoLevelCodedBatchVerification(ComponentRef):
    """Declare two-level coded batch checks; do not generate or check challenges."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/two-level-coded-batch-verification/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/verification-scheme", category_version="1",
        lifecycle_phase="online",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ValidatedMatrixCodePublication(ComponentRef):
    """Declare validated matrix-code publication; do not publish or validate."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/validated-matrix-code-publication/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/verification-scheme", category_version="1",
        lifecycle_phase="offline",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class OriginalDotWindow(ComponentRef):
    """Declare an original-dot verification window; do not perform its checks."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/original-dot-window/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/verification-scheme", category_version="1",
        lifecycle_phase="online",
        parameter_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "TwoLevelCodedBatchVerification",
    "ValidatedMatrixCodePublication",
    "OriginalDotWindow",
]


class PerEdgeRangeWorkAdmission(ComponentRef):
    """Declare per-edge range/work admission requirements without running checks."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/per-edge-range-work-admission/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/verification-scheme", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ += ["PerEdgeRangeWorkAdmission"]
