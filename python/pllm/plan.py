"""Public immutable compiled-plan facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pllm import _native


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """Opaque compiled plan exposing records but no execution methods."""

    _native: Any = field(repr=False)

    def __post_init__(self) -> None:
        from pllm import _native

        if not isinstance(self._native, _native.CompiledPlan):
            raise TypeError("compiled plan handle must come from pllm.compile")

    @property
    def logical_plan(self) -> bytes:
        return self._native.logical_plan

    @property
    def execution_plan(self) -> bytes:
        return self._native.execution_plan

    @property
    def plan_lock(self) -> bytes:
        return self._native.plan_lock

    @property
    def region_program(self) -> bytes:
        return self._native.region_program

    @property
    def configuration_digest(self) -> str:
        return self._native.configuration_digest

    @property
    def logical_plan_digest(self) -> str:
        return self._native.logical_plan_digest

    @property
    def execution_plan_digest(self) -> str:
        return self._native.execution_plan_digest

    @property
    def plan_lock_digest(self) -> str:
        return self._native.plan_lock_digest

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._native.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._native.output_shape


def _wrap(native: _native.CompiledPlan) -> CompiledPlan:
    return CompiledPlan(native)


def _unwrap(plan: CompiledPlan) -> _native.CompiledPlan:
    return plan._native


__all__ = ["CompiledPlan"]
