from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .authenticated_mpc import (
    BitDecompositionMask,
    InputMask,
    LinearCorrelation,
    MultiplicationTriple,
)


class InventoryMismatch(RuntimeError):
    """Raised when execution does not match the prepared operation plan."""


class InventoryExhausted(RuntimeError):
    """Raised when execution needs more prepared material."""


@dataclass(frozen=True, slots=True)
class PlanEntry:
    kind: str
    shape: tuple[int, ...]
    bit_width: int | None = None
    weight_digest: str | None = None
    weight: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class PreprocessingPlan:
    entries: list[PlanEntry] = field(default_factory=list)

    @property
    def operation_count(self) -> int:
        return len(self.entries)

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in self.entries:
            result[entry.kind] = result.get(entry.kind, 0) + 1
        return result


@dataclass(frozen=True, slots=True)
class PreparedEntry:
    plan: PlanEntry
    material: Any


def _shape(value: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


def _weight_digest(weight: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(weight, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


class RecordingPreprocessor:
    """Record the material requested by a secure arithmetic program.

    The delegate creates working material so the program can execute while its
    future preparation schedule is recorded. The resulting plan can then be
    generated before a real online request.
    """

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.modulus = delegate.modulus
        self.alpha = delegate.alpha
        self.plan = PreprocessingPlan()

    def input_mask(self, shape: Sequence[int]) -> InputMask:
        target = _shape(shape)
        self.plan.entries.append(PlanEntry("input_mask", target))
        return self.delegate.input_mask(target)

    def linear_correlation(self, weight: np.ndarray, input_shape: Sequence[int]) -> LinearCorrelation:
        target = _shape(input_shape)
        matrix = np.asarray(weight, dtype=np.int64).copy()
        self.plan.entries.append(
            PlanEntry(
                "linear_correlation",
                target,
                weight_digest=_weight_digest(matrix),
                weight=matrix,
            )
        )
        return self.delegate.linear_correlation(matrix, target)

    def multiplication_triple(self, shape: Sequence[int]) -> MultiplicationTriple:
        target = _shape(shape)
        self.plan.entries.append(PlanEntry("multiplication_triple", target))
        return self.delegate.multiplication_triple(target)

    def bit_mask(
        self,
        shape: Sequence[int],
        bit_width: int | None = None,
    ) -> BitDecompositionMask:
        target = _shape(shape)
        width = int(bit_width or self.modulus.bit_length())
        self.plan.entries.append(PlanEntry("bit_mask", target, bit_width=width))
        return self.delegate.bit_mask(target, width)


class PreparedInventory:
    """Single use material generated for one recorded execution plan."""

    def __init__(self, generator, entries: list[PreparedEntry]) -> None:
        self.generator = generator
        self.modulus = generator.modulus
        self.alpha = generator.alpha
        self._entries = entries
        self._position = 0

    @classmethod
    def generate(cls, plan: PreprocessingPlan, generator) -> "PreparedInventory":
        prepared: list[PreparedEntry] = []
        for entry in plan.entries:
            if entry.kind == "input_mask":
                material = generator.input_mask(entry.shape)
            elif entry.kind == "linear_correlation":
                if entry.weight is None:
                    raise InventoryMismatch("linear plan entry has no weight")
                material = generator.linear_correlation(entry.weight, entry.shape)
            elif entry.kind == "multiplication_triple":
                material = generator.multiplication_triple(entry.shape)
            elif entry.kind == "bit_mask":
                material = generator.bit_mask(entry.shape, entry.bit_width)
            else:
                raise InventoryMismatch(f"unknown plan operation: {entry.kind}")
            prepared.append(PreparedEntry(entry, material))
        return cls(generator, prepared)

    @property
    def consumed(self) -> int:
        return self._position

    @property
    def remaining(self) -> int:
        return len(self._entries) - self._position

    def _take(
        self,
        kind: str,
        shape: Sequence[int],
        *,
        bit_width: int | None = None,
        weight: np.ndarray | None = None,
    ):
        if self._position >= len(self._entries):
            raise InventoryExhausted("prepared material has been consumed")
        prepared = self._entries[self._position]
        expected = prepared.plan
        target = _shape(shape)
        if expected.kind != kind or expected.shape != target:
            raise InventoryMismatch(
                f"prepared operation {self._position} is {expected.kind}{expected.shape}, "
                f"but execution requested {kind}{target}"
            )
        if bit_width is not None and expected.bit_width != int(bit_width):
            raise InventoryMismatch("prepared bit width does not match execution")
        if weight is not None and expected.weight_digest != _weight_digest(weight):
            raise InventoryMismatch("prepared linear weight does not match execution")
        self._position += 1
        return prepared.material

    def input_mask(self, shape: Sequence[int]) -> InputMask:
        return self._take("input_mask", shape)

    def linear_correlation(self, weight: np.ndarray, input_shape: Sequence[int]) -> LinearCorrelation:
        return self._take("linear_correlation", input_shape, weight=weight)

    def multiplication_triple(self, shape: Sequence[int]) -> MultiplicationTriple:
        return self._take("multiplication_triple", shape)

    def bit_mask(
        self,
        shape: Sequence[int],
        bit_width: int | None = None,
    ) -> BitDecompositionMask:
        width = int(bit_width or self.modulus.bit_length())
        return self._take("bit_mask", shape, bit_width=width)
