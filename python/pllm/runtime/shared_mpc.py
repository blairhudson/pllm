from __future__ import annotations

import hashlib
import secrets
import sqlite3
from math import ceil, log2
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, Sequence

import msgpack
import numpy as np


class SharedMPCError(RuntimeError):
    """Raised when a two-party share protocol invariant fails."""


class BurnLedger(Protocol):
    def burn(self, session_id: str, party: int, kind: str, material_id: str) -> None: ...

    def burn_truncation(
        self,
        session_id: str,
        party: int,
        material_id: str,
        profile: tuple[int, int, int],
        element_count: int,
    ) -> None: ...


class PreprocessingBurnLedger:
    """Bounded process-local burn ledger; production needs a durable equivalent."""

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise SharedMPCError("preprocessing burn-ledger capacity must be positive")
        self.capacity = capacity
        self._lock = Lock()
        self._burned: set[tuple[str, int, str, str]] = set()
        self._truncation: dict[tuple[str, int], tuple[tuple[int, int, int], int]] = {}

    def burn(self, session_id: str, party: int, kind: str, material_id: str) -> None:
        key = (session_id, party, kind, material_id)
        with self._lock:
            if key in self._burned:
                raise SharedMPCError(f"{kind} preprocessing was already consumed")
            if len(self._burned) >= self.capacity:
                raise SharedMPCError("preprocessing burn ledger is full")
            self._burned.add(key)

    def burn_truncation(
        self,
        session_id: str,
        party: int,
        material_id: str,
        profile: tuple[int, int, int],
        element_count: int,
    ) -> None:
        material_key = (session_id, party, "truncation", material_id)
        session_key = (session_id, party)
        with self._lock:
            if material_key in self._burned:
                raise SharedMPCError("truncation preprocessing was already consumed")
            current_profile, used = self._truncation.get(session_key, (profile, 0))
            if current_profile != profile:
                raise SharedMPCError(
                    "truncation masks use inconsistent session security accounting"
                )
            if used + element_count > profile[2]:
                raise SharedMPCError("truncation session opening budget exhausted")
            if len(self._burned) >= self.capacity:
                raise SharedMPCError("preprocessing burn ledger is full")
            self._burned.add(material_key)
            self._truncation[session_key] = (profile, used + element_count)


class SQLitePreprocessingBurnLedger:
    """Restart-durable burn ledger; its database must be protected from rollback."""

    def __init__(self, path: str | Path) -> None:
        self._lock = Lock()
        self._connection = sqlite3.connect(Path(path), check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS burned ("
            "session_id TEXT NOT NULL, party INTEGER NOT NULL, kind TEXT NOT NULL, "
            "material_id TEXT NOT NULL, PRIMARY KEY (session_id, party, kind, material_id))"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS truncation_sessions ("
            "session_id TEXT NOT NULL, party INTEGER NOT NULL, security_bits INTEGER NOT NULL, "
            "per_element_bits INTEGER NOT NULL, budget INTEGER NOT NULL, used INTEGER NOT NULL, "
            "PRIMARY KEY (session_id, party))"
        )

    def burn(self, session_id: str, party: int, kind: str, material_id: str) -> None:
        with self._lock:
            try:
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO burned VALUES (?, ?, ?, ?)",
                        (session_id, party, kind, material_id),
                    )
            except sqlite3.IntegrityError as exc:
                raise SharedMPCError(f"{kind} preprocessing was already consumed") from exc

    def burn_truncation(
        self,
        session_id: str,
        party: int,
        material_id: str,
        profile: tuple[int, int, int],
        element_count: int,
    ) -> None:
        with self._lock:
            try:
                with self._connection:
                    row = self._connection.execute(
                        "SELECT security_bits, per_element_bits, budget, used "
                        "FROM truncation_sessions WHERE session_id = ? AND party = ?",
                        (session_id, party),
                    ).fetchone()
                    if row is None:
                        row = (*profile, 0)
                        self._connection.execute(
                            "INSERT INTO truncation_sessions VALUES (?, ?, ?, ?, ?, 0)",
                            (session_id, party, *profile),
                        )
                    if tuple(row[:3]) != profile:
                        raise SharedMPCError(
                            "truncation masks use inconsistent session security accounting"
                        )
                    if int(row[3]) + element_count > profile[2]:
                        raise SharedMPCError("truncation session opening budget exhausted")
                    self._connection.execute(
                        "INSERT INTO burned VALUES (?, ?, 'truncation', ?)",
                        (session_id, party, material_id),
                    )
                    self._connection.execute(
                        "UPDATE truncation_sessions SET used = used + ? "
                        "WHERE session_id = ? AND party = ?",
                        (element_count, session_id, party),
                    )
            except sqlite3.IntegrityError as exc:
                raise SharedMPCError("truncation preprocessing was already consumed") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()


_DEFAULT_BURN_LEDGER = PreprocessingBurnLedger(capacity=1_000_000)


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise SharedMPCError("shared tensor shape is invalid")
    result = tuple(value)
    if not result or any(item <= 0 for item in result):
        raise SharedMPCError("shared tensor shape is invalid")
    return result


def _bounded_elements(shape: tuple[int, ...], maximum: int) -> int:
    elements = 1
    for dimension in shape:
        if dimension > maximum // elements:
            raise SharedMPCError("shared tensor shape is too large")
        elements *= dimension
    return elements


def _ring_array(value: np.ndarray | Sequence[int] | int) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype == np.uint64:
        return np.ascontiguousarray(original)
    array = np.ascontiguousarray(original, dtype=np.int64)
    return array.view(np.uint64)


def _random_ring(shape: Sequence[int]) -> np.ndarray:
    target = _shape(shape)
    return np.frombuffer(secrets.token_bytes(int(np.prod(target)) * 8), dtype=np.uint64).reshape(
        target
    )


@dataclass(frozen=True, slots=True)
class SharedTensor:
    session_id: str
    tensor_id: str
    party: int
    values: np.ndarray
    scale: int = 1

    def __post_init__(self) -> None:
        if not self.session_id or not self.tensor_id:
            raise SharedMPCError("shared tensor identifiers are required")
        if type(self.party) is not int or self.party not in (0, 1):
            raise SharedMPCError("shared tensor party must be zero or one")
        if self.values.dtype != np.uint64 or not self.values.flags.c_contiguous:
            raise SharedMPCError("shared tensor must be a contiguous uint64 array")
        if self.values.ndim == 0 or self.values.size == 0:
            raise SharedMPCError("shared tensor cannot be scalar or empty")
        if type(self.scale) is not int or self.scale <= 0:
            raise SharedMPCError("shared tensor scale must be positive")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


@dataclass(frozen=True, slots=True)
class PublicQuantizedMatrix:
    coefficients: np.ndarray
    row_multipliers: np.ndarray
    scale: int

    def __post_init__(self) -> None:
        raw_coefficients = np.asarray(self.coefficients)
        raw_multipliers = np.asarray(self.row_multipliers)
        if not np.issubdtype(raw_coefficients.dtype, np.integer):
            raise SharedMPCError("quantized public coefficients must be integers")
        if not np.issubdtype(raw_multipliers.dtype, np.integer):
            raise SharedMPCError("quantized public row multipliers must be integers")
        if raw_coefficients.size and (
            int(raw_coefficients.min()) < -128 or int(raw_coefficients.max()) > 127
        ):
            raise SharedMPCError("quantized public coefficients must fit signed int8")
        coefficients = np.ascontiguousarray(raw_coefficients, dtype=np.int64)
        multipliers = np.ascontiguousarray(raw_multipliers, dtype=np.int64)
        if coefficients.ndim != 2 or coefficients.size == 0:
            raise SharedMPCError("quantized public matrix must be non-empty and two-dimensional")
        if multipliers.shape != (coefficients.shape[0],) or np.any(multipliers <= 0):
            raise SharedMPCError("quantized public row multipliers are invalid")
        if self.scale <= 0 or self.scale & (self.scale - 1):
            raise SharedMPCError("quantized public scale must be a positive power of two")
        coefficient_array = coefficients.astype(np.int8)
        coefficient_array.setflags(write=False)
        multipliers.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficient_array)
        object.__setattr__(self, "row_multipliers", multipliers)

    @classmethod
    def unit(cls, coefficients: np.ndarray) -> PublicQuantizedMatrix:
        values = np.asarray(coefficients)
        if values.ndim != 2:
            raise SharedMPCError("quantized public matrix must be two-dimensional")
        return cls(values, np.ones(values.shape[0], dtype=np.int64), 1)

    def scaled_bound(self, input_bound: int) -> int:
        if input_bound <= 0:
            raise SharedMPCError("public linear signed bound must be positive")
        bound = (
            max(
                sum(abs(int(value)) for value in row) * int(multiplier)
                for row, multiplier in zip(self.coefficients, self.row_multipliers, strict=True)
            )
            * input_bound
        )
        if bound >= 1 << 62:
            raise SharedMPCError("quantized public linear exceeds signed truncation range")
        return bound

    def output_bound(self, input_bound: int) -> int:
        bound = self.scaled_bound(input_bound)
        if self.scale == 1:
            return bound
        return (bound >> (self.scale.bit_length() - 1)) + 1


@dataclass(frozen=True, slots=True)
class BeaverTripleShare:
    session_id: str
    triple_id: str
    party: int
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id or not self.triple_id:
            raise SharedMPCError("Beaver triple identifiers are required")
        if self.party not in (0, 1):
            raise SharedMPCError("Beaver triple party must be zero or one")
        if self.a.dtype != np.uint64 or self.b.dtype != np.uint64 or self.c.dtype != np.uint64:
            raise SharedMPCError("Beaver triple shares must use uint64")
        if self.a.shape != self.b.shape or self.a.shape != self.c.shape or self.a.ndim == 0:
            raise SharedMPCError("Beaver triple share shapes do not match")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.a.shape


@dataclass(frozen=True, slots=True)
class OpeningFrame:
    session_id: str
    operation_id: str
    party: int
    d: np.ndarray
    e: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id or not self.operation_id or self.party not in (0, 1):
            raise SharedMPCError("opening frame identity is invalid")
        if (
            self.d.dtype != np.uint64
            or self.e.dtype != np.uint64
            or not self.d.flags.c_contiguous
            or not self.e.flags.c_contiguous
            or self.d.shape != self.e.shape
            or self.d.ndim == 0
            or self.d.size == 0
        ):
            raise SharedMPCError("opening frame tensors are invalid")

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "v": 1,
                "s": self.session_id,
                "o": self.operation_id,
                "p": self.party,
                "h": self.d.shape,
                "d": self.d.tobytes(),
                "e": self.e.tobytes(),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes, *, max_elements: int = 16_777_216) -> "OpeningFrame":
        if not payload or len(payload) > max_elements * 16 + 4096:
            raise SharedMPCError("opening frame is empty or too large")
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
            if not isinstance(value, dict) or set(value) != {"v", "s", "o", "p", "h", "d", "e"}:
                raise SharedMPCError("opening frame fields are invalid")
            if type(value["v"]) is not int or value["v"] != 1:
                raise SharedMPCError("opening frame version is unsupported")
            shape = _shape(value["h"])
            elements = _bounded_elements(shape, max_elements)
            expected = elements * 8
            if not isinstance(value["d"], bytes) or len(value["d"]) != expected:
                raise SharedMPCError("opening frame d payload is invalid")
            if not isinstance(value["e"], bytes) or len(value["e"]) != expected:
                raise SharedMPCError("opening frame e payload is invalid")
            if not isinstance(value["s"], str) or not isinstance(value["o"], str):
                raise SharedMPCError("opening frame identifiers are invalid")
            party = value["p"]
            if type(party) is not int or party not in (0, 1):
                raise SharedMPCError("opening frame party is invalid")
            return cls(
                session_id=value["s"],
                operation_id=value["o"],
                party=party,
                d=np.frombuffer(value["d"], dtype=np.uint64).reshape(shape).copy(),
                e=np.frombuffer(value["e"], dtype=np.uint64).reshape(shape).copy(),
            )
        except (KeyError, TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError) as exc:
            raise SharedMPCError("opening frame is malformed") from exc


@dataclass(frozen=True, slots=True)
class ValueOpeningFrame:
    session_id: str
    operation_id: str
    party: int
    value: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id or not self.operation_id or self.party not in (0, 1):
            raise SharedMPCError("value opening identity is invalid")
        if (
            self.value.dtype != np.uint64
            or not self.value.flags.c_contiguous
            or self.value.ndim == 0
            or self.value.size == 0
        ):
            raise SharedMPCError("value opening tensor is invalid")

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "v": 1,
                "s": self.session_id,
                "o": self.operation_id,
                "p": self.party,
                "h": self.value.shape,
                "x": self.value.tobytes(),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes, *, max_elements: int = 16_777_216) -> "ValueOpeningFrame":
        if not payload or len(payload) > max_elements * 8 + 8192:
            raise SharedMPCError("value opening frame is empty or too large")
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
            if not isinstance(value, dict) or set(value) != {"v", "s", "o", "p", "h", "x"}:
                raise SharedMPCError("value opening frame fields are invalid")
            if type(value["v"]) is not int or value["v"] != 1:
                raise SharedMPCError("value opening frame version is unsupported")
            shape = _shape(value["h"])
            elements = _bounded_elements(shape, max_elements)
            raw = value["x"]
            if not isinstance(raw, bytes) or len(raw) != elements * 8:
                raise SharedMPCError("value opening frame payload is invalid")
            if not isinstance(value["s"], str) or not isinstance(value["o"], str):
                raise SharedMPCError("value opening frame identifiers are invalid")
            party = value["p"]
            if type(party) is not int or party not in (0, 1):
                raise SharedMPCError("value opening frame party is invalid")
            return cls(
                session_id=value["s"],
                operation_id=value["o"],
                party=party,
                value=np.frombuffer(raw, dtype=np.uint64).reshape(shape).copy(),
            )
        except (KeyError, TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError) as exc:
            raise SharedMPCError("value opening frame is malformed") from exc


@dataclass(frozen=True, slots=True)
class TruncationMaskShare:
    session_id: str
    mask_id: str
    party: int
    bits: int
    random_bits: int
    maximum_magnitude: int
    session_security_bits: int
    per_element_security_bits: int
    opened_element_budget: int
    value: np.ndarray
    shifted: np.ndarray

    def __post_init__(self) -> None:
        if not self.session_id or not self.mask_id or self.party not in (0, 1):
            raise SharedMPCError("truncation mask identity is invalid")
        if not 0 < self.bits < 63:
            raise SharedMPCError("truncation mask bit count is invalid")
        if self.opened_element_budget <= 0:
            raise SharedMPCError("truncation mask security accounting is invalid")
        required_per_element_bits = self.session_security_bits + ceil(
            log2(self.opened_element_budget)
        )
        if self.maximum_magnitude <= 0:
            raise SharedMPCError("truncation mask magnitude is invalid")
        if (
            self.session_security_bits < 16
            or self.per_element_security_bits < required_per_element_bits
            or not self.bits < self.random_bits <= 61
            or self.random_bits <= self.per_element_security_bits
            or self.maximum_magnitude > 1 << (self.random_bits - self.per_element_security_bits)
        ):
            raise SharedMPCError("truncation mask security accounting is invalid")
        if (
            self.value.dtype != np.uint64
            or self.shifted.dtype != np.uint64
            or self.value.shape != self.shifted.shape
            or self.value.ndim == 0
        ):
            raise SharedMPCError("truncation mask tensors are invalid")


@dataclass(frozen=True, slots=True)
class MultiplicationState:
    operation_id: str
    triple: BeaverTripleShare
    output_scale: int


@dataclass(frozen=True, slots=True)
class TruncationState:
    operation_id: str
    mask: TruncationMaskShare
    output_scale: int
    offset: int


@dataclass(slots=True)
class SharedMPCStats:
    multiplication_rounds: int = 0
    opened_elements: int = 0
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0


class PreprocessingSource(Protocol):
    """Party-local source. Production implementations must use OT or VOLE."""

    session_id: str
    party: int

    def take_multiplication(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
    ) -> BeaverTripleShare: ...

    def take_truncation(
        self,
        session_id: str,
        operation_id: str,
        shape: Sequence[int],
        party: int,
        bits: int,
    ) -> TruncationMaskShare: ...


class PartyRuntime:
    """One semi-honest party's local view of additive ring-2^64 MPC."""

    def __init__(
        self,
        session_id: str,
        party: int,
        *,
        matrix_executor: Any = None,
        minimum_truncation_security_bits: int = 40,
        burn_ledger: BurnLedger | None = None,
    ) -> None:
        if not session_id or party not in (0, 1) or minimum_truncation_security_bits < 16:
            raise SharedMPCError("party runtime identity is invalid")
        self.session_id = session_id
        self.party = party
        self.stats = SharedMPCStats()
        self._burned_triples: set[str] = set()
        self._burned_masks: set[str] = set()
        self._matrix_executor = matrix_executor
        self._compiled_matrices: dict[tuple[int, int, bytes], Any] = {}
        self._minimum_truncation_security_bits = minimum_truncation_security_bits
        self._burn_ledger = burn_ledger or _DEFAULT_BURN_LEDGER
        self._truncation_profile: tuple[int, int, int] | None = None
        self._truncation_opened_elements = 0

    def _check(self, *values: SharedTensor) -> None:
        if not values:
            return
        shape = values[0].shape
        scale = values[0].scale
        for value in values:
            if value.session_id != self.session_id or value.party != self.party:
                raise SharedMPCError("shared tensor belongs to another party or session")
            if value.shape != shape:
                raise SharedMPCError("shared tensor shapes do not match")
            if value.scale != scale:
                raise SharedMPCError("shared tensor scales do not match")

    def add(self, left: SharedTensor, right: SharedTensor, tensor_id: str) -> SharedTensor:
        self._check(left, right)
        return SharedTensor(
            self.session_id,
            tensor_id,
            self.party,
            left.values + right.values,
            left.scale,
        )

    def add_public(
        self,
        value: SharedTensor,
        clear: np.ndarray | Sequence[int] | int,
        tensor_id: str,
    ) -> SharedTensor:
        self._check(value)
        result = value.values.copy()
        if self.party == 0:
            result += np.broadcast_to(_ring_array(clear), value.shape)
        return SharedTensor(self.session_id, tensor_id, self.party, result, value.scale)

    def mul_public(self, value: SharedTensor, clear: int, tensor_id: str) -> SharedTensor:
        self._check(value)
        return SharedTensor(
            self.session_id,
            tensor_id,
            self.party,
            value.values * _ring_array(clear),
            value.scale,
        )

    def mul_public_tensor(
        self,
        value: SharedTensor,
        clear: np.ndarray | Sequence[int],
        tensor_id: str,
    ) -> SharedTensor:
        self._check(value)
        clear_ring = np.broadcast_to(_ring_array(clear), value.shape)
        return SharedTensor(
            self.session_id,
            tensor_id,
            self.party,
            value.values * clear_ring,
            value.scale,
        )

    def sum_last(self, value: SharedTensor, tensor_id: str) -> SharedTensor:
        self._check(value)
        result = np.sum(value.values, axis=-1, keepdims=True, dtype=np.uint64)
        return SharedTensor(self.session_id, tensor_id, self.party, result, value.scale)

    def sum_axis(
        self,
        value: SharedTensor,
        axis: int,
        tensor_id: str,
        *,
        keepdims: bool = False,
    ) -> SharedTensor:
        self._check(value)
        result = np.sum(value.values, axis=axis, keepdims=keepdims, dtype=np.uint64)
        return SharedTensor(self.session_id, tensor_id, self.party, result, value.scale)

    def broadcast_last(
        self,
        value: SharedTensor,
        width: int,
        tensor_id: str,
    ) -> SharedTensor:
        self._check(value)
        if value.shape[-1] != 1 or width <= 0:
            raise SharedMPCError("shared tensor cannot be broadcast to requested width")
        result = np.ascontiguousarray(np.broadcast_to(value.values, (*value.shape[:-1], width)))
        return SharedTensor(self.session_id, tensor_id, self.party, result, value.scale)

    def reinterpret_scale(self, value: SharedTensor, scale: int, tensor_id: str) -> SharedTensor:
        self._check(value)
        return SharedTensor(
            self.session_id,
            tensor_id,
            self.party,
            value.values.copy(),
            scale,
        )

    def linear_public(
        self,
        value: SharedTensor,
        weight: np.ndarray,
        tensor_id: str,
        *,
        signed_bound: int,
    ) -> SharedTensor:
        self._check(value)
        raw = np.asarray(weight)
        if raw.ndim != 2 or raw.shape[1] != value.shape[-1]:
            raise SharedMPCError("public linear weight shape is invalid")
        if not np.issubdtype(raw.dtype, np.integer):
            raise SharedMPCError("public linear weights must contain integers")
        if raw.size and (int(raw.min()) < -128 or int(raw.max()) > 127):
            raise SharedMPCError("public linear weights must fit signed int8")
        matrix = np.ascontiguousarray(raw, dtype=np.int64)
        if signed_bound <= 0:
            raise SharedMPCError("public linear signed bound must be positive")
        row_bound = int(np.max(np.sum(np.abs(matrix), axis=1, dtype=np.int64))) * signed_bound
        if row_bound >= 1 << 63:
            raise SharedMPCError("public linear output exceeds signed ring range")
        if self._matrix_executor is None:
            ring_weight = matrix.view(np.uint64)
            result = np.matmul(value.values, ring_weight.T, dtype=np.uint64)
        else:
            int8_weight = np.ascontiguousarray(matrix, dtype=np.int8)
            digest = hashlib.sha256(int8_weight.tobytes()).digest()
            key = (int8_weight.shape[0], int8_weight.shape[1], digest)
            compiled = self._compiled_matrices.get(key)
            if compiled is None:
                compiled = self._matrix_executor.compile(int8_weight)
                self._compiled_matrices[key] = compiled
            result = compiled.wrap64(value.values)
        return SharedTensor(self.session_id, tensor_id, self.party, result, value.scale)

    def begin_multiply(
        self,
        left: SharedTensor,
        right: SharedTensor,
        triple: BeaverTripleShare,
        operation_id: str,
    ) -> tuple[MultiplicationState, OpeningFrame]:
        if left.shape != right.shape:
            raise SharedMPCError("multiplication tensor shapes do not match")
        self._check(left)
        self._check(right)
        if (
            triple.session_id != self.session_id
            or triple.party != self.party
            or triple.shape != left.shape
        ):
            raise SharedMPCError("Beaver triple does not match multiplication")
        if triple.triple_id in self._burned_triples:
            raise SharedMPCError("Beaver triple has already been consumed")
        self._burn_ledger.burn(self.session_id, self.party, "multiplication", triple.triple_id)
        self._burned_triples.add(triple.triple_id)
        frame = OpeningFrame(
            self.session_id,
            operation_id,
            self.party,
            left.values - triple.a,
            right.values - triple.b,
        )
        state = MultiplicationState(operation_id, triple, left.scale * right.scale)
        self.stats.uploaded_bytes += len(frame.pack())
        return state, frame

    def finish_multiply(
        self,
        state: MultiplicationState,
        local: OpeningFrame,
        peer: OpeningFrame,
        tensor_id: str,
    ) -> SharedTensor:
        if (
            local.session_id != self.session_id
            or peer.session_id != self.session_id
            or local.operation_id != state.operation_id
            or peer.operation_id != state.operation_id
            or local.party != self.party
            or peer.party == self.party
            or local.d.shape != state.triple.shape
            or peer.d.shape != state.triple.shape
        ):
            raise SharedMPCError("multiplication opening does not match state")
        d = local.d + peer.d
        e = local.e + peer.e
        result = state.triple.c + d * state.triple.b + e * state.triple.a
        if self.party == 0:
            result += d * e
        self.stats.multiplication_rounds += 1
        self.stats.opened_elements += int(d.size + e.size)
        self.stats.downloaded_bytes += len(peer.pack())
        return SharedTensor(
            self.session_id,
            tensor_id,
            self.party,
            np.ascontiguousarray(result),
            state.output_scale,
        )

    def begin_truncate(
        self,
        value: SharedTensor,
        mask: TruncationMaskShare,
        operation_id: str,
        *,
        signed_bound: int,
    ) -> tuple[TruncationState, ValueOpeningFrame]:
        self._check(value)
        if mask.session_id != self.session_id or mask.party != self.party:
            raise SharedMPCError("truncation mask belongs to another party or session")
        if mask.value.shape != value.shape:
            raise SharedMPCError("truncation mask shape does not match tensor")
        if mask.mask_id in self._burned_masks:
            raise SharedMPCError("truncation mask was already consumed")
        if signed_bound <= 0 or signed_bound > mask.maximum_magnitude:
            raise SharedMPCError("truncation input exceeds statistically hidden range")
        profile = (
            mask.session_security_bits,
            mask.per_element_security_bits,
            mask.opened_element_budget,
        )
        if mask.session_security_bits < self._minimum_truncation_security_bits:
            raise SharedMPCError("truncation mask misses session statistical-security target")
        if self._truncation_profile is None:
            self._truncation_profile = profile
        elif self._truncation_profile != profile:
            raise SharedMPCError("truncation masks use inconsistent session security accounting")
        if self._truncation_opened_elements + value.values.size > mask.opened_element_budget:
            raise SharedMPCError("truncation session opening budget exhausted")
        divisor = 1 << mask.bits
        if value.scale % divisor:
            raise SharedMPCError("tensor scale cannot be truncated by requested bits")
        self._burn_ledger.burn_truncation(
            self.session_id,
            self.party,
            mask.mask_id,
            profile,
            int(value.values.size),
        )
        self._burned_masks.add(mask.mask_id)
        self._truncation_opened_elements += int(value.values.size)
        offset = 1 << 62
        offset_share = np.uint64(offset if self.party == 0 else 0)
        opened_share = value.values + mask.value + offset_share
        return (
            TruncationState(operation_id, mask, value.scale // divisor, offset),
            ValueOpeningFrame(self.session_id, operation_id, self.party, opened_share),
        )

    def finish_truncate(
        self,
        state: TruncationState,
        local: ValueOpeningFrame,
        peer: ValueOpeningFrame,
        tensor_id: str,
    ) -> SharedTensor:
        if (
            state.operation_id != local.operation_id
            or local.operation_id != peer.operation_id
            or local.session_id != self.session_id
            or peer.session_id != self.session_id
            or local.party != self.party
            or peer.party == self.party
            or local.value.shape != state.mask.value.shape
            or peer.value.shape != state.mask.value.shape
        ):
            raise SharedMPCError("truncation opening does not match state")
        opened = local.value + peer.value
        bits = state.mask.bits
        public = (opened >> np.uint64(bits)) - np.uint64(state.offset >> bits)
        result = np.uint64(0) - state.mask.shifted
        if self.party == 0:
            result = result + public
        self.stats.opened_elements += int(opened.size)
        return SharedTensor(
            self.session_id,
            tensor_id,
            self.party,
            np.ascontiguousarray(result),
            state.output_scale,
        )


class ReferenceDealer:
    """Test-only dealer. Never use where either online party can inspect it."""

    def __init__(
        self,
        session_id: str,
        *,
        session_security_bits: int = 40,
        opened_element_budget: int | None = None,
    ) -> None:
        if not session_id:
            raise SharedMPCError("dealer session is required")
        self.session_id = session_id
        self._triple_ids: set[str] = set()
        self._mask_ids: set[str] = set()
        self.session_security_bits = session_security_bits
        self.opened_element_budget = opened_element_budget

    def split(
        self,
        clear: np.ndarray | Sequence[int],
        tensor_id: str,
        *,
        scale: int = 1,
    ) -> tuple[SharedTensor, SharedTensor]:
        value = _ring_array(clear)
        first = _random_ring(value.shape)
        second = value - first
        return (
            SharedTensor(self.session_id, tensor_id, 0, first, scale),
            SharedTensor(self.session_id, tensor_id, 1, second, scale),
        )

    def multiplication_triples(
        self,
        shape: Sequence[int],
        triple_id: str,
    ) -> tuple[BeaverTripleShare, BeaverTripleShare]:
        target = _shape(shape)
        if not triple_id or triple_id in self._triple_ids:
            raise SharedMPCError("Beaver triple identifier is empty or reused")
        self._triple_ids.add(triple_id)
        a = _random_ring(target)
        b = _random_ring(target)
        c = a * b
        a0 = _random_ring(target)
        b0 = _random_ring(target)
        c0 = _random_ring(target)
        return (
            BeaverTripleShare(self.session_id, triple_id, 0, a0, b0, c0),
            BeaverTripleShare(self.session_id, triple_id, 1, a - a0, b - b0, c - c0),
        )

    def truncation_masks(
        self,
        shape: Sequence[int],
        mask_id: str,
        *,
        bits: int,
        random_bits: int = 61,
    ) -> tuple[TruncationMaskShare, TruncationMaskShare]:
        target = _shape(shape)
        if not mask_id or mask_id in self._mask_ids:
            raise SharedMPCError("truncation mask identifier is empty or reused")
        if self.opened_element_budget is None:
            raise SharedMPCError("truncation session opening budget is required")
        opened_element_budget = self.opened_element_budget
        session_security_bits = self.session_security_bits
        if opened_element_budget <= 0 or session_security_bits < 16:
            raise SharedMPCError("truncation session security budget is invalid")
        per_element_security_bits = session_security_bits + ceil(log2(opened_element_budget))
        if not 0 < bits < random_bits <= 61 or random_bits <= per_element_security_bits:
            raise SharedMPCError("truncation mask bit range is invalid")
        self._mask_ids.add(mask_id)
        mask = _random_ring(target) & np.uint64((1 << random_bits) - 1)
        shifted = mask >> np.uint64(bits)
        maximum_magnitude = 1 << (random_bits - per_element_security_bits)
        value0 = _random_ring(target)
        shifted0 = _random_ring(target)
        return (
            TruncationMaskShare(
                self.session_id,
                mask_id,
                0,
                bits,
                random_bits,
                maximum_magnitude,
                session_security_bits,
                per_element_security_bits,
                opened_element_budget,
                value0,
                shifted0,
            ),
            TruncationMaskShare(
                self.session_id,
                mask_id,
                1,
                bits,
                random_bits,
                maximum_magnitude,
                session_security_bits,
                per_element_security_bits,
                opened_element_budget,
                mask - value0,
                shifted - shifted0,
            ),
        )


def reconstruct(first: SharedTensor, second: SharedTensor) -> np.ndarray:
    if (
        first.session_id != second.session_id
        or first.party == second.party
        or first.shape != second.shape
        or first.scale != second.scale
    ):
        raise SharedMPCError("shares cannot be reconstructed together")
    return first.values + second.values


def encode_fixed(value: np.ndarray | Sequence[float], scale: int) -> np.ndarray:
    if scale < 2 or scale & (scale - 1):
        raise SharedMPCError("fixed-point scale must be a power of two of at least two")
    scaled = np.rint(np.asarray(value, dtype=np.float64) * scale)
    if scaled.size == 0 or not np.all(np.isfinite(scaled)) or np.any(scaled >= float(1 << 63)):
        raise SharedMPCError("fixed-point value exceeds signed ring range")
    if np.any(scaled < -float(1 << 63)):
        raise SharedMPCError("fixed-point value exceeds signed ring range")
    return np.ascontiguousarray(scaled.astype(np.int64)).view(np.uint64)


def decode_fixed(value: np.ndarray, scale: int) -> np.ndarray:
    if value.dtype != np.uint64 or scale <= 0:
        raise SharedMPCError("fixed-point ring value or scale is invalid")
    return value.view(np.int64).astype(np.float64) / scale
