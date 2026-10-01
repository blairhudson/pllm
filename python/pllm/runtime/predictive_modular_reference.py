"""Bounded one-use predictive modular linear reference; never a live topology.

The issuer sees public W and a proposed public integer predictor A. Client owns
fresh input/output masks and A; provider owns W and W*r-s modulo q. Neither
client receives the provider correction nor provider receives client masks.
Exactness requires a public bound for *all* admitted activation inputs; a small
observed error on sampled prompts is never an exactness certificate.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

import numpy as np


class PredictiveModularError(ValueError):
    """Malformed one-use material or uncertified residue reconstruction."""


_DOMAIN = b"pllm.predictive_modular_reference.v1\0"
_MAX_ELEMENTS = 131_072


def _bounded_matrix(value: np.ndarray, *, shape: tuple[int, int] | None = None) -> np.ndarray:
    matrix = np.asarray(value)
    if (
        matrix.dtype != np.int8
        or matrix.ndim != 2
        or any(dim <= 0 for dim in matrix.shape)
        or matrix.size > _MAX_ELEMENTS
        or (shape is not None and matrix.shape != shape)
    ):
        raise PredictiveModularError(
            "integer weight/predictor matrix exceeds the reference contract"
        )
    snapshot = np.array(matrix, copy=True, order="C")
    snapshot.setflags(write=False)
    return snapshot


def certified_error_bound(
    weight: np.ndarray, predictor: np.ndarray, *, max_abs_input: int = 127
) -> np.ndarray:
    """Per-output worst-case |(W-A)x| for independent bounded input coordinates."""
    w = _bounded_matrix(weight)
    a = _bounded_matrix(predictor, shape=w.shape)
    if type(max_abs_input) is not int or not 1 <= max_abs_input <= 127:
        raise PredictiveModularError("input domain bound must be public signed i8")
    return max_abs_input * np.sum(
        np.abs(w.astype(np.int16) - a.astype(np.int16)), axis=1, dtype=np.int64
    )


def minimum_certified_ring_bits(max_error: int) -> int:
    """Smallest power-of-two ring with a strictly unique integer lift."""
    if type(max_error) is not int or max_error < 0 or max_error > (1 << 31) - 1:
        raise PredictiveModularError("public error bound exceeds the ring contract")
    return max(2, (2 * max_error).bit_length())


def sparse_predictor_options(weight: np.ndarray) -> dict[int, dict[str, object]]:
    """Offline public-weight analysis; larger than the one-use issuer capacity."""
    w = np.asarray(weight)
    if (
        w.dtype != np.int8
        or w.ndim != 2
        or any(dim <= 0 for dim in w.shape)
        or w.size > 16_777_216
        or w.shape[1] > 65535
    ):
        raise PredictiveModularError("public stage exceeds the bounded sparse analysis")
    row_count, columns = w.shape
    score = np.sum(np.abs(w.astype(np.int16)), axis=0, dtype=np.int64)
    ranked = np.lexsort((np.arange(columns), -score))
    options: dict[int, dict[str, object]] = {}
    for fraction in (10, 50, 90):
        selected = np.sort(ranked[: columns * fraction // 100])
        omitted = np.ones(columns, dtype=bool)
        omitted[selected] = False
        # A=W on selected public columns, zero elsewhere. This is the exact
        # certificate without making three dense copies of a real checkpoint.
        public_error = 127 * np.sum(np.abs(w[:, omitted].astype(np.int16)), axis=1, dtype=np.int64)
        maximum = int(np.max(public_error))
        options[fraction] = {
            "selected": selected,
            "public_worst_error": maximum,
            "certified_ring_bits": minimum_certified_ring_bits(maximum),
            "public_i8_predictor_bytes": row_count * len(selected) + 2 * len(selected),
            "observed_worst_error": 0,
            "client_integer_macs": 0,
        }
    return options


def _uniform(modulus: int, shape: tuple[int, int]) -> np.ndarray:
    return np.fromiter(
        (secrets.randbelow(modulus) for _ in range(shape[0] * shape[1])),
        dtype=np.uint64,
        count=shape[0] * shape[1],
    ).reshape(shape)


@dataclass(frozen=True, slots=True)
class ModularRequest:
    ticket: str
    binding: str
    masked_input: np.ndarray


@dataclass(frozen=True, slots=True)
class ModularResponse:
    ticket: str
    binding: str
    masked_output: np.ndarray


@dataclass(slots=True)
class ClientModularMaterial:
    """Trusted-client-only material; one request and at most one response."""

    ticket: str
    binding: str
    modulus: int
    max_abs_input: int
    output_bound: int
    predictor: np.ndarray = field(repr=False)
    input_mask: np.ndarray = field(repr=False)
    output_mask: np.ndarray = field(repr=False)
    _state: str = field(default="unused", init=False, repr=False)
    _input: np.ndarray | None = field(default=None, init=False, repr=False)

    def abort(self) -> None:
        self._state = "burned"
        self._input = None
        self.input_mask.fill(0)
        self.output_mask.fill(0)

    def mask_input(self, value: np.ndarray) -> ModularRequest:
        if self._state != "unused":
            raise PredictiveModularError("modular client ticket has already been consumed")
        self._state = "burned"  # Even a malformed attempt burns its material.
        x = np.asarray(value)
        if (
            x.dtype != np.int8
            or x.shape != self.input_mask.shape
            or np.any(np.abs(x.astype(np.int16)) > self.max_abs_input)
        ):
            self.abort()
            raise PredictiveModularError("client input exceeds its fixed public domain")
        self._input = np.array(x, dtype=np.int64, order="C", copy=True)
        masked = np.asarray(
            (self._input - self.input_mask.astype(np.int64)) % self.modulus, dtype=np.uint32
        )
        self._state = "waiting"
        return ModularRequest(self.ticket, self.binding, masked)

    def recover(self, response: ModularResponse) -> np.ndarray:
        if self._state != "waiting" or self._input is None:
            raise PredictiveModularError("modular client ticket is not awaiting a response")
        self._state = "burned"  # No retry if the provider response is invalid.
        try:
            if (
                type(response) is not ModularResponse
                or response.ticket != self.ticket
                or response.binding != self.binding
                or not isinstance(response.masked_output, np.ndarray)
                or response.masked_output.dtype != np.uint32
                or response.masked_output.shape != self.output_mask.shape
                or np.any(response.masked_output.astype(np.uint64) >= self.modulus)
            ):
                raise PredictiveModularError("masked response differs from the bound ticket")
            residue = (
                response.masked_output.astype(np.int64) + self.output_mask.astype(np.int64)
            ) % self.modulus
            prediction = self._input @ self.predictor.astype(np.int64).T
            half = self.modulus // 2
            delta = (residue - prediction + half) % self.modulus - half
            output = prediction + delta
            if np.any(np.abs(output) > self.output_bound):
                raise PredictiveModularError("recovered output exceeds the signed stage bound")
            return output
        finally:
            self.abort()


@dataclass(slots=True)
class ProviderModularMaterial:
    """Separate provider view: only W and correction, never client masks."""

    ticket: str
    binding: str
    modulus: int
    weight: np.ndarray = field(repr=False)
    correction: np.ndarray = field(repr=False)
    _used: bool = field(default=False, init=False, repr=False)

    def evaluate(self, request: ModularRequest) -> ModularResponse:
        if self._used:
            raise PredictiveModularError("provider correction has already been consumed")
        if type(request) is not ModularRequest or request.ticket != self.ticket:
            raise PredictiveModularError("provider request has no matching ticket")
        self._used = True
        try:
            if (
                request.binding != self.binding
                or not isinstance(request.masked_input, np.ndarray)
                or request.masked_input.dtype != np.uint32
                or request.masked_input.shape != (self.correction.shape[0], self.weight.shape[1])
                or np.any(request.masked_input.astype(np.uint64) >= self.modulus)
            ):
                raise PredictiveModularError("provider request differs from the committed stage")
            output = (
                request.masked_input.astype(np.int64) @ self.weight.astype(np.int64).T
                + self.correction.astype(np.int64)
            ) % self.modulus
            return ModularResponse(self.ticket, self.binding, np.asarray(output, dtype=np.uint32))
        finally:
            self.correction.fill(0)


def issue_certified_modular_stage(
    weight: np.ndarray,
    predictor: np.ndarray,
    *,
    modulus: int,
    rows: int,
    stage_id: str,
    plan_digest: str,
    max_abs_input: int = 127,
) -> tuple[ClientModularMaterial, ProviderModularMaterial]:
    """Issue one reference row only after proving all inputs fit the lift bound."""
    w = _bounded_matrix(weight)
    a = _bounded_matrix(predictor, shape=w.shape)
    if (
        type(modulus) is not int
        or modulus < 4
        or modulus > 1 << 32
        or modulus & (modulus - 1)
        or type(rows) is not int
        or not 1 <= rows <= 64
        or max(rows * w.shape[0], rows * w.shape[1]) > _MAX_ELEMENTS
        or type(stage_id) is not str
        or not 0 < len(stage_id) <= 128
        or not stage_id.isascii()
        or type(plan_digest) is not str
        or len(plan_digest) != 64
        or any(c not in "0123456789abcdef" for c in plan_digest)
    ):
        raise PredictiveModularError("stage identity or modulus exceeds the reference contract")
    bound = certified_error_bound(w, a, max_abs_input=max_abs_input)
    if np.any(2 * bound >= modulus):
        raise PredictiveModularError("predictor has no whole-input exact lift certificate")
    identity = hashlib.sha256(
        _DOMAIN
        + plan_digest.encode("ascii")
        + stage_id.encode("ascii")
        + modulus.to_bytes(8, "little")
        + bytes([max_abs_input])
        + np.asarray(w.shape, dtype="<u4").tobytes()
        + w.tobytes()
        + a.tobytes()
    ).hexdigest()
    ticket = secrets.token_hex(16)
    r = _uniform(modulus, (rows, w.shape[1]))
    s = _uniform(modulus, (rows, w.shape[0]))
    correction = np.asarray(
        (r.astype(np.int64) @ w.astype(np.int64).T - s.astype(np.int64)) % modulus,
        dtype=np.uint32,
    )
    client = ClientModularMaterial(
        ticket,
        identity,
        modulus,
        max_abs_input,
        max_abs_input * int(np.max(np.sum(np.abs(w.astype(np.int16)), axis=1))),
        a,
        r,
        s,
    )
    provider = ProviderModularMaterial(ticket, identity, modulus, w, correction)
    return client, provider


__all__ = [
    "ClientModularMaterial",
    "ModularRequest",
    "ModularResponse",
    "PredictiveModularError",
    "ProviderModularMaterial",
    "certified_error_bound",
    "issue_certified_modular_stage",
    "minimum_certified_ring_bits",
    "sparse_predictor_options",
]
