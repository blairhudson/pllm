from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class AuthenticationError(RuntimeError):
    """Raised when an authenticated share fails its integrity check."""


class MPCShapeError(ValueError):
    """Raised when shared values have incompatible shapes."""


@dataclass(slots=True)
class MPCStats:
    opened_values: int = 0
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0
    opening_rounds: int = 0
    multiplication_rounds: int = 0
    linear_rounds: int = 0

    @property
    def online_rounds(self) -> int:
        return self.opening_rounds


@dataclass(frozen=True, slots=True)
class PartyShare:
    value: np.ndarray
    mac: np.ndarray

    def copy(self) -> "PartyShare":
        return PartyShare(self.value.copy(), self.mac.copy())


@dataclass(frozen=True, slots=True)
class AuthenticatedValue:
    """Two additive shares with a server secret message authentication key.

    The server alone knows ``alpha``. The invariant is
    ``client_mac + server_mac = alpha * clear_value`` in the prime field.
    A client that changes a value share without a matching message
    authentication share passes an opening check with probability at most
    ``1 / (modulus - 1)`` for a single fixed forgery and a uniform nonzero key.
    """

    client: PartyShare
    server: PartyShare
    modulus: int

    @property
    def shape(self) -> tuple[int, ...]:
        return self.client.value.shape

    @property
    def size(self) -> int:
        return int(self.client.value.size)

    def tamper_client(
        self,
        *,
        value_delta: int | np.ndarray = 0,
        mac_delta: int | np.ndarray = 0,
    ) -> "AuthenticatedValue":
        p = self.modulus
        return AuthenticatedValue(
            client=PartyShare(
                (self.client.value.astype(np.int64) + value_delta) % p,
                (self.client.mac.astype(np.int64) + mac_delta) % p,
            ),
            server=self.server.copy(),
            modulus=p,
        )


@dataclass(frozen=True, slots=True)
class InputMask:
    clear_mask_for_client: np.ndarray
    shared_mask: AuthenticatedValue


@dataclass(frozen=True, slots=True)
class LinearCorrelation:
    random_input: AuthenticatedValue
    transformed_input: AuthenticatedValue
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MultiplicationTriple:
    a: AuthenticatedValue
    b: AuthenticatedValue
    c: AuthenticatedValue


@dataclass(frozen=True, slots=True)
class BitDecompositionMask:
    random_value: AuthenticatedValue
    random_bits: AuthenticatedValue
    bit_width: int


class TrustedPreprocessor:
    """Reference source for authenticated preprocessing material.

    This class is useful for correctness tests and for measuring the online
    protocol. Production use should replace it with the HE generated
    preprocessor included in this package or another reviewed correlation
    protocol.
    """

    def __init__(
        self,
        *,
        modulus: int = 65537,
        alpha: int | None = None,
        seed: int | None = None,
    ) -> None:
        if modulus <= 2:
            raise ValueError("modulus must be an odd prime larger than two")
        self.modulus = int(modulus)
        self.alpha = int(alpha if alpha is not None else secrets.randbelow(modulus - 1) + 1)
        if not 0 < self.alpha < self.modulus:
            raise ValueError("alpha must be nonzero in the field")
        self.rng = np.random.default_rng(seed if seed is not None else secrets.randbits(128))

    def random(self, shape: Sequence[int]) -> np.ndarray:
        return self.rng.integers(0, self.modulus, size=tuple(shape), dtype=np.int64)

    def share(self, clear: np.ndarray | Sequence[int] | int) -> AuthenticatedValue:
        p = self.modulus
        value = np.asarray(clear, dtype=np.int64) % p
        client_value = self.random(value.shape)
        server_value = (value - client_value) % p
        client_mac = self.random(value.shape)
        server_mac = (self.alpha * value - client_mac) % p
        return AuthenticatedValue(
            PartyShare(client_value, client_mac),
            PartyShare(server_value, server_mac),
            p,
        )

    def input_mask(self, shape: Sequence[int]) -> InputMask:
        clear = self.random(shape)
        return InputMask(clear_mask_for_client=clear, shared_mask=self.share(clear))

    def linear_correlation(
        self,
        weight: np.ndarray,
        input_shape: Sequence[int],
    ) -> LinearCorrelation:
        weight_mod = np.asarray(weight, dtype=np.int64) % self.modulus
        if weight_mod.ndim != 2:
            raise MPCShapeError("weight must be a matrix")
        shape = tuple(int(item) for item in input_shape)
        if not shape or shape[-1] != weight_mod.shape[1]:
            raise MPCShapeError("linear correlation input width mismatch")
        random_input = self.random(shape)
        transformed = _matmul_last(random_input, weight_mod, self.modulus)
        return LinearCorrelation(
            random_input=self.share(random_input),
            transformed_input=self.share(transformed),
            input_shape=shape,
            output_shape=transformed.shape,
        )

    def multiplication_triple(self, shape: Sequence[int]) -> MultiplicationTriple:
        a = self.random(shape)
        b = self.random(shape)
        c = (a * b) % self.modulus
        return MultiplicationTriple(self.share(a), self.share(b), self.share(c))

    def bit_mask(self, shape: Sequence[int], bit_width: int | None = None) -> BitDecompositionMask:
        width = int(bit_width or self.modulus.bit_length())
        if 1 << width <= self.modulus:
            raise ValueError("bit width must represent every field element")
        random_value = self.random(shape)
        bit_positions = np.arange(width, dtype=np.int64)
        bits = ((random_value[..., None] >> bit_positions) & 1).astype(np.int64)
        return BitDecompositionMask(
            random_value=self.share(random_value),
            random_bits=self.share(bits),
            bit_width=width,
        )


class AuthenticatedMPC:
    """Arithmetic simulator with both parties' shares in one process.

    This is not a distributed malicious-client-secure implementation.

    Intermediate values remain split between the client and server. The server
    is assumed to follow the protocol. Only values that the program explicitly
    opens become visible.
    """

    def __init__(self, preprocessor: TrustedPreprocessor) -> None:
        self.preprocessor = preprocessor
        self.modulus = preprocessor.modulus
        self._alpha = preprocessor.alpha
        self.stats = MPCStats()

    def _check_compatible(self, *values: AuthenticatedValue) -> None:
        if not values:
            return
        shape = values[0].shape
        for item in values:
            if item.modulus != self.modulus:
                raise MPCShapeError("field mismatch")
            if item.shape != shape:
                raise MPCShapeError("share shape mismatch")

    def _open_without_stats(self, value: AuthenticatedValue) -> np.ndarray:
        self._check_compatible(value)
        p = self.modulus
        clear = (value.client.value.astype(np.int64) + value.server.value.astype(np.int64)) % p
        mac = (value.client.mac.astype(np.int64) + value.server.mac.astype(np.int64)) % p
        expected = (self._alpha * clear) % p
        if not np.array_equal(mac, expected):
            raise AuthenticationError("authenticated share check failed")
        return clear

    def open_many(
        self,
        values: Sequence[AuthenticatedValue],
        *,
        to_client: bool = True,
        count_round: bool = True,
    ) -> list[np.ndarray]:
        clear_values = [self._open_without_stats(value) for value in values]
        count = sum(int(value.size) for value in values)
        self.stats.opened_values += count
        self.stats.uploaded_bytes += count * 16
        if to_client:
            self.stats.downloaded_bytes += count * 8
        if values and count_round:
            self.stats.opening_rounds += 1
        return clear_values

    def open(self, value: AuthenticatedValue, *, to_client: bool = True) -> np.ndarray:
        return self.open_many([value], to_client=to_client)[0]

    def input(self, clear: np.ndarray, mask: InputMask) -> AuthenticatedValue:
        value = np.asarray(clear, dtype=np.int64) % self.modulus
        if value.shape != mask.clear_mask_for_client.shape:
            raise MPCShapeError("input mask shape mismatch")
        epsilon = (value - mask.clear_mask_for_client) % self.modulus
        shared = mask.shared_mask
        client = shared.client.copy()
        server = PartyShare(
            (shared.server.value + epsilon) % self.modulus,
            (shared.server.mac + self._alpha * epsilon) % self.modulus,
        )
        self.stats.uploaded_bytes += int(epsilon.size) * 8
        return AuthenticatedValue(client, server, self.modulus)

    def public_value(self, clear: int | np.ndarray, shape: Sequence[int] | None = None) -> AuthenticatedValue:
        value = np.array(clear, dtype=np.int64, copy=True)
        if shape is not None:
            value = np.broadcast_to(value, tuple(shape)).copy()
        value %= self.modulus
        zeros = np.zeros_like(value, dtype=np.int64)
        return AuthenticatedValue(
            PartyShare(zeros.copy(), zeros.copy()),
            PartyShare(value, (self._alpha * value) % self.modulus),
            self.modulus,
        )

    def add(self, left: AuthenticatedValue, right: AuthenticatedValue) -> AuthenticatedValue:
        self._check_compatible(left, right)
        p = self.modulus
        return AuthenticatedValue(
            PartyShare(
                (left.client.value + right.client.value) % p,
                (left.client.mac + right.client.mac) % p,
            ),
            PartyShare(
                (left.server.value + right.server.value) % p,
                (left.server.mac + right.server.mac) % p,
            ),
            p,
        )

    def sub(self, left: AuthenticatedValue, right: AuthenticatedValue) -> AuthenticatedValue:
        self._check_compatible(left, right)
        p = self.modulus
        return AuthenticatedValue(
            PartyShare(
                (left.client.value - right.client.value) % p,
                (left.client.mac - right.client.mac) % p,
            ),
            PartyShare(
                (left.server.value - right.server.value) % p,
                (left.server.mac - right.server.mac) % p,
            ),
            p,
        )

    def add_public(self, value: AuthenticatedValue, clear: int | np.ndarray) -> AuthenticatedValue:
        p = self.modulus
        public = np.asarray(clear, dtype=np.int64) % p
        return AuthenticatedValue(
            value.client.copy(),
            PartyShare(
                (value.server.value + public) % p,
                (value.server.mac + self._alpha * public) % p,
            ),
            p,
        )

    def mul_public(self, value: AuthenticatedValue, clear: int | np.ndarray) -> AuthenticatedValue:
        p = self.modulus
        public = np.asarray(clear, dtype=np.int64) % p
        return AuthenticatedValue(
            PartyShare(
                (value.client.value * public) % p,
                (value.client.mac * public) % p,
            ),
            PartyShare(
                (value.server.value * public) % p,
                (value.server.mac * public) % p,
            ),
            p,
        )

    def reshape(self, value: AuthenticatedValue, shape: Sequence[int]) -> AuthenticatedValue:
        target = tuple(int(item) for item in shape)
        return AuthenticatedValue(
            PartyShare(value.client.value.reshape(target), value.client.mac.reshape(target)),
            PartyShare(value.server.value.reshape(target), value.server.mac.reshape(target)),
            value.modulus,
        )

    def transpose(self, value: AuthenticatedValue, axes: Sequence[int]) -> AuthenticatedValue:
        target = tuple(int(item) for item in axes)
        return AuthenticatedValue(
            PartyShare(np.transpose(value.client.value, target), np.transpose(value.client.mac, target)),
            PartyShare(np.transpose(value.server.value, target), np.transpose(value.server.mac, target)),
            value.modulus,
        )

    def take(self, value: AuthenticatedValue, key) -> AuthenticatedValue:
        return AuthenticatedValue(
            PartyShare(value.client.value[key], value.client.mac[key]),
            PartyShare(value.server.value[key], value.server.mac[key]),
            value.modulus,
        )

    def concatenate(self, values: Sequence[AuthenticatedValue], axis: int = 0) -> AuthenticatedValue:
        if not values:
            raise ValueError("at least one value is required")
        for value in values:
            if value.modulus != self.modulus:
                raise MPCShapeError("field mismatch")
        return AuthenticatedValue(
            PartyShare(
                np.concatenate([value.client.value for value in values], axis=axis),
                np.concatenate([value.client.mac for value in values], axis=axis),
            ),
            PartyShare(
                np.concatenate([value.server.value for value in values], axis=axis),
                np.concatenate([value.server.mac for value in values], axis=axis),
            ),
            self.modulus,
        )

    def broadcast_to(self, value: AuthenticatedValue, shape: Sequence[int]) -> AuthenticatedValue:
        target = tuple(int(item) for item in shape)
        return AuthenticatedValue(
            PartyShare(
                np.broadcast_to(value.client.value, target).copy(),
                np.broadcast_to(value.client.mac, target).copy(),
            ),
            PartyShare(
                np.broadcast_to(value.server.value, target).copy(),
                np.broadcast_to(value.server.mac, target).copy(),
            ),
            value.modulus,
        )

    def sum(self, value: AuthenticatedValue, axis=None, keepdims: bool = False) -> AuthenticatedValue:
        p = self.modulus
        return AuthenticatedValue(
            PartyShare(
                np.sum(value.client.value, axis=axis, keepdims=keepdims, dtype=np.int64) % p,
                np.sum(value.client.mac, axis=axis, keepdims=keepdims, dtype=np.int64) % p,
            ),
            PartyShare(
                np.sum(value.server.value, axis=axis, keepdims=keepdims, dtype=np.int64) % p,
                np.sum(value.server.mac, axis=axis, keepdims=keepdims, dtype=np.int64) % p,
            ),
            p,
        )

    def linear_public(self, value: AuthenticatedValue, weight: np.ndarray) -> AuthenticatedValue:
        weight_mod = np.asarray(weight, dtype=np.int64) % self.modulus
        if weight_mod.ndim != 2 or weight_mod.shape[1] != value.shape[-1]:
            raise MPCShapeError("public linear weight width mismatch")
        return AuthenticatedValue(
            PartyShare(
                _matmul_last(value.client.value, weight_mod, self.modulus),
                _matmul_last(value.client.mac, weight_mod, self.modulus),
            ),
            PartyShare(
                _matmul_last(value.server.value, weight_mod, self.modulus),
                _matmul_last(value.server.mac, weight_mod, self.modulus),
            ),
            self.modulus,
        )

    def linear(
        self,
        value: AuthenticatedValue,
        weight: np.ndarray,
        correlation: LinearCorrelation,
    ) -> AuthenticatedValue:
        return self.linear_many([value], [weight], [correlation])[0]

    def linear_many(
        self,
        values: Sequence[AuthenticatedValue],
        weights: Sequence[np.ndarray],
        correlations: Sequence[LinearCorrelation],
    ) -> list[AuthenticatedValue]:
        if not (len(values) == len(weights) == len(correlations)):
            raise ValueError("linear input lists must have the same length")
        differences: list[AuthenticatedValue] = []
        weight_mods: list[np.ndarray] = []
        for value, weight, correlation in zip(values, weights, correlations, strict=True):
            if value.shape != correlation.input_shape:
                raise MPCShapeError("linear input and correlation shape mismatch")
            weight_mod = np.asarray(weight, dtype=np.int64) % self.modulus
            if weight_mod.ndim != 2 or weight_mod.shape[1] != value.shape[-1]:
                raise MPCShapeError("linear weight width mismatch")
            differences.append(self.sub(value, correlation.random_input))
            weight_mods.append(weight_mod)
        opened = self.open_many(differences, to_client=False)
        outputs: list[AuthenticatedValue] = []
        for clear_difference, weight_mod, correlation in zip(
            opened, weight_mods, correlations, strict=True
        ):
            transformed_difference = _matmul_last(clear_difference, weight_mod, self.modulus)
            base = correlation.transformed_input
            outputs.append(
                AuthenticatedValue(
                    base.client.copy(),
                    PartyShare(
                        (base.server.value + transformed_difference) % self.modulus,
                        (base.server.mac + self._alpha * transformed_difference) % self.modulus,
                    ),
                    self.modulus,
                )
            )
        if values:
            self.stats.linear_rounds += 1
        return outputs

    def multiply(
        self,
        left: AuthenticatedValue,
        right: AuthenticatedValue,
        triple: MultiplicationTriple,
    ) -> AuthenticatedValue:
        return self.multiply_many([left], [right], [triple])[0]

    def multiply_many(
        self,
        left_values: Sequence[AuthenticatedValue],
        right_values: Sequence[AuthenticatedValue],
        triples: Sequence[MultiplicationTriple],
    ) -> list[AuthenticatedValue]:
        if not (len(left_values) == len(right_values) == len(triples)):
            raise ValueError("multiplication input lists must have the same length")
        differences: list[AuthenticatedValue] = []
        for left, right, triple in zip(left_values, right_values, triples, strict=True):
            self._check_compatible(left, right, triple.a, triple.b, triple.c)
            differences.append(self.sub(left, triple.a))
            differences.append(self.sub(right, triple.b))
        opened = self.open_many(differences, to_client=True)
        outputs: list[AuthenticatedValue] = []
        p = self.modulus
        for index, triple in enumerate(triples):
            d = opened[index * 2]
            e = opened[index * 2 + 1]
            client_value = (
                triple.c.client.value
                + d * triple.b.client.value
                + e * triple.a.client.value
            ) % p
            server_value = (
                triple.c.server.value
                + d * triple.b.server.value
                + e * triple.a.server.value
                + d * e
            ) % p
            client_mac = (
                triple.c.client.mac
                + d * triple.b.client.mac
                + e * triple.a.client.mac
            ) % p
            server_mac = (
                triple.c.server.mac
                + d * triple.b.server.mac
                + e * triple.a.server.mac
                + self._alpha * d * e
            ) % p
            outputs.append(
                AuthenticatedValue(
                    PartyShare(client_value, client_mac),
                    PartyShare(server_value, server_mac),
                    p,
                )
            )
        if left_values:
            self.stats.multiplication_rounds += 1
        return outputs

    def square(self, value: AuthenticatedValue, triple: MultiplicationTriple) -> AuthenticatedValue:
        return self.multiply(value, value, triple)

    def polynomial(
        self,
        value: AuthenticatedValue,
        coefficients: Sequence[int],
        triples: Iterable[MultiplicationTriple],
    ) -> AuthenticatedValue:
        """Evaluate an integer polynomial with Horner's rule."""
        coeffs = [int(item) % self.modulus for item in coefficients]
        if not coeffs:
            raise ValueError("polynomial needs at least one coefficient")
        if len(coeffs) == 1:
            return self.add_public(self.mul_public(value, 0), coeffs[0])
        triple_iter = iter(triples)
        result = self.add_public(self.mul_public(value, 0), coeffs[-1])
        for coefficient in reversed(coeffs[:-1]):
            try:
                triple = next(triple_iter)
            except StopIteration as exc:
                raise ValueError("insufficient multiplication triples") from exc
            result = self.multiply(result, value, triple)
            result = self.add_public(result, coefficient)
        return result

    def centered(self, value: np.ndarray) -> np.ndarray:
        raw = np.asarray(value, dtype=np.int64) % self.modulus
        return np.where(raw > self.modulus // 2, raw - self.modulus, raw)


def _matmul_last(value: np.ndarray, weight: np.ndarray, modulus: int) -> np.ndarray:
    x = np.asarray(value, dtype=np.int64)
    w = np.asarray(weight, dtype=np.int64)
    if x.shape[-1] != w.shape[1]:
        raise MPCShapeError("matrix multiplication width mismatch")
    return np.matmul(x, w.T) % int(modulus)


def demo() -> dict[str, object]:
    """Run a small authenticated arithmetic check for the command line."""
    preprocessor = TrustedPreprocessor(seed=1)
    runtime = AuthenticatedMPC(preprocessor)
    x = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    weight = np.asarray([[1, -1, 2, 0], [0, 3, -2, 1]], dtype=np.int64)
    shared = runtime.input(x, preprocessor.input_mask(x.shape))
    result = runtime.linear(shared, weight, preprocessor.linear_correlation(weight, x.shape))
    opened = runtime.centered(runtime.open(result))
    tamper_detected = False
    try:
        runtime.open(result.tamper_client(value_delta=1))
    except AuthenticationError:
        tamper_detected = True
    return {
        "exact": bool(np.array_equal(opened, x @ weight.T)),
        "output": opened.tolist(),
        "tamper_detected": tamper_detected,
        "security": "authenticated arithmetic with an honest server",
    }
