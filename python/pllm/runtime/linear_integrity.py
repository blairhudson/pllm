from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .secure_random import uniform_residues


class LinearIntegrityError(RuntimeError):
    """Raised when a server linear result fails its hidden check."""


@dataclass(frozen=True, slots=True)
class LinearCheckKey:
    challenge: np.ndarray
    input_projection: np.ndarray
    modulus: int
    model_fingerprint: str = ""
    stage_id: str = ""

    @property
    def check_count(self) -> int:
        return int(self.challenge.shape[0])


@dataclass(frozen=True, slots=True)
class LinearCheckResult:
    valid: bool
    expected: np.ndarray
    observed: np.ndarray


def create_linear_check_key(
    weight: np.ndarray,
    *,
    modulus: int,
    checks: int = 2,
    seed: int | None = None,
    model_fingerprint: str = "",
    stage_id: str = "",
) -> LinearCheckKey:
    """Create a client secret check key for a server matrix.

    A model publisher or another trusted build step creates this key. The model
    server must not receive the random challenge vectors. Each check reveals one
    random row combination of the matrix to the client.
    """
    matrix = np.asarray(weight, dtype=np.int64) % int(modulus)
    if matrix.ndim != 2:
        raise ValueError("weight must be a matrix")
    if checks < 1:
        raise ValueError("checks must be positive")
    if seed is not None:
        # Seeded challenges exist solely for reproducible test cases.
        rng = np.random.default_rng(seed)
        challenge = rng.integers(0, modulus, size=(checks, matrix.shape[0]), dtype=np.int64)
    else:
        challenge = uniform_residues(modulus, (checks, matrix.shape[0])).astype(np.int64)
    projection = challenge @ matrix % int(modulus)
    return LinearCheckKey(
        challenge=challenge,
        input_projection=projection,
        modulus=int(modulus),
        model_fingerprint=model_fingerprint,
        stage_id=stage_id,
    )


def check_linear_result(
    input_value: np.ndarray,
    output_value: np.ndarray,
    key: LinearCheckKey,
    *,
    raise_on_failure: bool = True,
) -> LinearCheckResult:
    x = np.asarray(input_value, dtype=np.int64) % key.modulus
    y = np.asarray(output_value, dtype=np.int64) % key.modulus
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear check expects two dimensional batches")
    if x.shape[0] != y.shape[0]:
        raise ValueError("input and output batch sizes differ")
    if x.shape[1] != key.input_projection.shape[1]:
        raise ValueError("input width does not match check key")
    if y.shape[1] != key.challenge.shape[1]:
        raise ValueError("output width does not match check key")
    expected = x @ key.input_projection.T % key.modulus
    observed = y @ key.challenge.T % key.modulus
    valid = bool(np.array_equal(expected, observed))
    if not valid and raise_on_failure:
        raise LinearIntegrityError("server linear result failed its hidden check")
    return LinearCheckResult(valid=valid, expected=expected, observed=observed)
