from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .authenticated_mpc import (
    AuthenticatedValue,
    BitDecompositionMask,
    InputMask,
    LinearCorrelation,
    MPCShapeError,
    MultiplicationTriple,
    PartyShare,
)


class HEPreprocessingUnavailable(RuntimeError):
    """Raised when the optional HE runtime cannot be loaded."""


@dataclass(slots=True)
class HEPreprocessingStats:
    context_generation_ms: float = 0.0
    client_encrypt_ms: float = 0.0
    server_compute_ms: float = 0.0
    client_decrypt_ms: float = 0.0
    uploaded_bytes: int = 0
    downloaded_bytes: int = 0
    public_context_bytes: int = 0
    multiplication_triples: int = 0
    linear_correlations: int = 0
    authenticated_values: int = 0

    @property
    def total_ms(self) -> float:
        return (
            self.context_generation_ms
            + self.client_encrypt_ms
            + self.server_compute_ms
            + self.client_decrypt_ms
        )


class HEAuthenticatedPreprocessor:
    """Create authenticated arithmetic material with BFV.

    The client owns the BFV secret key. The server owns the message
    authentication key and receives only the public BFV context. The reference
    implementation protects against a client that changes its local shares. It
    assumes the model server follows the preprocessing protocol.

    The implementation uses packed elementwise BFV for multiplication triples.
    Small server private linear correlations use one encrypted dot product per
    output row. That path is intended for validation, not large model service.
    """

    def __init__(
        self,
        *,
        modulus: int = 65537,
        poly_modulus_degree: int = 4096,
        alpha: int | None = None,
        seed: int | None = None,
        threads: int = 1,
        tenseal_path: str | Path | None = None,
        enable_linear_correlations: bool = True,
    ) -> None:
        if threads < 1:
            raise ValueError("threads must be at least one")
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        self.ts = self._load_tenseal(tenseal_path)
        self.modulus = int(modulus)
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.alpha = int(alpha if alpha is not None else secrets.randbelow(modulus - 1) + 1)
        self.rng = np.random.default_rng(seed if seed is not None else secrets.randbits(128))
        self.stats = HEPreprocessingStats()
        self.enable_linear_correlations = bool(enable_linear_correlations)

        started = time.perf_counter_ns()
        self.client_context = self.ts.context(
            self.ts.SCHEME_TYPE.BFV,
            self.poly_modulus_degree,
            self.modulus,
            n_threads=threads,
        )
        if self.enable_linear_correlations:
            self.client_context.generate_galois_keys()
        public_bytes = self.client_context.serialize(save_secret_key=False)
        self.server_context = self.ts.context_from(public_bytes, n_threads=threads)
        self.stats.context_generation_ms = (time.perf_counter_ns() - started) / 1e6
        self.stats.public_context_bytes = len(public_bytes)

    @staticmethod
    def _load_tenseal(path: str | Path | None):
        if path is not None:
            import sys

            sys.path.insert(0, str(path))
        try:
            import tenseal as ts
        except ImportError as exc:
            raise HEPreprocessingUnavailable(
                "Install the HE extra or provide a directory containing TenSEAL"
            ) from exc
        return ts

    def random(self, shape: Sequence[int]) -> np.ndarray:
        return self.rng.integers(0, self.modulus, size=tuple(shape), dtype=np.int64)

    def _check_size(self, size: int) -> None:
        if size > self.poly_modulus_degree:
            raise MPCShapeError(
                f"reference BFV packing supports at most {self.poly_modulus_degree} values, got {size}"
            )

    def _encrypt(self, value: np.ndarray):
        flat = np.asarray(value, dtype=np.int64).reshape(-1) % self.modulus
        self._check_size(int(flat.size))
        started = time.perf_counter_ns()
        encrypted = self.ts.bfv_vector(self.client_context, flat.tolist())
        payload = encrypted.serialize()
        self.stats.client_encrypt_ms += (time.perf_counter_ns() - started) / 1e6
        self.stats.uploaded_bytes += len(payload)
        return payload

    def _server_load(self, payload: bytes):
        return self.ts.bfv_vector_from(self.server_context, payload)

    def _decrypt(self, encrypted, count: int, shape: tuple[int, ...]) -> np.ndarray:
        payload = encrypted.serialize()
        self.stats.downloaded_bytes += len(payload)
        started = time.perf_counter_ns()
        client_value = self.ts.bfv_vector_from(self.client_context, payload)
        clear = np.asarray(client_value.decrypt()[:count], dtype=np.int64) % self.modulus
        self.stats.client_decrypt_ms += (time.perf_counter_ns() - started) / 1e6
        return clear.reshape(shape)

    def _authenticate_parts(
        self,
        client_value: np.ndarray,
        server_value: np.ndarray,
        *,
        encrypted_client=None,
    ) -> AuthenticatedValue:
        client = np.asarray(client_value, dtype=np.int64) % self.modulus
        server = np.asarray(server_value, dtype=np.int64) % self.modulus
        if client.shape != server.shape:
            raise MPCShapeError("client and server shares must have the same shape")
        flat_count = int(client.size)
        if encrypted_client is None:
            encrypted_client = self._server_load(self._encrypt(client))
        beta = self.random(client.shape)
        started = time.perf_counter_ns()
        encrypted_mac = encrypted_client * self.alpha - beta.reshape(-1).tolist()
        self.stats.server_compute_ms += (time.perf_counter_ns() - started) / 1e6
        client_mac = self._decrypt(encrypted_mac, flat_count, client.shape)
        server_mac = (self.alpha * server + beta) % self.modulus
        self.stats.authenticated_values += flat_count
        return AuthenticatedValue(
            PartyShare(client.copy(), client_mac),
            PartyShare(server.copy(), server_mac),
            self.modulus,
        )

    def share(self, clear: np.ndarray | Sequence[int] | int) -> AuthenticatedValue:
        value = np.asarray(clear, dtype=np.int64) % self.modulus
        client_value = self.random(value.shape)
        server_value = (value - client_value) % self.modulus
        return self._authenticate_parts(client_value, server_value)

    def input_mask(self, shape: Sequence[int]) -> InputMask:
        clear_mask = self.random(shape)
        return InputMask(clear_mask_for_client=clear_mask, shared_mask=self.share(clear_mask))

    def bit_mask(self, shape: Sequence[int], bit_width: int | None = None) -> BitDecompositionMask:
        width = int(bit_width or self.modulus.bit_length())
        if 1 << width <= self.modulus:
            raise ValueError("bit width must represent every field element")
        random_value = self.random(shape)
        positions = np.arange(width, dtype=np.int64)
        bits = ((random_value[..., None] >> positions) & 1).astype(np.int64)
        return BitDecompositionMask(
            random_value=self.share(random_value),
            random_bits=self.share(bits),
            bit_width=width,
        )

    def multiplication_triple(self, shape: Sequence[int]) -> MultiplicationTriple:
        target_shape = tuple(int(item) for item in shape)
        count = int(np.prod(target_shape, dtype=np.int64))
        self._check_size(count)
        a_client = self.random(target_shape)
        b_client = self.random(target_shape)
        a_server = self.random(target_shape)
        b_server = self.random(target_shape)
        c_server = self.random(target_shape)

        encrypted_a_payload = self._encrypt(a_client)
        encrypted_b_payload = self._encrypt(b_client)
        encrypted_a = self._server_load(encrypted_a_payload)
        encrypted_b = self._server_load(encrypted_b_payload)

        started = time.perf_counter_ns()
        encrypted_c_client = (
            encrypted_a * encrypted_b
            + encrypted_a * b_server.reshape(-1).tolist()
            + encrypted_b * a_server.reshape(-1).tolist()
            + ((a_server * b_server - c_server) % self.modulus).reshape(-1).tolist()
        )
        self.stats.server_compute_ms += (time.perf_counter_ns() - started) / 1e6
        c_client = self._decrypt(encrypted_c_client, count, target_shape)

        a = self._authenticate_parts(a_client, a_server, encrypted_client=encrypted_a)
        b = self._authenticate_parts(b_client, b_server, encrypted_client=encrypted_b)
        c = self._authenticate_parts(c_client, c_server, encrypted_client=encrypted_c_client)
        self.stats.multiplication_triples += count
        return MultiplicationTriple(a, b, c)

    def _encrypted_dot_outputs(
        self,
        encrypted_input,
        weight: np.ndarray,
        server_input_share: np.ndarray,
        server_output_share: np.ndarray,
    ) -> list:
        weight_mod = np.asarray(weight, dtype=np.int64) % self.modulus
        total_server_term = (server_input_share @ weight_mod.T - server_output_share) % self.modulus
        outputs = []
        for row_index, row in enumerate(weight_mod):
            result = encrypted_input.dot(row.tolist()) + int(total_server_term[row_index])
            outputs.append(result)
        return outputs

    def linear_correlation(
        self,
        weight: np.ndarray,
        input_shape: Sequence[int],
    ) -> LinearCorrelation:
        if not self.enable_linear_correlations:
            raise HEPreprocessingUnavailable("linear correlations require a context with rotation keys")
        weight_mod = np.asarray(weight, dtype=np.int64) % self.modulus
        if weight_mod.ndim != 2:
            raise MPCShapeError("weight must be a matrix")
        shape = tuple(int(item) for item in input_shape)
        if len(shape) != 2 or shape[-1] != weight_mod.shape[1]:
            raise MPCShapeError("reference HE linear correlations require shape (batch, input width)")
        batch = shape[0]
        input_client = self.random(shape)
        input_server = self.random(shape)
        output_shape = (batch, weight_mod.shape[0])
        output_server = self.random(output_shape)
        output_client = np.empty(output_shape, dtype=np.int64)

        encrypted_inputs = []
        encrypted_output_rows: list[list] = []
        for batch_index in range(batch):
            encrypted_payload = self._encrypt(input_client[batch_index])
            encrypted_input = self._server_load(encrypted_payload)
            encrypted_inputs.append(encrypted_input)
            started = time.perf_counter_ns()
            rows = self._encrypted_dot_outputs(
                encrypted_input,
                weight_mod,
                input_server[batch_index],
                output_server[batch_index],
            )
            self.stats.server_compute_ms += (time.perf_counter_ns() - started) / 1e6
            encrypted_output_rows.append(rows)
            for output_index, encrypted_output in enumerate(rows):
                output_client[batch_index, output_index] = self._decrypt(
                    encrypted_output, 1, (1,)
                )[0]

        # Input authentication reuses the encrypted row only for batch one. For
        # larger batches the reference path encrypts the whole flattened share.
        random_input = self._authenticate_parts(input_client, input_server)

        # Each output row is already available as a ciphertext. The reference
        # implementation authenticates the complete flattened output with one
        # additional encryption to keep the code simple and auditable.
        transformed = self._authenticate_parts(output_client, output_server)
        self.stats.linear_correlations += batch
        return LinearCorrelation(
            random_input=random_input,
            transformed_input=transformed,
            input_shape=shape,
            output_shape=output_shape,
        )

    def summary(self) -> dict[str, int | float]:
        return {
            "context_generation_ms": self.stats.context_generation_ms,
            "client_encrypt_ms": self.stats.client_encrypt_ms,
            "server_compute_ms": self.stats.server_compute_ms,
            "client_decrypt_ms": self.stats.client_decrypt_ms,
            "total_ms": self.stats.total_ms,
            "uploaded_bytes": self.stats.uploaded_bytes,
            "downloaded_bytes": self.stats.downloaded_bytes,
            "public_context_bytes": self.stats.public_context_bytes,
            "multiplication_triples": self.stats.multiplication_triples,
            "linear_correlations": self.stats.linear_correlations,
            "authenticated_values": self.stats.authenticated_values,
        }
