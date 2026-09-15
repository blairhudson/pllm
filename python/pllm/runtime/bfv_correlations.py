from __future__ import annotations

import os
from typing import Any

import msgpack
import numpy as np

from .masked_runtime import ModelError, centered_mod


def he_worker_threads() -> int:
    """Return the bounded TenSEAL worker count used by reference HE contexts.

    TenSEAL otherwise defaults to the host CPU count for every context. A test or
    multi-tenant client can create many contexts, so that default can accumulate
    hundreds of idle worker threads and eventually stall unrelated I/O. The
    reference runtime is latency-insensitive preprocessing and defaults to one
    worker; production correlation factories should set ``PLLM_HE_THREADS``
    explicitly or use a packed GPU backend.
    """
    raw = os.environ.get("PLLM_HE_THREADS", "1")
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ModelError("PLLM_HE_THREADS must be a positive integer") from exc


class BFVCorrelationClient:
    """Actual BFV client for offline one-time mask preprocessing.

    The server receives a public/evaluation context and Enc(r). It evaluates one
    encrypted dot product per output row and returns ciphertexts for W r. The
    client alone decrypts the transformed mask.
    """

    def __init__(
        self,
        *,
        dimension: int,
        plain_modulus: int = 65537,
        poly_modulus_degree: int = 4096,
        pydeps_path: str | None = None,
    ) -> None:
        if pydeps_path:
            import sys

            if pydeps_path not in sys.path:
                sys.path.insert(0, pydeps_path)
        import tenseal as ts

        self.ts = ts
        self.dimension = dimension
        self.plain_modulus = plain_modulus
        self.n_threads = he_worker_threads()
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_modulus_degree,
            plain_modulus=plain_modulus,
            encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC,
            n_threads=self.n_threads,
        )
        self.context.generate_galois_keys()
        self.public_context = self.context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=False,
        )

    def encrypt_mask(self, mask: np.ndarray) -> bytes:
        values = centered_mod(mask, self.plain_modulus).tolist()
        return self.ts.bfv_vector(self.context, values).serialize()

    def decrypt_transformed(self, payload: bytes) -> np.ndarray:
        value = msgpack.unpackb(payload, raw=False)
        ciphertexts = value["ciphertexts"]
        outputs: list[int] = []
        for item in ciphertexts:
            vector = self.ts.bfv_vector_from(self.context, item)
            outputs.append(int(vector.decrypt()[0]))
        return centered_mod(np.asarray(outputs, dtype=np.int64), self.plain_modulus)


class BFVCorrelationServer:
    def __init__(self, weight: np.ndarray, *, pydeps_path: str | None = None) -> None:
        if pydeps_path:
            import sys

            if pydeps_path not in sys.path:
                sys.path.insert(0, pydeps_path)
        import tenseal as ts

        self.ts = ts
        self.weight = np.asarray(weight, dtype=np.int64)
        self.n_threads = he_worker_threads()
        self.contexts: dict[str, Any] = {}

    def register_context(self, context_id: str, public_context: bytes) -> None:
        context = self.ts.context_from(public_context, n_threads=self.n_threads)
        if context.has_secret_key():
            raise ModelError("server context contains a secret key")
        self.contexts[context_id] = context

    def evaluate(
        self,
        context_id: str,
        encrypted_mask: bytes,
        *,
        output_mask: np.ndarray | None = None,
    ) -> bytes:
        context = self.contexts.get(context_id)
        if context is None:
            raise ModelError("unknown BFV context")
        if output_mask is not None:
            output_mask = np.asarray(output_mask, dtype=np.int64).reshape(-1)
            if output_mask.shape != (self.weight.shape[0],):
                raise ModelError("BFV output mask width mismatch")
        chunks: list[Any]
        chunk_size: int
        try:
            envelope = msgpack.unpackb(encrypted_mask, raw=False, strict_map_key=False)
        except Exception:
            envelope = None
        if isinstance(envelope, dict) and int(envelope.get("v", 0)) == 2:
            chunks = [self.ts.bfv_vector_from(context, item) for item in envelope["chunks"]]
            chunk_size = int(envelope["slot_count"])
            if int(envelope["size"]) != self.weight.shape[1]:
                raise ModelError("encrypted mask width mismatch")
        else:
            chunks = [self.ts.bfv_vector_from(context, encrypted_mask)]
            chunk_size = self.weight.shape[1]
        outputs: list[bytes] = []
        # Reference implementation: chunk inputs larger than one BFV SIMD
        # vector, compute one encrypted partial dot per chunk, then add partials.
        for row in self.weight:
            partials = []
            for index, encrypted in enumerate(chunks):
                start = index * chunk_size
                plain = row[start : start + chunk_size]
                partials.append(encrypted.dot(plain.tolist()))
            result = partials[0]
            for partial in partials[1:]:
                result += partial
            if output_mask is not None:
                result += int(output_mask[len(outputs)])
            outputs.append(result.serialize())
        return msgpack.packb({"ciphertexts": outputs}, use_bin_type=True)
