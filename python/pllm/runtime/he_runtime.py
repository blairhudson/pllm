from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
import secrets
import statistics
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Iterator, Protocol

import msgpack
import numpy as np

from .tokenizer import AlphabetTokenizer, DEFAULT_ALPHABET


class HEModelError(RuntimeError):
    pass


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
        raise HEModelError("PLLM_HE_THREADS must be a positive integer") from exc


def centered_mod(values: np.ndarray, modulus: int) -> np.ndarray:
    raw = np.asarray(values, dtype=np.int64) % modulus
    return np.where(raw > modulus // 2, raw - modulus, raw).astype(np.int64, copy=False)


def matvec_mod(weight: np.ndarray, vector: np.ndarray, modulus: int) -> np.ndarray:
    # Python/int64 is safe for the deliberately small reference models. Real
    # transformer stages use the bounded native modular GEMM path from Round 6.
    result = np.asarray(weight, dtype=np.int64) @ np.asarray(vector, dtype=np.int64)
    return centered_mod(result, modulus)


@dataclass(frozen=True, slots=True)
class ModelManifest:
    id: str
    owned_by: str
    tokenizer: str
    vocab_size: int
    protocol: str
    privacy_mode: str
    model_fingerprint: str
    max_output_tokens: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model_object(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "created": 0,
            "owned_by": self.owned_by,
            "he": {
                "protocol": self.protocol,
                "privacy_mode": self.privacy_mode,
                "tokenizer": self.tokenizer,
                "vocab_size": self.vocab_size,
                "model_fingerprint": self.model_fingerprint,
                **self.metadata,
            },
        }


@dataclass(slots=True)
class MaskCorrelation:
    id: str
    mask: np.ndarray
    transformed_mask: np.ndarray
    consumed: bool = False


class CorrelationPool:
    """Client-side one-time mask inventory.

    A correlation may only be consumed once. The server stores no corresponding
    plaintext mask; it receives a uniformly masked activation online.
    """

    def __init__(self, correlations: Iterable[MaskCorrelation] = ()) -> None:
        self._items = deque(correlations)
        self._lock = threading.Lock()
        self.generated = 0
        self.consumed = 0

    def put(self, correlation: MaskCorrelation) -> None:
        with self._lock:
            if correlation.consumed:
                raise HEModelError("cannot add a consumed correlation")
            self._items.append(correlation)
            self.generated += 1

    def extend(self, correlations: Iterable[MaskCorrelation]) -> None:
        for correlation in correlations:
            self.put(correlation)

    def take(self) -> MaskCorrelation:
        with self._lock:
            if not self._items:
                raise HEModelError("one-time correlation pool exhausted")
            item = self._items.popleft()
            if item.consumed:
                raise HEModelError("correlation reuse detected")
            item.consumed = True
            self.consumed += 1
            return item

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class CorrelationFactory(Protocol):
    def create(self, count: int) -> list[MaskCorrelation]: ...


class LocalCorrelationFactory:
    """Correctness/throughput baseline; not a private preprocessing protocol."""

    def __init__(self, weight: np.ndarray, modulus: int, seed: int = 42) -> None:
        self.weight = np.asarray(weight, dtype=np.int64)
        self.modulus = int(modulus)
        self.rng = np.random.default_rng(seed)

    def create(self, count: int) -> list[MaskCorrelation]:
        rows: list[MaskCorrelation] = []
        for _ in range(count):
            mask = self.rng.integers(0, self.modulus, size=self.weight.shape[1], dtype=np.int64)
            transformed = matvec_mod(self.weight, mask, self.modulus)
            rows.append(MaskCorrelation(secrets.token_hex(12), mask, transformed))
        return rows


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
            raise HEModelError("server context contains a secret key")
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
            raise HEModelError("unknown BFV context")
        if output_mask is not None:
            output_mask = np.asarray(output_mask, dtype=np.int64).reshape(-1)
            if output_mask.shape != (self.weight.shape[0],):
                raise HEModelError("BFV output mask width mismatch")
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
                raise HEModelError("encrypted mask width mismatch")
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


@dataclass(slots=True)
class MaskedLinearRequest:
    model: str
    correlation_id: str
    masked_input: np.ndarray

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "model": self.model,
                "correlation_id": self.correlation_id,
                "shape": list(self.masked_input.shape),
                "dtype": "i8",
                "data": self.masked_input.astype("<i8", copy=False).tobytes(),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "MaskedLinearRequest":
        value = msgpack.unpackb(payload, raw=False)
        if set(value) != {"model", "correlation_id", "shape", "dtype", "data"}:
            raise HEModelError("invalid masked-linear payload")
        if value["dtype"] != "i8":
            raise HEModelError("unsupported masked-linear dtype")
        array = np.frombuffer(value["data"], dtype="<i8").copy().reshape(value["shape"])
        return cls(str(value["model"]), str(value["correlation_id"]), array)


@dataclass(slots=True)
class MaskedLinearResponse:
    correlation_id: str
    masked_output: np.ndarray

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "correlation_id": self.correlation_id,
                "shape": list(self.masked_output.shape),
                "dtype": "i8",
                "data": self.masked_output.astype("<i8", copy=False).tobytes(),
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "MaskedLinearResponse":
        value = msgpack.unpackb(payload, raw=False)
        if set(value) != {"correlation_id", "shape", "dtype", "data"}:
            raise HEModelError("invalid masked-linear response")
        array = np.frombuffer(value["data"], dtype="<i8").copy().reshape(value["shape"])
        return cls(str(value["correlation_id"]), array)


class MaskedBigramModel:
    """Tiny exact private-inference model used by the functional API stack.

    The model is a dense transition matrix over a small alphabet. The online
    server receives only x+r and returns W(x+r). The client subtracts the
    one-time W r correlation and samples the next token locally.
    """

    def __init__(
        self,
        *,
        model_id: str = "he-bigram-demo",
        alphabet: str = DEFAULT_ALPHABET,
        phrase: str = "encrypted response\n",
        modulus: int = 65537,
    ) -> None:
        self.tokenizer = AlphabetTokenizer(alphabet)
        self.model_id = model_id
        self.modulus = modulus
        self.weight = np.full((self.tokenizer.vocab_size, self.tokenizer.vocab_size), -64, dtype=np.int64)
        tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(phrase) + [self.tokenizer.eos_token_id]
        for current, nxt in zip(tokens, tokens[1:]):
            self.weight[nxt, current] = 64
        self.weight[self.tokenizer.eos_token_id, self.tokenizer.eos_token_id] = 64
        fingerprint = hashlib.sha256(self.weight.astype("<i8").tobytes()).hexdigest()
        self.manifest = ModelManifest(
            id=model_id,
            owned_by="pllm",
            tokenizer=self.tokenizer.name,
            vocab_size=self.tokenizer.vocab_size,
            protocol="masked-linear-v1",
            privacy_mode="he_preprocessed",
            model_fingerprint=fingerprint,
            max_output_tokens=256,
            metadata={"demo": True, "modulus": modulus, "alphabet": alphabet},
        )

    def evaluate_masked(self, payload: bytes) -> bytes:
        request = MaskedLinearRequest.unpack(payload)
        if request.model != self.model_id:
            raise HEModelError("model mismatch")
        if request.masked_input.shape != (self.tokenizer.vocab_size,):
            raise HEModelError("masked input shape mismatch")
        output = matvec_mod(self.weight, request.masked_input, self.modulus)
        return MaskedLinearResponse(request.correlation_id, output).pack()


@dataclass(frozen=True, slots=True)
class MaskedBigramClientModel:
    """Public client metadata for the masked bigram model; contains no weights."""

    model_id: str
    tokenizer: AlphabetTokenizer
    modulus: int

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "MaskedBigramClientModel":
        alphabet = str(manifest.metadata.get("alphabet", DEFAULT_ALPHABET))
        modulus = int(manifest.metadata.get("modulus", 65537))
        return cls(manifest.id, AlphabetTokenizer(alphabet), modulus)


class MaskedBigramClientSession:
    def __init__(
        self,
        model: MaskedBigramModel | MaskedBigramClientModel,
        pool: CorrelationPool,
        *,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.pool = pool
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

    def request_for_token(self, token_id: int) -> tuple[MaskedLinearRequest, MaskCorrelation]:
        if not 0 <= token_id < self.model.tokenizer.vocab_size:
            raise HEModelError("token out of range")
        correlation = self.pool.take()
        one_hot = np.zeros(self.model.tokenizer.vocab_size, dtype=np.int64)
        one_hot[token_id] = 1
        masked = centered_mod(one_hot + correlation.mask, self.model.modulus)
        request = MaskedLinearRequest(self.model.model_id, correlation.id, masked)
        return request, correlation

    def complete_token(self, response_payload: bytes, correlation: MaskCorrelation) -> int:
        response = MaskedLinearResponse.unpack(response_payload)
        if response.correlation_id != correlation.id:
            raise HEModelError("correlation response mismatch")
        logits = centered_mod(response.masked_output - correlation.transformed_mask, self.model.modulus).astype(np.float64)
        if self.temperature <= 0:
            return int(np.argmax(logits))
        logits /= self.temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return int(self.rng.choice(len(probabilities), p=probabilities))


class StageExecutor(Protocol):
    async def execute(self, payloads: list[bytes]) -> list[bytes]: ...


class BigramStageExecutor:
    """Vectorised same-weight executor used by the reference gateway.

    This mirrors the production scheduler's key optimisation: requests from
    independent sessions at the same model stage become one matrix-matrix
    operation rather than many isolated matrix-vector calls.
    """

    def __init__(self, model: MaskedBigramModel) -> None:
        self.model = model

    async def execute(self, payloads: list[bytes]) -> list[bytes]:
        requests = [MaskedLinearRequest.unpack(payload) for payload in payloads]
        for request in requests:
            if request.model != self.model.model_id:
                raise HEModelError("model mismatch")
            if request.masked_input.shape != (self.model.tokenizer.vocab_size,):
                raise HEModelError("masked input shape mismatch")
        batch = np.stack([request.masked_input for request in requests], axis=0)
        outputs = centered_mod(batch @ self.model.weight.T, self.model.modulus)
        return [
            MaskedLinearResponse(request.correlation_id, output).pack()
            for request, output in zip(requests, outputs, strict=True)
        ]


class StageBatchScheduler:
    """Continuous same-stage batching for HE/masked model operations."""

    def __init__(
        self,
        executor: StageExecutor,
        *,
        max_batch_size: int = 32,
        max_wait_ms: float = 0.25,
        adaptive_wait: bool = True,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.executor = executor
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self.adaptive_wait = bool(adaptive_wait)
        self._queue: asyncio.Queue[tuple[bytes, asyncio.Future[bytes]]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self.batches = 0
        self.items = 0
        self.batch_sizes: list[int] = []

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, payload: bytes) -> bytes:
        await self.start()
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        await self._queue.put((payload, future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]

            # A fixed batching delay penalizes every sequential decode stage.
            # Yield once, drain requests that are already runnable, and only
            # spend the configured coalescing window after real contention is
            # observed. This preserves single-session latency while retaining
            # same-stage batching under concurrent load.
            should_wait = not self.adaptive_wait
            if self.adaptive_wait:
                await asyncio.sleep(0)
                while len(batch) < self.max_batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                should_wait = len(batch) > 1

            if should_wait and self.max_wait > 0 and len(batch) < self.max_batch_size:
                deadline = asyncio.get_running_loop().time() + self.max_wait
                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break
            payloads = [item[0] for item in batch]
            try:
                results = await self.executor.execute(payloads)
                if len(results) != len(batch):
                    raise HEModelError("stage executor returned wrong result count")
                for (_, future), result in zip(batch, results):
                    if not future.cancelled():
                        future.set_result(result)
            except Exception as exc:
                for _, future in batch:
                    if not future.cancelled():
                        future.set_exception(exc)
            finally:
                self.batches += 1
                self.items += len(batch)
                self.batch_sizes.append(len(batch))

    def stats(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "items": self.items,
            "mean_batch_size": statistics.fmean(self.batch_sizes) if self.batch_sizes else 0.0,
            "max_batch_size": max(self.batch_sizes, default=0),
        }
