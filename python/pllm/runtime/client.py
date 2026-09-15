from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import posixpath
import secrets
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
from urllib.parse import urlparse, urlunparse

import httpx
import msgpack
import numpy as np
from filelock import FileLock

from .secure_random import FieldRandom

from .bfv_correlations import BFVCorrelationClient, he_worker_threads
from .masked_runtime import (
    CorrelationPool,
    ModelError,
    MaskCorrelation,
    MaskedBigramClientModel,
    MaskedBigramClientSession,
    MaskedLinearRequest,
    MaskedLinearResponse,
    centered_mod,
)
from .protocol import (
    ProtocolEnvelope,
    encode_length_prefixed,
    iter_length_prefixed,
    pack_envelope,
    unpack_envelope,
)
from .preparation_protocol import (
    PreparationAck,
    PreparationRequest,
    SessionAuthorization,
    SessionAuthorizationAck,
    expand_output_mask,
    expand_preparation_mask,
)
from .stage_protocol import (
    BlindedStageCorrelation,
    BlindedStageRequest,
    BlindedStageResponse,
    DirectFHEStageRequest,
    DirectFHEStageResponse,
    blinded_correlation_from_wire,
    prepared_stage_batch_rows,
)
from .quantization import dequantize_matmul, quantize_activation_per_row
from .transformer_client import (
    ClientBundle,
    MaskedTransformerClientRuntime,
    PreparedInventory,
    PreparedInventoryLease,
    PreparedStageRows,
    RemoteLinear,
    RuntimeSnapshot,
    StageClientStats,
    StageMetadata,
    PreparedRemoteLinear,
    TransformerClientError,
)
from .responses import normalize_input, prompt_text
from .security import derive_session_key
from .tokenizer import AlphabetTokenizer
from .types import Response, ResponseEvent, ResponseUsage, new_id

if TYPE_CHECKING:
    from pllm.configuration import Experiment, ExperimentProfile

T = TypeVar("T")

_BUNDLE_CACHE_MODES = {"read-write", "read-only", "refresh", "off"}
_MAX_CLIENT_BUNDLE_BYTES = 8 * 1024 * 1024 * 1024


def _default_bundle_cache_dir() -> Path:
    base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "pllm" / "client-bundles"


def _normalized_inference_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) URL")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    if port is not None and port != (443 if scheme == "https" else 80):
        host = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    elif ":" in host:
        host = f"[{host}]"
    path = posixpath.normpath(parsed.path or "/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{scheme}://{host}{path.rstrip('/') or '/'}"


class ProtocolError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class _BundleIntegrityError(ProtocolError):
    pass


class ResponseStream(Generic[T]):
    def __init__(self, iterator: Iterator[T], close: Callable[[], None] | None = None) -> None:
        self.iterator = iterator
        self._close = close
        self._closed = False

    def __iter__(self) -> "ResponseStream[T]":
        return self

    def __next__(self) -> T:
        return next(self.iterator)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        callback, self._close = self._close, None
        try:
            iterator_close = getattr(self.iterator, "close", None)
            if iterator_close is not None:
                iterator_close()
        finally:
            if callback is not None:
                callback()

    def __enter__(self) -> "ResponseStream[T]":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(slots=True)
class PrivacyAudit:
    plaintext_prompt_bytes_sent: int = 0
    plaintext_token_ids_sent: int = 0
    public_context_bytes: int = 0
    encrypted_correlation_upload_bytes: int = 0
    encrypted_correlation_download_bytes: int = 0
    masked_online_upload_bytes: int = 0
    masked_online_download_bytes: int = 0
    correlation_count: int = 0
    online_steps: int = 0
    inference_stage_calls: int = 0
    token_lookup_cache_hits: int = 0
    token_lookup_cache_misses: int = 0
    kv_continuation_hits: int = 0
    kv_continuation_misses: int = 0
    direct_fhe_upload_bytes: int = 0
    direct_fhe_download_bytes: int = 0
    direct_fhe_steps: int = 0
    preparation_upload_bytes: int = 0
    preparation_download_bytes: int = 0
    inference_upload_bytes: int = 0
    inference_download_bytes: int = 0
    preparation_server_ns: int = 0
    inference_server_ns: int = 0
    correction_push_bytes: int = 0
    correction_push_ns: int = 0
    session_authorization_upload_bytes: int = 0
    session_authorization_download_bytes: int = 0
    preparation_attempts: int = 0
    preparation_rows: int = 0
    preparation_requests_during_online: int = 0
    preparation_failures: int = 0
    bundle_network_bytes: int = 0
    bundle_cache_hits: int = 0
    bundle_cache_misses: int = 0
    bundle_cache_corruptions: int = 0

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


class _Channel:
    def __init__(
        self, client: httpx.Client, base_url: str, api_key: str, session_id: str, mode: str
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session_id = session_id
        self.mode = mode
        self.socket: Any = None

    def _http(self, payload: bytes) -> bytes:
        response = self.client.post(
            f"/v1/runtime/sessions/{self.session_id}/execute",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/vnd.pllm.runtime+msgpack",
            },
            content=payload,
        )
        _raise(response)
        return response.content

    def _connect(self) -> None:
        if self.socket is not None:
            return
        from websockets.sync.client import connect

        parsed = urlparse(self.base_url)
        path = f"{parsed.path.rstrip('/')}/v1/runtime/ws/{self.session_id}"
        url = urlunparse(
            ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, path, "", "", "")
        )
        self.socket = connect(
            url,
            subprotocols=["pllm-runtime-v1"],
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=30,
            max_size=None,
        )

    def exchange(self, payload: bytes) -> bytes:
        if self.mode == "http":
            return self._http(payload)
        if self.mode not in {"websocket", "auto"}:
            raise ValueError(f"unsupported runtime transport {self.mode!r}")
        if self.mode == "auto" and self.socket is None:
            try:
                self._connect()
            except Exception:
                self.mode = "http"
                return self._http(payload)
        else:
            self._connect()
        # Never retry an accepted frame over HTTP: the nonce may already be consumed.
        self.socket.send(payload)
        value = self.socket.recv()
        return value.encode() if isinstance(value, str) else bytes(value)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


@dataclass(slots=True)
class _ModelCryptoState:
    model: MaskedBigramClientModel
    mode: str
    pool: CorrelationPool
    rng: FieldRandom
    bfv: BFVCorrelationClient | None = None
    context_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class _BFVStageClient:
    def __init__(
        self,
        *,
        plain_modulus: int,
        poly_modulus_degree: int = 4096,
        pydeps_path: str | None = None,
    ) -> None:
        if pydeps_path:
            import sys

            if pydeps_path not in sys.path:
                sys.path.insert(0, pydeps_path)
        import tenseal as ts

        self.ts = ts
        self.plain_modulus = int(plain_modulus)
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.slot_count = max(1, int(poly_modulus_degree) // 2)
        self.n_threads = he_worker_threads()
        self.context = ts.context(
            ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=poly_modulus_degree,
            plain_modulus=self.plain_modulus,
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

    def encrypt(self, mask: np.ndarray) -> bytes:
        values = np.asarray(mask, dtype=np.int64).reshape(-1) % self.plain_modulus
        chunks = [
            self.ts.bfv_vector(
                self.context, values[start : start + self.slot_count].tolist()
            ).serialize()
            for start in range(0, values.size, self.slot_count)
        ]
        return msgpack.packb(
            {
                "v": 2,
                "plain_modulus": self.plain_modulus,
                "size": int(values.size),
                "slot_count": self.slot_count,
                "chunks": chunks,
            },
            use_bin_type=True,
        )

    def decrypt(self, payload: bytes, *, stage_id: str) -> np.ndarray:
        value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        if int(value.get("v", 0)) != 1 or str(value.get("stage_id")) != stage_id:
            raise ModelError("invalid BFV stage correlation response")
        outputs = [
            int(self.ts.bfv_vector_from(self.context, item).decrypt()[0]) % self.plain_modulus
            for item in value["ciphertexts"]
        ]
        return np.asarray(outputs, dtype=np.uint32)


@dataclass(slots=True)
class _TransformerCryptoState:
    bundle: ClientBundle
    bundle_fingerprint: str
    mode: str
    privacy_mode: str = "public"
    privacy_protocol: str = "masked_w4a4"
    queues: dict[str, deque[Any]] = field(default_factory=lambda: defaultdict(deque))
    locks: dict[str, threading.Lock] = field(default_factory=lambda: defaultdict(threading.Lock))
    rng: FieldRandom = field(default_factory=FieldRandom)
    bfv_clients: dict[int, _BFVStageClient] = field(default_factory=dict)
    context_ids: dict[int, str] = field(default_factory=dict)
    blinded_owner_id: str = field(default_factory=lambda: new_id("owner"))
    token_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict)
    token_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    prepared_inventory: PreparedInventory | None = None
    prepared_inventory_spare: PreparedInventory | None = None
    retired_inventories: list[PreparedInventory] = field(default_factory=list)
    active_prepared_responses: int = 0
    refill_in_progress: bool = False
    preparation_verified: bool = False


@dataclass(slots=True)
class _TransformerConversationState:
    model_id: str
    bundle_fingerprint: str
    token_ids: list[int]
    rendered_context: str
    snapshot: RuntimeSnapshot
    next_logits: np.ndarray
    pending_token_ids: list[int]


class _BlindedCorrelationProvider:
    """Client inventory for output-blinded proprietary correlations."""

    def __init__(
        self,
        *,
        client: "RuntimeClient",
        session_id: str,
        state: _TransformerCryptoState,
        prefetch: int,
    ) -> None:
        self.owner = client
        self.session_id = session_id
        self.state = state
        self.prefetch = max(1, int(prefetch))
        self.model_id = state.bundle.model_id

    def take_many(self, stage: StageMetadata, count: int) -> list[BlindedStageCorrelation]:
        with self.state.locks[stage.id]:
            queue = self.state.queues[stage.id]
            missing = count - len(queue)
            if missing > 0:
                queue.extend(self._create(stage, max(missing, self.prefetch)))
            if len(queue) < count:
                raise ModelError(f"proprietary correlation inventory exhausted for {stage.id}")
            values = [queue.popleft() for _ in range(count)]
            output: list[BlindedStageCorrelation] = []
            for item in values:
                if not isinstance(item, BlindedStageCorrelation):
                    raise ModelError("mixed correlation protocols in proprietary inventory")
                if item.consumed:
                    raise ModelError("proprietary correlation reuse detected")
                item.consumed = True
                output.append(item)
            return output

    def _context(self, modulus: int) -> tuple[_BFVStageClient, str]:
        bfv = self.state.bfv_clients.get(modulus)
        context_id = self.state.context_ids.get(modulus)
        if bfv is not None and context_id is not None:
            return bfv, context_id
        bfv = _BFVStageClient(plain_modulus=modulus, pydeps_path=self.owner.tenseal_path)
        context_id = new_id(f"ctx{modulus}")
        response = self.owner.http.put(
            f"/v1/runtime/sessions/{self.session_id}/contexts/{context_id}",
            headers={**self.owner.headers, "Content-Type": "application/octet-stream"},
            content=bfv.public_context,
        )
        _raise(response)
        self.state.bfv_clients[modulus] = bfv
        self.state.context_ids[modulus] = context_id
        self.owner.audit.public_context_bytes += len(bfv.public_context)
        return bfv, context_id

    def _create(self, stage: StageMetadata, count: int) -> list[BlindedStageCorrelation]:
        if self.state.mode == "local-test":
            response = self.owner.http.post(
                f"/v1/runtime/sessions/{self.session_id}/correlations/proprietary/local-test",
                headers=self.owner.headers,
                json={
                    "count": count,
                    "stage_id": stage.id,
                    "owner_id": self.state.blinded_owner_id,
                },
            )
            _raise(response)
            value = msgpack.unpackb(response.content, raw=False, strict_map_key=False)
            rows = [blinded_correlation_from_wire(item) for item in value["items"]]
            self.owner.audit.correlation_count += len(rows)
            return rows
        if self.state.mode != "bfv":
            raise ValueError(f"unsupported proprietary correlation mode {self.state.mode!r}")
        bfv, context_id = self._context(int(stage.modulus))
        masks = [
            self.state.rng.integers(0, int(stage.modulus), size=stage.in_features, dtype=np.uint32)
            for _ in range(count)
        ]
        encrypted = [bfv.encrypt(mask) for mask in masks]
        upload = encode_length_prefixed(encrypted)
        self.owner.audit.encrypted_correlation_upload_bytes += len(upload)
        response = self.owner.http.post(
            f"/v1/runtime/sessions/{self.session_id}/correlations/proprietary/bfv/batch",
            headers={
                **self.owner.headers,
                "Content-Type": "application/octet-stream",
                "X-HE-Stage-ID": stage.id,
                "X-HE-Context-ID": context_id,
                "X-HE-Owner-ID": self.state.blinded_owner_id,
            },
            content=upload,
        )
        _raise(response)
        self.owner.audit.encrypted_correlation_download_bytes += len(response.content)
        payloads = list(iter_length_prefixed(response.content))
        if len(payloads) != count:
            raise ModelError("blinded BFV correlation batch returned the wrong item count")
        rows: list[BlindedStageCorrelation] = []
        for mask, payload in zip(masks, payloads, strict=True):
            envelope = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            correlation_id = str(envelope.get("correlation_id", ""))
            if not correlation_id:
                raise ModelError("blinded BFV correlation is missing its id")
            transformed = bfv.decrypt(payload, stage_id=stage.id)
            rows.append(
                BlindedStageCorrelation(
                    id=correlation_id,
                    stage_id=stage.id,
                    owner_id=self.state.blinded_owner_id,
                    mask=mask,
                    blinded_transformed_mask=transformed,
                    modulus=int(stage.modulus),
                    ring="prime",
                )
            )
        self.owner.audit.correlation_count += len(rows)
        return rows


class _BlindedRemoteLinear:
    """Fast output-blinded OLE-style stage client for proprietary mode."""

    def __init__(
        self,
        *,
        stages: dict[str, StageMetadata],
        correlations: _BlindedCorrelationProvider,
        session_id: str,
        state: _TransformerCryptoState,
        exchange: Callable[[str, list[bytes]], list[bytes]],
    ) -> None:
        self.stages = stages
        self.correlations = correlations
        self.session_id = session_id
        self.state = state
        self.exchange = exchange
        self.stats = StageClientStats()

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self.stages.get(stage_id)
        if stage is None:
            raise ModelError(f"unknown stage {stage_id!r}")
        value = np.asarray(activation, dtype=np.float32)
        if value.shape[-1] != stage.in_features:
            raise ModelError(
                f"stage {stage_id} expects {stage.in_features} features, got {value.shape}"
            )
        quantized = quantize_activation_per_row(value, bits=stage.activation_bits)
        rows = self.correlations.take_many(stage, quantized.rows)
        modulus = int(stage.modulus)
        if any(
            item.stage_id != stage_id
            or item.owner_id != self.state.blinded_owner_id
            or item.modulus != modulus
            or (item.ring or "prime") != "prime"
            for item in rows
        ):
            raise ModelError("proprietary correlation arithmetic profile mismatch")
        masks = np.stack([np.asarray(item.mask).reshape(stage.in_features) for item in rows])
        blinded_transformed = np.stack(
            [np.asarray(item.blinded_transformed_mask).reshape(stage.out_features) for item in rows]
        )
        masked = (quantized.values.astype(np.int64) - masks.astype(np.int64)) % modulus
        request = BlindedStageRequest(
            model=self.state.bundle.model_id,
            stage_id=stage_id,
            owner_id=self.state.blinded_owner_id,
            correlation_ids=tuple(item.id for item in rows),
            masked_input=masked.astype(np.uint32),
            activation_scales=quantized.scales,
            modulus=modulus,
            wire_bits=stage.wire_bits,
            ring="prime",
        )
        packed = request.pack()
        response_payloads = self.exchange(stage_id, [packed])
        if len(response_payloads) != 1:
            raise ModelError("blinded stage exchange returned the wrong result count")
        response = BlindedStageResponse.unpack(response_payloads[0])
        if (
            response.stage_id != stage_id
            or response.correlation_ids != request.correlation_ids
            or response.modulus != modulus
        ):
            raise ModelError("blinded stage response mismatch")
        combined = (
            response.masked_output.astype(np.uint64) + blinded_transformed.astype(np.uint64)
        ) % modulus
        integers = combined.astype(np.int64)
        integers = np.where(integers > modulus // 2, integers - modulus, integers)
        output = dequantize_matmul(
            integers,
            quantized.scales,
            stage.weight_scales,
            output_shape=quantized.original_shape[:-1] + (stage.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        self.stats.calls += 1
        self.stats.rows += quantized.rows
        self.stats.correlations += quantized.rows
        self.stats.upload_bytes += len(packed)
        self.stats.download_bytes += len(response_payloads[0])
        self.stats.server_ns += response.server_ns
        return output


class _DirectFHERemoteLinear:
    """Direct encrypted stage client used by proprietary mode."""

    def __init__(
        self,
        *,
        stages: dict[str, StageMetadata],
        owner: "RuntimeClient",
        session_id: str,
        state: _TransformerCryptoState,
        exchange: Callable[[str, list[bytes]], list[bytes]],
    ) -> None:
        self.stages = stages
        self.owner = owner
        self.session_id = session_id
        self.state = state
        self.exchange = exchange
        self.stats = StageClientStats()

    def _context(self, modulus: int) -> tuple[_BFVStageClient, str]:
        bfv = self.state.bfv_clients.get(modulus)
        context_id = self.state.context_ids.get(modulus)
        if bfv is not None and context_id is not None:
            return bfv, context_id
        bfv = _BFVStageClient(plain_modulus=modulus, pydeps_path=self.owner.tenseal_path)
        context_id = new_id(f"ctx{modulus}")
        response = self.owner.http.put(
            f"/v1/runtime/sessions/{self.session_id}/contexts/{context_id}",
            headers={**self.owner.headers, "Content-Type": "application/octet-stream"},
            content=bfv.public_context,
        )
        _raise(response)
        self.state.bfv_clients[modulus] = bfv
        self.state.context_ids[modulus] = context_id
        self.owner.audit.public_context_bytes += len(bfv.public_context)
        return bfv, context_id

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self.stages.get(stage_id)
        if stage is None:
            raise ModelError(f"unknown stage {stage_id!r}")
        value = np.asarray(activation, dtype=np.float32)
        if value.shape[-1] != stage.in_features:
            raise ModelError(
                f"stage {stage_id} expects {stage.in_features} features, got {value.shape}"
            )
        quantized = quantize_activation_per_row(value, bits=stage.activation_bits)
        bfv, context_id = self._context(stage.modulus)
        encrypted_rows = tuple(bfv.encrypt(row) for row in quantized.values)
        request = DirectFHEStageRequest(
            model=self.state.bundle.model_id,
            stage_id=stage_id,
            context_id=context_id,
            input_shape=quantized.original_shape,
            activation_scales=quantized.scales,
            encrypted_rows=encrypted_rows,
        )
        packed = request.pack()
        self.owner.audit.direct_fhe_upload_bytes += len(packed)
        result_payloads = self.exchange(stage_id, [packed])
        if len(result_payloads) != 1:
            raise ModelError("direct-FHE exchange returned the wrong result count")
        self.owner.audit.direct_fhe_download_bytes += len(result_payloads[0])
        response = DirectFHEStageResponse.unpack(result_payloads[0])
        if response.stage_id != stage_id or len(response.output_rows) != quantized.rows:
            raise ModelError("direct-FHE stage response mismatch")
        integer_rows = np.stack(
            [bfv.decrypt(item, stage_id=stage_id).astype(np.int64) for item in response.output_rows]
        )
        # BFV returns positive residues. Convert back to the signed dot-product range.
        integer_rows = np.where(
            integer_rows > stage.modulus // 2,
            integer_rows - stage.modulus,
            integer_rows,
        )
        output = dequantize_matmul(
            integer_rows,
            quantized.scales,
            stage.weight_scales,
            output_shape=quantized.original_shape[:-1] + (stage.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        self.stats.calls += 1
        self.stats.rows += quantized.rows
        self.stats.upload_bytes += len(packed)
        self.stats.download_bytes += len(result_payloads[0])
        self.stats.server_ns += response.server_ns
        self.owner.audit.direct_fhe_steps += 1
        return output


class _CorrelationSource:
    def __init__(
        self,
        *,
        client: "RuntimeClient",
        session_id: str,
        state: _ModelCryptoState,
        prefetch: int,
    ) -> None:
        self.owner = client
        self.session_id = session_id
        self.state = state
        self.model = state.model
        self.mode = state.mode
        self.prefetch = max(1, prefetch)
        if self.mode == "bfv" and self.state.bfv is None:
            self.state.bfv = BFVCorrelationClient(
                dimension=self.model.tokenizer.vocab_size,
                plain_modulus=self.model.modulus,
                pydeps_path=client.tenseal_path,
            )
            self.state.context_id = new_id("ctx")
            response = client.http.put(
                f"/v1/runtime/sessions/{session_id}/contexts/{self.state.context_id}",
                headers={**client.headers, "Content-Type": "application/octet-stream"},
                content=self.state.bfv.public_context,
            )
            _raise(response)
            client.audit.public_context_bytes += len(self.state.bfv.public_context)
        elif self.mode not in {"bfv", "local-test"}:
            raise ValueError(f"unsupported correlation mode {self.mode}")

    def take(self) -> MaskCorrelation:
        with self.state.lock:
            if len(self.state.pool) == 0:
                self._replenish_locked(self.prefetch)
            return self.state.pool.take()

    def replenish(self, count: int) -> int:
        if count <= 0:
            return len(self.state.pool)
        with self.state.lock:
            self._replenish_locked(count)
            return len(self.state.pool)

    def _replenish_locked(self, count: int) -> None:
        if self.mode == "local-test":
            response = self.owner.http.post(
                f"/v1/runtime/sessions/{self.session_id}/correlations/local-test",
                headers=self.owner.headers,
                json={"count": count},
            )
            _raise(response)
            value = msgpack.unpackb(response.content, raw=False)
            for item in value["items"]:
                shape = tuple(item["shape"])
                mask = np.frombuffer(item["mask"], dtype="<i8").copy().reshape(shape)
                transformed = np.frombuffer(item["transformed"], dtype="<i8").copy()
                self.state.pool.put(MaskCorrelation(str(item["id"]), mask, transformed))
            self.owner.audit.correlation_count += count
            return

        assert self.state.bfv is not None
        for _ in range(count):
            mask = self.state.rng.integers(
                0,
                self.model.modulus,
                size=self.model.tokenizer.vocab_size,
                dtype=np.int64,
            )
            encrypted = self.state.bfv.encrypt_mask(mask)
            self.owner.audit.encrypted_correlation_upload_bytes += len(encrypted)
            response = self.owner.http.post(
                f"/v1/runtime/sessions/{self.session_id}/correlations/bfv",
                headers={**self.owner.headers, "Content-Type": "application/octet-stream"},
                content=encrypted,
            )
            _raise(response)
            self.owner.audit.encrypted_correlation_download_bytes += len(response.content)
            transformed = self.state.bfv.decrypt_transformed(response.content)
            self.state.pool.put(MaskCorrelation(new_id("corr"), mask, transformed))
            self.owner.audit.correlation_count += 1


class RuntimeClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_model: str | None = None,
        session_transport: str = "http",
        correlation_mode: str = "bfv",
        preparation_base_url: str | None = None,
        preparation_api_key: str | None = None,
        correlation_prefetch: int = 4,
        prepared_inventory_rows: int = 64,
        background_inventory_refill: bool = True,
        token_cache_size: int = 512,
        bundle_cache_mode: str = "read-write",
        bundle_cache_dir: str | Path | None = None,
        tenseal_path: str | None = None,
        timeout: float = 300.0,
        http_client: httpx.Client | None = None,
        preparation_http_client: httpx.Client | None = None,
        experiment: Experiment | ExperimentProfile | None = None,
    ) -> None:
        from pllm.configuration import Experiment, ExperimentProfile

        if experiment is None:
            self.experiment = None
        elif isinstance(experiment, ExperimentProfile):
            self.experiment = experiment
        elif isinstance(experiment, Experiment):
            self.experiment = experiment.resolve()
        else:
            raise TypeError("experiment must be an Experiment or ExperimentProfile")
        if (
            self.experiment is not None
            and default_model is not None
            and default_model != self.experiment.model
        ):
            raise ValueError("default_model conflicts with Experiment model")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = self.experiment.model if self.experiment is not None else default_model
        self.session_transport = session_transport
        self.correlation_mode = correlation_mode
        self.correlation_prefetch = correlation_prefetch
        if prepared_inventory_rows < 1:
            raise ValueError("prepared_inventory_rows must be positive")
        self.prepared_inventory_rows = prepared_inventory_rows
        self.background_inventory_refill = background_inventory_refill
        self._closing = False
        self.token_cache_size = max(0, int(token_cache_size))
        if bundle_cache_mode not in _BUNDLE_CACHE_MODES:
            raise ValueError(
                "bundle_cache_mode must be read-write, read-only, refresh, or off"
            )
        self.bundle_cache_mode = bundle_cache_mode
        self._bundle_cache_explicit = bundle_cache_dir is not None
        self.bundle_cache_dir = Path(bundle_cache_dir).expanduser() if bundle_cache_dir else (
            _default_bundle_cache_dir()
        )
        self._bundle_endpoint = _normalized_inference_endpoint(self.base_url)
        self.tenseal_path = tenseal_path
        self._owns_http = http_client is None
        self.http = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._owns_preparation_http = (
            preparation_http_client is None and preparation_base_url is not None
        )
        self.preparation_http = preparation_http_client
        if self.preparation_http is None and preparation_base_url is not None:
            self.preparation_http = httpx.Client(
                base_url=preparation_base_url.rstrip("/"), timeout=timeout
            )
        if self.preparation_http is not None:
            if not preparation_api_key:
                raise ValueError("preparation_api_key is required for preparation service")
            if preparation_api_key == api_key:
                raise ValueError("inference and preparation credentials must be distinct")
            provider_urls = (self.http.base_url, self.preparation_http.base_url)
            origins = [
                (
                    item.scheme,
                    item.host,
                    item.port or (443 if item.scheme == "https" else 80),
                )
                for item in provider_urls
            ]
            if origins[0] == origins[1]:
                raise ValueError("preparation and inference require distinct origins")
            for name, provider_url in zip(
                ("base_url", "preparation_base_url"), provider_urls, strict=True
            ):
                if provider_url.scheme != "https" and provider_url.host not in {
                    "127.0.0.1",
                    "localhost",
                    "::1",
                }:
                    raise ValueError(f"{name} must use HTTPS outside loopback")
        self.preparation_headers = {
            "Authorization": f"Bearer {preparation_api_key}"
        }
        self._provider_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="pllm-provider"
        )
        self.cache: dict[str, Response] = {}
        self.histories: dict[str, str] = {}
        self.message_histories: dict[str, list[dict[str, str]]] = {}
        self.audit = PrivacyAudit()
        self._crypto_states: dict[str, _ModelCryptoState] = {}
        self._crypto_state_lock = threading.Lock()
        self._transformer_states: dict[str, _TransformerCryptoState] = {}
        self._transformer_state_lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self._online_active = 0
        self._preparation_active = 0
        self._transformer_conversations: dict[str, _TransformerConversationState] = {}
        self._transformer_conversation_lock = threading.Lock()
        self._model_manifests: dict[str, dict[str, Any]] = {}

    def _open_private_session(
        self,
        model_id: str,
        *,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], MaskedBigramClientModel, _CorrelationSource]:
        # Context creation/registration is serialized per client. The context and
        # one-time-correlation inventory are reused across Responses, while each
        # Response retains an isolated authenticated online session.
        with self._crypto_state_lock:
            existing = self._crypto_states.get(model_id)
            session_body: dict[str, Any] = {
                "model": model_id,
                "max_output_tokens": max_output_tokens,
            }
            if existing is not None and existing.context_id is not None:
                session_body["context_id"] = existing.context_id
            session_response = self.http.post(
                "/v1/runtime/sessions", headers=self.headers, json=session_body
            )
            _raise(session_response)
            session_value = session_response.json()
            runtime = session_value["manifest"]["runtime"]
            model = MaskedBigramClientModel(
                model_id,
                AlphabetTokenizer(str(runtime["alphabet"])),
                int(runtime["modulus"]),
            )
            state = existing
            if state is None:
                state = _ModelCryptoState(
                    model=model,
                    mode=self.correlation_mode,
                    pool=CorrelationPool(),
                    rng=FieldRandom(),
                )
                self._crypto_states[model_id] = state
            elif (
                state.model.tokenizer.vocab_size != model.tokenizer.vocab_size
                or state.model.modulus != model.modulus
                or state.mode != self.correlation_mode
            ):
                raise ModelError("server model manifest changed during client lifetime")
            source = _CorrelationSource(
                client=self,
                session_id=str(session_value["id"]),
                state=state,
                prefetch=self.correlation_prefetch,
            )
            return session_value, model, source

    def _model_manifest(self, model_id: str, *, refresh: bool = False) -> dict[str, Any]:
        cached = None if refresh else self._model_manifests.get(model_id)
        if cached is not None:
            return cached
        response = self.http.get(f"/v1/runtime/models/{model_id}", headers=self.headers)
        _raise(response)
        value = response.json()
        self._model_manifests[model_id] = value
        return value

    def _client_bundle_descriptor(self, model_id: str) -> dict[str, Any]:
        descriptor = self._model_manifest(model_id, refresh=True).get("client_bundle")
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "schema", "sha256", "size", "etag"
        }:
            raise ProtocolError("provider model descriptor lacks client bundle fingerprint", 409)
        fingerprint = str(descriptor["sha256"])
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise ProtocolError("provider client bundle fingerprint is invalid", 409)
        if descriptor["etag"] != f'"{fingerprint}"':
            raise ProtocolError("provider client bundle descriptor is invalid", 409)
        try:
            schema = int(descriptor["schema"])
            size = int(descriptor["size"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("provider client bundle descriptor is invalid", 409) from exc
        if schema < 1 or size < 1 or size > _MAX_CLIENT_BUNDLE_BYTES:
            raise ProtocolError("provider client bundle descriptor is invalid", 409)
        return descriptor

    @staticmethod
    def _ensure_private_directory(path: Path) -> None:
        existed = path.exists()
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not existed:
            path.chmod(0o700)

    def _bundle_cache_namespace(self, *, create: bool) -> bytes | None:
        key_path = self.bundle_cache_dir / ".identity-key"
        if not create:
            try:
                key = key_path.read_bytes()
            except FileNotFoundError:
                return None
            if len(key) != 32:
                raise OSError(f"invalid bundle cache identity key: {key_path}")
            return key

        self._ensure_private_directory(self.bundle_cache_dir)
        with FileLock(str(key_path) + ".lock"):
            try:
                key = key_path.read_bytes()
            except FileNotFoundError:
                key = secrets.token_bytes(32)
                descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(key)
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException:
                    try:
                        key_path.unlink()
                    except OSError:
                        pass
                    raise
            if len(key) != 32:
                raise OSError(f"invalid bundle cache identity key: {key_path}")
            return key

    def _bundle_cache_path(
        self,
        model_id: str,
        fingerprint: str | None = None,
        *,
        create_namespace: bool = True,
    ) -> Path | None:
        del fingerprint
        namespace = self._bundle_cache_namespace(create=create_namespace)
        if namespace is None:
            return None
        identity = json.dumps(
            {
                "endpoint": self._bundle_endpoint,
                "model": model_id,
                "credential": self.api_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        key = hmac.new(namespace, identity, hashlib.sha256).hexdigest()
        return self.bundle_cache_dir / key[:2] / f"{key}.msgpack"

    @staticmethod
    def _read_cached_bundle(
        path: Path,
        *,
        model_id: str,
        fingerprint: str,
        schema: int,
        size: int,
    ) -> tuple[ClientBundle | None, str]:
        if not path.is_file():
            return None, "miss"
        try:
            payload = path.read_bytes()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != fingerprint:
                try:
                    stale = ClientBundle.unpack(payload)
                except (TransformerClientError, ValueError, TypeError):
                    return None, "corrupt"
                if stale.model_id == model_id and stale.schema_version == schema:
                    return None, "miss"
                return None, "corrupt"
            bundle = ClientBundle.unpack(payload)
            if bundle.model_id != model_id or bundle.schema_version != schema:
                return None, "corrupt"
            return bundle, "hit"
        except (OSError, TransformerClientError, ValueError, TypeError):
            return None, "corrupt"

    def _download_client_bundle(
        self,
        model_id: str,
        *,
        fingerprint: str,
        schema: int,
        size: int,
    ) -> tuple[ClientBundle, bytes]:
        response = self.http.get(
            f"/v1/runtime/models/{model_id}/client-bundle",
            headers=self.headers,
        )
        _raise(response)
        payload = response.content
        self.audit.bundle_network_bytes += len(payload)
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != fingerprint
            or response.headers.get("X-PLLM-Bundle-SHA256") != fingerprint
        ):
            raise _BundleIntegrityError("provider client bundle fingerprint mismatch", 409)
        try:
            bundle = ClientBundle.unpack(payload)
        except TransformerClientError as exc:
            raise _BundleIntegrityError("provider client bundle is invalid", 409) from exc
        if bundle.model_id != model_id or bundle.schema_version != schema:
            raise _BundleIntegrityError("provider client bundle descriptor mismatch", 409)
        return bundle, payload

    @staticmethod
    def _write_cached_bundle(path: Path, payload: bytes) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
            try:
                directory = os.open(
                    path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                # Directory fsync is unsupported on some otherwise safe filesystems.
                pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def _load_client_bundle_descriptor(
        self,
        model_id: str,
        descriptor: dict[str, Any],
    ) -> ClientBundle:
        fingerprint = str(descriptor["sha256"])
        schema = int(descriptor["schema"])
        size = int(descriptor["size"])
        if self.bundle_cache_mode == "off":
            return self._download_client_bundle(
                model_id, fingerprint=fingerprint, schema=schema, size=size
            )[0]

        path: Path | None = None
        downloaded: tuple[ClientBundle, bytes] | None = None
        try:
            path = self._bundle_cache_path(
                model_id,
                fingerprint,
                create_namespace=self.bundle_cache_mode != "read-only",
            )
        except OSError as exc:
            if self._bundle_cache_explicit:
                raise ProtocolError(f"configured bundle cache is unavailable: {exc}") from exc
            return self._download_client_bundle(
                model_id, fingerprint=fingerprint, schema=schema, size=size
            )[0]

        if self.bundle_cache_mode == "read-only":
            try:
                if path is None:
                    bundle, status = None, "miss"
                else:
                    bundle, status = self._read_cached_bundle(
                        path,
                        model_id=model_id,
                        fingerprint=fingerprint,
                        schema=schema,
                        size=size,
                    )
            except OSError:
                bundle, status = None, "miss"
            if status == "hit":
                self.audit.bundle_cache_hits += 1
                assert bundle is not None
                return bundle
            if status == "corrupt":
                self.audit.bundle_cache_corruptions += 1
            else:
                self.audit.bundle_cache_misses += 1
            return self._download_client_bundle(
                model_id, fingerprint=fingerprint, schema=schema, size=size
            )[0]

        assert path is not None
        try:
            self._ensure_private_directory(path.parent)
            with FileLock(str(path) + ".lock"):
                if self.bundle_cache_mode == "read-write":
                    bundle, status = self._read_cached_bundle(
                        path,
                        model_id=model_id,
                        fingerprint=fingerprint,
                        schema=schema,
                        size=size,
                    )
                    if status == "hit":
                        self.audit.bundle_cache_hits += 1
                        assert bundle is not None
                        return bundle
                    if status == "corrupt":
                        self.audit.bundle_cache_corruptions += 1
                    else:
                        self.audit.bundle_cache_misses += 1
                downloaded = self._download_client_bundle(
                    model_id, fingerprint=fingerprint, schema=schema, size=size
                )
                self._write_cached_bundle(path, downloaded[1])
                return downloaded[0]
        except OSError as exc:
            if self._bundle_cache_explicit:
                raise ProtocolError(f"configured bundle cache is unavailable: {exc}") from exc
            if downloaded is not None:
                return downloaded[0]
            return self._download_client_bundle(
                model_id, fingerprint=fingerprint, schema=schema, size=size
            )[0]

    def _load_client_bundle_record(
        self,
        model_id: str,
        descriptor: dict[str, Any] | None = None,
    ) -> tuple[ClientBundle, str]:
        descriptor = descriptor or self._client_bundle_descriptor(model_id)
        for attempt in range(2):
            try:
                bundle = self._load_client_bundle_descriptor(model_id, descriptor)
                return bundle, str(descriptor["sha256"])
            except _BundleIntegrityError:
                if attempt:
                    raise
                current = self._client_bundle_descriptor(model_id)
                if current == descriptor:
                    raise
                descriptor = current
        raise AssertionError("unreachable")

    def _load_client_bundle(self, model_id: str) -> ClientBundle:
        return self._load_client_bundle_record(model_id)[0]

    @staticmethod
    def _remote_stages(state: _TransformerCryptoState) -> list[StageMetadata]:
        return [
            stage
            for stage in state.bundle.stages.values()
            if stage.client_weight is None and stage.id != "embed_tokens"
        ]

    def _transformer_state(self, model_id: str) -> _TransformerCryptoState:
        if self.experiment is not None and model_id != self.experiment.model:
            raise ValueError("request model conflicts with Experiment model")
        with self._transformer_state_lock:
            descriptor = self._client_bundle_descriptor(model_id)
            if self.experiment is not None:
                metadata = descriptor.get("metadata") or {}
                if (
                    metadata.get("client_runtime") != "masked_transformer_v1"
                    or metadata.get("privacy_mode") != "public"
                ):
                    raise ProtocolError(
                        "Experiment requires public masked_transformer_v1 inference", 409
                    )
            bundle_fingerprint = str(descriptor["sha256"])
            state = self._transformer_states.get(model_id)
            if state is None or state.bundle_fingerprint != bundle_fingerprint:
                bundle, bundle_fingerprint = self._load_client_bundle_record(
                    model_id, descriptor
                )
                state = _TransformerCryptoState(
                    bundle=bundle,
                    bundle_fingerprint=bundle_fingerprint,
                    mode=self.correlation_mode,
                    privacy_mode=str(bundle.privacy.get("mode", "public")),
                    privacy_protocol=str(bundle.privacy.get("protocol", "masked_w4a4")),
                )
                self._transformer_states[model_id] = state

            if self.experiment is not None and state.privacy_mode != "public":
                raise ProtocolError("Experiment requires public masked_transformer_v1 inference", 409)

            if state.privacy_mode == "public" and not state.preparation_verified:
                if self.preparation_http is None:
                    raise ProtocolError("public inference requires a preparation service", 400)
                inference_response = self.http.get("/v1/models", headers=self.headers)
                _raise(inference_response)
                inference_models = [
                    item
                    for item in inference_response.json().get("data", [])
                    if item.get("id") == model_id
                ]
                if len(inference_models) != 1:
                    raise ProtocolError("inference service does not serve the requested model", 404)
                inference = inference_models[0].get("runtime") or {}
                if (
                    inference.get("body_fingerprint")
                    != state.bundle.privacy.get("body_fingerprint")
                    or inference.get("stage_commitment")
                    != state.bundle.privacy.get("stage_commitment")
                    or inference.get("architecture")
                    != state.bundle.manifest.get("architecture")
                    or inference.get("stage_count")
                    != len(state.bundle.manifest.get("stages", []))
                ):
                    raise ProtocolError("inference and client model commitments do not match", 409)
                model_response = self.preparation_http.get(
                    "/v1/models",
                    headers=self.preparation_headers,
                )
                _raise(model_response)
                preparation_models = [
                    item
                    for item in model_response.json().get("data", [])
                    if item.get("id") == model_id
                ]
                if len(preparation_models) != 1:
                    raise ProtocolError("preparation service does not serve the requested model", 404)
                preparation = preparation_models[0].get("preparation") or {}
                remote_stages = self._remote_stages(state)
                if not remote_stages or any(
                    not stage.weight_digest or stage.seeded_profile is None
                    for stage in remote_stages
                ):
                    raise ProtocolError(
                        "provider bundle lacks stage weight or ring commitments", 409
                    )
                if (
                    preparation.get("protocol") != "seeded-inventory/v1"
                    or preparation.get("body_fingerprint")
                    != state.bundle.privacy.get("body_fingerprint")
                    or preparation.get("stage_commitment")
                    != state.bundle.privacy.get("stage_commitment")
                    or preparation.get("weight_bits")
                    != state.bundle.privacy.get("weight_bits")
                    or preparation.get("activation_bits")
                    != state.bundle.privacy.get("activation_bits")
                    or preparation.get("architecture") != state.bundle.manifest.get("architecture")
                    or preparation.get("stage_count")
                    != len(state.bundle.manifest.get("stages", []))
                ):
                    raise ProtocolError("preparation and inference model commitments do not match", 409)
                state.preparation_verified = True
            if self.experiment is not None and not state.preparation_verified:
                raise ProtocolError("Experiment requires seeded-inventory preparation", 409)
            return state

    def _prepare_inventory_locked(
        self,
        model_id: str,
        state: _TransformerCryptoState,
        rows: int,
    ) -> PreparedInventory:
        self._begin_preparation()
        try:
            return self._prepare_inventory(model_id, state, rows)
        finally:
            self._end_preparation()

    def _prepare_inventory(
        self,
        model_id: str,
        state: _TransformerCryptoState,
        rows: int,
    ) -> PreparedInventory:
        if self.preparation_http is None:
            raise ProtocolError("public inference requires a preparation service", 400)
        remote_stages = self._remote_stages(state)
        response = self.http.post(
            "/v1/runtime/inventories",
            headers=self.headers,
            json={
                "model": model_id,
                "rows": rows,
            },
        )
        _raise(response)
        value = response.json()
        inventory_id = str(value["id"])
        try:
            descriptor = value.get("preparation_authorization") or {}
            expected = {
                "body_fingerprint": state.bundle.privacy.get("body_fingerprint"),
                "stage_commitment": state.bundle.privacy.get("stage_commitment"),
                "weight_bits": state.bundle.privacy.get("weight_bits"),
                "activation_bits": state.bundle.privacy.get("activation_bits"),
            }
            if any(descriptor.get(name) != item for name, item in expected.items()):
                raise ProtocolError("inventory authorization commitments do not match", 409)
            weight_bits = expected["weight_bits"]
            activation_bits = expected["activation_bits"]
            if not isinstance(weight_bits, int) or not isinstance(activation_bits, int):
                raise ProtocolError("inventory authorization precision is invalid", 409)
            authorization = SessionAuthorization(
                session_id=inventory_id,
                model=model_id,
                body_fingerprint=str(expected["body_fingerprint"]),
                stage_commitment=str(expected["stage_commitment"]),
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                max_attempts=int(descriptor["max_attempts"]),
                rows=int(descriptor["rows"]),
                stage_ids=tuple(str(stage_id) for stage_id in descriptor["stage_ids"]),
            )
            if authorization.rows != rows or authorization.stage_ids != tuple(
                stage.id for stage in remote_stages
            ):
                raise ModelError("preparation inventory authorization mismatch")
            authorization_payload = authorization.pack()
            self.audit.session_authorization_upload_bytes += len(authorization_payload)
            authorized = self.preparation_http.post(
                f"/v1/preparation/inventories/{inventory_id}/authorize",
                headers={
                    **self.preparation_headers,
                    "Content-Type": "application/octet-stream",
                },
                content=authorization_payload,
            )
            _raise(authorized)
            self.audit.session_authorization_download_bytes += len(authorized.content)
            if SessionAuthorizationAck.unpack(authorized.content).session_id != inventory_id:
                raise ModelError("preparation inventory authorization mismatch")

            prepared_stages: dict[str, PreparedStageRows] = {}
            for stage in remote_stages:
                profile = stage.seeded_profile
                assert profile is not None
                request = PreparationRequest(
                    attempt_id=secrets.token_hex(16),
                    session_id=inventory_id,
                    model=model_id,
                    body_fingerprint=str(state.bundle.privacy["body_fingerprint"]),
                    stage_id=stage.id,
                    weight_digest=stage.weight_digest,
                    rows=rows,
                    in_features=stage.in_features,
                    out_features=stage.out_features,
                    seed=secrets.token_bytes(32),
                    weight_bits=int(state.bundle.privacy["weight_bits"]),
                    activation_bits=int(state.bundle.privacy["activation_bits"]),
                    signed_output_bound=profile.signed_output_bound,
                    ring=profile.ring,
                    modulus=profile.modulus,
                    wire_bits=profile.wire_bits,
                )
                payload = request.pack()
                self.audit.preparation_upload_bytes += len(payload)
                from .telemetry import record_protocol_bytes, start_protocol_span

                protocol_span = start_protocol_span(
                    stage.id,
                    len(payload),
                    0,
                    phase="offline",
                )
                try:
                    prepared = self.preparation_http.post(
                        f"/v1/preparation/inventories/{inventory_id}/stages/{stage.id}",
                        headers={
                            **self.preparation_headers,
                            "Content-Type": "application/octet-stream",
                        },
                        content=payload,
                    )
                    _raise(prepared)
                except Exception as exc:
                    protocol_span.record_exception(exc)
                    protocol_span.end()
                    raise
                self.audit.preparation_attempts += 1
                self.audit.preparation_rows += request.rows
                self.audit.preparation_download_bytes += len(prepared.content)
                try:
                    ack = PreparationAck.unpack(prepared.content)
                except Exception as exc:
                    protocol_span.record_exception(exc)
                    protocol_span.end()
                    raise
                if ack.attempt_id != request.attempt_id or ack.stage_id != stage.id:
                    error = ModelError("preparation inventory acknowledgement mismatch")
                    protocol_span.record_exception(error)
                    protocol_span.end()
                    raise error
                self.audit.correction_push_bytes += ack.correction_bytes
                self.audit.preparation_server_ns += ack.server_ns
                self.audit.correction_push_ns += ack.push_ns
                record_protocol_bytes("client", "preparation", len(payload), stage.id)
                record_protocol_bytes(
                    "preparation", "client", len(prepared.content), stage.id
                )
                protocol_span.set_attribute(
                    "pllm.preparation_inference.bytes", ack.correction_bytes
                )
                protocol_span.set_attribute(
                    "pllm.preparation_client.bytes", len(prepared.content)
                )
                protocol_span.end()
                prepared_stages[stage.id] = PreparedStageRows(
                    request=request,
                    input_mask=expand_preparation_mask(request),
                    output_mask=expand_output_mask(request),
                )
            sealed = self.http.post(
                f"/v1/runtime/inventories/{inventory_id}/ready",
                headers=self.headers,
            )
            _raise(sealed)
            if sealed.json().get("status") != "ready":
                raise ModelError("inference did not commit prepared inventory")
            return PreparedInventory(inventory_id, rows, prepared_stages)
        except BaseException:
            self.audit.preparation_failures += 1
            try:
                self.http.post(
                    f"/v1/runtime/inventories/{inventory_id}/cancel", headers=self.headers
                )
            except Exception:
                pass
            try:
                self.preparation_http.post(
                    f"/v1/preparation/inventories/{inventory_id}/cancel",
                    headers=self.preparation_headers,
                )
            except Exception:
                pass
            raise

    def _cancel_prepared_inventory(self, inventory: PreparedInventory) -> None:
        try:
            self.http.post(
                f"/v1/runtime/inventories/{inventory.id}/cancel", headers=self.headers
            )
        except Exception:
            pass
        if self.preparation_http is not None:
            try:
                self.preparation_http.post(
                    f"/v1/preparation/inventories/{inventory.id}/cancel",
                    headers=self.preparation_headers,
                )
            except Exception:
                pass

    def _prepared_inventory_is_live(self, inventory: PreparedInventory) -> bool:
        try:
            response = self.http.get(
                f"/v1/runtime/inventories/{inventory.id}", headers=self.headers
            )
        except httpx.HTTPError:
            raise
        if response.status_code in {404, 409, 410}:
            return False
        _raise(response)
        value = response.json()
        return (
            value.get("status") == "ready"
            and int(value.get("next_row", -1)) == inventory.claimed
        )

    def _install_prepared_inventory_locked(
        self,
        state: _TransformerCryptoState,
        inventory: PreparedInventory,
    ) -> None:
        previous = state.prepared_inventory
        state.prepared_inventory = inventory
        if previous is None or previous is inventory:
            return
        if previous.status()["reserved"] == 0:
            self._cancel_prepared_inventory(previous)
        else:
            state.retired_inventories.append(previous)

    def _background_refill(
        self,
        model_id: str,
        state: _TransformerCryptoState,
    ) -> None:
        with self._transformer_state_lock:
            inventory = state.prepared_inventory
            if (
                self._closing
                or state.active_prepared_responses != 0
                or inventory is None
                or state.prepared_inventory_spare is not None
                or state.refill_in_progress
            ):
                return
            state.refill_in_progress = True
            capacity = max(self.prepared_inventory_rows, inventory.capacity)
        try:
            spare = self._prepare_inventory_locked(model_id, state, capacity)
        except Exception:
            spare = None
        with self._transformer_state_lock:
            state.refill_in_progress = False
            if (
                spare is not None
                and not self._closing
                and state.active_prepared_responses == 0
                and state.prepared_inventory is inventory
                and state.prepared_inventory_spare is None
            ):
                state.prepared_inventory_spare = spare
            elif spare is not None:
                self._cancel_prepared_inventory(spare)

    def _finish_prepared_response(
        self,
        model_id: str,
        state: _TransformerCryptoState,
        inventory: PreparedInventoryLease,
    ) -> None:
        inventory.close()
        self._end_online()
        with self._transformer_state_lock:
            state.active_prepared_responses = max(0, state.active_prepared_responses - 1)
            retained: list[PreparedInventory] = []
            for retired in state.retired_inventories:
                if retired.status()["reserved"] == 0:
                    self._cancel_prepared_inventory(retired)
                else:
                    retained.append(retired)
            state.retired_inventories = retained
            if (
                self.background_inventory_refill
                and not self._closing
                and state.active_prepared_responses == 0
                and state.prepared_inventory_spare is None
            ):
                self._provider_executor.submit(self._background_refill, model_id, state)

    def _begin_online(self) -> None:
        with self._activity_lock:
            if self._preparation_active:
                raise ModelError("prepared inventory refill is in progress; retry when ready")
            self._online_active += 1

    def _end_online(self) -> None:
        with self._activity_lock:
            self._online_active = max(0, self._online_active - 1)

    def _begin_preparation(self) -> None:
        with self._activity_lock:
            if self._online_active:
                self.audit.preparation_requests_during_online += 1
                raise ModelError("preparation cannot run while inference is online")
            self._preparation_active += 1

    def _end_preparation(self) -> None:
        with self._activity_lock:
            self._preparation_active = max(0, self._preparation_active - 1)

    def _open_transformer_session(
        self,
        model_id: str,
        *,
        max_output_tokens: int,
        required_rows: int | None = None,
    ) -> tuple[dict[str, Any], _TransformerCryptoState, Any | None]:
        state = self._transformer_state(model_id)
        prepared_public = state.privacy_mode == "public"
        provider: Any | None
        with self._transformer_state_lock:
            if prepared_public:
                if state.refill_in_progress:
                    raise ModelError("prepared inventory refill is in progress; retry when ready")
                needed = max(1, int(required_rows or max_output_tokens + 1))
                inventory = state.prepared_inventory
                if inventory is not None and not self._prepared_inventory_is_live(inventory):
                    self._cancel_prepared_inventory(inventory)
                    state.prepared_inventory = None
                    inventory = None
                if (
                    (inventory is None or inventory.available < needed)
                    and state.prepared_inventory_spare is not None
                    and state.prepared_inventory_spare.available >= needed
                ):
                    spare = state.prepared_inventory_spare
                    state.prepared_inventory_spare = None
                    if self._prepared_inventory_is_live(spare):
                        inventory = spare
                        self._install_prepared_inventory_locked(state, inventory)
                    else:
                        self._cancel_prepared_inventory(spare)
                if inventory is None or inventory.available < needed:
                    raise ModelError(
                        "prepared inventory became unavailable before reservation; "
                        "retry the request"
                    )
                provider = inventory.reserve(needed)
            else:
                provider = None

            session_body: dict[str, Any] = {
                "model": model_id,
                "max_output_tokens": max_output_tokens,
            }
            if prepared_public:
                assert provider is not None
                session_body["execution"] = "seeded-preparation"
                session_body["inventory_id"] = provider.inventory_id
                session_body["inventory_start"] = provider.reservation_start
                session_body["inventory_rows"] = provider.reservation_rows
            if state.context_ids:
                session_body["context_ids"] = list(state.context_ids.values())
            online_started = False
            try:
                if prepared_public:
                    self._begin_online()
                    online_started = True
                session_response = self.http.post(
                    "/v1/runtime/sessions",
                    headers=self.headers,
                    json=session_body,
                )
                _raise(session_response)
                session_value = session_response.json()
                session_id = str(session_value["id"])
            except BaseException:
                if prepared_public and provider is not None:
                    provider.close()
                    if online_started:
                        self._end_online()
                raise
            if prepared_public:
                state.active_prepared_responses += 1

            if state.privacy_mode == "proprietary":
                if state.privacy_protocol in {"blinded_ole_w4a4", "guarded_blinded_w4a4"}:
                    if state.mode not in {"bfv", "local-test"}:
                        raise ValueError(f"unsupported proprietary correlation mode {state.mode!r}")
                    provider = _BlindedCorrelationProvider(
                        client=self,
                        session_id=session_id,
                        state=state,
                        prefetch=self.correlation_prefetch,
                    )
                elif state.privacy_protocol == "direct_bfv_w4a4":
                    provider = None
                else:
                    raise ModelError(
                        f"unsupported proprietary protocol {state.privacy_protocol!r}"
                    )
            elif state.privacy_mode != "public":
                raise ModelError(f"unsupported server privacy mode {state.privacy_mode!r}")
            return session_value, state, provider

    def preprocess(
        self,
        model_id: str,
        count: int,
        *,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        descriptor = self._model_manifest(model_id, refresh=True)
        target = max(0, int(count))
        if descriptor.get("metadata", {}).get("client_runtime") in {
            "masked_transformer_v1",
            "direct_fhe_transformer_v1",
            "blinded_ole_transformer_v1",
            "guarded_blinded_transformer_v1",
        }:
            if descriptor.get("metadata", {}).get("privacy_mode") == "public":
                state = self._transformer_state(model_id)
                selected = set(stages or (stage.id for stage in self._remote_stages(state)))
                expected = {stage.id for stage in self._remote_stages(state)}
                if selected != expected:
                    raise ValueError("prepared inventories require every remote stage")
                with self._transformer_state_lock:
                    if state.active_prepared_responses:
                        raise ModelError("prepared inventory refill requires an idle model")
                    prepared_inventory = state.prepared_inventory
                    if prepared_inventory is not None and not self._prepared_inventory_is_live(
                        prepared_inventory
                    ):
                        self._cancel_prepared_inventory(prepared_inventory)
                        state.prepared_inventory = None
                        prepared_inventory = None
                    if (
                        (prepared_inventory is None or prepared_inventory.available < target)
                        and state.prepared_inventory_spare is not None
                        and state.prepared_inventory_spare.available >= target
                    ):
                        spare = state.prepared_inventory_spare
                        state.prepared_inventory_spare = None
                        if self._prepared_inventory_is_live(spare):
                            prepared_inventory = spare
                            self._install_prepared_inventory_locked(state, prepared_inventory)
                        else:
                            self._cancel_prepared_inventory(spare)
                    if prepared_inventory is None or prepared_inventory.available < target:
                        if state.prepared_inventory_spare is not None:
                            self._cancel_prepared_inventory(state.prepared_inventory_spare)
                            state.prepared_inventory_spare = None
                        prepared_inventory = self._prepare_inventory_locked(
                            model_id,
                            state,
                            max(self.prepared_inventory_rows, target),
                        )
                        self._install_prepared_inventory_locked(state, prepared_inventory)
                        generated = prepared_inventory.capacity
                    else:
                        generated = 0
                    if self.background_inventory_refill:
                        spare = state.prepared_inventory_spare
                        if spare is not None and spare.available < target:
                            self._cancel_prepared_inventory(spare)
                            state.prepared_inventory_spare = None
                            spare = None
                        if spare is None:
                            state.prepared_inventory_spare = self._prepare_inventory_locked(
                                model_id,
                                state,
                                max(self.prepared_inventory_rows, target),
                            )
                            generated += state.prepared_inventory_spare.capacity
                    available = prepared_inventory.available
                return {
                    "object": "runtime.preprocessing_result",
                    "model": model_id,
                    "privacy_mode": "public",
                    "protocol": "seeded-inventory/v1",
                    "status": "ready",
                    "generated": generated,
                    "available_per_stage": {
                        stage_id: available for stage_id in expected
                    },
                }
            session, state, provider = self._open_transformer_session(model_id, max_output_tokens=1)
            if state.privacy_mode == "proprietary" and state.privacy_protocol == "direct_bfv_w4a4":
                complete = self.http.post(
                    f"/v1/runtime/sessions/{session['id']}/complete",
                    headers=self.headers,
                    json={"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
                )
                _raise(complete)
                return {
                    "object": "runtime.preprocessing_result",
                    "model": model_id,
                    "privacy_mode": "proprietary",
                    "protocol": "direct_bfv_w4a4",
                    "generated": 0,
                    "available_per_stage": {},
                    "message": "direct BFV reference mode does not use preprocessing",
                }
            assert provider is not None
            selected = set(stages or state.bundle.stages)
            unknown = selected.difference(state.bundle.stages)
            if unknown:
                raise ValueError(f"unknown transformer stages: {sorted(unknown)}")
            generated = 0
            inventory: dict[str, int] = {}
            for stage_id, stage in state.bundle.stages.items():
                if (
                    stage_id not in selected
                    or stage_id == "embed_tokens"
                    or stage.client_weight is not None
                ):
                    continue
                with state.locks[stage_id]:
                    queue = state.queues[stage_id]
                    missing = max(0, target - len(queue))
                    if missing:
                        queue.extend(provider._create(stage, missing))
                        generated += missing
                    inventory[stage_id] = len(queue)
            complete = self.http.post(
                f"/v1/runtime/sessions/{session['id']}/complete",
                headers=self.headers,
                json={"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
            )
            _raise(complete)
            return {
                "object": "runtime.preprocessing_result",
                "model": model_id,
                "generated": generated,
                "available_per_stage": inventory,
                "contexts": len(state.context_ids),
                "context_reused": bool(state.context_ids),
            }

        session, _, source = self._open_private_session(model_id, max_output_tokens=1)
        before = len(source.state.pool)
        available = source.replenish(target)
        complete = self.http.post(
            f"/v1/runtime/sessions/{session['id']}/complete",
            headers=self.headers,
            json={"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
        )
        _raise(complete)
        return {
            "object": "runtime.preprocessing_result",
            "model": model_id,
            "generated": available - before,
            "available": available,
            "context_reused": self.audit.public_context_bytes > 0,
        }

    def prepared_inventory_status(self, model: str) -> dict[str, Any]:
        with self._transformer_state_lock:
            state = self._transformer_states.get(model)
            if state is None or state.prepared_inventory is None:
                return {
                    "status": "not-loaded",
                    "capacity": 0,
                    "available": 0,
                    "reserved": 0,
                    "burned": 0,
                }
            return state.prepared_inventory.status()

    def prepared_rows_for_response(
        self,
        model: str,
        input: str,
        max_output_tokens: int,
        *,
        instructions: str | None = None,
    ) -> int:
        state = self._transformer_state(model)
        messages: list[dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": input})
        rendered = state.bundle.render_prompt(messages, add_generation_prompt=True)
        tokenizer = state.bundle.tokenizer()
        add_bos = bool(state.bundle.tokenizer_descriptor.get("add_bos_token", True))
        ids = tokenizer.encode(rendered, add_bos=add_bos)
        return len(ids or [int(state.bundle.config["bos_token_id"])]) + max(
            0, max_output_tokens - 1
        )

    def _ensure_prepared_inventory(
        self,
        model_id: str,
        state: _TransformerCryptoState,
        required_rows: int,
    ) -> None:
        with self._transformer_state_lock:
            candidates = (state.prepared_inventory, state.prepared_inventory_spare)
            if any(
                inventory is not None
                and inventory.available >= required_rows
                and self._prepared_inventory_is_live(inventory)
                for inventory in candidates
            ):
                return
        self.preprocess(model_id, count=required_rows)

    def close(self) -> None:
        self._closing = True
        self._provider_executor.shutdown(wait=True, cancel_futures=True)
        with self._transformer_state_lock:
            inventories: list[PreparedInventory] = []
            for state in self._transformer_states.values():
                if state.prepared_inventory is not None:
                    inventories.append(state.prepared_inventory)
                if state.prepared_inventory_spare is not None:
                    inventories.append(state.prepared_inventory_spare)
                inventories.extend(state.retired_inventories)
                state.prepared_inventory = None
                state.prepared_inventory_spare = None
                state.retired_inventories.clear()
        for inventory in inventories:
            self._cancel_prepared_inventory(inventory)
        if self._owns_preparation_http and self.preparation_http is not None:
            self.preparation_http.close()
        if self._owns_http:
            self.http.close()

    def list_models(self) -> dict[str, Any]:
        response = self.http.get("/v1/models", headers=self.headers)
        _raise(response)
        return response.json()

    def retrieve(self, response_id: str) -> Response:
        if response_id in self.cache:
            return self.cache[response_id]
        response = self.http.get(f"/v1/responses/{response_id}", headers=self.headers)
        _raise(response)
        return Response.from_dict(response.json())

    def cancel(self, response_id: str) -> Response:
        response = self.http.post(f"/v1/responses/{response_id}/cancel", headers=self.headers)
        _raise(response)
        if response_id in self.cache:
            self.cache[response_id].status = "cancelled"
            return self.cache[response_id]
        return Response.from_dict(response.json())

    def create(self, body: dict[str, Any]) -> Response | ResponseStream[ResponseEvent]:
        if body.get("stream"):
            return ResponseStream(self.events(body))
        final: Response | None = None
        for event in self.events(body):
            if event.type == "response.completed":
                final = Response.from_dict(event.response)
        if final is None:
            raise ProtocolError("response stream ended without response.completed")
        return final

    def events(self, body: dict[str, Any]) -> Iterator[ResponseEvent]:
        requested_model = body.get("model")
        if self.experiment is not None:
            model_id = self.experiment.model
            if requested_model is not None and str(requested_model) != model_id:
                raise ValueError("request model conflicts with Experiment model")
            body["model"] = model_id
        else:
            model_id = str(body.get("model") or self.default_model or "")
        if not model_id:
            models = self.list_models().get("data", [])
            private = [
                item.get("id")
                for item in models
                if item.get("runtime", {}).get("privacy_mode")
                not in {"manifest_only", "trusted_backend"}
            ]
            private = [str(item) for item in private if item]
            if len(private) == 1:
                model_id = private[0]
            elif not private:
                raise ValueError("the server exposes no private model")
            else:
                raise ValueError(
                    "model is required when the server exposes more than one private model"
                )
            body["model"] = model_id
        messages = normalize_input(body.get("input", ""), instructions=body.get("instructions"))
        rendered = prompt_text(messages)
        previous_id = body.get("previous_response_id")
        previous_history: str | None = None
        if previous_id:
            if str(previous_id) not in self.histories:
                raise ProtocolError("unknown previous_response_id in client-private cache", 404)
            previous_history = self.histories[str(previous_id)]
        descriptor = self._model_manifest(model_id, refresh=True)
        if self.experiment is not None:
            self._transformer_state(model_id)
        if descriptor.get("metadata", {}).get("client_runtime") in {
            "masked_transformer_v1",
            "direct_fhe_transformer_v1",
            "blinded_ole_transformer_v1",
            "guarded_blinded_transformer_v1",
        }:
            yield from self._transformer_events(
                body,
                messages,
                descriptor,
                previous_history=previous_history,
            )
            return
        if previous_history is not None:
            rendered = previous_history + "\n" + rendered
        # rendered remains local; only usage counts and opaque masked vectors cross the wire.
        session_value, model, correlations = self._open_private_session(
            model_id,
            max_output_tokens=int(body.get("max_output_tokens") or 64),
        )
        session_id = str(session_value["id"])
        response_id = str(session_value["response_id"])
        runtime = session_value["manifest"]["runtime"]
        channel = _Channel(self.http, self.base_url, self.api_key, session_id, self.session_transport)
        max_tokens = max(
            1,
            min(
                int(body.get("max_output_tokens") or 64),
                int(runtime.get("max_output_tokens", 4096)),
            ),
        )
        message_id = new_id("msg")
        seq = 0
        created = {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "status": "in_progress",
            "model": model_id,
            "output": [],
            "usage": None,
        }
        output_ids: list[int] = []
        output_chunks: list[str] = []
        key = derive_session_key(self.api_key, session_id)
        current = model.tokenizer.bos_token_id
        try:
            yield ResponseEvent.from_dict(
                {"type": "response.created", "sequence_number": seq, "response": created}
            )
            seq += 1
            yield ResponseEvent.from_dict(
                {"type": "response.in_progress", "sequence_number": seq, "response": created}
            )
            seq += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }
            )
            seq += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.content_part.added",
                    "sequence_number": seq,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                }
            )
            seq += 1

            for step in range(max_tokens):
                correlation = correlations.take()
                one_hot = np.zeros(model.tokenizer.vocab_size, dtype=np.int64)
                one_hot[current] = 1
                masked = centered_mod(one_hot + correlation.mask, model.modulus)
                request = MaskedLinearRequest(model.model_id, correlation.id, masked)
                envelope = ProtocolEnvelope.create(
                    request_id=f"{response_id}:{step}",
                    session_id=session_id,
                    model=model_id,
                    kind="masked.linear",
                    sequence=step,
                    payload=request.pack(),
                    metadata={},
                    key=key,
                )
                upload = pack_envelope(envelope)
                self.audit.masked_online_upload_bytes += len(upload)
                result_bytes = channel.exchange(upload)
                self.audit.masked_online_download_bytes += len(result_bytes)
                result = unpack_envelope(result_bytes)
                result.verify(key)
                if result.kind != "masked.linear.result" or result.sequence != step:
                    raise ModelError("mismatched runtime result frame")
                decoded = MaskedLinearResponse.unpack(result.payload)
                if decoded.correlation_id != correlation.id:
                    raise ModelError("correlation response mismatch")
                logits = centered_mod(
                    decoded.masked_output - correlation.transformed_mask, model.modulus
                ).astype(np.float64)
                temperature = float(body.get("temperature") or 0.0)
                if temperature <= 0:
                    current = int(np.argmax(logits))
                else:
                    logits = logits / temperature
                    logits -= logits.max()
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum()
                    current = int(
                        np.random.default_rng(42 + step).choice(len(probabilities), p=probabilities)
                    )
                self.audit.online_steps += 1
                if current == model.tokenizer.eos_token_id:
                    break
                output_ids.append(current)
                delta = model.tokenizer.decode([current])
                output_chunks.append(delta)
                yield ResponseEvent.from_dict(
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": seq,
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                        "logprobs": [],
                    }
                )
                seq += 1

            text = "".join(output_chunks)
            content = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
            item = {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [content],
            }
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_text.done",
                    "sequence_number": seq,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                }
            )
            seq += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.content_part.done",
                    "sequence_number": seq,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": content,
                }
            )
            seq += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_item.done",
                    "sequence_number": seq,
                    "output_index": 0,
                    "item": item,
                }
            )
            seq += 1
            input_tokens = len(rendered.encode("utf-8"))
            usage = ResponseUsage(input_tokens, len(output_ids), input_tokens + len(output_ids))
            final = Response(
                id=response_id,
                model=model_id,
                output=[item],
                status="completed",
                instructions=body.get("instructions"),
                metadata=body.get("metadata"),
                parallel_tool_calls=bool(body.get("parallel_tool_calls", True)),
                temperature=body.get("temperature"),
                top_p=body.get("top_p"),
                tools=list(body.get("tools") or []),
                tool_choice=body.get("tool_choice", "auto"),
                truncation=body.get("truncation", "disabled"),
                max_output_tokens=max_tokens,
                previous_response_id=body.get("previous_response_id"),
                store=bool(body.get("store", False)),
                usage=usage,
            )
            self.cache[response_id] = final
            self.histories[response_id] = rendered + "\nassistant: " + text
            complete = self.http.post(
                f"/v1/runtime/sessions/{session_id}/complete",
                headers=self.headers,
                json={"usage": usage.to_dict()},
            )
            _raise(complete)
            yield ResponseEvent.from_dict(
                {"type": "response.completed", "sequence_number": seq, "response": final.to_dict()}
            )
        finally:
            channel.close()

    def _transformer_events(
        self,
        body: dict[str, Any],
        messages: list[Any],
        descriptor: dict[str, Any],
        *,
        previous_history: str | None = None,
    ) -> Iterator[ResponseEvent]:
        model_id = str(body["model"])
        state = self._transformer_state(model_id)
        max_tokens = max(
            1,
            min(
                int(body.get("max_output_tokens") or 64),
                int(descriptor.get("context_length", 4096)),
            ),
        )
        temperature = float(body.get("temperature") or 0.0)
        raw_top_p = body.get("top_p")
        top_p = None if raw_top_p is None else float(raw_top_p)
        current_messages = normalize_input(
            body.get("input", ""), instructions=body.get("instructions")
        )
        previous_id = body.get("previous_response_id")
        structured: list[dict[str, str]] = []
        if previous_id:
            structured.extend(self.message_histories.get(str(previous_id), []))
        structured.extend({"role": row.role, "content": row.text} for row in current_messages)
        rendered = state.bundle.render_prompt(structured, add_generation_prompt=True)
        add_bos = bool(state.bundle.tokenizer_descriptor.get("add_bos_token", True))
        tokenizer = state.bundle.tokenizer()
        input_ids = tokenizer.encode(rendered, add_bos=add_bos)
        required_input_rows = len(input_ids or [int(state.bundle.config["bos_token_id"])])
        if previous_id:
            with self._transformer_conversation_lock:
                candidate = self._transformer_conversations.get(str(previous_id))
            if (
                candidate is not None
                and candidate.model_id == model_id
                and candidate.bundle_fingerprint == state.bundle_fingerprint
                and rendered.startswith(candidate.rendered_context)
            ):
                required_input_rows = len(
                    tokenizer.encode(rendered[len(candidate.rendered_context) :], add_bos=False)
                ) + len(candidate.pending_token_ids)
        required_rows = required_input_rows + max(0, max_tokens - 1)
        if state.privacy_mode == "public":
            self._ensure_prepared_inventory(model_id, state, required_rows)
        session_value, state, provider = self._open_transformer_session(
            model_id,
            max_output_tokens=max_tokens,
            required_rows=required_rows,
        )
        session_id = str(session_value["id"])
        response_id = str(session_value["response_id"])
        channel: _Channel | None = None

        def abandon_transformer_session() -> None:
            if channel is not None:
                channel.close()
            if isinstance(provider, PreparedInventoryLease):
                self._finish_prepared_response(model_id, state, provider)
            try:
                self.http.post(
                    f"/v1/runtime/sessions/{session_id}/cancel",
                    headers=self.headers,
                )
            except Exception:
                pass

        try:
            channel = _Channel(self.http, self.base_url, self.api_key, session_id, self.session_transport)
        except BaseException:
            abandon_transformer_session()
            raise
        assert channel is not None
        key = derive_session_key(self.api_key, session_id)
        sequence = 0

        def exchange(stage_id: str, payloads: list[bytes]) -> list[bytes]:
            nonlocal sequence
            self.audit.inference_stage_calls += 1
            compact_rows = (
                prepared_stage_batch_rows(payloads[0]) if len(payloads) == 1 else None
            )
            if len(payloads) > 1 or compact_rows is not None:
                upload = encode_length_prefixed(payloads)
                if state.privacy_protocol != "direct_bfv_w4a4":
                    self.audit.masked_online_upload_bytes += len(upload)
                response = self.http.post(
                    f"/v1/runtime/sessions/{session_id}/stages/{stage_id}",
                    headers={**self.headers, "Content-Type": "application/octet-stream"},
                    content=upload,
                )
                _raise(response)
                if state.privacy_protocol != "direct_bfv_w4a4":
                    self.audit.masked_online_download_bytes += len(response.content)
                results = list(iter_length_prefixed(response.content))
                self.audit.online_steps += compact_rows or len(payloads)
                return results
            envelope = ProtocolEnvelope.create(
                request_id=f"{response_id}:{sequence}",
                session_id=session_id,
                model=model_id,
                kind="masked.transformer.stage",
                sequence=sequence,
                payload=payloads[0],
                metadata={"stage_id": stage_id},
                key=key,
            )
            sequence += 1
            upload = pack_envelope(envelope)
            if state.privacy_protocol != "direct_bfv_w4a4":
                self.audit.masked_online_upload_bytes += len(upload)
            raw = channel.exchange(upload)
            if state.privacy_protocol != "direct_bfv_w4a4":
                self.audit.masked_online_download_bytes += len(raw)
            result = unpack_envelope(raw)
            result.verify(key)
            if result.kind != "masked.transformer.stage.result":
                raise ModelError("unexpected transformer stage result")
            self.audit.online_steps += 1
            return [result.payload]

        if state.privacy_mode == "public":
            if not isinstance(provider, PreparedInventoryLease):
                abandon_transformer_session()
                raise ModelError("public mode requires a prepared inventory lease")
            remote = PreparedRemoteLinear(
                model_id=model_id,
                body_fingerprint=str(state.bundle.privacy["body_fingerprint"]),
                stages=state.bundle.stages,
                inference=exchange,
                inventory=provider,
            )
        elif state.privacy_protocol == "direct_bfv_w4a4":
            remote = _DirectFHERemoteLinear(
                stages=state.bundle.stages,
                owner=self,
                session_id=session_id,
                state=state,
                exchange=exchange,
            )
        elif state.privacy_protocol in {"blinded_ole_w4a4", "guarded_blinded_w4a4"}:
            if not isinstance(provider, _BlindedCorrelationProvider):
                raise ModelError("blinded proprietary mode requires its correlation provider")
            remote = _BlindedRemoteLinear(
                stages=state.bundle.stages,
                correlations=provider,
                session_id=session_id,
                state=state,
                exchange=exchange,
            )
        else:
            if provider is None:
                raise ModelError("public mode requires a correlation provider")
            remote = RemoteLinear(state.bundle.stages, provider, exchange)
        try:
            runtime = MaskedTransformerClientRuntime(
                state.bundle,
                remote,
                token_cache=state.token_cache,
                token_cache_size=self.token_cache_size,
                token_cache_lock=state.token_cache_lock,
            )
        except BaseException:
            abandon_transformer_session()
            raise
        message_id = new_id("msg")
        event_sequence = 0
        output_ids: list[int] = []
        output_chunks: list[str] = []
        previous_text = ""
        session_completed = False

        created = {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "status": "in_progress",
            "model": model_id,
            "output": [],
            "usage": None,
        }
        try:
            yield ResponseEvent.from_dict(
                {"type": "response.created", "sequence_number": event_sequence, "response": created}
            )
            event_sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.in_progress",
                    "sequence_number": event_sequence,
                    "response": created,
                }
            )
            event_sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_item.added",
                    "sequence_number": event_sequence,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                }
            )
            event_sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.content_part.added",
                    "sequence_number": event_sequence,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                }
            )
            event_sequence += 1

            continuation_used = False
            prior: _TransformerConversationState | None = None
            if previous_id:
                with self._transformer_conversation_lock:
                    candidate = self._transformer_conversations.get(str(previous_id))
                    if (
                        candidate is not None
                        and candidate.model_id == model_id
                        and candidate.bundle_fingerprint == state.bundle_fingerprint
                    ):
                        prior = candidate

            suffix: list[int] | None = None
            if prior is not None and rendered.startswith(prior.rendered_context):
                # Preserve the exact previously sampled token IDs. Re-encoding
                # decoded assistant text is not guaranteed to round-trip for
                # byte fallback tokens, normalising tokenizers or special tokens.
                suffix_text = rendered[len(prior.rendered_context) :]
                suffix = runtime.tokenizer.encode(suffix_text, add_bos=False)
                full_input_ids = [*prior.token_ids, *suffix]
            else:
                full_input_ids = runtime.encode_prompt(rendered)
                if (
                    prior is not None
                    and len(full_input_ids) >= len(prior.token_ids)
                    and full_input_ids[: len(prior.token_ids)] == prior.token_ids
                ):
                    suffix = full_input_ids[len(prior.token_ids) :]

            if prior is not None and suffix is not None:
                runtime.restore(prior.snapshot)
                continuation_used = True
                pending_input_ids = [*prior.pending_token_ids, *suffix]
                logits = (
                    runtime.forward_ids(pending_input_ids)[-1]
                    if pending_input_ids
                    else prior.next_logits.copy()
                )
                input_ids = full_input_ids
                caches = runtime.caches
            else:
                input_ids, logits, caches = runtime.prepare_ids(full_input_ids)
            pending_token_ids: list[int] = []
            for step in range(max_tokens):
                token = runtime.sample(logits, temperature=temperature, top_p=top_p)
                if token == int(runtime.cfg["eos_token_id"]):
                    break
                output_ids.append(token)
                text_so_far = runtime.tokenizer.decode(output_ids)
                delta = (
                    text_so_far[len(previous_text) :]
                    if text_so_far.startswith(previous_text)
                    else text_so_far
                )
                previous_text = text_so_far
                output_chunks.append(delta)
                yield ResponseEvent.from_dict(
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": event_sequence,
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta,
                        "logprobs": [],
                    }
                )
                event_sequence += 1
                if step + 1 >= max_tokens:
                    pending_token_ids = [token]
                    break
                logits, caches = runtime.decode_step(token, caches, len(input_ids) + step)

            self.audit.token_lookup_cache_hits += runtime.token_cache_hits
            self.audit.token_lookup_cache_misses += runtime.token_cache_misses
            if previous_id:
                if continuation_used:
                    self.audit.kv_continuation_hits += 1
                else:
                    self.audit.kv_continuation_misses += 1

            text = "".join(output_chunks)
            content = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
            item = {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [content],
            }
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_text.done",
                    "sequence_number": event_sequence,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                    "logprobs": [],
                }
            )
            event_sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.content_part.done",
                    "sequence_number": event_sequence,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": content,
                }
            )
            event_sequence += 1
            yield ResponseEvent.from_dict(
                {
                    "type": "response.output_item.done",
                    "sequence_number": event_sequence,
                    "output_index": 0,
                    "item": item,
                }
            )
            event_sequence += 1
            usage = ResponseUsage(len(input_ids), len(output_ids), len(input_ids) + len(output_ids))
            final = Response(
                id=response_id,
                model=model_id,
                output=[item],
                status="completed",
                instructions=body.get("instructions"),
                metadata=body.get("metadata"),
                parallel_tool_calls=bool(body.get("parallel_tool_calls", True)),
                temperature=body.get("temperature"),
                top_p=body.get("top_p"),
                tools=list(body.get("tools") or []),
                tool_choice=body.get("tool_choice", "auto"),
                truncation=body.get("truncation", "disabled"),
                max_output_tokens=max_tokens,
                previous_response_id=body.get("previous_response_id"),
                store=bool(body.get("store", False)),
                usage=usage,
            )
            self.cache[response_id] = final
            self.histories[response_id] = rendered + text
            self.message_histories[response_id] = [
                *structured,
                {"role": "assistant", "content": text},
            ]
            with self._transformer_conversation_lock:
                self._transformer_conversations[response_id] = _TransformerConversationState(
                    model_id=model_id,
                    bundle_fingerprint=state.bundle_fingerprint,
                    token_ids=[*input_ids, *output_ids],
                    rendered_context=rendered + text,
                    snapshot=runtime.snapshot(),
                    next_logits=np.asarray(logits, dtype=np.float32).copy(),
                    pending_token_ids=pending_token_ids,
                )
            complete = self.http.post(
                f"/v1/runtime/sessions/{session_id}/complete",
                headers=self.headers,
                json={"usage": usage.to_dict()},
            )
            _raise(complete)
            session_completed = True
            yield ResponseEvent.from_dict(
                {
                    "type": "response.completed",
                    "sequence_number": event_sequence,
                    "response": final.to_dict(),
                }
            )
        finally:
            if isinstance(remote, PreparedRemoteLinear):
                self.audit.preparation_upload_bytes += remote.stats.preparation_upload_bytes
                self.audit.preparation_download_bytes += remote.stats.preparation_download_bytes
                self.audit.inference_upload_bytes += remote.stats.inference_upload_bytes
                self.audit.inference_download_bytes += remote.stats.inference_download_bytes
                self.audit.preparation_server_ns += remote.stats.preparation_server_ns
                self.audit.inference_server_ns += remote.stats.inference_server_ns
                self.audit.correction_push_bytes += remote.stats.correction_push_bytes
                self.audit.correction_push_ns += remote.stats.correction_push_ns
                self._finish_prepared_response(model_id, state, remote.inventory)
                if not session_completed:
                    try:
                        self.http.post(
                            f"/v1/responses/{response_id}/cancel", headers=self.headers
                        )
                    except Exception:
                        pass
            channel.close()


class ResponsesResource:
    def __init__(self, core: RuntimeClient) -> None:
        self.core = core

    def create(self, **kwargs: Any) -> Response | ResponseStream[ResponseEvent]:
        return self.core.create(kwargs)

    def retrieve(self, response_id: str) -> Response:
        return self.core.retrieve(response_id)

    def cancel(self, response_id: str) -> Response:
        return self.core.cancel(response_id)


class ModelsResource:
    def __init__(self, core: RuntimeClient) -> None:
        self.core = core

    def list(self) -> dict[str, Any]:
        return self.core.list_models()


class RuntimeResource:
    """Runtime control plane and native stage-extension surface.

    The methods under ``client.runtime`` are intentionally outside the OpenAI API
    schema. They let runtime plugins import weights and execute opaque encrypted
    stage frames while ordinary application code continues to use
    ``client.responses.create(...)``.
    """

    def __init__(self, core: RuntimeClient) -> None:
        self.core = core

    def preprocess(
        self,
        *,
        model: str,
        correlations: int = 64,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.core.preprocess(model, correlations, stages=stages)

    def capabilities(self) -> dict[str, Any]:
        response = self.core.http.get("/v1/runtime/capabilities", headers=self.core.headers)
        _raise(response)
        return response.json()

    def engines(self) -> dict[str, Any]:
        response = self.core.http.get("/v1/runtime/engines", headers=self.core.headers)
        _raise(response)
        return response.json()

    def inspect_model(self, **source: Any) -> dict[str, Any]:
        response = self.core.http.post(
            "/v1/runtime/models/inspect", headers=self.core.headers, json=source
        )
        _raise(response)
        return response.json()

    def load_model(self, *, engine: str, **source: Any) -> dict[str, Any]:
        response = self.core.http.post(
            "/v1/runtime/models/load",
            headers=self.core.headers,
            json={"engine": engine, **source},
        )
        _raise(response)
        return response.json()

    def unload_model(self, model: str) -> dict[str, Any]:
        response = self.core.http.delete(
            f"/v1/runtime/models/{model}", headers=self.core.headers
        )
        _raise(response)
        return response.json()

    def execute_stage(
        self,
        *,
        engine: str,
        model: str,
        stage: str,
        payloads: list[bytes],
    ) -> list[bytes]:
        if not payloads:
            raise ValueError("payloads cannot be empty")
        response = self.core.http.post(
            f"/v1/runtime/engines/{engine}/models/{model}/stages/{stage}",
            headers={**self.core.headers, "Content-Type": "application/octet-stream"},
            content=encode_length_prefixed(payloads),
        )
        _raise(response)
        results = list(iter_length_prefixed(response.content))
        if len(results) != len(payloads):
            raise ModelError("Runtime engine returned the wrong result count")
        return results

    @property
    def privacy_audit(self) -> PrivacyAudit:
        return self.core.audit


class OpenAI:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        model: str | None = None,
        session_transport: str | None = None,
        correlation_mode: str | None = None,
        preparation_base_url: str | None = None,
        preparation_api_key: str | None = None,
        correlation_prefetch: int | None = None,
        prepared_inventory_rows: int | None = None,
        background_inventory_refill: bool = True,
        token_cache_size: int | None = None,
        bundle_cache_mode: str | None = None,
        bundle_cache_dir: str | Path | None = None,
        tenseal_path: str | None = None,
        timeout: float | None = None,
        http_client: httpx.Client | None = None,
        preparation_http_client: httpx.Client | None = None,
        experiment: Experiment | ExperimentProfile | None = None,
    ) -> None:
        from pllm.settings import ClientSettings

        settings = ClientSettings.load().merged(
            api_key=api_key,
            base_url=base_url,
            model=default_model or model,
            transport=session_transport,
            correlation_mode=correlation_mode,
            preparation_base_url=preparation_base_url,
            preparation_api_key=preparation_api_key,
            correlation_prefetch=correlation_prefetch,
            prepared_inventory_rows=prepared_inventory_rows,
            token_cache_size=token_cache_size,
            bundle_cache_mode=bundle_cache_mode,
            bundle_cache_dir=bundle_cache_dir,
            timeout=timeout,
        )
        self._core = RuntimeClient(
            base_url=settings.base_url,
            api_key=settings.api_key,
            default_model=settings.model,
            session_transport=settings.transport,
            correlation_mode=settings.correlation_mode,
            preparation_base_url=settings.preparation_base_url,
            preparation_api_key=settings.preparation_api_key,
            correlation_prefetch=settings.correlation_prefetch,
            prepared_inventory_rows=settings.prepared_inventory_rows,
            background_inventory_refill=background_inventory_refill,
            token_cache_size=settings.token_cache_size,
            bundle_cache_mode=settings.bundle_cache_mode,
            bundle_cache_dir=settings.bundle_cache_dir,
            tenseal_path=tenseal_path,
            timeout=settings.timeout,
            http_client=http_client,
            preparation_http_client=preparation_http_client,
            experiment=experiment,
        )
        self.responses = ResponsesResource(self._core)
        self.models = ModelsResource(self._core)
        self.runtime = RuntimeResource(self._core)

    @property
    def privacy_audit(self) -> PrivacyAudit:
        return self._core.audit

    def preprocess(
        self,
        model: str | None = None,
        *,
        count: int | None = None,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        model_id = model or self._core.default_model
        if model_id is None:
            raise ValueError("model is required")
        target = count if count is not None else self._core.prepared_inventory_rows
        return self._core.preprocess(model_id, count=target, stages=stages)

    def prepared_inventory_status(self, model: str | None = None) -> dict[str, Any]:
        model_id = model or self._core.default_model
        if model_id is None:
            raise ValueError("model is required")
        return self._core.prepared_inventory_status(model_id)

    def prepared_rows_for_response(
        self,
        input: str,
        max_output_tokens: int,
        *,
        model: str | None = None,
        instructions: str | None = None,
    ) -> int:
        model_id = model or self._core.default_model
        if model_id is None:
            raise ValueError("model is required")
        return self._core.prepared_rows_for_response(
            model_id,
            input,
            max_output_tokens,
            instructions=instructions,
        )

    def close(self) -> None:
        self._core.close()

    def __enter__(self) -> "OpenAI":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AsyncResponsesResource:
    def __init__(self, resource: ResponsesResource) -> None:
        self.resource = resource

    async def create(self, **kwargs: Any):
        if kwargs.get("stream"):
            stream = await asyncio.to_thread(self.resource.create, **kwargs)
            assert isinstance(stream, ResponseStream)

            async def iterate() -> AsyncIterator[ResponseEvent]:
                sentinel = object()
                try:
                    while True:
                        event = await asyncio.to_thread(_next_or, stream, sentinel)
                        if event is sentinel:
                            break
                        yield event
                finally:
                    await asyncio.to_thread(stream.close)

            return iterate()
        return await asyncio.to_thread(self.resource.create, **kwargs)

    async def retrieve(self, response_id: str) -> Response:
        return await asyncio.to_thread(self.resource.retrieve, response_id)

    async def cancel(self, response_id: str) -> Response:
        return await asyncio.to_thread(self.resource.cancel, response_id)


class AsyncModelsResource:
    def __init__(self, resource: ModelsResource) -> None:
        self.resource = resource

    async def list(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.resource.list)


class AsyncRuntimeResource:
    def __init__(self, resource: RuntimeResource) -> None:
        self.resource = resource

    async def preprocess(
        self,
        *,
        model: str,
        correlations: int = 64,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.resource.preprocess,
            model=model,
            correlations=correlations,
            stages=stages,
        )

    async def capabilities(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.resource.capabilities)

    async def engines(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.resource.engines)

    async def inspect_model(self, **source: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.resource.inspect_model, **source)

    async def load_model(self, *, engine: str, **source: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.resource.load_model, engine=engine, **source)

    async def unload_model(self, model: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.resource.unload_model, model)

    async def execute_stage(
        self,
        *,
        engine: str,
        model: str,
        stage: str,
        payloads: list[bytes],
    ) -> list[bytes]:
        return await asyncio.to_thread(
            self.resource.execute_stage,
            engine=engine,
            model=model,
            stage=stage,
            payloads=payloads,
        )

    @property
    def privacy_audit(self) -> PrivacyAudit:
        return self.resource.privacy_audit


class AsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.sync = OpenAI(**kwargs)
        self.responses = AsyncResponsesResource(self.sync.responses)
        self.models = AsyncModelsResource(self.sync.models)
        self.runtime = AsyncRuntimeResource(self.sync.runtime)

    @property
    def privacy_audit(self) -> PrivacyAudit:
        return self.sync.privacy_audit

    async def preprocess(
        self,
        model: str | None = None,
        *,
        count: int | None = None,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.sync.preprocess,
            model,
            count=count,
            stages=stages,
        )

    async def prepared_rows_for_response(
        self,
        input: str,
        max_output_tokens: int,
        *,
        model: str | None = None,
        instructions: str | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.sync.prepared_rows_for_response,
            input,
            max_output_tokens,
            model=model,
            instructions=instructions,
        )

    async def close(self) -> None:
        await asyncio.to_thread(self.sync.close)

    async def __aenter__(self) -> "AsyncOpenAI":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


def _next_or(iterator: Iterator[Any], sentinel: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel


def _raise(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        body = response.json()
        detail = body.get("detail", body)
        if isinstance(detail, dict) and "error" in detail:
            message = str(detail["error"].get("message", detail))
        else:
            message = str(detail)
    except Exception:
        body = response.text
        message = response.text
    raise ProtocolError(message, response.status_code, body)
