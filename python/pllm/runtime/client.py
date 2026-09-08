from __future__ import annotations

import asyncio
import base64
import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict, deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar
from urllib.parse import urlparse, urlunparse

import httpx
import msgpack
import numpy as np

from .secure_random import FieldRandom

from .he_runtime import (
    BFVCorrelationClient,
    CorrelationPool,
    HEModelError,
    MaskCorrelation,
    MaskedBigramClientModel,
    MaskedBigramClientSession,
    MaskedLinearRequest,
    MaskedLinearResponse,
    centered_mod,
    he_worker_threads,
)
from .protocol import (
    HEEnvelope,
    encode_length_prefixed,
    iter_length_prefixed,
    pack_envelope,
    unpack_envelope,
)
from .stage_protocol import (
    BlindedStageCorrelation,
    BlindedStageRequest,
    BlindedStageResponse,
    DirectFHEStageRequest,
    DirectFHEStageResponse,
    StageCorrelation,
    blinded_correlation_from_wire,
    correlation_from_wire,
)
from .tiled_bfv import TiledBFVClient
from .quantization import dequantize_matmul, quantize_activation_per_row
from .transformer_client import (
    ClientBundle,
    CorrelationInventory,
    MaskedTransformerClientRuntime,
    RemoteLinear,
    RuntimeSnapshot,
    StageClientStats,
    StageMetadata,
    TwoProviderRemoteLinear,
)
from .responses import normalize_input, prompt_text
from .security import derive_session_key
from .tokenizer import AlphabetTokenizer
from .types import Response, ResponseEvent, ResponseUsage, new_id

T = TypeVar("T")


class HEAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ResponseStream(Generic[T]):
    def __init__(self, iterator: Iterator[T], close: Callable[[], None] | None = None) -> None:
        self.iterator = iterator
        self._close = close

    def __iter__(self) -> "ResponseStream[T]":
        return self

    def __next__(self) -> T:
        return next(self.iterator)

    def close(self) -> None:
        if self._close is not None:
            self._close()
            self._close = None

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
    token_lookup_cache_hits: int = 0
    token_lookup_cache_misses: int = 0
    kv_continuation_hits: int = 0
    kv_continuation_misses: int = 0
    direct_fhe_upload_bytes: int = 0
    direct_fhe_download_bytes: int = 0
    direct_fhe_steps: int = 0
    direct_share_primary_upload_bytes: int = 0
    direct_share_primary_download_bytes: int = 0
    direct_share_secondary_upload_bytes: int = 0
    direct_share_secondary_download_bytes: int = 0
    direct_share_primary_server_ns: int = 0
    direct_share_secondary_server_ns: int = 0

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
            f"/v1/he/sessions/{self.session_id}/execute",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/vnd.openai.he+msgpack",
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
        path = f"{parsed.path.rstrip('/')}/v1/he/ws/{self.session_id}"
        url = urlunparse(
            ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, path, "", "", "")
        )
        self.socket = connect(
            url,
            subprotocols=["he-responses-v1"],
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            open_timeout=30,
            max_size=None,
        )

    def exchange(self, payload: bytes) -> bytes:
        if self.mode == "http":
            return self._http(payload)
        if self.mode not in {"websocket", "auto"}:
            raise ValueError(f"unsupported HE transport {self.mode!r}")
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
            raise HEModelError("invalid BFV stage correlation response")
        outputs = [
            int(self.ts.bfv_vector_from(self.context, item).decrypt()[0]) % self.plain_modulus
            for item in value["ciphertexts"]
        ]
        return np.asarray(outputs, dtype=np.uint32)


@dataclass(slots=True)
class _TransformerCryptoState:
    bundle: ClientBundle
    mode: str
    privacy_mode: str = "public"
    privacy_protocol: str = "masked_w4a4"
    queues: dict[str, deque[Any]] = field(default_factory=lambda: defaultdict(deque))
    locks: dict[str, threading.Lock] = field(default_factory=lambda: defaultdict(threading.Lock))
    rng: FieldRandom = field(default_factory=FieldRandom)
    bfv_clients: dict[int, _BFVStageClient] = field(default_factory=dict)
    tiled_bfv_clients: dict[int, TiledBFVClient] = field(default_factory=dict)
    context_ids: dict[int, str] = field(default_factory=dict)
    tiled_context_lock: threading.Lock = field(default_factory=threading.Lock)
    blinded_owner_id: str = field(default_factory=lambda: new_id("owner"))
    token_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict)
    token_cache_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(slots=True)
class _TransformerConversationState:
    model_id: str
    token_ids: list[int]
    rendered_context: str
    snapshot: RuntimeSnapshot
    next_logits: np.ndarray


class _TransformerCorrelationProvider:
    def __init__(
        self,
        *,
        client: "HEClientCore",
        session_id: str,
        state: _TransformerCryptoState,
        prefetch: int,
    ) -> None:
        self.owner = client
        self.session_id = session_id
        self.state = state
        self.prefetch = max(1, int(prefetch))
        self.model_id = state.bundle.model_id

    def take_many(self, stage: StageMetadata, count: int) -> list[StageCorrelation]:
        with self.state.locks[stage.id]:
            queue = self.state.queues[stage.id]
            missing = count - len(queue)
            if missing > 0:
                queue.extend(self._create(stage, max(missing, self.prefetch)))
            if len(queue) < count:
                raise HEModelError(f"correlation inventory exhausted for {stage.id}")
            values = [queue.popleft() for _ in range(count)]
            for item in values:
                if item.consumed:
                    raise HEModelError("correlation reuse detected")
                item.consumed = True
            return values

    def _create(self, stage: StageMetadata, count: int) -> list[StageCorrelation]:
        if self.state.mode == "local-test":
            response = self.owner.http.post(
                f"/v1/he/sessions/{self.session_id}/correlations/local-test",
                headers=self.owner.headers,
                json={"count": count, "stage_id": stage.id, "ring": stage.ring},
            )
            _raise(response)
            value = msgpack.unpackb(response.content, raw=False, strict_map_key=False)
            rows = [correlation_from_wire(item) for item in value["items"]]
            self.owner.audit.correlation_count += len(rows)
            return rows
        if self.state.mode != "bfv":
            raise ValueError(f"unsupported transformer correlation mode {self.state.mode!r}")
        modulus = int(stage.modulus)
        with self.state.tiled_context_lock:
            bfv = self.state.tiled_bfv_clients.get(modulus)
            context_id = self.state.context_ids.get(modulus)
            if bfv is None or context_id is None:
                bfv = TiledBFVClient(
                    stage.in_features,
                    stage.out_features,
                    plain_modulus=modulus,
                    threads=he_worker_threads(),
                    tenseal_path=self.owner.tenseal_path,
                )
                context_id = new_id(f"ctx{modulus}")
                register = self.owner.http.put(
                    f"/v1/he/sessions/{self.session_id}/contexts/{context_id}",
                    headers={
                        **self.owner.headers,
                        "Content-Type": "application/octet-stream",
                    },
                    content=bfv.public_context,
                )
                _raise(register)
                self.state.tiled_bfv_clients[modulus] = bfv
                self.state.context_ids[modulus] = context_id
                self.owner.audit.public_context_bytes += len(bfv.public_context)
        bfv = bfv.for_shape(stage.in_features, stage.out_features)
        masks = [
            self.state.rng.integers(0, modulus, size=stage.in_features, dtype=np.uint32)
            for _ in range(count)
        ]
        encrypted = bfv.encrypt_many(np.stack(masks))
        group_sizes = bfv.group_sizes(len(masks))
        upload = encode_length_prefixed(encrypted)
        self.owner.audit.encrypted_correlation_upload_bytes += len(upload)
        response = self.owner.http.post(
            f"/v1/he/sessions/{self.session_id}/correlations/bfv/batch",
            headers={
                **self.owner.headers,
                "Content-Type": "application/octet-stream",
                "X-HE-Stage-ID": stage.id,
                "X-HE-Context-ID": context_id,
            },
            content=upload,
        )
        _raise(response)
        self.owner.audit.encrypted_correlation_download_bytes += len(response.content)
        payloads = list(iter_length_prefixed(response.content))
        if len(payloads) != len(group_sizes):
            raise HEModelError("BFV correlation batch returned the wrong group count")
        transformed_rows = bfv.decrypt_many(payloads, group_sizes)
        rows: list[StageCorrelation] = []
        for mask, transformed in zip(masks, transformed_rows, strict=True):
            rows.append(
                StageCorrelation(
                    id=new_id("corr"),
                    stage_id=stage.id,
                    ring="prime",
                    modulus=modulus,
                    mask=mask,
                    transformed_mask=transformed,
                )
            )
        self.owner.audit.correlation_count += len(rows)
        return rows


class _BlindedCorrelationProvider:
    """Client inventory for output-blinded proprietary correlations."""

    def __init__(
        self,
        *,
        client: "HEClientCore",
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
                raise HEModelError(f"proprietary correlation inventory exhausted for {stage.id}")
            values = [queue.popleft() for _ in range(count)]
            output: list[BlindedStageCorrelation] = []
            for item in values:
                if not isinstance(item, BlindedStageCorrelation):
                    raise HEModelError("mixed correlation protocols in proprietary inventory")
                if item.consumed:
                    raise HEModelError("proprietary correlation reuse detected")
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
            f"/v1/he/sessions/{self.session_id}/contexts/{context_id}",
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
                f"/v1/he/sessions/{self.session_id}/correlations/proprietary/local-test",
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
            f"/v1/he/sessions/{self.session_id}/correlations/proprietary/bfv/batch",
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
            raise HEModelError("blinded BFV correlation batch returned the wrong item count")
        rows: list[BlindedStageCorrelation] = []
        for mask, payload in zip(masks, payloads, strict=True):
            envelope = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            correlation_id = str(envelope.get("correlation_id", ""))
            if not correlation_id:
                raise HEModelError("blinded BFV correlation is missing its id")
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
            raise HEModelError(f"unknown stage {stage_id!r}")
        value = np.asarray(activation, dtype=np.float32)
        if value.shape[-1] != stage.in_features:
            raise HEModelError(
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
            raise HEModelError("proprietary correlation arithmetic profile mismatch")
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
            raise HEModelError("blinded stage exchange returned the wrong result count")
        response = BlindedStageResponse.unpack(response_payloads[0])
        if (
            response.stage_id != stage_id
            or response.correlation_ids != request.correlation_ids
            or response.modulus != modulus
        ):
            raise HEModelError("blinded stage response mismatch")
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
        owner: "HEClientCore",
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
            f"/v1/he/sessions/{self.session_id}/contexts/{context_id}",
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
            raise HEModelError(f"unknown stage {stage_id!r}")
        value = np.asarray(activation, dtype=np.float32)
        if value.shape[-1] != stage.in_features:
            raise HEModelError(
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
            raise HEModelError("direct-FHE exchange returned the wrong result count")
        self.owner.audit.direct_fhe_download_bytes += len(result_payloads[0])
        response = DirectFHEStageResponse.unpack(result_payloads[0])
        if response.stage_id != stage_id or len(response.output_rows) != quantized.rows:
            raise HEModelError("direct-FHE stage response mismatch")
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
        client: "HEClientCore",
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
                f"/v1/he/sessions/{session_id}/contexts/{self.state.context_id}",
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
                f"/v1/he/sessions/{self.session_id}/correlations/local-test",
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
                f"/v1/he/sessions/{self.session_id}/correlations/bfv",
                headers={**self.owner.headers, "Content-Type": "application/octet-stream"},
                content=encrypted,
            )
            _raise(response)
            self.owner.audit.encrypted_correlation_download_bytes += len(response.content)
            transformed = self.state.bfv.decrypt_transformed(response.content)
            self.state.pool.put(MaskCorrelation(new_id("corr"), mask, transformed))
            self.owner.audit.correlation_count += 1


class HEClientCore:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        default_model: str | None = None,
        he_transport: str = "http",
        correlation_mode: str = "bfv",
        execution_strategy: str = "bfv",
        secondary_base_url: str | None = None,
        secondary_api_key: str | None = None,
        correlation_prefetch: int = 4,
        token_cache_size: int = 512,
        tenseal_path: str | None = None,
        timeout: float = 300.0,
        http_client: httpx.Client | None = None,
        secondary_http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.he_transport = he_transport
        self.correlation_mode = correlation_mode
        if execution_strategy not in {"bfv", "two-provider"}:
            raise ValueError("execution_strategy must be 'bfv' or 'two-provider'")
        self.execution_strategy = execution_strategy
        self.correlation_prefetch = correlation_prefetch
        self.token_cache_size = max(0, int(token_cache_size))
        self.tenseal_path = tenseal_path
        self._owns_http = http_client is None
        self.http = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._owns_secondary_http = secondary_http_client is None and secondary_base_url is not None
        self.secondary_http = secondary_http_client
        if self.secondary_http is None and secondary_base_url is not None:
            self.secondary_http = httpx.Client(
                base_url=secondary_base_url.rstrip("/"), timeout=timeout
            )
        if execution_strategy == "two-provider" and self.secondary_http is None:
            raise ValueError("two-provider execution requires secondary_base_url")
        if execution_strategy == "two-provider":
            assert self.secondary_http is not None
            provider_urls = (self.http.base_url, self.secondary_http.base_url)
            if provider_urls[0] == provider_urls[1]:
                raise ValueError("two-provider execution requires distinct provider origins")
            for provider_url in provider_urls:
                if provider_url.scheme != "https" and provider_url.host not in {
                    "127.0.0.1",
                    "localhost",
                    "::1",
                }:
                    raise ValueError("two-provider execution requires HTTPS except on loopback")
        if execution_strategy == "two-provider":
            assert self.secondary_http is not None
            provider_urls = (self.http.base_url, self.secondary_http.base_url)
            if provider_urls[0] == provider_urls[1]:
                raise ValueError("two-provider execution requires distinct provider origins")
            for provider_url in provider_urls:
                if provider_url.scheme != "https" and provider_url.host not in {
                    "127.0.0.1",
                    "localhost",
                    "::1",
                }:
                    raise ValueError("two-provider execution requires HTTPS except on loopback")
        self.secondary_headers = {
            "Authorization": f"Bearer {secondary_api_key or api_key}"
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
                "/v1/he/sessions", headers=self.headers, json=session_body
            )
            _raise(session_response)
            session_value = session_response.json()
            he = session_value["manifest"]["he"]
            model = MaskedBigramClientModel(
                model_id,
                AlphabetTokenizer(str(he["alphabet"])),
                int(he["modulus"]),
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
                raise HEModelError("server model manifest changed during client lifetime")
            source = _CorrelationSource(
                client=self,
                session_id=str(session_value["id"]),
                state=state,
                prefetch=self.correlation_prefetch,
            )
            return session_value, model, source

    def _model_manifest(self, model_id: str) -> dict[str, Any]:
        cached = self._model_manifests.get(model_id)
        if cached is not None:
            return cached
        response = self.http.get(f"/v1/he/models/{model_id}", headers=self.headers)
        _raise(response)
        value = response.json()
        self._model_manifests[model_id] = value
        return value

    def _open_transformer_session(
        self,
        model_id: str,
        *,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], _TransformerCryptoState, Any | None]:
        with self._transformer_state_lock:
            state = self._transformer_states.get(model_id)
            if state is None:
                bundle_response = self.http.get(
                    f"/v1/he/models/{model_id}/client-bundle",
                    headers=self.headers,
                )
                _raise(bundle_response)
                bundle = ClientBundle.unpack(bundle_response.content)
                state = _TransformerCryptoState(
                    bundle=bundle,
                    mode=self.correlation_mode,
                    privacy_mode=str(bundle.privacy.get("mode", "public")),
                    privacy_protocol=str(bundle.privacy.get("protocol", "masked_w4a4")),
                )
                self._transformer_states[model_id] = state

            has_secondary = False
            if self.execution_strategy == "two-provider":
                if state.privacy_mode != "public":
                    raise HEAPIError("two-provider execution requires public model weights", 400)
                if self.secondary_http is None:
                    raise HEAPIError("two-provider execution requires a secondary provider", 400)
                model_response = self.secondary_http.get(
                    "/v1/models",
                    headers=self.secondary_headers,
                )
                _raise(model_response)
                secondary_models = [
                    item
                    for item in model_response.json().get("data", [])
                    if item.get("id") == model_id
                ]
                if len(secondary_models) != 1:
                    raise HEAPIError("secondary provider does not serve the requested model", 404)
                secondary_he = secondary_models[0].get("he") or {}
                remote_stages = [
                    stage
                    for stage in state.bundle.stages.values()
                    if stage.client_weight is None and stage.id != "embed_tokens"
                ]
                if not remote_stages or any(not stage.weight_digest for stage in remote_stages):
                    raise HEAPIError("provider bundle lacks stage weight commitments", 409)
                if (
                    secondary_he.get("privacy_mode") != "public"
                    or secondary_he.get("body_fingerprint")
                    != state.bundle.privacy.get("body_fingerprint")
                    or secondary_he.get("architecture") != state.bundle.manifest.get("architecture")
                    or secondary_he.get("stage_count")
                    != len(state.bundle.manifest.get("stages", []))
                ):
                    raise HEAPIError("provider model commitments do not match", 409)
                has_secondary = True

            session_body: dict[str, Any] = {
                "model": model_id,
                "max_output_tokens": max_output_tokens,
            }
            if state.context_ids:
                session_body["context_ids"] = list(state.context_ids.values())
            session_response = self.http.post(
                "/v1/he/sessions",
                headers=self.headers,
                json=session_body,
            )
            _raise(session_response)
            session_value = session_response.json()
            session_id = str(session_value["id"])

            if has_secondary:
                assert self.secondary_http is not None
                secondary_response = self.secondary_http.post(
                    "/v1/he/sessions",
                    headers=self.secondary_headers,
                    json={"model": model_id, "max_output_tokens": max_output_tokens},
                )
                _raise(secondary_response)
                provider = str(secondary_response.json()["id"])
            elif state.privacy_mode == "public":
                if state.mode not in {"bfv", "local-test"}:
                    raise ValueError(f"unsupported transformer correlation mode {state.mode!r}")
                provider: Any | None = _TransformerCorrelationProvider(
                    client=self,
                    session_id=session_id,
                    state=state,
                    prefetch=self.correlation_prefetch,
                )
            elif state.privacy_mode == "proprietary":
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
                    raise HEModelError(
                        f"unsupported proprietary protocol {state.privacy_protocol!r}"
                    )
            else:
                raise HEModelError(f"unsupported server privacy mode {state.privacy_mode!r}")
            return session_value, state, provider

    def preprocess(
        self,
        model_id: str,
        count: int,
        *,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        descriptor = self._model_manifest(model_id)
        target = max(0, int(count))
        if descriptor.get("metadata", {}).get("client_runtime") in {
            "masked_transformer_v1",
            "direct_fhe_transformer_v1",
            "blinded_ole_transformer_v1",
            "guarded_blinded_transformer_v1",
        }:
            session, state, provider = self._open_transformer_session(model_id, max_output_tokens=1)
            if state.privacy_mode == "proprietary" and state.privacy_protocol == "direct_bfv_w4a4":
                complete = self.http.post(
                    f"/v1/he/sessions/{session['id']}/complete",
                    headers=self.headers,
                    json={"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
                )
                _raise(complete)
                return {
                    "object": "he.preprocessing_result",
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
                f"/v1/he/sessions/{session['id']}/complete",
                headers=self.headers,
                json={"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
            )
            _raise(complete)
            return {
                "object": "he.preprocessing_result",
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
            f"/v1/he/sessions/{session['id']}/complete",
            headers=self.headers,
            json={"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}},
        )
        _raise(complete)
        return {
            "object": "he.preprocessing_result",
            "model": model_id,
            "generated": available - before,
            "available": available,
            "context_reused": self.audit.public_context_bytes > 0,
        }

    def close(self) -> None:
        self._provider_executor.shutdown(wait=True, cancel_futures=True)
        if self._owns_secondary_http and self.secondary_http is not None:
            self.secondary_http.close()
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
            raise HEAPIError("response stream ended without response.completed")
        return final

    def events(self, body: dict[str, Any]) -> Iterator[ResponseEvent]:
        model_id = str(body.get("model") or self.default_model or "")
        if not model_id:
            models = self.list_models().get("data", [])
            private = [
                item.get("id")
                for item in models
                if item.get("he", {}).get("privacy_mode")
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
                raise HEAPIError("unknown previous_response_id in client-private cache", 404)
            previous_history = self.histories[str(previous_id)]
        descriptor = self._model_manifest(model_id)
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
        he = session_value["manifest"]["he"]
        channel = _Channel(self.http, self.base_url, self.api_key, session_id, self.he_transport)
        max_tokens = max(
            1, min(int(body.get("max_output_tokens") or 64), int(he.get("max_output_tokens", 4096)))
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
                envelope = HEEnvelope.create(
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
                    raise HEModelError("mismatched HE result frame")
                decoded = MaskedLinearResponse.unpack(result.payload)
                if decoded.correlation_id != correlation.id:
                    raise HEModelError("correlation response mismatch")
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
                f"/v1/he/sessions/{session_id}/complete",
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
        session_value, state, provider = self._open_transformer_session(
            model_id,
            max_output_tokens=int(body.get("max_output_tokens") or 64),
        )
        session_id = str(session_value["id"])
        response_id = str(session_value["response_id"])
        current_messages = normalize_input(
            body.get("input", ""), instructions=body.get("instructions")
        )
        previous_id = body.get("previous_response_id")
        structured: list[dict[str, str]] = []
        if previous_id:
            structured.extend(self.message_histories.get(str(previous_id), []))
        structured.extend({"role": row.role, "content": row.text} for row in current_messages)
        rendered = state.bundle.render_prompt(structured, add_generation_prompt=True)
        channel = _Channel(self.http, self.base_url, self.api_key, session_id, self.he_transport)
        key = derive_session_key(self.api_key, session_id)
        sequence = 0

        def exchange(stage_id: str, payloads: list[bytes]) -> list[bytes]:
            nonlocal sequence
            if len(payloads) > 1:
                upload = encode_length_prefixed(payloads)
                if state.privacy_protocol != "direct_bfv_w4a4":
                    self.audit.masked_online_upload_bytes += len(upload)
                response = self.http.post(
                    f"/v1/he/sessions/{session_id}/stages/{stage_id}",
                    headers={**self.headers, "Content-Type": "application/octet-stream"},
                    content=upload,
                )
                _raise(response)
                if state.privacy_protocol != "direct_bfv_w4a4":
                    self.audit.masked_online_download_bytes += len(response.content)
                results = list(iter_length_prefixed(response.content))
                self.audit.online_steps += len(payloads)
                return results
            envelope = HEEnvelope.create(
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
                raise HEModelError("unexpected transformer stage result")
            self.audit.online_steps += 1
            return [result.payload]

        def direct_exchange(client, headers, target_session_id, stage_id, payloads, secondary):
            upload = encode_length_prefixed(payloads)
            response = client.post(
                f"/v1/he/sessions/{target_session_id}/stages/{stage_id}",
                headers={**headers, "Content-Type": "application/octet-stream"},
                content=upload,
            )
            _raise(response)
            self.audit.masked_online_upload_bytes += len(upload)
            self.audit.masked_online_download_bytes += len(response.content)
            if secondary:
                self.audit.direct_share_secondary_upload_bytes += len(upload)
                self.audit.direct_share_secondary_download_bytes += len(response.content)
            else:
                self.audit.direct_share_primary_upload_bytes += len(upload)
                self.audit.direct_share_primary_download_bytes += len(response.content)
                self.audit.online_steps += len(payloads)
            return list(iter_length_prefixed(response.content))

        secondary_session_id = provider if self.execution_strategy == "two-provider" else None
        if secondary_session_id is not None:
            assert isinstance(secondary_session_id, str)
            assert self.secondary_http is not None
            remote = TwoProviderRemoteLinear(
                model_id=model_id,
                stages=state.bundle.stages,
                primary=lambda stage_id, payloads: direct_exchange(
                    self.http, self.headers, session_id, stage_id, payloads, False
                ),
                secondary=lambda stage_id, payloads: direct_exchange(
                    self.secondary_http,
                    self.secondary_headers,
                    secondary_session_id,
                    stage_id,
                    payloads,
                    True,
                ),
                executor=self._provider_executor,
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
                raise HEModelError("blinded proprietary mode requires its correlation provider")
            remote = _BlindedRemoteLinear(
                stages=state.bundle.stages,
                correlations=provider,
                session_id=session_id,
                state=state,
                exchange=exchange,
            )
        else:
            if provider is None:
                raise HEModelError("public mode requires a correlation provider")
            remote = RemoteLinear(state.bundle.stages, provider, exchange)
        runtime = MaskedTransformerClientRuntime(
            state.bundle,
            remote,
            token_cache=state.token_cache,
            token_cache_size=self.token_cache_size,
            token_cache_lock=state.token_cache_lock,
        )
        max_tokens = max(
            1,
            min(
                int(body.get("max_output_tokens") or 64),
                int(descriptor.get("context_length", 4096)),
            ),
        )
        temperature = float(body.get("temperature") or 0.0)
        top_p = body.get("top_p")
        top_p = None if top_p is None else float(top_p)
        message_id = new_id("msg")
        event_sequence = 0
        output_ids: list[int] = []
        output_chunks: list[str] = []
        previous_text = ""

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
                    if candidate is not None and candidate.model_id == model_id:
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
                logits = runtime.forward_ids(suffix)[-1] if suffix else prior.next_logits.copy()
                input_ids = full_input_ids
                caches = runtime.caches
            else:
                input_ids, logits, caches = runtime.prepare_ids(full_input_ids)
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
                logits, caches = runtime.decode_step(token, caches, len(input_ids) + step)

            if isinstance(remote, TwoProviderRemoteLinear):
                self.audit.direct_share_primary_server_ns += remote.stats.primary_server_ns
                self.audit.direct_share_secondary_server_ns += remote.stats.secondary_server_ns
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
                    token_ids=[*input_ids, *output_ids],
                    rendered_context=rendered + text,
                    snapshot=runtime.snapshot(),
                    next_logits=np.asarray(logits, dtype=np.float32).copy(),
                )
            complete = self.http.post(
                f"/v1/he/sessions/{session_id}/complete",
                headers=self.headers,
                json={"usage": usage.to_dict()},
            )
            _raise(complete)
            if secondary_session_id is not None:
                assert self.secondary_http is not None
                secondary_complete = self.secondary_http.post(
                    f"/v1/he/sessions/{secondary_session_id}/complete",
                    headers=self.secondary_headers,
                    json={"usage": usage.to_dict()},
                )
                _raise(secondary_complete)
            yield ResponseEvent.from_dict(
                {
                    "type": "response.completed",
                    "sequence_number": event_sequence,
                    "response": final.to_dict(),
                }
            )
        finally:
            channel.close()


class ResponsesResource:
    def __init__(self, core: HEClientCore) -> None:
        self.core = core

    def create(self, **kwargs: Any) -> Response | ResponseStream[ResponseEvent]:
        return self.core.create(kwargs)

    def retrieve(self, response_id: str) -> Response:
        return self.core.retrieve(response_id)

    def cancel(self, response_id: str) -> Response:
        return self.core.cancel(response_id)


class ModelsResource:
    def __init__(self, core: HEClientCore) -> None:
        self.core = core

    def list(self) -> dict[str, Any]:
        return self.core.list_models()


class HEExtensionsResource:
    """HE control plane and native stage-extension surface.

    The methods under ``client.he`` are intentionally outside the OpenAI API
    schema. They let runtime plugins import weights and execute opaque encrypted
    stage frames while ordinary application code continues to use
    ``client.responses.create(...)``.
    """

    def __init__(self, core: HEClientCore) -> None:
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
        response = self.core.http.get("/v1/he/capabilities", headers=self.core.headers)
        _raise(response)
        return response.json()

    def engines(self) -> dict[str, Any]:
        response = self.core.http.get("/v1/he/engines", headers=self.core.headers)
        _raise(response)
        return response.json()

    def inspect_model(self, **source: Any) -> dict[str, Any]:
        response = self.core.http.post(
            "/v1/he/models/inspect", headers=self.core.headers, json=source
        )
        _raise(response)
        return response.json()

    def load_model(self, *, engine: str, **source: Any) -> dict[str, Any]:
        response = self.core.http.post(
            "/v1/he/models/load",
            headers=self.core.headers,
            json={"engine": engine, **source},
        )
        _raise(response)
        return response.json()

    def unload_model(self, model: str) -> dict[str, Any]:
        response = self.core.http.delete(f"/v1/he/models/{model}", headers=self.core.headers)
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
            f"/v1/he/engines/{engine}/models/{model}/stages/{stage}",
            headers={**self.core.headers, "Content-Type": "application/octet-stream"},
            content=encode_length_prefixed(payloads),
        )
        _raise(response)
        results = list(iter_length_prefixed(response.content))
        if len(results) != len(payloads):
            raise HEModelError("HE engine returned the wrong result count")
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
        he_transport: str | None = None,
        correlation_mode: str | None = None,
        execution_strategy: str | None = None,
        secondary_base_url: str | None = None,
        secondary_api_key: str | None = None,
        correlation_prefetch: int | None = None,
        token_cache_size: int | None = None,
        tenseal_path: str | None = None,
        timeout: float | None = None,
        http_client: httpx.Client | None = None,
        secondary_http_client: httpx.Client | None = None,
    ) -> None:
        from pllm.settings import ClientSettings

        settings = ClientSettings.load().merged(
            api_key=api_key,
            base_url=base_url,
            model=default_model or model,
            transport=he_transport,
            correlation_mode=correlation_mode,
            execution_strategy=execution_strategy,
            secondary_base_url=secondary_base_url,
            secondary_api_key=secondary_api_key,
            correlation_prefetch=correlation_prefetch,
            token_cache_size=token_cache_size,
            timeout=timeout,
        )
        self._core = HEClientCore(
            base_url=settings.base_url,
            api_key=settings.api_key,
            default_model=settings.model,
            he_transport=settings.transport,
            correlation_mode=settings.correlation_mode,
            execution_strategy=settings.execution_strategy,
            secondary_base_url=settings.secondary_base_url,
            secondary_api_key=settings.secondary_api_key,
            correlation_prefetch=settings.correlation_prefetch,
            token_cache_size=settings.token_cache_size,
            tenseal_path=tenseal_path,
            timeout=settings.timeout,
            http_client=http_client,
            secondary_http_client=secondary_http_client,
        )
        self.responses = ResponsesResource(self._core)
        self.models = ModelsResource(self._core)
        self.he = HEExtensionsResource(self._core)

    @property
    def privacy_audit(self) -> PrivacyAudit:
        return self._core.audit

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
                while True:
                    event = await asyncio.to_thread(_next_or, stream, sentinel)
                    if event is sentinel:
                        break
                    yield event

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


class AsyncHEExtensionsResource:
    def __init__(self, resource: HEExtensionsResource) -> None:
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
        self.he = AsyncHEExtensionsResource(self.sync.he)

    @property
    def privacy_audit(self) -> PrivacyAudit:
        return self.sync.privacy_audit

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
    raise HEAPIError(message, response.status_code, body)
