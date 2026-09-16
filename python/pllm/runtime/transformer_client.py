from __future__ import annotations

import math
import secrets
import threading
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, cast

import msgpack
import numpy as np

from .native import CompiledMatrix, MaskedGEMM
from .preparation_protocol import (
    PreparationAck,
    PreparationRequest,
    SeededRingProfile,
    derive_online_attempt_id,
    expand_output_mask,
    expand_preparation_mask,
)
from .quantization import dequantize_matmul, quantize_activation_per_row, signed_qmax
from .protocol import ProtocolError
from .stage_protocol import (
    MaskedStageRequest,
    MaskedStageResponse,
    PreparedStageBatchRequest,
    PreparedStageBatchResponse,
    RingKind,
    StageCorrelation,
    unmask_stage_output,
)
from .telemetry import record_protocol_bytes, record_protocol_operation, start_protocol_span
from .tokenizer import AlphabetTokenizer, Tokenizer


class TransformerClientError(RuntimeError):
    pass


@dataclass(slots=True)
class PreparedStageRows:
    request: PreparationRequest
    input_mask: np.ndarray
    output_mask: np.ndarray


@dataclass(slots=True)
class PreparedInventoryLease:
    inventory_id: str
    stages: dict[str, PreparedStageRows]
    start: int
    rows: int
    _owner: "PreparedInventory" = field(repr=False)
    _offsets: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closed: bool = False

    @property
    def reservation_start(self) -> int:
        return self.start

    @property
    def reservation_rows(self) -> int:
        return self.rows

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            consumed = min(
                (self._offsets.get(stage_id, 0) for stage_id in self.stages),
                default=0,
            )
        self._owner._finish(self.rows, consumed)

    def take(self, stage_id: str, count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        stage = self.stages.get(stage_id)
        if stage is None or count <= 0:
            raise TransformerClientError("prepared inventory stage request is invalid")
        with self._lock:
            offset = self._offsets.get(stage_id, 0)
            if offset + count > self.rows:
                raise TransformerClientError(f"prepared inventory exhausted for {stage_id}")
            begin = self.start + offset
            end = begin + count
            self._offsets[stage_id] = offset + count
        attempts = [derive_online_attempt_id(stage.request, row) for row in range(begin, end)]
        return stage.input_mask[begin:end], stage.output_mask[begin:end], attempts


@dataclass(slots=True)
class PreparedInventory:
    id: str
    capacity: int
    stages: dict[str, PreparedStageRows]
    _reserved: int = 0
    _active: int = 0
    _burned: int = 0
    _consumed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def available(self) -> int:
        with self._lock:
            return self.capacity - self._reserved

    @property
    def claimed(self) -> int:
        with self._lock:
            return self._reserved

    def reserve(self, rows: int) -> PreparedInventoryLease:
        if rows <= 0:
            raise TransformerClientError("prepared inventory reservation must be positive")
        with self._lock:
            if self._reserved + rows > self.capacity:
                raise TransformerClientError("prepared inventory does not have enough rows")
            start = self._reserved
            self._reserved += rows
            self._active += rows
        return PreparedInventoryLease(self.id, self.stages, start, rows, self)

    def _finish(self, rows: int, consumed: int) -> None:
        with self._lock:
            self._active -= rows
            self._consumed += consumed
            self._burned += rows - consumed

    def status(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "id": self.id,
                "status": "ready",
                "capacity": self.capacity,
                "available": self.capacity - self._reserved,
                "reserved": self._active,
                "burned": self._burned,
                "consumed": self._consumed,
            }


@dataclass(frozen=True, slots=True)
class StageMetadata:
    id: str
    op: str
    in_features: int
    out_features: int
    weight_bits: int
    activation_bits: int
    ring: RingKind
    modulus: int
    wire_bits: int
    weight_scales: np.ndarray
    bias: np.ndarray | None = None
    role: str | None = None
    layer_index: int | None = None
    client_weight: np.ndarray | None = None
    weight_digest: str = ""
    client_weight_scales: np.ndarray | None = None
    client_weight_layout: str = "linear"
    client_aux_weight: np.ndarray | None = None
    client_aux_scales: np.ndarray | None = None
    seeded_profile: SeededRingProfile | None = None

    @property
    def scales(self) -> np.ndarray:
        return self.weight_scales


ClientStage = StageMetadata


@dataclass(slots=True)
class ClientBundle:
    model_id: str
    manifest: dict[str, Any]
    cfg: dict[str, Any]
    tokenizer_descriptor: dict[str, Any]
    stages: dict[str, StageMetadata]
    arrays: dict[str, np.ndarray]
    privacy: dict[str, Any]
    schema_version: int = 1
    _local_kernel: MaskedGEMM | None = field(default=None, init=False, repr=False)
    _local_matrices: dict[str, CompiledMatrix] = field(default_factory=dict, init=False, repr=False)
    _local_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def config(self) -> dict[str, Any]:
        return self.cfg

    @property
    def local_tensors(self) -> dict[str, np.ndarray]:
        return self.arrays

    @classmethod
    def unpack(cls, payload: bytes) -> "ClientBundle":
        try:
            value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
        except Exception as exc:
            raise TransformerClientError("invalid transformer client bundle") from exc
        if not isinstance(value, dict) or int(value.get("v", 0)) not in {1, 2}:
            raise TransformerClientError("unsupported transformer client bundle")
        version = int(value["v"])
        new_format = {"v", "manifest", "runtime", "tokenizer", "arrays", "stages", "privacy"}
        legacy_format = {
            "v",
            "runtime",
            "model",
            "manifest",
            "config",
            "tokenizer",
            "stages",
            "local_tensors",
            "privacy",
        }
        reference_format = legacy_format | {"client_weights"}
        keys = set(value)
        if version == 1 and keys == new_format:
            runtime_config = dict(value["runtime"])
            array_rows = value["arrays"]
            weight_rows: dict[str, Any] = {}
        elif (
            version == 1
            and keys == legacy_format
            and value.get("runtime")
            in {
                "masked_transformer",
                "direct_fhe_transformer",
                "blinded_ole_transformer",
                "guarded_blinded_transformer",
            }
        ):
            runtime_config = dict(value["config"])
            array_rows = value["local_tensors"]
            weight_rows = {}
        elif (
            version == 2
            and keys == reference_format
            and value.get("runtime")
            in {
                "masked_transformer",
                "direct_fhe_transformer",
                "blinded_ole_transformer",
                "guarded_blinded_transformer",
            }
        ):
            runtime_config = dict(value["config"])
            array_rows = value["local_tensors"]
            weight_rows = value["client_weights"]
        else:
            raise TransformerClientError("unsupported transformer client bundle")
        manifest = dict(value["manifest"])
        if str(value.get("model", manifest.get("id", ""))) != str(manifest.get("id", "")):
            raise TransformerClientError("client bundle model descriptor mismatch")
        privacy = dict(value["privacy"])
        client_weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for weight_id, weight_row in weight_rows.items():
            if not isinstance(weight_row, dict) or set(weight_row) != {
                "dtype",
                "shape",
                "data",
                "scales",
            }:
                raise TransformerClientError(f"invalid client weight {weight_id}")
            shape = tuple(int(item) for item in weight_row["shape"])
            if weight_row["dtype"] != "i1" or len(shape) != 2:
                raise TransformerClientError(f"invalid client weight shape for {weight_id}")
            matrix = np.frombuffer(weight_row["data"], dtype=np.int8).copy()
            scales = np.frombuffer(weight_row["scales"], dtype="<f4").copy()
            if matrix.size != int(np.prod(shape, dtype=np.int64)) or scales.shape != (shape[0],):
                raise TransformerClientError(f"invalid client weight length for {weight_id}")
            client_weights[str(weight_id)] = (matrix.reshape(shape), scales)
        stage_specs = {
            str(row["id"]): row for row in manifest.get("stages", []) if isinstance(row, dict)
        }
        stages: dict[str, StageMetadata] = {}
        for stage_id, row in value["stages"].items():
            scales = np.frombuffer(row["weight_scales"], dtype="<f4").copy()
            out_features = int(row["out_features"])
            if scales.shape != (out_features,):
                raise TransformerClientError(f"invalid weight scale count for {stage_id}")
            bias_payload = row.get("bias")
            bias = None if bias_payload is None else np.frombuffer(bias_payload, dtype="<f4").copy()
            if bias is not None and bias.shape != (out_features,):
                raise TransformerClientError(f"invalid bias count for {stage_id}")
            spec = stage_specs.get(stage_id, {})
            seeded_profile = None
            if "seeded_profile" in row:
                try:
                    seeded_profile = SeededRingProfile.from_dict(row["seeded_profile"])
                except ProtocolError as exc:
                    raise TransformerClientError(
                        f"invalid seeded ring profile for {stage_id}"
                    ) from exc
            client_weight = None
            client_weight_scales = None
            client_weight_layout = "linear"
            weight_row = row.get("client_weight")
            if weight_row is not None:
                if privacy.get("mode") != "public" or stage_id not in {"token_lookup", "lm_head"}:
                    raise TransformerClientError(
                        "client stage weights require public boundary stages"
                    )
                if not isinstance(weight_row, dict):
                    raise TransformerClientError(f"invalid client weight for {stage_id}")
                if version == 2:
                    if set(weight_row) != {"ref", "layout"}:
                        raise TransformerClientError(
                            f"invalid client weight reference for {stage_id}"
                        )
                    reference = client_weights.get(str(weight_row["ref"]))
                    if reference is None:
                        raise TransformerClientError(
                            f"unknown client weight reference for {stage_id}"
                        )
                    client_weight, client_weight_scales = reference
                    client_weight_layout = str(weight_row["layout"])
                    expected_shape = (
                        (int(row["in_features"]), min(out_features, client_weight.shape[1]))
                        if client_weight_layout == "embedding"
                        else (out_features, int(row["in_features"]))
                    )
                    if client_weight_layout not in {"linear", "embedding", "transposed_embedding"}:
                        raise TransformerClientError(f"invalid client weight layout for {stage_id}")
                    if (
                        client_weight_layout != "embedding"
                        and client_weight.shape != expected_shape
                    ):
                        raise TransformerClientError(f"invalid client weight shape for {stage_id}")
                    if client_weight_layout == "embedding" and (
                        client_weight.shape[0] != int(row["in_features"])
                        or client_weight.shape[1] > out_features
                    ):
                        raise TransformerClientError(f"invalid client weight shape for {stage_id}")
                else:
                    if set(weight_row) != {"dtype", "shape", "data"}:
                        raise TransformerClientError(f"invalid client weight for {stage_id}")
                    shape = tuple(int(item) for item in weight_row["shape"])
                    if weight_row["dtype"] != "i1" or shape != (
                        out_features,
                        int(row["in_features"]),
                    ):
                        raise TransformerClientError(f"invalid client weight shape for {stage_id}")
                    client_weight = np.frombuffer(weight_row["data"], dtype=np.int8)
                    if client_weight.size != int(np.prod(shape, dtype=np.int64)):
                        raise TransformerClientError(f"invalid client weight length for {stage_id}")
                    client_weight = client_weight.reshape(shape)
                    client_weight_scales = scales
                    if stage_id == "token_lookup":
                        client_weight_layout = "transposed_embedding"
            client_aux_weight = None
            client_aux_scales = None
            aux_row = row.get("client_aux_weight")
            if aux_row is not None:
                if version != 2 or stage_id != "token_lookup" or set(aux_row) != {"ref"}:
                    raise TransformerClientError(f"invalid auxiliary client weight for {stage_id}")
                reference = client_weights.get(str(aux_row["ref"]))
                if reference is None:
                    raise TransformerClientError(f"unknown auxiliary client weight for {stage_id}")
                client_aux_weight, client_aux_scales = reference
                if client_aux_weight.shape[1] != int(row["in_features"]):
                    raise TransformerClientError(
                        f"invalid auxiliary client weight shape for {stage_id}"
                    )
            local_width = 0
            if client_weight is not None:
                local_width = (
                    client_weight.shape[1]
                    if client_weight_layout == "embedding"
                    else client_weight.shape[0]
                )
            if client_aux_weight is not None:
                local_width += client_aux_weight.shape[0]
            if client_weight is not None and local_width != out_features:
                raise TransformerClientError(f"invalid local client weight width for {stage_id}")
            stages[stage_id] = StageMetadata(
                id=stage_id,
                op=str(row["op"]),
                in_features=int(row["in_features"]),
                out_features=out_features,
                weight_bits=int(row["weight_bits"]),
                activation_bits=int(row["activation_bits"]),
                ring=cast(RingKind, str(row.get("ring", "prime"))),
                modulus=int(row["modulus"]),
                wire_bits=int(row.get("wire_bits", 32)),
                weight_scales=scales,
                bias=bias,
                role=spec.get("role"),
                layer_index=spec.get("layer_index"),
                client_weight=client_weight,
                weight_digest=str(row.get("weight_digest", "")),
                client_weight_scales=client_weight_scales,
                client_weight_layout=client_weight_layout,
                client_aux_weight=client_aux_weight,
                client_aux_scales=client_aux_scales,
                seeded_profile=seeded_profile,
            )
        # Keep the legacy embedding stage name as a read-only alias. Round 8
        # fuses the token embedding and Gemma PLE table into ``token_lookup``,
        # while older clients and manifests refer to ``embed_tokens``.
        if "token_lookup" in stages and "embed_tokens" not in stages:
            stages["embed_tokens"] = stages["token_lookup"]

        arrays: dict[str, np.ndarray] = {}
        for key, row in array_rows.items():
            if row.get("dtype") != "f4":
                raise TransformerClientError(f"unsupported client tensor dtype for {key}")
            shape = tuple(int(item) for item in row["shape"])
            array = np.frombuffer(row["data"], dtype="<f4").copy()
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise TransformerClientError(f"client tensor byte length mismatch for {key}")
            arrays[str(key)] = array.reshape(shape)
        local_ids = {stage.id for stage in stages.values() if stage.client_weight is not None}
        advertised_ids = set(privacy.get("local_quantized_stages", []))
        if advertised_ids and advertised_ids != local_ids:
            raise TransformerClientError("local stage weight advertisement mismatch")
        return cls(
            model_id=str(manifest["id"]),
            manifest=manifest,
            cfg=runtime_config,
            tokenizer_descriptor=dict(value["tokenizer"]),
            stages=stages,
            arrays=arrays,
            privacy=privacy,
            schema_version=version,
        )

    def local_linear(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self.stages[stage_id]
        if (
            stage.client_weight is None
            or stage.client_weight_scales is None
            or stage.client_weight_layout != "linear"
        ):
            raise TransformerClientError(f"stage {stage_id!r} has no local weight")
        quantized = quantize_activation_per_row(activation, bits=stage.activation_bits)
        with self._local_lock:
            matrix = self._local_matrices.get(stage_id)
            if matrix is None:
                if self._local_kernel is None:
                    self._local_kernel = MaskedGEMM()
                matrix = self._local_kernel.compile(stage.client_weight)
                self._local_matrices[stage_id] = matrix
        integer = matrix.clear(quantized.values)
        result = dequantize_matmul(
            integer,
            quantized.scales,
            stage.client_weight_scales,
            output_shape=quantized.original_shape[:-1] + (stage.out_features,),
        )
        if stage.bias is not None:
            result = result + stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)

    def local_token_lookup(self, token_ids: np.ndarray) -> np.ndarray:
        stage = self.stages["token_lookup"]
        if stage.client_weight is None or stage.client_weight_scales is None:
            raise TransformerClientError("token lookup has no local weight")
        ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= stage.in_features):
            raise TransformerClientError("token id outside model vocabulary")
        if stage.client_weight_layout == "embedding":
            result = (
                stage.client_weight[ids].astype(np.float32) * stage.client_weight_scales[ids, None]
            )
        elif stage.client_weight_layout == "transposed_embedding":
            qmax = signed_qmax(stage.activation_bits)
            integer = stage.client_weight[:, ids].T.astype(np.int32) * qmax
            scales = np.full(ids.size, np.float32(1.0 / qmax), dtype=np.float32)
            result = dequantize_matmul(integer, scales, stage.client_weight_scales)
        else:
            raise TransformerClientError("token lookup has invalid local weight layout")
        if stage.client_aux_weight is not None:
            assert stage.client_aux_scales is not None
            qmax = signed_qmax(stage.activation_bits)
            integer = stage.client_aux_weight[:, ids].T.astype(np.int32) * qmax
            scales = np.full(ids.size, np.float32(1.0 / qmax), dtype=np.float32)
            auxiliary = dequantize_matmul(integer, scales, stage.client_aux_scales)
            result = np.concatenate((result, auxiliary), axis=-1)
        if stage.bias is not None:
            result = result + stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)

    def tensor(self, suffix: str, *, default: np.ndarray | None = None) -> np.ndarray:
        if suffix in self.arrays:
            return self.arrays[suffix]
        marker = suffix if suffix.startswith(".") else "." + suffix
        matches = [value for key, value in self.arrays.items() if key.endswith(marker)]
        if len(matches) == 1:
            return matches[0]
        if not matches and default is not None:
            return default
        if not matches:
            raise TransformerClientError(f"missing client tensor {suffix!r}")
        raise TransformerClientError(f"ambiguous client tensor suffix {suffix!r}")

    def render_prompt(
        self,
        messages: list[Any],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        rows: list[dict[str, Any]] = []
        for item in messages:
            if hasattr(item, "to_prompt_dict"):
                row = dict(item.to_prompt_dict())
            elif isinstance(item, dict):
                row = dict(item)
            else:
                row = {"role": "user", "content": str(item)}
            row["role"] = "system" if row.get("role") == "developer" else str(row.get("role"))
            row["content"] = str(row.get("content", ""))
            rows.append(row)
        template = self.tokenizer_descriptor.get("chat_template") or self.manifest.get(
            "chat_template"
        )
        if not template:
            suffix = "\nassistant:" if add_generation_prompt else ""
            return "\n".join(f"{row['role']}: {row['content']}" for row in rows) + suffix
        try:
            from jinja2 import StrictUndefined
            from jinja2.sandbox import SandboxedEnvironment

            environment = SandboxedEnvironment(
                undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True
            )

            def raise_exception(message: str) -> None:
                raise TransformerClientError(str(message))

            environment.globals["raise_exception"] = raise_exception
            return str(
                environment.from_string(str(template)).render(
                    messages=rows,
                    add_generation_prompt=bool(add_generation_prompt),
                    bos_token=self.tokenizer_descriptor.get("bos_token", ""),
                    eos_token=self.tokenizer_descriptor.get("eos_token", ""),
                    tools=tools,
                )
            )
        except BaseException:
            # Custom Transformers templates may require extensions such as the
            # ``generation`` tag. Falling back remains deterministic and local.
            suffix = "\nassistant:" if add_generation_prompt else ""
            return "\n".join(f"{row['role']}: {row['content']}" for row in rows) + suffix

    def tokenizer(self) -> Tokenizer:
        kind = str(self.tokenizer_descriptor.get("kind", self.tokenizer_descriptor.get("type", "")))
        if kind == "alphabet":
            return AlphabetTokenizer(str(self.tokenizer_descriptor["alphabet"]))
        if kind == "byte":
            return ModelByteTokenizer(
                vocab_size=int(self.tokenizer_descriptor.get("vocab_size", self.cfg["vocab_size"])),
                bos_token_id=int(
                    self.tokenizer_descriptor.get("bos_token_id", self.cfg["bos_token_id"])
                ),
                eos_token_id=int(
                    self.tokenizer_descriptor.get("eos_token_id", self.cfg["eos_token_id"])
                ),
            )
        if kind == "sentencepiece":
            try:
                import sentencepiece as spm
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise TransformerClientError("sentencepiece is required for this model") from exc
            processor = spm.SentencePieceProcessor()
            if not processor.LoadFromSerializedProto(bytes(self.tokenizer_descriptor["model"])):
                raise TransformerClientError("failed to load SentencePiece tokenizer")
            return SentencePieceTokenizer(
                processor,
                int(self.cfg["bos_token_id"]),
                int(self.cfg["eos_token_id"]),
            )
        if kind == "tokenizer_json":
            try:
                from tokenizers import Tokenizer as HFTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise TransformerClientError(
                    "tokenizers is required for tokenizer.json models"
                ) from exc
            inner = HFTokenizer.from_buffer(bytes(self.tokenizer_descriptor["model"]))
            return TokenizersJSONTokenizer(
                inner,
                int(self.cfg["bos_token_id"]),
                int(self.cfg["eos_token_id"]),
            )
        raise TransformerClientError(f"unsupported tokenizer type {kind!r}")


TransformerClientBundle = ClientBundle


@dataclass(frozen=True, slots=True)
class ModelByteTokenizer:
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    name: str = "pllm-model-byte-v1"

    def __post_init__(self) -> None:
        if self.vocab_size < 258:
            raise ValueError("byte tokenizer requires at least 258 tokens")

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        # Gemma-style test checkpoints reserve 0/1 and store bytes at 2..257.
        offset = 2 if {self.bos_token_id, self.eos_token_id} == {0, 1} else 0
        values = [int(item) + offset for item in text.encode("utf-8")]
        return ([self.bos_token_id] + values) if add_bos else values

    def decode(self, tokens: list[int]) -> str:
        offset = 2 if {self.bos_token_id, self.eos_token_id} == {0, 1} else 0
        data = bytes(
            max(0, min(255, int(token) - offset))
            for token in tokens
            if 0 <= int(token) - offset <= 255
        )
        return data.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class SentencePieceTokenizer:
    inner: Any
    bos_token_id: int
    eos_token_id: int
    name: str = "sentencepiece"

    @property
    def vocab_size(self) -> int:
        return int(self.inner.GetPieceSize())

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        values = [int(item) for item in self.inner.EncodeAsIds(text)]
        return ([self.bos_token_id] + values) if add_bos else values

    def decode(self, tokens: list[int]) -> str:
        return str(self.inner.DecodeIds([int(item) for item in tokens]))


@dataclass(frozen=True, slots=True)
class TokenizersJSONTokenizer:
    inner: Any
    bos_token_id: int
    eos_token_id: int
    name: str = "tokenizer-json"

    @property
    def vocab_size(self) -> int:
        return int(self.inner.get_vocab_size())

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        values = [int(item) for item in self.inner.encode(text).ids]
        return ([self.bos_token_id] + values) if add_bos else values

    def decode(self, tokens: list[int]) -> str:
        return str(self.inner.decode([int(item) for item in tokens], skip_special_tokens=True))


class CorrelationProvider(Protocol):
    def take_many(self, stage: StageMetadata, count: int) -> list[StageCorrelation]: ...


class StageExchange(Protocol):
    def __call__(self, stage_id: str, payloads: list[bytes]) -> list[bytes]: ...


@dataclass(slots=True)
class StageClientStats:
    calls: int = 0
    rows: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    correlations: int = 0
    server_ns: int = 0
    preparation_upload_bytes: int = 0
    preparation_download_bytes: int = 0
    inference_upload_bytes: int = 0
    inference_download_bytes: int = 0
    preparation_server_ns: int = 0
    inference_server_ns: int = 0
    correction_push_bytes: int = 0
    correction_push_ns: int = 0
    attempts: int = 0
    failures: int = 0


class CorrelationInventory:
    """Reusable inventory adapter with hard one-time consumption semantics."""

    def __init__(self, source: CorrelationProvider, *, prefetch: int = 4) -> None:
        self.source = source
        self.prefetch = max(1, int(prefetch))
        self._queues: dict[str, deque[StageCorrelation]] = defaultdict(deque)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def take_many(self, stage: StageMetadata, count: int) -> list[StageCorrelation]:
        with self._locks[stage.id]:
            queue = self._queues[stage.id]
            if len(queue) < count:
                queue.extend(self.source.take_many(stage, max(self.prefetch, count - len(queue))))
            output = [queue.popleft() for _ in range(count)]
            for item in output:
                if item.consumed:
                    raise TransformerClientError("correlation reuse detected")
                item.consumed = True
            return output


class RemoteLinear:
    def __init__(
        self,
        stages: dict[str, StageMetadata],
        correlations: CorrelationProvider,
        exchange: StageExchange,
    ) -> None:
        self.stages = stages
        self.correlations = correlations
        self.exchange = exchange
        self.stats = StageClientStats()

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self.stages.get(stage_id)
        if stage is None:
            raise TransformerClientError(f"unknown stage {stage_id!r}")
        value = np.asarray(activation, dtype=np.float32)
        if value.shape[-1] != stage.in_features:
            raise TransformerClientError(
                f"stage {stage_id} expects {stage.in_features} features, got {value.shape}"
            )
        quantized = quantize_activation_per_row(value, bits=stage.activation_bits)
        rows = self.correlations.take_many(stage, quantized.rows)
        if len(rows) != quantized.rows:
            raise TransformerClientError("correlation provider returned the wrong row count")
        ring = rows[0].ring or stage.ring
        modulus = rows[0].modulus
        if any(
            item.stage_id != stage_id
            or (item.ring or stage.ring) != ring
            or item.modulus != modulus
            for item in rows
        ):
            raise TransformerClientError("correlation arithmetic profile mismatch")
        masks = np.stack([np.asarray(item.mask).reshape(stage.in_features) for item in rows])
        transformed = np.stack(
            [np.asarray(item.transformed_mask).reshape(stage.out_features) for item in rows]
        )
        aggregate = StageCorrelation(
            id="batch-" + secrets.token_hex(12),
            stage_id=stage_id,
            mask=masks,
            transformed_mask=transformed,
            modulus=modulus,
            ring=ring,
        )
        masked = (quantized.values.astype(np.int64) + masks.astype(np.int64)) % modulus
        masked = masked.astype(np.uint32)
        owner_state = getattr(self.correlations, "state", None)
        owner_bundle = getattr(owner_state, "bundle", None)
        model_id = getattr(owner_bundle, "model_id", None) or getattr(
            self.correlations, "model_id", None
        )
        if model_id is None:
            raise TransformerClientError("correlation provider does not expose a model id")
        request = MaskedStageRequest(
            model=str(model_id),
            stage_id=stage_id,
            correlation_id=aggregate.id,
            masked_input=masked,
            activation_scales=quantized.scales,
            modulus=modulus,
            wire_bits=stage.wire_bits,
            ring=ring,
        )
        packed = request.pack()
        response_payloads = self.exchange(stage_id, [packed])
        if len(response_payloads) != 1:
            raise TransformerClientError("stage exchange returned the wrong result count")
        response = MaskedStageResponse.unpack(response_payloads[0])
        accumulators = unmask_stage_output(response, aggregate)
        output = dequantize_matmul(
            accumulators,
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


class PreparedRemoteLinear:
    def __init__(
        self,
        model_id,
        body_fingerprint,
        stages,
        inventory,
        inference,
    ) -> None:
        self.model_id = model_id
        self.body_fingerprint = body_fingerprint
        self.stages = stages
        self.inference = inference
        self.inventory = inventory
        self.stats = StageClientStats()

    @staticmethod
    def _result(payloads: list[bytes], request_ids: list[str], stage: StageMetadata):
        if len(payloads) != len(request_ids):
            raise TransformerClientError("inference provider returned the wrong result count")
        results = [MaskedStageResponse.unpack(payload) for payload in payloads]
        if stage.seeded_profile is None or any(
            result.correlation_id != request_id
            or result.stage_id != stage.id
            or result.ring != stage.seeded_profile.ring
            or result.modulus != stage.seeded_profile.modulus
            or result.wire_bits != stage.seeded_profile.wire_bits
            for result, request_id in zip(results, request_ids, strict=True)
        ):
            raise TransformerClientError("service returned a mismatched seeded ring result")
        return results

    def __call__(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self.stages.get(stage_id)
        if stage is None:
            raise TransformerClientError(f"unknown stage {stage_id!r}")
        value = np.asarray(activation, dtype=np.float32)
        if value.shape[-1] != stage.in_features:
            raise TransformerClientError(
                f"stage {stage_id} expects {stage.in_features} features, got {value.shape}"
            )
        quantized = quantize_activation_per_row(value, bits=stage.activation_bits)
        profile = stage.seeded_profile
        if profile is None:
            raise TransformerClientError("stage lacks a seeded ring profile")
        clear = (
            quantized.values.reshape(quantized.rows, stage.in_features).astype(np.int64, copy=False)
            % profile.modulus
        )
        mask, output_mask, attempt_ids = self.inventory.take(stage_id, quantized.rows)
        complement = (clear - mask.astype(np.int64)) % profile.modulus
        batch_id = secrets.token_hex(16) if quantized.rows > 1 else None
        if batch_id is not None:
            inference_requests = [
                PreparedStageBatchRequest(
                    batch_id=batch_id,
                    correlation_ids=tuple(attempt_ids),
                    masked_input=complement.astype(np.uint32),
                    wire_bits=profile.wire_bits,
                ).pack()
            ]
        else:
            inference_requests = [
                MaskedStageRequest(
                    model=self.model_id,
                    stage_id=stage_id,
                    correlation_id=attempt_ids[0],
                    masked_input=complement.astype(np.uint32),
                    activation_scales=np.ones(1, dtype=np.float32),
                    modulus=profile.modulus,
                    wire_bits=profile.wire_bits,
                    ring=profile.ring,
                    body_fingerprint=self.body_fingerprint,
                    weight_digest=stage.weight_digest,
                    weight_bits=stage.weight_bits,
                    activation_bits=stage.activation_bits,
                    session_id=self.inventory.inventory_id,
                    out_features=stage.out_features,
                    signed_output_bound=profile.signed_output_bound,
                ).pack()
            ]
        inference_upload_bytes = sum(map(len, inference_requests))
        self.stats.inference_upload_bytes += inference_upload_bytes
        self.stats.upload_bytes += inference_upload_bytes
        record_protocol_operation(stage_id)
        record_protocol_bytes("client", "inference", inference_upload_bytes, stage_id)
        protocol_span = start_protocol_span(stage_id, 0, inference_upload_bytes)
        try:
            inference_payloads = self.inference(stage_id, inference_requests)
            self.stats.inference_download_bytes += sum(map(len, inference_payloads))
            self.stats.download_bytes += sum(map(len, inference_payloads))
            record_protocol_bytes(
                "inference", "client", sum(map(len, inference_payloads)), stage_id
            )
            if batch_id is not None:
                if len(inference_payloads) != 1:
                    raise TransformerClientError(
                        "inference provider returned the wrong batch result count"
                    )
                batch_result = PreparedStageBatchResponse.unpack(
                    inference_payloads[0],
                    max_rows=quantized.rows,
                    max_tensor_elements=quantized.rows * stage.out_features,
                )
                if (
                    batch_result.batch_id != batch_id
                    or batch_result.wire_bits != profile.wire_bits
                    or batch_result.masked_output.shape != (quantized.rows, stage.out_features)
                ):
                    raise TransformerClientError(
                        "service returned a mismatched prepared batch result"
                    )
                masked_output = batch_result.masked_output
                inference_server_ns = batch_result.server_ns
            else:
                inference_results = self._result(inference_payloads, attempt_ids, stage)
                masked_output = inference_results[0].masked_output
                inference_server_ns = inference_results[0].server_ns
        except Exception as exc:
            protocol_span.record_exception(exc)
            protocol_span.end()
            raise
        protocol_span.set_attribute(
            "pllm.inference_client.bytes", sum(map(len, inference_payloads))
        )
        protocol_span.end()
        combined = (masked_output.astype(np.int64) + output_mask.astype(np.int64)) % profile.modulus
        if profile.ring == "u16":
            accumulators = combined.astype(np.uint16).view(np.int16).astype(np.int64)
        elif profile.ring == "u24":
            accumulators = np.where(combined >= 1 << 23, combined - (1 << 24), combined)
        else:
            accumulators = combined.astype(np.uint32).view(np.int32).astype(np.int64)
        output = dequantize_matmul(
            accumulators,
            quantized.scales,
            stage.weight_scales,
            output_shape=quantized.original_shape[:-1] + (stage.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        self.stats.calls += 1
        self.stats.rows += quantized.rows
        self.stats.inference_server_ns += inference_server_ns
        self.stats.server_ns += inference_server_ns
        return np.ascontiguousarray(output, dtype=np.float32)


MaskedStageClient = RemoteLinear


@dataclass(slots=True)
class LayerCache:
    key: np.ndarray | None = None
    value: np.ndarray | None = None
    length: int = 0

    def append(self, key: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        required = self.length + key.shape[0]
        key_storage = self.key
        value_storage = self.value
        if key_storage is None or value_storage is None or required > key_storage.shape[0]:
            capacity = max(64, 1 << (required - 1).bit_length())
            next_key = np.empty((capacity, *key.shape[1:]), dtype=key.dtype)
            next_value = np.empty((capacity, *value.shape[1:]), dtype=value.dtype)
            if self.length:
                if key_storage is None or value_storage is None:
                    raise TransformerClientError("KV cache storage is unavailable")
                next_key[: self.length] = key_storage[: self.length]
                next_value[: self.length] = value_storage[: self.length]
            self.key, self.value = next_key, next_value
            key_storage, value_storage = next_key, next_value
        key_storage[self.length : required] = key
        value_storage[self.length : required] = value
        self.length = required
        return key_storage[:required], value_storage[:required]

    def copy_active(self) -> "LayerCache":
        return LayerCache(
            None if self.key is None else self.key[: self.length].copy(),
            None if self.value is None else self.value[: self.length].copy(),
            self.length,
        )


@dataclass(slots=True)
class RuntimeSnapshot:
    position: int
    caches: list[LayerCache]
    shared_kv: dict[str, tuple[np.ndarray, np.ndarray]]


class MaskedTransformerClientRuntime:
    """Client-owned nonlinear, attention, KV-cache and sampling runtime.

    Every learned dense projection—including token lookup, Gemma PLE and LM
    head—remains remote. Only public tokenizer assets and normalization vectors
    are present in the client bundle.
    """

    def __init__(
        self,
        bundle: ClientBundle,
        remote: Callable[[str, np.ndarray], np.ndarray],
        *,
        token_cache: OrderedDict[int, np.ndarray] | None = None,
        token_cache_size: int = 512,
        token_cache_lock: threading.Lock | None = None,
    ) -> None:
        self.bundle = bundle
        self.remote = remote
        self.cfg = bundle.cfg
        self.config = self.cfg
        self.tokenizer = bundle.tokenizer()
        self.hidden = int(self.cfg["hidden_size"])
        self.layers = int(self.cfg["num_hidden_layers"])
        self.heads = int(self.cfg["num_attention_heads"])
        self.default_kv_heads = int(self.cfg.get("num_key_value_heads", self.heads))
        self.default_head_dim = int(self.cfg.get("head_dim", self.hidden // self.heads))
        self.eps = float(self.cfg.get("rms_norm_eps", 1e-6))
        self.norm_offset = float(self.cfg.get("norm_offset", 0.0))
        self.layer_types = list(self.cfg.get("layer_types") or ["full_attention"] * self.layers)
        self.sliding_window = self.cfg.get("sliding_window")
        self.shared_count = int(self.cfg.get("num_kv_shared_layers", 0) or 0)
        self.first_shared = self.layers - self.shared_count
        self.attention_k_eq_v = bool(self.cfg.get("attention_k_eq_v", False))
        self.ple_dim = int(self.cfg.get("hidden_size_per_layer_input", 0) or 0)
        self.embedding_multiplier = float(
            self.cfg.get("embedding_multiplier", math.sqrt(self.hidden))
        )
        self.block_style = str(self.cfg.get("block_style", "gemma4"))
        self.qk_norm = bool(self.cfg.get("qk_norm", True))
        self.v_norm = bool(self.cfg.get("v_norm", False))
        self.caches = [LayerCache() for _ in range(self.layers)]
        self.shared_kv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._producer_by_type = self._shared_producers()
        self.position = 0
        self.rng = np.random.default_rng(42)
        self.token_cache = token_cache if token_cache is not None else OrderedDict()
        self.token_cache_size = max(0, int(token_cache_size))
        self.token_cache_lock = token_cache_lock or threading.Lock()
        self.token_cache_hits = 0
        self.token_cache_misses = 0
        self.token_lookup_batch = max(1, int(self.cfg.get("token_lookup_batch", 16)))

    def reset(self) -> None:
        self.caches = [LayerCache() for _ in range(self.layers)]
        self.shared_kv.clear()
        self.position = 0

    def encode_prompt(self, prompt: str) -> list[int]:
        add_bos = bool(self.bundle.tokenizer_descriptor.get("add_bos_token", True))
        ids = self.tokenizer.encode(prompt, add_bos=add_bos)
        return ids or [int(self.cfg["bos_token_id"])]

    def prepare(self, prompt: str) -> tuple[list[int], np.ndarray, list[LayerCache]]:
        return self.prepare_ids(self.encode_prompt(prompt))

    def prepare_ids(self, ids: list[int]) -> tuple[list[int], np.ndarray, list[LayerCache]]:
        if not ids:
            ids = [int(self.cfg["bos_token_id"])]
        self.reset()
        values = np.asarray(ids, dtype=np.int64).reshape(-1)
        logits = self._forward(values, final_logits_only=True)[-1]
        return list(ids), logits, self.caches

    def forward_ids(self, ids: list[int] | np.ndarray) -> np.ndarray:
        values = np.asarray(ids, dtype=np.int64).reshape(-1)
        if values.size == 0:
            raise TransformerClientError("at least one token is required")
        return self._forward(values)

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            position=int(self.position),
            caches=[row.copy_active() for row in self.caches],
            shared_kv={
                key: (value[0].copy(), value[1].copy()) for key, value in self.shared_kv.items()
            },
        )

    def restore(self, snapshot: RuntimeSnapshot) -> None:
        self.position = int(snapshot.position)
        self.caches = [row.copy_active() for row in snapshot.caches]
        self.shared_kv = {
            key: (value[0].copy(), value[1].copy()) for key, value in snapshot.shared_kv.items()
        }

    def decode_step(
        self,
        token_id: int,
        caches: list[LayerCache] | None,
        position: int | None = None,
    ) -> tuple[np.ndarray, list[LayerCache]]:
        if caches is not None and caches is not self.caches:
            self.caches = caches
        logits = self._forward(np.asarray([token_id], dtype=np.int64))[-1]
        return logits, self.caches

    def sample(
        self, logits: np.ndarray, *, temperature: float = 0.0, top_p: float | None = None
    ) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))
        values = np.asarray(logits, dtype=np.float64) / float(temperature)
        values -= values.max()
        probabilities = np.exp(values)
        probabilities /= probabilities.sum()
        if top_p is not None and top_p < 1.0:
            order = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[order])
            keep = cumulative <= max(0.0, float(top_p))
            keep[0] = True
            mask = np.zeros_like(probabilities, dtype=bool)
            mask[order[keep]] = True
            probabilities = np.where(mask, probabilities, 0.0)
            probabilities /= probabilities.sum()
        return int(self.rng.choice(probabilities.size, p=probabilities))

    def generate_steps(
        self,
        prompt: str,
        *,
        max_output_tokens: int,
        temperature: float = 0.0,
        top_p: float | None = None,
    ):
        ids, logits, caches = self.prepare(prompt)
        for step in range(max_output_tokens):
            token = self.sample(logits, temperature=temperature, top_p=top_p)
            if token == int(self.cfg["eos_token_id"]):
                break
            yield token, self.tokenizer.decode([token])
            if step + 1 == max_output_tokens:
                break
            logits, caches = self.decode_step(token, caches, len(ids) + step)

    def _forward(self, ids: np.ndarray, *, final_logits_only: bool = False) -> np.ndarray:
        hidden, ple_tokens = self._token_lookup(ids)
        hidden = hidden * self.embedding_multiplier
        positions = np.arange(self.position, self.position + ids.size, dtype=np.int64)
        ple = self._per_layer_inputs(hidden, ple_tokens) if self.ple_dim else None
        for index in range(self.layers):
            hidden = self._layer(
                index, hidden, positions, None if ple is None else ple[:, index, :]
            )
        hidden = self._norm(hidden, self._tensor("model.norm.weight", self.hidden))
        head = self.bundle.stages["lm_head"]
        head_input = hidden[-1:] if final_logits_only and head.client_weight is not None else hidden
        logits = (
            self.bundle.local_linear("lm_head", head_input)
            if head.client_weight is not None
            else self.remote("lm_head", head_input)
        )
        multiplier = self.cfg.get("output_multiplier")
        if multiplier is not None:
            logits = logits * float(multiplier)
        cap = self.cfg.get("final_logit_softcapping")
        if cap:
            logits = float(cap) * np.tanh(logits / float(cap))
        self.position += ids.size
        return logits.astype(np.float32, copy=False)

    def _token_lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if "token_lookup" in self.bundle.stages:
            stage = self.bundle.stages["token_lookup"]
            valid = np.clip(np.asarray(ids, dtype=np.int64).reshape(-1), 0, stage.in_features - 1)
            if stage.client_weight is not None:
                values = self.bundle.local_token_lookup(valid)
                hidden = values[:, : self.hidden]
                rest = values[:, self.hidden :]
                return hidden, (rest if rest.size else None)
            values = np.empty((valid.size, stage.out_features), dtype=np.float32)
            missing: list[int] = []
            missing_set: set[int] = set()
            existing_hits = 0
            if self.token_cache_size:
                with self.token_cache_lock:
                    for token in valid:
                        key = int(token)
                        cached = self.token_cache.get(key)
                        if cached is not None:
                            self.token_cache.move_to_end(key)
                            existing_hits += 1
                        elif key not in missing_set:
                            missing.append(key)
                            missing_set.add(key)
                        # Duplicate misses in the same batch are deduplicated,
                        # but are not counted as pre-existing LRU hits.
            else:
                missing = list(dict.fromkeys(int(token) for token in valid))
            self.token_cache_hits += existing_hits

            for start in range(0, len(missing), self.token_lookup_batch):
                batch = missing[start : start + self.token_lookup_batch]
                one_hot = np.zeros((len(batch), stage.in_features), dtype=np.float32)
                if batch:
                    one_hot[np.arange(len(batch)), np.asarray(batch, dtype=np.int64)] = 1.0
                    looked_up = self.remote("token_lookup", one_hot)
                    self.token_cache_misses += len(batch)
                    with self.token_cache_lock:
                        for token, row in zip(batch, looked_up, strict=True):
                            self.token_cache[int(token)] = np.asarray(row, dtype=np.float32).copy()
                            self.token_cache.move_to_end(int(token))
                        while (
                            self.token_cache_size and len(self.token_cache) > self.token_cache_size
                        ):
                            self.token_cache.popitem(last=False)

            with self.token_cache_lock:
                for row, token in enumerate(valid):
                    key = int(token)
                    cached = self.token_cache.get(key)
                    if cached is None:
                        raise TransformerClientError(
                            f"token lookup cache failed to materialize token {key}"
                        )
                    self.token_cache.move_to_end(key)
                    values[row] = cached
            if not self.token_cache_size:
                # The temporary materialisation above is needed to assemble the
                # response rows, but cache_size=0 must not retain token-derived
                # model material across calls.
                with self.token_cache_lock:
                    self.token_cache.clear()
            hidden = values[:, : self.hidden]
            rest = values[:, self.hidden :]
            return hidden, (rest if rest.size else None)
        hidden = self._embedding(ids, "embed_tokens")
        token_ple = (
            self._embedding(ids, "embed_tokens_per_layer")
            if "embed_tokens_per_layer" in self.bundle.stages
            else None
        )
        return hidden, token_ple

    def _embedding(self, ids: np.ndarray, stage_id: str) -> np.ndarray:
        stage = self.bundle.stages[stage_id]
        one_hot = np.zeros((ids.size, stage.in_features), dtype=np.float32)
        valid = np.clip(ids, 0, stage.in_features - 1)
        one_hot[np.arange(ids.size), valid] = 1.0
        return self.remote(stage_id, one_hot)

    def _per_layer_inputs(
        self, embedding: np.ndarray, token_ple: np.ndarray | None
    ) -> np.ndarray | None:
        if token_ple is None:
            return None
        token = token_ple.reshape(embedding.shape[0], self.layers, self.ple_dim)
        token = token * math.sqrt(self.ple_dim)
        if "model.per_layer_model_projection" in self.bundle.stages:
            context = self.remote("model.per_layer_model_projection", embedding)
            context = context * (self.hidden**-0.5)
            context = context.reshape(embedding.shape[0], self.layers, self.ple_dim)
            context = self._norm(
                context,
                self._tensor("per_layer_projection_norm.weight", self.ple_dim),
            )
            return (token + context) * (2**-0.5)
        return token

    def _layer(
        self,
        index: int,
        hidden: np.ndarray,
        positions: np.ndarray,
        per_layer_input: np.ndarray | None,
    ) -> np.ndarray:
        residual = hidden
        normed = self._norm(
            hidden, self._tensor(f"layers.{index}.input_layernorm.weight", self.hidden)
        )
        attention = self._attention(index, normed, positions)
        if self.block_style == "gemma4":
            attention = self._norm(
                attention,
                self._tensor(f"layers.{index}.post_attention_layernorm.weight", self.hidden),
            )
        hidden = residual + attention

        residual = hidden
        if self.block_style == "gemma4":
            normed = self._norm(
                hidden,
                self._tensor(f"layers.{index}.pre_feedforward_layernorm.weight", self.hidden),
            )
        else:
            normed = self._norm(
                hidden,
                self._tensor(f"layers.{index}.post_attention_layernorm.weight", self.hidden),
            )
        gate_up = self.remote(f"layers.{index}.mlp.gate_up_proj", normed)
        gate, up = np.split(gate_up, 2, axis=-1)
        mlp = self._activation(gate) * up
        mlp = self.remote(f"layers.{index}.mlp.down_proj", mlp)
        if self.block_style == "gemma4":
            mlp = self._norm(
                mlp,
                self._tensor(f"layers.{index}.post_feedforward_layernorm.weight", self.hidden),
            )
        hidden = residual + mlp

        if per_layer_input is not None:
            gate_stage = f"layers.{index}.per_layer_input_gate"
            projection_stage = f"layers.{index}.per_layer_projection"
            if gate_stage not in self.bundle.stages or projection_stage not in self.bundle.stages:
                raise TransformerClientError(
                    f"Gemma PLE stages are missing for layer {index}; dense PLE weights are never shipped to the client"
                )
            residual = hidden
            ple_gate = self.remote(gate_stage, hidden)
            ple_gate = self._activation(ple_gate) * per_layer_input
            ple = self.remote(projection_stage, ple_gate)
            ple = self._norm(
                ple,
                self._tensor(f"layers.{index}.post_per_layer_input_norm.weight", self.hidden),
            )
            hidden = residual + ple
        return hidden

    def _attention(self, index: int, hidden: np.ndarray, positions: np.ndarray) -> np.ndarray:
        layer_type = self.layer_types[index]
        head_dim = self._layer_value(index, "head_dim", self.default_head_dim)
        kv_heads = self._layer_value(index, "num_key_value_heads", self.default_kv_heads)
        q_width = self.heads * head_dim
        kv_width = kv_heads * head_dim
        shared = self.shared_count > 0 and index >= self.first_shared
        if shared:
            q_raw = self.remote(f"layers.{index}.self_attn.q_proj", hidden)
            query = q_raw.reshape(hidden.shape[0], self.heads, head_dim)
            if layer_type not in self.shared_kv:
                raise TransformerClientError(f"shared KV for {layer_type!r} is unavailable")
            key_cache, value_cache = self.shared_kv[layer_type]
        else:
            qkv = self.remote(f"layers.{index}.self_attn.qkv_proj", hidden)
            query = qkv[:, :q_width].reshape(hidden.shape[0], self.heads, head_dim)
            key = qkv[:, q_width : q_width + kv_width].reshape(hidden.shape[0], kv_heads, head_dim)
            if qkv.shape[-1] == q_width + kv_width:
                value = key.copy()
            else:
                value = qkv[:, q_width + kv_width : q_width + 2 * kv_width].reshape(
                    hidden.shape[0], kv_heads, head_dim
                )
            if self.qk_norm:
                key = self._head_norm(key, f"layers.{index}.self_attn.k_norm.weight")
            if self.v_norm:
                value = self._norm_unscaled(value)
            key = self._rope(key, positions, index)
            cache = self.caches[index]
            key_cache, value_cache = cache.append(key, value)
            if self._producer_by_type.get(layer_type) == index:
                self.shared_kv[layer_type] = (key_cache, value_cache)
        if self.qk_norm:
            query = self._head_norm(query, f"layers.{index}.self_attn.q_norm.weight")
        query = self._rope(query, positions, index)

        groups = self.heads // kv_heads
        grouped_query = query.reshape(hidden.shape[0], kv_heads, groups, head_dim)
        output = np.empty((hidden.shape[0], self.heads, head_dim), dtype=np.float32)
        for row, position in enumerate(positions):
            end = min(int(position) + 1, key_cache.shape[0])
            start = 0
            if layer_type == "sliding_attention" and self.sliding_window:
                start = max(0, end - int(self.sliding_window))
            keys = key_cache[start:end]
            values = value_cache[start:end]
            scores = np.einsum("kgd,tkd->kgt", grouped_query[row], keys)
            configured_scaling = self.cfg.get("attention_scaling")
            scores *= float(
                1.0 / math.sqrt(head_dim) if configured_scaling is None else configured_scaling
            )
            scores -= scores.max(axis=-1, keepdims=True)
            probabilities = np.exp(scores).astype(np.float32)
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            output[row] = np.einsum("kgt,tkd->kgd", probabilities, values).reshape(
                self.heads, head_dim
            )
        return self.remote(
            f"layers.{index}.self_attn.o_proj", output.reshape(hidden.shape[0], q_width)
        )

    def _shared_producers(self) -> dict[str, int]:
        result: dict[str, int] = {}
        if self.shared_count <= 0:
            return result
        for index in range(self.first_shared):
            result[self.layer_types[index]] = index
        return result

    def _head_norm(self, value: np.ndarray, suffix: str) -> np.ndarray:
        return self._norm(value, self._tensor(suffix, value.shape[-1]))

    def _rope(self, value: np.ndarray, positions: np.ndarray, index: int) -> np.ndarray:
        params: Any = self.cfg.get("rope_parameters") or {}
        layer_type = self.layer_types[index]
        if isinstance(params, dict) and isinstance(params.get(layer_type), dict):
            params = params[layer_type]
        if not isinstance(params, dict):
            params = {}
        theta = float(params.get("rope_theta", self.cfg.get("rope_theta", 10000.0)))
        if theta == 0:
            return value
        partial = float(params.get("partial_rotary_factor", 1.0))
        dim = value.shape[-1]
        rotary_dim = min(dim, int(dim * partial))
        rotary_dim -= rotary_dim % 2
        if rotary_dim <= 0:
            return value
        frequency = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        angles = positions.astype(np.float32)[:, None] * frequency[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[:, None, :]
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[:, None, :]
        current = value[..., :rotary_dim]
        half = rotary_dim // 2
        rotated_half = np.concatenate([-current[..., half:], current[..., :half]], axis=-1)
        result = value.copy()
        result[..., :rotary_dim] = current * cos + rotated_half * sin
        return result

    def _norm(self, value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        result = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return result * (weight.astype(np.float32) + self.norm_offset)

    def _norm_unscaled(self, value: np.ndarray) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)

    def _tensor(self, suffix: str, width: int) -> np.ndarray:
        default = (
            np.zeros(width, dtype=np.float32)
            if self.norm_offset
            else np.ones(width, dtype=np.float32)
        )
        return self.bundle.tensor(suffix, default=default)

    def _layer_value(self, index: int, key: str, default: int) -> int:
        per_layer = self.cfg.get("per_layer_config") or {}
        for candidate in (str(index), index, self.layer_types[index]):
            row = per_layer.get(candidate) if isinstance(per_layer, dict) else None
            if isinstance(row, dict) and row.get(key) is not None:
                return int(row[key])
        return default

    def _activation(self, value: np.ndarray) -> np.ndarray:
        name = str(self.cfg.get("hidden_activation", "silu"))
        if name in {"silu", "swish"}:
            return value / (1.0 + np.exp(-value))
        if name in {"gelu", "gelu_pytorch_tanh", "gelu_new"}:
            return (
                0.5
                * value
                * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)))
            )
        if name == "relu":
            return np.maximum(value, 0)
        raise TransformerClientError(f"unsupported activation {name!r}")


GemmaNumpyRuntime = MaskedTransformerClientRuntime
