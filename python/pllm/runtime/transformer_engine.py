from __future__ import annotations

import asyncio
import hashlib
import json
import re
import math
import os
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import msgpack
import numpy as np
from filelock import FileLock

from .engine import EngineCapabilities
from .bfv_correlations import BFVCorrelationServer
from .models import ModelManifest, StageSpec, gemma4_stage_plan, transformer_stage_plan
from .native_kernels import MaskedGEMM
from .preparation_protocol import (
    CorrectionPush,
    PreparationRequest,
    SeededRingProfile,
    SessionAuthorization,
    expand_output_mask,
    expand_preparation_mask,
    seeded_ring_profile,
)
from .quantization import (
    QuantizedWeight,
    choose_plain_modulus,
    choose_wire_bits,
    quantize_weight_per_row,
    signed_qmax,
)
from .safetensors_store import SafeTensorStore, TensorStoreError
from .stage_protocol import MaskedStageRequest, MaskedStageResponse, RingKind, StageCorrelation
from .tiled_bfv import TiledBFVError, TiledBFVServer, tiled_context_modulus


class TransformerEngineError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class StageMetadata:
    id: str
    op: str
    in_features: int
    out_features: int
    weight_bits: int
    activation_bits: int
    modulus: int
    wire_bits: int
    weight_scales: np.ndarray
    bias: np.ndarray | None = None
    role: str | None = None
    layer_index: int | None = None
    ring: str = "prime"
    weight_digest: str = ""

    def pack(self) -> bytes:
        return msgpack.packb(
            {
                "v": 1,
                "id": self.id,
                "op": self.op,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "weight_bits": self.weight_bits,
                "activation_bits": self.activation_bits,
                "modulus": self.modulus,
                "wire_bits": self.wire_bits,
                "weight_scales": self.weight_scales.astype("<f4", copy=False).tobytes(),
                "bias": None
                if self.bias is None
                else self.bias.astype("<f4", copy=False).tobytes(),
                "role": self.role,
                "layer_index": self.layer_index,
                "ring": self.ring,
                "weight_digest": self.weight_digest,
            },
            use_bin_type=True,
        )

    @classmethod
    def unpack(cls, payload: bytes) -> "StageMetadata":
        row = msgpack.unpackb(payload, raw=False)
        if int(row.get("v", 0)) != 1:
            raise TransformerEngineError("unsupported stage metadata")
        scales = np.frombuffer(row["weight_scales"], dtype="<f4").copy()
        out_features = int(row["out_features"])
        if scales.shape != (out_features,):
            raise TransformerEngineError("invalid stage scale count")
        bias_payload = row.get("bias")
        bias = None if bias_payload is None else np.frombuffer(bias_payload, dtype="<f4").copy()
        if bias is not None and bias.shape != (out_features,):
            raise TransformerEngineError("invalid stage bias count")
        return cls(
            id=str(row["id"]),
            op=str(row["op"]),
            in_features=int(row["in_features"]),
            out_features=out_features,
            weight_bits=int(row["weight_bits"]),
            activation_bits=int(row["activation_bits"]),
            modulus=int(row["modulus"]),
            wire_bits=int(row["wire_bits"]),
            weight_scales=scales,
            bias=bias,
            role=row.get("role"),
            layer_index=row.get("layer_index"),
            ring=str(row.get("ring", "prime")),
            weight_digest=str(row.get("weight_digest", "")),
        )


@dataclass(slots=True)
class StageRuntime:
    spec: StageSpec
    weight: QuantizedWeight
    modulus: int
    wire_bits: int
    source_keys: tuple[str, ...]
    bias: np.ndarray | None = None
    calls: int = 0
    rows: int = 0
    server_ns: int = 0
    compiled_weight: Any = None
    _weight_digest: str = field(init=False)
    _signed_output_bound: int = field(init=False)

    def __post_init__(self) -> None:
        values = self.weight.values.astype(np.int8, copy=False)
        self._weight_digest = hashlib.sha256(values.tobytes()).hexdigest()
        row_l1 = int(np.max(np.sum(np.abs(values.astype(np.int16)), axis=1, dtype=np.int64)))
        self._signed_output_bound = signed_qmax(self.spec.activation_bits) * row_l1

    @property
    def metadata(self) -> StageMetadata:
        return StageMetadata(
            id=self.spec.id,
            op=self.spec.op,
            in_features=self.spec.in_features,
            out_features=self.spec.out_features,
            weight_bits=self.weight.bits,
            activation_bits=self.spec.activation_bits,
            modulus=self.modulus,
            wire_bits=self.wire_bits,
            weight_scales=self.weight.scales,
            bias=self.bias,
            role=self.spec.role,
            layer_index=self.spec.layer_index,
            weight_digest=self.weight_digest,
        )

    @property
    def weight_digest(self) -> str:
        return self._weight_digest

    @property
    def signed_output_bound(self) -> int:
        return self._signed_output_bound

    @property
    def seeded_profile(self) -> SeededRingProfile:
        return seeded_ring_profile(self.signed_output_bound)

    def public_descriptor(
        self,
        *,
        include_weight: bool = False,
        include_seeded_profile: bool = False,
    ) -> dict[str, Any]:
        descriptor = {
            "id": self.spec.id,
            "op": self.spec.op,
            "in_features": self.spec.in_features,
            "out_features": self.spec.out_features,
            "weight_bits": self.weight.bits,
            "activation_bits": self.spec.activation_bits,
            "ring": "prime",
            "weight_digest": self.weight_digest,
            "modulus": self.modulus,
            "wire_bits": self.wire_bits,
            "weight_scales": self.weight.scales.astype("<f4", copy=False).tobytes(),
            "bias": None if self.bias is None else self.bias.astype("<f4", copy=False).tobytes(),
            "source_keys": list(self.source_keys),
            "role": self.spec.role,
            "layer_index": self.spec.layer_index,
        }
        if include_weight:
            descriptor["client_weight"] = {
                "dtype": "i1",
                "shape": list(self.weight.values.shape),
                "data": self.weight.values.astype(np.int8, copy=False).tobytes(),
            }
        if include_seeded_profile:
            descriptor["seeded_profile"] = self.seeded_profile.to_dict()
        return descriptor


@dataclass(slots=True)
class LoadedTransformer:
    manifest: ModelManifest
    store: SafeTensorStore
    config: dict[str, Any]
    stages: dict[str, StageRuntime]
    local_tensors: dict[str, np.ndarray]
    tokenizer: dict[str, Any]
    bfv_contexts: dict[str, bytes] = field(default_factory=dict)
    bfv_servers: dict[tuple[str, str], BFVCorrelationServer | TiledBFVServer] = field(
        default_factory=dict
    )
    bfv_lock: threading.Lock = field(default_factory=threading.Lock)


def _body_fingerprint(stages: dict[str, StageRuntime]) -> str:
    body = []
    for stage_id, runtime in sorted(stages.items()):
        if stage_id in {"token_lookup", "lm_head"}:
            continue
        body.append(
            {
                "id": stage_id,
                "op": runtime.spec.op,
                "in": runtime.spec.in_features,
                "out": runtime.spec.out_features,
                "weight_bits": runtime.spec.weight_bits,
                "activation_bits": runtime.spec.activation_bits,
                "weight_digest": runtime.weight_digest,
                "weight_scales": runtime.weight.scales.astype("<f4", copy=False).tobytes(),
                "bias": None
                if runtime.bias is None
                else runtime.bias.astype("<f4", copy=False).tobytes(),
            }
        )
    return hashlib.sha256(msgpack.packb(body, use_bin_type=True)).hexdigest()


def _seeded_stage_commitment(stages: dict[str, StageRuntime]) -> str:
    body = [
        {
            "id": stage_id,
            "weight": runtime.weight_digest,
            "in": runtime.spec.in_features,
            "out": runtime.spec.out_features,
            "wb": runtime.spec.weight_bits,
            "ab": runtime.spec.activation_bits,
            "profile": runtime.seeded_profile.to_dict(),
        }
        for stage_id, runtime in sorted(stages.items())
        if stage_id not in {"token_lookup", "lm_head"}
    ]
    return hashlib.sha256(msgpack.packb(body, use_bin_type=True)).hexdigest()


@dataclass(frozen=True, slots=True)
class ArchitectureProfile:
    family: str
    stage_plan: str
    block_style: str
    norm_offset: float
    embedding_multiplier: float
    attention_scaling: float | None


def classify_architecture(
    manifest: ModelManifest,
    config: dict[str, Any],
) -> ArchitectureProfile:
    architecture = (manifest.architecture + " " + str(config.get("model_type", ""))).lower()
    hidden = int(manifest.hidden_size)
    if "muse_glimmer" in architecture or "museglimmer" in architecture:
        raise TransformerEngineError(
            "Muse Glimmer requires its normalized embedding and gated-attention adapter; "
            "this runtime currently supports Gemma 4 and Llama-compatible decoder graphs"
        )
    if bool(config.get("enable_moe_block")) or int(config.get("num_experts", 0) or 0) > 0:
        raise TransformerEngineError(
            "sparse MoE checkpoints require a route-private expert adapter and are not yet supported"
        )
    if any(name in architecture for name in ("gemma2", "gemma_2", "gemma3", "gemma_3")) and not any(
        name in architecture for name in ("gemma4", "gemma_4")
    ):
        raise TransformerEngineError(
            "Gemma 2/3 use a different residual/norm graph; use Gemma 4 or a Llama-compatible checkpoint"
        )
    if "gemma4" in architecture or "gemma_4" in architecture:
        return ArchitectureProfile(
            family="gemma4",
            stage_plan="gemma4",
            block_style="gemma4",
            norm_offset=0.0,
            embedding_multiplier=math.sqrt(hidden),
            attention_scaling=1.0,
        )

    supported = (
        "llama",
        "mistral",
        "qwen2",
        "qwen_2",
        "qwen3",
        "qwen_3",
        "smollm",
        "granite",
        "olmo",
        "yi",
        "internlm",
    )
    if not any(name in architecture for name in supported):
        raise TransformerEngineError(
            f"unsupported decoder architecture {manifest.architecture!r}; "
            "expected Gemma 4 or a Llama-compatible q/k/v/o + gated-MLP checkpoint"
        )
    if config.get("rope_scaling") not in (None, {}, {"type": "default"}, {"rope_type": "default"}):
        # The base/default RoPE path is exact. More involved YaRN/LongRoPE
        # variants need family-specific position transforms.
        rope = config.get("rope_scaling")
        rope_type = rope.get("rope_type", rope.get("type")) if isinstance(rope, dict) else None
        if rope_type not in (None, "default", "linear"):
            raise TransformerEngineError(
                f"RoPE scaling {rope_type!r} is not implemented by the local runtime client"
            )
    return ArchitectureProfile(
        family="llama-compatible",
        stage_plan="llama",
        block_style="llama",
        norm_offset=0.0,
        embedding_multiplier=1.0,
        attention_scaling=None,
    )


class MaskedTransformerEngine:
    """Safetensors-backed W4A4 Transformer engine with HE-generated masks.

    Online inputs are exact modular one-time-pad masked activations. The server
    owns every learned dense matrix; the client owns tokenisation, nonlinearities,
    attention/KV state, sampling, and the BFV secret key.
    """

    capabilities = EngineCapabilities(
        name="masked-transformer-w4a4",
        model_sources=("huggingface", "safetensors", "vllm", "mlx-lm"),
        protocols=("masked.stage/v3", "prepared-correction/v2", "bfv-correlation/v1"),
        online_fhe=False,
        preprocessed=True,
        continuous_batching=True,
        notes=(
            "all learned dense matrices remain server-side",
            "exact modular online GEMM over one-time masked W4A4 activations",
            "client executes nonlinearities, attention/KV and sampling",
        ),
    )

    def __init__(
        self,
        *,
        weight_bits: int = 4,
        activation_bits: int = 4,
        modulus: int | None = None,
        threads: int | None = None,
        native_library: str | Path | None = None,
        tenseal_path: str | None = None,
        local_correlation_seed: int = 20260817,
        compiled_cache_dir: str | Path | None = None,
        streaming_threshold_elements: int = 50_000_000,
        quantization_chunk_rows: int = 64,
    ) -> None:
        if modulus is not None and (modulus <= 2 or modulus >= 2**31):
            raise ValueError("modulus must satisfy 2 < p < 2^31")
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)
        self.fixed_modulus = None if modulus is None else int(modulus)
        # Backward-compatible attribute. With automatic profiles this is the
        # largest configured stage modulus after model load.
        self.modulus = self.fixed_modulus
        self.tenseal_path = tenseal_path
        self.kernel = MaskedGEMM(native_library, threads=threads)
        self.compiled_cache_dir = Path(
            compiled_cache_dir or (Path.home() / ".cache" / "pllm" / "compiled")
        )
        self.streaming_threshold_elements = max(1, int(streaming_threshold_elements))
        self.quantization_chunk_rows = max(1, int(quantization_chunk_rows))
        self.models: dict[str, LoadedTransformer] = {}
        self._rng = np.random.default_rng(local_correlation_seed)
        self._rng_lock = threading.Lock()

    async def load(self, manifest: ModelManifest) -> None:
        if manifest.id in self.models:
            raise TransformerEngineError(f"model {manifest.id!r} is already loaded")
        source = Path(manifest.source)
        store = SafeTensorStore(source)
        raw_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        config = (
            raw_config.get("text_config")
            if isinstance(raw_config.get("text_config"), dict)
            else raw_config
        )
        config = dict(config)

        profile = classify_architecture(manifest, config)
        if profile.stage_plan == "gemma4":
            stages = gemma4_stage_plan(config, include_lm_head=True)
        else:
            stages = transformer_stage_plan(
                hidden_size=manifest.hidden_size,
                intermediate_size=manifest.intermediate_size,
                num_hidden_layers=manifest.num_hidden_layers,
                num_attention_heads=manifest.num_attention_heads,
                num_key_value_heads=manifest.num_key_value_heads,
                head_dim=manifest.head_dim,
                vocab_size=manifest.vocab_size,
                include_embedding=True,
                include_lm_head=True,
            )
        # Token-addressed tables are fused into one remote lookup. All learned
        # dense projections, including Gemma 4 PLE gate/projection matrices,
        # remain server-owned. The client bundle contains only normalization
        # vectors and public tokenizer assets.
        ple = int(config.get("hidden_size_per_layer_input", 0) or 0)
        stages = [
            stage for stage in stages if stage.id not in {"embed_tokens_per_layer", "embed_tokens"}
        ]
        stages.insert(
            0,
            StageSpec(
                id="token_lookup",
                op="embedding",
                in_features=manifest.vocab_size,
                out_features=manifest.hidden_size + manifest.num_hidden_layers * ple,
                weight_keys=("model.embed_tokens.weight", "embed_tokens.weight"),
                transpose_weight=True,
                role="token_lookup",
                metadata={"ple_width": manifest.num_hidden_layers * ple},
            ),
        )
        stages = [
            replace(
                stage,
                weight_bits=self.weight_bits,
                activation_bits=self.activation_bits,
            )
            for stage in stages
        ]
        manifest.stages = stages
        manifest.metadata.update(
            {
                "client_runtime": "masked_transformer_v1",
                "runtime": "masked_transformer",
                "engine": self.capabilities.name,
                "native_masked_gemm": self.kernel.available,
                "kernel_backend": self.kernel.backend,
                "weight_bits": self.weight_bits,
                "activation_bits": self.activation_bits,
                "model_family": profile.family,
                "block_style": profile.block_style,
                "privacy_mode": "public",
                "privacy_protocol": f"masked_w{self.weight_bits}a{self.activation_bits}",
                "online_fhe": False,
                "preprocessed": True,
                "model_weight_correlations_disclosed": True,
                "model_privacy_threat_model": "public_weights",
            }
        )

        runtimes = {
            stage.id: await asyncio.to_thread(self._load_stage, store, stage, manifest)
            for stage in stages
        }
        moduli = sorted({runtime.modulus for runtime in runtimes.values()})
        manifest.metadata.update(
            {
                "prime_modulus": max(moduli),
                "plain_moduli": moduli,
                "stage_specific_moduli": self.fixed_modulus is None,
                "body_fingerprint": _body_fingerprint(runtimes),
                "seeded_stage_commitment": _seeded_stage_commitment(runtimes),
            }
        )
        if self.modulus is None:
            self.modulus = max(moduli)
        config.update(
            {
                "model_family": profile.family,
                "block_style": profile.block_style,
                "norm_offset": profile.norm_offset,
                "embedding_multiplier": profile.embedding_multiplier,
                "attention_scaling": profile.attention_scaling,
                "hidden_activation": config.get("hidden_activation")
                or config.get("hidden_act")
                or "silu",
            }
        )
        self.models[manifest.id] = LoadedTransformer(
            manifest=manifest,
            store=store,
            config=config,
            stages=runtimes,
            local_tensors=self._load_local_tensors(store),
            tokenizer=self._load_tokenizer_descriptor(source, manifest, config),
        )

    async def unload(self, model_id: str) -> None:
        if self.models.pop(model_id, None) is None:
            raise TransformerEngineError(f"unknown model {model_id!r}")

    def model_manifest(self, model_id: str) -> ModelManifest:
        return self._model(model_id).manifest

    def _resolve_stage_sources(
        self,
        store: SafeTensorStore,
        stage: StageSpec,
        manifest: ModelManifest,
    ) -> list[tuple[str, bool]]:
        keys = stage.weight_keys or self._default_weight_keys(stage)
        if stage.id == "token_lookup":
            output = [
                (
                    store.resolve_first(("model.embed_tokens.weight", "embed_tokens.weight")),
                    True,
                )
            ]
            try:
                output.append(
                    (
                        store.resolve_first(
                            (
                                "model.embed_tokens_per_layer.weight",
                                "embed_tokens_per_layer.weight",
                            )
                        ),
                        True,
                    )
                )
            except TensorStoreError:
                pass
            return output
        if stage.op == "lm_head":
            candidates = list(keys)
            if manifest.tied_embeddings:
                candidates = ["model.embed_tokens.weight", "embed_tokens.weight", *candidates]
            return [(store.resolve_first(candidates), bool(stage.transpose_weight))]
        resolved = [(store.resolve(key), False) for key in keys]
        if stage.transpose_weight:
            if len(resolved) != 1:
                raise TransformerEngineError(f"stage {stage.id} cannot transpose fused weights")
            resolved[0] = (resolved[0][0], True)
        return resolved

    def _oriented_shape(
        self,
        store: SafeTensorStore,
        key: str,
        transpose: bool,
    ) -> tuple[int, int]:
        shape = store.tensor_shape(key)
        if len(shape) != 2:
            raise TransformerEngineError(f"tensor {key!r} must be rank two, got {shape}")
        return (shape[1], shape[0]) if transpose else (shape[0], shape[1])

    def _stage_cache_key(
        self,
        store: SafeTensorStore,
        stage: StageSpec,
        sources: list[tuple[str, bool]],
    ) -> str:
        payload = {
            "stage": stage.to_dict(),
            "bits": stage.weight_bits or self.weight_bits,
            "sources": [
                {**row, "transpose": transpose}
                for row, (_, transpose) in zip(
                    store.source_signature(key for key, _ in sources),
                    sources,
                    strict=True,
                )
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _iter_oriented_rows(
        self,
        store: SafeTensorStore,
        key: str,
        *,
        transpose: bool,
        chunk_rows: int,
    ):
        source_shape = store.tensor_shape(key)
        if len(source_shape) != 2:
            raise TransformerEngineError(f"tensor {key!r} must be rank two")
        dtype = store.tensor_dtype(key).upper()
        if dtype not in {"F16", "BF16", "F32", "F64"}:
            # Packed MLX and other quantized layouts currently use the existing
            # dequantizer. This remains bounded for normal projection shards.
            value = store.get_linear((key,))
            if transpose:
                value = value.T
            for start in range(0, value.shape[0], chunk_rows):
                yield np.ascontiguousarray(value[start : start + chunk_rows], dtype=np.float32)
            return
        out_features = source_shape[1] if transpose else source_shape[0]
        selections = (
            (slice(None), slice(start, min(out_features, start + chunk_rows)))
            if transpose
            else (slice(start, min(out_features, start + chunk_rows)), slice(None))
            for start in range(0, out_features, chunk_rows)
        )
        for value in store.iter_slices(key, selections, dtype=np.float32):
            if transpose:
                value = value.T
            yield np.ascontiguousarray(value, dtype=np.float32)

    def _quantize_sources(
        self,
        store: SafeTensorStore,
        stage: StageSpec,
        sources: list[tuple[str, bool]],
    ) -> QuantizedWeight:
        shapes = [self._oriented_shape(store, key, transpose) for key, transpose in sources]
        in_features = shapes[0][1]
        if any(shape[1] != in_features for shape in shapes):
            raise TransformerEngineError(f"stage {stage.id} source widths differ: {shapes}")
        out_features = sum(shape[0] for shape in shapes)
        expected = (stage.out_features, stage.in_features)
        if (out_features, in_features) != expected:
            raise TransformerEngineError(
                f"stage {stage.id} expected {expected} but sources produce "
                f"{(out_features, in_features)} from {[key for key, _ in sources]}"
            )
        bits = stage.weight_bits or self.weight_bits
        elements = out_features * in_features
        if elements < self.streaming_threshold_elements:
            matrices: list[np.ndarray] = []
            for key, transpose in sources:
                value = store.get_linear((key,))
                matrices.append(np.ascontiguousarray(value.T if transpose else value))
            matrix = np.concatenate(matrices, axis=0) if len(matrices) > 1 else matrices[0]
            return quantize_weight_per_row(matrix, bits=bits)

        cache_key = self._stage_cache_key(store, stage, sources)
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage.id)
        root = self.compiled_cache_dir / cache_key[:2] / cache_key
        root.mkdir(parents=True, exist_ok=True)
        values_path = root / f"{safe_stage}.i8"
        scales_path = root / f"{safe_stage}.scales.f32"
        metadata_path = root / f"{safe_stage}.json"
        metadata = {
            "cache_key": cache_key,
            "shape": [out_features, in_features],
            "bits": bits,
            "sources": [key for key, _ in sources],
        }
        expected_bytes = elements

        def open_cached() -> QuantizedWeight | None:
            if not (
                metadata_path.exists()
                and values_path.exists()
                and scales_path.exists()
                and values_path.stat().st_size == expected_bytes
                and scales_path.stat().st_size == out_features * 4
            ):
                return None
            try:
                if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                    return None
            except Exception:
                return None
            values = np.memmap(
                values_path,
                dtype=np.int8,
                mode="r",
                shape=(out_features, in_features),
            )
            scales = np.memmap(
                scales_path,
                dtype="<f4",
                mode="r",
                shape=(out_features,),
            )
            return QuantizedWeight(values, scales, bits)

        cached = open_cached()
        if cached is not None:
            return cached

        lock_path = root / f"{safe_stage}.compile.lock"
        with FileLock(str(lock_path)):
            # Another worker may have finished while this process waited.
            cached = open_cached()
            if cached is not None:
                return cached

            nonce = f"{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}"
            temporary_values = root / f"{safe_stage}.{nonce}.i8.tmp"
            temporary_scales = root / f"{safe_stage}.{nonce}.scales.f32.tmp"
            values = np.memmap(
                temporary_values,
                dtype=np.int8,
                mode="w+",
                shape=(out_features, in_features),
            )
            scales = np.memmap(
                temporary_scales,
                dtype="<f4",
                mode="w+",
                shape=(out_features,),
            )
            try:
                qmax = signed_qmax(bits)
                output_offset = 0
                for key, transpose in sources:
                    for chunk in self._iter_oriented_rows(
                        store,
                        key,
                        transpose=transpose,
                        chunk_rows=self.quantization_chunk_rows,
                    ):
                        rows = chunk.shape[0]
                        max_abs = np.max(np.abs(chunk), axis=1)
                        row_scales = np.where(max_abs > 0, max_abs / qmax, 1.0).astype(np.float32)
                        quantized = np.clip(
                            np.rint(chunk / row_scales[:, None]),
                            -qmax,
                            qmax,
                        ).astype(np.int8)
                        values[output_offset : output_offset + rows] = quantized
                        scales[output_offset : output_offset + rows] = row_scales
                        output_offset += rows
                if output_offset != out_features:
                    raise TransformerEngineError(
                        f"streaming quantizer wrote {output_offset} rows, expected {out_features}"
                    )
                values.flush()
                scales.flush()
                del values, scales
                temporary_values.replace(values_path)
                temporary_scales.replace(scales_path)
                metadata_tmp = root / f"{safe_stage}.{nonce}.json.tmp"
                metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                metadata_tmp.replace(metadata_path)
            finally:
                for temporary in (temporary_values, temporary_scales):
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass

        cached = open_cached()
        if cached is None:  # pragma: no cover - atomic commit invariant
            raise TransformerEngineError(f"compiled stage cache is incomplete for {stage.id}")
        return cached

    def _load_stage(
        self, store: SafeTensorStore, stage: StageSpec, manifest: ModelManifest
    ) -> StageRuntime:
        sources = self._resolve_stage_sources(store, stage, manifest)
        resolved = [key for key, _ in sources]
        source_shapes = [self._oriented_shape(store, key, transpose) for key, transpose in sources]
        quantized = self._quantize_sources(store, stage, sources)
        bias = store.get_optional(stage.bias_keys, dtype=np.float32) if stage.bias_keys else None
        if bias is None and stage.id != "token_lookup":
            bias_parts: list[np.ndarray | None] = []
            any_bias = False
            for (resolved_key, _), shape in zip(sources, source_shapes, strict=True):
                candidate = resolved_key.removesuffix(".weight") + ".bias"
                item = store.get_optional((candidate,), dtype=np.float32)
                if item is not None:
                    item = np.asarray(item, dtype=np.float32).reshape(-1)
                    if item.shape != (shape[0],):
                        raise TransformerEngineError(
                            f"bias {candidate} has shape {item.shape}, expected {(shape[0],)}"
                        )
                    any_bias = True
                bias_parts.append(item)
            if any_bias:
                bias = np.concatenate(
                    [
                        np.zeros(shape[0], dtype=np.float32) if item is None else item
                        for item, shape in zip(bias_parts, source_shapes, strict=True)
                    ]
                )
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32).reshape(-1)
            if bias.shape != (stage.out_features,):
                raise TransformerEngineError(
                    f"stage {stage.id} bias has shape {bias.shape}, expected {(stage.out_features,)}"
                )
        store.drop(*resolved)
        if self.fixed_modulus is not None:
            modulus = self.fixed_modulus
        elif stage.op == "embedding" or stage.role == "token_lookup":
            modulus = choose_plain_modulus(
                1,
                weight_bits=stage.weight_bits or self.weight_bits,
                activation_bits=stage.activation_bits or self.activation_bits,
            )
        else:
            modulus = choose_plain_modulus(
                stage.in_features,
                weight_bits=stage.weight_bits or self.weight_bits,
                activation_bits=stage.activation_bits or self.activation_bits,
            )
        return StageRuntime(
            spec=stage,
            weight=quantized,
            modulus=modulus,
            wire_bits=choose_wire_bits(modulus),
            source_keys=tuple(resolved),
            bias=bias,
            compiled_weight=self.kernel.compile(quantized.values),
        )

    @staticmethod
    def _default_weight_keys(stage: StageSpec) -> tuple[str, ...]:
        if stage.id == "embed_tokens":
            return ("model.embed_tokens.weight", "embed_tokens.weight")
        if stage.id == "lm_head":
            return ("lm_head.weight",)
        if stage.id.endswith("qkv_proj"):
            prefix = stage.id.removesuffix("qkv_proj")
            return (prefix + "q_proj.weight", prefix + "k_proj.weight", prefix + "v_proj.weight")
        if stage.id.endswith("gate_up_proj"):
            prefix = stage.id.removesuffix("gate_up_proj")
            return (prefix + "gate_proj.weight", prefix + "up_proj.weight")
        return (stage.id + ".weight",)

    @staticmethod
    def _load_local_tensors(store: SafeTensorStore) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key in store.keys:
            lower = key.lower()
            if lower.endswith("norm.weight") or lower.endswith("norm.scale"):
                result[key] = store.get(key, dtype=np.float32)
        return result

    @staticmethod
    def _load_tokenizer_descriptor(
        source: Path, manifest: ModelManifest, config: dict[str, Any]
    ) -> dict[str, Any]:
        custom = source / "pllm_tokenizer.json"
        if custom.exists():
            value = json.loads(custom.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or "type" not in value:
                raise TransformerEngineError("invalid pllm_tokenizer.json")
            return value

        tokenizer_config: dict[str, Any] = {}
        tokenizer_config_path = source / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            try:
                tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise TransformerEngineError("invalid tokenizer_config.json") from exc
        eos_value = config.get("eos_token_id", 1)
        if isinstance(eos_value, list):
            eos_value = eos_value[0] if eos_value else 1
        common = {
            "vocab_size": manifest.vocab_size,
            "bos_token_id": int(config.get("bos_token_id", 2) or 2),
            "eos_token_id": int(eos_value or 1),
            "chat_template": tokenizer_config.get("chat_template") or manifest.chat_template,
            "bos_token": tokenizer_config.get("bos_token", ""),
            "eos_token": tokenizer_config.get("eos_token", ""),
            "add_bos_token": bool(tokenizer_config.get("add_bos_token", True)),
        }
        if config.get("pllm_test_tokenizer") == "byte":
            return {"type": "byte", **common}
        sentencepiece = source / "tokenizer.model"
        if sentencepiece.exists():
            return {"type": "sentencepiece", "model": sentencepiece.read_bytes(), **common}
        tokenizer_json = source / "tokenizer.json"
        if tokenizer_json.exists():
            return {"type": "tokenizer_json", "model": tokenizer_json.read_bytes(), **common}
        raise TransformerEngineError(
            "private runtime client requires tokenizer.model, tokenizer.json, or pllm_tokenizer.json"
        )

    async def execute_stage(
        self, model_id: str, stage: StageSpec, payloads: list[bytes]
    ) -> list[bytes]:
        model = self._model(model_id)
        runtime = self._runtime(model_id, stage.id)
        requests = [MaskedStageRequest.unpack(payload) for payload in payloads]
        if not requests:
            return []
        profiles = {
            (request.ring or "prime", request.modulus, request.wire_bits)
            for request in requests
        }
        if len(profiles) != 1:
            raise TransformerEngineError("masked stage arithmetic profile mismatch")
        ring, modulus, wire_bits = profiles.pop()
        prime_profile = ring == "prime" and modulus == runtime.modulus and wire_bits == runtime.wire_bits
        wrapping_profile = ring in {"u16", "u24", "u32"} and (
            modulus, wire_bits
        ) == (1 << wire_bits, wire_bits)
        if not prime_profile and not wrapping_profile:
            raise TransformerEngineError("masked stage arithmetic profile mismatch")
        if wrapping_profile and runtime.signed_output_bound >= 1 << (wire_bits - 1):
            raise TransformerEngineError("stage signed output exceeds wrapping ring range")
        for request in requests:
            if request.model != model_id or request.stage != stage.id:
                raise TransformerEngineError("masked stage request route mismatch")
            if (
                request.masked_input.ndim != 2
                or request.masked_input.shape[1] != runtime.spec.in_features
            ):
                raise TransformerEngineError("masked stage input width mismatch")
            if request.body_fingerprint and request.body_fingerprint != model.manifest.metadata.get(
                "body_fingerprint"
            ):
                raise TransformerEngineError("masked stage model fingerprint mismatch")
            if request.weight_digest and request.weight_digest != runtime.weight_digest:
                raise TransformerEngineError("masked stage weight commitment mismatch")
            if request.weight_bits and request.weight_bits != runtime.spec.weight_bits:
                raise TransformerEngineError("masked stage weight quantization mismatch")
            if request.activation_bits and request.activation_bits != runtime.spec.activation_bits:
                raise TransformerEngineError("masked stage activation quantization mismatch")
            if request.session_id and (
                request.out_features != runtime.spec.out_features
                or SeededRingProfile(
                    request.signed_output_bound,
                    request.ring or "prime",  # type: ignore[arg-type]
                    request.modulus,
                    request.wire_bits,
                ) != runtime.seeded_profile
            ):
                raise TransformerEngineError("masked stage seeded profile mismatch")

        row_counts = [request.masked_input.shape[0] for request in requests]
        combined = np.ascontiguousarray(
            requests[0].masked_input
            if len(requests) == 1
            else np.concatenate([r.masked_input for r in requests], axis=0),
            dtype=np.uint32,
        )
        started = time.perf_counter_ns()
        output = await asyncio.to_thread(
            runtime.compiled_weight.wrap32 if ring == "u32" else runtime.compiled_weight.modular,
            combined,
            *(() if ring == "u32" else ((modulus,) if wrapping_profile else (runtime.modulus,))),
        )
        elapsed = time.perf_counter_ns() - started
        runtime.calls += len(requests)
        runtime.rows += int(combined.shape[0])
        runtime.server_ns += elapsed

        results: list[bytes] = []
        offset = 0
        for request, count in zip(requests, row_counts):
            chunk = output[offset : offset + count]
            offset += count
            results.append(
                MaskedStageResponse(
                    correlation_id=request.correlation_id,
                    masked_output=chunk,
                    modulus=modulus,
                    wire_bits=wire_bits,
                    server_ns=int(elapsed * count / max(1, combined.shape[0])),
                    stage_id=stage.id,
                    ring=cast(RingKind, ring if wrapping_profile else "prime"),
                ).pack()
            )
        return results

    def validate_seeded_preparation(self, request: PreparationRequest) -> None:
        model = self._model(request.model)
        runtime = self._runtime(request.model, request.stage_id)
        if request.body_fingerprint != model.manifest.metadata.get("body_fingerprint"):
            raise TransformerEngineError("preparation model body fingerprint mismatch")
        if request.weight_digest != runtime.weight_digest:
            raise TransformerEngineError("preparation stage weight commitment mismatch")
        if request.in_features != runtime.spec.in_features:
            raise TransformerEngineError("preparation stage input width mismatch")
        if request.out_features != runtime.spec.out_features:
            raise TransformerEngineError("preparation stage output width mismatch")
        if (
            request.weight_bits != runtime.spec.weight_bits
            or request.activation_bits != runtime.spec.activation_bits
        ):
            raise TransformerEngineError("preparation stage quantization mismatch")
        if request.profile != runtime.seeded_profile:
            raise TransformerEngineError("preparation stage ring profile mismatch")

    async def prepare_seeded_stage(self, request: PreparationRequest) -> CorrectionPush:
        self.validate_seeded_preparation(request)
        runtime = self._runtime(request.model, request.stage_id)
        mask = expand_preparation_mask(request)
        output_mask = expand_output_mask(request)
        started = time.perf_counter_ns()
        transformed = await asyncio.to_thread(
            runtime.compiled_weight.wrap32 if request.ring == "u32" else runtime.compiled_weight.modular,
            mask,
            *(() if request.ring == "u32" else (request.modulus,)),
        )
        correction = (
            transformed.astype(np.int64) - output_mask.astype(np.int64)
        ) % request.modulus
        elapsed = time.perf_counter_ns() - started
        runtime.calls += 1
        runtime.rows += request.rows
        runtime.server_ns += elapsed
        return CorrectionPush(
            attempt_id=request.attempt_id,
            session_id=request.session_id,
            model=request.model,
            body_fingerprint=request.body_fingerprint,
            server_ns=elapsed,
            stage_id=request.stage_id,
            weight_digest=request.weight_digest,
            rows=request.rows,
            in_features=request.in_features,
            out_features=request.out_features,
            weight_bits=request.weight_bits,
            activation_bits=request.activation_bits,
            signed_output_bound=request.signed_output_bound,
            ring=request.ring,
            modulus=request.modulus,
            wire_bits=request.wire_bits,
            correction=correction.astype(np.uint32),
        )

    async def stage_metadata(self, model_id: str, stage_id: str) -> StageMetadata:
        return self._runtime(model_id, stage_id).metadata

    def seeded_profile(self, model_id: str, stage_id: str) -> SeededRingProfile:
        return self._runtime(model_id, stage_id).seeded_profile

    def seeded_stage_ids(self, model_id: str) -> tuple[str, ...]:
        model = self._model(model_id)
        return tuple(
            stage_id
            for stage_id in model.stages
            if stage_id not in {"token_lookup", "lm_head"}
        )

    def validate_seeded_correction(self, correction: CorrectionPush) -> None:
        model = self._model(correction.model)
        runtime = self._runtime(correction.model, correction.stage_id)
        if (
            correction.body_fingerprint != model.manifest.metadata.get("body_fingerprint")
            or correction.weight_digest != runtime.weight_digest
            or correction.in_features != runtime.spec.in_features
            or correction.out_features != runtime.spec.out_features
            or correction.weight_bits != runtime.spec.weight_bits
            or correction.activation_bits != runtime.spec.activation_bits
            or correction.profile != runtime.seeded_profile
        ):
            raise TransformerEngineError("correction stage metadata mismatch")

    def validate_seeded_activation(self, request: MaskedStageRequest) -> None:
        model = self._model(request.model)
        runtime = self._runtime(request.model, request.stage_id)
        if (
            request.body_fingerprint != model.manifest.metadata.get("body_fingerprint")
            or request.weight_digest != runtime.weight_digest
            or request.masked_input.shape[-1] != runtime.spec.in_features
            or request.out_features != runtime.spec.out_features
            or request.weight_bits != runtime.spec.weight_bits
            or request.activation_bits != runtime.spec.activation_bits
            or SeededRingProfile(
                request.signed_output_bound,
                request.ring or "prime",  # type: ignore[arg-type]
                request.modulus,
                request.wire_bits,
            )
            != runtime.seeded_profile
        ):
            raise TransformerEngineError("prepared activation stage metadata mismatch")

    def seeded_session_authorization(
        self,
        model_id: str,
        session_id: str,
        max_attempts: int,
    ) -> SessionAuthorization:
        model = self._model(model_id)
        remote = [
            runtime
            for stage_id, runtime in model.stages.items()
            if stage_id not in {"token_lookup", "lm_head"}
        ]
        weight_bits = {runtime.spec.weight_bits for runtime in remote}
        activation_bits = {runtime.spec.activation_bits for runtime in remote}
        if len(weight_bits) != 1 or len(activation_bits) != 1:
            raise TransformerEngineError("seeded stages require uniform quantization")
        return SessionAuthorization(
            session_id=session_id,
            model=model_id,
            body_fingerprint=str(model.manifest.metadata["body_fingerprint"]),
            stage_commitment=str(model.manifest.metadata["seeded_stage_commitment"]),
            weight_bits=next(iter(weight_bits)),
            activation_bits=next(iter(activation_bits)),
            max_attempts=max_attempts,
            rows=max_attempts // len(self.seeded_stage_ids(model_id)),
            stage_ids=self.seeded_stage_ids(model_id),
        )

    def validate_seeded_session_authorization(
        self,
        authorization: SessionAuthorization,
    ) -> None:
        model = self._model(authorization.model)
        max_budget = max(1, len(model.manifest.stages)) * (
            max(1, model.manifest.context_length) + 1
        )
        if authorization.max_attempts > max_budget:
            raise TransformerEngineError("session authorization attempt budget exceeds model limit")
        expected = self.seeded_session_authorization(
            authorization.model,
            authorization.session_id,
            authorization.max_attempts,
        )
        if authorization != expected:
            raise TransformerEngineError("session authorization model commitment mismatch")

    async def client_bundle_bytes(self, model_id: str) -> bytes:
        return self.client_bundle(model_id)

    def create_local_correlations(
        self,
        model_id: str,
        stage_id: str,
        count: int,
        *,
        ring: str | None = None,
        rows: int | None = None,
        seed: int | None = None,
    ) -> list[StageCorrelation]:
        runtime = self._runtime(model_id, stage_id)
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        lock = threading.Lock() if seed is not None else self._rng_lock
        output: list[StageCorrelation] = []
        with lock:
            for _ in range(int(count)):
                mask = rng.integers(
                    0, runtime.modulus, size=runtime.spec.in_features, dtype=np.uint32
                )
                transformed = runtime.compiled_weight.modular(
                    mask[None, :],
                    runtime.modulus,
                )[0]
                output.append(
                    StageCorrelation(
                        id=secrets.token_hex(12),
                        stage_id=stage_id,
                        mask=mask,
                        transformed_mask=transformed,
                        modulus=runtime.modulus,
                        ring="prime",
                    )
                )
        return output

    def register_bfv_context(self, model_id: str, context_id: str, public_context: bytes) -> None:
        model = self._model(model_id)
        tiled_modulus = tiled_context_modulus(public_context)
        if tiled_modulus is None:
            # Validate immediately and retain public bytes for lazy per-stage registration.
            first = next(iter(model.stages.values()))
            validator = BFVCorrelationServer(first.weight.values[:1], pydeps_path=self.tenseal_path)
            validator.register_context(context_id, public_context)
        else:
            if not any(stage.modulus == tiled_modulus for stage in model.stages.values()):
                raise TransformerEngineError("tiled BFV context modulus is not used by this model")
            try:
                TiledBFVServer(public_context, np.zeros((1, 1), dtype=np.int8))
            except TiledBFVError as exc:
                raise TransformerEngineError("invalid tiled BFV public context") from exc
        with model.bfv_lock:
            model.bfv_contexts[context_id] = public_context
            for key in tuple(model.bfv_servers):
                if key[1] == context_id:
                    del model.bfv_servers[key]

    def evaluate_bfv_correlation(
        self,
        model_id: str,
        stage_id: str,
        context_id: str,
        encrypted_mask: bytes,
    ) -> bytes:
        model = self._model(model_id)
        runtime = self._runtime(model_id, stage_id)
        with model.bfv_lock:
            public = model.bfv_contexts.get(context_id)
            if public is None:
                raise TransformerEngineError("unknown BFV context")
            tiled_modulus = tiled_context_modulus(public)
            if tiled_modulus is not None and tiled_modulus != runtime.modulus:
                raise TransformerEngineError("tiled BFV context modulus does not match stage")
            key = (stage_id, context_id)
            server = model.bfv_servers.get(key)
            if server is None:
                if tiled_context_modulus(public) is not None:
                    server = TiledBFVServer(
                        public, runtime.weight.values, threads=self.kernel.threads
                    )
                else:
                    server = BFVCorrelationServer(
                        runtime.weight.values, pydeps_path=self.tenseal_path
                    )
                    server.register_context(context_id, public)
                model.bfv_servers[key] = server
        if isinstance(server, TiledBFVServer):
            return server.evaluate_many([encrypted_mask])[0]
        raw = server.evaluate(context_id, encrypted_mask)
        value = msgpack.unpackb(raw, raw=False)
        return msgpack.packb(
            {
                "v": 1,
                "stage_id": stage_id,
                "out_features": runtime.spec.out_features,
                "ciphertexts": value["ciphertexts"],
            },
            use_bin_type=True,
        )

    def evaluate_bfv_correlations(
        self,
        model_id: str,
        stage_id: str,
        context_id: str,
        encrypted_masks: list[bytes],
        *,
        cancel_event: threading.Event | None = None,
    ) -> list[bytes]:
        if not encrypted_masks:
            return []
        model = self._model(model_id)
        runtime = self._runtime(model_id, stage_id)
        with model.bfv_lock:
            public = model.bfv_contexts.get(context_id)
            if public is None:
                raise TransformerEngineError("unknown BFV context")
            tiled_modulus = tiled_context_modulus(public)
            if tiled_modulus is not None and tiled_modulus != runtime.modulus:
                raise TransformerEngineError("tiled BFV context modulus does not match stage")
            key = (stage_id, context_id)
            server = model.bfv_servers.get(key)
            if server is None and tiled_context_modulus(public) is not None:
                server = TiledBFVServer(
                    public, runtime.weight.values, threads=self.kernel.threads
                )
                model.bfv_servers[key] = server
        if isinstance(server, TiledBFVServer):
            return server.evaluate_many(encrypted_masks, cancel_event=cancel_event)
        results = []
        for payload in encrypted_masks:
            if cancel_event is not None and cancel_event.is_set():
                raise TransformerEngineError("BFV evaluation cancelled")
            results.append(
                self.evaluate_bfv_correlation(model_id, stage_id, context_id, payload)
            )
        return results

    def client_bundle(self, model_id: str, *, include_local_weights: bool = True) -> bytes:
        model = self._model(model_id)
        local_stage_ids = {"token_lookup", "lm_head"} if include_local_weights else set()
        stage_descriptors = {
            sid: runtime.public_descriptor(
                include_weight=False,
                include_seeded_profile=include_local_weights and sid not in local_stage_ids,
            )
            for sid, runtime in model.stages.items()
        }
        client_weights: dict[str, dict[str, Any]] = {}

        def add_client_weight(weight_id: str, weight: QuantizedWeight) -> None:
            client_weights[weight_id] = {
                "dtype": "i1",
                "shape": list(weight.values.shape),
                "data": weight.values.astype(np.int8, copy=False).tobytes(),
                "scales": weight.scales.astype("<f4", copy=False).tobytes(),
            }

        if include_local_weights:
            token_lookup = model.stages["token_lookup"]
            lm_head = model.stages["lm_head"]
            tied = (
                model.manifest.tied_embeddings
                and token_lookup.source_keys
                and lm_head.source_keys
                and token_lookup.source_keys[0] == lm_head.source_keys[0]
            )
            if tied:
                add_client_weight("tied_embeddings", lm_head.weight)
                stage_descriptors["lm_head"]["client_weight"] = {
                    "ref": "tied_embeddings",
                    "layout": "linear",
                }
                stage_descriptors["token_lookup"]["client_weight"] = {
                    "ref": "tied_embeddings",
                    "layout": "embedding",
                }
                auxiliary_width = token_lookup.spec.out_features - lm_head.spec.in_features
                if auxiliary_width:
                    auxiliary = QuantizedWeight(
                        np.ascontiguousarray(token_lookup.weight.values[-auxiliary_width:]),
                        np.ascontiguousarray(token_lookup.weight.scales[-auxiliary_width:]),
                        token_lookup.weight.bits,
                    )
                    add_client_weight("token_lookup_aux", auxiliary)
                    stage_descriptors["token_lookup"]["client_aux_weight"] = {
                        "ref": "token_lookup_aux"
                    }
            else:
                for stage_id, runtime in (("token_lookup", token_lookup), ("lm_head", lm_head)):
                    add_client_weight(stage_id, runtime.weight)
                    stage_descriptors[stage_id]["client_weight"] = {
                        "ref": stage_id,
                        "layout": (
                            "transposed_embedding" if stage_id == "token_lookup" else "linear"
                        ),
                    }
        tensors = {
            key: {
                "shape": list(value.shape),
                "dtype": "f4",
                "data": value.astype("<f4", copy=False).tobytes(),
            }
            for key, value in model.local_tensors.items()
        }
        config = dict(model.config)
        norm_offset = float(config.get("norm_offset", 0.0))
        block_style = str(config.get("block_style", "llama"))
        config.update(
            {
                "rms_norm_centered": bool(norm_offset),
                "norm_offset": norm_offset,
                "embedding_multiplier": float(config.get("embedding_multiplier", 1.0)),
                "block_style": block_style,
                "model_family": config.get("model_family", "llama-compatible"),
                "qk_norm": block_style == "gemma4"
                or any(
                    key.endswith(("q_norm.weight", "k_norm.weight")) for key in model.local_tensors
                ),
                "v_norm": block_style == "gemma4",
                "prime_modulus": max(runtime.modulus for runtime in model.stages.values()),
                "plain_moduli": sorted({runtime.modulus for runtime in model.stages.values()}),
                "attention_scaling": config.get("attention_scaling"),
            }
        )
        return msgpack.packb(
            {
                "v": 2,
                "runtime": "masked_transformer",
                "model": model_id,
                "manifest": model.manifest.to_dict(),
                "config": config,
                "tokenizer": model.tokenizer,
                "stages": stage_descriptors,
                "local_tensors": tensors,
                "client_weights": client_weights,
                "privacy": {
                    "mode": "public",
                    "protocol": f"masked_w{self.weight_bits}a{self.activation_bits}",
                    "online_fhe": False,
                    "preprocessed": True,
                    "model_weight_correlations_disclosed": True,
                    "dense_weights_in_bundle": bool(local_stage_ids),
                    "local_quantized_stages": sorted(local_stage_ids),
                    "tied_embedding_quantization": (
                        "per_token_row_v1" if model.manifest.tied_embeddings else None
                    ),
                    "token_ids_remote": False,
                    "plaintext_activations_remote": False,
                    "plaintext_logits_remote": False,
                    "client_intermediate_activations": True,
                    "model_privacy_threat_model": "public_weights",
                    "body_fingerprint": model.manifest.metadata["body_fingerprint"],
                    "stage_commitment": model.manifest.metadata["seeded_stage_commitment"],
                    "weight_bits": self.weight_bits,
                    "activation_bits": self.activation_bits,
                },
            },
            use_bin_type=True,
        )

    def metrics(self) -> dict[str, Any]:
        return {
            model_id: {
                "stages": {
                    stage_id: {
                        "calls": runtime.calls,
                        "rows": runtime.rows,
                        "server_ms": runtime.server_ns / 1e6,
                        "rows_per_second": (
                            runtime.rows / (runtime.server_ns / 1e9) if runtime.server_ns else None
                        ),
                    }
                    for stage_id, runtime in model.stages.items()
                }
            }
            for model_id, model in self.models.items()
        }

    def stats(self) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "execute_calls": sum(
                s["calls"] for m in metrics.values() for s in m["stages"].values()
            ),
            "execute_items": sum(s["rows"] for m in metrics.values() for s in m["stages"].values()),
            "models": metrics,
        }

    def _model(self, model_id: str) -> LoadedTransformer:
        model = self.models.get(model_id)
        if model is None:
            raise TransformerEngineError(f"unknown model {model_id!r}")
        return model

    def _runtime(self, model_id: str, stage_id: str) -> StageRuntime:
        model = self._model(model_id)
        runtime = model.stages.get(stage_id)
        if runtime is None:
            raise TransformerEngineError(f"unknown stage {stage_id!r}")
        return runtime


# Stable public aliases used by the Round 8 tests and external integrations.
SafetensorsW4A4Engine = MaskedTransformerEngine
