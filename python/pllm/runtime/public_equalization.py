"""Source-bound public offline calibration for per-channel linear equalization."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import msgpack
import numpy as np


PROFILE_VERSION = "pllm.public_equalization.v1"
ALPHA_NUMERATOR = 3
ALPHA_DENOMINATOR = 4
MAX_PROFILE_BYTES = 16 * 1024 * 1024
MAX_STAGES = 1024
MAX_CALIBRATION_SEQUENCES = 8
MAX_CALIBRATION_TOKENS = 128
MIN_SCALE = np.float32(1 / 64)
MAX_SCALE = np.float32(64)


class PublicEqualizationError(ValueError):
    pass


def validate_input_scale(scale: np.ndarray, in_features: int) -> np.ndarray:
    raw = np.asarray(scale)
    if (
        raw.dtype != np.float32
        or raw.shape != (in_features,)
        or not np.all(np.isfinite(raw))
        or np.any(raw < MIN_SCALE)
        or np.any(raw > MAX_SCALE)
    ):
        raise PublicEqualizationError("equalization requires a bounded float32 input scale")
    return np.ascontiguousarray(raw, dtype="<f4")


def equalize_activation(activation: np.ndarray, scale: np.ndarray) -> np.ndarray:
    value = np.asarray(activation, dtype=np.float32)
    validate_input_scale(scale, int(value.shape[-1]))
    if not np.all(np.isfinite(value)):
        raise PublicEqualizationError("equalization requires finite activations")
    result = value / scale
    if not np.all(np.isfinite(result)):
        raise PublicEqualizationError("equalization exceeded the float32 domain")
    return np.ascontiguousarray(result, dtype=np.float32)


def equalize_weight_chunk(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2:
        raise PublicEqualizationError("equalization requires a weight matrix")
    validate_input_scale(scale, value.shape[1])
    if not np.all(np.isfinite(value)):
        raise PublicEqualizationError("equalization requires finite weights")
    result = value * scale[None, :]
    if not np.all(np.isfinite(result)):
        raise PublicEqualizationError("equalized weight exceeds the float32 domain")
    return np.ascontiguousarray(result, dtype=np.float32)


def fit_input_scale(activation_max: np.ndarray, weight_max: np.ndarray) -> np.ndarray:
    """Bounded SmoothQuant-style weight/activation redistribution, fixed α=3/4."""
    activation = np.asarray(activation_max, dtype=np.float32)
    weight = np.asarray(weight_max, dtype=np.float32)
    if (
        activation.ndim != 1
        or activation.shape != weight.shape
        or not activation.size
        or not np.all(np.isfinite(activation))
        or not np.all(np.isfinite(weight))
        or np.any(activation < 0)
        or np.any(weight < 0)
    ):
        raise PublicEqualizationError("calibration maxima must be finite nonnegative vectors")
    floor = np.float32(1e-6)
    numerator = np.maximum(activation, floor) ** (ALPHA_NUMERATOR / ALPHA_DENOMINATOR)
    denominator = np.maximum(weight, floor) ** (1 - ALPHA_NUMERATOR / ALPHA_DENOMINATOR)
    raw = numerator / denominator
    normalizer = np.exp(np.mean(np.log(raw), dtype=np.float64))
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise PublicEqualizationError("calibration normalization is invalid")
    scale = np.clip(raw / normalizer, MIN_SCALE, MAX_SCALE).astype(np.float32)
    return validate_input_scale(scale, activation.size)


@dataclass(frozen=True, slots=True)
class PublicEqualizationProfile:
    source_lock_digest: str
    calibration_tokens: tuple[tuple[int, ...], ...]
    stage_scales: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if type(self.source_lock_digest) is not str or len(self.source_lock_digest) != 64 or any(c not in "0123456789abcdef" for c in self.source_lock_digest):
            raise PublicEqualizationError("profile requires a source-lock digest")
        if not 1 <= len(self.calibration_tokens) <= MAX_CALIBRATION_SEQUENCES or any(
            not 1 <= len(row) <= MAX_CALIBRATION_TOKENS
            or any(type(token) is not int or token < 0 or token >= 1 << 31 for token in row)
            for row in self.calibration_tokens
        ):
            raise PublicEqualizationError("profile requires bounded public calibration tokens")
        if not 1 <= len(self.stage_scales) <= MAX_STAGES:
            raise PublicEqualizationError("profile requires bounded body stages")
        frozen = {}
        for stage_id, scale in self.stage_scales.items():
            if type(stage_id) is not str or not stage_id or len(stage_id) > 256:
                raise PublicEqualizationError("profile has an invalid stage ID")
            scale = np.asarray(scale)
            if scale.ndim != 1:
                raise PublicEqualizationError("profile stage scale must be rank one")
            frozen[stage_id] = np.frombuffer(
                validate_input_scale(scale, scale.size).tobytes(), dtype="<f4",
            )
        object.__setattr__(self, "stage_scales", MappingProxyType(frozen))

    def pack(self) -> bytes:
        payload = msgpack.packb({
            "v": PROFILE_VERSION,
            "source_lock_digest": self.source_lock_digest,
            "alpha_numerator": ALPHA_NUMERATOR,
            "alpha_denominator": ALPHA_DENOMINATOR,
            "calibration_tokens": [list(row) for row in self.calibration_tokens],
            "stage_scales": {
                key: self.stage_scales[key].astype("<f4", copy=False).tobytes()
                for key in sorted(self.stage_scales)
            },
        }, use_bin_type=True)
        if len(payload) > MAX_PROFILE_BYTES:
            raise PublicEqualizationError("calibration profile exceeds 16 MiB")
        return payload

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.pack()).hexdigest()

    @property
    def calibration_digest(self) -> str:
        return hashlib.sha256(
            b"pllm.public_calibration_tokens.v1\0"
            + msgpack.packb(self.calibration_tokens, use_bin_type=True)
        ).hexdigest()

    @classmethod
    def unpack(cls, payload: bytes, *, expected_digest: str | None = None) -> "PublicEqualizationProfile":
        if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_PROFILE_BYTES:
            raise PublicEqualizationError("calibration profile exceeds its byte budget")
        if expected_digest is not None and hashlib.sha256(payload).hexdigest() != expected_digest:
            raise PublicEqualizationError("calibration profile digest mismatch")
        try:
            record = msgpack.unpackb(payload, raw=False, strict_map_key=True)
            if not isinstance(record, dict) or set(record) != {
                "v", "source_lock_digest", "alpha_numerator", "alpha_denominator",
                "calibration_tokens", "stage_scales",
            } or (record["v"], record["alpha_numerator"], record["alpha_denominator"]) != (
                PROFILE_VERSION, ALPHA_NUMERATOR, ALPHA_DENOMINATOR,
            ):
                raise PublicEqualizationError("invalid calibration profile schema")
            scales = record["stage_scales"]
            if not isinstance(scales, dict) or not 1 <= len(scales) <= MAX_STAGES:
                raise PublicEqualizationError("invalid calibration stage inventory")
            parsed = {
                stage_id: np.frombuffer(raw, dtype="<f4").copy()
                for stage_id, raw in scales.items()
                if type(stage_id) is str and type(raw) is bytes and len(raw) % 4 == 0
            }
            if len(parsed) != len(scales):
                raise PublicEqualizationError("invalid calibration stage scale bytes")
            profile = cls(
                record["source_lock_digest"],
                tuple(tuple(row) for row in record["calibration_tokens"]),
                parsed,
            )
        except (KeyError, TypeError, ValueError, msgpack.ExtraData, msgpack.FormatError) as exc:
            raise PublicEqualizationError("invalid calibration profile") from exc
        if profile.pack() != payload:
            raise PublicEqualizationError("noncanonical calibration profile")
        return profile


def profile_path(root: str | Path, digest: str) -> Path:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PublicEqualizationError("calibration digest must be lowercase SHA-256")
    return Path(root) / f"pllm-public-equalization-{digest}.msgpack"


def load_public_equalization_profile(
    root: str | Path, digest: str, source_lock_digest: str,
) -> PublicEqualizationProfile:
    path = profile_path(root, digest)
    if not path.is_file() or path.stat().st_size > MAX_PROFILE_BYTES:
        raise PublicEqualizationError("declared public calibration profile is unavailable")
    profile = PublicEqualizationProfile.unpack(path.read_bytes(), expected_digest=digest)
    if profile.source_lock_digest != source_lock_digest:
        raise PublicEqualizationError("calibration profile belongs to a different checkpoint")
    return profile


def fit_public_equalization_profile(
    source: Any,
    calibration_tokens: tuple[tuple[int, ...], ...],
    *,
    threads: int = 4,
) -> PublicEqualizationProfile:
    """Fit only from caller-declared public offline tokens and a locked source."""
    from pllm import load_model, lower_model
    from pllm.profiles import MaskedLinearCpu

    from .model_binding import compile_runtime_model
    from .safetensors_store import SafeTensorStore
    from .transformer_client import ClientBundle
    from .transformer_engine import MaskedTransformerEngine

    manifest = load_model(source)
    lock = manifest.source_lock_digest
    if lock is None:
        raise PublicEqualizationError("calibration requires a locked checkpoint")
    # Validate the cohort before allocating checkpoint weights or issuing work.
    PublicEqualizationProfile(
        lock, calibration_tokens, {"preflight": np.ones(1, dtype=np.float32)},
    )
    root = Path(manifest.source)
    config = (root / "config.json").read_bytes()
    bound = max(map(len, calibration_tokens))
    plan = lower_model(config, batch=1, max_input_tokens=bound, max_new_tokens=1)
    composition = MaskedLinearCpu(source)
    if not plan.runtime_schedule(composition).complete:
        raise PublicEqualizationError("calibration requires a complete baseline schedule")
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=threads)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    stages = engine.models[manifest.id].stages
    store = SafeTensorStore(root)
    input_max: dict[str, np.ndarray] = {}
    weight_max: dict[str, np.ndarray] = {}

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = stages[stage_id]
        value = np.asarray(activation, dtype=np.float32)
        observed = np.max(np.abs(value), axis=0)
        prior = input_max.get(stage_id)
        input_max[stage_id] = observed if prior is None else np.maximum(prior, observed)
        sources = [store.get(key, dtype=np.float32) for key in stage.spec.weight_keys]
        weight = sources[0] if len(sources) == 1 else np.concatenate(sources)
        if stage_id not in weight_max:
            weight_max[stage_id] = np.max(np.abs(weight), axis=0)
        output = np.ascontiguousarray(value @ weight.T, dtype=np.float32)
        del sources, weight
        store.drop(*stage.spec.weight_keys)
        if stage.bias is not None:
            output += stage.bias
        return output

    for ids in calibration_tokens:
        compiled.runtime(remote).prepare_ids(list(ids))
    body_stages = set(stages) - {"token_lookup", "lm_head"}
    if set(input_max) != body_stages or set(weight_max) != body_stages:
        raise PublicEqualizationError("public calibration did not cover every body stage")
    scales = {stage_id: fit_input_scale(input_max[stage_id], weight_max[stage_id]) for stage_id in sorted(body_stages)}
    del engine, compiled, bundle
    gc.collect()
    return PublicEqualizationProfile(lock, calibration_tokens, scales)
