from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np

from .engine import EngineCapabilities
from .models import ModelManifest, StageSpec
from .stage_protocol import (
    BlindedStageCorrelation,
    BlindedStageRequest,
    BlindedStageResponse,
)
from .transformer_engine import MaskedTransformerEngine, TransformerEngineError


@dataclass(frozen=True, slots=True)
class _ServerBlind:
    model_id: str
    owner_id: str
    stage_id: str
    correlation_id: str
    created_ns: int


class BlindedTransformerEngine(MaskedTransformerEngine):
    """Fast proprietary-weight mode using output-blinded HE preprocessing.

    Offline, the client chooses ``r`` and receives only ``W r + s`` while the
    server retains the secret and one-time record needed to derive the fresh output mask ``s``. Online, the client sends
    ``x-r`` and the server returns ``W(x-r)-s``. Adding the two client-visible
    values yields ``W x``. Unlike public mode, the client never receives a plain
    ``(r, W r)`` equation. Unlike direct BFV, the latency-sensitive online path
    is ordinary exact modular GEMM.

    This removes passive correlation leakage but still assumes an honest client
    runtime: a modified client that can issue arbitrary valid stage queries can
    use the stage as a chosen-input oracle. Strong malicious-client model
    privacy requires authenticated secret shares and secure nonlinearities.
    """

    capabilities = EngineCapabilities(
        name="blinded-transformer-proprietary",
        model_sources=("huggingface", "safetensors", "vllm", "mlx-lm"),
        protocols=("blinded-ole.stage/v1", "bfv-blinded-correlation/v1"),
        online_fhe=False,
        preprocessed=True,
        continuous_batching=True,
        notes=(
            "all learned dense matrices remain server-side",
            "HE-preprocessed output-blinded OLE-style correlations",
            "client never receives unblinded (r, W r) equations",
            "fast exact modular online GEMM",
            "model privacy assumes an honest client runtime",
        ),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._blinds: dict[tuple[str, str, str, str], _ServerBlind] = {}
        self._blind_lock = threading.RLock()
        self._blind_rng = np.random.default_rng(secrets.randbits(128))
        self._blind_key = secrets.token_bytes(32)
        self._blind_generated = 0
        self._blind_consumed = 0

    async def load(self, manifest: ModelManifest) -> None:
        await super().load(manifest)
        loaded = self.models[manifest.id]
        loaded.manifest.metadata.update(
            {
                "privacy_mode": "proprietary",
                "privacy_protocol": "blinded_ole_w4a4",
                "client_runtime": "blinded_ole_transformer_v1",
                "online_fhe": False,
                "preprocessed": True,
                "model_weight_correlations_disclosed": False,
                "output_blinded_correlations": True,
                "model_privacy_threat_model": "honest_but_curious_client",
                "malicious_client_model_privacy": False,
            }
        )

    def _derive_output_mask(
        self,
        model_id: str,
        owner_id: str,
        stage_id: str,
        correlation_id: str,
        modulus: int,
        out_features: int,
    ) -> np.ndarray:
        """Expand a server-secret correlation label into pseudorandom residues.

        Only a one-time label is retained per correlation. Using 64-bit rejection
        sampling maps pseudorandom words uniformly into the ring while avoiding
        a materialized output-mask inventory.
        """
        label = b"\0".join(
            item.encode("utf-8")
            for item in ("pllm-blind-v1", model_id, owner_id, stage_id, correlation_id)
        )
        seed = hmac.new(self._blind_key, label, hashlib.sha256).digest()
        limit = (1 << 64) - ((1 << 64) % int(modulus))
        values: list[np.ndarray] = []
        remaining = int(out_features)
        counter = 0
        while remaining:
            # Rejection is extremely rare for the small stage moduli used here,
            # but the loop makes the distribution exact over Z_p.
            take = max(remaining + 8, 32)
            raw = hashlib.shake_256(seed + counter.to_bytes(8, "little")).digest(take * 8)
            candidates = np.frombuffer(raw, dtype="<u8")
            candidates = candidates[candidates < limit]
            if candidates.size:
                chunk = (candidates[:remaining] % int(modulus)).astype(np.uint32)
                values.append(chunk)
                remaining -= int(chunk.size)
            counter += 1
        return np.concatenate(values)

    def _store_blind(
        self,
        model_id: str,
        owner_id: str,
        stage_id: str,
        correlation_id: str,
    ) -> None:
        key = (model_id, owner_id, stage_id, correlation_id)
        blind = _ServerBlind(
            model_id=model_id,
            owner_id=owner_id,
            stage_id=stage_id,
            correlation_id=correlation_id,
            created_ns=time.time_ns(),
        )
        with self._blind_lock:
            if key in self._blinds:
                raise TransformerEngineError("duplicate proprietary correlation id")
            self._blinds[key] = blind
            self._blind_generated += 1

    def _consume_blinds(
        self,
        model_id: str,
        owner_id: str,
        stage_id: str,
        correlation_ids: tuple[str, ...],
    ) -> np.ndarray:
        if len(set(correlation_ids)) != len(correlation_ids):
            raise TransformerEngineError("proprietary correlation ids must be unique")
        keys = [(model_id, owner_id, stage_id, item) for item in correlation_ids]
        with self._blind_lock:
            missing = [item[-1] for item in keys if item not in self._blinds]
            if missing:
                raise TransformerEngineError(
                    "unknown, expired, or already consumed proprietary correlation"
                )
            # Pop before computation. A failed or cancelled request burns the
            # label rather than risking reuse after a retry.
            records = [self._blinds.pop(item) for item in keys]
            self._blind_consumed += len(records)
        runtime = self._runtime(model_id, stage_id)
        return np.stack([
            self._derive_output_mask(
                row.model_id,
                row.owner_id,
                row.stage_id,
                row.correlation_id,
                runtime.modulus,
                runtime.spec.out_features,
            )
            for row in records
        ])

    def create_local_blinded_correlations(
        self,
        model_id: str,
        stage_id: str,
        owner_id: str,
        count: int,
        *,
        seed: int | None = None,
    ) -> list[BlindedStageCorrelation]:
        """Insecure deterministic test helper; the server sees both masks."""
        runtime = self._runtime(model_id, stage_id)
        rng = np.random.default_rng(seed) if seed is not None else self._blind_rng
        output: list[BlindedStageCorrelation] = []
        lock = threading.Lock() if seed is not None else self._blind_lock
        with lock:
            for _ in range(int(count)):
                mask = rng.integers(
                    0, runtime.modulus, size=runtime.spec.in_features, dtype=np.uint32
                )
                correlation_id = secrets.token_hex(12)
                output_mask = self._derive_output_mask(
                    model_id, owner_id, stage_id, correlation_id,
                    runtime.modulus, runtime.spec.out_features,
                )
                wr = runtime.compiled_weight.modular(mask[None, :], runtime.modulus)[0]
                blinded = (wr.astype(np.uint64) + output_mask.astype(np.uint64)) % runtime.modulus
                self._store_blind(model_id, owner_id, stage_id, correlation_id)
                output.append(
                    BlindedStageCorrelation(
                        id=correlation_id,
                        stage_id=stage_id,
                        owner_id=owner_id,
                        mask=mask,
                        blinded_transformed_mask=blinded.astype(np.uint32),
                        modulus=runtime.modulus,
                        ring="prime",
                    )
                )
        return output

    def evaluate_bfv_blinded_correlation(
        self,
        model_id: str,
        stage_id: str,
        owner_id: str,
        context_id: str,
        encrypted_mask: bytes,
    ) -> bytes:
        model = self._model(model_id)
        runtime = self._runtime(model_id, stage_id)
        correlation_id = secrets.token_hex(12)
        output_mask = self._derive_output_mask(
            model_id, owner_id, stage_id, correlation_id,
            runtime.modulus, runtime.spec.out_features,
        )
        with model.bfv_lock:
            public = model.bfv_contexts.get(context_id)
            if public is None:
                raise TransformerEngineError("unknown BFV context")
            key = (stage_id, context_id)
            server = model.bfv_servers.get(key)
            if server is None:
                from .bfv_correlations import BFVCorrelationServer

                server = BFVCorrelationServer(runtime.weight.values, pydeps_path=self.tenseal_path)
                server.register_context(context_id, public)
                model.bfv_servers[key] = server
        raw = server.evaluate(
            context_id,
            encrypted_mask,
            output_mask=output_mask,
        )
        value = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        self._store_blind(model_id, owner_id, stage_id, correlation_id)
        return msgpack.packb(
            {
                "v": 1,
                "stage_id": stage_id,
                "correlation_id": correlation_id,
                "out_features": runtime.spec.out_features,
                "ciphertexts": value["ciphertexts"],
            },
            use_bin_type=True,
        )

    def evaluate_bfv_blinded_correlations(
        self,
        model_id: str,
        stage_id: str,
        owner_id: str,
        context_id: str,
        encrypted_masks: list[bytes],
    ) -> list[bytes]:
        return [
            self.evaluate_bfv_blinded_correlation(
                model_id, stage_id, owner_id, context_id, payload
            )
            for payload in encrypted_masks
        ]

    async def execute_stage(
        self,
        model_id: str,
        stage: StageSpec,
        payloads: list[bytes],
    ) -> list[bytes]:
        runtime = self._runtime(model_id, stage.id)
        requests = [BlindedStageRequest.unpack(payload) for payload in payloads]
        if not requests:
            return []
        for request in requests:
            if request.model != model_id or request.stage_id != stage.id:
                raise TransformerEngineError("blinded stage request route mismatch")
            if request.modulus != runtime.modulus or request.wire_bits != runtime.wire_bits:
                raise TransformerEngineError("blinded stage arithmetic profile mismatch")
            if (request.ring or "prime") != "prime":
                raise TransformerEngineError("blinded stage currently requires a prime ring")
            if request.masked_input.ndim != 2 or request.masked_input.shape[1] != runtime.spec.in_features:
                raise TransformerEngineError("blinded stage input width mismatch")
            if len(request.correlation_ids) != request.masked_input.shape[0]:
                raise TransformerEngineError("blinded stage row/correlation mismatch")

        row_counts = [request.masked_input.shape[0] for request in requests]
        combined = np.ascontiguousarray(
            np.concatenate([request.masked_input for request in requests], axis=0),
            dtype=np.uint32,
        )
        # Consume all output masks before executing the matrix operation.
        output_masks = np.concatenate(
            [
                self._consume_blinds(
                    model_id,
                    request.owner_id,
                    stage.id,
                    request.correlation_ids,
                )
                for request in requests
            ],
            axis=0,
        )
        started = time.perf_counter_ns()
        output = await asyncio.to_thread(
            runtime.compiled_weight.modular, combined, runtime.modulus
        )
        output = (
            output.astype(np.int64) - output_masks.astype(np.int64)
        ) % runtime.modulus
        output = output.astype(np.uint32)
        elapsed = time.perf_counter_ns() - started
        runtime.calls += len(requests)
        runtime.rows += int(combined.shape[0])
        runtime.server_ns += elapsed

        results: list[bytes] = []
        offset = 0
        for request, count in zip(requests, row_counts, strict=True):
            chunk = output[offset : offset + count]
            offset += count
            results.append(
                BlindedStageResponse(
                    stage_id=stage.id,
                    correlation_ids=request.correlation_ids,
                    masked_output=chunk,
                    modulus=runtime.modulus,
                    wire_bits=runtime.wire_bits,
                    server_ns=int(elapsed * count / max(1, combined.shape[0])),
                    ring="prime",
                ).pack()
            )
        return results

    def create_local_correlations(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise TransformerEngineError(
            "proprietary mode does not expose unblinded model-dependent correlations"
        )

    def evaluate_bfv_correlation(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise TransformerEngineError(
            "proprietary mode requires output-blinded correlation generation"
        )

    def evaluate_bfv_correlations(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise TransformerEngineError(
            "proprietary mode requires output-blinded correlation generation"
        )

    def client_bundle(
        self, model_id: str, *, include_local_weights: bool = False
    ) -> bytes:
        value = msgpack.unpackb(
            super().client_bundle(model_id, include_local_weights=False),
            raw=False,
            strict_map_key=False,
        )
        value["runtime"] = "blinded_ole_transformer"
        value["privacy"] = {
            **dict(value.get("privacy") or {}),
            "mode": "proprietary",
            "protocol": "blinded_ole_w4a4",
            "online_fhe": False,
            "preprocessed": True,
            "model_weight_correlations_disclosed": False,
            "output_blinded_correlations": True,
            "client_intermediate_activations": True,
            "model_privacy_threat_model": "honest_but_curious_client",
            "malicious_client_model_privacy": False,
            "direct_fhe_reference_available": True,
            "dense_weights_in_bundle": False,
            "local_quantized_stages": [],
        }
        return msgpack.packb(value, use_bin_type=True)

    def proprietary_inventory_stats(self) -> dict[str, int]:
        with self._blind_lock:
            return {
                "available": len(self._blinds),
                "generated": self._blind_generated,
                "consumed": self._blind_consumed,
                "materialized_output_mask_bytes": 0,
                "derived_blind_records": len(self._blinds),
            }
