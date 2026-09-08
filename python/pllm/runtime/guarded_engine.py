from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .blinded_engine import BlindedTransformerEngine
from .engine import EngineCapabilities
from .models import ModelManifest, StageSpec
from .stage_protocol import BlindedStageRequest, BlindedStageResponse
from .transformer_engine import TransformerEngineError


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    """Extraction-hardening controls for the fast proprietary protocol.

    These controls reduce bulk chosen-input access. They do not replace a
    malicious-secure MPC protocol because a permitted query still reveals a
    complete intermediate stage result to the client.
    """

    max_rows_per_request: int = 4096
    max_rows_per_owner_stage: int = 16384
    max_requests_per_owner_minute: int = 4096
    output_dither_bound: int = 0

    def __post_init__(self) -> None:
        if self.max_rows_per_request <= 0:
            raise ValueError("max_rows_per_request must be positive")
        if self.max_rows_per_owner_stage <= 0:
            raise ValueError("max_rows_per_owner_stage must be positive")
        if self.max_requests_per_owner_minute <= 0:
            raise ValueError("max_requests_per_owner_minute must be positive")
        if self.output_dither_bound < 0:
            raise ValueError("output_dither_bound must be non-negative")


class GuardedBlindedTransformerEngine(BlindedTransformerEngine):
    """Fast proprietary inference with explicit extraction hardening.

    The cryptographic core is the output-blinded protocol from
    :class:`BlindedTransformerEngine`. This wrapper adds per-owner query
    budgets, bounded row counts, audit counters and optional integer dither.
    The result is useful operational hardening, but it is intentionally labelled
    *not malicious secure*. A modified client can still learn stage outputs for
    the requests that the policy permits.
    """

    capabilities = EngineCapabilities(
        name="guarded-transformer-proprietary",
        model_sources=("huggingface", "safetensors", "vllm", "mlx-lm"),
        protocols=("guarded-blinded.stage/v1", "bfv-blinded-correlation/v1"),
        online_fhe=False,
        he_preprocessed=True,
        continuous_batching=True,
        notes=(
            "all learned dense matrices remain server-side",
            "output-blinded HE preprocessing",
            "per-owner extraction budgets and request limits",
            "optional bounded output dither",
            "extraction hardened, not simulation-secure against a malicious client",
        ),
    )

    def __init__(
        self,
        *args: Any,
        guard_policy: GuardPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guard_policy = guard_policy or GuardPolicy()
        self._guard_lock = threading.RLock()
        self._owner_stage_rows: dict[tuple[str, str, str], int] = defaultdict(int)
        self._owner_request_times: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._guard_rejected = 0
        self._guard_accepted = 0
        self._dither_rng = np.random.default_rng(secrets.randbits(128))

    async def load(self, manifest: ModelManifest) -> None:
        await super().load(manifest)
        loaded = self.models[manifest.id]
        loaded.manifest.metadata.update(
            {
                "privacy_protocol": "guarded_blinded_w4a4",
                "client_runtime": "guarded_blinded_transformer_v1",
                "model_privacy_threat_model": "extraction_hardened_honest_client",
                "malicious_client_model_privacy": False,
                "guard_policy": {
                    "max_rows_per_request": self.guard_policy.max_rows_per_request,
                    "max_rows_per_owner_stage": self.guard_policy.max_rows_per_owner_stage,
                    "max_requests_per_owner_minute": self.guard_policy.max_requests_per_owner_minute,
                    "output_dither_bound": self.guard_policy.output_dither_bound,
                },
            }
        )

    def _admit(self, model_id: str, requests: list[BlindedStageRequest]) -> None:
        now = time.monotonic()
        policy = self.guard_policy
        with self._guard_lock:
            # Validate the complete batch before mutating counters.
            increments: dict[tuple[str, str, str], int] = defaultdict(int)
            request_increments: dict[tuple[str, str], int] = defaultdict(int)
            for request in requests:
                rows = int(request.masked_input.shape[0])
                if rows > policy.max_rows_per_request:
                    self._guard_rejected += 1
                    raise TransformerEngineError("guarded proprietary row limit exceeded")
                key = (model_id, request.owner_id, request.stage_id)
                increments[key] += rows
                request_increments[(model_id, request.owner_id)] += 1
            for key, rows in increments.items():
                if self._owner_stage_rows[key] + rows > policy.max_rows_per_owner_stage:
                    self._guard_rejected += 1
                    raise TransformerEngineError("guarded proprietary owner/stage budget exhausted")
            for owner_key, count in request_increments.items():
                queue = self._owner_request_times[owner_key]
                while queue and now - queue[0] >= 60.0:
                    queue.popleft()
                if len(queue) + count > policy.max_requests_per_owner_minute:
                    self._guard_rejected += 1
                    raise TransformerEngineError("guarded proprietary request-rate limit exceeded")
            for key, rows in increments.items():
                self._owner_stage_rows[key] += rows
            for owner_key, count in request_increments.items():
                self._owner_request_times[owner_key].extend([now] * count)
            self._guard_accepted += len(requests)

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
        self._admit(model_id, requests)
        for request in requests:
            if request.model != model_id or request.stage_id != stage.id:
                raise TransformerEngineError("guarded stage request route mismatch")
            if request.modulus != runtime.modulus or request.wire_bits != runtime.wire_bits:
                raise TransformerEngineError("guarded stage arithmetic profile mismatch")
            if (request.ring or "prime") != "prime":
                raise TransformerEngineError("guarded stage currently requires a prime ring")
            if request.masked_input.ndim != 2 or request.masked_input.shape[1] != runtime.spec.in_features:
                raise TransformerEngineError("guarded stage input width mismatch")
            if len(request.correlation_ids) != request.masked_input.shape[0]:
                raise TransformerEngineError("guarded stage row/correlation mismatch")

        row_counts = [request.masked_input.shape[0] for request in requests]
        combined = np.ascontiguousarray(
            np.concatenate([request.masked_input for request in requests], axis=0),
            dtype=np.uint32,
        )
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
        output = (output.astype(np.int64) - output_masks.astype(np.int64)) % runtime.modulus
        bound = self.guard_policy.output_dither_bound
        if bound:
            output = (
                output
                + self._dither_rng.integers(
                    -bound, bound + 1, size=output.shape, dtype=np.int64
                )
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

    def client_bundle(self, model_id: str) -> bytes:
        import msgpack

        value = msgpack.unpackb(super().client_bundle(model_id), raw=False, strict_map_key=False)
        value["runtime"] = "guarded_blinded_transformer"
        privacy = dict(value.get("privacy") or {})
        privacy.update(
            {
                "protocol": "guarded_blinded_w4a4",
                "model_privacy_threat_model": "extraction_hardened_honest_client",
                "malicious_client_model_privacy": False,
                "formal_security_profile": "none",
                "warning": (
                    "Guarding limits extraction volume but does not hide permitted intermediate "
                    "stage outputs from a modified client."
                ),
                "guard_policy": {
                    "max_rows_per_request": self.guard_policy.max_rows_per_request,
                    "max_rows_per_owner_stage": self.guard_policy.max_rows_per_owner_stage,
                    "max_requests_per_owner_minute": self.guard_policy.max_requests_per_owner_minute,
                    "output_dither_bound": self.guard_policy.output_dither_bound,
                },
            }
        )
        value["privacy"] = privacy
        return msgpack.packb(value, use_bin_type=True)

    def guard_stats(self) -> dict[str, int]:
        with self._guard_lock:
            return {
                "accepted_requests": self._guard_accepted,
                "rejected_requests": self._guard_rejected,
                "tracked_owner_stages": len(self._owner_stage_rows),
                "tracked_owners": len(self._owner_request_times),
            }
