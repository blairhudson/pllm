from __future__ import annotations

import asyncio
import time
from typing import Any

import msgpack

from .engine import EngineCapabilities
from .models import ModelManifest, StageSpec
from .stage_protocol import DirectFHEStageRequest, DirectFHEStageResponse
from .transformer_engine import MaskedTransformerEngine, TransformerEngineError


class DirectFHETransformerEngine(MaskedTransformerEngine):
    """Direct BFV hybrid runtime for proprietary-weight checkpoints.

    Unlike the public-weight masked mode, the client never receives pairs of
    ``(r, W r)``. It encrypts each W4A4 activation row and the server evaluates
    the plaintext model matrix directly over BFV ciphertexts. The client still
    executes attention and nonlinearities, so the model-confidentiality claim is
    against an honest-but-curious client, not an actively malicious client that
    modifies the runtime to build a layer oracle.
    """

    capabilities = EngineCapabilities(
        name="direct-bfv-transformer-proprietary",
        model_sources=("huggingface", "safetensors", "vllm", "mlx-lm"),
        protocols=("direct-bfv.stage/v1",),
        online_fhe=True,
        he_preprocessed=False,
        continuous_batching=False,
        notes=(
            "all learned dense matrices remain server-side",
            "client receives no model-dependent one-time correlation equations",
            "direct BFV ciphertext-plaintext stage evaluation",
            "model privacy assumes an honest-but-curious client runtime",
        ),
    )

    async def load(self, manifest: ModelManifest) -> None:
        await super().load(manifest)
        loaded = self.models[manifest.id]
        loaded.manifest.metadata.update(
            {
                "privacy_mode": "proprietary",
                "privacy_protocol": "direct_bfv_w4a4",
                "client_runtime": "direct_fhe_transformer_v1",
                "online_fhe": True,
                "he_preprocessed": False,
                "model_weight_correlations_disclosed": False,
                "model_privacy_threat_model": "honest_but_curious_client",
                "malicious_client_model_privacy": False,
            }
        )

    async def execute_stage(
        self,
        model_id: str,
        stage: StageSpec,
        payloads: list[bytes],
    ) -> list[bytes]:
        runtime = self._runtime(model_id, stage.id)
        requests = [DirectFHEStageRequest.unpack(payload) for payload in payloads]
        if not requests:
            return []
        for request in requests:
            if request.model != model_id or request.stage_id != stage.id:
                raise TransformerEngineError("direct-FHE stage request route mismatch")
            if request.input_shape[-1] != runtime.spec.in_features:
                raise TransformerEngineError("direct-FHE stage input width mismatch")

        async def evaluate(request: DirectFHEStageRequest) -> bytes:
            started = time.perf_counter_ns()
            rows: list[bytes] = []
            for encrypted in request.encrypted_rows:
                rows.append(
                    await asyncio.to_thread(
                        self.evaluate_bfv_correlation,
                        model_id,
                        stage.id,
                        request.context_id,
                        encrypted,
                    )
                )
            output_rows = tuple(rows)
            elapsed = time.perf_counter_ns() - started
            runtime.calls += 1
            runtime.rows += len(output_rows)
            runtime.server_ns += elapsed
            return DirectFHEStageResponse(
                stage_id=stage.id,
                output_rows=output_rows,
                server_ns=elapsed,
            ).pack()

        # Keep evaluation serial per request in the reference implementation.
        # TenSEAL's underlying context/evaluator objects are not treated as
        # concurrently mutable here. Production GPU backends can override this.
        return [await evaluate(request) for request in requests]

    def create_local_correlations(self, *args: Any, **kwargs: Any):  # pragma: no cover - defensive
        raise TransformerEngineError("proprietary mode does not expose model-dependent correlations")

    def evaluate_bfv_correlations(self, *args: Any, **kwargs: Any):  # pragma: no cover - defensive
        raise TransformerEngineError("proprietary mode uses direct encrypted stage evaluation")

    def client_bundle(
        self, model_id: str, *, include_local_weights: bool = False
    ) -> bytes:
        value = msgpack.unpackb(
            super().client_bundle(model_id, include_local_weights=False),
            raw=False,
            strict_map_key=False,
        )
        value["runtime"] = "direct_fhe_transformer"
        value["privacy"] = {
            **dict(value.get("privacy") or {}),
            "mode": "proprietary",
            "protocol": "direct_bfv_w4a4",
            "online_fhe": True,
            "he_preprocessed": False,
            "model_weight_correlations_disclosed": False,
            "client_intermediate_activations": True,
            "model_privacy_threat_model": "honest_but_curious_client",
            "malicious_client_model_privacy": False,
            "dense_weights_in_bundle": False,
            "local_quantized_stages": [],
        }
        return msgpack.packb(value, use_bin_type=True)
