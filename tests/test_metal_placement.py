"""Metal ownership composition must dispatch locally and preserve the full numeric state."""

import importlib.util
import platform

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import AppleMetal, Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles, ClientPrefixLayers
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or importlib.util.find_spec("mlx") is None,
    reason="Apple Silicon and pllm.run[metal] are required",
)


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize(
    "placement", [ClientLinearRoles(["qkv_projection", "attention_output"]), ClientPrefixLayers(1),
                  ClientLinearRoles(["qkv_projection", "attention_output"], prefix_layers=1)]
)
def test_metal_prepared_placement_matches_cpu_logits_kv_and_reuses_snapshots(
    tmp_path, monkeypatch, family, placement
):
    from pllm.runtime.metal import MetalCompiledMatrix
    from pllm.runtime.dashboard import client_body_placement_snapshot
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime

    root = create_tiny_llama_checkpoint(
        tmp_path / family,
        model_type=family,
        num_hidden_layers=2,
        qk_norm=family == "qwen3",
        with_qkv_bias=family == "qwen2",
    )
    states, gpu_rows = [], []
    forward = SemanticDecoderRuntime._forward
    clear = MetalCompiledMatrix.clear

    def record_forward(self, ids, **kwargs):
        result = forward(self, ids, **kwargs)
        states.append(
            (
                result.copy(),
                [
                    (cache.key[: cache.length].copy(), cache.value[: cache.length].copy())
                    for cache in self.caches
                ],
            )
        )
        return result

    def record_clear(self, inputs):
        gpu_rows.append(inputs.shape[0])
        return clear(self, inputs)

    monkeypatch.setattr(SemanticDecoderRuntime, "_forward", record_forward)
    monkeypatch.setattr(MetalCompiledMatrix, "clear", record_clear)
    observations = []
    for kernel in (Cpu(threads=1), AppleMetal(min_rows=2)):
        candidate = Experiment(
            "metal-placement",
            MaskedLinearCpu(
                Model.path(str(root), model_id="metal-placement"),
                kernels=kernel,
                placement=placement,
                quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
                inventory=PreparedInventory("request-sized", rows=1, refill="on-demand"),
            ),
            Deployment.local(root=str(tmp_path)),
            ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2),
        )
        states.clear()
        with build_roles(candidate) as roles, roles.client(bundle_cache_mode="off") as client:
            result = client.responses.create(
                input="Private input.", max_output_tokens=2, temperature=0
            )
            audit = client.privacy_audit
            assert audit.plaintext_prompt_bytes_sent == audit.plaintext_token_ids_sent == 0
            assert audit.preparation_requests_during_online == 0
            observations.append((result.output_text, result.usage, list(states)))
            state = client._core._transformer_state("metal-placement")
            compiled = client._core._compiled_public_decoder(
                state,
                max_input_tokens=result.usage.input_tokens,
                max_new_tokens=2,
            )
            matrices = state.bundle._body_metal_matrices
            storage = client_body_placement_snapshot(client, "metal-placement")
            assert storage is not None
            assert storage["native_body_i8_snapshot_bytes"] > 0  # CPU decode fallback.
            if isinstance(kernel, AppleMetal):
                assert (
                    storage["metal_body_i8_snapshot_bytes"]
                    == storage["client_body_i8_weight_bytes"]
                )
                assert set(matrices) == {
                    s.id
                    for s in state.bundle.stages.values()
                    if s.layer_index is not None and s.client_weight is not None
                }
                before = {key: id(value) for key, value in matrices.items()}
                compiled.validate()
                assert {
                    key: id(value) for key, value in state.bundle._body_metal_matrices.items()
                } == before
            else:
                assert storage["metal_body_i8_snapshot_bytes"] == 0
                assert not matrices and not gpu_rows
    assert gpu_rows and min(gpu_rows) >= 2  # No one-row decode GPU dispatch.
    assert observations[0][:2] == observations[1][:2]
    assert len(observations[0][2]) == len(observations[1][2]) >= 2
    for (left_logits, left_kv), (right_logits, right_kv) in zip(
        observations[0][2], observations[1][2], strict=True
    ):
        np.testing.assert_array_equal(left_logits, right_logits)
        for (lk, lv), (rk, rv) in zip(left_kv, right_kv, strict=True):
            np.testing.assert_array_equal(lk, rk)
            np.testing.assert_array_equal(lv, rv)


@pytest.mark.parametrize("remote_head", [False, True])
def test_provider_gpu_snapshots_match_actual_role_ownership_including_head(tmp_path, remote_head):
    import asyncio
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=2,
                                       tie_word_embeddings=False, with_qkv_bias=True)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1,
        metal_min_rows=2, remote_output_head=remote_head,
        client_prefix_layers=1, client_linear_roles=("attention_output", "qkv_projection"))
    asyncio.run(engine.load(load_hf_directory(root, model_id="ownership")))
    matrices = engine._metal_stages["ownership"]
    assert set(matrices) == set(engine.seeded_stage_ids("ownership"))
    assert ("lm_head" in matrices) == remote_head
    assert "token_lookup" not in matrices
