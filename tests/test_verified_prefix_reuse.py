"""Verified reuse retains a sealed, composition-budgeted verification lineage."""

from dataclasses import replace
import importlib.util

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import AppleMetal, Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import VerifiedMaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles, OutputHeadAtInference
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import TransformerClientError
from pllm.state import ClientPrefixReuse
from pllm.verification import FreivaldsVerify


def make_experiment(tmp_path, family="qwen2", remote_head=False, metal=False):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", model_type=family, seed=83, num_hidden_layers=2,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3", tie_word_embeddings=False,
    )
    return Experiment(
        "verified-combinations",
        VerifiedMaskedLinearCpu(
            Model.path(str(root), model_id="verified-combinations"),
            kernels=AppleMetal(min_rows=2) if metal else Cpu(threads=1),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
            cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128),
            placement=ClientLinearRoles(["qkv_projection", "attention_output"], prefix_layers=1),
            inventory=PreparedInventory(rows=1, refill="on-demand"),
            delivery=ClientBundleTransport("artifacts", compression="zlib"),
            boundary=OutputHeadAtInference() if remote_head else None,
        ),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=8, max_input_tokens=128, max_new_tokens=2),
    )


def capture(runtime, logits):
    return np.asarray(logits).copy(), [
        (row.key[:row.length].copy(), row.value[:row.length].copy()) for row in runtime.caches
    ], runtime.snapshot().state_basis


@pytest.mark.parametrize("family,remote_head,metal", [
    ("qwen2", False, False), ("qwen3", True, False), ("qwen2", False, True),
])
def test_verified_reuse_composes_placement_artifacts_inventory_and_metal(
    tmp_path, monkeypatch, family, remote_head, metal,
):
    if metal and importlib.util.find_spec("mlx") is None:
        pytest.skip("Apple Metal runtime is unavailable")
    experiment = make_experiment(tmp_path, family, remote_head, metal)
    assert experiment.resolve().verification_target_failure_bits == 52
    prefills, continued = [], []
    original, prepare = SemanticDecoderRuntime.continue_ids, SemanticDecoderRuntime.prepare_ids

    def continuation(runtime, ids, **kwargs):
        result = original(runtime, ids, **kwargs)
        continued.append(capture(runtime, result[-1]))
        return result

    def prefill(runtime, ids):
        result = prepare(runtime, ids)
        prefills.append(capture(runtime, result[1]))
        return result

    monkeypatch.setattr(SemanticDecoderRuntime, "continue_ids", continuation)
    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prefill)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        bundle_cache_dir=tmp_path / "cache"
    ) as client:
        request = dict(model=experiment.name, temperature=0, max_output_tokens=2)
        client.responses.create(input="abcdef " * 8, **request)
        branch = "abcdef " * 7 + "new suffix"
        cached = client.responses.create(input=branch, **request)
        fresh = client.responses.create(input=branch, store=False, **request)
        assert cached.output_text == fresh.output_text and cached.usage == fresh.usage
        assert len(continued) == 1
        np.testing.assert_array_equal(continued[0][0], prefills[-1][0])
        for actual, expected in zip(continued[0][1], prefills[-1][1], strict=True):
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(left, right)
        assert continued[0][2].valid() and continued[0][2].verification_failure_bits == 52
        assert continued[0][2].verification_lineage == 2  # two independently issued inventories
        assert prefills[-1][2].verification_lineage == 1
        before = client.privacy_audit.inference_stage_calls
        repeat = client.responses.create(input=branch, **request)
        assert repeat.output_text == fresh.output_text
        assert client.privacy_audit.inference_stage_calls - before == 2 + int(remote_head)
        assert client.privacy_audit.prefill_cache_hits == 2
        assert client.privacy_audit.prefill_prefix_tokens_reused > 0
        assert client.privacy_audit.plaintext_prompt_bytes_sent == client.privacy_audit.plaintext_token_ids_sent == 0
        saved = client._core._transformer_conversations[repeat.id].snapshot.state_basis
        assert saved.valid() and saved.verification_lineage == 3
        assert not replace(saved, verification_lineage=1).valid()
        assert not replace(saved, verification_failure_bits=0).valid()
        assert not replace(saved, verification_inventory_digest="f" * 64).valid()
        cache = client._core._transformer_states[experiment.name].prefill_cache
        assert cache.verification_failure_bits == 52
        assert all(entry.state_basis.verification_failure_bits == 52 for entry in cache._entries.values())
        assert client._core.artifact_cache_stats.object_requests > 0


@pytest.mark.parametrize("failure", ["provenance", "lineage"])
def test_verified_cache_rejects_missing_provenance_or_exhausted_budget_before_stage_work(tmp_path, monkeypatch, failure):
    experiment = make_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        bundle_cache_dir=tmp_path / "cache"
    ) as client:
        request = dict(model=experiment.name, temperature=0, max_output_tokens=2)
        client.responses.create(input="abcdef " * 8, **request)
        cache = client._core._transformer_states[experiment.name].prefill_cache
        original = SemanticDecoderRuntime._validate_verification_basis

        def restricted(runtime, basis):
            if failure == "lineage":
                runtime._verification_lineage_limit = 1
            original(runtime, basis)

        monkeypatch.setattr(SemanticDecoderRuntime, "_validate_verification_basis", restricted)
        if failure == "provenance":
            get = cache.get

            def forged(*args, **kwargs):
                value = get(*args, **kwargs)
                if value is not None:
                    value[0].state_basis = replace(value[0].state_basis, verification_failure_bits=0)
                return value

            monkeypatch.setattr(cache, "get", forged)
        before = client.privacy_audit.inference_stage_calls
        retained = {key: row.state_basis for key, row in cache._entries.items()}
        burned = client.privacy_audit.prepared_stage_rows_burned
        with pytest.raises(TransformerClientError, match="execution basis|verification provenance|failure budget"):
            client.responses.create(input="abcdef " * 8, **request)
        assert client.privacy_audit.inference_stage_calls == before
        assert {key: row.state_basis for key, row in cache._entries.items()} == retained
        assert client.privacy_audit.prepared_stage_rows_burned > burned


def test_verified_cache_allocation_cannot_exceed_native_failure_bit_limit(tmp_path):
    experiment = make_experiment(tmp_path)
    assert experiment.with_params(pipeline=experiment.pipeline.with_params(
        verification=FreivaldsVerify(target_failure_bits=68)
    )).resolve().verification_target_failure_bits == 80
    with pytest.raises(ValueError):
        experiment.with_params(pipeline=experiment.pipeline.with_params(
            verification=FreivaldsVerify(target_failure_bits=69))).resolve()


def test_provider_textual_inventory_id_cannot_merge_verification_epochs():
    from pllm.runtime.transformer_client import PreparedInventory

    first, second = (PreparedInventory("same-provider-id", 1, {}) for _ in range(2))
    assert len(first._verification_identity) == 32
    assert first._verification_identity != second._verification_identity


@pytest.mark.parametrize("remote_head", [False, True])
def test_verified_cached_single_output_needs_no_new_remote_checks(tmp_path, remote_head):
    experiment = make_experiment(tmp_path, remote_head=remote_head)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(
        bundle_cache_dir=tmp_path / "cache"
    ) as client:
        request = dict(model=experiment.name, input="abcdef " * 8, temperature=0, max_output_tokens=1)
        first = client.responses.create(**request)
        before = client.privacy_audit.inference_stage_calls
        repeat = client.responses.create(**request)
        assert repeat.output_text == first.output_text and repeat.usage == first.usage
        assert client.privacy_audit.inference_stage_calls == before
        assert client.privacy_audit.prefill_cache_hits == 1
        saved = client._core._transformer_conversations[repeat.id].snapshot.state_basis
        assert saved.valid() and saved.verification_failure_bits == 52
        assert saved.verification_lineage == 1  # a reservation alone adds no verification event
