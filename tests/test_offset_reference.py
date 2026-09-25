"""Two independent loaded native stage kernels, with client-local reconstruction."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_reference import OffsetReferenceError, TwoOnlineOffsetReference
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, RemoteLinear, quantize_activation_per_row
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _workers(tmp_path: Path, model_type: str = "qwen2"):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1,
        model_type=model_type, with_qkv_bias=model_type == "qwen2",
        qk_norm=model_type == "qwen3",
    )
    model_id = "offset-model"
    manifest = load_hf_directory(root, model_id=model_id)
    first = MaskedTransformerEngine(threads=1)
    second = MaskedTransformerEngine(threads=1)
    asyncio.run(first.load(manifest))
    asyncio.run(second.load(manifest))
    bundle = ClientBundle.unpack(first.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2)
    compiled = compile_runtime_model(plan, bundle)

    def exchange(engine):
        def invoke(stage_id: str, payloads: list[bytes]) -> list[bytes]:
            stage = engine.models[model_id].stages[stage_id].spec
            return asyncio.run(engine.execute_stage(model_id, stage, payloads))
        return invoke

    return compiled, first, second, exchange(first), exchange(second)


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_offset_reference_matches_prepared_numeric_session_across_phases(
    tmp_path: Path, model_type: str,
) -> None:
    compiled, first, second, exchange_a, exchange_b = _workers(tmp_path, model_type)
    reference = TwoOnlineOffsetReference(
        compiled, first, second, model_id="offset-model",
        exchange_a=exchange_a, exchange_b=exchange_b,
    )

    class LocalCorrelations:
        model_id = "offset-model"

        def take_many(self, stage, count):
            return first.create_local_correlations("offset-model", stage.id, count)

    prepared = RemoteLinear(compiled._bundle.stages, LocalCorrelations(), exchange_a)
    reference_session = compiled.session(reference)
    prepared_session = compiled.session(prepared)
    token_ids = [0, 2]
    reference_session.prefill_ids(token_ids)
    prepared_session.prefill_ids(token_ids)
    np.testing.assert_allclose(
        reference_session.logits, prepared_session.logits, rtol=0, atol=1e-4,
    )
    assert reference_session.select_next() == prepared_session.select_next()
    reference_session.decode_selected()
    prepared_session.decode_selected()
    np.testing.assert_allclose(
        reference_session.logits, prepared_session.logits, rtol=0, atol=1e-4,
    )
    assert reference_session.select_next() == prepared_session.select_next()
    costs = reference.costs
    assert costs.stages == prepared.stats.calls
    assert costs.rows >= costs.stages
    assert costs.total_stage_body_bytes > 0
    assert all((costs.client_to_worker_a_bytes, costs.client_to_worker_b_bytes))
    assert all((costs.worker_a_to_client_bytes, costs.worker_b_to_client_bytes))
    assert costs.worker_a_stage_ns > 0 and costs.worker_b_stage_ns > 0
    assert costs.total_integer_macs > 0
    assert costs.worker_a_integer_macs == costs.worker_b_integer_macs


def test_offset_reference_rejects_same_worker_and_forged_response(tmp_path: Path) -> None:
    compiled, first, second, exchange_a, exchange_b = _workers(tmp_path)
    with pytest.raises(OffsetReferenceError, match="distinct loaded"):
        TwoOnlineOffsetReference(
            compiled, first, first, model_id="offset-model",
            exchange_a=exchange_a, exchange_b=exchange_b,
        )

    def forged(stage_id: str, requests: list[bytes]) -> list[bytes]:
        original = MaskedStageResponse.unpack(exchange_b(stage_id, requests)[0])
        return [MaskedStageResponse(
            correlation_id="forged", masked_output=original.masked_output,
            modulus=original.modulus, wire_bits=original.wire_bits,
            server_ns=original.server_ns, stage_id=original.stage_id, ring=original.ring,
        ).pack()]

    reference = TwoOnlineOffsetReference(
        compiled, first, second, model_id="offset-model",
        exchange_a=exchange_a, exchange_b=forged,
    )
    remote = next(stage for stage in compiled._stages if stage.client_weight_layout is None)
    with pytest.raises(OffsetReferenceError, match="result differs"):
        reference(remote.stage_id, np.zeros((1, remote.in_features), dtype=np.float32))
    assert reference.costs.stages == 0


def test_offset_reference_shares_are_fresh_and_bounded_before_issuance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled, first, second, exchange_a, exchange_b = _workers(tmp_path)
    observed_a: list[MaskedStageRequest] = []
    observed_b: list[MaskedStageRequest] = []

    def observe(exchange, observed):
        def call(stage_id: str, payloads: list[bytes]) -> list[bytes]:
            observed.extend(MaskedStageRequest.unpack(payload) for payload in payloads)
            return exchange(stage_id, payloads)
        return call

    reference = TwoOnlineOffsetReference(
        compiled, first, second, model_id="offset-model",
        exchange_a=observe(exchange_a, observed_a),
        exchange_b=observe(exchange_b, observed_b),
    )
    stage = next(stage for stage in compiled._stages if stage.client_weight_layout is None)
    activation = np.full((1, stage.in_features), 0.125, dtype=np.float32)
    reference(stage.stage_id, activation)
    reference(stage.stage_id, activation)
    profile = compiled._bundle.stages[stage.stage_id].seeded_profile
    clear = quantize_activation_per_row(activation, bits=stage.activation_bits).values
    assert len(observed_a) == len(observed_b) == 2
    for left, right in zip(observed_a, observed_b, strict=True):
        np.testing.assert_array_equal(
            (left.masked_input.astype(np.int64) + right.masked_input.astype(np.int64))
            % profile.modulus,
            clear.astype(np.int64) % profile.modulus,
        )
        assert np.array_equal(left.activation_scales, np.ones(1, np.float32))
        assert left.correlation_id != right.correlation_id
    assert observed_a[0].correlation_id != observed_a[1].correlation_id
    assert observed_a[0].session_id == observed_a[1].session_id
    assert observed_b[0].session_id == observed_b[1].session_id
    assert observed_a[0].session_id != observed_b[0].session_id
    assert not np.array_equal(observed_a[0].masked_input, observed_a[1].masked_input)
    assert reference.costs.stages == 2

    def no_mask(_size):
        raise AssertionError("mask sampled before stage admission")

    monkeypatch.setattr("pllm.runtime.offset_reference.secrets.token_bytes", no_mask)
    with pytest.raises(OffsetReferenceError, match="bounded tensor policy"):
        reference(stage.stage_id, np.zeros((5, stage.in_features), dtype=np.float32))


def test_offset_reference_rejects_out_of_bound_reconstruction(tmp_path: Path) -> None:
    compiled, first, second, exchange_a, exchange_b = _workers(tmp_path)

    def out_of_bound(stage_id: str, payloads: list[bytes]) -> list[bytes]:
        result = MaskedStageResponse.unpack(exchange_b(stage_id, payloads)[0])
        shifted = np.asarray(
            (result.masked_output.astype(np.int64) + result.modulus // 2) % result.modulus,
            dtype=np.uint32,
        )
        return [MaskedStageResponse(
            correlation_id=result.correlation_id, masked_output=shifted,
            modulus=result.modulus, wire_bits=result.wire_bits,
            server_ns=result.server_ns, stage_id=result.stage_id, ring=result.ring,
        ).pack()]

    reference = TwoOnlineOffsetReference(
        compiled, first, second, model_id="offset-model",
        exchange_a=exchange_a, exchange_b=out_of_bound,
    )
    stage = next(stage for stage in compiled._stages if stage.client_weight_layout is None)
    with pytest.raises(OffsetReferenceError, match="signed stage bound"):
        reference(stage.stage_id, np.zeros((1, stage.in_features), dtype=np.float32))
    assert reference.costs.stages == 0


def test_offset_reference_rejects_different_worker_body_before_issuance(tmp_path: Path) -> None:
    compiled, first, _second, exchange_a, _exchange_b = _workers(tmp_path / "first")
    different_root = create_tiny_llama_checkpoint(
        tmp_path / "other", num_hidden_layers=1, model_type="qwen2",
        with_qkv_bias=True, gate_weight_scale=0.02,
    )
    other = MaskedTransformerEngine(threads=1)
    asyncio.run(other.load(load_hf_directory(different_root, model_id="offset-model")))
    with pytest.raises(OffsetReferenceError, match="body differs"):
        TwoOnlineOffsetReference(
            compiled, first, other, model_id="offset-model",
            exchange_a=exchange_a, exchange_b=lambda _stage, _requests: [],
        )


def test_offset_reference_benchmark_reports_costs_without_token_ids() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, str(root / "scripts/benchmark_offset_reference.py"),
            "--model-type", "qwen2", "--repeats", "1",
        ],
        cwd=root, capture_output=True, text=True, check=True, timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["schema"] == "pllm.topology_reference_benchmark.v4"
    assert report["scope"] == "client_only_and_in_process_offset; not_deployed_network"
    assert report["offset_backend"] == "in-process"
    assert report["all_selected_tokens_match"] is True
    assert report["worst_logit_difference"] == 0
    assert report["input_token_count"] == report["generated_token_count"] == 2
    assert report["samples"][0]["offset_stage_calls"] == 8
    assert report["samples"][0]["offset_integer_macs"] == 55_296
    assert report["samples"][0]["client_only_body_integer_macs"] == 27_648
    assert report["samples"][0]["client_only_online_network_bytes"] == 0
    assert report["client_only_checkpoint_artifact_bytes"] > 0
    assert report["client_only_quantized_weight_bytes"] > 0
    assert report["client_only_compiled_bundle_bytes"] > 0
    assert report["client_only_cold_checkpoint_transfer_bytes"] is None
    assert report["client_only_peak_memory_bytes"] is None
    assert report["client_only_topology_digest"] != report["two_online_topology_digest"]
    assert report["samples"][0]["offset_stage_body_bytes"] > 0
    assert sum(report["samples"][0]["offset_per_edge_bytes"].values()) == (
        report["samples"][0]["offset_stage_body_bytes"]
    )
    assert "token_ids" not in result.stdout
    assert "prompt" not in result.stdout


def test_offset_loopback_benchmark_reports_matched_results_without_secret_material() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark_offset_reference.py"),
         "--model-type", "qwen2", "--repeats", "1", "--offset-backend", "loopback"],
        cwd=root, capture_output=True, text=True, check=True, timeout=60,
    )
    report = json.loads(result.stdout)
    assert report["schema"] == "pllm.topology_reference_benchmark.v4"
    assert report["offset_backend"] == "loopback"
    assert "worker_CPU_reported_separately" in report["offset_cpu_scope"]
    assert report["total_wire_bytes"] is None
    assert report["full_response_compute_cap_checked"] is False
    assert report["all_selected_tokens_match"] is True
    assert report["worst_logit_difference"] == 0
    assert report["samples"][0]["offset_stage_calls"] == 8
    assert report["samples"][0]["offset_integer_macs"] == 55_296
    assert report["median_offset_aggregate_online_cpu_seconds"] > 0
    assert set(report["samples"][0]["offset_online_cpu_seconds_by_role"]) == {
        "client", "worker_a", "worker_b",
    }
    assert set(report["samples"][0]["offset_worker_lifetime_peak_rss_bytes"]) == {
        "worker_a", "worker_b",
    }
    assert report["samples"][0]["offset_http_body_bytes_all_links"] > (
        report["samples"][0]["offset_stage_body_bytes"]
    )
    assert report["samples"][0]["client_only_body_integer_macs"] == 27_648
    assert report["client_only_cold_checkpoint_transfer_bytes"] is None
    assert "token_ids" not in result.stdout
    assert "prompt" not in result.stdout


def test_offset_chat_cohort_uses_normal_tokenized_gateway_input() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark_offset_reference.py"),
         "--model-type", "qwen2", "--cohort", "chat", "--repeats", "1"],
        cwd=root, capture_output=True, text=True, check=True, timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["cohort"] == "chat"
    assert report["input_token_count"] > 2
    assert report["all_selected_tokens_match"] is True
    assert report["worst_logit_difference"] == 0
    assert "user: A" not in result.stdout


def test_prepared_control_matches_checkpoint_and_token_cohort_without_payloads() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/benchmark_offset_reference.py"),
         "--model-type", "qwen2", "--cohort", "chat", "--repeats", "1",
         "--offset-backend", "loopback", "--include-prepared"],
        cwd=root, capture_output=True, text=True, check=True, timeout=180,
    )
    report = json.loads(result.stdout)
    control = report["prepared"]
    assert report["schema"] == "pllm.topology_reference_benchmark.v4"
    assert control["model_fingerprint"] == report["model_body_fingerprint"]
    assert control["input_token_count"] == report["input_token_count"]
    assert control["generated_token_count"] == report["generated_token_count"] == 2
    assert control["completed_runs"] == 1
    assert control["body_counter_set_present"]
    assert control["initial_inventory_all_link_body_bytes"] > 0
    assert control["recorded_body_bytes_including_initial_inventory"] >= (
        control["initial_inventory_all_link_body_bytes"]
        + control["median_online_all_link_body_bytes"]
    )
    assert any(
        edge["source"] == "preparation" and edge["destination"] == "inference"
        and edge["serialized_body_bytes"] > 0
        for edge in control["initial_inventory_body_bytes_by_edge"]
    )
    assert control["median_online_all_link_body_bytes"] > 0
    assert control["generated_selection_compared"] is False
    assert report["all_selected_tokens_match"] is True
    assert "user: A" not in result.stdout
    assert "token_ids" not in result.stdout
