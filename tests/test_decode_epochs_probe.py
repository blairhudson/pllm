"""Tiny functional reference checks; never a real-checkpoint performance cohort."""
import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("decode_epochs_probe",
    Path(__file__).resolve().parents[1] / "scripts/probe_decode_epochs.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    root = create_tiny_llama_checkpoint(tmp_path_factory.mktemp("decode-epochs") / "model",
        model_type="qwen2", seed=179, hidden_size=32, intermediate_size=64,
        head_dim=8, num_hidden_layers=2, with_qkv_bias=True)
    engine, compiled, pipeline = probe.load_reference(root, input_tokens=5, cap=9, bound=32)
    yield engine, compiled, pipeline
    asyncio.run(engine.unload(compiled._bundle.model_id))


def cell(reference, **kwargs):
    return probe.run_cell(*reference, [3, 7, 9, 12, 4], cap=9, **kwargs)


@pytest.mark.parametrize("epoch", [1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("used,termination", [(9, "explicit"), (3, "explicit"), (3, "eos"),
                                               (0, "explicit"), (0, "eos")])
def test_same_work_exact_logits_all_kv_pending_and_outputs(reference, epoch, used, termination):
    control = cell(reference, epoch=9, used=used, termination=termination)
    candidate = cell(reference, epoch=epoch, used=used, termination=termination)
    for key in ("state", "output_commitment", "emitted_count", "pending_count", "executed_generated_rows"):
        assert candidate[key] == control[key]
    for result in (control, candidate):
        rows = result["stage_rows"]
        assert rows["prepared_stage_rows_issued"] == rows["prepared_stage_rows_reserved"]
        assert rows["prepared_stage_rows_reserved"] == (
            rows["prepared_stage_rows_claimed"] + rows["prepared_stage_rows_burned"])
        assert rows.get("prepared_stage_rows_discarded", 0) == 0
    if used == 3 and epoch == 4:
        assert candidate["measured_serialized_stage_body_bytes"]["preparation_to_inference_corrections"] < (
            control["measured_serialized_stage_body_bytes"]["preparation_to_inference_corrections"])


def test_prefill_terminal_pruning_is_compiler_derived(reference):
    _, compiled, _ = reference
    cold = compiled.prepared_stage_rows(5, 0)
    resumed = compiled.prepared_stage_rows(5, 0, continuation=True)
    assert sum(cold.values()) < sum(resumed.values())
    assert set(cold) == set(resumed)


def test_every_executed_row_matches_independent_clear_native_target(reference):
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
    engine, compiled, _ = reference
    def clear(sid, value):
        stage = engine.models[compiled._bundle.model_id].stages[sid]
        q = quantize_activation_per_row(value, bits=8)
        result = dequantize_matmul(stage.compiled_weight.clear(q.values), q.scales,
            stage.weight.scales, output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
        if stage.bias is not None:
            result += stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)
    runtime = compiled.runtime(clear)
    observed = {}
    try:
        _, logits, _ = runtime.prepare_ids([3, 7, 9, 12, 4])
        observed[runtime.position] = probe.state_bits(runtime, logits, b"oracle")
        for _ in range(8):
            token = runtime.sample(logits, temperature=0)
            logits, _ = runtime.decode_step(token, runtime.caches, runtime.position)
            observed[runtime.position] = probe.state_bits(runtime, logits, b"oracle")
        for epoch in (1, 4, 8):
            checked = {}
            def capture(candidate, values):
                checked[candidate.position] = probe.state_bits(candidate, values, b"oracle")
            cell(reference, epoch=epoch, capture=capture)
            assert checked == observed
    finally:
        runtime.reset()


def test_failed_preparation_discards_published_rows_and_charges_partial_bodies(reference, monkeypatch):
    engine, compiled, _ = reference
    original = engine.prepare_seeded_stage
    calls = 0
    async def fail(request):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("preparation failed")
        return await original(request)
    monkeypatch.setattr(engine, "prepare_seeded_stage", fail)
    material = probe.EpochMaterial(engine, compiled)
    with pytest.raises(RuntimeError, match="preparation failed"):
        material.prepare(5, 3)
    assert not material.active and not material.corrections
    assert material.audit["prepared_stage_rows_issued"] > 0
    assert material.audit["prepared_stage_rows_issued"] == material.audit["prepared_stage_rows_discarded"]
    assert material.bytes["preparation_to_inference_corrections"] > 0


def test_full_cap_forecast_has_no_raw_correction_saving(reference):
    _, compiled, _ = reference
    for epoch in (1, 4, 8):
        forecast = probe.geometry_forecast(compiled, input_rows=5, cap=9, used=9, epoch=epoch)
        assert forecast["packed_correction_saving_bytes"] == 0
        assert forecast["whole_request_saving_fraction"] is None


def test_preparation_idle_no_online_fallback_burn_and_no_reuse(reference):
    engine, compiled, _ = reference
    material = probe.EpochMaterial(engine, compiled)
    material.prepare(1, 0)
    lease = material.lease
    sid = next(iter(lease.stages))
    _, _, tickets = lease.take(sid, 1)
    assert len(tickets) == 1
    with pytest.raises(RuntimeError, match="idle"):
        material.prepare(1, 0)
    with pytest.raises(Exception, match="exhausted"):
        lease.take(sid, 1)
    material.finish()
    material.finish()
    assert material.audit["prepared_stage_rows_reserved"] == (
        material.audit["prepared_stage_rows_claimed"] + material.audit["prepared_stage_rows_burned"])
    with pytest.raises(RuntimeError, match="fallback"):
        material(sid, np.zeros((1, compiled._bundle.stages[sid].in_features), np.float32))
    material.prepare(1, 0)
    _, _, fresh = material.lease.take(sid, 1)
    assert fresh != tickets
    material.finish()


@pytest.mark.parametrize("epoch", [1, 4, 9])
@pytest.mark.parametrize("fail_after", [0, 2, 4])
def test_failure_or_cancel_retires_all_reserved_material(reference, epoch, fail_after):
    result = cell(reference, epoch=epoch, fail_after=fail_after)
    assert result["status"] == "failed_or_cancelled"
    assert result["emitted_count"] == fail_after
    rows = result["stage_rows"]
    assert rows["prepared_stage_rows_reserved"] == rows["prepared_stage_rows_claimed"] + rows["prepared_stage_rows_burned"]


def test_over_bound_resume_rejects_before_issuance(reference, monkeypatch):
    engine, compiled, pipeline = reference
    def forbidden(*args, **kwargs):
        pytest.fail("must reject before fresh inventory")
    monkeypatch.setattr(probe.EpochMaterial, "prepare", forbidden)
    with pytest.raises(Exception, match="bound|exceed|range"):
        probe.run_cell(engine, compiled, pipeline, list(range(3, 34)), cap=9, epoch=4)


@pytest.mark.parametrize("kwargs", [{"epoch": 0}, {"epoch": True}, {"epoch": 4, "used": 10},
                                     {"epoch": 4, "termination": "unknown"}])
def test_invalid_workload_rejects(reference, kwargs):
    with pytest.raises(ValueError):
        cell(reference, **kwargs)


def test_committed_current_codec_binding_and_frames(reference):
    from pllm.runtime.residue_codec import compiled_row_layout
    _, compiled, pipeline = reference
    assert pipeline.components["linear"].params["request_encoding"] == "stage_packed"
    assert pipeline.components["linear"].params["output_encoding"] == "row_residues"
    assert compiled_row_layout(compiled, namespace="prepared", packed_input=True)
    assert all(stage.output_residue_bits for stage in compiled._bundle.stages.values()
               if stage.client_weight is None)


def test_full_ring_stage_matches_independent_packed_integer_oracle():
    from types import SimpleNamespace
    from pllm import _native
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.preparation_protocol import seeded_ring_profile
    from pllm.runtime.residue_codec import unpack_row_response
    from pllm.runtime.stage_protocol import PreparedStageBatchRequest

    weights = np.full((2, 1024), 127, np.int8)
    weights[1] *= -1
    profile = seeded_ring_profile(127 * 127 * 1024)
    assert profile.wire_bits == 32
    widths = bytes([25, 25])
    metadata = SimpleNamespace(seeded_profile=profile, output_residue_bits=widths)
    compiled = SimpleNamespace(_bundle=SimpleNamespace(model_id="wide", stages={"stage": metadata}))
    stage = SimpleNamespace(compiled_weight=MaskedGEMM(threads=1).compile(weights))
    engine = SimpleNamespace(models={"wide": SimpleNamespace(stages={"stage": stage})})
    rng = np.random.default_rng(29)
    inputs = rng.integers(0, 1 << 25, (2, 1024), dtype=np.uint32)
    corrections = rng.integers(0, 1 << 32, (2, 2), dtype=np.uint32)
    request = PreparedStageBatchRequest("11" * 16, ("22" * 16, "33" * 16), inputs, 32, 25)
    payload = request.pack()
    material = probe.EpochMaterial(engine, compiled)
    material.active = True
    material.corrections["stage"] = [corrections, 0]
    try:
        reply, = material.inference("stage", [payload])
        raw, _ = unpack_row_response(reply, ticket=request.batch_id, stage="stage", widths=widths,
                                     rows=2, bits=32, namespace="prepared")
        actual = np.frombuffer(_native.unpack_residue_rows(raw, widths, 2), "<u4").reshape(2, 2)
        oracle = (inputs.astype(np.int64) @ weights.astype(np.int64).T
                  + corrections.astype(np.int64)) % (1 << 25)
        np.testing.assert_array_equal(actual, oracle)
        assert material.bytes["client_to_inference_online"] == len(payload)
        assert material.bytes["inference_to_client_online"] == len(reply)
        with pytest.raises(RuntimeError, match="one-use ticket reused"):
            material.inference("stage", [payload])
    finally:
        material.finish()


def test_unexpected_exception_retires_runtime_and_material(reference, monkeypatch):
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    original = SemanticDecoderRuntime.reset
    retired = []
    def reset(runtime):
        original(runtime)
        retired.append(runtime)
    monkeypatch.setattr(SemanticDecoderRuntime, "reset", reset)
    def fail(*args):
        raise OSError("unexpected capture error")
    with pytest.raises(OSError, match="unexpected"):
        cell(reference, epoch=4, capture=fail)
    runtime = retired[-1]
    assert runtime.position == 0 and not runtime.remote.active
    assert runtime.remote.audit["prepared_stage_rows_reserved"] == (
        runtime.remote.audit["prepared_stage_rows_claimed"] + runtime.remote.audit["prepared_stage_rows_burned"])


def test_historical_control_comes_from_conserved_ledger():
    control = probe.historical_control()
    assert control["pair_bytes"] == 493575247
    assert control["classes"]["corrections"] == 139060689
    assert control["classes"]["inputs"] == 90443518
    assert control["classes"]["public_bundle"] == 124905091


@pytest.mark.parametrize("input_tokens,cap,bound", [(17, 32, 256), (16, 129, 256), (16, 32, 1024), (16, 32, 32)])
def test_memory_workload_caps_reject_before_checkpoint_read(input_tokens, cap, bound):
    with pytest.raises(ValueError, match="reference workload"):
        probe.memory_admission("/does-not-exist", input_tokens=input_tokens, cap=cap, bound=bound)
