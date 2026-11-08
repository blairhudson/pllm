"""Metadata-only estimates for immutable local F32 tensors and request scratch."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import pllm
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
from pllm.quantization import SymmetricPerRow
from pllm.runtime.benchmark_memory import _semantic_local_allocation, estimate_memory
from pllm.runtime.bundle_storage import CHUNK_BYTES
from pllm.runtime.decoder_memory import decoder_memory
from pllm.runtime.semantic_tensors import required_client_tensors
from pllm.runtime.semantic_workspace import semantic_local_workspace_bytes
from pllm.sources import _tiny_model_config
from pllm.state import ClientPrefixReuse

from absolute_position_decoder_oracle import tiny_checkpoint


def test_native_absolute_decoder_uses_qualified_live_roots_and_local_scratch():
    config, _ = tiny_checkpoint()
    plan = pllm.lower_model(config, batch=1, max_input_tokens=6, max_new_tokens=3)
    pipeline = MaskedLinearCpu(pllm.Model("local-memory"))
    memory = decoder_memory(plan, plan.runtime_schedule(pipeline))
    assert memory["mode"] == "full_kv_bounded_live_roots_v2"
    assert memory["working_bytes"] > 0 and memory["state_bytes"] > 0
    kinds = {op["operator"] for op in plan.prefill["operations"]}
    assert {"layer_norm", "gelu_tanh", "token_lookup"} <= kinds
    for operation in plan.prefill["operations"]:
        if operation["operator"] in {"layer_norm", "gelu_tanh", "token_lookup"}:
            assert semantic_local_workspace_bytes(operation, operation["output_shape"]) > 0


@pytest.mark.parametrize("kind", ["layer_norm", "gelu_tanh"])
def test_row_scratch_not_an_extra_output_matrix(kind):
    operation = {"operator": kind, "output_shape": [1, 1024, 64]}
    scratch = semantic_local_workspace_bytes(operation, [1, 1024, 64])
    assert scratch == 4 * 64 + 1024 * 64 + 64
    assert scratch < 4 * 1024 * 64
    assert semantic_local_workspace_bytes(operation, [1, 1, 64]) < scratch


def test_lookup_workspace_uses_requested_rows_not_table_cardinality():
    operation = {"operator": "token_lookup", "output_shape": [1, 128, 64],
                 "attributes": {"weight_shape": [1024, 64]}}
    scratch = semantic_local_workspace_bytes(operation, [1, 32, 64])
    operation["attributes"]["weight_shape"] = [1048576, 64]
    assert semantic_local_workspace_bytes(operation, [1, 32, 64]) == scratch
    assert scratch < 4 * 32 * 64


def test_lookup_full_request_masks_cross_previous_chunk_bound(monkeypatch):
    from pllm.runtime import semantic_numeric
    from pllm.runtime.bundle_storage import immutable_f32_array
    from pllm.runtime.semantic_tensors import lookup_contract

    # Genuine compiled Qwen3 geometry permits width one via its independent
    # attention head width. No absolute-position table/context cap is involved.
    rows = 2 * (CHUNK_BYTES // 8) + 1
    # Complete tiny Qwen3 contract, as used by the decoder numeric fixture;
    # only model dimensions and context differ for this metadata-only probe.
    config = {
        "model_type": "qwen3", "hidden_size": 4, "intermediate_size": 4,
        "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 2, "vocab_size": 8,
        "max_position_embeddings": 9, "rms_norm_eps": 0.000001,
        "rope_theta": 1000000, "hidden_act": "silu", "attention_bias": False,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0, "rope_scaling": None, "sliding_window": None,
        "use_sliding_window": False, "use_cache": True, "max_window_layers": 1,
    } | {
        "hidden_size": 1, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": 1,
        "num_key_value_heads": 1, "head_dim": 2, "vocab_size": 2,
        "max_position_embeddings": rows + 1,
    }
    plan = pllm.lower_model(config, batch=1, max_input_tokens=rows, max_new_tokens=1)
    operation_id = next(op["id"] for op in plan.prefill["operations"]
                        if op["operator"] == "token_lookup")
    operation, compiled_rows, width = semantic_numeric._compiled_local_geometry(
        plan, "prefill", operation_id)
    _, _, table_shape = lookup_contract(operation, legacy_vocabulary=config["vocab_size"])
    assert compiled_rows == rows and width == 1 and table_shape == (2, 1)
    table = immutable_f32_array(np.array([1.5, 2.5], dtype="<f4").tobytes(), table_shape)
    # Arithmetic-only kernel sentinel, not an executable bound tensor owner.
    # Request limits and table geometry come from the real native operation.
    kernel = SimpleNamespace(operation=operation, max_rows=compiled_rows,
                             width=width, table_shape=table_shape, weight=table)
    ids = np.zeros(compiled_rows, dtype=np.int64)
    observed = []
    forward_any = np.any

    def observe_any(value, *args, **kwargs):
        observed.append((value.shape, value.dtype, value.nbytes))
        return forward_any(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(semantic_numeric.np, "any", observe_any)
        result = semantic_numeric._BoundLocalNumeric.lookup(kernel, ids)
    assert result.shape == (compiled_rows, width)
    assert np.all(result == np.float32(1.5))
    assert observed == [((compiled_rows,), np.dtype(bool), compiled_rows)] * 2
    scratch = semantic_local_workspace_bytes(operation, operation["output_shape"])
    finite = min(compiled_rows * width, CHUNK_BYTES // 4)
    assert scratch == 11 * compiled_rows + finite + 64
    # Inputs + observed full-request mask must fit independently of output-root
    # accounting. This is an estimate check, not an observed physical peak.
    assert scratch >= ids.nbytes + max(size for _, _, size in observed)
    with pytest.raises(ValueError, match="compiled output bounds"):
        semantic_local_workspace_bytes(operation, [1, compiled_rows + 1, width])


@pytest.mark.parametrize("actual", [[1, 129, 4], [1, 1, 3], [1, 1, 5],
                                    [True, 1, 4], [1, 0, 4], [1, 4], [1, 1, 4.0]])
def test_workspace_rejects_geometry_not_admitted_by_compiled_operation(actual):
    with pytest.raises(ValueError):
        semantic_local_workspace_bytes({"operator": "layer_norm", "output_shape": [1, 128, 4]}, actual)


def test_local_storage_conversion_transport_and_binding_scratch_are_separate_estimates():
    requirements = {"position": (1024, 64), "gamma": (64,), "beta": (64,)}
    half = _semantic_local_allocation(requirements, source_item_sizes=dict.fromkeys(requirements, 2))
    full = _semantic_local_allocation(requirements)
    retained = 4 * (1024 * 64 + 128)
    assert half["retained_bytes"] == full["retained_bytes"] == retained
    assert full["producer_conversion_serialization_bytes"] - half["producer_conversion_serialization_bytes"] == retained // 2
    assert full["transport_import_overlap_bytes"] == 2 * retained
    assert full["extra_snapshot_copy_bytes"] == 0
    assert full["binding_chunk_scratch_bytes"] == min(1024 * 64, CHUNK_BYTES // 4)
    assert full["binding_control_bytes"] > 0


def test_native_plan_requirements_deduplicate_repeated_phase_and_operator_uses():
    config, _ = tiny_checkpoint()
    plan = pllm.lower_model(config, batch=1, max_input_tokens=6, max_new_tokens=3)
    required = required_client_tensors(plan)
    estimate = _semantic_local_allocation(required)
    assert estimate["retained_bytes"] == sum(4 * math.prod(shape) for shape in required.values())
    assert "transformer.wpe.weight" in required
    assert len(required) < sum(op["operator"] in {"layer_norm", "token_lookup"}
                               for phase in (plan.prefill, plan.decode) for op in phase["operations"]) * 2


def test_existing_inflight_concurrency_multiplies_request_workspace_not_shared_local_storage():
    config = _tiny_model_config()
    base = MaskedLinearCpu(pllm.Model.tiny(), quantization=SymmetricPerRow(
        weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"))
    serial = base.with_params(cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=16))
    parallel = base.with_params(cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=16,
        inflight=True, inflight_max_requests=3, inflight_max_bytes=1 << 20))
    assert resolve_runtime_composition(serial) is not None
    assert resolve_runtime_composition(parallel) is not None
    estimates = [estimate_memory(config, pipeline, max_input_tokens=16, max_output_tokens=2)
                 for pipeline in (serial, parallel)]
    assert estimates[1]["concurrent_response_bound"] == 3
    for key in ("client_tensor_work_bytes", "client_live_tensor_bytes", "client_state_work_bytes"):
        assert estimates[1]["components"][key] == 3 * estimates[0]["components"][key]
    assert (estimates[1]["components"]["semantic_local_allocation"]
            == estimates[0]["components"]["semantic_local_allocation"])
