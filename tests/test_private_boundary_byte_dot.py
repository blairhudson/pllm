"""Selector/marshalling tests; no model execution or native kernel invocation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pllm.runtime.native import CompiledMatrix, NativeKernelError
from pllm.runtime.model_binding import RuntimeBindingError, _validate_private_boundary_contract
from pllm.runtime.private_boundary_protocol import BoundaryDescriptor, BoundaryError


def descriptor(**options):
    return BoundaryDescriptor(**{name: "a" * 64 for name in (
        "source_digest", "checkpoint_digest", "model_config_digest", "composition_digest",
        "schedule_digest", "head_weight_digest", "head_scales_digest", "lookup_digest")},
        vocabulary=3, hidden=17, signed_output_bound=17 * 127 * 127, **options)


def test_descriptor_kernel_is_committed_and_default_bytes_unchanged():
    baseline = descriptor()
    assert "head_kernel" not in baseline.to_dict()
    assert descriptor(head_kernel="existing").digest == baseline.digest
    selected = descriptor(head_kernel="byte_dot")
    assert selected.digest != baseline.digest
    assert BoundaryDescriptor.from_dict(selected.to_dict()) == selected
    assert replace(selected, head_kernel="existing") == baseline
    for invalid in (True, None, "sdot", ""):
        with pytest.raises(BoundaryError, match="head kernel"):
            descriptor(head_kernel=invalid)


def test_descriptor_rejects_byte_dot_weight_bound_before_execution():
    with pytest.raises(BoundaryError, match="byte-dot matrix"):
        replace(descriptor(head_kernel="byte_dot"), vocabulary=262144, hidden=1025)


def matrix(executor):
    result = object.__new__(CompiledMatrix)
    result.shape, result.weight_bytes = (3, 17), 51
    result._matrix = object()
    result.owner = SimpleNamespace(_executor=executor)
    return result


def test_wrapper_marshals_full_ring_one_row_without_weight_copy():
    calls = []
    expected = np.array([[0, 1 << 31, 2**32 - 1]], dtype="<u4")
    def execute(weights, data):
        calls.append((weights, data))
        return expected.tobytes()
    compiled = matrix(SimpleNamespace(wrap32_byte_dot=execute))
    inputs = np.arange(17, dtype=np.uint32).reshape(1, 17)
    inputs[0, -1] = 2**32 - 1
    np.testing.assert_array_equal(compiled.wrap32_byte_dot(inputs), expected)
    assert calls == [(compiled._matrix, inputs.astype("<u4").tobytes())]


@pytest.mark.parametrize("shape", [(0, 17), (2, 17), (1, 16), (17,)])
def test_wrapper_rejects_malformed_dimensions(shape):
    compiled = matrix(SimpleNamespace())
    with pytest.raises(NativeKernelError):
        compiled.wrap32_byte_dot(np.zeros(shape, dtype=np.uint32))


@pytest.mark.parametrize("shape", [(0, 17), (2, 17), (1, 16), (17,), (1, 1, 17)])
def test_malformed_geometry_rejects_before_input_conversion(monkeypatch, shape):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed input reached dtype scanning or copying")
    monkeypatch.setattr("pllm.runtime.native._integer_array", forbidden)
    with pytest.raises(NativeKernelError, match="one integer input row"):
        matrix(SimpleNamespace()).wrap32_byte_dot(np.zeros(shape, dtype=np.uint64))


@pytest.mark.parametrize("native", [False, True])
def test_live_payload_rejects_before_conversion_for_both_backends(monkeypatch, native):
    compiled = matrix(SimpleNamespace())
    compiled.shape, compiled.weight_bytes = (1 << 27, 1), 1 << 27
    compiled._matrix = object() if native else None
    def forbidden(*args, **kwargs):
        pytest.fail("unadmitted geometry reached input conversion")
    monkeypatch.setattr("pllm.runtime.native._integer_array", forbidden)
    with pytest.raises(NativeKernelError, match="native bounds"):
        compiled.wrap32_byte_dot(np.zeros((1, 1), dtype=np.uint64))


def test_reference_complete_output_uses_bounded_scalar_owners(monkeypatch):
    compiled = matrix(SimpleNamespace())
    compiled._matrix = None
    compiled._reference = np.resize(np.array([-128, 127, -1, 0, 1], np.int8), compiled.shape)
    inputs = np.resize(np.array([0, 128, 255, 0x80000000, 0xffffffff], np.uint64), (1, 17))
    expected = np.asarray((inputs.astype(object) @ compiled._reference.astype(object).T) % 2**32, np.uint32)
    def forbidden(*args, **kwargs):
        pytest.fail("byte-dot reference expanded complete matrix through generic wrap32")
    monkeypatch.setattr(compiled, "wrap32", forbidden)
    np.testing.assert_array_equal(compiled.wrap32_byte_dot(inputs), expected)


def test_wrapper_missing_installed_api_fails_closed():
    with pytest.raises(NativeKernelError, match="rebuild installed extension"):
        matrix(SimpleNamespace()).wrap32_byte_dot(np.zeros((1, 17), np.uint32))


def test_source_contract_requires_independent_kernel_selection():
    # Hand-built public graph: this test never invokes native lowering or a model.
    graph = {"query_sequence": 1, "maximum_key_sequence": 1, "operations": [
        {"id": "lookup", "operator": "token_lookup", "inputs": ["input.tokens"],
         "attributes": {"weight": "tied.weight"}, "output_shape": [1, 1, 17]},
        {"id": "head", "operator": "output_head", "attributes": {
            "weight": "tied.weight", "tied": True}, "output_shape": [1, 3]}]}
    document = {"config_digest": "b" * 64, "prefill": graph, "decode": graph}
    phases = {phase: {"lookup_operation": "lookup", "head_operation": "head",
                      "lookup_rows": 1, "head_rows": 1} for phase in ("prefill", "decode")}
    contract = {"schema_version": "pllm.private_boundary.v1", "model_plan_digest": "a" * 64,
        "model_config_digest": "b" * 64, "composition_digest": "c" * 64,
        "weight_id": "tied.weight", "vocabulary": 3, "hidden_width": 17,
        "owners": ["boundary_a", "boundary_b"], "lookup_protocol": "packed128_xor_dpf_v2",
        "head_protocol": "two_party_additive_full_head_v1", "weight_bits": 8, "activation_bits": 8,
        "head_ring_bits": 24, "head_signed_output_bound": 17 * 127 * 127,
        "embedding_numeric": "tied_per_token_row_f32_v1",
        "head_numeric": "ordinary_symmetric_per_row_integer_then_f32_v1",
        "lookup_batch_bound": 64, "lookup_query_bound": 1, "matrix_bytes": 51,
        "scale_bytes": 12, "lookup_scratch_bytes": 192, "client_weight_bytes": 0, **phases}
    args = dict(plan_digest="a" * 64, composition_digest="c" * 64,
                vocabulary=3, hidden=17, activation_bits=8)
    _validate_private_boundary_contract(contract, document, **args)
    selected = {**contract, "head_kernel": "byte_dot"}
    _validate_private_boundary_contract(selected, document, boundary_params={"head_kernel": "byte_dot"}, **args)
    with pytest.raises(RuntimeBindingError):
        _validate_private_boundary_contract(selected, document, **args)
    with pytest.raises(RuntimeBindingError):
        _validate_private_boundary_contract(contract, document, boundary_params={"head_kernel": "byte_dot"}, **args)
