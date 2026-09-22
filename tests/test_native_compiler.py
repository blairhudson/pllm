from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct

import pytest

from pllm import _native


pytestmark = pytest.mark.rust
ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "schemas" / "fixtures"


def compiled_fixture() -> _native.CompiledPlan:
    assert _native.capabilities()["implementation"] == "rust"
    return _native.compile_plan((FIXTURES / "compile-request.valid.json").read_bytes())


def silu_compiled_fixture(size: int):
    document = json.loads((FIXTURES / "compile-request.valid.json").read_bytes())
    contract = json.loads(_native.silu_q7_contract())
    components = document["configuration"]["pipeline"]["components"]
    components["nonlinear"] = {"component": contract["component_id"], "params": {}}
    components["nonlinear_schedule"] = {
        "component": contract["schedule_component_id"],
        "params": {"max_elements": 128},
    }
    document["context"]["compiler"] = {
        "id": contract["compiler_id"],
        "version": contract["compiler_version"],
        "digest": contract["compiler_artifact_digest"],
    }
    tensor = {"numeric": "signed_fixed_q7", "shape": [size]}
    operation = {
        "id": "layer.0.mlp.silu",
        "operator": "silu",
        "output": tensor,
        "input_representation": "arithmetic_label",
        "output_representation": "arithmetic_label",
    }
    method = document["methods"][0]
    method.update(
        {
            "id": contract["method_id"],
            "operator": "silu",
            "input_representation": "arithmetic_label",
            "output_representation": "arithmetic_label",
            "properties": {
                "online_parties": 1,
                "needs_online_preparation": False,
                "needs_client_weights": False,
                "uses_he": False,
                "experimental": True,
            },
            "artifact_digest": contract["method_artifact_digest"],
        }
    )
    document["kernels"][0].update(
        {
            "id": contract["kernel_id"],
            "method_id": method["id"],
            "input_numeric": "signed_fixed_q7",
            "output_numeric": "signed_fixed_q7",
            "implementation": "pllm_garble_silu_quadratic_q7",
            "artifact_digest": contract["kernel_artifact_digest"],
        }
    )
    document.update(
        {
            "input": tensor,
            "input_representation": "arithmetic_label",
            "output": tensor,
            "output_representation": "arithmetic_label",
            "operations": [operation],
            "privacy_contract": {
                **document["privacy_contract"],
                "allow_experimental": True,
                "required_claims": [],
            },
            "candidate_evidence": [],
            "assurance_results": [],
        }
    )

    def digest(domain, value):
        canonical = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(domain.encode() + b"\0" + canonical).hexdigest()

    document["context"]["composition_digest"] = digest(
        "pllm.pipeline.v2", document["configuration"]["pipeline"]
    )

    semantic = digest(
        "pllm.semantic_region.v1",
        {
            "input_shape": [size],
            "output_shape": [size],
            "operations": [{"id": operation["id"], "operator": "silu", "output_shape": [size]}],
        },
    )
    numeric = digest(
        "pllm.numeric_region.v1",
        {
            "semantic_graph_digest": semantic,
            "input_numeric": "signed_fixed_q7",
            "output_numeric": "signed_fixed_q7",
            "operations": [{"output_numeric": "signed_fixed_q7"}],
        },
    )
    protected = digest(
        "pllm.protected_region.v1",
        {
            "numeric_graph_digest": numeric,
            "input_representation": "arithmetic_label",
            "output_representation": "arithmetic_label",
            "operations": [
                {
                    "input_representation": "arithmetic_label",
                    "output_representation": "arithmetic_label",
                }
            ],
        },
    )
    document["context"]["semantic_graph"]["digest"] = semantic
    document["context"]["numeric_graph"]["id"] = contract["numeric_graph_id"]
    document["context"]["numeric_graph"]["digest"] = numeric
    document["context"]["protected_graph"]["id"] = contract["protected_graph_id"]
    document["context"]["protected_graph"]["digest"] = protected
    document["context"]["privacy_contract"]["digest"] = digest(
        "pllm.privacy_contract.v1", document["privacy_contract"]
    )
    encoded = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    return _native.compile_plan(encoded)


def gated_multiply_region(
    mode: str = "prefill",
    *,
    intermediate_size: int = 1,
    method: str = "pllm/r03-crt/v1",
    schedule: str = "pllm/scalar/v1",
    max_tensor_elements: int = 1,
):
    config = json.dumps(
        {
            "hidden_act": "silu",
            "hidden_size": 2,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": 8,
            "model_type": "qwen2",
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": True,
            "vocab_size": 8,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    plan = _native.lower_model(config, 1, 1, 1)
    (region,) = _native.lower_gated_multiply_q7(plan, mode, method, schedule, max_tensor_elements)
    return region


def test_compiled_plan_matches_canonical_fixtures_and_digests():
    plan = compiled_fixture()

    assert plan.logical_plan == (FIXTURES / "logical-plan.valid.json").read_bytes().removesuffix(
        b"\n"
    )
    assert plan.execution_plan == (
        FIXTURES / "execution-plan.valid.json"
    ).read_bytes().removesuffix(b"\n")
    assert plan.plan_lock == (FIXTURES / "plan-lock.valid.json").read_bytes().removesuffix(b"\n")
    assert plan.region_program == (
        FIXTURES / "region-program.valid.json"
    ).read_bytes().removesuffix(b"\n")
    assert (
        plan.configuration_digest
        == "cd051de9c3dfdfe2e3f13d1316582a844d5494529cf772c58502a44021875329"
    )
    assert (
        plan.logical_plan_digest
        == "2af9881d478e537177d8ab84637327ee454cc7c7a71a213fc03b01ea2b488703"
    )
    assert (
        plan.execution_plan_digest
        == "07348c6bb80569b71d6cb548c0e6fa10d9523a4392cf1a635e849adebbe05266"
    )
    assert (
        plan.plan_lock_digest == "fe8373bcc198486ebd03c29a877036faf32d54e4cfe2fc4d87e23738dab1f77a"
    )
    assert plan.input_shape == (2, 3)
    assert plan.output_shape == (2, 2)


def test_compiled_plan_is_opaque_and_immutable():
    plan = compiled_fixture()

    with pytest.raises(TypeError):
        _native.CompiledPlan()
    with pytest.raises(AttributeError):
        plan.logical_plan = b"{}"
    with pytest.raises(AttributeError):
        plan.extra = object()
    assert not hasattr(plan, "__dict__")


def test_compiled_plan_executes_wrap32_matrix_oracle():
    plan = compiled_fixture()
    weights = bytes((1, 254, 127, 128, 3, 252))
    values = (0, 1, 0xFFFFFFFF, 0x80000000, 7, 11)
    input_bytes = struct.pack("<6I", *values)

    signed_weights = (1, -2, 127, -128, 3, -4)
    expected = []
    for batch in range(2):
        for row in range(2):
            total = sum(
                signed_weights[row * 3 + column] * values[batch * 3 + column] for column in range(3)
            )
            expected.append(total & 0xFFFFFFFF)
    expected_bytes = struct.pack("<4I", *expected)

    assert plan.execute_wrap32(weights, input_bytes) == expected_bytes
    assert plan.execute_wrap32(weights, input_bytes, threads=2, simd=False) == expected_bytes


def test_compiler_diagnostics_are_structured_deterministic_utf8():
    malformed_messages = []
    for _ in range(2):
        with pytest.raises(ValueError) as caught:
            _native.compile_plan(b"\xff")
        malformed_messages.append(str(caught.value))
    assert malformed_messages[0] == malformed_messages[1]
    assert json.loads(malformed_messages[0]) == [
        {
            "code": "E_INVALID_DOCUMENT",
            "message": "invalid compile document: expected value at line 1 column 1",
            "subject_id": "compile_document",
        }
    ]

    document = json.loads((FIXTURES / "compile-request.valid.json").read_bytes())
    with pytest.raises(ValueError) as caught:
        _native.compile_plan(json.dumps(document, indent=2, sort_keys=True).encode())
    assert json.loads(str(caught.value)) == [
        {
            "code": "E_INVALID_DOCUMENT",
            "message": "compile document must use canonical compact sorted JSON bytes",
            "subject_id": "compile_document",
        }
    ]


@pytest.mark.parametrize(
    ("weights", "input_bytes", "threads", "message"),
    [
        (b"\x00" * 6, b"\x00\x00\x00", 1, "uint32 buffer length must be divisible by four"),
        (b"\x00" * 5, b"\x00" * 24, 1, "weights must be a nonempty [out,in] int8 matrix"),
        (b"\x00" * 6, b"\x00" * 4, 1, "input must have shape [batch,in]"),
        (b"\x00" * 6, b"\x00" * 24, 0, "threads must be in 1..=32"),
    ],
)
def test_compiled_plan_rejects_malformed_execution_inputs(weights, input_bytes, threads, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        compiled_fixture().execute_wrap32(weights, input_bytes, threads=threads)


def test_q7_silu_crosses_native_transport_boundary_and_burns_material():
    values = (-128, -64, 0, 64, 128)
    plan = silu_compiled_fixture(len(values))
    materials = tuple(plan.prepare_silu_q7_material() for _ in values)
    gates = tuple(material.gate for material in materials)
    evaluator = plan.prepare_silu_q7_evaluator(gates)
    with pytest.raises(ValueError, match="already bound"):
        plan.prepare_silu_q7_evaluator(gates)
    labels = tuple(material.encode(value) for material, value in zip(materials, values))
    outputs = evaluator.evaluate(labels)
    assert tuple(material.decode(output) for material, output in zip(materials, outputs)) == (
        -32,
        -24,
        0,
        40,
        96,
    )

    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate(labels)

    failed_materials = tuple(plan.prepare_silu_q7_material() for _ in values)
    failed_evaluator = plan.prepare_silu_q7_evaluator(
        tuple(material.gate for material in failed_materials)
    )
    failed_labels = tuple(
        material.encode(value) for material, value in zip(failed_materials, values)
    )
    with pytest.raises(ValueError, match="expected 5 labels"):
        failed_evaluator.evaluate(failed_labels[:-1])
    with pytest.raises(ValueError, match="already consumed"):
        failed_evaluator.evaluate(failed_labels)

    invalid_material = plan.prepare_silu_q7_material()
    with pytest.raises(ValueError, match="outside -128..=128"):
        invalid_material.encode(129)


def test_q7_silu_rejects_oversized_label_before_copy_and_burns_material():
    plan = silu_compiled_fixture(1)
    material = plan.prepare_silu_q7_material()
    evaluator = plan.prepare_silu_q7_evaluator((material.gate,))

    with pytest.raises(ValueError, match="label exceeds its byte bound"):
        evaluator.evaluate((b"x" * 1025,))
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate((material.encode(0),))


def test_q7_silu_copies_mutable_sequences_once():
    plan = silu_compiled_fixture(1)
    material = plan.prepare_silu_q7_material()

    class MutatingSequence(tuple):
        def __new__(cls, first: bytes):
            return super().__new__(cls, (first,))

        def __init__(self, first: bytes) -> None:
            self.calls = 0

        def __getitem__(self, index):
            if index != 0:
                raise IndexError
            self.calls += 1
            if self.calls == 1:
                return super().__getitem__(index)
            return b"x" * 16_385

    gates = MutatingSequence(material.gate)
    plan.prepare_silu_q7_evaluator(gates)
    assert gates.calls == 1


def test_gated_q7_multiply_crosses_opaque_native_boundary_and_burns_material():
    region = gated_multiply_region()
    assert region.multiply_operation_id == "layer.0.gated_multiply"
    assert len(region.digest) == 64
    material = region.prepare_material()
    assert len(material.evaluator_payload) == 245_397
    evaluator = region.prepare_evaluator(material.evaluator_payload)
    with pytest.raises(ValueError, match="already bound"):
        region.prepare_evaluator(material.evaluator_payload)

    output = evaluator.evaluate(material.encode_gate(-65), material.encode_up(127))
    assert material.decode(output) == -24
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate(material.encode_gate(-65), material.encode_up(127))
    with pytest.raises(AttributeError):
        region.multiply_operation_id = "forged"


def test_gated_q7_multiply_oversized_label_burns_the_program():
    region = gated_multiply_region("decode")
    material = region.prepare_material()
    evaluator = region.prepare_evaluator(material.evaluator_payload)
    with pytest.raises(ValueError, match="label exceeds its byte bound"):
        evaluator.evaluate(b"x" * 1025, material.encode_up(1))
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate(material.encode_gate(1), material.encode_up(1))


@pytest.mark.parametrize("invalid", [bytearray(b"label"), memoryview(b"label"), "label"])
def test_gated_q7_multiply_non_bytes_label_burns_before_conversion(invalid):
    region = gated_multiply_region("decode")
    material = region.prepare_material()
    evaluator = region.prepare_evaluator(material.evaluator_payload)
    with pytest.raises(ValueError, match="must be bytes"):
        evaluator.evaluate(invalid, material.encode_up(1))
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate(material.encode_gate(1), material.encode_up(1))


def test_gated_q7_multiply_rejects_unknown_decoder_mode():
    with pytest.raises(ValueError, match="decoder mode must be prefill or decode"):
        gated_multiply_region("training")


def test_gated_q7_components_cross_native_tensor_boundary_atomically():
    region = gated_multiply_region(
        intermediate_size=4,
        schedule="pllm/independent-lanes/v1",
        max_tensor_elements=4,
    )
    assert region.method_component_id == "pllm/r03-crt/v1"
    assert region.schedule_component_id == "pllm/independent-lanes/v1"
    assert region.max_tensor_elements == 4
    material = region.prepare_material()
    assert material.element_count == 4
    assert len(material.evaluator_payload) == 980_573
    gates = (-128, -65, 64, 128)
    ups = (127, -96, -96, 128)
    gate_labels = material.encode_gates(gates)
    up_labels = material.encode_ups(ups)
    evaluator = region.prepare_evaluator(material.evaluator_payload)
    outputs = evaluator.evaluate_tensor(gate_labels, up_labels)
    assert material.decode_tensor(outputs) == [-32, 18, -30, 96]
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate_tensor(gate_labels, up_labels)


def test_gated_q7_tensor_rejects_one_bad_lane_and_burns_bundle():
    region = gated_multiply_region(
        intermediate_size=2,
        schedule="pllm/independent-lanes/v1",
        max_tensor_elements=2,
    )
    material = region.prepare_material()
    gates = list(material.encode_gates((5, 6)))
    gates[1] += b"x"
    ups = material.encode_ups((7, 8))
    evaluator = region.prepare_evaluator(material.evaluator_payload)
    with pytest.raises(ValueError, match="label encoding is invalid"):
        evaluator.evaluate_tensor(gates, ups)
    with pytest.raises(ValueError, match="already consumed"):
        evaluator.evaluate_tensor(material.encode_gates((5, 6)), ups)


def test_gated_q7_tensor_burns_before_python_sequence_callbacks():
    region = gated_multiply_region(
        intermediate_size=2,
        schedule="pllm/independent-lanes/v1",
        max_tensor_elements=2,
    )
    material = region.prepare_material()
    gates = material.encode_gates((5, 6))
    ups = material.encode_ups((7, 8))
    evaluator = region.prepare_evaluator(material.evaluator_payload)

    class ReentrantSequence(tuple):
        def __new__(cls, values):
            return super().__new__(cls, values)

        def __init__(self, values) -> None:
            self.rejected = False

        def __getitem__(self, index):
            if not self.rejected:
                self.rejected = True
                with pytest.raises(ValueError, match="already consumed"):
                    evaluator.evaluate_tensor(gates, ups)
            return super().__getitem__(index)

    reentrant_gates = ReentrantSequence(gates)
    outputs = evaluator.evaluate_tensor(reentrant_gates, ups)
    assert reentrant_gates.rejected is True
    assert material.decode_tensor(outputs) == [0, 0]


def test_gated_q7_tensor_bounds_values_before_copy_and_supports_cancellation():
    region = gated_multiply_region(
        intermediate_size=2,
        schedule="pllm/independent-lanes/v1",
        max_tensor_elements=2,
    )
    material = region.prepare_material()
    payload = material.evaluator_payload
    with pytest.raises(ValueError, match="expected 2 gate values, got 5"):
        material.encode_gates(range(5))
    assert material.cancel() is True
    assert material.cancel() is False
    with pytest.raises(ValueError, match="was not issued or was already bound"):
        region.prepare_evaluator(payload)


def test_gated_q7_method_selection_keeps_dense_baseline_available():
    region = gated_multiply_region(method="pllm/binary-table/v1")
    material = region.prepare_material()
    assert region.method_component_id == "pllm/binary-table/v1"
    assert len(material.evaluator_payload) == 2_387_674
    evaluator = region.prepare_evaluator(material.evaluator_payload)
    output = evaluator.evaluate(material.encode_gate(-65), material.encode_up(127))
    assert material.decode(output) == -24


@pytest.mark.parametrize(
    ("method", "schedule", "elements", "message"),
    [
        ("paper-name", "pllm/scalar/v1", 1, "unsupported.*method"),
        ("pllm/r03-crt/v1", "paper-name", 1, "unsupported.*schedule"),
        ("pllm/r03-crt/v1", "pllm/scalar/v1", 2, "requires one element"),
        ("pllm/r03-crt/v1", "pllm/independent-lanes/v1", 5, "supports 2..=4"),
    ],
)
def test_gated_q7_rejects_incompatible_component_selection(
    method: str, schedule: str, elements: int, message: str
):
    with pytest.raises(ValueError, match=message):
        gated_multiply_region(
            method=method,
            schedule=schedule,
            max_tensor_elements=elements,
        )
