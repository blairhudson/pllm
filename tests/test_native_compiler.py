from __future__ import annotations

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


def test_compiled_plan_matches_canonical_fixtures_and_digests():
    plan = compiled_fixture()

    assert plan.logical_plan == (FIXTURES / "logical-plan.valid.json").read_bytes().removesuffix(b"\n")
    assert plan.execution_plan == (FIXTURES / "execution-plan.valid.json").read_bytes().removesuffix(b"\n")
    assert plan.plan_lock == (FIXTURES / "plan-lock.valid.json").read_bytes().removesuffix(b"\n")
    assert plan.region_program == (FIXTURES / "region-program.valid.json").read_bytes().removesuffix(b"\n")
    assert plan.configuration_digest == "43cb9fa05e87d1fe88daf1d2573b42bc0794b84e505c90be07d3c647816cea32"
    assert plan.logical_plan_digest == "ccb6dc4466602d04fd5097f4b0a49aa555de876e4e99427eed7816a67b2a6df1"
    assert plan.execution_plan_digest == "768546d8d9ef3420ba13a02887251e6172e6d608ca1b84ede26248472f28a035"
    assert plan.plan_lock_digest == "61a45e7af9eb10e4825871657c1ee5b2a85959c1ebeeae48c00adfd4168b64ca"
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
                signed_weights[row * 3 + column] * values[batch * 3 + column]
                for column in range(3)
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
