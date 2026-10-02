"""Rust protocol checked against independent unbounded Python integer algebra."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pllm import _native


def fixture(seed=12):
    rng = np.random.default_rng(seed)
    return tuple(rng.integers(-5, 6, shape, dtype=np.int8) for shape in ((9, 4), (9, 4), (3, 9)))


@pytest.mark.rust
@pytest.mark.parametrize("bits", (24, 32, 64))
@pytest.mark.parametrize("mode", ("dense", "derived", "contracted", "seeded"))
def test_native_modular_oracle_with_wrap(bits, mode):
    weights = fixture()
    rng = np.random.default_rng(674)
    clear = rng.integers(0, 1 << 63, (4, 4), dtype=np.uint64) & np.uint64((1 << bits) - 1)
    clear[0] = [0, 1, (1 << bits) - 1, 1 << (bits - 1)]
    output, metadata = _native.projected_polynomial_probe(
        mode,
        bits,
        4,
        4,
        9,
        3,
        *(w.tobytes() for w in weights),
        clear.astype("<u8").tobytes(),
        b"b" * 32,
    )
    x = clear.astype(object)
    gate = x @ weights[0].astype(object).T
    up = x @ weights[1].astype(object).T
    expected = (((gate * gate + 256 * gate) * up) @ weights[2].astype(object).T) % (1 << bits)
    actual = np.frombuffer(output, dtype="<u8").reshape(4, 3)
    np.testing.assert_array_equal(actual, expected.astype(np.uint64))
    measured = json.loads(metadata)
    estimate = json.loads(_native.projected_polynomial_estimate(mode, bits, 4, 4, 9, 3))
    assert estimate["material_bytes"] == measured["material_bytes"]
    assert estimate["opening_bytes"] == measured["opening_bytes"]
    assert measured["backend"] == "rust-pllm-garble/pllm-core"
    assert measured["opening_bytes"][0] == measured["opening_bytes"][1]
    if mode == "seeded":
        assert measured["material_bytes"][0] < measured["expanded_array_storage_bytes_per_party"][0]


@pytest.mark.rust
@pytest.mark.parametrize("bad", ("shape", "ring", "mode", "bytes", "residue", "binding"))
def test_native_boundary_rejects_invalid_requests(bad):
    weights = fixture()
    args = [
        "seeded",
        24,
        1,
        4,
        9,
        3,
        *(w.tobytes() for w in weights),
        np.zeros((1, 4), dtype="<u8").tobytes(),
        b"b" * 32,
    ]
    if bad == "shape":
        args[2] = 17
    elif bad == "ring":
        args[1] = 8
    elif bad == "mode":
        args[0] = "unknown"
    elif bad == "bytes":
        args[9] += b"x"
    elif bad == "residue":
        args[9] = np.full((1, 4), (1 << 24), dtype="<u8").tobytes()
    else:
        args[10] = b"bad"
    with pytest.raises(ValueError):
        _native.projected_polynomial_probe(*args)


@pytest.mark.rust
def test_compiler_projection_keeps_complete_cost_and_numeric_gates_closed():
    from pllm import Model, lower_model
    from pllm.metrics import ProjectedPolynomialCostProbe
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from test_shared_resources import CONFIG

    plan = lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=8)
    composition = MaskedLinearCpu(
        Model.hf("Qwen/Qwen2.5-0.5B-Instruct"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    dense = ProjectedPolynomialCostProbe(mode="dense").project(
        plan, composition, response_new_tokens=8
    )
    seeded = ProjectedPolynomialCostProbe().project(plan, composition, response_new_tokens=8)
    assert seeded["plan_digest"] == dense["plan_digest"] == plan.digest
    assert all(layer["executed_rows"] == 46 for layer in seeded["layers"])
    assert (
        seeded["totals"]["material_a"] + seeded["totals"]["material_b"]
        < dense["totals"]["material_a"]
    )
    assert seeded["totals"]["peer_a"] == dense["totals"]["peer_a"]
    assert seeded["known_matrix_mac_ratio_to_offset"] > 1
    for placement in seeded["placements"].values():
        assert (
            sum(edge["body_bytes"] for edge in placement["body_bytes_by_edge"])
            == placement["known_all_link_body_bytes"]
        )
        assert placement["unknown_required_work"]
        assert placement["complete_total_body_bytes"] is None
        assert not placement["byte_admitted"]
    with pytest.raises(ValueError, match="response|decode|bound"):
        ProjectedPolynomialCostProbe().project(plan, composition, response_new_tokens=9)


@pytest.mark.rust
@pytest.mark.parametrize(
    "args",
    [
        ("seeded", 24, 0, 32, 128, 32),
        ("seeded", 24, 1 << 21, 32, 128, 32),
        ("seeded", 12, 1, 32, 128, 32),
        ("seeded", 64, 1 << 20, 1 << 20, 1 << 20, 1 << 20),
    ],
)
def test_native_cost_rejects_invalid_or_excessive_shapes(args):
    with pytest.raises(ValueError):
        _native.projected_polynomial_estimate(*args)
