"""Pure synthetic bounds/reports, not model values or trained results."""
from fractions import Fraction
import importlib
import importlib.abc
import itertools
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def experiment():
    class OfflineOnly(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {
                "pllm", "torch", "numpy", "transformers", "safetensors", "tokenizers",
                "requests", "urllib", "http", "socket", "ssl", "httpx", "psutil",
            }:
                raise AssertionError("non-pure import forbidden: "+fullname)
            return None
    guard = OfflineOnly()
    root = str(Path(__file__).resolve().parents[1])
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, root)
    try:
        yield importlib.import_module("benchmarks.research.maverick_observed_no_wrap")
    finally:
        sys.meta_path.remove(guard)
        sys.path.remove(root)


@pytest.mark.parametrize("scale,expected", [(Fraction(181, 65536), 317), (Fraction(1, 32768), 4),
                                          (Fraction(3, 32768), 10), (Fraction(1, 1 << 133), 0)])
def test_exact_selected_weight_rounding_and_ties(experiment, scale, expected):
    assert experiment.weight_bound(scale) == expected
    assert experiment.weight_bound(scale) == experiment.nearest_integer(
        256*experiment.round_bound(448*scale, 24))


@pytest.mark.parametrize("scale", [0, -1, Fraction(1)+(Fraction(1, 1 << 23)), 1 << 46, 0.1, True])
def test_no_scale_policy_widening_or_float_guesses(experiment, scale):
    with pytest.raises((ValueError, TypeError)):
        experiment.weight_bound(scale)


@pytest.mark.parametrize("row", [(0,), (1,), (3, -4), (317, -317), (-9, 2, 1)])
@pytest.mark.parametrize("z", [Fraction(0), Fraction(1), Fraction(1000000)])
def test_r2_matches_existing_public_math_checker(experiment, row, z):
    actual = experiment.bounds.check_row_real_l2(row, z)
    upper = experiment.r2_upper(sum(v*v for v in row), sum(abs(v) for v in row), z)
    assert (upper["r2_inequality"] == "closed") == (actual.math_status == experiment.bounds.Status.CERTIFIED)
    assert actual.source_status == actual.whole_graph_status == experiment.bounds.Status.UNRESOLVED
    assert upper["scaled_squared_bound"] == dict(actual.facts)["scaled_bound_squared"]


def test_uniform_row_enclosure_is_not_a_fabricated_source_row(experiment):
    z = Fraction(experiment.bounds.H, 256*6)
    bound = experiment.r2_upper(3*3*3, 3*3, z)
    assert bound["r2_inequality"] == "closed"
    for row in itertools.product((-3, 0, 3), repeat=3):
        actual = experiment.bounds.check_row_real_l2(row, z)
        assert actual.math_status == experiment.bounds.Status.CERTIFIED


def test_field_boundary_and_no_failed_bound_clamping(experiment):
    h = experiment.bounds.H
    closed = experiment.r2_upper(1, 1, Fraction(2*h-1, 512))
    assert closed["r2_inequality"] == "closed" and closed["integer_dot_absolute_upper"] == h
    failed = experiment.r2_upper(1, 1, Fraction(2*h+1, 512))
    assert failed["r2_inequality"] == "unresolved" and failed["integer_dot_absolute_upper"] > h
    assert experiment.r2_upper(0, 2*h+1, 0)["r2_inequality"] == "unresolved"


def test_finite_square_enclosure_and_rounded_rms_are_reused(experiment):
    p = experiment.power
    assert experiment.round_bound(p(64)*p(64), 24) is None
    largest_bf16_below = p(64)-p(56)
    assert experiment.round_bound(largest_bf16_below**2, 24) is not None
    for gamma in (Fraction(0), Fraction(197, 1024), Fraction(29)):
        result = experiment.norm_bound(gamma)
        certificate = experiment.bounds.check_rms_bound(2560, gamma, 51*p(64))
        assert result == dict(certificate.facts)
        assert result["epsilon_margin"] > 0
        assert result["quantized_l2_bound"] == 256*result["norm_l2_bound"]+Fraction(51, 2)
        assert result["norm_l2_bound"] > 0  # additive underflow terms not removed


def test_attention_and_ff_keep_all_rounding_margins(experiment):
    b = experiment.bounds
    for v in (Fraction(0), Fraction(3, 2), Fraction(15360)):
        context = experiment.attention_context(v)
        for n in range(1, 9):
            mass = b.K/(1-n*(b.U32+b.ETA32))+n*b.E
            cn = b.R*(b.B**(n+1)*v*mass+2*n*b.ETA32*b.B**n)+b.ETA16
            assert context >= 64*cn
        assert context > 64*v
    assert experiment.mlp_product(2, 3, 5, 7) == b.K*min(2*7, 3*5)+99*b.E
    assert experiment.mlp_product(0, 0, 0, 0) > 0


def synthetic_gamma():
    gamma = {"model.norm.weight": Fraction(1)}
    for layer in range(36):
        for role in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            gamma[f"model.layers.{layer}.{role}.weight"] = Fraction(1)
    gamma["model.layers.0.self_attn.k_norm.weight"] = Fraction(44)
    return gamma


def test_actual_appropriate_norm_roles_not_global_gamma(experiment):
    gamma = synthetic_gamma()
    result = experiment.analyze(Fraction(181, 65536), gamma)
    assert result["claim_groups"] == 253 and result["semantic_claims"] == 2017
    assert result["conditional_closed_groups"] == 180
    assert result["conditional_closed_semantic_claims"] == 1440
    first = result["records"][0]
    assert first["norm_source"] == "model.layers.0.input_layernorm.weight" and first["observed_gamma_max"] == 1
    assert all("post_attention_layernorm" in row["norm_source"] for row in result["records"] if row["role"] in ("gate", "up", "down"))
    changed = dict(gamma)
    changed["model.layers.0.self_attn.k_norm.weight"] = Fraction(1024)
    assert experiment.analyze(Fraction(181, 65536), changed) == result
    head = result["records"][-1]
    assert head["r2_inequality"] == "unresolved" and "BF16" in head["reason"]
    assert "max_abs_integer_coefficient_upper" not in head
    assert result["source_status"] == result["whole_graph_status"] == "unresolved"
    assert result["new_source_bytes"] == result["new_HTTP_attempts"] == 0


def test_no_canonical_reply_substitute_when_upstream_fails(experiment):
    gamma = synthetic_gamma()
    gamma["model.layers.0.input_layernorm.weight"] = Fraction(29)
    gamma["model.layers.0.post_attention_layernorm.weight"] = Fraction(29)
    result = experiment.analyze(Fraction(181, 65536), gamma)
    rows = {row["role"]: row for row in result["records"] if row["layer"] == 0}
    assert rows["o"]["r2_inequality"] == rows["down"]["r2_inequality"] == "unresolved"
    assert "integer_dot_absolute_upper" not in rows["o"]
    assert "integer_dot_absolute_upper" not in rows["down"]
    assert "honest V" in rows["o"]["reason"] and "honest gate/up" in rows["down"]["reason"]


def report_fixtures(experiment, tmp_path):
    scales = []
    for layer in range(36):
        for role, (m, d) in experiment.ROLES.items():
            family = "self_attn" if role in ("q", "k", "v", "o") else "mlp"
            count = m*d//16384
            scales.append({"name": f"model.layers.{layer}.{family}.{role}_proj.weight_scale_inv",
                           "elements": count, "counts": {"positive_finite_within_bound": count}})
    identity = {"repository": "Qwen/Qwen3-4B-FP8", "revision": experiment.PIN, "safe_error": None}
    scale_report = {**identity, "tensor_ranges_and_counts": scales,
                    "inspected_tensors": 252, "inspected_elements": 221760,
                    "positive_maximum": {"exact_fraction": "181/65536"}}
    vectors = []
    for name, value in synthetic_gamma().items():
        dimension = 128 if "self_attn" in name else 2560
        vectors.append({"name": name, "shape": [dimension], "elements": dimension,
                        "counts": {"positive_finite": dimension},
                        "maximum_absolute": {"exact_fraction": str(value)}})
    gamma_report = {**identity, "vectors": vectors, "inspected_coefficients": 196096, "unread_vectors": []}
    a, b = tmp_path/"scale.log", tmp_path/"gamma.log"
    a.write_text(json.dumps({"plan": "synthetic prelude"})+"\n"+json.dumps(scale_report, indent=2))
    b.write_text(json.dumps(gamma_report))
    return a, b, scale_report, gamma_report


def test_ordinary_sequence_report_and_actual_cli_parser(experiment, tmp_path, capsys):
    a, b, _, _ = report_fixtures(experiment, tmp_path)
    assert experiment.observations(a, b) == (Fraction(181, 65536), synthetic_gamma())
    experiment.main(["--scales-report", str(a), "--gamma-report", str(b)])
    output = json.loads(capsys.readouterr().out)
    assert output["derived_Q8_coefficient_upper"] == 317
    assert output["source_status"] == output["whole_graph_status"] == "unresolved"
    assert output["records"][0]["row_l2_squared_upper"] == 2560*317*317
    assert "actual row norms" in output["missing_detail"]


@pytest.mark.parametrize("mutation", ["pin", "scale_count", "gamma_count", "nonfinite", "float_max", "missing_max"])
def test_incomplete_or_incompatible_observations_fail_not_guess(experiment, tmp_path, mutation):
    a, b, scales, gamma = report_fixtures(experiment, tmp_path)
    if mutation == "pin":
        scales["revision"] = "wrong"
    elif mutation == "scale_count":
        scales["tensor_ranges_and_counts"].pop()
    elif mutation == "gamma_count":
        gamma["vectors"].pop()
    elif mutation == "nonfinite":
        gamma["vectors"][0]["counts"] = {"nan": 2560}
    elif mutation == "float_max":
        scales["positive_maximum"]["exact_fraction"] = 0.1
    else:
        del scales["positive_maximum"]
    a.write_text(json.dumps(scales)); b.write_text(json.dumps(gamma))
    with pytest.raises((ValueError, KeyError)):
        experiment.observations(a, b)
