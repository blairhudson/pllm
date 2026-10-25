"""Offline R2 inequalities from retained scale/Gamma observations, not weights.

Only ordinary report paths are read. No network, payload, model or SDK imports.
All closures are local sufficient inequalities CONDITIONAL ON F and induction;
source authentication, source completeness and whole-graph no-wrap stay unresolved.
An unresolved sufficient inequality is not an observed wrapped dot product.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from math import isqrt
from pathlib import Path

from benchmarks.research import maverick_no_wrap_bounds as bounds
from benchmarks.research import maverick_weight_product_bounds as products
from benchmarks.research.maverick_totality_bounds import nearest_integer, power, round_bound

PIN = "96b30dc13593a244a5e59e84687309f53c375cfa"
ROLES = {"q": (4096, 2560), "k": (1024, 2560), "v": (1024, 2560),
         "o": (2560, 4096), "gate": (9728, 2560), "up": (9728, 2560), "down": (2560, 9728)}


def exact(value):
    if type(value) not in (int, Fraction, str):
        raise TypeError("exact integer/rational required; no floats or bools")
    return Fraction(value)


def ceil_sqrt(value):
    value = exact(value)
    if value < 0:
        raise ValueError("negative squared bound")
    root = isqrt(value.numerator // value.denominator)
    return root + (root*root < value)


def r2_upper(a_upper, l_upper, z):
    """R2 monotonic in actual A,L: inputs are UPPER BOUNDS, never observed rows."""
    if any(type(n) is not int or n < 0 for n in (a_upper, l_upper)) or exact(z) < 0:
        raise ValueError("invalid exact row/norm upper bounds")
    z = exact(z)
    budget = bounds.H - Fraction(l_upper, 2)
    lhs = 65536*a_upper*z*z
    closed = budget >= 0 and lhs <= budget*budget
    # Ceiling each nonnegative summand gives a rationally verified integer bound.
    upper = ceil_sqrt(lhs) + (l_upper+1)//2
    # When closed, H is another proved bound. Its minimum is not a numeric clip.
    return {"r2_inequality": "closed" if closed else "unresolved", "row_l2_squared_upper": a_upper,
            "row_l1_upper": l_upper, "input_l2_upper": z, "remaining_field_budget": budget,
            "scaled_squared_bound": lhs, "squared_margin": budget*budget-lhs,
            "integer_dot_absolute_upper": min(bounds.H, upper) if closed else upper}


def weight_bound(scale):
    scale = exact(scale)
    if not 0 < scale <= products.SCALE_MAX or round_bound(scale, 8) != scale:
        raise ValueError("outside selected positive BF16 scale premise")
    maximum = max(c.hi*power(c.exponent) for c in products.WEIGHT_CLASSES)
    effective = round_bound(maximum*scale, 24)
    if effective is None:
        raise ValueError("nonfinite weight endpoint")
    # Monotonic R32 and nearest-even bound EVERY admitted finite payload value.
    return nearest_integer(256*effective)


def report_objects(path):
    path = Path(path)
    if path.stat().st_size > 1 << 20:
        raise ValueError("expected a bounded ordinary report, not a payload")

    def no_float(text):
        raise ValueError("report contains machine-float data")
    decoder = json.JSONDecoder(parse_float=no_float, parse_constant=no_float)
    text, position, records = path.read_text(), 0, []
    while text[position:].strip():
        position += len(text[position:])-len(text[position:].lstrip())
        record, position = decoder.raw_decode(text, position)
        records.append(record)
    return records


def observations(scale_path, gamma_path):
    scales = [r for r in report_objects(scale_path) if isinstance(r, dict) and "tensor_ranges_and_counts" in r]
    gammas = [r for r in report_objects(gamma_path) if isinstance(r, dict) and "vectors" in r]
    if len(scales) != 1 or len(gammas) != 1:
        raise ValueError("need one ordinary scale and Gamma observation")
    scales, gammas = scales[0], gammas[0]
    if any(r.get("revision") != PIN or r.get("repository") != "Qwen/Qwen3-4B-FP8"
           or r.get("safe_error") is not None for r in (scales, gammas)):
        raise ValueError("incompatible or incomplete report identity")
    expected_scales = {}
    for layer in range(36):
        for role, (m, d) in ROLES.items():
            family = "self_attn" if role in ("q", "k", "v", "o") else "mlp"
            expected_scales[f"model.layers.{layer}.{family}.{role}_proj.weight_scale_inv"] = m*d//16384
    rows = scales["tensor_ranges_and_counts"]
    if (len(rows) != 252 or {r["name"] for r in rows} != set(expected_scales)
            or scales["inspected_tensors"] != 252 or scales["inspected_elements"] != 221760):
        raise ValueError("missing scale observation coverage")
    for row in rows:
        if row["elements"] != expected_scales[row["name"]] or row["counts"] != {
                "positive_finite_within_bound": row["elements"]}:
            raise ValueError("scale report is not complete positive-finite coverage")
    gamma_shapes = {"model.norm.weight": 2560}
    for layer in range(36):
        for role, d in (("input_layernorm", 2560), ("post_attention_layernorm", 2560),
                        ("self_attn.q_norm", 128), ("self_attn.k_norm", 128)):
            gamma_shapes[f"model.layers.{layer}.{role}.weight"] = d
    rows = gammas["vectors"]
    if (len(rows) != 145 or {r["name"] for r in rows} != set(gamma_shapes)
            or gammas["inspected_coefficients"] != 196096 or gammas.get("unread_vectors")):
        raise ValueError("missing Gamma observation coverage")
    gamma = {}
    for row in rows:
        if (row["shape"] != [gamma_shapes[row["name"]]] or row["elements"] != gamma_shapes[row["name"]]
                or sum(row["counts"].values()) != row["elements"]
                or set(row["counts"]) - {"positive_finite", "negative_finite", "positive_zero", "negative_zero"}):
            raise ValueError("Gamma coverage/nonfinite mismatch")
        value = exact(row["maximum_absolute"]["exact_fraction"])
        if value < 0 or round_bound(value, 8) != value:
            raise ValueError("Gamma maximum is not finite BF16")
        gamma[row["name"]] = value
    return exact(scales["positive_maximum"]["exact_fraction"]), gamma


def norm_bound(gamma):
    # F implies each BF16 norm input's F32 square is finite, hence |x_j|<2**64.
    # sqrt(2560)<51. This is an F-derived enclosure, NOT an observed embedding norm.
    certificate = bounds.check_rms_bound(2560, gamma, 51*power(64))
    if certificate.math_status != bounds.Status.CERTIFIED:
        raise ValueError("rounded RMS bound unresolved")
    return dict(certificate.facts)


def attention_context(v_bound):
    n = 8
    c = 1-n*(bounds.U32+bounds.ETA32)
    mass = bounds.K/c+n*bounds.E
    coordinate = bounds.R*(bounds.B**(n+1)*v_bound*mass
                           + 2*n*bounds.ETA32*bounds.B**n)+bounds.ETA16
    return 64*coordinate  # sqrt(4096), including fourfold GQA replication


def mlp_product(gate_bound, up_bound, gate_l2, up_l2):
    # Exact-real SiLU rounded at BF16 endpoints cannot enlarge |gate|.
    return bounds.K*min(gate_bound*up_l2, up_bound*gate_l2)+99*bounds.E


def analyze(scale, gamma):
    w = weight_bound(scale)
    records = []

    def row(layer, role, z=None, reason=None, norm_name=None):
        m, d = ROLES[role]
        result = {"layer": layer, "role": role, "rows": m, "input_dimension": d,
                  "semantic_claims": 8, "max_abs_integer_coefficient_upper": w,
                  "bound_kind": "universal coefficient enclosure; not observed row norms"}
        if norm_name:
            result.update(norm_source=norm_name, observed_gamma_max=gamma[norm_name])
        if reason:
            result.update(r2_inequality="unresolved", reason=reason)
        else:
            result.update(r2_upper(d*w*w, d*w, z))
            if result["r2_inequality"] == "unresolved":
                result["reason"] = "conservative_R2_upper_inequality_not_met; not observed wrap"
        records.append(result)
        return result

    for layer in range(36):
        input_name = f"model.layers.{layer}.input_layernorm.weight"
        post_name = f"model.layers.{layer}.post_attention_layernorm.weight"
        input_norm, post_norm = norm_bound(gamma[input_name]), norm_bound(gamma[post_name])
        for role in ("q", "k", "v"):
            v = row(layer, role, input_norm["norm_l2_bound"], norm_name=input_name)
        if v["r2_inequality"] == "closed":
            v_bound = round_bound(Fraction(v["integer_dot_absolute_upper"], bounds.D), 8)
            row(layer, "o", attention_context(v_bound), norm_name=input_name)
        else:
            row(layer, "o", reason="honest V relation not established: v R2 unresolved", norm_name=input_name)
        for role in ("gate", "up"):
            up = row(layer, role, post_norm["norm_l2_bound"], norm_name=post_name)
        if up["r2_inequality"] == "closed":
            component = round_bound(Fraction(up["integer_dot_absolute_upper"], bounds.D), 8)
            # Both gate/up share the same honest post-norm z. Frobenius dominates
            # operator norm for EVERY matrix in this coefficient enclosure.
            operator = ceil_sqrt(9728*2560*w*w)
            l2 = bounds.R*operator*post_norm["quantized_l2_bound"]/bounds.D
            ff = mlp_product(component, component, l2, l2)
            row(layer, "down", ff, norm_name=post_name)
        else:
            row(layer, "down", reason="honest gate/up relations not established: R2 unresolved", norm_name=post_name)
    records.append({"layer": None, "role": "head", "rows": 151936, "input_dimension": 2560,
                    "semantic_claims": 1, "norm_source": "model.norm.weight",
                    "observed_gamma_max": gamma["model.norm.weight"], "r2_inequality": "unresolved",
                    "reason": "BF16 head row quantities and tying uninspected; FP8 bound inapplicable"})
    by_role = {role: dict(Counter(row["r2_inequality"] for row in records if row["role"] == role))
               for role in (*ROLES, "head")}
    return {"repository": "Qwen/Qwen3-4B-FP8", "revision": PIN,
            "field_modulus": bounds.P, "centered_integer_bound_H": bounds.H,
            "integer_dot_denominator": bounds.D,
            "observed_global_scale_max": scale, "derived_Q8_coefficient_upper": w,
            "payload_domain": "all finite E4M3FN values; actual payload NaN exclusion remains unproved",
            "missing_detail": "scale report retains ranges/counts, not per-tensor/block maxima or actual row norms",
            "scale_bound_scope": "observed global maximum applied uniformly; no matrix-specific maximum inferred",
            "input_norm_enclosure": "51*2**64 follows from F finite RMS squares, not embedding observations",
            "F_and_inductive_prefix": "UNPROVED; all closed cells are conditional local R2 obligations",
            "source_authentication": "trusted supplied reports/immutable publisher, not whole LFS integrity",
            "source_status": bounds.Status.UNRESOLVED.value,
            "whole_graph_status": bounds.Status.UNRESOLVED.value,
            "by_role": by_role, "claim_groups": len(records), "semantic_claims": sum(r["semantic_claims"] for r in records),
            "conditional_closed_groups": sum(r["r2_inequality"] == "closed" for r in records),
            "conditional_closed_semantic_claims": sum(r["semantic_claims"] for r in records if r["r2_inequality"] == "closed"),
            "new_source_bytes": 0, "new_HTTP_attempts": 0, "records": records}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scales-report", type=Path, required=True)
    cli.add_argument("--gamma-report", type=Path, required=True)
    args = cli.parse_args(argv)
    scale, gamma = observations(args.scales_report, args.gamma_report)
    print(json.dumps(analyze(scale, gamma), default=lambda x: str(x) if isinstance(x, Fraction) else x, indent=2))


if __name__ == "__main__":
    main()
