"""Bounded exact-adapter controls. Standard library only; no checkpoint loading.

These are clear algebra/bit-identity references, not protected implementations.
All model-specific claims are imported from hashed, frozen evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "ARCHITECTURE.md",
    "docs/research/network-first-principles.md",
    "docs/research/network-algebra.md",
    "docs/research/joint-gated-reference.md",
    "docs/research/prepared-region-gate.md",
    "docs/research/fixed-weight-correlation-gate.md",
    "docs/research/sota-breakthrough-reference.md",
    "docs/research/residual-delegation-reference.md",
    "python/pllm/runtime/rank_interface_reference.py",
    "python/pllm/runtime/mlp_channel_extraction.py",
    "docs/evidence/research-prepared-region-gate.json",
    "docs/evidence/network-algebra-qwen25-2026-10-06.json",
    "docs/evidence/rank-interface-output-oracle-qwen25-2026-09-29.json",
    "docs/evidence/affine-residual-oracle-qwen25-2026-09-29.json",
    "docs/evidence/extracted-mlp-channels-qwen25-2026-09-29.json",
)


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def f32(number: float) -> float:
    return value(bits(number))


def rational_f32_bits(number: Fraction) -> int:
    """Independent exact nearest-even oracle for finite rational results.

    Zero is positive; signed-zero propagation is tested separately with bit copies.
    No hardware float conversion is used here.
    """
    if number == 0:
        return 0
    sign = int(number < 0) << 31
    number = abs(number)
    n, d = number.numerator, number.denominator
    exponent = n.bit_length() - d.bit_length()
    if number < (Fraction(2) ** exponent):
        exponent -= 1
    shift = max(exponent - 23, -149)
    scaled = number / (Fraction(2) ** shift)
    q, r = divmod(scaled.numerator, scaled.denominator)
    q += int(2 * r > scaled.denominator or (2 * r == scaled.denominator and q % 2))
    if q == 0:
        return sign
    if q >= 1 << 24:
        q >>= 1
        shift += 1
    if shift == -149 and q < 1 << 23:
        return sign | q
    biased = shift + 23 + 127
    if biased >= 255:
        raise ValueError("oracle fixture overflow")
    return sign | (biased << 23) | (q - (1 << 23))


def _oracle_op(a: int, b: int, multiply: bool) -> int:
    x, y = Fraction(value(a)), Fraction(value(b))
    return rational_f32_bits(x * y if multiply else x + y)


def quotient(signatures: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exact ordered DAG signatures, not algebraic rank or approximate equality.

    Production signatures must include source, operation, row bytes, scales, bias,
    quantization, reduction order, backend and every position/state dependency.
    """
    representatives: list[int] = []
    inverse: list[int] = []
    seen: dict[tuple[int, ...], int] = {}
    for j, signature in enumerate(signatures):
        if signature not in seen:
            seen[signature] = len(representatives)
            representatives.append(j)
        inverse.append(seen[signature])
    return tuple(representatives), tuple(inverse)


def ordered_block(gates: tuple[float, ...], ups: tuple[float, ...], weights: tuple[float, ...],
                  *, compact: bool) -> int:
    """Synthetic original: h=RN(RN(g*g)*u), y=ordered RN-add(RN(w*h)).

    Square is a declared toy nonlinearity, NOT a SiLU replacement. Compact path
    copies exact representative bits but keeps every original down-projection node.
    """
    if not len(gates) == len(ups) == len(weights) or not gates:
        raise ValueError("nonempty aligned channels required")
    signatures = tuple((bits(g), bits(u)) for g, u in zip(gates, ups, strict=True))
    reps, inverse = quotient(signatures) if compact else (tuple(range(len(gates))), tuple(range(len(gates))))
    h = [f32(f32(gates[j] * gates[j]) * ups[j]) for j in reps]
    total = 0.0
    for j, weight in enumerate(weights):
        total = f32(total + f32(weight * h[inverse[j]]))
    return bits(total)


def ordered_block_oracle(gates: tuple[float, ...], ups: tuple[float, ...],
                         weights: tuple[float, ...]) -> int:
    total = 0
    for g, u, w in zip(gates, ups, weights, strict=True):
        square = _oracle_op(bits(g), bits(g), True)
        h = _oracle_op(square, bits(u), True)
        term = _oracle_op(bits(w), h, True)
        total = _oracle_op(total, term, False)
    return total


def _program_binding(gate_rows: tuple, up_rows: tuple, down: tuple) -> str:
    if (not gate_rows or len(gate_rows) != len(up_rows) or len(down) != len(gate_rows)
            or not gate_rows[0] or any(len(row) != len(gate_rows[0]) for row in (*gate_rows, *up_rows))):
        raise ValueError("invalid source program shapes")
    words = [[bits(x) for x in row] for row in (*gate_rows, *up_rows, down)]
    if any(w & 0x7F800000 == 0x7F800000 for row in words for w in row):
        raise ValueError("finite source coefficients required")
    payload = ["synthetic-ordered-binary32-square-gate-separate-mul-add-v1", words]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CompiledQuotient:
    source_digest: str
    representatives: tuple[int, ...]
    inverse: tuple[int, ...]


def compile_quotient(gate_rows: tuple, up_rows: tuple, down: tuple) -> CompiledQuotient:
    """Compile from source coefficient bits only; no inputs/calibration observed."""
    binding = _program_binding(gate_rows, up_rows, down)
    signatures = tuple(tuple(bits(x) for x in (*g, *u))
                       for g, u in zip(gate_rows, up_rows, strict=True))
    representatives, inverse = quotient(signatures)
    return CompiledQuotient(binding, representatives, inverse)


def evaluate_compiled(compiled: CompiledQuotient, gate_rows: tuple, up_rows: tuple,
                      down: tuple, inputs: tuple[float, ...]) -> int:
    if _program_binding(gate_rows, up_rows, down) != compiled.source_digest:
        raise ValueError("source binding mismatch")
    # In production, the compiler certificate must be authenticated, not supplied
    # by an untrusted caller. Recompile this bounded fixture to reject forged maps.
    if compiled != compile_quotient(gate_rows, up_rows, down):
        raise ValueError("invalid quotient certificate")
    if len(inputs) != len(gate_rows[0]):
        raise ValueError("input width mismatch")

    def dot(row: tuple) -> float:
        total = 0.0
        for w, x in zip(row, inputs, strict=True):
            total = f32(total + f32(f32(w) * f32(x)))
        return total

    gates, ups = tuple(map(dot, gate_rows)), tuple(map(dot, up_rows))
    h = [f32(f32(gates[j] * gates[j]) * ups[j]) for j in compiled.representatives]
    total = 0.0
    for j, w in enumerate(down):
        total = f32(total + f32(f32(w) * h[compiled.inverse[j]]))
    return bits(total)


def source_program_oracle(gate_rows: tuple, up_rows: tuple, down: tuple, inputs: tuple) -> int:
    def dot(row: tuple) -> int:
        total = 0
        for w, x in zip(row, inputs, strict=True):
            total = _oracle_op(total, _oracle_op(bits(w), bits(x), True), False)
        return total

    total = 0
    for g, u, w in zip(gate_rows, up_rows, down, strict=True):
        gate, up = dot(g), dot(u)
        h = _oracle_op(_oracle_op(gate, gate, True), up, True)
        total = _oracle_op(total, _oracle_op(bits(w), h, True), False)
    return total


def rounding_certificate(prefix: int, tail_l1_bound: int) -> int | None:
    """Certify RN32(exact_integer_dot), before the original scale/bias nodes.

    Public B bounds the unknown tail by [-B,B]. Monotonic RN is constant on this
    interval iff both endpoint bits agree. No token-margin or statistical test.
    """
    if type(prefix) is not int or type(tail_l1_bound) is not int or tail_l1_bound < 0:
        raise ValueError("integer prefix and nonnegative public bound required")
    if abs(prefix) + tail_l1_bound > 1 << 53:
        raise ValueError("bounded fixture requires exact binary64 integer conversion")
    low, high = bits(float(prefix - tail_l1_bound)), bits(float(prefix + tail_l1_bound))
    return low if low == high else None


def certified_dot(weights: tuple[int, ...], inputs: tuple[int, ...], prefix_length: int,
                  input_bound: int) -> tuple[int, bool]:
    if (not weights or len(weights) != len(inputs) or not 0 <= prefix_length <= len(weights)
            or type(input_bound) is not int or input_bound < 0
            or any(type(x) is not int or abs(x) > input_bound for x in inputs)
            or any(type(w) is not int for w in weights)):
        raise ValueError("invalid bounded integer-dot fixture")
    prefix = sum(weights[j] * inputs[j] for j in range(prefix_length))
    bound = input_bound * sum(abs(w) for w in weights[prefix_length:])
    prediction = rounding_certificate(prefix, bound)
    if prediction is not None:
        return prediction, True
    exact = sum(w * x for w, x in zip(weights, inputs, strict=True))
    if abs(exact) > 1 << 53:
        raise ValueError("bounded fixture exceeds binary64 exact integers")
    return bits(float(exact)), False


# Complete synthetic decoder state: hidden, KV cell, position parity, sampler RNG.
STATES = tuple((h, k, p, r) for h in range(4) for k in range(4) for p in range(2) for r in range(4))


def transition(state: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, k, p, r = state
    logits = (h + k, 3 - h + p)
    # A declared exact finite sampler; RNG advances exactly once per transition.
    token = int((r % (sum(logits) + 1)) >= logits[0])
    return ((h * h + k + token + p) % 4, (k + h + p) % 4, 1 - p, (3 * r + 1) % 4)


def observables(state: tuple[int, int, int, int]) -> tuple[int, ...]:
    h, k, p, r = state
    logits = (h + k, 3 - h + p)
    token = int((r % (sum(logits) + 1)) >= logits[0])
    return (*state, *logits, token)


def transition_table() -> tuple[int, ...]:
    """Independent table-generation expression rather than calling transition()."""
    index = {s: i for i, s in enumerate(STATES)}
    result = []
    for h, k, p, r in STATES:
        chosen = 0 if r % (h + k + 3 - h + p + 1) < h + k else 1
        result.append(index[((h ** 2 + k + chosen + p) & 3,
                             (k + h + p) & 3, p ^ 1, (3 * r + 1) & 3)])
    return tuple(result)


def lift_step(share: list[int], table: tuple[int, ...], prime: int = 257) -> list[int]:
    if len(share) != len(table) or any(not 0 <= j < len(table) for j in table):
        raise ValueError("invalid public transition table")
    output = [0] * len(share)
    # Scatter-add, NOT overwrite: transition need not be a permutation.
    for j, target in enumerate(table):
        output[target] = (output[target] + share[j]) % prime
    return output


def decode_share(share: list[int], prime: int = 257) -> list[int]:
    return [sum(a * observables(s)[j] for a, s in zip(share, STATES, strict=True)) % prime
            for j in range(7)]


def geometry(prompt: int, outputs: int, *, layers: int = 24, width: int = 4864) -> dict:
    if min(prompt, outputs, layers, width) < 1:
        raise ValueError("positive dimensions required")
    rows = prompt + outputs - 1
    full = layers * width * rows
    pruned = width * ((layers - 1) * rows + outputs)
    return {
        "input_tokens": prompt, "output_tokens": outputs, "layers": layers, "width": width,
        "executed_rows_per_unpruned_layer": rows,
        "unpruned_gated_elements": full,
        "terminal_pruned_gated_elements": pruned,
        "cut_payload_6_ring32_words_per_element_bytes": 24 * pruned,
        "one_and_per_lane_half_gate_material_bytes": 32 * pruned,
        "two_ring32_input_openings_both_workers_bytes": 16 * pruned,
        "scope": "analytical components, not all-link total or float backend admission",
    }


def integer_tail_veto(input_width: int, activation_bound: int = 127, weight_bound: int = 128) -> dict:
    """Full-domain bound for the specified exact-integer -> RN32 certificate.

    One unexamined nonzero integer weight contributes public uncertainty B>=A.
    RN32 cells in the entire bounded sum range have width at most max(1, ULP).
    This says nothing about later scaled/rounded nonlinear results or tight
    source-proved correlated input domains.
    """
    if min(input_width, activation_bound, weight_bound) < 1:
        raise ValueError("positive bounds required")
    maximum = input_width * activation_bound * weight_bound
    max_cell = 1 << max(0, maximum.bit_length() - 1 - 23)
    return {
        "input_width": input_width, "activation_abs_bound": activation_bound,
        "weight_abs_bound": weight_bound, "max_abs_integer_dot": maximum,
        "max_rounding_cell_width_integer_units": max_cell,
        "minimum_nonzero_tail_interval_width": 2 * activation_bound,
        "any_nonzero_tail_vetoed": 2 * activation_bound > max_cell,
        "scope": "symmetric full-domain public L1 tail bound, one integer RN32 cast only",
    }


def frozen_geometry(algebra: dict) -> dict:
    stages = algebra["stages"]
    gated = [s for s in stages if s["stage"].endswith(".mlp.gate_up_proj")]
    if not gated or any(s["shape"] != gated[0]["shape"] for s in gated):
        raise ValueError("unsupported frozen gate geometry")
    pair_width, hidden = gated[0]["shape"]
    if pair_width % 2 or len(stages) != 4 * len(gated):
        raise ValueError("incomplete frozen stage coverage")
    return {
        "layers": len(gated), "intermediate_width": pair_width // 2, "hidden_width": hidden,
        "max_body_input_width": max(s["shape"][1] for s in stages),
        "historically_omitted_signed_duplicate_rows": sum(s["row_dictionary"]["omitted_rows"] for s in stages),
        "historical_body_stage_count": len(stages),
        "gate_up_integer_tail_veto": integer_tail_veto(hidden),
        "all_body_integer_tail_veto": integer_tail_veto(max(s["shape"][1] for s in stages)),
    }


def quotient_ledger(*, original: int, compact: int, rows: int, word_bytes: int = 4,
                    one_use_bytes_per_lane: int = 0, requests: int = 1) -> dict:
    """Conditional cut layout, one layer; unchanged body/artifacts remain unpriced.

    Two workers send g,u shares (4 words); client returns h shares (2 words).
    Both workers receive an original-to-compact uint32 map. Client receives K.
    mu counts total Preparation-to-both-worker lane material, not per worker.
    """
    if not (0 < compact <= original and rows > 0 and requests > 0 and word_bytes > 0
            and one_use_bytes_per_lane >= 0):
        raise ValueError("invalid quotient ledger")
    setup = {"source_to_worker_0_map": 4 * original,
             "source_to_worker_1_map": 4 * original,
             "source_to_client_width": 4}
    links = {
        "worker_0_to_client": requests * rows * compact * 2 * word_bytes,
        "worker_1_to_client": requests * rows * compact * 2 * word_bytes,
        "client_to_worker_0": requests * rows * compact * word_bytes,
        "client_to_worker_1": requests * rows * compact * word_bytes,
        "preparation_to_both_workers_material": requests * rows * compact * one_use_bytes_per_lane,
    }
    gross = requests * rows * (original - compact) * (6 * word_bytes + one_use_bytes_per_lane)
    return {
        "setup_links_bytes": setup, "repeated_links_bytes": links,
        "priced_component_total_bytes": sum(setup.values()) + sum(links.values()),
        "component_net_saving_bytes": gross - sum(setup.values()),
        "gross_saving_per_request_bytes": gross // requests,
        "fresh_client_random_words": requests * rows * compact,
        "unpriced": ["unchanged body and checkpoint delivery to both workers", "source certificate",
                     "framing/authentication", "protected float/share conversions", "failures and fallback",
                     "complete client RSS and scratch", "preprocessing beyond explicit mu"],
        "whole_response_promotion": False,
    }


def controls() -> dict:
    fixtures = (
        ((1.25, 1.25, -0.5, 1.25), (2.0, 2.0, 0.25, 2.0), (1.0, -2.0, 3.0, 0.5)),
        ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (float(1 << 24), 1.0, -float(1 << 24))),
        ((2.0 ** -60, 2.0 ** -60), (1.0, 1.0), (1.0, 1.0)),
    )
    for gates, ups, weights in fixtures:
        assert ordered_block(gates, ups, weights, compact=True) == ordered_block_oracle(gates, ups, weights)
        assert ordered_block(gates, ups, weights, compact=False) == ordered_block_oracle(gates, ups, weights)
    gate_rows = ((1.0, 0.25), (-0.5, 2.0), (1.0, 0.25), (1.0, 0.25))
    up_rows = ((2.0, -0.25), (1.0, 0.5), (2.0, -0.25), (2.0, -0.25))
    down = (1.0, 0.5, -2.0, 4.0)
    compiled = compile_quotient(gate_rows, up_rows, down)
    program_cases = 0
    for x in (-2.0, -0.25, 1.0, 4.0):
        for y in (-0.5, 0.25, 2.0):
            assert evaluate_compiled(compiled, gate_rows, up_rows, down, (x, y)) == source_program_oracle(
                gate_rows, up_rows, down, (x, y))
            program_cases += 1
    # Reassociating a down reduction is not part of the valid quotient.
    a, b, c = float(1 << 24), 1.0, -float(1 << 24)
    original, fused = bits(f32(f32(a + b) + c)), bits(f32(f32(a + c) + b))
    assert original != fused
    # Rational rank one does not certify a rounding-preserving factorization.
    # W1=[2^24,1,-2^24]^T, W2=[1,1,1], x=1. Both matrices
    # have rank one. Exact algebra W2*W1=1, original ordered float result=0.
    rank_witness = (original, bits(f32(1.0)))
    assert rank_witness[0] != rank_witness[1]
    # Signed-zero input identity is bitwise, even when mathematical values coincide.
    reps, _ = quotient(((bits(0.0),), (bits(-0.0),)))
    assert len(reps) == 2
    rng = random.Random(20261016)
    table = transition_table()
    checked = 0
    for initial in STATES:
        truth = initial
        left = [rng.randrange(257) for _ in STATES]
        j = STATES.index(initial)
        right = [(int(i == j) - a) % 257 for i, a in enumerate(left)]
        for _ in range(6):
            left, right = lift_step(left, table), lift_step(right, table)
            truth = transition(truth)
            result = tuple((a + b) % 257 for a, b in zip(decode_share(left), decode_share(right), strict=True))
            assert result == observables(truth)
            checked += 1
    certificates, fallbacks = 0, 0
    for prefix in (0, 1, -1, (1 << 25) + 4, -(1 << 25) - 4, 1 << 24, (1 << 24) + 1):
        for bound in range(5):
            predicted = rounding_certificate(prefix, bound)
            for tail in range(-bound, bound + 1):
                exact = rational_f32_bits(Fraction(prefix + tail))
                if predicted is not None:
                    assert predicted == exact
                    certificates += 1
                else:
                    fallbacks += 1
    # A genuine skipped nonzero tail, and an adversarial rounding-boundary miss.
    positive = certified_dot(((1 << 25) + 4, 1), (1, 1), 1, 1)
    negative = certified_dot((1 << 24, 1), (1, 1), 1, 1)
    assert positive[1] and not negative[1]
    return {
        "ordered_quotient_fixtures_exact": len(fixtures),
        "source_compiled_quotient_input_cases_exact": program_cases,
        "source_compiled_quotient_widths": [len(gate_rows), len(compiled.representatives)],
        "synthetic_program_source_digest": compiled.source_digest,
        "unsafe_reassociation_witness_bits": [original, fused],
        "rank_one_reduction_order_witness_bits": list(rank_witness),
        "signed_zero_distinct_classes": len(reps),
        "lift_states": len(STATES), "lift_steps_full_state_logits_selection_rng_exact": checked,
        "rounding_certified_tail_cases": certificates, "rounding_rejected_tail_cases": fallbacks,
        "nonzero_tail_skipped": positive[1], "boundary_forces_exact_fallback": not negative[1],
    }


def source_snapshot() -> dict:
    result = {}
    for relative in (*SOURCES, "scripts/probe_exact_adapter_frontier.py", "tests/test_exact_adapter_frontier.py",
                     "docs/research/exact-adapter-frontier-reference.md"):
        data = (ROOT / relative).read_bytes()
        result[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return result


def report() -> dict:
    frozen = json.loads((ROOT / "docs/evidence/research-prepared-region-gate.json").read_text())
    algebra = json.loads((ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json").read_text())
    if frozen["source"]["body_fingerprint"] != algebra["source"]["body_fingerprint"]:
        raise ValueError("frozen artifacts have different body identities")
    metadata = frozen_geometry(algebra)
    return {
        "schema": "pllm.exact_adapter_frontier.v1",
        "scope": "synthetic exact controls and analytical conditional ledgers; no secure backend",
        "source_snapshot": source_snapshot(),
        "frozen_source_identity": frozen["source"],
        "frozen_algebra_source_identity": algebra["source"],
        "frozen_geometry_and_vetoes": metadata,
        "checkpoint_tensor_loading": False,
        "checkpoint_duplicate_count": None, "checkpoint_rounding_certificate_hit_rate": None,
        "controls": controls(),
        "qwen_geometry": [geometry(150, n, layers=metadata["layers"], width=metadata["intermediate_width"])
                          for n in (8, 64)],
        "quotient_synthetic_ledger": quotient_ledger(original=8, compact=2, rows=157,
                                                     one_use_bytes_per_lane=32),
        "finite_lift": {
            "toy_states": len(STATES), "field_word_bytes": 2,
            "client_to_each_worker_initial_share_bytes": 2 * len(STATES),
            "each_worker_to_client_final_observable_share_bytes": 2 * 7,
            "both_worker_transition_tables_uint16_bytes": 2 * 2 * len(STATES),
            "both_worker_output_tables_uint16_bytes": 2 * 2 * 7 * len(STATES),
            "one_use_initial_mask_words": len(STATES),
            "dense_raw_float32_unary_initial_shares_bytes": 2 * 4 * (1 << 32),
            "dense_raw_float32_pair_initial_shares_bytes": 2 * 4 * (1 << 64),
            "client_target_bytes": 256 * (1 << 20),
            "security": "ideal independent uniform shares; deterministic fixture RNG is not cryptographic",
            "extra_costs": ["source/state indexing", "authentication/framing", "both-worker source artifacts",
                            "private entry decoding", "failure retirement", "real checkpoint reachability proof"],
        },
        "rounding_certificate": {
            "map": "prefix P and public tail bound B -> unique RN32 integer cell, else exact fallback",
            "source_contract": "exact integer dot then one RN32 cast; not arbitrary ordered float dot",
            "checkpoint_admission": False,
            "private_certificate_cost": None,
            "unpadded_miss_pattern_disclosure": True,
            "fixed_padding_worst_case_saving_bytes": 0,
            "optimistic_expected_component_bytes": "A + V + (1-h)*F; beats F iff A+V < h*F",
            "worst_case_component_bytes": "A + V + F; add artifacts, setup, failures and RNG/state rollback",
        },
        "decision": "no whole-model adapter admitted; only tighter source-certified rounded-block equivalence remains open",
        "whole_client_peak_bytes": None, "full_wire_bytes": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
