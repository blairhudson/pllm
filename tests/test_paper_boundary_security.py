"""Bounded P2-B algebra and negative-contract witnesses, not cryptographic proofs.

No model, checkpoint, service, build, or benchmark is started. The coalition model
uses an ideal dense XOR point share, not the implemented AES DPF. Native checks
exercise the already-installed extension; its hash is recorded by standalone
replay. All seeds, inputs and masks in this file are public synthetic fixtures.

Replay: PYTHONPATH=python .venv/bin/python tests/test_paper_boundary_security.py \
    --output paper/artifacts/security/boundary-security-checks.json
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import hashlib
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

import pytest


def _center(value, bits):
    modulus = 1 << bits
    value %= modulus
    return value - modulus if value >= modulus // 2 else value


def check_head_residue_exhaustive():
    """Exhaust all Z/16Z shares of small signed two-coordinate dot products."""
    cases = 0
    modulus = 16
    for weights in product(range(-2, 3), repeat=2):
        bound = sum(abs(w) for w in weights)
        bits = max(1, (2 * bound).bit_length())
        assert 2 * bound < 1 << bits
        for activation in product(range(-1, 2), repeat=2):
            expected = sum(w * x for w, x in zip(weights, activation, strict=True))
            for right in product(range(modulus), repeat=2):
                left = tuple((x - r) % modulus for x, r in zip(activation, right, strict=True))
                outputs = [sum(w * x for w, x in zip(weights, share, strict=True)) % modulus
                           for share in (left, right)]
                assert _center(sum(outputs), 4) == expected
                assert _center(sum(y % (1 << bits) for y in outputs), bits) == expected
                cases += 1
    return {"cases": cases, "ring_modulus": modulus, "interpretation": "finite algebra check"}


def _xor_lookup(table, selector):
    value = 0
    for i, record in enumerate(table):
        if selector & (1 << i):
            value ^= record
    return value


def _coalition_histograms(token):
    """Correlated lookup -> body -> nonlinear hidden; independent protocol coins."""
    roles = ("P", "I", "B0", "B1")
    coalitions = [tuple(c) for n in range(5) for c in combinations(roles, n)]
    histograms = {c: Counter() for c in coalitions}
    table, modulus, weight = (1, 2, 3), 4, 3
    x = table[token]
    hidden = (x * x + 1) % modulus
    for r, s, a, selector in product(range(4), range(4), range(4), range(8)):
        correction = (weight * r - s) % modulus
        masked = (x - r) % modulus
        reply = (weight * masked + correction) % modulus
        other_selector = selector ^ (1 << token)
        left = (hidden - a) % modulus
        views = {
            "P": (r, s, correction),
            "I": (correction, masked, reply),
            "B0": (selector, _xor_lookup(table, selector), left, weight * left % modulus),
            "B1": (other_selector, _xor_lookup(table, other_selector), a, weight * a % modulus),
        }
        assert (reply + s) % modulus == weight * x % modulus
        for coalition in coalitions:
            histograms[coalition][tuple(views[role] for role in coalition)] += 1
    return histograms


def check_joint_coalition_views():
    first, second = _coalition_histograms(0), _coalition_histograms(1)
    allowed, forbidden = [], []
    for coalition in first:
        roles = set(coalition)
        admitted = not ({"P", "I"} <= roles or {"B0", "B1"} <= roles)
        same = first[coalition] == second[coalition]
        assert same == admitted, coalition
        (allowed if admitted else forbidden).append(list(coalition))
    assert len(allowed) == 9 and len(forbidden) == 7
    return {"assignments_per_input": 512, "allowed": allowed, "forbidden": forbidden,
            "interpretation": "exact ideal-XOR toy distributions; not AES-DPF security evidence"}


def check_coin_dependent_leakage_breaks_payload_composition():
    """Equal payload/timing marginals do not imply equal joint observations."""
    joint = {
        q: Counter(((q - a) % 4, a % 2) for a in range(4))
        for q in range(4)
    }
    shares = [Counter({left: sum(count for (l, _), count in view.items() if l == left)
                       for left in range(4)}) for view in joint.values()]
    timing = [Counter({bit: sum(count for (_, t), count in view.items() if t == bit)
                       for bit in range(2)}) for view in joint.values()]
    assert all(view == shares[0] for view in shares)
    assert all(view == timing[0] for view in timing)
    assert set(joint[0]).isdisjoint(joint[1])
    assert joint[0] == joint[2] and joint[1] == joint[3]
    for q, view in joint.items():
        assert all((left + leaked_bit) % 2 == q % 2 for left, leaked_bit in view)
    return {"modulus": 4, "assignments_per_input": 4,
            "payload_marginals_equal": True, "timing_marginals_equal": True,
            "joint_view_recovers_q_parity": True,
            "scope": "synthetic T=a mod 2 leakage; no actual runtime timing exploit established"}


def check_prepared_causal_simulator_and_conditional_masks():
    """Offline c leaves r uniform, while s=Ar-c need not remain uniform."""
    modulus, cases = 4, 0
    for weight in range(modulus):
        for x in range(modulus):
            real = Counter(
                ((weight * r - s) % modulus, (x - r) % modulus, (weight * x - s) % modulus)
                for r, s in product(range(modulus), repeat=2)
            )
            # Match the causal order: sample c offline, then u online; derive v.
            simulated = Counter(
                (c, u, (weight * u + c) % modulus)
                for c in range(modulus) for u in range(modulus)
            )
            assert real == simulated
            cases += 1
        for c in range(modulus):
            compatible = [(r, s) for r, s in product(range(modulus), repeat=2)
                          if (weight * r - s) % modulus == c]
            assert Counter(r for r, _ in compatible) == Counter(range(modulus))
            assert all(s == (weight * r - c) % modulus for r, s in compatible)
            if weight == 0:
                assert {s for _, s in compatible} == {(-c) % modulus}
    return {"modulus": modulus, "weight_input_cases": cases,
            "offline_c_then_online_u_matches_real": True,
            "r_conditioned_on_c_uniform": True, "zero_matrix_c_determines_s": True,
            "scope": "finite joint-distribution and conditional-mask check"}


def check_packed_body_collusion_recovers_only_transmitted_residue():
    """With A=0 and one-bit input coding, different W8A8 inputs can coincide."""
    cases = 0
    inputs = ((1, 127), (3, 127))  # Shared qmax anchor makes both quantizations attainable.
    for r0, r1, s in product(range(4), repeat=3):
        r = (r0, r1)
        correction = (-s) % 2
        views = [(r, s, correction, tuple((x - mask) % 2 for x, mask in zip(value, r, strict=True)),
                  correction) for value in inputs]
        assert views[0] == views[1]
        assert tuple((u + mask) % 2 for u, mask in zip(views[0][3], r, strict=True)) == (1, 1)
        cases += 1
    widths = (8, 9, 16, 24, 32)
    for bits in widths:
        residues = {x % (1 << bits): x for x in range(-127, 128)}
        assert len(residues) == 255
        assert all(_center(residue, bits) == x for residue, x in residues.items())
    return {"one_bit_collision_cases": cases, "indistinguishable_inputs": inputs,
            "w8a8_injective_widths_checked": list(widths),
            "scope": "degenerate zero-matrix example; P+I remains conservatively prohibited"}


def check_output_mask_is_essential():
    modulus, weight, x = 256, 3, 17
    recovered_products = set()
    for r in range(modulus):
        masked = (x - r) % modulus
        wrong_correction = weight * r % modulus
        recovered_products.add((weight * masked + wrong_correction) % modulus)
    assert recovered_products == {weight * x % modulus}
    return {"omitted_output_mask_reveals": "W*x", "example": weight * x % modulus,
            "scope": "counterfactual variant; implementation uses output-s"}


def check_cross_role_randomness_reuse_breaks_mixed_pairs():
    modulus, body_x, hidden, r = 256, 17, 29, 211
    body_masked = (body_x - r) % modulus
    # Counterfactual: use the body's r as the boundary's right share.
    left, right = (hidden - r) % modulus, r
    assert (body_masked + right) % modulus == body_x  # I+B1
    assert (left + r) % modulus == hidden  # P+B0
    return {"I+B1_recovers_body_x": body_x, "P+B0_recovers_hidden": hidden,
            "scope": "counterfactual shared randomness; production head draws independent OS bytes"}


def check_head_mask_reuse_reveals_differences():
    modulus, a, first, second = 256, 211, 17, 29
    left0, left1 = (first - a) % modulus, (second - a) % modulus
    recovered = (left1 - left0) % modulus
    assert recovered == (second - first) % modulus
    return {"recovered_hidden_difference": recovered,
            "scope": "counterfactual reuse; production draws a new vector per head call"}


def check_underbound_residue_alias_passes_a_global_range_check():
    from pllm import _native

    expected, claimed_bound, bits = 6, 6, 3
    # An undersized public layout can map a valid result to another valid result.
    a, b = struct.pack("<I", 123), struct.pack("<I", (expected - 123) % (1 << 32))
    packed = [_native.offset_pack_rows(value, bytes([bits]), 1) for value in (a, b)]
    decoded = struct.unpack("<q", _native.offset_reconstruct_rows(*packed, bytes([bits]), 1))[0]
    assert decoded == -2 and decoded != expected and abs(decoded) <= claimed_bound
    return {"true_result": expected, "undersized_bits": bits, "decoded_result": decoded,
            "global_bound": claimed_bound, "range_check_passes": True,
            "scope": "missing trusted-bound assumption, not an accepted honest-worker certificate"}


def check_native_head_against_integer_oracle():
    import numpy as np
    from pllm import _native
    from pllm.runtime.native import MaskedGEMM

    weights = ((-128, 127, -1, 0), (0, 0, 0, 0), (127, 127, 127, 127), (-128, -128, -128, -128))
    activation = (-127, 127, -127, -127)
    right = (0xffffffff, 0x80000000, 7, 0)
    left = tuple((x - r) % (1 << 32) for x, r in zip(activation, right, strict=True))
    matrix = MaskedGEMM(threads=1).compile(np.asarray(weights, np.int8))
    widths = bytes(max(1, (2 * 127 * sum(abs(w) for w in row)).bit_length()) for row in weights)
    packed = []
    for share in (left, right):
        output = matrix.wrap32(np.asarray([share], dtype=np.uint32))
        packed.append(_native.offset_pack_rows(output.astype("<u4", copy=False).tobytes(), widths, 1))
    actual = list(struct.unpack("<4q", _native.offset_reconstruct_rows(*packed, widths, 1)))
    expected = [sum(w * x for w, x in zip(row, activation, strict=True)) for row in weights]
    assert actual == expected
    return {"expected": expected, "actual": actual, "widths": list(widths),
            "scope": "installed native wrap32 and row codec, independent Python integer oracle"}


def check_cancelled_owner_retains_rederivable_seed():
    import numpy as np
    from pllm.runtime.preparation_protocol import PreparationRequest, expand_preparation_mask
    from pllm.runtime.transformer_client import PreparedStageRows

    request = PreparationRequest(attempt_id="1" * 32, session_id="public-test-session",
        model="public-test-model", body_fingerprint="2" * 64, stage_id="public-test-stage",
        weight_digest="3" * 64, rows=1, in_features=2, out_features=2,
        weight_bits=8, activation_bits=8, signed_output_bound=100,
        ring="u16", modulus=1 << 16, wire_bits=16, seed=bytes(range(32)))
    owner = PreparedStageRows(request=request)
    before, _ = owner.take_masks(0, 1)
    owner.cancel()
    assert owner.request.seed == bytes(range(32))
    regenerated = expand_preparation_mask(owner.request)
    assert np.array_equal(before, regenerated)
    assert owner._mask_rows is not None
    with pytest.raises(ValueError):
        owner.take_masks(0, 1)
    return {"cancelled_api_rejects": True, "retained_seed_rederives_prior_mask": True,
            "scope": "client-owner custody witness; not proof of provider post-ACK retention"}


def check_retired_dpf_keys_survive_in_saved_transcripts():
    from pllm import _native

    records, width, selected = 129, 2, 128
    table = b"".join(i.to_bytes(width, "little") for i in range(records))
    context = hashlib.sha256(b"public-boundary-security-test-context").digest()

    def fresh(party):
        return _native.PrivatePageServer(table, width, party).bind_context(context)

    workers = [fresh(p) for p in (0, 1)]
    query0, query1, decoder = _native.private_page_issue(
        workers[0].descriptor()[0], records, width, selected, leaf_bits=128)
    saved = (query0, query1)
    replies = [worker.evaluate(query) for worker, query in zip(workers, saved, strict=True)]
    decoder.cancel()
    with pytest.raises(ValueError):
        decoder.decode(*replies)
    with pytest.raises(ValueError):
        workers[0].evaluate(saved[0])
    # A fresh process-equivalent owner has no persistent context/replay ledger.
    replayed = [fresh(p).evaluate(query) for p, query in enumerate(saved)]
    assert replayed == replies
    recovered = bytes(a ^ b for a, b in zip(replayed[0][65:], replayed[1][65:], strict=True))
    assert recovered == selected.to_bytes(width, "little")
    return {"same_owner_replay_rejects": True, "fresh_owner_accepts_saved_keys": True,
            "recovered_synthetic_record": int.from_bytes(recovered, "little"),
            "scope": "both historical key shares; forbidden cumulative B0+B1 coalition"}


CHECKS = (
    check_head_residue_exhaustive,
    check_joint_coalition_views,
    check_coin_dependent_leakage_breaks_payload_composition,
    check_prepared_causal_simulator_and_conditional_masks,
    check_packed_body_collusion_recovers_only_transmitted_residue,
    check_output_mask_is_essential,
    check_cross_role_randomness_reuse_breaks_mixed_pairs,
    check_head_mask_reuse_reveals_differences,
    check_underbound_residue_alias_passes_a_global_range_check,
    check_native_head_against_integer_oracle,
    check_cancelled_owner_retains_rederivable_seed,
    check_retired_dpf_keys_survive_in_saved_transcripts,
)


@pytest.mark.parametrize("check", CHECKS, ids=lambda check: check.__name__)
def test_paper_boundary_security(check):
    check()


def _evidence():
    from pllm import _native
    import pllm.runtime.private_boundary_transport as boundary_transport

    root = Path(__file__).resolve().parents[1]
    files = (
        Path(__file__).resolve(), Path(_native.__file__).resolve(),
        Path(boundary_transport.__file__).resolve(),
        root / "crates/pllm-garble/src/private_pages.rs",
        root / "crates/pllm-core/src/mask_stream.rs",
        root / "crates/pllm-core/src/offset_transport.rs",
        root / "python/pllm/runtime/preparation_protocol.py",
        root / "python/pllm/runtime/transformer_engine.py",
        root / "python/pllm/runtime/transformer_client.py",
        root / "paper/SECURITY-AND-CORRECTNESS.md",
    )
    return {
        "schema": "pllm.paper.boundary_security_checks.v1",
        "audit_revision": "post-review-untimed-coin-independent-leakage-v2",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": sys.version, "platform": platform.platform(),
        "artifacts": {str(path.relative_to(root) if path.is_relative_to(root) else path):
                      hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "scope": "bounded synthetic algebra/leakage/custody checks; no model load or cryptographic security proof",
        "theorem_scope": "untimed admitted honest execution with cryptographic-coin-independent semantic leakage",
        "native_build_matches_source": "not established; existing binary hashed, no rebuild performed",
        "checks": {check.__name__: {"status": "pass", **check()} for check in CHECKS},
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    raw = json.dumps(_evidence(), indent=2, sort_keys=True) + "\n"
    if options.output is None:
        print(raw, end="")
    else:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(raw, encoding="utf-8")
        print(options.output)
