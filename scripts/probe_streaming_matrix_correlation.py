"""Bounded exact-ring consumer and historical-body accounting; no crypto claims."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
INGRESS = ROOT / "docs/evidence/research-bootstrap-ingress.json"
NARROW = ROOT / "docs/evidence/research-narrow-ingress.json"
DEMAND = ROOT / "docs/evidence/research-attested-linear-bootstrap.json"
CONTROL = ROOT / "docs/evidence/research-all-link-frontier.json"
GEOMETRY = ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"
SOURCE_EXCERPTS = {
    "2025/787": "Within such existing protocols, we provide comparable online performance while improving the storage and offline dealer-to-party communication requirements by more than 3 orders of magnitude.",
    "2025/1660": "produce 10 million COTs on-the-fly and reducing the memory from 337 MB and 624 MB to 20 MB, respectively.",
    "2026/2030": "PCFs for correlated oblivious transfer (COT) and vector oblivious linear evaluation (VOLE) correlations over Z_M for an arbitrary modulus M",
}


def matvec(w: list[list[int]], x: list[int], bits: int) -> list[int]:
    if not w or not x or any(len(row) != len(x) for row in w):
        raise ValueError("invalid geometry")
    if not 1 <= bits <= 32:
        raise ValueError("invalid ring")
    return [sum(a * b for a, b in zip(row, x, strict=True)) % (1 << bits) for row in w]


def ideal_vector_ot(w: list[list[int]], r: list[int], bits: int,
                    rng: random.Random) -> tuple[list[int], list[int]]:
    """Trusted ideal chosen OT. Python RNG only generates public toy fixtures.

    Sender owns T and q. Receiver privately selects each bit of full-ring r.
    Quotient-ring masks reduce message width without changing exact-ring sum.
    No choice travels to Sender. Public delta lets Receiver derive both messages.
    """
    matvec(w, r, bits)
    if any(not 0 <= a < 1 << bits for a in r):
        raise ValueError("mask outside ring")
    s, q = [0] * len(w), [0] * len(w)
    for i, a in enumerate(r):
        for ell in range(bits):
            modulus = 1 << (bits - ell)
            for j, row in enumerate(w):
                t = rng.randrange(modulus)
                selected = (((a >> ell) & 1) * row[i] - t) % modulus
                s[j] = (s[j] + (selected << ell)) % (1 << bits)
                q[j] = (q[j] + (t << ell)) % (1 << bits)
    return s, q


class TicketBook:
    """Ideal external monotonic authority, NOT rollback-safe Python ownership."""

    def __init__(self, capacity: int, expiry: int):
        if capacity <= 0 or expiry <= 0:
            raise ValueError("invalid lifetime")
        self.capacity, self.expiry = capacity, expiry
        self.burned: set[tuple[str, int]] = set()

    def claim(self, context: str, index: int, now: int, valid: bool = True) -> None:
        if not context or type(index) is not int or not 0 <= index < self.capacity:
            raise ValueError("invalid identity")
        identity = context, index
        if identity in self.burned:
            raise ValueError("reused index")
        if len(self.burned) >= self.capacity:
            raise ValueError("total-use capacity exhausted")
        # Burn before checking expiry/malformed/cancel outcomes.
        self.burned.add(identity)
        if now >= self.expiry or not valid:
            raise ValueError("expired or cancelled/malformed")


def tv(left: Counter, right: Counter) -> Fraction:
    nl, nr = sum(left.values()), sum(right.values())
    return sum((abs(Fraction(left[k], nl) - Fraction(right[k], nr))
                for k in left.keys() | right.keys()), Fraction()) / 2


def privacy_screen() -> dict:
    # Full provider view (q, x-r), not just one opening. W=[3] in Z_8.
    def views(x: int, correlated: bool) -> Counter:
        return Counter(( (3*r - s) % 8, (x-r) % 8)
                       for r in range(8)
                       for s in ([2*r % 8] if correlated else range(8)))
    independent = tv(views(0, False), views(1, False))
    correlated = tv(views(0, True), views(1, True))
    # Reuse reveals x2-x1 regardless of W or output masking.
    repeated = [((2-r) % 8, (5-r) % 8) for r in range(8)]
    assert all((b-a) % 8 == 3 for a, b in repeated)
    # Low-dimensional shared pad exposes lane difference even at fresh index.
    rank_one = Counter(((0-r) % 8, (0-r) % 8) for r in range(8))
    shifted = Counter(((0-r) % 8, (1-r) % 8) for r in range(8))
    return {"full_view_independent_masks_tv": str(independent),
            "s_equals_2r_W_equals_3_tv": str(correlated),
            "rank_one_mask_tv": str(tv(rank_one, shifted)),
            "reused_r_opening_difference": 3,
            "provider_readable_seed": "r=Expand(seed,index); x=(x-r)+r",
            "scope": "exhaustive toy shortcut attacks, not PCF security certification"}


def public_delta_screen() -> dict:
    recovered = 0
    for w in range(8):
        for ell in range(3):
            modulus = 1 << (3-ell)
            for bit, t in itertools.product(range(2), range(modulus)):
                selected = (bit*w-t) % modulus
                recovered_t = (bit*w-selected) % modulus
                assert recovered_t == t
                other = (selected+(1-2*bit)*w) % modulus
                assert other == ((1-bit)*w-t) % modulus
                recovered += 1
    # Sender knows every T, not merely q. Ideal OT sends no choice transcript.
    def view(x: int) -> Counter:
        return Counter((*masks, (x-r) % 8)
                       for masks in itertools.product(range(8), range(4), range(2))
                       for r in range(8))
    return {"public_delta_recovered_other_messages": recovered,
            "full_sender_masks_and_opening_tv": str(tv(view(0), view(1))),
            "scope": "explicit ideal adapter only; not an OT communication lower bound"}


def algebra_screen() -> dict:
    rng = random.Random(78716602030)
    checked = 0
    for bits in (3, 16, 24, 32):
        w = [[-128, 127, -3], [1, 0, 7]]
        for _ in range(12):
            r = [rng.randrange(1 << bits) for _ in range(3)]
            s, q = ideal_vector_ot(w, r, bits, rng)
            assert [(a-b) % (1 << bits) for a, b in zip(matvec(w, r, bits), s)] == q
            x = [rng.randrange(-128, 128) for _ in r]
            z = [(a-b) % (1 << bits) for a, b in zip(x, r)]
            y = [(a+b+c) % (1 << bits) for a, b, c in zip(matvec(w, z, bits), q, s)]
            assert y == matvec(w, x, bits)
            checked += len(y)
    # Independent exhaustive ring3 consumer, including arbitrary signed W.
    for r, s, x in itertools.product(range(8), repeat=3):
        q, z = (3*r-s) % 8, (x-r) % 8
        assert (3*z+q+s) % 8 == 3*x % 8
    # Enumerate quotient masks independently: s is uniform for each private r.
    for r in range(8):
        distribution = Counter()
        for masks in itertools.product(range(8), range(4), range(2)):
            s = sum((1 << ell) * ((((r >> ell) & 1)*3-t) % (1 << (3-ell)))
                    for ell, t in enumerate(masks)) % 8
            distribution[s] += 1
        assert distribution == Counter({s: 8 for s in range(8)})
    return {"ideal_ot_checked_outputs": checked, "exhaustive_scalar_consumers": 512,
            "quotient_mask_uniformity_enumerations": 512,
            "rings": [3, 16, 24, 32], "cryptography_implemented": False}


def budget(pairs: int, target: Fraction, reuse: bool, bootstrap: int = 0,
           renewal: int = 0, capacity: int = 1) -> Fraction:
    if pairs <= 0 or min(bootstrap, renewal) < 0 or capacity <= 0 or not 0 < target < 1:
        raise ValueError("invalid budget")
    c0, t0, removed = 124_905_091, 368_670_156, 139_060_689
    control = (c0 if reuse else 0) + pairs * (t0 if reuse else c0+t0)
    # Available TOTAL replacement cost per pair; setup, renewal and failures debit it.
    return (pairs*removed - target*control - bootstrap
            - ((pairs+capacity-1)//capacity)*renewal) / pairs


def unique_stages(stages: list[dict]) -> dict:
    indexed = {s["stage"]: s for s in stages}
    if len(indexed) != len(stages):
        raise ValueError("duplicate stage identity")
    return indexed


def validate_geometry(ingress: list[dict], narrow: list[dict], demand: list[dict],
                      geometry: list[dict], control_stages: dict) -> None:
    i, n, d, g = (unique_stages(s) for s in (ingress, narrow, demand, geometry))
    if not set(i) == set(n) == set(d) == set(g) == set(control_stages):
        raise ValueError("stage identity set mismatch")
    for name, stage in i.items():
        if stage["original_ring_bits"] != g[name]["ring_bits"]:
            raise ValueError("original ring contract mismatch")
        if stage["stage_bits"] != n[name]["stage_bits"]:
            raise ValueError("encoded width contract mismatch")
        if not stage["shape"] == n[name]["shape"] == d[name]["shape"] == g[name]["shape"]:
            raise ValueError("stage shape mismatch")
        if not stage["weight_sha256"] == n[name]["weight_sha256"] == d[name]["weight_sha256"] == g[name]["weight_sha256"]:
            raise ValueError("weight identity mismatch")


def cost_screen() -> dict:
    ingress = json.loads(INGRESS.read_text())["cost"]
    control = json.loads(CONTROL.read_text())["control"]
    narrow = json.loads(NARROW.read_text())["checkpoint_geometry"]
    demand = json.loads(DEMAND.read_text())["budget"]
    geometry = json.loads(GEOMETRY.read_text())
    assert ingress["control_pair_bytes"] == control["pair_bytes"] == 493_575_247
    assert sum(control["classes"].values()) == control["pair_bytes"]
    assert demand["baseline"] == control
    assert narrow["lock"]["body_fingerprint"] == control["body_fingerprint"]
    assert geometry["source"]["body_fingerprint"] == control["body_fingerprint"]
    validate_geometry(ingress["stages"], narrow["stages"], demand["stages"],
                      geometry["stages"], control["stages"])
    ids = unique_stages(narrow["stages"])
    rows = unique_stages(demand["stages"])
    assert set(ids) == set(rows) == set(control["stages"])
    stages = []
    for s in ingress["stages"]:
        name, (m, n), k, r = s["stage"], s["shape"], s["original_ring_bits"], s["rows"]
        assert ids[name]["shape"] == rows[name]["shape"] == [m, n]
        assert ids[name]["weight_sha256"] == rows[name]["weight_sha256"] == s["weight_sha256"]
        assert r == rows[name]["pair_rows"]
        macs = r*m*n
        assert macs == s["body_macs"]
        # Pack each selected output vector separately, each quotient width.
        selected = r*n*sum((m*(k-ell)+7)//8 for ell in range(k))
        stages.append({"stage": name, "weight_sha256": s["weight_sha256"],
                       "shape": [m, n], "ring_bits": k, "rows": r,
                       "body_macs": macs, "client_ot_lane_additions": k*macs,
                       "selected_vector_bytes": selected,
                       "ROT_switch_bits": r*n*k,
                       "one_row_client_generated_r_s_bytes": (n+m)*(k//8),
                       "one_row_provider_generated_q_bytes": m*(k//8)})
    assert len(stages) == 96 and sum(s["rows"] for s in stages) == 15297
    totals = {key: sum(s[key] for s in stages) for key in
              ("body_macs", "client_ot_lane_additions", "selected_vector_bytes", "ROT_switch_bits")}
    totals["two_chosen_message_ciphertext_payload_bytes"] = 2*totals["selected_vector_bytes"]
    totals["packed_ROT_switch_bytes_lower_bound"] = (totals["ROT_switch_bits"]+7)//8
    return {"totals": totals, "stages": stages,
            "cost_scope": "explicit two-message vector-OT adapter only; not a communication lower bound",
            "OT_layout": "full-ring mask bits; quotient masks; two ciphertexts plus ROT choice switches",
            "horizons": [{"reuse": reuse, "pairs": h,
                          "total_new_cost_headroom_per_pair": {
                              str(p): str(budget(h, Fraction(p, 100), reuse)) for p in (20, 35, 50)}}
                         for reuse in (True, False) for h in (1, 2, 4, 16, 64, 1024)],
            "warm_headroom": {str(p): str(Fraction(139_060_689)-Fraction(p, 100)*368_670_156)
                              for p in (20, 35, 50)},
            "ceilings": {"first_pair": str(Fraction(139_060_689, 493_575_247)),
                         "recurring_reused_public": str(Fraction(139_060_689, 368_670_156))},
            "historical_not_latest_low_or_wire": True,
            "horizon_unit": "new150-token-prefix/eight-output request then exact repeat"}


def report() -> dict:
    sources = (Path(__file__), ROOT / "tests/test_streaming_matrix_correlation_probe.py",
               ROOT / "docs/research/streaming-matrix-correlation-reference.md",
               INGRESS, NARROW, DEMAND, CONTROL, GEOMETRY)
    return {"schema": "pllm.streaming_matrix_correlation_screen.v1", "complete": True,
            "scope": "standard-library bounded algebra, ideal OT and conditional body accounting",
            "algebra": algebra_screen(), "privacy": privacy_screen(),
            "public_delta": public_delta_screen(), "cost": cost_screen(),
            "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            "primary_sources": [{"url": "https://eprint.iacr.org/"+i,
                                 "evidence_level": "primary abstract verified; full PDF HTTP403",
                                 "excerpt": text,
                                 "normalized_excerpt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                 "pdf_sha256": None} for i, text in SOURCE_EXCERPTS.items()],
            "bounded_full_paper_followup": {
                "pages_checked": ["https://cs.idc.ac.il/~elette/", "https://u.cs.biu.ac.il/~nofarie/",
                                  "https://archisman.org/", "https://pascholl.github.io/publications",
                                  "https://lisakohl.me/", "https://www.cwi.nl/people/lisa-kohl",
                                  "https://ir.cwi.nl/ppl/9821", "https://ldr709.gitlab.io/"],
                "result": "author links only same ePrint copies; no alternate full paper located; Dory arXiv search found no author copy",
                "full_papers_verified": False},
            "runtime": {"python": sys.version, "executable": sys.executable,
                        "native_crypto_or_sdk_imported": False},
            "unknown_not_zero": ["matrix-PCF key bytes and generation", "reviewed security parameters",
                                  "client and provider expansion CPU", "whole client RSS",
                                  "authentication/verification/openings", "key renewal capacity and failure budget",
                                  "all-link setup/provider distribution/new helper/failed attempts/framing"],
            "client_peak_rss_bytes": None, "client_cpu_seconds": None,
            "full_wire_bytes": None, "complete_base_bytes": None, "per_token_bytes": None,
            "decision": "Ideal matrix consumer survives algebra. Explicit vector-OT adapter rejected by bytes/work. No source-certified compact client-light matrix generator found."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "complete": True}))


if __name__ == "__main__":
    main()
