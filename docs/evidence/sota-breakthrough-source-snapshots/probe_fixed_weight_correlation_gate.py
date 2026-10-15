"""Bounded fixed-W transcript/cost falsification. Not a production PCG or benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
MOD = 1 << 32
W = ((-128, 127, -3, 19, 0), (7, -11, 0, 1, 23), (2, 5, -17, 31, -9))
MODES = ("prepared", "zero_tape", "zero_local", "direct_weight")


def natural(x, minimum=0):
    if type(x) is not int or x < minimum:
        raise ValueError("invalid integer")
    return x


def validate_w(w):
    if not w or not w[0] or any(len(row) != len(w[0]) for row in w):
        raise ValueError("invalid matrix shape")
    if any(type(v) is not int or not -128 <= v <= 127 for row in w for v in row):
        raise ValueError("expected signed-i8 W")


def mv(w, x):
    validate_w(w)
    if len(x) != len(w[0]) or any(type(v) is not int for v in x):
        raise ValueError("invalid vector")
    return tuple(sum(a*b for a, b in zip(row, x, strict=True)) % MOD for row in w)


def combine(a, b, sign=1):
    return tuple((x + sign*y) % MOD for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class Binding:
    weight_sha256: str
    shape: tuple[int, int]
    stage: str
    epoch: str
    capacity: int
    ring_bits: int = 32
    layout: str = "row-major-signed-i8/full-ring32/little-endian"
    roles: tuple[str, ...] = ("Client", "Preparation-offline", "Inference")

    def digest(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()


def binding(w, capacity, epoch="fixture-epoch-0"):
    validate_w(w)
    natural(capacity, 1)
    if not epoch:
        raise ValueError("empty epoch")
    return Binding(hashlib.sha256(bytes(v % 256 for row in w for v in row)).hexdigest(),
                   (len(w), len(w[0])), "arbitrary-fixed-W", epoch, capacity)


def expand(seed, context, index, label, count):
    """Domain-separated SHAKE fixture; conditional on private-root PRG security."""
    if len(seed) != 32 or not 0 <= natural(index) < context.capacity:
        raise ValueError("invalid expansion key/index")
    natural(count)
    domain = json.dumps([context.digest(), index, label], separators=(",", ":")).encode()
    raw = hashlib.shake_256(seed + domain).digest(4*count)
    return tuple(int.from_bytes(raw[i:i+4], "little") for i in range(0, len(raw), 4))


@dataclass(frozen=True)
class ClientKey:
    context: Binding
    mode: str
    r_root: bytes
    pad_root: bytes | None
    output_tape: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[int, ...], ...]

    def row(self, index):
        m, n = self.context.shape
        r = expand(self.r_root, self.context, index, "r", n)
        if self.mode == "prepared":
            s = expand(self.pad_root, self.context, index, "s", m)
        elif self.mode == "zero_tape":
            s = self.output_tape[index]
        else:
            q = (expand(self.pad_root, self.context, index, "q", m)
                 if self.mode == "direct_weight" else (0,)*m)
            s = combine(mv(self.weights, r), q, -1)
        return r, s


@dataclass(frozen=True)
class InferenceKey:
    context: Binding
    mode: str
    q_root: bytes | None
    correction_tape: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[int, ...], ...]

    def row(self, index):
        if not 0 <= natural(index) < self.context.capacity:
            raise ValueError("column exhausted")
        if self.mode == "prepared":
            return self.correction_tape[index]
        if self.mode == "direct_weight":
            return expand(self.q_root, self.context, index, "q", self.context.shape[0])
        return (0,)*self.context.shape[0]

    def evaluate(self, index, opening):
        return combine(mv(self.weights, opening), self.row(index))


def setup(w, capacity, mode, rng, epoch="fixture-epoch-0"):
    """Preparation's offline view. Only returned role key enters each online role.

    Deterministic RNG is for replayable public tests, never secret deployment.
    No candidate claims ideal independent-s output for zero-correction modes.
    """
    if mode not in MODES:
        raise ValueError("unknown construction")
    w = tuple(tuple(row) for row in w)
    ctx = binding(w, capacity, epoch)
    r_root, pad_root = rng.randbytes(32), rng.randbytes(32)
    tape = []
    if mode in ("prepared", "zero_tape"):
        for j in range(capacity):
            wr = mv(w, expand(r_root, ctx, j, "r", len(w[0])))
            if mode == "prepared":
                s = expand(pad_root, ctx, j, "s", len(w))
                tape.append(combine(wr, s, -1))
            else:
                tape.append(wr)
    client = ClientKey(ctx, mode, r_root,
                       pad_root if mode in ("prepared", "direct_weight") else None,
                       tuple(tape) if mode == "zero_tape" else (),
                       w if mode in ("zero_local", "direct_weight") else ())
    inference = InferenceKey(ctx, mode, pad_root if mode == "direct_weight" else None,
                             tuple(tape) if mode == "prepared" else (), w)
    return client, inference


class Authority:
    """Ideal EXTERNAL monotonic authority; process restart keeps this owner alive.

    Not an implementation of anti-rollback storage or authenticated transport.
    Copies/snapshots of this authority are outside its assumption.
    """

    def __init__(self, context):
        self.context = context
        self.burned = set()

    def reserve(self, index, received_binding, outcome="ok"):
        if not 0 <= natural(index) < self.context.capacity or index in self.burned:
            raise ValueError("exhausted or replayed column")
        self.burned.add(index)  # Commit before exposure or any failure outcome.
        if received_binding != self.context.digest() or outcome != "ok":
            raise ValueError("binding/expiry/cancel/malformed failure; column burned")


def query(client, inference, authority, index, x):
    authority.reserve(index, client.context.digest())
    if client.context != inference.context:
        raise ValueError("role binding mismatch; column burned")
    r, s = client.row(index)
    opening = combine(tuple(x), r, -1)
    reply = inference.evaluate(index, opening)
    return combine(reply, s), {"binding": client.context.digest(), "index": index,
                              "client_to_inference": list(opening),
                              "inference_to_client": list(reply)}


def tv(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return sum((abs(Fraction(a[k], na)-Fraction(b[k], nb))
                for k in a.keys() | b.keys()), Fraction()) / 2


def leakage_controls():
    """Exhaustive SMALL rings only. Full inference view, not marginal pad tests."""
    def views(x, mode):
        out = Counter()
        for r in range(8):
            for s in (range(8) if mode == "ideal" else [(3*r) % 8]):
                q = (3*r-s) % 8
                out[(q, (x-r) % 8)] += 1
        return out
    ideal = tv(views(0, "ideal"), views(1, "ideal"))
    zero = tv(views(0, "zero"), views(1, "zero"))
    # W=2: r in ker(W) makes s=0 cheap but reveals W*x in W*z.
    kernel = [Counter((2*(x-r) % 8,) for r in (0, 4)) for x in (0, 1)]
    # r=2*a has full scalar rank as a rational map, but not surjective over Z_8.
    even = [Counter(((x-2*a) % 8,) for a in range(8)) for x in (0, 1)]
    # Public rank-one basis (1,3): left annihilator (3,-1) exposes 3*x0-x1.
    rank = [Counter(((x[0]-a) % 8, (x[1]-3*a) % 8) for a in range(8))
            for x in ((0, 0), (0, 1))]
    # Same hidden basis across many columns also leaks relations: two masks
    # r1=a, r2=3*a; joint openings reveal z2-3*z1=x2-3*x1.
    joint = [Counter(((-a) % 8, (x-3*a) % 8) for a in range(8)) for x in (0, 1)]
    fresh_joint = [Counter(((x[0]-a) % 8, (x[1]-b) % 8)
                           for a, b in itertools.product(range(8), repeat=2))
                   for x in ((0, 0), (1, 5))]
    return {"ideal_full_view_tv": str(ideal), "zero_correction_full_view_tv": str(zero),
            "kernel_mask_tv": str(tv(*kernel)), "even_multiplier_tv": str(tv(*even)),
            "public_low_rank_tv": str(tv(*rank)), "cross_column_relation_tv": str(tv(*joint)),
            "fresh_two_column_zero_correction_tv": str(tv(*fresh_joint)),
            "reused_r_reveals_delta": all(((5-r)-(2-r)) % 8 == 3 for r in range(8)),
            "q0_is_ideal_independent_s": False,
            "q0_ideal_distribution_distinguisher": "q is always zero, not uniform",
            "production_security_certified": False}


def linear_mask_gate(b):
    """Uniform coefficient vector -> uniform B*a over ring32 iff rank(B mod2)=n.

    Only a public linear-mask compiler screen; says nothing about secure PRGs.
    """
    if not b or not b[0] or any(len(row) != len(b[0]) for row in b):
        raise ValueError("invalid basis")
    if any(type(v) is not int for row in b for v in row):
        raise ValueError("invalid basis coefficients")
    rows = [sum((v & 1) << j for j, v in enumerate(row)) for row in b]
    pivots = {}
    for row in rows:
        while row:
            pivot = row.bit_length()-1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return {"rank_mod2": len(pivots), "input_dimension": len(b),
            "surjective_ring32": len(pivots) == len(b),
            "scope": "public linear masks with uniform ring coefficients only"}


def ledger(m, n, rows, capacity, mode, discarded=0, attempts=1):
    """Exact payload/operation counts for SPECIFIED ring32 stage protocol.

    Every epoch prepaid in full. Restart burns count in discarded, including
    reserved columns on failed attempts. attempts multiplies prepaid setup cohorts.
    Online directed totals count successful rows only; failed-attempt online
    traffic remains unknown, not zero, regardless of prepaid column retirement.
    No connection with historical serving/body envelopes is asserted.
    """
    for v in (m, n, rows, capacity, attempts):
        natural(v, 1)
    natural(discarded)
    if mode not in MODES:
        raise ValueError("unknown mode")
    epochs = ((rows+discarded+capacity-1)//capacity)*attempts
    issued = epochs*capacity
    local = mode in ("zero_local", "direct_weight")
    # Canonical binding: digest32 + epoch-id16 + shape/ring/capacity4*4 = 64.
    # One binding per Preparation->role. Row framing: index8 + binding-digest32.
    # W already at Preparation/Inference; new client delivery only when needed.
    links = {"preparation_to_client": epochs*(64+32+(32 if mode in
               ("prepared", "direct_weight") else 0)) + (m*n if local else 0),
             "preparation_to_inference": epochs*(64+(32 if mode == "direct_weight" else 0)),
             "client_to_inference": rows*(40+4*n),
             "inference_to_client": rows*(40+4*m)}
    if mode == "prepared":
        links["preparation_to_inference"] += issued*4*m
    elif mode == "zero_tape":
        links["preparation_to_client"] += issued*4*m
    client_prg_words = rows*(n+(m if mode in ("prepared", "direct_weight") else 0))
    return {"mode": mode, "shape": [m, n], "successful_rows": rows, "capacity": capacity,
            "epochs": epochs, "issued_columns": issued, "unused_or_burned_columns": issued-rows,
            "directed_bytes": links, "all_link_bytes": sum(links.values()),
            "directed_totals_exclude_failed_attempt_online_traffic": True,
            "failed_attempt_online_bytes": {"client_to_inference": None,
                                            "inference_to_client": None},
            "client_W_bytes": m*n if local else 0,
            "client_dense_MACs": rows*m*n if local else 0,
            "client_vector_additions": rows*(n+m+(m if mode == "direct_weight" else 0)),
            "client_PRG_output_words": client_prg_words,
            "client_PRG_output_bytes": 4*client_prg_words,
            "inference_PRG_output_bytes": rows*4*m if mode == "direct_weight" else 0,
            "preparation_dense_MACs": issued*m*n if mode in ("prepared", "zero_tape") else 0,
            "preparation_paged_W_read_bytes": issued*m*n if mode in ("prepared", "zero_tape") else 0,
            "preparation_PRG_output_bytes": issued*4*(n+(m if mode == "prepared" else 0))
            if mode in ("prepared", "zero_tape") else 0,
            "secret_tape_live_bytes": capacity*4*m if mode in ("prepared", "zero_tape") else 0,
            "secret_tape_owner": {"prepared": "Inference", "zero_tape": "Client"}.get(mode),
            "client_logical_streaming_workspace_bytes": 4*(2*n+3*m),
            "client_root_live_bytes": 64 if mode in ("prepared", "direct_weight") else 32,
            "client_external_tape_write_bytes": issued*4*m if mode == "zero_tape" else 0,
            "client_external_tape_read_bytes": rows*4*m if mode == "zero_tape" else 0,
            "client_paged_W_read_bytes": rows*m*n if local else 0,
            "inference_W_read_bytes": rows*m*n,
            "inference_tape_read_bytes": rows*4*m if mode == "prepared" else 0,
            "inference_tape_write_bytes": issued*4*m if mode == "prepared" else 0,
            "failed_post_expansion_client_work": None,
            "external_storage_authentication_bytes": None,
            "renewal_overlap_extra_bytes": None,
            "authentication_proof_bytes": None, "client_process_RSS_bytes": None,
            "timing_seconds": None,
            "scope": "prepaid setup plus successful-row payload/framing only; failed-attempt online traffic, authenticated transport, durable authority and runtime overhead unpriced"}


def lookup_ledger(m, n, group=4):
    """Public binary subset-sum tables: exact W*r, no dense MAC, not free work.

    Per group store all 2^g W-subset sums (ring32). For each of 32 bitplanes,
    fetch one m-vector per group and accumulate with bit shift. Full table
    distribution explicitly charged; no cache or selective-fetch privacy claim.
    """
    for v in (m, n, group):
        natural(v, 1)
    if group > 12:
        raise ValueError("bounded groups only")
    sizes = [min(group, n-i) for i in range(0, n, group)]
    entries = sum(1 << g for g in sizes)*m
    return {"group": group, "table_bytes": 4*entries,
            "setup_vector_additions": sum((1 << g)-1 for g in sizes)*m,
            "client_MACs_per_row": 0, "client_additions_per_row": 32*len(sizes)*m,
            "logical_table_read_bytes_per_row": 32*len(sizes)*m*4,
            "dense_W_MACs_per_row": m*n,
            "table_distribution_bytes": 4*entries,
            "admission": "not cheap merely because MAC count is zero"}


def lookup_tables(w, group=4):
    validate_w(w)
    if not 1 <= natural(group, 1) <= 4 or len(w)*len(w[0]) > 128:
        raise ValueError("bounded fixture only")
    tables = []
    for i in range(0, len(w[0]), group):
        g = min(group, len(w[0])-i)
        table = [(0,)*len(w)]
        for mask in range(1, 1 << g):
            low = mask & -mask
            b = low.bit_length()-1
            table.append(tuple((table[mask ^ low][j]+row[i+b]) % MOD
                               for j, row in enumerate(w)))
        tables.append(tuple(table))
    return tuple(tables)


def lookup_stage_ledger(m, n, rows, capacity, discarded=0, group=4):
    """Cost of precomputed public subset tables with zero correction transcript."""
    result = ledger(m, n, rows, capacity, "zero_local", discarded)
    tables = lookup_ledger(m, n, group)
    result["mode"] = "zero_lookup"
    result["directed_bytes"]["preparation_to_client"] += tables["table_bytes"]-m*n
    result["all_link_bytes"] = sum(result["directed_bytes"].values())
    result["client_W_bytes"] = 0
    result["client_dense_MACs"] = 0
    result["client_paged_W_read_bytes"] = 0
    result["client_vector_additions"] += rows*tables["client_additions_per_row"]
    result["public_subset_table_bytes"] = tables["table_bytes"]
    result["client_external_public_table_write_bytes"] = tables["table_bytes"]
    result["client_external_public_table_read_bytes"] = rows*tables["logical_table_read_bytes_per_row"]
    result["preparation_table_additions"] = tables["setup_vector_additions"]
    result["preparation_table_W_read_bytes"] = m*n
    result["client_subset_index_bit_extractions"] = rows*32*n
    result["client_subset_output_shifts"] = rows*tables["client_additions_per_row"]
    result["table_access_leakage"] = "client-local table I/O only; external observer obliviousness not provided"
    return result


def lookup_mv(w, x, group=4):
    validate_w(w)
    if len(x) != len(w[0]) or any(type(v) is not int for v in x):
        raise ValueError("invalid vector")
    tables = lookup_tables(w, group)
    out = [0]*len(w)
    for bit in range(32):
        for j, table in enumerate(tables):
            i = j*group
            mask = sum(((x[i+b] >> bit) & 1) << b
                       for b in range(min(group, len(x)-i)))
            for row, v in enumerate(table[mask]):
                out[row] = (out[row]+(v << bit)) % MOD
    return tuple(out)


def admission(candidate, baseline):
    """Necessary bounded gates, not sufficient whole-client/timing qualification."""
    return {"traffic_strictly_beats_baseline": candidate["all_link_bytes"] < baseline["all_link_bytes"],
            "no_added_dense_client_product": candidate["client_dense_MACs"] == 0,
            "client_ring_additions_no_greater_than_baseline":
                candidate["client_vector_additions"] <= baseline["client_vector_additions"],
            "not_a_client_output_tape": candidate["client_external_tape_write_bytes"] == 0,
            "secure_fixed_W_generator_implemented": False,
            "scale_admitted": False}


SOURCE_PAGES = {
    "2025/1223": {28: "A uniformly random", 29: "The computation of Expand", 30: "two t-sparse elements", 33: "would leak information", 35: "Regular   128 5 27", 36: "should never be", 42: "additional random inputs"},
    "2025/1637": {22: "1839 s", 23: "homomorphically evaluate C", 24: "relative noise bound"},
    "2026/2030": {14: "Assume DCR and the Generalized Riemann hypothesis", 15: "inner-product-membership weak PRF", 17: "OLE) over"},
    "2026/2298": {72: "leave this extension to future work", 75: "which we do not price here"},
}
TEXT_HASHES = {
    "2025/1223": "4a91404c17a157c0449f625c2c3532bcb722447ad23d03d9e58b53d2262b5821",
    "2025/1637": "921ff8447f56f84fd738b0d9c6f2bd9f2848bab333d11fe30f82ad224b55c0e1",
    "2026/2030": "aec7fd02840d18d286315d7a63c1b5f8a3a05e1fc212efe7fe84ad1480eba86e",
    "2026/2298": "c6642f525f8f96394b93dc5e65998790d37674dc36bb2597a29950fc92879b93",
}


def provenance(root=ROOT):
    manifest = json.loads((root/"docs/evidence/compact-setup-fulltext-review.json").read_text())
    result = []
    for paper in manifest["papers"]:
        if paper["id"] not in SOURCE_PAGES:
            continue
        pdf = (root/paper["path"]).read_bytes()
        if hashlib.sha256(pdf).hexdigest() != paper["sha256"]:
            raise ValueError("pinned PDF hash drift")
        textpath = "target/compact-setup-fulltext/"+paper["id"].replace("/", "-")+".txt"
        raw = (root/textpath).read_bytes()
        if hashlib.sha256(raw).hexdigest() != TEXT_HASHES[paper["id"]]:
            raise ValueError("pinned text hash drift")
        pages = raw.decode().split("\f")
        extracts = []
        for page, needle in SOURCE_PAGES[paper["id"]].items():
            text = pages[page-1]
            if needle not in text:
                raise ValueError(f"source excerpt drift: {paper['id']}:{page}")
            at = text.index(needle)
            extracts.append({"pdf_page": page, "page_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "excerpt": text[max(0, at-80):at+400]})
        result.append({"id": paper["id"], "url": paper["url"], "pdf": paper["path"],
                       "pdf_sha256": paper["sha256"], "pdf_bytes": len(pdf),
                       "text_path": textpath, "text_sha256": hashlib.sha256(raw).hexdigest(),
                       "reviewed_excerpts": extracts})
    return {"papers": result, "latest_PDF_verification": "blocked: HTTP 403 for all four PDF URLs on 2026-10-15",
            "available_revisions_only": True, "abstracts_used_for_construction_claims": False}


def report():
    rng = random.Random(610031)
    transcripts, outputs = {}, 0
    for mode in MODES:
        client, inference = setup(W, 7, mode, rng)
        authority = Authority(client.context)
        traces = []
        for j in range(7):
            x = tuple(rng.randrange(-MOD, MOD) for _ in W[0])
            got, trace = query(client, inference, authority, j, x)
            # Independent signed-bigint oracle, never mv or lookup expansion.
            oracle = tuple(sum(W[a][b]*x[b] for b in range(5)) & 0xffffffff for a in range(3))
            if got != oracle or lookup_mv(W, x) != oracle:
                raise AssertionError("independent exact-ring oracle failed")
            trace.update({"chosen_input": list(x), "reconstructed": list(got)})
            traces.append(trace)
            outputs += len(got)
        transcripts[mode] = {"public_fixture_only": True, "rows": traces,
                             "client_key_fields": list(client.__dict__),
                             "inference_key_fields": list(inference.__dict__)}
    costs = {}
    for label, shape, rows, cap, waste in (("small", (3, 5), 7, 4, 2),
                                          ("stage_analytic_only", (9728, 896), 288, 128, 1)):
        baseline = ledger(*shape, rows, cap, "prepared", waste)
        costs[label] = {mode: {**ledger(*shape, rows, cap, mode, waste),
                              "gate": admission(ledger(*shape, rows, cap, mode, waste), baseline)}
                        for mode in MODES}
        table = lookup_stage_ledger(*shape, rows, cap, waste)
        costs[label]["zero_lookup"] = {**table, "gate": admission(table, baseline)}
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
              ("scripts/probe_fixed_weight_correlation_gate.py", "tests/test_fixed_weight_correlation_gate.py",
               "docs/research/fixed-weight-correlation-gate.md")}
    return {"schema": "pllm.fixed_weight_correlation_gate.v1", "ring_bits": 32,
            "matrix": W, "checked_outputs": outputs, "transcripts": transcripts,
            "leakage": leakage_controls(), "ledgers": costs,
            "public_linear_mask_controls": {
                "even_full_rational_rank": linear_mask_gate(((2, 0), (0, 2))),
                "public_rank_one": linear_mask_gate(((1,), (3,))),
                "surjective_control": linear_mask_gate(((1, 2, 3), (0, 1, 4)))},
            "lookup_control": {"small": lookup_ledger(3, 5),
                               "stage_analytic_only": lookup_ledger(9728, 896)},
            "source_provenance": provenance(), "implementation_hashes": hashes,
            "conclusion": "No cheap-client construction admitted. Zero correction can hide inputs but does not remove masked-image production cost.",
            "production_crypto_implemented": False, "serving_saving_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "checked_outputs": result["checked_outputs"],
                      "scale_admitted": False}))


if __name__ == "__main__":
    main()
