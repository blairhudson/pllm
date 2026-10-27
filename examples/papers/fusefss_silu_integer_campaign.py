"""Astra F19: finite public integer SiLU model, not protected execution.

Run: python3 -I -B examples/papers/fusefss_silu_integer_campaign.py
Reads only frozen public constants; no table regeneration or project imports.
The independent equation oracle and source-derived bit pipeline compare every
component (Y,D,A,B,J,Z), not only outputs. Public range checks here are a model,
not secret membership proofs, native lifecycle checks, or FSS construction.
"""

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
from time import monotonic


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "paper/reproduction-contracts/fusefss/gate-set/fixtures.json"
VALUES = ROOT / "crates/pllm-garble/src/fixed_activation_table_values.rs"
FIXTURE_SHA256 = "ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87"
TABLE_SHA256 = "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f"
RING = 1 << 50
LIMIT = 1 << 37
ENDPOINT_Y = (-(1 << 31), -(1 << 31) + 1, (1 << 31) - 2, (1 << 31) - 1)
INVALID_X = (-(1 << 49), -LIMIT - 1, LIMIT, (1 << 49) - 1)
CORNERS = (-65, -64, -1, 0, 63, 64, -65473, -65472, 65535, 65536)
PUBLIC_MODEL_SCHEDULE = (
    "signed50_decode", "range_check", "safe_substitute", "sticky_and",
    "narrow38", "tr6_to32", "sign_extend50", "negative_compare",
    "negate", "abs_select", "unsigned_lt1024", "clamp_select",
    "lookup", "original_input_select", "subtract50", "stage_output",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_tables(fixture_path=FIXTURE, values_path=VALUES):
    """Fail closed on missing/mutated fixture, hash, or any Rust table entry.

    Provenance: fixed_activation_tables.rs's independent frozen-entry/hash test
    pins this file and u64LE digest. gate-set/reference.py:certify_tables uses
    rational exponential bounds to settle floors; it is NOT run/imported here.
    """
    raw = fixture_path.read_bytes()
    require(sha256(raw).hexdigest() == FIXTURE_SHA256, "frozen fixture hash mismatch")
    fixture = json.loads(raw)
    frozen = tuple(fixture["tables"]["silu"])
    require(len(frozen) == 1024, "expected all 1024 frozen entries")
    require(all(type(v) is int and 0 <= v < RING for v in frozen), "invalid entry")
    require(fixture["sha256_u64le"]["silu"] == TABLE_SHA256, "fixture table hash mismatch")
    digest = sha256()
    for value in frozen:
        digest.update(value.to_bytes(8, "little"))
    require(digest.hexdigest() == TABLE_SHA256, "u64LE table hash mismatch")
    # Read inert public Rust constants, never bindings, material or source weights.
    match = re.search(
        r"static SILU: \[u64; 1024\] = \[([\d\s,]+)\];", values_path.read_text()
    )
    require(match is not None, "missing frozen Rust SiLU array")
    source = tuple(int(v.strip()) for v in match[1].split(",") if v.strip())
    require(len(source) == 1024, "expected all 1024 Rust entries")
    for index, (actual, expected) in enumerate(zip(source, frozen)):
        require(actual == expected, f"Rust/frozen table mismatch at {index}")
    return frozen, source


def oracle(x, table):
    """Independent integer equations: floor BEFORE abs; original X branch."""
    require(type(x) is int and -LIMIT <= x < LIMIT, "oracle outside signed38")
    y = x // 64
    d = int(y >= 0)
    a = abs(y)
    b = int(a < 1024)
    j = min(a, 1023)
    return y, d, a, b, j, (d * x - table[j]) % RING


def signed(word, bits):
    return word - (1 << bits) if word & (1 << (bits - 1)) else word


def pipeline(word, table, trace=None):
    """Public Python model of gates.rs:silu_inner, NOT native execution.

    Caller must check actual signed50 membership before this unsafe narrowing.
    The reduced TR word is sign-extended before negation, including -2**31.
    """
    if trace is None:
        trace = []
    x38 = word & ((1 << 38) - 1)
    trace.append("narrow38")
    y32 = x38 >> 6
    trace.append("tr6_to32")
    y = signed(y32, 32)
    trace.append("sign_extend50")
    negative = int(y < 0)
    trace.append("negative_compare")
    negated = (-y) & (RING - 1)
    trace.append("negate")
    absolute = negated if negative else y
    trace.append("abs_select")
    in_table = int(absolute < 1024)
    trace.append("unsigned_lt1024")
    index = absolute if in_table else 1023
    trace.append("clamp_select")
    correction = table[index]
    trace.append("lookup")
    selected = 0 if negative else word
    trace.append("original_input_select")
    result = (selected - correction) & (RING - 1)
    trace.append("subtract50")
    return y, 1 - negative, absolute, in_table, index, result


def checked_lane(word, table, sticky=True):
    """Public control model: membership, zero substitution, then narrowing."""
    require(type(word) is int and 0 <= word < RING, "noncanonical ring50 word")
    trace = []
    x = signed(word, 50)
    trace.append("signed50_decode")
    valid = -LIMIT <= x < LIMIT
    trace.append("range_check")
    safe = word if valid else 0
    trace.append("safe_substitute")
    sticky = bool(sticky & valid)
    trace.append("sticky_and")
    result = pipeline(safe, table, trace)
    trace.append("stage_output")
    return valid, safe, sticky, result, tuple(trace)


def request_model(words, table):
    """Length-three public schedule; only final aggregate client decision.

    Internal values and trace exist for arithmetic testing only. No holder
    transcript, secret-safe implementation, timing claim or lifecycle simulation.
    Rejected requests disclose no staged outputs in this client-interface model.
    """
    require(len(words) == 3, "fixed public request length is three")
    sticky, staged, trace = True, [], []
    for word in words:
        _, _, sticky, result, schedule = checked_lane(word, table, sticky)
        staged.append(result[-1])
        trace.extend(schedule)
    trace.append("client_aggregate_decision")
    return sticky, tuple(staged) if sticky else None, tuple(trace)


def valid_cases():
    """Stream fixed disjoint ranges; memory independent of case count."""
    yield from range(-65536, 65600)
    for y in ENDPOINT_Y:
        for remainder in range(64):
            yield 64 * y + remainder


def run_campaign(timeout_seconds=30):
    started = monotonic()
    require(0 < timeout_seconds <= 300, "timeout must be in (0,300]")

    def deadline():
        require(monotonic() - started < timeout_seconds, "campaign deadline exceeded")

    frozen, source = load_tables()
    # Independent interval arithmetic establishes cardinality and no overlap;
    # streamed order and per-group counts below check the generator itself.
    intervals = ((-65536, 65599),) + tuple((64*y, 64*y+63) for y in ENDPOINT_Y)
    for i, (lo, hi) in enumerate(intervals):
        require(-LIMIT <= lo <= hi < LIMIT, "invalid campaign interval")
        for other_lo, other_hi in intervals[:i]:
            require(hi < other_lo or other_hi < lo, "campaign ranges overlap")
    require(sum(hi-lo+1 for lo, hi in intervals) == 131392, "interval cardinality")
    counts, previous = [0] * 5, [None] * 5
    for ordinal, x in enumerate(valid_cases()):
        if ordinal % 1024 == 0:
            deadline()
        group = 0 if ordinal < 131136 else 1 + (ordinal - 131136) // 64
        require(group < 5, "extra case")
        lo, hi = intervals[group]
        require(lo <= x <= hi, "case outside expected group")
        require(x == (lo if previous[group] is None else previous[group] + 1),
                "duplicate or missing case")
        previous[group] = x
        counts[group] += 1
        expected = oracle(x, frozen)
        actual = pipeline(x % RING, source)
        require(actual == expected, f"tuple mismatch X={x}: {actual} != {expected}")
    require(counts == [131136, 64, 64, 64, 64], "streamed cardinality mismatch")
    corner_results = {}
    for x in CORNERS:
        actual = pipeline(x % RING, source)
        require(actual == oracle(x, frozen), f"corner mismatch {x}")
        corner_results[str(x)] = actual
    for x in INVALID_X:
        valid, safe, sticky, actual, _ = checked_lane(x % RING, source)
        require(not valid and safe == 0 and not sticky, "invalid not substituted")
        require(actual == oracle(0, frozen), "unsafe invalid narrowing")
    # All eight validity patterns, repeated with each invalid sentinel in every
    # failing position. False must remain sticky even after later valid lanes.
    expected_trace = PUBLIC_MODEL_SCHEDULE * 3 + ("client_aggregate_decision",)
    requests = accepted = rejected = 0
    for pattern in product((False, True), repeat=3):
        for invalid in INVALID_X:
            deadline()
            xs = tuple(valid_x if ok else invalid
                       for ok, valid_x in zip(pattern, (-65, 64, 65536)))
            decision, outputs, trace = request_model(tuple(x % RING for x in xs), source)
            require(trace == expected_trace, "public model schedule mismatch")
            require(decision == all(pattern), "aggregate acceptance mismatch")
            expected_outputs = tuple(oracle(x, frozen)[-1] for x in xs) if all(pattern) else None
            require(outputs == expected_outputs, "client aggregate disclosure mismatch")
            sticky = True
            for x, expected_valid in zip(xs, pattern):
                valid, _, next_sticky, _, _ = checked_lane(x % RING, source, sticky)
                require(valid == expected_valid and next_sticky == (sticky and expected_valid),
                        "sticky validity mismatch")
                sticky = next_sticky
            requests += 1
            accepted += int(decision)
            rejected += int(not decision)
    deadline()
    return {
        "experiment": "astra-f19-fusefss-silu-integer-campaign",
        "status": "passed", "failures": 0,
        "scope": "finite public integer arithmetic and control models only",
        "valid_cases": sum(counts), "central_cases": counts[0],
        "endpoint_cases": sum(counts[1:]), "explicit_corners": corner_results,
        "disjoint_ranges_checked": True, "tuple_fields": ["Y", "D", "A", "B", "J", "Z"],
        "invalid_sentinels": list(INVALID_X), "validity_patterns": 8,
        "request_models": requests, "accepted_models": accepted, "rejected_models": rejected,
        "identical_public_model_schedule_steps": len(expected_trace),
        "table_entries": len(frozen), "fixture_sha256": FIXTURE_SHA256,
        "table_sha256_u64le": TABLE_SHA256,
        "construction": "UNMET", "simulator": "UNMET",
        "not_tested": ["secret membership proof", "FSS material or keys", "protected/native kernels",
                       "native lifecycle, burn, cancellation or replay", "HE", "SDK", "trained execution"],
        "unchanged_limits": {"tape_bytes_per_holder": 32 << 30, "native_resident_bytes": 64 << 20,
                             "conditional_bytes_per_gate": "1229.936", "full_ledger": "required"},
        "elapsed_seconds": monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/fusefss-silu-integer-campaign.json")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    report = run_campaign(args.timeout_seconds)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
