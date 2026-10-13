"""Deterministic public-schedule screen. No SDK, tensor, provider or timing imports."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "docs/evidence/epoch-prefix-sdk-qwen25-2026-10-06.json"
MAX_CAP = 64
STAGES = 96
FULL_PREFILL_STAGES = 93
TERMINAL_STAGES = 3
SOURCES = (
    CONTROL,
    "docs/research/epoch-prefix-sdk-reference.md",
    "python/pllm/preparation.py",
    "python/pllm/runtime/client.py",
    "python/pllm/runtime/model_binding.py",
    "python/pllm/runtime/semantic_stages.py",
    "python/pllm/runtime/continuation_admission.py",
    "python/pllm/runtime/preparation_protocol.py",
    "python/pllm/runtime/preparation_server.py",
    "python/pllm/runtime/residue_codec.py",
    "crates/pllm-core/src/offset_transport.rs",
    "crates/pllm-compiler/src/decoder_runtime_schedule.rs",
    "crates/pllm-compiler/src/decoder_continuation.rs",
    "scripts/probe_adaptive_correction_epochs.py",
    "tests/test_adaptive_correction_epochs_probe.py",
    "docs/research/adaptive-correction-epochs-reference.md",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside public probe bound")
    return value


def schedule(cap, start, *, geometric=False):
    integer(cap, 1, MAX_CAP)
    integer(start, 1, MAX_CAP)
    result = []
    remaining = cap
    size = start
    while remaining:
        used = min(size, remaining)
        result.append(used)
        remaining -= used
        if geometric:
            size = min(64, size * 2)
    return tuple(result)


def validate(parts, cap):
    integer(cap, 1, MAX_CAP)
    if not parts or sum(parts) != cap:
        raise ValueError("schedule must cover exact cap")
    for size in parts:
        integer(size, 1, 64)
    # The final epoch may be clipped by remaining allowance.
    if any(b < a for a, b in zip(parts[:-2], parts[1:-1])):
        raise ValueError("non-final epochs must not shrink")


def payload_bytes(stage_rows, bits_per_row=None):
    """Exact public layout formula when supplied; absent widths remain unknown."""
    if bits_per_row is None:
        return None
    if set(stage_rows) != set(bits_per_row) or any(
            type(bits) is not int or bits <= 0 for bits in bits_per_row.values()):
        raise ValueError("exact public stage layout required")
    return sum((rows * bits_per_row[stage] + 7) // 8 for stage, rows in stage_rows.items())


def account(parts, *, prompt, output, cap):
    """EOS output=0..cap-1; cap stop output=cap. EOS itself is not emitted."""
    validate(parts, cap)
    integer(prompt, 1, 128)
    integer(output, 0, cap)
    selections = output + 1 if output < cap else cap
    covered = sessions = 0
    for size in parts:
        sessions += 1
        covered += size
        if covered >= selections:
            break
    prefill = FULL_PREFILL_STAGES * prompt + TERMINAL_STAGES
    issued = prefill + STAGES * (covered - 1)
    claimed = prefill + STAGES * (selections - 1)
    return {
        "output": output, "stop": "eos" if output < cap else "cap",
        "selections": selections, "covered_selections": covered,
        "decode_executions": selections - 1, "sessions": sessions,
        "pauses": sessions - 1, "issued_rows": issued, "reserved_rows": issued,
        "claimed_rows": claimed, "burned_rows": issued - claimed,
        "discarded_rows": 0, "preparation_requests": STAGES * sessions,
        "logical_correction_frames": STAGES * sessions,
        "correction_acknowledgements": STAGES * sessions,
        "session_admissions": sessions, "session_completions": sessions,
        "max_executed_history_bound": prompt + cap - 1,
        "correction_payload_bytes": None, "correction_envelope_bytes": None,
        "ack_bytes": None, "admission_control_bytes": None,
        "all_link_tcp_stream_bytes": None, "physical_wire_bytes": None,
        "latency_seconds": None, "client_peak_rss_bytes": None,
    }


def priors(cap):
    """Public synthetic counts over output stop positions, not learned prompts."""
    early = min(3, cap - 1)
    return {
        "early80_full20": {early: 80, cap: 20},
        "early50_full50": {early: 50, cap: 50},
        "uniform_eos_plus_full": dict.fromkeys(range(cap + 1), 1),
        "early20_full80": {early: 20, cap: 80},
    }


def expectation(parts, counts, *, cap, penalty, prompt=36):
    total = sum(counts.values())
    rows = [account(parts, prompt=prompt, output=n, cap=cap) for n in counts]
    fields = ("issued_rows", "burned_rows", "pauses", "logical_correction_frames")
    result = {field: sum(row[field] * counts[row["output"]] for row in rows) / total
              for field in fields}
    result["objective_row_equivalents"] = result["issued_rows"] + penalty * result["pauses"]
    result["full_cap_pauses"] = len(parts) - 1
    result["schedule"] = list(parts)
    return result


def optimize(cap, counts, penalty):
    """Exact monotone DP, integer count-weighted row objective; penalty is NOT time."""
    integer(cap, 1, MAX_CAP)
    integer(penalty, 0, 1_000_000)
    if not counts or any(type(n) is not int or not 0 <= n <= cap
                         or type(c) is not int or c <= 0 for n, c in counts.items()):
        raise ValueError("invalid public stop counts")
    required = [(n + 1 if n < cap else cap, count) for n, count in counts.items()]

    @lru_cache(None)
    def solve(boundary, previous):
        if boundary == cap:
            return 0, ()
        survival = sum(count for needed, count in required if needed > boundary)
        remaining = cap - boundary
        choices = range(previous, remaining + 1) if remaining >= previous else (remaining,)
        candidates = []
        for size in choices:
            tail_cost, tail = solve(boundary + size, size)
            cost = survival * (STAGES * size + (penalty if boundary else 0)) + tail_cost
            parts = (size, *tail)
            candidates.append((cost, len(parts), parts))
        best = min(candidates)
        return best[0], best[2]

    return solve(0, 1)[1]


class Lifecycle:
    """Tiny independent exclusion/burn control, not an installed runtime emulator."""

    def __init__(self):
        self.phase = "idle"
        self.issued = self.claimed = self.burned = self.discarded = 0
        self.pending_rows = 0
        self.session = 0

    def prepare(self, rows):
        if self.phase != "idle" or type(rows) is not int or rows <= 0:
            raise ValueError("Preparation requires idle and positive fresh rows")
        self.issued += rows
        self.pending_rows = rows
        self.phase = "prepared"

    def admit(self, *, acknowledged=True):
        if self.phase != "prepared":
            raise ValueError("fresh preparation required")
        self.session += 1
        self.phase = "online"
        if not acknowledged:
            self.retire(confirmed=True)
            raise ValueError("continuation acknowledgement mismatch")

    def claim(self, rows):
        if self.phase != "online" or not 0 <= rows <= self.pending_rows:
            raise ValueError("invalid claim")
        self.claimed += rows
        self.pending_rows -= rows

    def retire(self, *, confirmed=True):
        if self.phase not in {"online", "prepared"}:
            raise ValueError("no material to retire")
        if self.phase == "online":
            self.burned += self.pending_rows
        else:
            self.discarded += self.pending_rows
        self.pending_rows = 0
        self.phase = "idle" if confirmed else "closed"


def toy_oracle(parts, *, cap, eos_after=None, seed=11):
    """Integer recurrence + RNG control; compare session splitting, not SDK numerics."""
    validate(parts, cap)
    rng = random.Random(seed)
    state = 7
    output = []
    pending = None
    session = 1
    boundary = parts[0]
    for step in range(cap):
        token = (state + rng.randrange(97)) % 101
        if eos_after == step:
            break
        output.append(token)
        if step + 1 == cap:
            pending = token
            break
        if step + 1 == boundary:
            session += 1
            boundary += parts[session - 1]
        # A selected token crosses sessions without being sampled again.
        state = (state * 31 + token) % 65521
    return {"state": state, "output": output, "pending": pending, "rng": rng.getstate()}


def measured_controls():
    report = json.loads((ROOT / CONTROL).read_text())
    if report.get("complete") is not True:
        raise ValueError("retained cohort incomplete")
    result = []
    for cell in report["results"]:
        if cell["workload"] not in {"early", "full"}:
            continue
        response = cell["client"]["responses"][0]
        cap = 64 if cell["workload"] == "early" else 8
        parts = schedule(cap, 4 if cell["variant"] == "epochs" else cap)
        predicted = account(parts, prompt=response["input_tokens"],
                            output=response["output_tokens"], cap=cap)
        audit = cell["client"]["audit"]
        for name in ("issued", "reserved", "claimed", "burned"):
            if predicted[f"{name}_rows"] != audit[f"prepared_stage_rows_{name}"]:
                raise ValueError("Qwen source-derived row regression mismatch")
        result.append({"workload": cell["workload"], "variant": cell["variant"],
                       "replicate": cell["replicate"],
                       "issued_rows": predicted["issued_rows"],
                       "claimed_rows": predicted["claimed_rows"],
                       "burned_rows": predicted["burned_rows"],
                       "measured_correction_body_bytes": audit["correction_push_bytes"],
                       "measured_client_peak_rss_bytes": cell["client"]["client_peak_rss_bytes"]})
    return result


def build_report():
    comparisons = []
    for cap in (8, 16, 32, 64):
        for name, counts in priors(cap).items():
            for penalty in (0, 96, 384, 1536):
                fixed = [expectation(schedule(cap, e), counts, cap=cap, penalty=penalty)
                         for e in range(1, cap + 1)]
                best = min(fixed, key=lambda row: (row["objective_row_equivalents"],
                                                  row["full_cap_pauses"], row["schedule"]))
                candidates = {f"geometric_{start}": schedule(cap, start, geometric=True)
                              for start in (1, 2, 4, 8)}
                candidates["public_monotone_dp"] = optimize(cap, counts, penalty)
                comparisons.append({"cap": cap, "prior": name, "public_stop_counts": counts,
                                    "pause_penalty_row_equivalents": penalty,
                                    "best_fixed": best,
                                    "candidates": {key: expectation(parts, counts, cap=cap,
                                                                   penalty=penalty)
                                                   for key, parts in candidates.items()}})
    exhaustive = []
    for cap in range(1, MAX_CAP + 1):
        policies = {f"fixed_{e}": schedule(cap, e) for e in range(1, cap + 1)}
        policies.update({f"geometric_{e}": schedule(cap, e, geometric=True) for e in (1, 2, 4, 8)})
        for comparison in comparisons:
            if comparison["cap"] == cap:
                key = f"dp_{comparison['prior']}_{comparison['pause_penalty_row_equivalents']}"
                policies[key] = tuple(comparison["candidates"]["public_monotone_dp"]["schedule"])
        h = hashlib.sha256()
        cases = 0
        for name, parts in policies.items():
            for stop in range(cap + 1):
                row = account(parts, prompt=36, output=stop, cap=cap)
                assert row["reserved_rows"] == row["claimed_rows"] + row["burned_rows"]
                h.update(json.dumps([name, list(parts), row], sort_keys=True).encode())
                cases += 1
        exhaustive.append({"cap": cap, "cases": cases, "all_stop_ledger_sha256": h.hexdigest()})
    hashes = {path: digest((ROOT / path).read_bytes()) for path in SOURCES}
    return {
        "schema": "pllm.adaptive_correction_epochs.screen.v1", "complete": True,
        "verdict": "conditional row/frame frontier only; native/all-link/RSS target unproven",
        "scope": "stdlib synthetic ledger and oracle; no new measured bytes, CPU, RSS or latency",
        "source_hashes": hashes,
        "source_set_sha256": digest(json.dumps(hashes, sort_keys=True).encode()),
        "numeric_contract": "retained Qwen2.5 i8 weights/activations, prefix_f32, full KV; unchanged",
        "row_profile": {"decode_remote_stages": STAGES,
                        "prefill_full_row_stages": FULL_PREFILL_STAGES,
                        "prefill_terminal_row_stages": TERMINAL_STAGES,
                        "status": "source counting rule + aggregate retained cohort regression; not compiled parity"},
        "retained_measured_controls": measured_controls(),
        "comparisons": comparisons, "exhaustive_caps_and_stops": exhaustive,
        "exhaustive_cases": sum(row["cases"] for row in exhaustive),
        "bytes": {"adaptive_all_link_tcp_stream_bytes": None, "adaptive_ack_bytes": None,
                  "adaptive_correction_payload_bytes": None, "adaptive_lifecycle_control_bytes": None},
        "gates": {"native_schedule_sdk_parity": "not run; adaptive policy absent from SDK",
                  "remote_body_work_fraction": None, "required_remote_body_fraction": 0.90,
                  "whole_client_peak_rss_bytes": None, "maximum_client_rss_bytes": 256 << 20,
                  "full_cap_memory_admission_retained": True,
                  "matching_fresh_all_link_measurements": "required; no RTT/latency extrapolation"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    args.output.write_text(text)


if __name__ == "__main__":
    main()
