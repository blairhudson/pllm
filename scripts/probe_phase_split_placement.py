"""Bounded phase-ownership reference; not an implemented compiled executor.

Only public config, native semantic schedules and retained geometry are read.
Byte objective is explicit unframed arithmetic, NOT measured protocol traffic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "docs/evidence/placement-frontier-screen-2026-10-01.json"
PRIOR_SHA256 = "384fb471b90b33eae1591708adf0fefd4671c86a652dabd9606363b1495edf12"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Event:
    name: str
    prefill_rows: int
    decode_steps: int
    continuation: bool = False

    def __post_init__(self):
        if not self.name or any(type(n) is not int or n < 0 for n in (self.prefill_rows, self.decode_steps)):
            raise ValueError("invalid event")
        if type(self.continuation) is not bool:
            raise ValueError("invalid continuation")


@dataclass(frozen=True)
class Owners:
    """Explicit extra owners, never interpreted as measured whole-client RSS.

    Resident uses immutable native i8 snapshot plus configurable overlapping
    Python i8 owner and two scale owners. Paged uses shared file once, two scale
    owners, page indices, and explicitly supplied resident I/O/kernel windows.
    """
    storage: str = "resident"
    overlapping_i8_copies: int = 1
    paged_window_bytes: int | None = None
    base_peak_bytes: int | None = None
    other_peak_bytes: int | None = None
    common_cold_bytes: int | None = None

    def __post_init__(self):
        if self.storage not in {"resident", "paged"}:
            raise ValueError("unknown owner storage")
        for x in (self.overlapping_i8_copies, self.paged_window_bytes, self.base_peak_bytes,
                  self.other_peak_bytes, self.common_cold_bytes):
            if x is not None and (type(x) is not int or x < 0):
                raise ValueError("invalid owner charge")

    def charge(self, item):
        w, s = item["weight_bytes"], item["scale_bytes"]
        if self.storage == "resident":
            return (1 + self.overlapping_i8_copies) * w + 2 * s, 0
        if self.paged_window_bytes is None:
            raise ValueError("paged resident windows unknown; cannot select")
        indices = 256 + 128 * ((w + (1 << 20) - 1) // (1 << 20)) + 8 * max(item["in"], item["out"])
        return 2 * s + indices + self.paged_window_bytes, w


def phase_rows(item, event, phase):
    if phase == "decode":
        return event.decode_steps
    if event.prefill_rows and item["terminal"] and not event.continuation:
        return 1
    return event.prefill_rows


def vectors(items, events):
    # One cap per nonempty prefill event; decode cap is per executed row.
    totals = [sum(phase_rows(i, e, "prefill") * i["macs"] for i in items) for e in events]
    totals.append(sum(i["macs"] for i in items))
    return totals


def item_use(item, mask, events, owners):
    # mask bit 0: client prefill; bit 1: client decode. 0/1/2/3 all legal.
    used = [phase_rows(item, e, "prefill") * item["macs"] if mask & 1 else 0 for e in events]
    used.append(item["macs"] if mask & 2 else 0)
    memory, _ = owners.charge(item) if mask else (0, 0)
    return (*used, memory)


def item_value(item, mask, events):
    rows = sum((phase_rows(item, e, "prefill") if mask & 1 else 0)
               + (e.decode_steps if mask & 2 else 0) for e in events)
    # Cold raw i8/scales union delivery once, not once per selected phase.
    return rows * item["remote_row_bytes"] - (item["weight_bytes"] + item["scale_bytes"] if mask else 0)


def validate(items, events, candidates):
    if not items or not events or len(events) > 8 or len(items) > 512:
        raise ValueError("unbounded/empty problem")
    if len({i["id"] for i in items}) != len(items) or len({e.name for e in events}) != len(events):
        raise ValueError("duplicate identity")
    for i in items:
        if any(type(i[k]) is not int or i[k] <= 0 for k in
               ("macs", "weight_bytes", "scale_bytes", "remote_row_bytes", "in", "out")):
            raise ValueError("invalid stage cost")
        if type(i["correction_row_bytes"]) is not int or not 0 < i["correction_row_bytes"] < i["remote_row_bytes"]:
            raise ValueError("invalid correction cost")
    if len(candidates) > 12 or len(set(candidates)) != len(candidates) or not set(candidates) <= {i["id"] for i in items}:
        raise ValueError("invalid bounded candidate set")


def choose(items, events, candidates, owners, *, extra_memory_limit, max_states=100_000, fixed=False):
    """Exact sparse multiple-choice knapsack over declared candidate universe.

    Noncandidates stay remote but count in every MAC denominator. State includes
    EACH prefill event, per-row decode, union client resident-owner bytes.
    State exhaustion rejects; never silently beam-prunes or claims full optimum.
    """
    validate(items, events, candidates)
    if type(extra_memory_limit) is not int or extra_memory_limit < 0 or not 1 <= max_states <= 1_000_000:
        raise ValueError("invalid search bound")
    by_id = {i["id"]: i for i in items}
    caps = (*[n // 10 for n in vectors(items, events)], extra_memory_limit)
    states = {(0,) * len(caps): (0, ())}
    peak = 1
    for name in candidates:
        item = by_id[name]
        next_states = {}
        for used, (saving, path) in states.items():
            for mask in ((0, 3) if fixed else range(4)):
                addition = item_use(item, mask, events, owners)
                key = tuple(a + b for a, b in zip(used, addition, strict=True))
                if any(a > b for a, b in zip(key, caps, strict=True)):
                    continue
                candidate = (saving + item_value(item, mask, events), (*path, mask))
                old = next_states.get(key)
                if old is None or candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    next_states[key] = candidate
                if len(next_states) > max_states:
                    raise ValueError("exact planner state bound exceeded")
        states = next_states
        peak = max(peak, len(states))
    _, (saving, path) = min(states.items(), key=lambda pair: (-pair[1][0], pair[0][-1], pair[1][1]))
    selection = {name: mask for name, mask in zip(candidates, path, strict=True) if mask}
    return {"selection": selection, "net_arithmetic_byte_saving": saving,
            "peak_states": peak, "candidate_universe": candidates,
            "optimality": "exact only within declared bounded candidate universe",
            "fixed_phase_control": fixed}


def brute_force(items, events, candidates, owners, *, extra_memory_limit, fixed=False):
    """Independent tiny Cartesian oracle; intentionally does not call DP helpers."""
    validate(items, events, candidates)
    if len(candidates) > 7:
        raise ValueError("oracle limited to seven groups")
    best = None
    for path in itertools.product((0, 3) if fixed else range(4), repeat=len(candidates)):
        selected = dict(zip(candidates, path, strict=True))
        totals = [0] * (len(events) + 1)
        locals_ = [0] * len(totals)
        memory = saving = 0
        for item in items:
            mask = selected.get(item["id"], 0)
            for index, event in enumerate(events):
                rows = 1 if item["terminal"] and event.prefill_rows > 0 and not event.continuation else event.prefill_rows
                totals[index] += rows * item["macs"]
                if mask & 1:
                    locals_[index] += rows * item["macs"]
                    saving += rows * item["remote_row_bytes"]
                if mask & 2:
                    saving += event.decode_steps * item["remote_row_bytes"]
            totals[-1] += item["macs"]
            if mask & 2:
                locals_[-1] += item["macs"]
            if mask:
                saving -= item["weight_bytes"] + item["scale_bytes"]
                memory += owners.charge(item)[0]
        if memory > extra_memory_limit or any(a * 10 > b for a, b in zip(locals_, totals, strict=True)):
            continue
        key = (-saving, memory, path)
        if best is None or key < best[0]:
            best = (key, selected)
    return {"selection": {k: v for k, v in best[1].items() if v},
            "net_arithmetic_byte_saving": -best[0][0]}


def replay(items, events, selection, owners, *, extra_memory_limit):
    validate(items, events, list(selection))
    if any(type(m) is not int or m not in (1, 2, 3) for m in selection.values()):
        raise ValueError("invalid ownership mask")
    memory = disk = cold = native = scales = python_i8 = 0
    provider_union = provider_cold = 0
    event_rows = []
    for i in items:
        mask = selection.get(i["id"], 0)
        if mask:
            m, d = owners.charge(i)
            memory += m
            disk += d
            cold += i["weight_bytes"] + i["scale_bytes"]
            native += i["weight_bytes"] if owners.storage == "resident" else 0
            python_i8 += owners.overlapping_i8_copies * i["weight_bytes"] if owners.storage == "resident" else 0
            scales += 2 * i["scale_bytes"]
        if mask != 3:
            provider_union += i["weight_bytes"] + i["scale_bytes"]
    provider_cold = 2 * provider_union  # two distinct provider recipients, uncompressed payload floor
    totals = vectors(items, events)
    local_prefill = [0] * len(events)
    local_decode = sum(i["macs"] for i in items if selection.get(i["id"], 0) & 2)
    for index, e in enumerate(events):
        phases = {}
        for phase, bit in (("prefill", 1), ("decode", 2)):
            remote_rows = local_rows = correction_bytes = online_bytes = 0
            identities = []
            for i in items:
                rows = phase_rows(i, e, phase)
                if selection.get(i["id"], 0) & bit:
                    local_rows += rows
                    if phase == "prefill":
                        local_prefill[index] += rows * i["macs"]
                else:
                    remote_rows += rows
                    correction_bytes += rows * i["correction_row_bytes"]
                    online_bytes += rows * (i["remote_row_bytes"] - i["correction_row_bytes"])
                    # Count only; no material creation or reuse. Every executed row
                    # receives phase/event-qualified fresh one-use correction.
                    if rows:
                        identities.append({"namespace": [e.name, phase, i["id"]], "fresh_rows": rows})
            phases[phase] = {"remote_stage_rows": remote_rows, "client_stage_rows": local_rows,
                             "fresh_correction_rows": remote_rows, "correction_payload_bytes": correction_bytes,
                             "online_payload_bytes": online_bytes, "fresh_namespaces": identities,
                             "client_cpu_seconds": None, "preparation_cpu_seconds": None,
                             "latency_seconds": None}
        event_rows.append({"event": e.name, "continuation": e.continuation, "phases": phases,
                           "ownership_switches": [i["id"] for i in items if selection.get(i["id"], 0) in (1, 2)],
                           "transition_cpu_seconds": None, "transition_bytes": None})
    if memory > extra_memory_limit or any(a * 10 > b for a, b in zip((*local_prefill, local_decode), totals, strict=True)):
        raise ValueError("replay violates per-phase remote MAC or union-owner cap")
    execution = sum(p["correction_payload_bytes"] + p["online_payload_bytes"]
                    for e in event_rows for p in e["phases"].values())
    peak_floor = None if owners.base_peak_bytes is None or owners.other_peak_bytes is None else owners.base_peak_bytes + owners.other_peak_bytes + memory
    return {"scope": "unframed arithmetic/owner cost replay, not measured response",
            "compiled_phase_split_executor_implemented": False,
            "union_client_groups": sorted(selection), "union_client_resident_owner_bytes": memory,
            "client_native_i8_snapshot_bytes": native, "client_overlapping_python_i8_bytes": python_i8,
            "client_scale_owner_bytes": scales, "client_paged_shared_file_bytes": disk,
            "client_cold_raw_weight_scale_payload_bytes": cold,
            "provider_union_weight_scale_payload_bytes_per_provider": provider_union,
            "two_provider_cold_raw_payload_floor_bytes": provider_cold,
            "provider_checkpoint_distribution_bytes": None,
            "admission_burn_only_rows": None, "control_seed_manifest_framing_bytes": None,
            "client_base_cpu_seconds": None, "preparation_base_cpu_seconds": None,
            "client_prefill_decode_buffer_peak_bytes": None,
            "whole_client_peak_owner_floor_bytes": peak_floor,
            "whole_client_peak_rss_bytes": None, "tier_qualified": False,
            "common_cold_bytes": owners.common_cold_bytes,
            "execution_arithmetic_payload_bytes": execution,
            "priced_payload_bytes": execution + cold,
            "complete_all_link_body_bytes": None, "complete_all_link_wire_bytes": None,
            "phase_mac_admission": [{"phase": e.name + "/prefill", "client_macs": a, "body_macs": b}
                                    for e, a, b in zip(events, local_prefill, totals[:-1], strict=True)]
                                   + [{"phase": "decode/per-row", "client_macs": local_decode, "body_macs": totals[-1]}],
            "events": event_rows, "numeric_identity": digest([i["identity"] for i in items])}


def load_public_problem(*, input_rows=150, decode_steps=7):
    """Native header-only lowering; no model resolution, weights or checkpoint."""
    from pllm import Model, lower_model
    from pllm.kernels import Cpu
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols.masked_linear import MaskedLinear
    from pllm.runtime.semantic_stages import scheduled_stage_specs, semantic_stage_role

    raw = PRIOR.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_SHA256:
        raise ValueError("retained source hash mismatch")
    retained = json.loads(raw)["retained_inputs"]
    pipeline = MaskedLinearCpu(Model("public/geometry-only"), kernels=Cpu(threads=1))
    pipeline = pipeline.with_params(linear=MaskedLinear(prefill_pruning="terminal"))
    plan = lower_model(retained["config"], batch=1, max_input_tokens=input_rows, max_new_tokens=decode_steps + 1)
    schedule = plan.runtime_schedule(pipeline)
    if not schedule.complete or schedule.protected_execution:
        raise ValueError("unsupported public schedule")
    specs = {s.id: s for s in scheduled_stage_specs(plan, pipeline) if s.layer_index is not None}
    all_weights = [key for s in specs.values() for key in s.weight_keys]
    if len(set(all_weights)) != len(all_weights):
        raise ValueError("aliased body weights need union-owner normalization")
    prior = {s["id"]: s for s in retained["stages"]}
    if specs.keys() != prior.keys():
        raise ValueError("source geometry/native grouped-stage set differs")
    by_phase = {}
    for phase in ("prefill", "decode"):
        ops = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        by_phase[phase] = {}
        for step in schedule.to_dict()[phase]["steps"]:
            if step["layer"] is None or not step["weight_ids"]:
                continue
            if step["executor"] != "remote_stage":
                raise ValueError("baseline body executor unsupported")
            role = semantic_stage_role(step, ops)
            key = (step["layer"], role)
            if key in by_phase[phase]:
                raise ValueError("ambiguous semantic group")
            by_phase[phase][key] = step
    items = []
    for name, s in specs.items():
        p = prior[name]
        if (s.in_features, s.out_features, s.layer_index, s.role) != (p["in_features"], p["out_features"], p["layer"], p["role"]):
            raise ValueError("source-locked geometry mismatch")
        steps = [by_phase[phase][s.layer_index, s.role] for phase in ("prefill", "decode")]
        contract = lambda step: (step["weight_ids"], step["operation_ids"], step["input_ids"],
                                 [(o["stage_offset"], o["stage_width"]) for o in step["outputs"]])
        if contract(steps[0]) != contract(steps[1]) or tuple(steps[0]["weight_ids"]) != s.weight_keys:
            raise ValueError("phase numeric group contract differs")
        # Bias bytes would require independent binding; retained scale/weight-only
        # ledger lacks them. QKV public local bias is unchanged, not newly moved.
        output_bytes = (p["wire_bits"] + 7) // 8 * s.out_features
        items.append({"id": name, "in": s.in_features, "out": s.out_features,
                      "macs": s.in_features * s.out_features,
                      "weight_bytes": p["i8_weight_bytes"], "scale_bytes": p["f32_scale_bytes"],
                      "terminal": steps[0].get("terminal_row_only", False),
                      "correction_row_bytes": output_bytes,
                      "remote_row_bytes": 4 * s.in_features + 2 * output_bytes,
                      "identity": {"source": retained["source"], "weight_ids": s.weight_keys,
                                   "plan_digest": plan.digest, "schedule_digest": schedule.digest,
                                   "composition_digest": digest(pipeline.to_spec()),
                                   "ordered_group_contract": contract(steps[0]),
                                   "weight_sha256": p["weight_sha256"], "scales_sha256": p["scales_sha256"],
                                   "signed_output_bound": p["signed_output_bound"], "wire_bits": p["wire_bits"],
                                   "numeric": "existing W8A8 integer reduction, existing local scales/bias/nonlinear"}})
    archived = retained["archives"]["cold"]["prepared-request-sized"]
    if sum(e["serialized_body_bytes"] for e in archived["edges"]) != archived["total_body_bytes"]:
        raise ValueError("retained measured directed ledger does not conserve")
    context = {"plan_digest": plan.digest, "schedule_digest": schedule.digest,
               "composition_digest": digest(pipeline.to_spec()), "source": retained["source"],
               "retained_measured_control": {"workload": "39 input/32 output, historical dense schedule",
                                             "body_bytes": archived["total_body_bytes"],
                                             "online_body_bytes": archived["online_body_bytes"],
                                             "scope": "context only; never prorated into phase measurements"},
               "schedule": schedule.to_dict()}
    from pllm.runtime import semantic_stages
    from pllm.runtime._native_support import extension
    import pllm.profiles
    native = extension()
    if native is None or not getattr(native, "__file__", None):
        raise ValueError("installed native schedule implementation unavailable")
    installed = [Path(semantic_stages.__file__), Path(pllm.profiles.__file__), Path(native.__file__)]
    context["installed_implementation_hashes"] = {}
    for path in installed:
        with path.open("rb") as handle:
            context["installed_implementation_hashes"][str(path)] = hashlib.file_digest(handle, "sha256").hexdigest()
    context["trust_identity"] = {"numeric": "unchanged W8A8", "protected_execution": False,
                                 "providers": "existing prepared masked-linear public trust boundary",
                                 "new_client_weights_require_explicit_permission": True,
                                 "corrections": "fresh event/phase/stage/row; never reused across ownership switches"}
    context["arithmetic_pricing_assumptions"] = {"masked_input_bytes_per_coordinate": 4,
        "output_and_correction_bytes_per_coordinate": "ceil(retained wire_bits / 8)",
        "scope": "unframed arithmetic only, not actual packed/residue protocol bodies",
        "provider_cold_floor_in_selection_objective": False,
        "provider_distribution_control": "full existing checkpoints held fixed; no claimed checkpoint saving"}
    return items, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.groups <= 10:
        parser.error("groups must be in [1,10]")
    items, context = load_public_problem()
    # Public bounded universe: smallest groups, with explicit pruned last-layer
    # attention/MLP candidates. Entire 96-group body remains denominator.
    special = sorted([i for i in items if i["terminal"]], key=lambda i: (i["macs"], i["id"]))
    ordered = special + sorted([i for i in items if not i["terminal"]], key=lambda i: (i["macs"], i["id"]))
    candidates = [i["id"] for i in ordered[:args.groups]]
    owners = Owners()
    scenarios = {"initial": [Event("cold", 150, 7)],
                 "cold-repeat": [Event("cold", 150, 7), Event("exact-repeat-hit", 0, 7)],
                 "continuation": [Event("cold", 150, 7), Event("new-suffix", 20, 7, True)]}
    results = {}
    for name, events in scenarios.items():
        entries = {}
        for fixed in (True, False):
            selected = choose(items, events, candidates, owners, extra_memory_limit=64 << 20, fixed=fixed)
            if len(candidates) <= 7:
                oracle = brute_force(items, events, candidates, owners, extra_memory_limit=64 << 20, fixed=fixed)
                if any(selected[k] != oracle[k] for k in oracle):
                    raise AssertionError("DP/oracle disagreement")
            entries["fixed" if fixed else "split"] = {"planner": selected,
                "replay": replay(items, events, selected["selection"], owners, extra_memory_limit=64 << 20)}
        results[name] = entries
    paths = [Path(__file__), ROOT / "tests/test_phase_split_placement_probe.py", PRIOR,
             ROOT / "python/pllm/runtime/semantic_stages.py", ROOT / "python/pllm/runtime/transformer_engine.py",
             ROOT / "python/pllm/runtime/benchmark_memory.py",
             ROOT / "crates/pllm-compiler/src/decoder_runtime_schedule.rs"]
    evidence = {"schema": "phase_split_placement.synthetic.v1", "scope": "public schedule and bounded cost reference only",
                "complete_response_measured": False, "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "context": context, "results": results,
                "missing_gates": ["phase-aware compiler/runtime/provider inventory", "exact native logits/KV parity",
                                  "phase CPU/latency", "whole-client RSS fresh replicates", "all-link distribution/framing/transitions"]}
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({name: {kind: {"saving": entry["planner"]["net_arithmetic_byte_saving"],
                                  "selection": entry["planner"]["selection"],
                                  "union_owner_bytes": entry["replay"]["union_client_resident_owner_bytes"]}
                             for kind, entry in entries.items()} for name, entries in results.items()}, indent=2))
    return evidence


if __name__ == "__main__":
    main()
