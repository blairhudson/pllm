"""Exact-payload selective correlation cost gate; no generator or serving execution.

Run with the existing frontier SDK environment. Only header-only model lowering
is used: no checkpoint resolution, loading, network, native crypto or benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = {
    "research-sota-combinations-qwen25.json": "5fb73b6883a0e42ac42b65b33a84cdfe0c6646d1e459f0df72639c38756ad54d",
    "research-sota-combinations-qwen25.plan.json": "3f8902f7f621adfd60dc301907ef144fbacc5f35c29c11bf02c689aaaf525204",
    "research-all-link-frontier.json": "401c5a8c12f90b8df76949a8d320411988b77afeb58e9d9c14e43aed3551190c",
    "research-bootstrap-ingress.json": "c842d8d2f78a042db5432078d8065895d24172f6e64992c476ab3ca39db99b09",
    "research-narrow-ingress.json": "8c55c563422a38ddc0c6d0187379c38450b0702b4031912166e7f4bd4c7e105d",
    "research-attested-linear-bootstrap.json": "f8ea98839aaf19ec698cc381b35fc92a7fe54e48477fdcf4929ea99fff66ad78",
    "network-algebra-qwen25-2026-10-06.json": "8c276df98557694ae5c54ee97d7d42e78b3fa146a5bf272b3f1826fb0bb23e3e",
    "placement-frontier-screen-2026-10-01.json": "384fb471b90b33eae1591708adf0fefd4671c86a652dabd9606363b1495edf12",
    "compact-setup-fulltext-review.json": "4797390ba0ed1d42eebeccb5add20d829b3dc4938a05301dc57b6e99c555f38c",
}
CASES = ("150/both", "150/both-placement", "288/both", "288/both-placement")
SUBSETS = ("whole_group", "gate_up_only", "prefill_only", "decode_only", "gate_up_prefill", "gate_up_bulk_prefill")
TARGETS = (Fraction(1, 5), Fraction(3, 10), Fraction(7, 20))
HORIZONS = (1, 2, 4, 16, 64, 4096)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def natural(value):
    require(type(value) is int and value >= 0, "nonnegative integer required")
    return value


def unique_object(pairs):
    value = dict(pairs)
    require(len(value) == len(pairs), "duplicate JSON key")
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(root=ROOT):
    sources = {}
    for filename, expected in LOCKS.items():
        path = root / "docs/evidence" / filename
        require(sha(path) == expected, f"source hash drift: {filename}")
        sources[filename] = json.loads(path.read_text(), object_pairs_hook=unique_object)
    return sources


def index(stages):
    result = {s["stage"]: s for s in stages}
    require(len(result) == len(stages), "duplicate stage identity")
    return result


def packed_bits(bits):
    return (natural(bits) + 7) // 8


def validate_geometry(sources):
    """Join independent frozen geometry, widths, demand and serializer payloads."""
    control = sources["research-all-link-frontier.json"]["control"]
    narrow = sources["research-narrow-ingress.json"]["checkpoint_geometry"]
    original = sources["network-algebra-qwen25-2026-10-06.json"]
    bootstrap = sources["research-bootstrap-ingress.json"]["cost"]
    demand = sources["research-attested-linear-bootstrap.json"]["budget"]
    ni, oi, bi, di = (index(s) for s in (
        narrow["stages"], original["stages"], bootstrap["stages"], demand["stages"]))
    require(set(ni) == set(oi) == set(bi) == set(di) == set(control["stages"]),
            "missing stage identity")
    require(len(ni) == 96, "unexpected stage count")
    require(control == demand["baseline"], "historical control mismatch")
    require(control["pair_bytes"] == bootstrap["control_pair_bytes"] == 493575247,
            "historical total mismatch")
    require(sum(control["classes"].values()) == control["pair_bytes"], "nonconserving control")
    require(control["body_fingerprint"] == narrow["lock"]["body_fingerprint"]
            == original["source"]["body_fingerprint"] == bootstrap["control_body_fingerprint"],
            "body fingerprint mismatch")
    for key in ("model", "revision", "checkpoint_digest", "composition_digest", "source_lock_digest"):
        require(narrow["lock"][key] == original["source"][key], f"source lock mismatch: {key}")
    result = {}
    for name, s in ni.items():
        b, o, d = bi[name], oi[name], di[name]
        require(s["shape"] == b["shape"] == o["shape"] == d["shape"], "shape mismatch")
        require(s["weight_sha256"] == b["weight_sha256"] == o["weight_sha256"]
                == d["weight_sha256"], "weight fingerprint mismatch")
        m, n = s["shape"]
        require(m > 0 and n > 0, "invalid shape")
        ring = o["ring_bits"]
        require(ring == b["original_ring_bits"] and ring in (24, 32), "ring mismatch")
        width = s["stage_bits"]
        require(width == b["stage_bits"] == o["annihilator"]["uniform_input_bits"],
                "input width mismatch")
        hist = s["row_residue_width_histogram"]
        require(all(str(int(w)) == w and 0 < int(w) <= width <= ring
                    and type(c) is int and c > 0 for w, c in hist.items()), "invalid residue width")
        require(sum(hist.values()) == m and max(map(int, hist)) == width, "histogram mismatch")
        output_bits = sum(int(w) * c for w, c in hist.items())
        rows = natural(b["rows"])
        require(rows == d["pair_rows"] and b["body_macs"] == rows * m * n,
                "row multiplicity mismatch")
        require(b["output_payload"] == packed_bits(rows * output_bits)
                and b["old_input_payload"] == packed_bits(rows * n * width), "serializer payload mismatch")
        for link, payload in (("preparation->inference", b["output_payload"]),
                              ("inference->client", b["output_payload"]),
                              ("client->inference", b["old_input_payload"])):
            require(control["stages"][name][link] >= payload, "stage body below payload")
        result[name] = {"shape": [m, n], "ring_bits": ring, "input_bits": width,
                        "output_bits_per_row": output_bits, "width_histogram": hist,
                        "weight_sha256": s["weight_sha256"], "historical_rows": rows}
    require(sum(s["historical_rows"] for s in result.values()) == 15297, "total row mismatch")
    require(sum(b["output_payload"] for b in bi.values()) == 138864059, "total payload mismatch")
    return result


def compile_headers(sources, geometry):
    """Use semantic graph roles, never stage-name parsing or checkpoint tensors."""
    from pllm import Experiment, lower_model
    from pllm.runtime.semantic_stages import scheduled_prepared_phase_rows, scheduled_stage_specs

    plan_doc = sources["research-sota-combinations-qwen25.plan.json"]
    header = sources["placement-frontier-screen-2026-10-01.json"]
    original = sources["network-algebra-qwen25-2026-10-06.json"]["source"]
    for key in ("model", "revision", "checkpoint_digest", "source_lock_digest"):
        require(header["source"][key] == original[key], "header source lock mismatch")
    lowered = lower_model(header["retained_inputs"]["config"], batch=1,
                          max_input_tokens=384, max_new_tokens=8)
    preflight = {c["case"]: c for c in plan_doc["preflight"]}
    require(len(preflight) == len(plan_doc["preflight"]), "duplicate preflight case")
    headers = {}
    roles = None
    for case in plan_doc["suite"]["cases"]:
        name = case["name"]
        if name not in CASES:
            continue
        exp = Experiment.from_spec(case["target"]["parameters"]["model"]["experiment"])
        pf = preflight[name]
        require(lowered.digest == pf["admission"]["estimate"]["model_plan_digest"], "compiled plan drift")
        require(pf["source_lock_digest"] == original["source_lock_digest"], "preflight lock drift")
        specs = {s.id: s for s in scheduled_stage_specs(lowered, exp.pipeline) if s.layer_index is not None}
        require(set(specs) == set(geometry), "compiled stage set mismatch")
        this_roles = {}
        for stage_id, s in specs.items():
            require([s.out_features, s.in_features] == geometry[stage_id]["shape"], "compiled shape drift")
            this_roles[stage_id] = {"role": s.role, "layer": s.layer_index}
        require(roles is None or roles == this_roles, "semantic role drift")
        roles = this_roles
        requests = pf["preparation_work_window"]["requests"]
        input_tokens = int(name.split("/")[0])  # workload label, never stage identity
        for request, count in (("cold", input_tokens), ("repeat", 0)):
            actual = scheduled_prepared_phase_rows(lowered, exp.pipeline, count, 7)
            require(actual == requests[request]["phase_rows"], "compiled phase demand drift")
        phase_rows = {phase: {stage: sum(requests[r]["phase_rows"][phase][stage] for r in requests)
                              for stage in requests["cold"]["phase_rows"][phase]}
                      for phase in ("prefill", "decode")}
        remote_ids = set(phase_rows["decode"])
        fractions = {}
        for phase in ("prefill", "decode"):
            full = scheduled_prepared_phase_rows(lowered,
                Experiment.from_spec(plan_doc["suite"]["cases"][0]["target"]["parameters"]["model"]["experiment"]).pipeline,
                input_tokens, 7)[phase]
            total = sum(full[s] * math.prod(geometry[s]["shape"]) for s in full)
            remote = sum(full[s] * math.prod(geometry[s]["shape"]) for s in remote_ids)
            require(remote * 10 >= total * 9, "remote phase MAC floor violated")
            fractions[phase] = str(Fraction(remote, total))
        headers[name] = {"phase_rows": phase_rows, "remote_mac_fractions": fractions,
                         "compiled_plan_digest": lowered.digest}
    require(set(headers) == set(CASES), "missing compiled case")
    return roles, headers


def choose_subset(items, subset, lane_cost=None):
    require(subset in (*SUBSETS, "cost_threshold"), "unknown subset")
    if subset == "cost_threshold":
        if lane_cost is None:
            return None
        require(Fraction(lane_cost) >= 0, "negative generator cost")
        return [i for i in items if Fraction(i["payload_bytes"], i["lanes"]) > lane_cost]
    return [i for i in items if
            (subset not in ("gate_up_only", "gate_up_prefill", "gate_up_bulk_prefill") or i["role"] == "mlp_gate_up")
            and (subset not in ("prefill_only", "gate_up_prefill", "gate_up_bulk_prefill") or i["phase"] == "prefill")
            and (subset != "gate_up_bulk_prefill" or i["rows"] > 1)
            and (subset != "decode_only" or i["phase"] == "decode")]


def additional_cost(rows, lanes, capacity, *, burned=0, setup=0, renewal=0,
                    public_creation=0, public_distribution=0, lane_cost=0, control=0):
    """One global finite row-ticket pool; all deliveries are all-link totals.

    A row ticket names one stage invocation, not one request or output lane.
    Lanes must include all generated lanes, including burned/failure outputs.
    Per-stage capacities need separate pools; matrix triples use the function below.
    Initial issuance is setup; renewal charges only subsequent pools. Unknown
    required parameters propagate, never become zero. Unused slots are reported.
    """
    values = (rows, lanes, burned, setup, renewal, public_creation, public_distribution, control)
    for value in values:
        if value is not None:
            natural(value)
    if capacity is not None:
        require(type(capacity) is int and capacity > 0, "positive capacity required")
    if lane_cost is not None:
        require(Fraction(lane_cost) >= 0, "negative lane cost")
    if any(v is None for v in (*values, capacity, lane_cost)):
        return {"bytes": None, "pools": None, "unused_rows": None}
    pools = (rows + burned + capacity - 1) // capacity
    total = (setup if pools else 0) + max(0, pools - 1) * renewal
    total += public_creation + public_distribution + control + Fraction(lane_cost) * lanes
    return {"bytes": str(total), "pools": pools, "unused_rows": pools * capacity - rows - burned}


def break_even(total, public, removed, lanes, horizon, target, reuse, extra=0):
    """Exact conditional allowance per lane for the explicitly credited removal.

    Mandatory-payload credit is conservative, not a necessary upper bound on a
    candidate that also proves removal of some currently unallocated overhead.
    Crediting all correction residual gives the separate optimistic upper bound.
    """
    for v in (total, public, removed, lanes, horizon):
        natural(v)
    require(0 < public < total and 0 <= removed <= total - public and lanes > 0
            and horizon > 0 and 0 <= target < 1, "invalid budget")
    if extra is None:
        return None
    require(Fraction(extra) >= 0, "negative additional bytes")
    denominator = public + horizon * (total - public) if reuse else horizon * total
    return (horizon * removed - target * denominator - Fraction(extra)) / (horizon * lanes)


def matrix_columns(items, geometry, block=128, horizons=(1, 4, 16)):
    """Conditional padded exact-ring adaptation, fresh left mask each triple.

    Not a construction for the field-only paper. Each stage/block pool admits
    at most block fresh columns. Two dense opening shares charged each pool.
    No reuse across independent triples; no keys, auth or PCG costs assumed free.
    """
    require(type(block) is int and block > 0, "invalid block")
    result = []
    for horizon in horizons:
        natural(horizon)
        require(horizon > 0, "invalid horizon")
        openings = triples = adaptation_macs = unused = 0
        by_stage = {}
        for i in items:
            by_stage[i["stage"]] = by_stage.get(i["stage"], 0) + i["rows"]
        for stage, rows in by_stage.items():
            s = geometry[stage]
            m, n = s["shape"]
            pools = (horizon * rows + block - 1) // block
            blocks = ((m + block - 1) // block) * ((n + block - 1) // block)
            triples += pools * blocks
            openings += pools * blocks * 2 * packed_bits(block * block * s["ring_bits"])
            adaptation_macs += horizon * rows * m * n
            unused += (pools * block - horizon * rows) * blocks
        result.append({"pairs": horizon, "triples": triples, "opening_bytes": openings,
                       "unused_block_columns": unused, "extra_client_dense_macs": adaptation_macs})
    return result


def build_report():
    from probe_silent_linear_controls import reconcile

    sources = load_sources()
    geometry = validate_geometry(sources)
    roles, headers = compile_headers(sources, geometry)
    controls = reconcile(sources["research-sota-combinations-qwen25.json"])
    results = sources["research-sota-combinations-qwen25.json"]["results"]
    by_id = {r["id"]: r for r in results}
    cases = []
    historical = None
    for control in controls:
        case = control["case"]
        items = []
        for phase, rows_by_stage in headers[case]["phase_rows"].items():
            for stage, rows in rows_by_stage.items():
                if not rows:
                    continue
                s = geometry[stage]
                items.append({"stage": stage, "phase": phase, "rows": rows,
                              "role": roles[stage]["role"], "lanes": rows * s["shape"][0],
                              # Complete bytes in the exact bit volume. Round down:
                              # saved v3 cannot locate every job/frame padding byte.
                              "payload_bytes": rows * s["output_bits_per_row"] // 8})
        payload = sum(i["payload_bytes"] for i in items)
        if case == "150/both":
            for stage, s in geometry.items():
                require(sum(headers[case]["phase_rows"][p][stage] for p in ("prefill", "decode"))
                        == s["historical_rows"], "historical/current geometry demand mismatch")
            old = sources["research-all-link-frontier.json"]["control"]
            historical = {"pair_bytes": old["pair_bytes"], "correction_bytes": old["classes"]["corrections"],
                          "public_bytes": old["classes"]["public_bundle"], "subsets": {}}
            for subset in SUBSETS:
                chosen = choose_subset(items, subset)
                removed = sum(i["payload_bytes"] for i in chosen)
                lanes = sum(i["lanes"] for i in chosen)
                upper = removed + old["classes"]["corrections"] - payload
                historical["subsets"][subset] = {
                    "mandatory_payload_removed": removed, "optimistic_removed": upper,
                    "first_pair_zero_cost_max": str(Fraction(upper, old["pair_bytes"])),
                    "first_pair_20pct_mandatory_lane_allowance": str(break_even(old["pair_bytes"],
                        old["classes"]["public_bundle"], removed, lanes, 1, Fraction(1, 5), True)),
                    "first_pair_20pct_optimistic_lane_ceiling": str(break_even(old["pair_bytes"],
                        old["classes"]["public_bundle"], upper, lanes, 1, Fraction(1, 5), True))}
        samples = []
        for sample in control["samples"]:
            audit = by_id[sample["result_id"]]["execution"]["details"]["runtime_audit"]
            require(len(audit["windows"]) == 2, "missing request audit")
            for phase in ("prefill", "decode"):
                recorded_rows = sum(w[f"preparation_{phase}_rows_issued"] for w in audit["windows"])
                require(recorded_rows == sum(i["rows"] for i in items if i["phase"] == phase),
                        "runtime issued-row mismatch")
            require(audit["totals"]["prepared_stage_rows_burned"] == 0
                    and audit["totals"]["prepared_stage_rows_discarded"] == 0, "unexpected unused issuance")
            residual = sample["correction_bytes"] - payload
            require(residual >= 0, "correction serializer below payload")
            subsets = {}
            for subset in SUBSETS:
                selected = choose_subset(items, subset)
                removed = sum(i["payload_bytes"] for i in selected)
                lanes = sum(i["lanes"] for i in selected)
                upper = removed + residual  # deliberately grants all remaining correction bodies
                total, public = sample["covered_pair_bytes"], sample["public_delivery_bytes"]
                subsets[subset] = {
                    "payload_removed": removed, "optimistic_removed": upper, "output_lanes": lanes,
                    "row_tickets": sum(i["rows"] for i in selected),
                    "input_mask_lanes": sum(i["rows"] * geometry[i["stage"]]["shape"][1] for i in selected),
                    "zero_cost_maxima": [str(Fraction(upper, total)), str(Fraction(upper, total - public))],
                    "20pct_payload_lane_ceiling": [str(break_even(total, public, removed, lanes, h,
                                                                  TARGETS[0], True)) for h in HORIZONS],
                    "20pct_optimistic_lane_ceiling": [str(break_even(total, public, upper, lanes, h,
                                                                     TARGETS[0], True)) for h in HORIZONS],
                    "20pct_first_pair_headroom_mandatory_optimistic": [str(Fraction(s) - TARGETS[0] * total)
                                                                     for s in (removed, upper)],
                    "warm_payload_lane_ceilings_20_30_35": [str(Fraction(removed - t * (total - public), lanes))
                                                            for t in TARGETS],
                }
            samples.append({k: sample[k] for k in ("replicate", "covered_pair_bytes", "public_delivery_bytes",
                           "correction_bytes", "known_base_bytes", "known_output_dependent_bytes", "unattributed_bytes")}
                           | {"correction_payload": payload, "correction_residual": residual, "subsets": subsets})
        representative = sorted(samples, key=lambda s: s["covered_pair_bytes"])[1]
        subsets = representative["subsets"]
        representative_replicate = representative["replicate"]
        for sample in samples:
            sample.pop("subsets")
        thresholds = sorted({Fraction(i["payload_bytes"], i["lanes"]) for i in items})
        frontier = []
        # All exact crossing points define the bounded analytical search. Report
        # four readable parameter slices, not invented generator measurements.
        for cost in (Fraction(), Fraction(1), Fraction(2), Fraction(3)):
            chosen = choose_subset(items, "cost_threshold", cost)
            frontier.append({"lane_cost": str(cost), "stage_phase_groups": len(chosen),
                             "net_payload_saving": str(sum(i["payload_bytes"] - cost * i["lanes"] for i in chosen))})
        bulk = choose_subset(items, "gate_up_bulk_prefill")
        first = max(bulk, key=lambda i: Fraction(i["payload_bytes"], i["lanes"]))
        cases.append({"case": case, "remote_mac_fractions": headers[case]["remote_mac_fractions"],
                      "samples": samples, "representative_replicate": representative_replicate,
                      "subsets": subsets, "cost_threshold_slices": frontier,
                      "exact_threshold_count": len(thresholds),
                      "bulk_gate_semantic_layers": sorted(roles[i["stage"]]["layer"] for i in bulk),
                      "highest_density_bulk_gate": first | roles[first["stage"]],
                      "matrix128_gate_up_prefill": matrix_columns(choose_subset(items, "gate_up_prefill"), geometry)})
    paper_manifest = sources["compact-setup-fulltext-review.json"]["papers"]
    papers = []
    for paper in paper_manifest:
        if paper["id"] not in ("2025/1223", "2026/2298"):
            continue
        require(sha(ROOT / paper["path"]) == paper["sha256"], "paper hash drift")
        extraction = ROOT / "target/compact-setup-fulltext" / (paper["id"].replace("/", "-") + ".txt")
        extraction_sha = {"2025/1223": "4a91404c17a157c0449f625c2c3532bcb722447ad23d03d9e58b53d2262b5821",
                          "2026/2298": "c6642f525f8f96394b93dc5e65998790d37674dc36bb2597a29950fc92879b93"}[paper["id"]]
        require(sha(extraction) == extraction_sha, "paper extraction drift")
        papers.append({k: paper[k] for k in ("id", "path", "sha256", "reviewed_pages")}
                      | {"text_sha256": extraction_sha})
    load_sources()  # Detect source mutation during header-only replay.
    from pllm import lower_model
    from pllm.runtime.semantic_stages import scheduled_stage_specs
    return {"schema": "pllm.selective_correlation_budget.v1", "complete": True,
            "scope": "conditional covered application bodies, exact payload subset; not TCP or physical wire",
            "horizon_unit": "new-prefix eight-output cold request plus exact repeat; public reuse symmetric",
            "ceiling_order": ["first_cold_plus_repeat", "warm_all_public_reused"],
            "allowance_scope": {
                "payload": "Conservative allowance crediting mandatory payload alone; not a necessary upper bound if additional overhead removal is proved.",
                "optimistic": "Necessary upper bound crediting mandatory payload plus all correction residual; residual attribution/removability unproved.",
                "matrix128_gate_up_prefill": "Terminal-inclusive gate/up-prefill control, distinct from recommended bulk-only subset."},
            "horizons": list(HORIZONS), "cases": cases, "historical_control": historical, "papers": papers,
            "geometry": {"stage_count": len(geometry), "roles": roles,
                         "historical_output_payload": 138864059,
                         "historical_stage_correction_bodies": sum(s["preparation->inference"] for s in
                             sources["research-all-link-frontier.json"]["control"]["stages"].values())},
            "source_hashes": LOCKS,
            "installed_header_implementation_hashes": {
                "lower_model": sha(Path(inspect.getfile(lower_model))),
                "scheduled_stage_specs": sha(Path(inspect.getfile(scheduled_stage_specs)))},
            "implementation_hashes": {p: sha(ROOT / p) for p in (
                "scripts/probe_selective_correlation_budget.py", "scripts/probe_silent_linear_controls.py",
                "python/pllm/runtime/semantic_stages.py", "tests/test_selective_correlation_budget_probe.py",
                "python/pllm/runtime/preparation_protocol.py", "crates/pllm-core/src/offset_transport.rs",
                "docs/research/selective-correlation-budget-reference.md")},
            "candidate": {"generator": None, "security_parameters": None, "client_crypto_work": None,
                          "client_peak_rss": None, "low_qualified": None, "medium_qualified": None,
                          "new_numeric_identity": None, "new_verifier_identity": None,
                          "setup_and_role_key_bytes": None, "capacity": None,
                          "public_cartridge_creation_and_distribution_bytes": None},
            "limitations": [
                "Partial v3 attribution is not a complete base/per-token decomposition; residual remains unattributed.",
                "Payload_removed counts whole bytes in exact phase/stage bit volume rounded down; unlocated serializer/job padding is residual.",
                "Payload removal is conditional on compatible generator/serializer; controls and ACKs retained.",
                "Optimistic subset bound gives every correction residual to selected subset; not measured attribution.",
                "All original MACs retain existing placement, separately >=90% remote per phase; extra crypto is additional.",
                "Header-only lowering validates saved plan, semantic groups and phase demand; no checkpoint tensors read.",
                "Finite capacity, burns, unused slots, both role keys, public creation/distribution and renewals require pricing.",
                "Matrix128 is an exact-ring conditional cost control, not the paper's field construction or client-light Gen.",
                "Provider checkpoint/tokenizer distribution absent in source boundary; cannot claim full bootstrap/wire savings.",
                "No RAM arithmetic is fresh-process Low <=256 MiB or Medium <=2 GiB qualification.",
            ]}


def publication_summary(report):
    """Compact deterministic evidence; full replay retains every analytical field."""
    cases = []
    for case in report["cases"]:
        subsets = {}
        for name, s in case["subsets"].items():
            subsets[name] = [s["payload_removed"], s["optimistic_removed"], s["output_lanes"],
                             s["input_mask_lanes"], *s["zero_cost_maxima"],
                             s["20pct_payload_lane_ceiling"], s["warm_payload_lane_ceilings_20_30_35"],
                             s["20pct_optimistic_lane_ceiling"], s["20pct_first_pair_headroom_mandatory_optimistic"]]
        cases.append({"case": case["case"], "representative_replicate": case["representative_replicate"],
                      "remote_mac_fractions": case["remote_mac_fractions"], "subsets": subsets,
                      "samples": case["samples"], "matrix128_gate_up_prefill": case["matrix128_gate_up_prefill"],
                      "highest_density_bulk_gate": case["highest_density_bulk_gate"],
                      "bulk_gate_semantic_layers": case["bulk_gate_semantic_layers"]})
    return {"schema": "pllm.selective_correlation_budget_summary.v1", "complete": True,
            "scope": report["scope"], "horizon_unit": report["horizon_unit"],
            "horizons": report["horizons"], "subset_columns": [
                "mandatory_payload_removed", "optimistic_removed", "output_lanes", "input_mask_lanes",
                "zero_cost_first_pair_fraction", "zero_cost_warm_fraction",
                "20pct_mandatory_payload_lane_allowance_at_horizons", "warm_mandatory_lane_allowances_20_30_35",
                "20pct_optimistic_all_residual_lane_ceiling_at_horizons", "20pct_first_pair_headroom_mandatory_optimistic"],
            "allowance_scope": report["allowance_scope"],
            "cases": cases, "historical_control": report["historical_control"],
            "papers": report["papers"], "source_hashes": report["source_hashes"],
            "implementation_hashes": report["implementation_hashes"],
            "installed_header_implementation_hashes": report["installed_header_implementation_hashes"],
            "candidate": report["candidate"], "limitations": report["limitations"],
            "full_report_canonical_sha256": hashlib.sha256(json.dumps(report, sort_keys=True,
                                                      separators=(",", ":")).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", action="store_true", help="write compact publication evidence")
    args = parser.parse_args()
    result = build_report()
    if args.summary:
        result = publication_summary(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "complete": True}))


if __name__ == "__main__":
    main()
