"""Matched ordinary SDK decode-cache cohort; no installs/builds or private evidence.

Six frozen cells, three fresh-client replicates, three seeded requests per client.
Pending SDK/work-accounting boundaries reject execution rather than infer savings.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time
import weakref

from benchmark_conversation_restore import commitment
from benchmark_epoch_prefix_sdk import MemoryAdmissionRejected, admission as base_admission
from benchmark_growing_epochs_sdk import (case as role_case, semantic_geometry, validate_audit,
    bounded_commitment, complete_state_commitment, freeze_installed_implementation,
    installed_bootstrap, validate_evaluated_state)
from frontier_probe_support import fresh_workers

ROOT = Path(__file__).resolve().parents[1]
LOW_BYTES = 268435456
POLICIES = {"ordinary": (None, False), "fixed1": (1, False), "decode": (1, True)}
SCENARIOS = {"same_seed": (7, 7, 7), "distinct_seeds": (7, 11, 19)}
PROMPT = "Explain how water moves from tree roots to leaves, then describe how dry weather changes that process. Give a detailed answer."
DECODE_MAX_BYTES = 32 << 20
OWNER_LIMITS = {"max_nodes": 128, "max_staged": 8, "max_pages": 8192,
    "max_files": 16384, "max_disk_bytes": 256 << 20, "max_index_bytes": 2 << 20,
    "max_snapshot_bytes": 8 << 20, "max_transient_bytes": 32 << 20, "max_tokens": 128}
# Parent-confirmed cumulative audit names. Verify installed behavior before runs;
# these are execution measurements, not hit-label estimates.
WORK_COUNTERS = {
    "requested_prefill_tokens": "generation_requested_prefill_tokens",
    "requested_decode_steps": "generation_requested_decode_steps",
    "executed_prefill_tokens": "generation_remote_prefill_tokens",
    "executed_terminal_prefill_rows": "generation_remote_terminal_prefill_rows",
    "executed_decode_steps": "generation_remote_decode_steps",
}
CACHE_COUNTERS = {"root_hits": "decode_cache_root_hits", "branch_hits": "decode_cache_branch_hits",
    "misses": "decode_cache_misses", "skipped_remote_rows": "decode_cache_skipped_remote_rows",
    "skipped_prefill_tokens": "decode_cache_skipped_prefill_tokens",
    "skipped_decode_steps": "decode_cache_skipped_decode_steps",
    "warm_no_preparation": "decode_cache_fully_warm_responses"}
STORAGE_FIELDS = ("disk_bytes", "disk_files", "read_bytes", "written_bytes", "node_count",
    "page_count", "staged_count", "hits", "misses", "publications", "peak_transient_bytes",
    "unpublished_bytes", "unpublished_files")
STORAGE_COUNTERS = ("read_bytes", "written_bytes", "hits", "misses", "publications")


def owner_factory(directory, key):
    """Existing caller-owned API; fresh directory, no disk-discovered trusted head."""
    from pllm.state.decode import DecodeCache
    return DecodeCache(directory, key, **OWNER_LIMITS)


def experiment(base, policy, cap=8):
    from pllm.preparation import PreparedInventory
    from pllm.state import ClientPrefixReuse
    epoch, decode = POLICIES[policy]
    return base.with_params(
        pipeline__inventory=PreparedInventory("request-sized", rows=1, allocation="demand",
            refill="on-demand", stage_window=4, stage_window_bytes=64 << 20,
            decode_epoch_tokens=epoch),
        pipeline__cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=128,
            generated_prefixes=True, decode=decode,
            **({"decode_max_bytes": DECODE_MAX_BYTES} if decode else {})),
        budget__requests=3, budget__max_input_tokens=128, budget__max_new_tokens=cap)


def cohort_order(keys, replicates):
    return [(replicate, key) for replicate in range(replicates)
            for key in (keys if replicate % 2 == 0 else list(reversed(keys)))]


def api_boundary():
    import pllm
    from pllm.runtime.client import PrivacyAudit
    missing = []
    if "decode_cache" not in inspect.signature(pllm.OpenAI).parameters:
        missing.append("OpenAI(decode_cache=owner)")
    try:
        from pllm.state.decode import DecodeCache
        params = inspect.signature(DecodeCache).parameters
        if any(key not in params for key in ("directory", "key", *OWNER_LIMITS)):
            missing.append("DecodeCache constructor bounds")
    except ImportError:
        missing.append("pllm.state.decode.DecodeCache")
    counters = PrivacyAudit().to_dict()
    missing.extend(name for name in (*WORK_COUNTERS.values(), *CACHE_COUNTERS.values()) if name not in counters)
    return {"validated": False, "structurally_ready": not missing, "missing": missing,
        "note": "Structural presence is not functionality proof; parent must validate generated integration tests before running."}


def admission(exp, policy, *, require=True):
    report = base_admission(exp, require=False)
    reasons = list(report["reasons"])
    cache_bound = None
    if policy == "decode":
        composition_bound = exp.pipeline.components["cache"].params.get("decode_max_bytes")
        priced = report["topology"].get("estimate", {}).get("components", {}).get("client_decode_owner_bytes")
        if (type(composition_bound) is not int or composition_bound != DECODE_MAX_BYTES
                or OWNER_LIMITS["max_transient_bytes"] > composition_bound
                or type(priced) is not int or priced != composition_bound):
            reasons.append("composition decode owner envelope not verified in native admission")
        cache_bound = {"composition_bytes": composition_bound, "already_charged_bytes": priced,
            "owner_transient_limit_bytes": OWNER_LIMITS["max_transient_bytes"],
            "scope": "additional client owner envelope already in native topology estimate; not added twice"}
    required = report["required_bytes"]
    if required > report["topology"]["host"]["admission_budget_bytes"]:
        reasons.append("decode storage owners exceed post-reserve headroom")
    disk_required = report["topology"].get("candidates", {}).get("native", {}).get("required_disk_bytes")
    disk_free = report["topology"]["host"].get("disk_free")
    if policy == "decode":
        if type(disk_required) is not int or type(disk_free) is not int:
            reasons.append("decode disk admission scope unavailable")
        elif disk_required + OWNER_LIMITS["max_disk_bytes"] > disk_free:
            reasons.append("encrypted decode cache exceeds disk headroom after existing reserve")
    report = {**report, "decode_owner": cache_bound, "decode_owner_extra_bytes": 0,
        "required_bytes": required, "admitted": report["admitted"] and not reasons, "reasons": reasons,
        "disk_capacity_bytes": OWNER_LIMITS["max_disk_bytes"] if policy == "decode" else 0,
        "required_disk_bytes": disk_required + OWNER_LIMITS["max_disk_bytes"]
            if policy == "decode" and type(disk_required) is int else disk_required}
    if require and not report["admitted"]:
        raise MemoryAdmissionRejected(report)
    return report


def storage_metrics(owner):
    if owner is None:
        return {name: 0 for name in STORAGE_FIELDS}
    raw = owner.storage_metrics
    if not raw:  # Unbound owner did no storage I/O yet.
        return {name: 0 for name in STORAGE_FIELDS}
    return {name: raw.get(name) if type(raw.get(name)) is int and raw[name] >= 0 else None
            for name in STORAGE_FIELDS}


def counter_delta(before, after):
    """Monotonic measured counters only; owner high-water gauges remain separate."""
    return {key: value - before.get(key, 0) for key, value in after.items()
            if type(value) is int and type(before.get(key, 0)) is int and value >= before.get(key, 0)}


@contextmanager
def harness_sampler_seed(seed, *, runtime_class=None, salt):
    """Seed actual sampling runtime once; retain bounded salted witnesses only."""
    import numpy as np
    if runtime_class is None:
        from pllm.runtime.semantic_executor import SemanticDecoderRuntime
        runtime_class = SemanticDecoderRuntime
    original, seen = runtime_class.sample, weakref.WeakSet()
    witness = {"initial": None, "final": None, "trajectory": None, "samples": 0, "runtimes": 0}

    def seeded(runtime, *args, **kwargs):
        if runtime not in seen:
            if witness["runtimes"]:
                raise RuntimeError("multiple sampled runtimes in serial request")
            runtime.rng = np.random.default_rng(seed)
            seen.add(runtime)
            witness["runtimes"] += 1
            witness["initial"] = bounded_commitment(salt, runtime.rng.bit_generator.state)
        before = bounded_commitment(salt, runtime.rng.bit_generator.state)
        result = original(runtime, *args, **kwargs)
        after = bounded_commitment(salt, runtime.rng.bit_generator.state)
        witness["samples"] += 1
        witness["final"] = after
        witness["trajectory"] = bounded_commitment(salt, {
            "prior": witness["trajectory"], "before": before, "after": after,
            "token": int(result), "sample": witness["samples"]})
        return result

    runtime_class.sample = seeded
    try:
        yield witness
    finally:
        runtime_class.sample = original


def sampling_response(client, prompt, cap, seed, salt, index):
    from pllm.runtime.types import Response
    started, cpu = time.perf_counter(), time.process_time()
    events, response = [], None
    with client.tokenizer_scope(), harness_sampler_seed(seed, salt=salt) as rng:
        with client.responses.create(input=prompt, max_output_tokens=cap, temperature=.7,
                top_p=.9, stream=True, store=True) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    events.append(time.perf_counter())
                elif event.type in {"response.completed", "response.incomplete"}:
                    response = Response.from_dict(event.response) if isinstance(event.response, dict) else event.response
    elapsed, used_cpu = time.perf_counter() - started, time.process_time() - cpu
    if (response is None or response.usage is None or response.status not in {"completed", "incomplete"}
            or (response.status == "incomplete" and (response.incomplete_details or {}).get("reason") != "max_output_tokens")
            or not 0 <= response.usage.output_tokens <= cap or len(events) != response.usage.output_tokens):
        raise RuntimeError("terminal response/event accounting failed")
    expected_samples = response.usage.output_tokens + int(response.status == "completed")
    if rng["runtimes"] != 1 or rng["samples"] != expected_samples or rng["final"] is None:
        raise RuntimeError("actual request sampler/RNG witness unavailable")
    state = complete_state_commitment(client, response, salt)
    state.update(rng=dict(rng), rng_scope="actual seeded sampling runtime; initial/final/per-selection trajectory")
    return {"index": index, "harness_numpy_seed": seed, "status": response.status,
        "sampling_fixture": "harness-only SemanticDecoderRuntime.sample RNG replacement once per actual runtime; serial request scoped",
        "state": state, "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens, "request_seconds": elapsed,
        "request_cpu_seconds": used_cpu, "first_text_seconds": events[0] - started if events else None,
        "decode_tps": (len(events) - 1) / (events[-1] - events[0]) if len(events) > 1 else None,
        "cache_temperature": "cold" if index == 0 else "prior_requests_completed; branch warmth measured, not assumed"}


def measured_work(geometry, row):
    counters = row["audit_delta"]
    counts = {key: counters.get(name) for key, name in WORK_COUNTERS.items()}
    cache = {key: counters.get(name) for key, name in CACHE_COUNTERS.items()}
    valid = all(type(value) is int and value >= 0 for value in counts.values())
    if not valid:
        return {"available": False, "counts": counts, "cache": cache,
            "requested_equivalent_body_macs": None, "actually_executed_body_macs": None,
            "actually_executed_remote_body_macs": None, "remote_fraction_of_executed_body_macs": None,
            "remote_90_percent_verified": None}
    p, d = counts["requested_prefill_tokens"], counts["requested_decode_steps"]
    ep, et, ed = counts["executed_prefill_tokens"], counts["executed_terminal_prefill_rows"], counts["executed_decode_steps"]
    expected_decode = max(0, row["output_tokens"] - 1) if row["status"] == "incomplete" else row["output_tokens"]
    # Fresh prefill prunes terminal stages to one row. Proper-prefix reuse runs
    # the native continuation schedule, whose terminal stages keep every suffix
    # row. Ordinary resident prefix reuse can take this path even on repeats.
    terminal_rows = min(ep, 1) if ep == p else ep
    if (p != row["input_tokens"] or d != expected_decode
            or ep > p or ed > d or et != terminal_rows):
        raise RuntimeError("requested/executed semantic work conservation failed")
    validate_evaluated_state(row, d)
    requested = executed = remote = requested_remote_rows = actual_remote_rows = 0
    for stage in geometry:
        requested_rows = (1 if stage["terminal_prefill"] else p) + d
        actual_rows = (et if stage["terminal_prefill"] else ep) + ed
        requested += requested_rows * stage["macs_per_row"]
        macs = actual_rows * stage["macs_per_row"]
        executed += macs
        remote += macs if stage["remote"] else 0
        requested_remote_rows += requested_rows if stage["remote"] else 0
        actual_remote_rows += actual_rows if stage["remote"] else 0
    return {"available": True, "counts": counts, "cache": cache,
        "requested_equivalent_body_macs": requested, "actually_executed_body_macs": executed,
        "actually_executed_remote_body_macs": remote, "eliminated_body_macs": requested - executed,
        "requested_equivalent_remote_stage_rows": requested_remote_rows,
        "actual_remote_stage_rows": actual_remote_rows,
        "eliminated_remote_stage_rows": requested_remote_rows - actual_remote_rows,
        "remote_fraction_of_executed_body_macs": remote / executed if executed else None,
        "remote_fraction_of_requested_equivalent_body_macs": remote / requested if requested else None,
        "remote_90_percent_verified": remote * 10 >= executed * 9 if executed else None,
        "zero_executed_work": executed == 0,
        "denominator": "actually executed scheduled body dense linear MACs; eliminated work excluded; equivalent work reported separately",
        "whole_body_compute_fraction": None}


def validate_request(row, policy, geometry):
    from pllm.runtime.topology_accounting import ACK_COVERAGE
    ledger, delta = row["request_ledger"], row["audit_delta"]
    rng = row["state"].get("rng", {})
    if (not isinstance(rng, dict) or rng.get("runtimes") != 1
            or rng.get("samples") != row["output_tokens"] + int(row["status"] == "completed")
            or any(not isinstance(rng.get(key), str) or len(rng[key]) != 64
                   or not set(rng[key]) <= set("0123456789abcdef")
                   for key in ("initial", "final", "trajectory"))
            or any(not isinstance(row["state"].get(key), str)
                   for key in ("history", "rendered_history"))):
        raise RuntimeError("actual history/RNG witnesses unavailable")
    if not ledger["tracked_body_counter_set_present"] or ledger["body_counter_coverage"] != ACK_COVERAGE:
        raise RuntimeError("request body window unavailable")
    work = measured_work(geometry, row)
    if not work["available"] or work["remote_90_percent_verified"] is False:
        raise RuntimeError("actual remote semantic work unavailable or gate failed")
    if delta.get("prepared_stage_rows_claimed") != work["actual_remote_stage_rows"]:
        raise RuntimeError("claimed prepared rows do not reconcile actual semantic remote rows")
    if policy == "decode":
        cache = work["cache"]
        if any(type(value) is not int or value < 0 for value in cache.values()):
            raise RuntimeError("decode hit/miss/skipped-work counters unavailable")
        if cache["warm_no_preparation"]:
            if (cache["warm_no_preparation"] != 1 or not work["zero_executed_work"]
                    or any(delta.get(name) != 0 for name in ("prepared_stage_rows_issued",
                        "prepared_stage_rows_reserved", "preparation_upload_bytes",
                        "preparation_download_bytes", "correction_push_bytes"))):
                raise RuntimeError("fully warm lookup minted fresh preparation")
        if cache["skipped_decode_steps"] > work["counts"]["requested_decode_steps"]:
            raise RuntimeError("skipped decode work exceeds requested work")
        if (cache["skipped_prefill_tokens"] > work["counts"]["requested_prefill_tokens"]
                or cache["skipped_decode_steps"] > work["counts"]["requested_decode_steps"] - work["counts"]["executed_decode_steps"]
                or cache["skipped_remote_rows"] > work["eliminated_remote_stage_rows"]):
            raise RuntimeError("cache skipped work exceeds actual eliminated work")
    row["semantic_work"] = work
    return work


def client_worker(payload):
    import pllm
    from pllm.runtime.benchmark_response_group import inspect_private_cleanup, prefix_owners, require_concurrency_contract
    from pllm.runtime.topology_accounting import prepared_body_accounting
    started = time.perf_counter()
    client, owner, owners, failure, initialized = None, None, (), None, None
    rows = []
    try:
        if payload["policy"] == "decode":
            owner = owner_factory(Path(payload["cache"]).parent / "decode-cache", bytes.fromhex(payload["decode_key"]))
        client = pllm.OpenAI(experiment=pllm.Experiment.from_spec(payload["experiment"]),
            **payload["connection"], bundle_cache_dir=payload["cache"],
            background_inventory_refill=False, timeout=900, **({"decode_cache": owner} if owner else {}))
        require_concurrency_contract(client)
        with client.tokenizer_scope():
            client.runtime.capabilities()
            client.prepared_rows_for_response(payload["prompts"][0], payload["cap"], store=True)
        initialized = time.perf_counter()
        for index, seed in enumerate(payload["seeds"]):
            before, disk_before = client.privacy_audit.to_dict(), storage_metrics(owner)
            row = sampling_response(client, payload["prompts"][index], payload["cap"], seed,
                                    bytes.fromhex(payload["salt"]), index)
            after, disk_after = client.privacy_audit.to_dict(), storage_metrics(owner)
            delta = counter_delta(before, after)
            row.update({"audit_delta": delta, "request_ledger": prepared_body_accounting({"privacy": delta}),
                "disk_io": {name: disk_after[name] - disk_before[name]
                    if disk_after[name] is not None and disk_before[name] is not None else None
                    for name in STORAGE_COUNTERS}, "storage_after": disk_after})
            rows.append(row)
    except BaseException as error:
        failure = {"stage": "client_execution", "error_type": type(error).__name__}
    finally:
        if client is not None:
            owners = prefix_owners(client)
            try:
                client.close()
            except BaseException as error:
                failure = {"stage": "client_retirement", "error_type": type(error).__name__}
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                failure = {"stage": "owner_retirement", "error_type": type(error).__name__}
    audit = client.privacy_audit.to_dict() if client else {}
    cleanup = None
    try:
        if client is not None:
            cleanup = inspect_private_cleanup(client, owners)
            if client._core._prepared_queue.requests or client._core._online_active or client._core._preparation_active:
                raise RuntimeError("prepared owner did not retire")
        ledger = validate_audit(audit) if failure is None else prepared_body_accounting({"privacy": audit})
    except BaseException as error:
        failure = {"stage": "client_validation", "error_type": type(error).__name__}
        ledger = prepared_body_accounting({"privacy": audit})
    disk = storage_metrics(owner)
    disk_residual = {name: disk[name] - sum(row["disk_io"][name] for row in rows)
        if disk[name] is not None and all(row["disk_io"][name] is not None for row in rows) else None
        for name in STORAGE_COUNTERS}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"responses": rows, "audit": audit, "ledger": ledger, "cleanup": cleanup,
        "failure": failure, "storage_lifetime": disk, "storage_setup_cleanup_or_failed_request_residual": disk_residual,
        "storage_bounds": OWNER_LIMITS if owner else None,
        "client_lifetime_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "client_peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        "client_setup_seconds": initialized - started if initialized is not None else None,
        "client_window_seconds": time.perf_counter() - started,
        "publisher_artifact_distribution_bytes": None, "publisher_artifact_disk_bytes": None}


def case(exp, scenario, policy, replicate, salt, run, *, tiny=False):
    memory = admission(exp, policy)
    prompt = "Explain sky." if tiny else PROMPT
    workload = {"prompts": [prompt] * 3, "cap": 8, "seeds": SCENARIOS[scenario],
                "policy": policy, "decode_key": os.urandom(32).hex()}
    result = role_case(exp, workload, scenario, policy, replicate, salt, run)
    result["admitted_host_bytes"] = memory["required_bytes"]
    return result


def verify_results(results, geometries, *, require_complete=False, replicates=3):
    reference, peaks, variation = {}, {policy: [] for policy in POLICIES}, {}
    seen = set()
    for result in results:
        scenario, policy, replicate = result["workload"], result["variant"], result["replicate"]
        identity = (scenario, policy, replicate)
        if identity in seen:
            raise RuntimeError("duplicate cohort cell")
        seen.add(identity)
        client = result["client"]
        if client.get("failure") or len(client["responses"]) != 3:
            raise RuntimeError("failed or incomplete three-request client")
        validate_audit(client["audit"])
        peaks[policy].append(client["client_peak_rss_bytes"])
        for index, row in enumerate(client["responses"]):
            if row["index"] != index or row["harness_numpy_seed"] != SCENARIOS[scenario][index]:
                raise RuntimeError("request order or harness seed changed")
            state = {key: row[key] for key in ("status", "state", "input_tokens", "output_tokens")}
            key = (scenario, index)
            reference.setdefault(key, state)
            if state != reference[key]:
                raise RuntimeError("exact logits/KV/history/pending/output parity failed")
            if row["input_tokens"] + 7 > 128:
                raise RuntimeError("full-cap continuation bound exceeded")
            validate_request(row, policy, geometries[scenario + "/" + policy]["stages"])
            if scenario == "same_seed" and policy == "decode" and index > 0:
                work = row["semantic_work"]
                if (not work["zero_executed_work"] or work["cache"]["warm_no_preparation"] != 1
                        or work["cache"]["root_hits"] < 1
                        or work["cache"]["branch_hits"] < work["counts"]["requested_decode_steps"]):
                    raise RuntimeError("declared same-seed warm root/branch path not exercised")
        digests = [row["state"]["output"] for row in client["responses"]]
        if scenario == "same_seed" and len(set(digests)) != 1:
            raise RuntimeError("same-seed sampler continuity failed")
        variation.setdefault(scenario, []).append({"policy": policy, "replicate": replicate,
            "distinct_output_commitments": len(set(digests))})
        if policy == "decode":
            storage = client["storage_lifetime"]
            if (any(storage.get(key) is None for key in STORAGE_FIELDS)
                    or storage["disk_bytes"] > OWNER_LIMITS["max_disk_bytes"]
                    or storage["publications"] < 1 or storage["written_bytes"] < 1):
                raise RuntimeError("cold encrypted publication/storage accounting unavailable")
            if any(sum(row["disk_io"][key] for row in client["responses"]) > storage[key]
                   for key in STORAGE_COUNTERS):
                raise RuntimeError("request disk counters exceed complete lifecycle")
    if require_complete and seen != {(s, p, r) for s in SCENARIOS for p in POLICIES for r in range(replicates)}:
        raise RuntimeError("cohort cell coverage incomplete")
    return {"exact_state_and_output_match": True, "checked_requests": 3 * len(results),
        "largest_client_peak_by_policy": {p: max(values) if values else None for p, values in peaks.items()},
        "low_gate_by_policy": {p: max(values) <= LOW_BYTES if values else None for p, values in peaks.items()},
        "low_threshold_bytes": LOW_BYTES, "sampler_variation": variation,
        "actual_remote_work_checked": True,
        "zero_executed_work_fraction": None,
        "note": "Zero-work warm requests have no remote fraction; no client dense body multiply admitted."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tiny", "--smoke", action="store_true")
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(client_worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("supply a new --output")
    if not (1 if args.tiny else 3) <= args.replicates <= 12:
        parser.error("replicates must be 3..12 (tiny permits 1..12)")
    import pllm
    from pllm.metrics import BuildRef
    from pllm.model_loader import resolve_model
    factory = runpy.run_path(str(ROOT / "examples/prepared_sdk.py"))["build_experiment"]
    if args.tiny:
        base = factory()
    else:
        declared = pllm.Experiment.from_spec(json.loads((args.artifacts / "indexed.json").read_bytes()))
        base = factory(declared.pipeline.model, tokenizer=declared.pipeline.components["tokenizer"])
    experiments = {s + "/" + p: experiment(base, p) for s in SCENARIOS for p in POLICIES}
    admissions = {key: admission(exp, key.split("/")[1], require=False) for key, exp in experiments.items()}
    required = max(value["required_bytes"] for value in admissions.values())
    source = resolve_model(base.pipeline.model)
    geometry = {key: semantic_geometry(exp, source) for key, exp in experiments.items()}
    files = [Path(__file__), ROOT / "tests/test_benchmark_decode_cache_sdk.py",
        ROOT / "examples/benchmarks/decode_cache_sdk.py", ROOT / "docs/research/decode-cache-benchmark-reference.md",
        ROOT / "scripts/benchmark_growing_epochs_sdk.py", ROOT / "scripts/benchmark_epoch_prefix_sdk.py",
        ROOT / "scripts/benchmark_conversation_restore.py", ROOT / "scripts/frontier_probe_support.py",
        ROOT / "examples/prepared_sdk.py"]
    def hashes():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    frozen, salt = hashes(), os.urandom(32)
    build, boundary = BuildRef.capture().to_spec(), api_boundary()
    freeze_installed_implementation(build)
    output = {"schema": "pllm.decode_cache_sdk_benchmark.v1", "complete": False, "tiny": args.tiny,
        "planned_clients": 6 * args.replicates, "planned_requests": 18 * args.replicates,
        "sampling": {"temperature": .7, "top_p": .9,
                      "seed_kind": "harness-only SemanticDecoderRuntime.sample: seed actual runtime once; original sampler preserved; bounded RNG witnesses",
                     "scenarios": SCENARIOS}, "api_boundary": boundary, "storage_bounds": OWNER_LIMITS,
        "experiments": {key: exp.to_spec() for key, exp in experiments.items()},
        "config_hashes": {key: hashlib.sha256(exp.canonical_bytes()).hexdigest() for key, exp in experiments.items()},
         "build": build, "source_lock_digest": source.source_lock_digest, "frozen_sources": frozen,
         "execution_isolation": {"bootstrap": "-I -S; explicit installed wheel paths; no startup/site/.pth hooks",
             "sdk_native_file_lock_checked_per_child": True,
             "verification_cost": "charged inside process birth CPU/RSS and complete cell wall windows"},
        "semantic_geometry": geometry, "memory_admission": admissions, "required_host_bytes": required,
        "cohort_commitment": commitment(salt, {"prompt": "Explain sky." if args.tiny else PROMPT,
            "sampling": SCENARIOS, "replicates": args.replicates}),
        "results": [], "attempts": [],
        "windows": {"client": "process birth through response state hashing, client and disk-owner close",
            "provider_cpu": "birth through client exit; shutdown excluded",
            "provider_rss": "50ms sampled after startup through client exit; startup peak unknown",
            "all_link_tcp": "native topology/relay startup through role and relay shutdown",
            "covered_bodies": "client construction through close, v3 controls/ACK included",
            "disk": "owner bind through close, every encrypted read/write including cold publication and orphans",
            "request": "ordinary SDK request start through terminal stream; hashing and publication separately charged in lifecycle",
            "aggregate_cpu": "launcher plus all reaped client/provider children through complete cleanup"},
        "limits": ["no cross-salt cohort merging; no private output-dependent optimization",
            "source/canonical numeric and full-cap bounds identical; ordinary and fixed1 controls separate",
            "partial v3 partitions conserve known base + known output-dependent + residual",
            "no inferred avoided correction bytes or rows from cache hit labels",
            "publisher/tokenizer/checkpoint artifact distribution and disk unknown, separate",
            "fresh-interpreter encrypted restart tested by parent, not this cohort",
            "semantic body linear MAC denominator excludes attention, nonlinear work, head, masks and Preparation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(output, indent=2) + "\n")
        temp.replace(args.output)
    blocked = [key for key, value in admissions.items() if not value["admitted"]]
    if blocked or not boundary["structurally_ready"]:
        output["failure"] = {"stage": "preflight", "blocked_cells": blocked,
            "missing_api": boundary["missing"]}
    save()
    if args.preflight_only:
        print(json.dumps({"admitted": not blocked, "api_ready": boundary["structurally_ready"],
            "required_host_bytes": required, "planned_clients": output["planned_clients"]}))
        return
    if "failure" in output:
        raise RuntimeError("decode-cache preflight blocked; redacted report saved")
    active, guard, stage = None, None, "cohort_admission"
    try:
        with fresh_workers(Path(__file__).resolve(), memory_bytes=required, timeout=1800) as (run, guard, _):
            if run.build != build:
                raise RuntimeError("installed build changed before cohort")
            for replicate, key in cohort_order(list(experiments), args.replicates):
                scenario, policy = key.split("/")
                active = {"scenario": scenario, "policy": policy, "replicate": replicate, "status": "started"}
                output["attempts"].append(active)
                save()
                stage = "cell_execution"
                result = case(experiments[key], scenario, policy, replicate, salt, run, tiny=args.tiny)
                output["results"].append(result)
                save()
                stage = "cell_validation"
                if (result["client"].get("failure") or {}).get("error_type") in {"KeyboardInterrupt", "SystemExit"}:
                    raise KeyboardInterrupt()
                if hashes() != frozen:
                    raise RuntimeError("measurement sources changed")
                verify_results(output["results"], geometry)
                active["status"] = "completed"
                save()
            stage = "cohort_validation"
            output["checks"] = verify_results(output["results"], geometry, require_complete=True, replicates=args.replicates)
    except BaseException as error:
        if active is not None:
            active["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "rejected_or_failed"
        output["failure"] = {"stage": stage, "error_type": type(error).__name__}
        if isinstance(error, MemoryAdmissionRejected):
            output["failure"]["admission"] = error.report
        save()
        raise RuntimeError("incomplete decode-cache cohort; redacted progress saved") from None
    finally:
        if guard is not None:
            output["memory_guard"] = {"failed": guard.error is not None,
                "minimum_available_bytes": guard.minimum_available_bytes,
                "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes}
            save()
    output["complete"] = True
    save()
    print(json.dumps({"output": str(args.output), "checks": output["checks"]}))


if __name__ == "__main__":
    installed_bootstrap(__file__)
    main()
