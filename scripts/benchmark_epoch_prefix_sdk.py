"""Matched installed-SDK epoch/pending-prefix responses under native link caps.

Fresh client and provider processes/cache for every cell. Private inputs, salts
and credentials travel over stdin; evidence contains salted state commitments.
Ordinary streaming SDK calls execute natural EOS and the requested output cap.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import tempfile
import threading
import time

from benchmark_conversation_restore import commitment, state_commitment
from frontier_probe_support import fresh_workers


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "early": {"prompts": ["Reply with exactly: Hello world."], "cap": 64},
    "full": {"prompts": ["Explain how water moves from tree roots to leaves, then describe how dry weather changes that process. Give a detailed answer."], "cap": 8},
    "shared": {"prompts": ["Explain how water moves from tree roots to leaves, then describe how dry weather changes that process. Give a detailed answer."] * 2, "cap": 8},
    "unrelated": {"prompts": ["Explain how water moves from tree roots to leaves, then describe how dry weather changes that process. Give a detailed answer.",
                             "Explain how bees find flowers, then describe how their colony stores food for winter. Give a detailed answer."], "cap": 8},
}
VARIANTS = {"control": (None, False), "epochs": (4, False),
             "inflight": (None, True), "combined": (4, True)}


class MemoryAdmissionRejected(RuntimeError):
    """Retain bounded admission evidence without serializing worker errors."""

    def __init__(self, report):
        self.report = report
        super().__init__("SDK group memory admission rejected: " + json.dumps({
            "required_host_bytes": report["required_bytes"],
            "admission_budget_bytes": report["topology"]["host"]["admission_budget_bytes"],
            "reasons": report["reasons"]}))


def experiment(base, workload, variant):
    from pllm.preparation import PreparedInventory
    from pllm.state import ClientPrefixReuse

    span, sharing = VARIANTS[variant]
    # The continuation bound includes executed prompt and decode history. The
    # 64-output early-EOS workload needs 128; short-output grouped cells use 64.
    bound = 128 if workload["cap"] > 8 else 64
    return base.with_params(
        pipeline__inventory=PreparedInventory("request-sized", rows=1, allocation="demand",
            refill="on-demand", stage_window=4, stage_window_bytes=64 << 20,
            decode_epoch_tokens=span),
        pipeline__cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=bound,
            inflight=sharing, **({"inflight_max_requests": 2, "inflight_max_bytes": 32 << 20}
                                if sharing else {})),
        budget__requests=len(workload["prompts"]), budget__max_input_tokens=bound,
        budget__max_new_tokens=workload["cap"])


def collect_response(client, prompt, cap, salt, index):
    """Read only public events, then hash the already-retained exact state."""
    from pllm.runtime.types import Response

    started = time.perf_counter()
    created = None
    events, response = [], None
    with client.tokenizer_scope(), client.responses.create(
            input=prompt, max_output_tokens=cap, temperature=0, store=True, stream=True) as stream:
        for event in stream:
            now = time.perf_counter()
            if event.type == "response.created":
                created = now
            elif event.type == "response.output_text.delta":
                events.append(now)
            elif event.type in {"response.completed", "response.incomplete"}:
                response = Response.from_dict(event.response) if isinstance(event.response, dict) else event.response
    elapsed = time.perf_counter() - started
    if (response is None or response.usage is None or created is None
            or response.status not in {"completed", "incomplete"}
            or (response.status == "incomplete" and (response.incomplete_details or {}).get(
                "reason") != "max_output_tokens")
            or not 0 < response.usage.output_tokens <= cap
            or len(events) != response.usage.output_tokens):
        raise RuntimeError("terminal response/event accounting failed")
    return {"index": index, "status": response.status, "state": state_commitment(client, response, salt),
        "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
        "request_seconds": elapsed, "request_to_created_seconds": created - started,
        "first_text_seconds": events[0] - started,
        "created_to_first_text_seconds": events[0] - created,
        "decode_tps": (len(events) - 1) / (events[-1] - events[0]) if len(events) > 1 else None}


def validate_audit(audit):
    from pllm.runtime.topology_accounting import CONTROL_COVERAGE, prepared_body_accounting

    ledger = prepared_body_accounting({"privacy": audit})
    if (not ledger["tracked_body_counter_set_present"] or ledger["body_counter_coverage"] != CONTROL_COVERAGE
            or any(audit[name] != 0 for name in (
                "plaintext_prompt_bytes_sent", "plaintext_token_ids_sent", "preparation_requests_during_online"))
            or audit["prepared_stage_rows_reserved"] != (
                audit["prepared_stage_rows_claimed"] + audit["prepared_stage_rows_burned"])):
        raise RuntimeError("private boundary or one-use row accounting failed")
    # The ordinary ledger receives the lifetime audit here, including bundle and
    # shutdown. Explicitly bind that wider window instead of inheriting its label.
    return {**ledger, "measurement_window": "client_construction_through_close",
        "scope": "lifetime covered application bodies, including bundle, preparation and lifecycle controls",
        "window_includes_bundle_and_preparation": True,
        "unmeasured": ["HTTP/TLS/WebSocket framing and uninstrumented control endpoints",
            "correction acknowledgements and retries absent from client audit",
            "checkpoint and public tokenizer artifact distribution", "physical wire"]}


def client_worker(p):
    import pllm
    from pllm.runtime.benchmark_response_group import (
        inspect_private_cleanup, prefix_owners, require_concurrency_contract,
    )

    exp = pllm.Experiment.from_spec(p["experiment"])
    salt = bytes.fromhex(p["salt"])
    started = time.perf_counter()
    client = pllm.OpenAI(experiment=exp, **p["connection"], bundle_cache_dir=p["cache"],
                         background_inventory_refill=False, timeout=900)
    owners = ()
    try:
        require_concurrency_contract(client)
        with client.tokenizer_scope():
            client.runtime.capabilities()
            for prompt in p["prompts"]:
                client.prepared_rows_for_response(prompt, p["cap"], store=True)
        initialized = time.perf_counter()
        barrier = threading.Barrier(len(p["prompts"]))

        def branch(index):
            barrier.wait(timeout=60)
            return collect_response(client, p["prompts"][index], p["cap"], salt, index)

        with ThreadPoolExecutor(max_workers=len(p["prompts"])) as pool:
            futures = [pool.submit(branch, i) for i in range(len(p["prompts"]))]
            rows = [future.result() for future in futures]
        group_seconds = time.perf_counter() - initialized
    finally:
        owners = prefix_owners(client)
        client.close()
    audit = client.privacy_audit.to_dict()
    cleanup = inspect_private_cleanup(client, owners)
    if client._core._prepared_queue.requests or client._core._online_active or client._core._preparation_active:
        raise RuntimeError("prepared execution owner did not retire")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"responses": rows, "audit": audit, "ledger": validate_audit(audit), "cleanup": cleanup,
        "client_lifetime_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "client_peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        "client_setup_seconds": initialized - started, "group_seconds": group_seconds,
        "client_window_seconds": time.perf_counter() - started}


def admission(exp, *, require=True):
    from pllm.metrics import benchmark_memory
    from pllm.runtime.benchmark_response_group import group_memory_contract

    topology = benchmark_memory(exp, backend="native", enforced_wan=True)
    group = group_memory_contract(topology, requests=exp.budget.requests,
                                  max_output_tokens=exp.budget.max_new_tokens)
    # The group estimator prices client/roles/retained snapshots. This separate
    # allowance covers this launcher, sampler and opaque native pacing relays.
    required = group["required_bytes"] + (256 << 20)
    reasons = list(topology["candidates"]["native"]["reasons"])
    if required > topology["host"]["admission_budget_bytes"]:
        reasons.append("complete group and launcher exceed post-reserve headroom")
    result = {"topology": topology, "group": group, "launcher_extra_bytes": 256 << 20,
              "required_bytes": required, "admitted": topology["admitted"] and not reasons,
              "reasons": reasons}
    if require and not result["admitted"]:
        raise MemoryAdmissionRejected(result)
    return result


def case(exp, workload, name, variant, replicate, salt, run):
    import psutil
    from pllm.deployment import WanConditions
    from pllm.runtime.servers import build_roles

    memory = admission(exp)
    peak, stop, errors = {}, threading.Event(), []
    with tempfile.TemporaryDirectory(prefix="pllm-epoch-prefix-sdk-") as root:
        wall, cpu = time.perf_counter(), time.process_time()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        with build_roles(exp, wan=WanConditions(), log_dir=Path(root) / "logs") as roles:
            processes = {status.role: psutil.Process(status.pid) for status in roles.statuses}

            def sample():
                try:
                    while not stop.is_set():
                        for role, process in processes.items():
                            peak[role] = max(peak.get(role, 0), process.memory_info().rss)
                        stop.wait(0.05)
                except psutil.Error as error:
                    errors.append(type(error).__name__)

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            try:
                connection = {"base_url": roles.inference_url, "preparation_base_url": roles.preparation_url,
                    "api_key": roles._inference_key, "preparation_api_key": roles._preparation_key}
                client = run({"experiment": exp.to_spec(), **workload, "salt": salt.hex(),
                    "connection": connection, "cache": str(Path(root) / "client-cache")})
                process_cpu = {}
                for role, process in processes.items():
                    used = process.cpu_times()
                    process_cpu[role] = used.user + used.system
                network = roles.resource_samples()["wan_emulation"]
                if (network["total_stream_bytes"] != sum(network["directed_stream_bytes"].values())
                        or network["conditions_digest"] != WanConditions().digest or not network["enforced"]):
                    raise RuntimeError("native link accounting failed")
            finally:
                stop.set()
                sampler.join()
        if errors or set(peak) != set(processes):
            raise RuntimeError("provider resource sampling incomplete")
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    launcher_cpu = time.process_time() - cpu
    return {"workload": name, "variant": variant, "replicate": replicate, "client": client,
        "provider_birth_cpu_seconds": process_cpu, "provider_sampled_peak_rss_bytes": peak,
        "launcher_cpu_seconds": launcher_cpu, "reaped_children_cpu_seconds": child_cpu,
        "aggregate_cleanup_cpu_seconds": launcher_cpu + child_cpu,
        "complete_case_seconds": time.perf_counter() - wall, "wan": network,
        "admitted_host_bytes": memory["required_bytes"]}


def verify_results(results, *, tiny=False):
    """Never rank text matches in place of complete state and lifecycle checks."""
    reference, matches = {}, []
    for result in results:
        name, variant = result["workload"], result["variant"]
        client = result["client"]
        state = [{key: row[key] for key in ("index", "status", "state", "input_tokens", "output_tokens")}
                 for row in client["responses"]]
        reference.setdefault(name, state)
        matches.append(state == reference[name])
        audit = client["audit"]
        if name in {"shared", "unrelated"}:
            expected_joins = 1 if name == "shared" and VARIANTS[variant][1] else 0
            if audit["inflight_joins"] != expected_joins:
                raise RuntimeError("pending-prefix workload was not exercised as declared")
        if not tiny:
            cap = WORKLOADS[name]["cap"]
            if name == "early":
                if any(row["status"] != "completed" or not row["output_tokens"] < cap for row in state):
                    raise RuntimeError("declared natural-EOS workload did not finish early")
            elif any(row["status"] != "incomplete" or row["output_tokens"] != cap for row in state):
                raise RuntimeError("declared full-cap workload did not use its complete allowance")
    if not all(matches):
        raise RuntimeError("complete logits/KV/history/pending/output parity failed")
    return {"exact_state_and_output_match": True, "checked_response_states": sum(
        len(r["client"]["responses"]) for r in results), "workload_lifecycles_checked": not tiny,
        "pending_joins_checked": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--processes", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--workload", choices=tuple(WORKLOADS), action="append")
    parser.add_argument("--artifacts", type=Path, default=Path("target/prepared-frontier-artifacts"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(client_worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("supply a new --output")
    import pllm
    from pllm.model_loader import resolve_model

    factory = runpy.run_path(str(ROOT / "examples/prepared_sdk.py"))["build_experiment"]
    if args.tiny:
        base = factory()
    else:
        source = pllm.Experiment.from_spec(json.loads((args.artifacts / "indexed.json").read_bytes()))
        base = factory(source.pipeline.model, tokenizer=source.pipeline.components["tokenizer"])
    workloads = {name: dict(WORKLOADS[name]) for name in (args.workload or WORKLOADS)}
    if args.tiny:
        for name, work in workloads.items():
            work["cap"] = 8
            work["prompts"] = (["Explain sky.", "Explain sea."] if name == "unrelated"
                               else ["Explain sky."] * len(work["prompts"]))
    experiments, admissions = {}, {}
    for name, workload in workloads.items():
        for variant in (VARIANTS if len(workload["prompts"]) > 1 else ("control", "epochs")):
            key = name + "/" + variant
            experiments[key] = experiment(base, workload, variant)
            admissions[key] = admission(experiments[key], require=False)
    required = max(value["required_bytes"] for value in admissions.values())
    source = resolve_model(base.pipeline.model)
    files = [Path(__file__), Path(__file__).with_name("frontier_probe_support.py"),
             Path(__file__).with_name("benchmark_conversation_restore.py"), ROOT / "examples/prepared_sdk.py",
             ROOT / "tests/test_epoch_prefix_response_benchmark.py"]
    hashes = lambda: {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    frozen, salt = hashes(), os.urandom(32)
    output = {"schema": "pllm.epoch_prefix_sdk_benchmark.v1", "complete": False,
        "experiments": {key: exp.to_spec() for key, exp in experiments.items()},
        "memory_admission": admissions, "required_host_bytes": required,
        "source_lock_digest": source.source_lock_digest, "frozen_sources": frozen,
        "workload_commitment": commitment(salt, workloads), "results": [],
        "scope": ["fresh installed SDK client, fresh separate providers and bundle cache for every cell",
            "shared native 100/40 Mbps access caps; no added RTT; opaque TCP-stream counts, not physical wire",
            "cached checkpoint and pre-positioned public tokenizer; distribution excluded",
            "complete cell includes provider startup, client import/setup, requests, exact state audit and cleanup",
            "aggregate CPU includes launcher/relays and all reaped client/provider children through cleanup",
            "provider CPU from birth through client exit; provider RSS sampled every 50 ms after startup",
            "client lifetime peak includes state audits; no summed independent RSS peaks or whole-device memory",
            "request-to-created includes queue/prefix wait, setup and admission; isolated queue wait unmeasured",
            "first text includes all waiting; N-1 text-delta throughput includes epoch pauses",
            "barrier-aligned callers; natural pending joins required, no forced prefill barriers",
            "full logits, active KV, executed history, pending token and output use private salted commitments"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2) + "\n")
        temporary.replace(args.output)

    blocked = [key for key, value in admissions.items() if not value["admitted"]]
    if blocked:
        output["failure"] = {"stage": "memory_admission", "blocked_cells": blocked}
    if args.preflight_only or blocked:
        from pllm.metrics import BuildRef
        output["build"] = BuildRef.capture().to_spec()
    save()
    if args.preflight_only:
        print(json.dumps({"admitted": not blocked, "required_host_bytes": required,
                          "cases": len(experiments)}))
        return
    if blocked:
        raise RuntimeError("SDK group memory admission rejected; bounded details saved to " + str(args.output))
    active_cell, guard, stage = None, None, "cohort_admission"
    try:
        with fresh_workers(Path(__file__).resolve(), memory_bytes=required, timeout=1200) as (run, guard, _):
            output["build"] = run.build
            save()
            for replicate in range(args.processes):
                keys = list(experiments) if replicate % 2 == 0 else list(reversed(experiments))
                for key in keys:
                    name, variant = key.split("/")
                    active_cell = {"workload": name, "variant": variant, "replicate": replicate}
                    stage = "cell_execution"
                    print(json.dumps({"starting": key, "replicate": replicate}), flush=True)
                    result = case(experiments[key], workloads[name], name, variant, replicate, salt, run)
                    stage = "cell_validation"
                    if hashes() != frozen:
                        raise RuntimeError("benchmark sources changed during cohort")
                    output["results"].append(result)
                    output["memory_guard"] = {"failed": guard.error is not None,
                        "minimum_available_bytes": guard.minimum_available_bytes,
                        "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes}
                    save()
                    verify_results(output["results"], tiny=args.tiny)
            stage = "cohort_validation"
            output["checks"] = verify_results(output["results"], tiny=args.tiny)
    except BaseException as error:
        output["failure"] = {"stage": "memory_admission" if isinstance(error, MemoryAdmissionRejected) else stage,
            "cell": active_cell, "error_type": type(error).__name__}
        if isinstance(error, MemoryAdmissionRejected):
            output["failure"]["admission"] = error.report
        if guard is not None:
            output["memory_guard"] = {"failed": guard.error is not None,
                "minimum_available_bytes": guard.minimum_available_bytes,
                "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes}
        save()
        raise
    output["complete"] = True
    save()
    print(json.dumps({"output": str(args.output), "cases": len(output["results"]), "checks": output["checks"]}))


if __name__ == "__main__":
    main()
