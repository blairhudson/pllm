"""Frozen cap-64 growing-epoch cohort through fresh installed SDK interpreters.

No builds or installs. Private prompts, cohort salt and credentials go to children
on stdin. Evidence contains counters and salted complete-state commitments only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

from benchmark_epoch_prefix_sdk import (
    MemoryAdmissionRejected, admission,
)
from benchmark_conversation_restore import commitment, state_commitment
from frontier_probe_support import fresh_workers

ROOT = Path(__file__).resolve().parents[1]
LOW_BYTES = 268435456
_IMPLEMENTATION_LOCK = None
VARIANTS = {"control": {}, "fixed4": {"decode_epoch_tokens": 4},
            "fixed8": {"decode_epoch_tokens": 8},
            "growing": {"decode_epoch_schedule": (4, 60)}}
WORKLOADS = {
    "early": {"prompts": ["Reply with exactly: Hello world."], "cap": 64},
    "full": {"prompts": ["Explain how water moves from tree roots to leaves, then describe how dry weather changes that process. Give a detailed answer."], "cap": 64},
}


def _python_bootstrap(target, *, module=False):
    """No site startup/.pth/user-site hooks; explicitly add installed wheel paths."""
    paths = list(dict.fromkeys(sysconfig.get_paths()[name] for name in ("purelib", "platlib")))
    setup = ("import sys,runpy,importlib.util,pathlib;"
        f"sys.prefix={sys.prefix!r};sys.exec_prefix={sys.exec_prefix!r};"
        f"sys.path.extend({paths!r});"
        "spec=importlib.util.find_spec('pllm');"
        f"roots=[pathlib.Path(p).resolve() for p in {paths!r}];"
        "assert spec is not None and any(pathlib.Path(spec.origin).resolve().is_relative_to(p) for p in roots), 'installed SDK unavailable';")
    if _IMPLEMENTATION_LOCK is not None:
        sdk, native = _IMPLEMENTATION_LOCK
        setup += ("import pllm,hashlib,json;root=pathlib.Path(pllm.__file__).parent;"
            "sources={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'}};"
            f"assert hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest()=={sdk!r}, 'SDK implementation changed';"
            "native=importlib.util.find_spec('pllm._native');"
            f"assert hashlib.sha256(pathlib.Path(native.origin).read_bytes()).hexdigest()=={native!r}, 'native implementation changed';")
    if module:
        return setup + f"sys.argv=[{target!r},*sys.argv[1:]];runpy.run_module({target!r},run_name='__main__',alter_sys=True)"
    directory = str(Path(target).resolve().parent)
    return setup + (f"import pllm;sys.path.insert(0,{directory!r});"
        f"sys.argv=[{str(target)!r},*sys.argv[1:]];runpy.run_path({str(target)!r},run_name='__main__')")


def isolated_command(command):
    if not isinstance(command, (list, tuple)) or not command:
        return command
    if str(Path(command[0]).resolve()) != str(Path(sys.executable).resolve()):
        return command
    tail = list(command[1:])
    while tail and tail[0] in {"-I", "-S", "-s", "-E", "-u"}:
        tail.pop(0)
    if len(tail) >= 2 and tail[0] == "-m":
        return [command[0], "-I", "-S", "-c", _python_bootstrap(tail[1], module=True), *tail[2:]]
    if tail and Path(tail[0]).suffix == ".py":
        return [command[0], "-I", "-S", "-c", _python_bootstrap(tail[0]), *tail[1:]]
    raise RuntimeError("unrecognized Python child bootstrap")


def freeze_installed_implementation(build):
    from pllm.metrics.benchmark_runner import inspect_build
    from pllm.metrics import BuildRef
    observed = inspect_build(BuildRef(**build))
    if observed["digest"] != build["expected_digest"]:
        raise RuntimeError("installed implementation changed before freeze")
    attributes = observed["attributes"]
    global _IMPLEMENTATION_LOCK
    _IMPLEMENTATION_LOCK = (hashlib.sha256(json.dumps(attributes["sdk_sources"],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest(), attributes["native_sha256"])


def installed_bootstrap(script):
    """Script-only adapter; also isolates historical runner and SDK role children."""
    if not (sys.flags.isolated and sys.flags.no_site):
        os.execv(sys.executable, [sys.executable, "-I", "-S", "-c",
            _python_bootstrap(str(Path(script).resolve())), *sys.argv[1:]])
        raise RuntimeError("isolated bootstrap did not replace process")
    original = subprocess.Popen
    class InstalledPopen(original):
        def __init__(self, args, *positional, **kwargs):
            super().__init__(isolated_command(args), *positional, **kwargs)
    subprocess.Popen = InstalledPopen


def bounded_commitment(salt, value):
    stack, budget, nodes = [(value, 0)], 16 << 20, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 32768:
            raise RuntimeError("state witness structure exceeds bound")
        if isinstance(item, str):
            budget -= 6 * len(item) + 2
        elif isinstance(item, dict):
            if 2 * len(item) + len(stack) + nodes > 32768:
                raise RuntimeError("state witness structure exceeds bound")
            budget -= 2 + 2 * len(item)
            stack.extend((part, depth + 1) for pair in item.items() for part in pair)
        elif isinstance(item, (tuple, list)):
            if len(item) + len(stack) + nodes > 32768:
                raise RuntimeError("state witness structure exceeds bound")
            budget -= 2 + len(item)
            stack.extend((part, depth + 1) for part in item)
        elif item is None or type(item) in {int, float, bool}:
            budget -= 64
        else:
            raise RuntimeError("unsupported state witness value")
        if budget < 0:
            raise RuntimeError("state witness bytes exceed bound")
    return commitment(salt, value)


def complete_state_commitment(client, response, salt):
    saved = client._core._transformer_conversations[response.id]
    return {**state_commitment(client, response, salt),
        "history": bounded_commitment(salt, client._core.message_histories[response.id]),
        "rendered_history": bounded_commitment(salt, {
            "context": saved.rendered_context, "history": client._core.histories[response.id]}),
        "pending_count": len(saved.pending_token_ids),
        "rng": None, "rng_scope": "greedy fixture; RNG trajectory not asserted"}


def validate_evaluated_state(row, steps):
    expected = row["output_tokens"] - 1 if row["status"] == "incomplete" else row["output_tokens"]
    pending = int(row["status"] == "incomplete")
    if (type(steps) is not int or steps != expected
            or type(row["state"].get("pending_count")) is not int
            or type(row["state"].get("position")) is not int
            or row["state"].get("pending_count") != pending
            or row["state"].get("position") != row["input_tokens"] + expected):
        raise RuntimeError("evaluated position/pending/semantic step conservation failed")


def collect_response(client, prompt, cap, salt, index):
    from pllm.runtime.types import Response
    started, created, events, response = time.perf_counter(), None, [], None
    with client.tokenizer_scope(), client.responses.create(input=prompt,
            max_output_tokens=cap, temperature=0, store=True, stream=True) as stream:
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
            or response.status == "incomplete" and (response.incomplete_details or {}).get("reason") != "max_output_tokens"
            or not 0 < response.usage.output_tokens <= cap or len(events) != response.usage.output_tokens):
        raise RuntimeError("terminal response/event accounting failed")
    return {"index": index, "status": response.status,
        "state": complete_state_commitment(client, response, salt),
        "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
        "request_seconds": elapsed, "request_to_created_seconds": created - started,
        "first_text_seconds": events[0] - started, "created_to_first_text_seconds": events[0] - created,
        "decode_tps": (len(events) - 1) / (events[-1] - events[0]) if len(events) > 1 else None}


def experiment(base, workload, variant):
    from pllm.preparation import PreparedInventory
    from pllm.state import ClientPrefixReuse

    return base.with_params(
        pipeline__inventory=PreparedInventory("request-sized", rows=1, allocation="demand",
            refill="on-demand", stage_window=4, stage_window_bytes=64 << 20,
            **VARIANTS[variant]),
        pipeline__cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=128),
        budget__requests=1, budget__max_input_tokens=128,
        budget__max_new_tokens=workload["cap"])


def validate_audit(audit):
    from pllm.runtime.topology_accounting import ACK_COVERAGE, prepared_body_accounting

    rows = {name: audit.get("prepared_stage_rows_" + name)
            for name in ("issued", "reserved", "claimed", "burned", "discarded")}
    if (any(type(value) is not int or value < 0 for value in rows.values())
            or rows["issued"] != rows["reserved"] + rows["discarded"]
            or rows["reserved"] != rows["claimed"] + rows["burned"]):
        raise RuntimeError("retired lifetime issued/reserved/discarded row accounting failed")
    ledger = prepared_body_accounting({"privacy": audit})
    if (not ledger["tracked_body_counter_set_present"]
            or ledger["body_counter_coverage"] != ACK_COVERAGE
            or any(audit.get(key) != 0 for key in ("plaintext_prompt_bytes_sent",
                "plaintext_token_ids_sent", "preparation_requests_during_online", "inflight_joins"))
            or audit.get("prepared_stage_rows_reserved") != (
                audit.get("prepared_stage_rows_claimed", -1) + audit.get("prepared_stage_rows_burned", -1))):
        raise RuntimeError("v3 private boundary or one-use row accounting failed")
    part = ledger["traffic_attribution"]
    if (part["total_all_link_body_bytes"] != part["known_base_all_link_body_bytes"]
            + part["known_output_dependent_all_link_body_bytes"]
            + part["unattributed_all_link_body_bytes"]):
        raise RuntimeError("traffic partition conservation failed")
    return {**ledger, "measurement_window": "client_construction_through_close",
        "scope": "lifetime covered application bodies including cold bundle, preparation, ACK and retirement",
        "unmeasured": ["HTTP/TLS/WebSocket framing and uninstrumented endpoints",
            "failed transport attempts absent from audit counters",
            "checkpoint/tokenizer/publisher artifact distribution outside measured links", "physical wire"]}


def client_worker(payload):
    import pllm
    from pllm.runtime.benchmark_response_group import (
        inspect_private_cleanup, prefix_owners, require_concurrency_contract,
    )
    from pllm.runtime.topology_accounting import prepared_body_accounting

    started = time.perf_counter()
    client, owners, rows, failure, initialized = None, (), [], None, None
    try:
        client = pllm.OpenAI(experiment=pllm.Experiment.from_spec(payload["experiment"]),
            **payload["connection"], bundle_cache_dir=payload["cache"],
            background_inventory_refill=False, timeout=900)
        require_concurrency_contract(client)
        with client.tokenizer_scope():
            client.runtime.capabilities()
            client.prepared_rows_for_response(payload["prompts"][0], payload["cap"], store=True)
        initialized = time.perf_counter()
        rows = [collect_response(client, payload["prompts"][0], payload["cap"],
                                 bytes.fromhex(payload["salt"]), 0)]
    except BaseException as error:
        failure = {"stage": "client_execution", "error_type": type(error).__name__}
    finally:
        if client is not None:
            owners = prefix_owners(client)
            try:
                client.close()
            except BaseException as error:
                failure = {"stage": "client_retirement", "error_type": type(error).__name__}
    audit = client.privacy_audit.to_dict() if client is not None else {}
    cleanup = None
    try:
        if client is not None:
            cleanup = inspect_private_cleanup(client, owners)
            if client._core._prepared_queue.requests or client._core._online_active or client._core._preparation_active:
                raise RuntimeError("prepared execution owner did not retire")
        ledger = validate_audit(audit) if failure is None else prepared_body_accounting({"privacy": audit})
    except BaseException as error:
        failure = {"stage": "client_validation", "error_type": type(error).__name__}
        ledger = prepared_body_accounting({"privacy": audit})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"responses": rows, "audit": audit, "ledger": ledger, "cleanup": cleanup,
        "failure": failure, "client_lifetime_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "client_peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        "client_setup_seconds": initialized - started if initialized is not None else None,
        "client_window_seconds": time.perf_counter() - started,
        "publisher_artifact_distribution_bytes": None, "publisher_artifact_disk_bytes": None}


def case(exp, workload, name, variant, replicate, salt, run):
    """Role CPU birth-to-client-exit; RSS sampled after startup; streams through close."""
    import psutil
    from pllm.deployment import WanConditions
    from pllm.runtime.servers import build_roles

    memory = admission(exp)
    peak, stop, errors = {}, threading.Event(), []
    wall, cpu = time.perf_counter(), time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with tempfile.TemporaryDirectory(prefix="pllm-growing-epochs-") as root:
        roles = build_roles(exp, wan=WanConditions(), log_dir=Path(root) / "logs")
        with roles:
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
                try:
                    client = run({"experiment": exp.to_spec(), **workload, "salt": salt.hex(),
                        "connection": {"base_url": roles.inference_url,
                            "preparation_base_url": roles.preparation_url,
                            "api_key": roles._inference_key, "preparation_api_key": roles._preparation_key},
                        "cache": str(Path(root) / "client-cache")})
                except BaseException as error:
                    client = {"failure": {"stage": "worker_process", "error_type": type(error).__name__},
                        "responses": [], "audit": {}, "ledger": None,
                        "client_peak_rss_bytes": None, "client_lifetime_cpu_seconds": None}
                process_cpu = {}
                for role, process in processes.items():
                    used = process.cpu_times()
                    process_cpu[role] = used.user + used.system
            finally:
                stop.set()
                sampler.join()
        # Public resource_samples schedules on the relay event loop, now closed.
        # Read the same counter owner only after its thread has joined.
        network = roles._native_network.pacer.snapshot()
        if (network["total_stream_bytes"] != sum(network["directed_stream_bytes"].values())
                or network["conditions_digest"] != WanConditions().digest or not network["enforced"]):
            raise RuntimeError("native link accounting failed")
    if errors or set(peak) != set(processes):
        raise RuntimeError("provider resource sampling incomplete")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    launcher_cpu = time.process_time() - cpu
    return {"workload": name, "variant": variant, "replicate": replicate, "client": client,
        "provider_birth_cpu_seconds": process_cpu, "provider_sampled_peak_rss_bytes": peak,
        "aggregate_cleanup_cpu_seconds": launcher_cpu + child_cpu,
        "complete_case_seconds": time.perf_counter() - wall, "wan": network,
        "admitted_host_bytes": memory["required_bytes"]}


def work_counts(geometry, input_tokens, decode_steps):
    """Public scheduled body projection MACs, not whole CPU or attention FLOPs."""
    totals = {"body_macs": 0, "remote_body_macs": 0}
    for stage in geometry:
        rows = (1 if stage["terminal_prefill"] else input_tokens) + decode_steps
        macs = rows * stage["macs_per_row"]
        totals["body_macs"] += macs
        totals["remote_body_macs"] += macs if stage["remote"] else 0
    total = totals["body_macs"]
    return {**totals, "remote_body_linear_mac_fraction": totals["remote_body_macs"] / total if total else None,
        "remote_90_percent_verified": bool(total and totals["remote_body_macs"] * 10 >= total * 9),
        "denominator": "scheduled transformer-body dense linear MACs; excludes head, attention, normalization, masking and preparation",
        "whole_body_compute_fraction": None}


def semantic_geometry(exp, source):
    from pllm import lower_model
    from pllm.runtime.semantic_stages import scheduled_stage_specs, semantic_stage_role

    config = json.loads((source.path / "config.json").read_bytes())
    plan = lower_model(config, batch=1, max_input_tokens=128, max_new_tokens=exp.budget.max_new_tokens)
    schedule = plan.runtime_schedule(exp.pipeline).to_dict()
    stages = {(s.layer_index, s.role): s for s in scheduled_stage_specs(plan, exp.pipeline)
              if s.layer_index is not None}
    phases = {}
    for phase in ("prefill", "decode"):
        operations = {op["id"]: op for op in getattr(plan, phase)["operations"]}
        phases[phase] = {(step["layer"], semantic_stage_role(step, operations)): step
            for step in schedule[phase]["steps"] if step["weight_ids"] and step["layer"] is not None
            and step["executor"] in {"remote_stage", "verified_remote_stage", "client_linear"}}
    if set(phases["prefill"]) != set(stages) or set(phases["decode"]) != set(stages) or not stages:
        raise RuntimeError("semantic body geometry incomplete")
    rows = []
    for key, stage in sorted(stages.items()):
        pre, dec = phases["prefill"][key], phases["decode"][key]
        if pre["executor"] != dec["executor"] or pre["executor"] not in {"remote_stage", "client_linear"}:
            raise RuntimeError("semantic body ownership unsupported")
        rows.append({"stage": stage.id, "macs_per_row": stage.in_features * stage.out_features,
            "remote": pre["executor"] == "remote_stage",
            "terminal_prefill": bool(pre.get("terminal_row_only", False))})
    return {"plan_digest": plan.digest, "schedule_digest": hashlib.sha256(
        json.dumps(schedule, sort_keys=True).encode()).hexdigest(), "stages": rows}


def verify_results(results, *, tiny=False):
    references = {}
    for result in results:
        client, name = result["client"], result["workload"]
        if client.get("failure"):
            raise RuntimeError("client attempt failed; bounded counters retained")
        if len(client["responses"]) != 1:
            raise RuntimeError("one cold response required")
        row = client["responses"][0]
        state = {key: row[key] for key in ("status", "state", "input_tokens", "output_tokens")}
        references.setdefault(name, state)
        if state != references[name]:
            raise RuntimeError("complete logits/KV/history/pending/output parity failed")
        if row["input_tokens"] + (8 if tiny else 64) - 1 > 128:
            raise RuntimeError("full allowance exceeds continuation bound")
        if not tiny:
            if name == "early" and not (row["status"] == "completed" and 0 < row["output_tokens"] < 64):
                raise RuntimeError("declared natural EOS not observed")
            if name == "full" and not (row["status"] == "incomplete" and row["output_tokens"] == 64):
                raise RuntimeError("declared full cap not observed")
        validate_evaluated_state(row, client["audit"].get("generation_executed_decode_steps"))
        validate_audit(client["audit"])
    peaks = [r["client"]["client_peak_rss_bytes"] for r in results]
    return {"exact_state_and_output_match": True, "checked_response_states": len(results),
        "workload_lifecycles_checked": not tiny,
        "largest_client_lifetime_peak_rss_bytes": max(peaks) if peaks else None,
        "low_rss_gate": max(peaks) <= LOW_BYTES if peaks else None,
        "low_threshold_bytes": LOW_BYTES,
        "admission_scope": "full-cap shared topology, retained state, caller metadata, launcher and relays; not measured RSS"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tiny", "--smoke", action="store_true")
    parser.add_argument("--replicates", "--processes", type=int, default=3)
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
    workloads = {name: dict(WORKLOADS[name]) for name in dict.fromkeys(args.workload or WORKLOADS)}
    if args.tiny:
        workloads = {name: {"prompts": ["Explain sky."], "cap": 8} for name in workloads}
    experiments = {name + "/" + variant: experiment(base, work, variant)
                   for name, work in workloads.items() for variant in VARIANTS}
    admissions = {key: admission(exp, require=False) for key, exp in experiments.items()}
    required = max(value["required_bytes"] for value in admissions.values())
    source = resolve_model(base.pipeline.model)
    files = [Path(__file__), ROOT / "scripts/benchmark_epoch_prefix_sdk.py",
        ROOT / "scripts/frontier_probe_support.py", ROOT / "scripts/benchmark_conversation_restore.py",
        ROOT / "examples/prepared_sdk.py", ROOT / "examples/benchmarks/growing_epochs_sdk.py",
        ROOT / "tests/test_benchmark_growing_epochs_sdk.py",
        ROOT / "docs/research/growing-epochs-sdk-reference.md"]
    def hashes():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    frozen, salt = hashes(), os.urandom(32)
    build = BuildRef.capture().to_spec()
    freeze_installed_implementation(build)
    geometries = {key: semantic_geometry(exp, source) for key, exp in experiments.items()}
    output = {"schema": "pllm.growing_epochs_sdk_benchmark.v1", "complete": False,
        "tiny": args.tiny, "planned_cells": len(experiments) * args.replicates,
        "experiments": {key: exp.to_spec() for key, exp in experiments.items()},
        "config_hashes": {key: hashlib.sha256(exp.canonical_bytes()).hexdigest() for key, exp in experiments.items()},
        "memory_admission": admissions, "required_host_bytes": required,
        "source_lock_digest": source.source_lock_digest, "frozen_sources": frozen,
        "build": build, "semantic_geometry": geometries,
        "execution_isolation": {"bootstrap": "-I -S; explicit installed wheel paths; no startup/site/.pth hooks",
            "sdk_native_file_lock_checked_per_child": True,
            "verification_cost": "charged inside process birth CPU/RSS and complete cell wall windows"},
        "cohort_commitment": commitment(salt, {"workloads": workloads, "replicates": args.replicates}),
        "results": [], "attempts": [],
        "windows": {"client_cpu_rss": "process birth through close and state hashing",
            "provider_cpu": "process birth through client exit; excludes role shutdown",
            "provider_rss": "50 ms samples after startup through client exit; startup peak unknown",
            "tcp_streams": "native relays from topology startup through provider/network close",
            "covered_bodies": "client construction through close, v3 counters only",
            "aggregate_cpu": "complete cell including launcher, relays and reaped client/role children",
            "ttft": "request start through first text; all admission/preparation waiting charged",
            "decode_tps": "N-1 / first-to-last text delta; all epoch pauses charged"},
        "limits": ["no private-output policy optimization or cross-salt pooling",
            "cold one-response client: no generated-prefix reuse savings claimed",
            "full-cap geometry remains admitted, including unused issued/reserved rows",
            "v3 base/output attribution may be partial; residual conserves totals",
            "publisher/checkpoint/tokenizer distribution and disk separate and unknown",
            "remote body gate means scheduled body projection MACs, not whole CPU or all operator FLOPs",
            "physical wire, Internet RTT and independent launcher peak RSS unknown"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2) + "\n")
        temporary.replace(args.output)
    blocked = [key for key, report in admissions.items() if not report["admitted"]]
    if blocked:
        output["failure"] = {"stage": "memory_admission", "blocked_cells": blocked}
    save()
    if args.preflight_only:
        print(json.dumps({"admitted": not blocked, "required_host_bytes": required,
                          "cells": output["planned_cells"]}))
        return
    if blocked:
        raise RuntimeError("memory admission rejected; bounded report saved")
    guard, active, stage = None, None, "cohort_admission"
    try:
        with fresh_workers(Path(__file__).resolve(), memory_bytes=required, timeout=1800) as (run, guard, _):
            if run.build != build:
                raise RuntimeError("installed build changed before cohort")
            for replicate in range(args.replicates):
                keys = list(experiments) if replicate % 2 == 0 else list(reversed(experiments))
                for key in keys:
                    name, variant = key.split("/")
                    active = {"workload": name, "variant": variant, "replicate": replicate, "status": "started"}
                    output["attempts"].append(active)
                    save()
                    stage = "cell_execution"
                    result = case(experiments[key], workloads[name], name, variant, replicate, salt, run)
                    output["results"].append(result)
                    stage = "cell_validation"
                    worker_failure = result["client"].get("failure") or {}
                    if worker_failure.get("error_type") in {"KeyboardInterrupt", "SystemExit"}:
                        save()
                        raise KeyboardInterrupt()
                    if hashes() != frozen:
                        raise RuntimeError("benchmark sources changed during cohort")
                    if not result["client"].get("failure"):
                        row = result["client"]["responses"][0]
                        steps = result["client"]["audit"].get("generation_executed_decode_steps")
                        validate_evaluated_state(row, steps)
                        result["semantic_work"] = work_counts(geometries[key]["stages"], row["input_tokens"], steps)
                        if not result["semantic_work"]["remote_90_percent_verified"]:
                            raise RuntimeError("remote body projection MAC gate failed")
                    save()
                    verify_results(output["results"], tiny=args.tiny)
                    active["status"] = "completed"
                    save()
            stage = "cohort_validation"
            output["checks"] = verify_results(output["results"], tiny=args.tiny)
            output["checks"]["remote_90_percent_verified_all_cells"] = all(
                r["semantic_work"]["remote_90_percent_verified"] for r in output["results"])
    except BaseException as error:
        if active is not None:
            active["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "rejected_or_failed"
        output["failure"] = {"stage": stage, "error_type": type(error).__name__}
        if isinstance(error, MemoryAdmissionRejected):
            output["failure"]["admission"] = error.report
        save()
        raise RuntimeError("incomplete cohort; redacted progress saved") from None
    finally:
        if guard is not None:
            output["memory_guard"] = {"failed": guard.error is not None,
                "minimum_available_bytes": guard.minimum_available_bytes,
                "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes}
            save()
    output["complete"] = True
    save()
    print(json.dumps({"output": str(args.output), "cases": len(output["results"]), "checks": output["checks"]}))


if __name__ == "__main__":
    installed_bootstrap(__file__)
    main()
