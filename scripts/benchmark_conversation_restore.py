"""Match RAM continuation, fresh-client replay and encrypted SDK restoration.

Three turns per cell; fresh providers/cache per cell and fresh interpreters at
each restart. Private history, keys and trusted heads travel only over stdin or
the launcher's private pipes. The saved evidence contains salted commitments.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import tempfile
import threading
import time

from frontier_probe_support import fresh_workers


def commitment(salt, value):
    return hmac.new(salt, json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()


def state_commitment(client, response, salt):
    import numpy as np
    state = client._core._transformer_conversations[response.id]
    digest = hmac.new(salt, digestmod=hashlib.sha256)
    arrays = [state.next_logits]
    for cache in state.snapshot.caches:
        arrays.extend((cache.key[:cache.length], cache.value[:cache.length]))
    for array in arrays:
        digest.update(json.dumps(array.shape).encode())
        for row in np.atleast_2d(array):
            digest.update(np.ascontiguousarray(row).view(np.uint8))
    return {"position": state.snapshot.position, "words": sum(a.size for a in arrays),
            "bits": digest.hexdigest(), "tokens": commitment(salt, state.token_ids),
            "pending": commitment(salt, state.pending_token_ids),
            "output": commitment(salt, response.output_text)}


def client_worker(p):
    import pllm
    from pllm.state import ConversationCheckpoint
    from pllm.runtime.conversation_checkpoint import _Storage
    from pllm.runtime.topology_accounting import prepared_body_accounting

    storage = {"read_bytes": 0, "written_bytes": 0}
    close_storage = _Storage.close

    def measured_close(owner, runtime):
        storage["read_bytes"] += owner.read_bytes
        storage["written_bytes"] += owner.written_bytes
        return close_storage(owner, runtime)

    _Storage.close = measured_close
    salt, key = bytes.fromhex(p["salt"]), bytes.fromhex(p["key"])
    experiment = pllm.Experiment.from_spec(p["experiment"])
    started = time.perf_counter()
    with pllm.OpenAI(experiment=experiment, **p["connection"],
        bundle_cache_dir=p["cache"], background_inventory_refill=False) as client:
        rows = []
        private = p.get("private", {})
        with client.tokenizer_scope(), client.conversation(max_state_bytes=32 << 20,
            max_output_tokens=p["output_tokens"], temperature=0) as chat:
            restore_wall = restore_cpu = 0.0
            if p["mode"] == "restore" and private:
                wall, cpu = time.perf_counter(), time.process_time()
                chat.restore(private["directory"], key, ConversationCheckpoint.from_spec(private["head"]))
                restore_wall, restore_cpu = time.perf_counter() - wall, time.process_time() - cpu
            for index in p["indices"]:
                text = p["prompts"][index]
                value = [*private["messages"], {"role": "user", "content": text}] if p["mode"] == "replay" and private else text
                event_times = []
                original = client._core.events

                def timed_events(body):
                    for event in original(body):
                        if event.type == "response.output_text.delta":
                            event_times.append(time.perf_counter())
                        yield event

                client._core.events = timed_events
                before = client.privacy_audit.to_dict()
                wall, cpu = time.perf_counter(), time.process_time()
                try:
                    response = chat.send(value)
                finally:
                    client._core.events = original
                elapsed, used_cpu = time.perf_counter() - wall, time.process_time() - cpu
                after = client.privacy_audit.to_dict()
                delta = {k: value - before[k] for k, value in after.items()
                         if type(value) is int and type(before.get(k)) is int}
                if len(event_times) != response.usage.output_tokens:
                    raise RuntimeError("text-delta events do not cover every generated token")
                row = {"turn": index, "state": state_commitment(client, response, salt),
                    "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
                    "request_seconds": elapsed, "request_cpu_seconds": used_cpu,
                    "first_text_seconds": event_times[0] - wall if event_times else None,
                    "decode_tps": ((response.usage.output_tokens - 1) / (event_times[-1] - event_times[0])
                        if len(event_times) > 1 and response.usage.output_tokens > 1 else None),
                    "retained_state_bytes": chat.state_bytes,
                    "request_ledger": prepared_body_accounting({"privacy": delta}),
                    "fresh_correction_rows": delta["prepared_stage_rows_issued"],
                    "continuation_hits": delta["kv_continuation_batched_hits"]}
                private = {"messages": client._core.message_histories[response.id]}
                if p["mode"] == "restore" and index + 1 < len(p["prompts"]):
                    directory = str(Path(p["checkpoint_root"]) / f"turn-{index}")
                    wall, cpu = time.perf_counter(), time.process_time()
                    head = chat.checkpoint(directory, key)
                    row.update(publication_seconds=time.perf_counter() - wall,
                               publication_cpu_seconds=time.process_time() - cpu)
                    row["checkpoint_bytes"] = sum(f.stat().st_size for f in Path(directory).iterdir() if f.is_file())
                    private.update(directory=directory, head=head.to_spec())
                rows.append(row)
        audit = client.privacy_audit.to_dict()
        ledger = prepared_body_accounting({"privacy": audit})
        if (not ledger["tracked_body_counter_set_present"] or audit["plaintext_prompt_bytes_sent"]
                or audit["plaintext_token_ids_sent"] or audit["preparation_requests_during_online"]):
            raise RuntimeError("conversation accounting or private-boundary check failed")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"turns": rows, "audit": audit, "ledger": ledger, "restore_seconds": restore_wall,
        "restore_cpu_seconds": restore_cpu, "client_lifetime_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "client_peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        "client_window_seconds": time.perf_counter() - started, "storage": storage, "private": private}


def case(p, run):
    import pllm
    import psutil
    from pllm.deployment import WanConditions
    from pllm.metrics import benchmark_memory
    from pllm.runtime.servers import build_roles

    experiment = pllm.Experiment.from_spec(p["experiment"])
    memory = benchmark_memory(experiment, backend="native", enforced_wan=True)
    if not memory["admitted"]:
        raise RuntimeError("conversation topology failed memory admission")
    peak, stop, errors = {}, threading.Event(), []
    with tempfile.TemporaryDirectory(prefix="pllm-conversation-benchmark-") as root:
        (Path(root) / "checkpoints").mkdir(mode=0o700)
        wall, cpu = time.perf_counter(), time.process_time()
        with build_roles(experiment, wan=WanConditions(), log_dir=Path(root) / "logs") as roles:
            processes = {s.role: psutil.Process(s.pid) for s in roles.statuses}

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
            clients, private = [], {}
            connection = {"base_url": roles.inference_url, "preparation_base_url": roles.preparation_url,
                          "api_key": roles._inference_key, "preparation_api_key": roles._preparation_key}
            try:
                groups = [list(range(len(p["prompts"])))] if p["mode"] == "ram" else [[i] for i in range(len(p["prompts"]))]
                for indices in groups:
                    payload = {**p, "indices": indices, "private": private, "cache": str(Path(root) / "cache"),
                               "checkpoint_root": str(Path(root) / "checkpoints")}
                    result = run({**payload, "connection": connection})
                    private = result.pop("private")
                    clients.append(result)
                process_cpu = {role: process.cpu_times().user + process.cpu_times().system
                               for role, process in processes.items()}
                network = roles.resource_samples()["wan_emulation"]
            finally:
                stop.set()
                sampler.join()
        if errors:
            raise RuntimeError("provider resource sampling incomplete")
    return {"mode": p["mode"], "replicate": p["replicate"], "clients": clients,
        "provider_birth_cpu_seconds": process_cpu, "provider_sampled_peak_rss_bytes": peak,
        "launcher_cpu_seconds": time.process_time() - cpu,
        "launcher_peak_rss_bytes": None,
        "complete_case_seconds": time.perf_counter() - wall, "wan": network,
        "admitted_host_bytes": memory["candidates"]["native"]["required_host_bytes"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(client_worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists() or not 1 <= args.processes <= 3:
        parser.error("supply a new --output and 1–3 --processes")
    import pllm
    from pllm.metrics import benchmark_memory
    from pllm.model_loader import resolve_model
    if args.experiment:
        experiment = pllm.Experiment.from_spec(json.loads(args.experiment.read_bytes()))
    else:
        factory = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/prepared_sdk.py"))["build_experiment"]
        base = pllm.Experiment.from_spec(json.loads(Path("target/prepared-frontier-artifacts/indexed.json").read_bytes()))
        experiment = factory(base.pipeline.model, tokenizer=base.pipeline.components["tokenizer"])
    experiment = experiment.with_params(budget__requests=3)
    prompts = json.loads(args.prompts.read_bytes()) if args.prompts else [
        "Explain how a tree gets water from its roots to its leaves. Keep the explanation clear and concrete.",
        "How does dry weather change that process?", "What changes during the winter?"]
    if len(prompts) != 3 or any(type(p) is not str or not 1 <= len(p) <= 8192 for p in prompts):
        parser.error("expected three bounded text turns")
    source = resolve_model(experiment.pipeline.model)
    memory = benchmark_memory(experiment, backend="native", enforced_wan=True)
    if not memory["admitted"]:
        raise RuntimeError("conversation topology failed memory admission")
    # The ordinary estimate prices the complete role graph. Add a conservative
    # launcher/storage allowance over the explicitly bounded 32 MiB snapshot,
    # transient encrypted pages and 2 MiB conversation metadata.
    extra_bytes = 512 << 20
    required_bytes = memory["candidates"]["native"]["required_host_bytes"] + extra_bytes
    files = [Path(__file__), Path(__file__).with_name("frontier_probe_support.py")]
    hashes = lambda: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    frozen, salt = hashes(), os.urandom(32)
    output = {"schema": "pllm.conversation_restore_benchmark.v1", "complete": False,
        "experiment": experiment.to_spec(),
        "memory_admission": {"topology": memory, "checkpoint_and_launcher_extra_bytes": extra_bytes,
                             "required_host_bytes": required_bytes},
        "source_lock_digest": source.source_lock_digest, "frozen_sources": frozen,
        "workload_commitment": commitment(salt, prompts), "results": [],
        "scope": ["prepared SDK; three turns; same verified public cache survives client restarts",
            "cold cache and fresh separate provider processes for each cell; shared native 100/40 caps",
            "client lifetime CPU/RSS includes imports, publication/restoration and state commitment audits",
            "provider CPU from birth; provider RSS sampled every 50 ms after startup; startup peak unknown",
            "launcher owns relay/sampler; window CPU separate, independent peak RSS unmeasured",
            "first text and N-1 text-delta throughput; event count must equal generated tokens",
            "logical encrypted checkpoint reads/writes; filesystem cache and physical disk I/O unmeasured",
            "no physical wire or checkpoint distribution",
            "trusted checkpoint keys/heads stay with private launcher; providers receive fresh one-use material"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with fresh_workers(Path(__file__).resolve(), memory_bytes=required_bytes, timeout=2700) as (run, guard, _):
        output["build"] = run.build
        for replicate in range(args.processes):
            for mode in (("ram", "replay", "restore") if replicate % 2 == 0 else ("restore", "replay", "ram")):
                result = case({"experiment": experiment.to_spec(), "mode": mode, "replicate": replicate,
                    "prompts": prompts, "salt": salt.hex(), "key": os.urandom(32).hex(),
                    "output_tokens": experiment.budget.max_new_tokens}, run)
                if hashes() != frozen:
                    raise RuntimeError("benchmark sources changed during cohort")
                output["results"].append(result)
                output["memory_guard"] = {"failed": guard.error is not None,
                    "minimum_available_bytes": guard.minimum_available_bytes,
                    "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes}
                args.output.write_text(json.dumps(output, indent=2) + "\n")
    states = [[row["state"] for client in r["clients"] for row in client["turns"]] for r in output["results"]]
    output["exact_state_and_output_match"] = all(s == states[0] for s in states)
    output["complete"] = len(states) == args.processes * 3 and output["exact_state_and_output_match"]
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if not output["exact_state_and_output_match"]:
        raise RuntimeError("complete logits/KV/history/output parity failed; do not rank")


if __name__ == "__main__":
    main()
