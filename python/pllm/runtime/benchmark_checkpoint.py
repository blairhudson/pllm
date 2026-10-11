"""Installed checkpoint SDK cost target with a fresh-interpreter continuation."""
from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
import time

import numpy as np

from pllm._benchmark_contract import TargetUnavailable, canonical, digest
from pllm._benchmark_worker import peak_rss
from pllm.state import CheckpointHead, CheckpointSession
from .benchmark_decoder_restart import bind, _MAX_SOURCE


def _hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def _state(session):
    # Complete active KV is checked one layer at a time, with no retained copies.
    runtime, owner = session._runtime, session._owner
    owner.begin(runtime, np.array([0], np.int64), False)
    result = hashlib.sha256()
    for layer in range(runtime.layers):
        owner.load(runtime, layer)
        cache = runtime.caches[layer]
        for values in (cache.key, cache.value):
            result.update(values[:runtime.position].tobytes())
        owner._retire(runtime)
    return result.hexdigest()


def child(request):
    compiled, _, unused, remote = bind(request["source"], request["bound"], request["outputs"])
    unused.reset()
    with CheckpointSession(compiled, remote, request["directory"], bytes.fromhex(request["key"]),
                           storage=request["storage"], page_rows=request["page_rows"],
                           head=CheckpointHead.from_spec(request["head"])) as session:
        restored = _hash(session.logits)
        continued = _hash(session.advance())
        result = {"restored": restored, "continued": continued, "pending": session.pending_token,
                  "position": session.position, "state": _state(session)}
        metrics = session.storage_metrics
    return {"result": result, "storage": metrics, "cpu_seconds": time.process_time(),
            "peak_rss_bytes": peak_rss(), "material_body_bytes": remote.material_bytes}


def execute_checkpoint(p, m):
    from pllm.configuration import Experiment
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow

    experiment = Experiment.from_spec(p["model"]["experiment"])
    required = MaskedLinearCpu(experiment.pipeline.model,
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    if (dict(experiment.pipeline.components) != dict(required.components)
            or p["model"]["profile"] != "baseline.masked_linear_cpu" or p["model"]["config"] is not None):
        raise TargetUnavailable("checkpoint_requires_exact_reference_composition")
    # The final extra row is a bounded KV audit, not executed history.
    if p["tokens"] + p["steps"] + 2 > experiment.budget.max_input_tokens:
        raise TargetUnavailable("checkpoint_continuation_bound", status="blocked")
    m.admit(768 << 20)
    resolved = resolve_model(experiment.pipeline.model)
    if resolved.path is None:
        raise TargetUnavailable("local_checkpoint_source_required")
    cpu, wall = time.process_time(), time.perf_counter()
    compiled, contract, reference, remote = bind(resolved.path,
        experiment.budget.max_input_tokens, experiment.budget.max_new_tokens)
    m.details["source_loading"] = {"cpu_seconds": time.process_time() - cpu,
                                   "wall_seconds": time.perf_counter() - wall}
    prompt = np.random.default_rng(p["seed"]).integers(2, int(reference.cfg["vocab_size"]), p["tokens"]).tolist()

    def execute(index):
        root, key = m.directory / f"checkpoint-{index}", secrets.token_bytes(32)
        frames = []
        before = remote.material_bytes
        with CheckpointSession(compiled, remote, root, key, storage=p["storage"], page_rows=p["page_rows"]) as session:
            frames.append(_hash(session.append(prompt)))
            for _ in range(p["steps"]):
                frames.append(_hash(session.advance()))
            head, storage = session.head, session.storage_metrics
        request = {"source": str(resolved.path), "bound": experiment.budget.max_input_tokens,
            "outputs": experiment.budget.max_new_tokens, "directory": str(root), "key": key.hex(),
            "head": head.to_spec(), "storage": p["storage"], "page_rows": p["page_rows"]}
        restarted = json.loads(subprocess.run([sys.executable, "-I", "-m", __name__],
            input=canonical(request), capture_output=True, check=True,
            timeout=m.sampling.timeout_seconds).stdout)
        return {"frames": frames, "child": restarted, "storage": storage,
                "material_body_bytes": remote.material_bytes - before}

    try:
        observed = m.run(execute)
        m.details.update(target_peak_rss_bytes=peak_rss(), target_cpu_from_birth_seconds=time.process_time())
        m.details["timings"]["cpu_seconds"] = [own + row["child"]["cpu_seconds"] for own, row in
            zip(m.details["timings"]["cpu_seconds"], observed, strict=True)]
        oracle_cpu = time.process_time()
        scores = reference.prepare_ids(prompt)[1]
        expected = [_hash(scores)]
        for _ in range(p["steps"]):
            scores = reference.forward_ids([int(np.argmax(scores))])[-1]
            expected.append(_hash(scores))
        scores = reference.forward_ids([int(np.argmax(scores))])[-1]
        state = hashlib.sha256()
        checked = len(expected) * scores.size + 2 * scores.size
        for cache in reference.caches:
            for values in (cache.key, cache.value):
                state.update(values[:reference.position].tobytes())
                checked += values[:reference.position].size
        final = {"restored": expected[-1], "continued": _hash(scores), "pending": int(np.argmax(scores)),
                 "position": reference.position, "state": state.hexdigest()}
        if any(row["frames"] != expected or row["child"]["result"] != final for row in observed):
            raise TargetUnavailable("checkpoint_exact_parity_failed", status="failed")
        m.oracle = {"kind": "uninterrupted_full_logits_kv_pending_exact_bits", "outcome": "passed",
                    "checked_values": checked * len(observed)}
        m.details.update(oracle_cpu_seconds=time.process_time() - oracle_cpu,
            storage=p["storage"], page_rows=p["page_rows"],
            target_cpu_owners=["checkpoint_issuer", "fresh_restore_client_with_local_prepared_roles"],
            samples=[{"storage": row["storage"], "material_body_bytes": row["material_body_bytes"],
                      "child": {k: v for k, v in row["child"].items() if k != "result"}}
                     for row in observed],
            model={"id": str(experiment.pipeline.model.source), "source_lock_digest": resolved.source_lock_digest,
                   "checkpoint_digest": resolved.checkpoint_digest}, plan_digest=compiled.model_plan_digest,
            component_ids=sorted({c.component for c in experiment.pipeline.components.values()}))
        for name in ("disk_bytes", "read_bytes", "written_bytes", "peak_live_kv_bytes"):
            values = [row["child"]["storage"][name] if name in {"disk_bytes", "peak_live_kv_bytes"}
                      else row["storage"][name] + row["child"]["storage"][name] for row in observed]
            m.metric(f"checkpoint.{name}", max(values), kind="bytes" if name in {"read_bytes", "written_bytes"} else "allocated",
                     scope="private_storage_and_layer_payload", window="publication_restart_and_audit")
        m.metric("restore_child.peak_rss", max(row["child"]["peak_rss_bytes"] for row in observed),
                 kind="memory", role="restore_child", scope="process_lifetime_high_water")
        m.fixture = {"kind": "checkpoint", "digest": resolved.source_lock_digest,
                     "metadata": {"public_seed": p["seed"], "source_float32_bound_bytes": _MAX_SOURCE}}
        m.comparison.update(source_digest=resolved.source_lock_digest, numeric_digest=compiled.digest,
            state_digest=contract.digest, privacy_digest=digest({"scope": "local_prepared_reference"}),
            workload_digest=digest({k: p[k] for k in ("tokens", "steps", "seed")}),
            output_digest=m.commitment({"frames": expected, "continued": final}),
            measurement_digest=digest({"scope": "checkpoint_publication_fresh_restart_decode_full_kv_audit_v1"}))
        m.limitations += ["Bounded checkpoint decoder with fresh in-process prepared arithmetic, not HTTP roles or a persistent OpenAI conversation.",
            "Caller retains key and trusted head; immutable branch pages and abandoned generations remain charged on disk.",
            "Target time includes publication, fresh interpreter/source loading, continued decoding, hashing and complete KV audit; initial issuer loading is separate.",
            "Issuer and restarted-client peaks remain separate; no sum of lifetime peaks or whole-response performance claim."]
    finally:
        reference.reset()


if __name__ == "__main__":
    request = json.loads(sys.stdin.buffer.read((1 << 20) + 1))
    print(json.dumps(child(request)))
