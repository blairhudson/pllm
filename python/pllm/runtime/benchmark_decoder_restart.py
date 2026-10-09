"""Small, trusted-client decoder restart reference; no serving checkpoint importer.

Only the fixture issuer owns its AEAD key and separately retained head. Prepared
arithmetic uses fresh local material; both simulated roles share this process.
No inventory, seed, credential or process-local snapshot seal is persisted.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

import msgpack
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pllm._benchmark_contract import TargetUnavailable, canonical, digest
from pllm._benchmark_worker import peak_rss

_SCHEMA = "pllm.decoder_restart_reference.v1"
_MAX_FRAME = 8 << 20
_MAX_SOURCE = 16 << 20


class FreshPreparedReference:
    """Original prepared correction algebra, with a fresh one-use row owner per call."""

    def __init__(self, engine, compiled):
        self.engine, self.compiled = engine, compiled
        self.attempts = set()
        self.rows = self.material_bytes = 0

    def __call__(self, stage_id, activation):
        from .preparation_protocol import PreparationRequest, prepared_mask_rows
        from .quantization import centered_residues, dequantize_matmul, quantize_activation_per_row
        from .stage_protocol import pack_residues, unpack_residues
        model_id = self.compiled._bundle.model_id
        stage = self.engine.models[model_id].stages[stage_id]
        q = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        rows = q.values.shape[0]
        profile = stage.seeded_profile
        req = PreparationRequest(secrets.token_hex(16), secrets.token_hex(16), model_id,
            self.compiled._bundle.privacy["body_fingerprint"], stage_id, stage.weight_digest,
            rows, stage.spec.in_features, stage.spec.out_features, stage.spec.weight_bits,
            stage.spec.activation_bits, stage.signed_output_bound, profile.ring, profile.modulus,
            profile.wire_bits, secrets.token_bytes(32))
        if req.attempt_id in self.attempts:
            raise ValueError("fresh restart material repeated an attempt")
        self.attempts.add(req.attempt_id)
        cursor = prepared_mask_rows(req)
        if cursor is None:
            raise TargetUnavailable("native_restart_material_required")
        try:
            raw_r, raw_s = cursor.take(0, rows)
            r = np.frombuffer(raw_r, "<u4").reshape(q.values.shape)
            s = np.frombuffer(raw_s, "<u4").reshape(rows, stage.spec.out_features)
            correction = asyncio.run(self.engine.prepare_seeded_stage(req)).correction
            masked = (q.values.astype(np.uint32) - r) & np.uint32(profile.modulus - 1)
            frame = pack_residues(masked, profile.wire_bits)
            masked = unpack_residues(frame, masked.shape, profile.wire_bits)
            product = (stage.compiled_weight.wrap32(masked) if profile.wire_bits == 32
                       else stage.compiled_weight.modular(masked, profile.modulus))
            response = (product + correction) & np.uint32(profile.modulus - 1)
            reply = pack_residues(response, profile.wire_bits)
            actual = (unpack_residues(reply, response.shape, profile.wire_bits) + s) & np.uint32(profile.modulus - 1)
            self.rows += rows
            self.material_bytes += len(frame) + len(reply) + correction.size * (profile.wire_bits // 8)
            result = dequantize_matmul(centered_residues(actual, profile.modulus), q.scales,
                stage.weight.scales, output_shape=q.original_shape[:-1] + (stage.spec.out_features,))
            if stage.bias is not None:
                result += stage.bias
            return np.ascontiguousarray(result, dtype=np.float32)
        finally:
            cursor.cancel()


def bind(path, bound, outputs=4):
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from .loaders import load_hf_directory
    from .model_binding import compile_runtime_model
    from .safetensors_store import SafeTensorStore
    from .transformer_client import ClientBundle
    from .transformer_engine import MaskedTransformerEngine
    path = Path(path)
    if (path / "config.json").stat().st_size > 2 << 20:
        raise TargetUnavailable("restart_reference_configuration_limit", status="blocked")
    config = json.loads((path / "config.json").read_bytes())
    plan = lower_model(config, batch=1, max_input_tokens=bound, max_new_tokens=outputs)
    headers = SafeTensorStore(path)
    payload = sum(math.prod(headers.tensor_shape(key)) * 4 for key in headers.keys)
    if payload > _MAX_SOURCE:
        raise TargetUnavailable("restart_reference_source_limit", status="blocked")
    model = "restart-reference"
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id=model)))
    bundle = ClientBundle.unpack(engine.client_bundle(model))
    pipeline = MaskedLinearCpu(Model(model), quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    compiled = compile_runtime_model(plan, bundle, composition=pipeline)
    contract = plan.continuation_schedule(pipeline)
    remote = FreshPreparedReference(engine, compiled)
    runtime = compiled.runtime(remote)
    runtime.install_continuation(contract)
    return compiled, contract, runtime, remote


def _aad(compiled, contract, revision):
    return canonical({"schema": _SCHEMA, "binding": compiled.digest,
                      "continuation": contract.digest, "revision": revision})


def _validate_owner(compiled, contract, runtime):
    from pllm.configuration import Pipeline
    from .model_binding import _state_binding_digest
    compiled.validate()
    composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    expected = _state_binding_digest(compiled._plan, compiled._bundle, composition.digest())
    if runtime._state_binding_digest != expected:
        raise ValueError("checkpoint runtime differs from its compiled owner")
    runtime.install_continuation(contract)


def publish(root, key, revision, compiled, contract, runtime, logits, history, *, fail=False):
    if type(revision) is not int or revision < 0 or type(key) is not bytes or len(key) != 32:
        raise ValueError("invalid checkpoint owner")
    _validate_owner(compiled, contract, runtime)
    contract.admit(runtime.position, 2, memory_bytes=64 << 20)
    saved = runtime.snapshot_for_response("research-response")
    runtime._validate_full_snapshot(saved)
    basis = saved.state_basis
    if (basis is None or not basis.valid() or basis.verification_failure_bits
            or basis.phase not in {"completed_prefill", "incremental"}):
        raise ValueError("restart requires unverified qualified response state")
    contract.admit(saved.position, 2, memory_bytes=64 << 20)
    logits = np.asarray(logits)
    if (logits.dtype != np.float32 or logits.shape != (int(runtime.cfg["vocab_size"]),)
            or not np.all(np.isfinite(logits)) or len(history) != saved.position
            or any(type(token) is not int or not 0 <= token < logits.size for token in history)):
        raise ValueError("restart logits or executed history differs")
    tensors = [getattr(row, kind)[:row.length] for row in saved.caches for kind in ("key", "value")]
    if sum(value.nbytes for value in tensors) + logits.nbytes + 8 * len(history) > _MAX_FRAME // 2:
        raise ValueError("restart payload exceeds admitted bound")
    document = {"schema": _SCHEMA, "binding": compiled.digest, "position": saved.position,
        "phase": basis.phase, "native_phase": basis.native_phase_digest, "execution": basis.execution_digest,
        "prefill_extent": basis.prefill_extent, "history": history, "pending": int(np.argmax(logits)),
        "logits": logits.tobytes(), "tensors": [value.tobytes() for value in tensors]}
    raw = msgpack.packb(document, use_bin_type=True)
    if len(raw) > _MAX_FRAME:
        raise ValueError("restart frame exceeds admitted bound")
    nonce = secrets.token_bytes(12)
    frame = nonce + AESGCM(key).encrypt(nonce, raw, _aad(compiled, contract, revision))
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="pending-", dir=root)
    try:
        with os.fdopen(fd, "wb") as sink:
            sink.write(frame)
            sink.flush()
            os.fsync(sink.fileno())
        if fail:
            raise ValueError("injected publication interruption")
        os.link(name, root / str(revision))
        return {"revision": revision, "digest": hashlib.sha256(frame).hexdigest()}, len(frame)
    finally:
        Path(name).unlink(missing_ok=True)


def restore(root, key, head, compiled, contract, runtime):
    from .transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis
    _validate_owner(compiled, contract, runtime)
    if (not isinstance(head, dict) or set(head) != {"revision", "digest"}
            or type(head["revision"]) is not int or head["revision"] < 0):
        raise ValueError("invalid trusted checkpoint head")
    with (Path(root) / str(head["revision"])).open("rb") as source:
        frame = source.read(_MAX_FRAME + 29)
    if (not 28 < len(frame) <= _MAX_FRAME + 28 or hashlib.sha256(frame).hexdigest() != head["digest"]):
        raise ValueError("checkpoint differs from trusted head")
    raw = AESGCM(key).decrypt(frame[:12], frame[12:], _aad(compiled, contract, head["revision"]))
    row = msgpack.unpackb(raw, raw=False, strict_map_key=True, max_map_len=16,
        max_array_len=4096, max_bin_len=_MAX_FRAME, max_str_len=1024)
    fields = {"schema", "binding", "position", "phase", "native_phase", "execution", "prefill_extent",
              "history", "pending", "logits", "tensors"}
    if (type(row) is not dict or set(row) != fields or row["schema"] != _SCHEMA
            or row["binding"] != compiled.digest or type(row["position"]) is not int
            or not 1 <= row["position"] <= 4096 or row["phase"] not in {"completed_prefill", "incremental"}):
        raise ValueError("checkpoint source/state contract differs")
    contract.admit(row["position"], 2, memory_bytes=64 << 20)
    vocab = int(runtime.cfg["vocab_size"])
    if (type(row["logits"]) is not bytes or len(row["logits"]) != vocab * 4
            or type(row["history"]) is not list or len(row["history"]) != row["position"]
            or any(type(token) is not int or not 0 <= token < vocab for token in row["history"])
            or type(row["pending"]) is not int or not 0 <= row["pending"] < vocab):
        raise ValueError("checkpoint history or pending-token contract differs")
    logits = np.frombuffer(row["logits"], np.float32)
    if not np.all(np.isfinite(logits)) or row["pending"] != int(np.argmax(logits)):
        raise ValueError("checkpoint pending token lacks greedy-logit provenance")
    declarations = sorted(compiled._plan.decode["state_inputs"], key=lambda item: (item["layer"], item["kind"]))
    if type(row["tensors"]) is not list or len(row["tensors"]) != len(declarations):
        raise ValueError("checkpoint KV cardinality differs")
    arrays = []
    for value, state in zip(row["tensors"], declarations, strict=True):
        shape = (row["position"], state["shape"][1], state["shape"][3])
        if type(value) is not bytes or len(value) != math.prod(shape) * 4:
            raise ValueError("checkpoint KV shape differs")
        arrays.append(np.frombuffer(value, np.float32).reshape(shape))
    caches = [LayerCache(arrays[index], arrays[index + 1], row["position"]) for index in range(0, len(arrays), 2)]
    # AEAD verification is the trusted local import boundary. Never reuse the
    # originating process's snapshot seal or any preparation inventory.
    owner = secrets.token_hex(16)
    basis = _snapshot_state_basis(row["phase"], row["position"], runtime._state_binding_digest,
        row["native_phase"], row["execution"], owner, row["prefill_extent"])
    saved = RuntimeSnapshot(row["position"], caches, {}, runtime._state_binding_digest, basis)
    runtime.restore_for_response(saved, owner)
    return logits, row["history"], row["pending"]


def advance(runtime, logits, history, steps=2):
    hashes = []
    for _ in range(steps):
        token = int(np.argmax(logits))
        logits = runtime.continue_ids([token])[-1]
        history.append(token)
        hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
    kv = hashlib.sha256()
    for row in runtime.caches:
        kv.update(row.key[:row.length].tobytes())
        kv.update(row.value[:row.length].tobytes())
    return {"logits": hashes, "state": kv.hexdigest(), "history": digest(history), "position": runtime.position}


def child(request):
    start, cpu = time.perf_counter(), time.process_time()
    compiled, contract, runtime, remote = bind(request["source"], request["bound"], request["outputs"])
    bootstrap = {"wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}
    start, cpu = time.perf_counter(), time.process_time()
    logits, history, pending = restore(request["directory"], bytes.fromhex(request["key"]),
        request["head"], compiled, contract, runtime)
    restore_cost = {"wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}
    if runtime.position != len(history) or pending != int(np.argmax(logits)):
        raise ValueError("pending token was counted as executed state")
    result = advance(runtime, logits, history)
    return {"result": result, "bootstrap": bootstrap, "restore": restore_cost,
        "cpu_seconds": time.process_time(), "peak_rss_bytes": peak_rss(),
        "fresh_attempts": len(remote.attempts), "material_body_bytes": remote.material_bytes}


def execute_restart(p, m):
    from pllm.configuration import Experiment
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    if p["storage"] != "sealed_local" or p["tokens"] > 128:
        raise TargetUnavailable("decoder_restart_reference_contract")
    m.admit(512 << 20)
    experiment = Experiment.from_spec(p["model"]["experiment"])
    expected_pipeline = MaskedLinearCpu(experiment.pipeline.model,
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    if (dict(experiment.pipeline.components) != dict(expected_pipeline.components)
            or p["model"]["profile"] != "baseline.masked_linear_cpu" or p["model"]["config"] is not None):
        raise TargetUnavailable("decoder_restart_requires_exact_reference_composition")
    resolved = resolve_model(experiment.pipeline.model)
    if resolved.path is None:
        raise TargetUnavailable("local_restart_source_required")
    before, cpu = time.perf_counter(), time.process_time()
    compiled, contract, runtime, remote = bind(resolved.path, experiment.budget.max_input_tokens,
                                              experiment.budget.max_new_tokens)
    if p["tokens"] + 3 > experiment.budget.max_input_tokens:
        raise TargetUnavailable("restart_continuation_bound", status="blocked")
    history = np.random.default_rng(p["seed"]).integers(2, int(runtime.cfg["vocab_size"]), p["tokens"]).tolist()
    logits = runtime.prepare_ids(history)[1]
    token = int(np.argmax(logits))
    logits = runtime.forward_ids([token])[-1]
    history.append(token)
    bootstrap = {"wall_seconds": time.perf_counter() - before, "cpu_seconds": time.process_time() - cpu,
        "material_body_bytes": remote.material_bytes, "fresh_attempts": len(remote.attempts)}
    key = secrets.token_bytes(32)
    before, cpu = time.perf_counter(), time.process_time()
    head, frame_bytes = publish(m.directory / "checkpoint", key, 0, compiled, contract, runtime, logits, history)
    publication = {"wall_seconds": time.perf_counter() - before, "cpu_seconds": time.process_time() - cpu,
                   "private_file_bytes": frame_bytes}
    # Control starts from the identical response-owned continuation basis.
    saved = runtime.snapshot_for_response("research-response")
    runtime.restore_for_response(saved, "research-response")
    expected = advance(runtime, logits, history.copy())
    request = {"source": str(resolved.path), "bound": experiment.budget.max_input_tokens,
        "outputs": experiment.budget.max_new_tokens, "directory": str(m.directory / "checkpoint"),
        "key": key.hex(), "head": head}
    observed = m.run(lambda _: json.loads(subprocess.run([sys.executable, "-I", "-m", __name__],
        input=canonical(request), capture_output=True, check=True, timeout=m.sampling.timeout_seconds).stdout))
    if any(row["result"] != expected or row["fresh_attempts"] < 1 for row in observed):
        raise TargetUnavailable("decoder_restart_parity_failed", status="failed")
    m.details["timings"]["cpu_seconds"] = [own + row["cpu_seconds"] for own, row in
        zip(m.details["timings"]["cpu_seconds"], observed, strict=True)]
    m.details.update(target_peak_rss_bytes=peak_rss(), target_cpu_from_birth_seconds=time.process_time(),
        target_cpu_owners=["fixture_issuer", "restarted_client_with_local_prepared_roles"],
        component_ids=sorted({component.component for component in experiment.pipeline.components.values()}),
        model={"id": str(experiment.pipeline.model.model_id or experiment.pipeline.model.source),
               "source_lock_digest": resolved.source_lock_digest, "checkpoint_digest": resolved.checkpoint_digest},
        plan_digest=compiled.model_plan_digest, continuation_digest=contract.digest,
        bootstrap=bootstrap, publication=publication,
        restarted=[{key: value for key, value in row.items() if key != "result"} for row in observed])
    m.metric("restore_child.peak_rss", max(row["peak_rss_bytes"] for row in observed), kind="memory",
        role="restore_child", window="process_birth_through_continued_decode", scope="process_lifetime_high_water")
    m.oracle = {"kind": "bit_exact_uninterrupted_decoder_logits_kv_pending_and_history", "outcome": "passed",
        "checked_values": len(observed) * (2 * int(runtime.cfg["vocab_size"]) + sum(
            2 * row.length * math.prod(row.key.shape[1:]) for row in runtime.caches))}
    m.fixture = {"kind": "checkpoint", "digest": resolved.source_lock_digest,
        "metadata": {"public_token_seed": p["seed"], "source_float32_bound_bytes": _MAX_SOURCE,
                     "generated_checkpoint": experiment.pipeline.model.kind == "tiny"}}
    m.comparison.update(source_digest=resolved.source_lock_digest, numeric_digest=compiled.digest,
        state_digest=contract.digest, privacy_digest=digest({"scope": "local_prepared_reference"}),
        workload_digest=digest({"tokens": p["tokens"], "steps": 2, "seed": p["seed"]}),
        output_digest=m.commitment(expected), measurement_digest=digest({"scope": _SCHEMA}))
    m.limitations += ["Bounded checkpoint-backed restart reference with locally simulated prepared roles, not a persistent SDK conversation.",
        "Trusted issuer retains key and revision; persistent key management, anti-rollback service and concurrent writers remain absent.",
        "Greedy pending tokens, logits and executed history are encrypted; no masks, inventory, credentials or old process seal persist.",
        "Bootstrap, publication, restarted-client import/restore and fresh material remain separately charged; no HTTP, full-wire or whole-response saving is established."]


if __name__ == "__main__":
    request = json.loads(sys.stdin.buffer.read((1 << 20) + 1))
    print(json.dumps(child(request)))
