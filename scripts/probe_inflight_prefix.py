"""Bounded trusted-client single-flight screen, not a provider or SDK integration.

Only completed native prefix_f32 full-KV snapshots are fanned out. A barrier
admits all cold waiters before prefix execution; branches retain private copies.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import dataclass
from contextlib import ExitStack, contextmanager
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import sys
import time

import numpy as np

INFLIGHT_ROOT = Path(__file__).resolve().parents[1]
INFLIGHT_CONTROL = INFLIGHT_ROOT / "docs/evidence/research-prepared-study-qwen25-confirm.json"
INFLIGHT_GEOMETRY = INFLIGHT_ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"


def inflight_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inflight_checkpoint_files(path):
    # Keep logical symlink paths: replacing a snapshot link must also fail.
    return {str(p.absolute()): inflight_hash(p) for p in sorted(Path(path).rglob("*")) if p.is_file()}


def inflight_checkpoint_pin():
    from huggingface_hub import snapshot_download
    from decoder_probe_support import MODEL, REVISION
    path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True,
        allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json",
                        "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                        "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]))
    files = inflight_checkpoint_files(path)
    if not (path / "config.json").is_file() or not any(p.endswith(".safetensors") for p in files):
        raise RuntimeError("pinned cached config/checkpoint files unavailable")
    return {"path": str(path), "files": files, "model": MODEL, "revision": REVISION}


def inflight_verify_checkpoint(pin):
    if pin and inflight_checkpoint_files(pin["path"]) != pin["files"]:
        raise RuntimeError("pinned checkpoint/config namespace or bytes changed")


def inflight_control_admission(fixture, pin):
    geometry = json.loads(INFLIGHT_GEOMETRY.read_text())
    control = json.loads(INFLIGHT_CONTROL.read_text())
    source = geometry["source"]
    if not geometry["complete"] or not control["complete"]:
        raise ValueError("incomplete historical control/geometry")
    lock = fixture.lock()
    for name in ("model", "revision", "checkpoint_digest", "source_lock_digest", "body_fingerprint"):
        if lock[name] != source[name]:
            raise ValueError("pinned source differs from historical control geometry")
    if fixture.source.path.resolve() != Path(pin["path"]).resolve():
        raise ValueError("loaded checkpoint differs from pre-worker pinned snapshot")
    confirmations = [r for r in control["results"] if r["execution"]["case"]["name"] == "prepared/digit8"]
    if len(confirmations) != 3 or any(run["model_fingerprint"] != source["body_fingerprint"]
            for row in confirmations for run in row["execution"]["details"]["report"]["runs"]):
        raise ValueError("historical confirmation body identity mismatch")
    stages = {row["stage"]: row for row in geometry["stages"]}
    body = {k: v for k, v in fixture.model.stages.items() if v.spec.role not in {"token_lookup", "lm_head"}}
    if body.keys() != stages.keys():
        raise ValueError("historical stage geometry set mismatch")
    for name, stage in body.items():
        expected = stages[name]
        from pllm.runtime.transformer_engine import _weight_chunks
        actual = hashlib.sha256()
        for chunk in _weight_chunks(stage.weight.values):
            actual.update(chunk.tobytes())
        if (list(stage.weight.values.shape) != expected["shape"]
                or stage.weight_digest != expected["weight_sha256"]
                or actual.hexdigest() != expected["weight_sha256"]):
            raise ValueError("actual W8 stage weight differs from pinned geometry")
        # Imported accessor hashes actual streamed W8 chunks, not just descriptor.
        stage.output_residue_bits
    return {"geometry_sha256": inflight_hash(INFLIGHT_GEOMETRY),
            "control_sha256": inflight_hash(INFLIGHT_CONTROL),
            "verified_body_stage_weights": len(stages), "source_lock": source}


@contextmanager
def inflight_fresh_output(path):
    # Exclusive creation rejects old files AND symlinks before expensive work.
    with Path(path).open("x") as stream:
        owned = os.fstat(stream.fileno())
        try:
            yield stream
        except BaseException:
            stream.close()
            try:
                current = Path(path).lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                Path(path).unlink()
            raise


def inflight_payload(snapshot, logits):
    return logits.nbytes + sum(c.key.nbytes + c.value.nbytes for c in snapshot.caches)


def inflight_erase(snapshot, logits):
    for c in snapshot.caches:
        c.key.fill(0)
        c.value.fill(0)
    logits.fill(0)


@dataclass
class InflightLease:
    owner: object
    node: object
    snapshot: object
    logits: np.ndarray
    reserved: int
    closed: bool = False

    def close(self):
        if not self.closed:
            self.closed = True
            inflight_erase(self.snapshot, self.logits)
            self.owner._drop(self.node, self.reserved)


class InflightPrefixOwner:
    """Single event-loop owner; capability identity is the trusted-client boundary.

    Explicit prefix node plus private suffix branches forms a one-level DAG.
    No persistent cache, automatic cross-client batching, or material reuse.
    Reservations include producer runtime/snapshot overlap and each branch's
    snapshot/runtime overlap. Caller must cap runtime capacity to that allowance.
    """

    def __init__(self, client, source, composition, binding, *, max_nodes=4,
                 max_waiters=8, max_bytes=64 << 20):
        if not (1 <= max_nodes <= 64 and 1 <= max_waiters <= 64
                and 0 < max_bytes <= 256 << 20):
            raise ValueError("invalid owner bounds")
        self.client, self.source, self.composition, self.binding = client, source, composition, binding
        self.max_nodes, self.max_waiters, self.max_bytes = max_nodes, max_waiters, max_bytes
        self._secret = secrets.token_bytes(32)
        self._nodes = {}
        self._retiring = {}
        self.bytes = self.peak_bytes = self.waiters = self.executions = self.copy_bytes = 0
        self.retired = self.cancelled_work = 0
        self.wait_seconds = self.copy_cpu_seconds = 0.0
        self._loop = None

    def _commit(self, snapshot, logits):
        digest = hmac.new(self._secret, digestmod="sha256")
        digest.update(str((snapshot.position, snapshot.semantic_binding_digest,
                           snapshot.state_basis)).encode())
        for c in snapshot.caches:
            digest.update(c.key.tobytes())
            digest.update(c.value.tobytes())
        digest.update(logits.tobytes())
        return digest.digest()

    def _charge(self, count):
        if self.bytes + count > self.max_bytes:
            raise MemoryError("snapshot/runtime overlap exceeds owner RAM bound")
        self.bytes += count
        self.peak_bytes = max(self.bytes, self.peak_bytes)

    async def acquire(self, client, source, composition, binding, ids, execute,
                      *, payload_bound, branch_overlap=None, producer_overlap=None,
                      reduction="prefix_f32"):
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("owner belongs to another event loop")
        self._loop = loop
        if client is not self.client or (source, composition, binding) != (
                self.source, self.composition, self.binding) or reduction != "prefix_f32":
            raise ValueError("trusted client/source/composition/numeric ownership mismatch")
        values = np.array(ids, dtype=np.int64, copy=True)
        if values.ndim != 1 or not 0 < len(values) <= 4096 or np.any(values < 0) or np.any(values > 0xffffffff):
            raise ValueError("invalid bounded prefix")
        if type(payload_bound) is not int or not 0 < payload_bound <= self.max_bytes:
            raise ValueError("invalid payload bound")
        overlap = 2 * payload_bound if branch_overlap is None else branch_overlap
        if type(overlap) is not int or overlap < 2 * payload_bound:
            raise ValueError("branch overlap must cover snapshot and restored runtime")
        producer = 2 * payload_bound if producer_overlap is None else producer_overlap
        if type(producer) is not int or producer < 2 * payload_bound:
            raise ValueError("producer overlap must cover runtime and published snapshot")
        key = hmac.new(self._secret, values.astype("<u4").tobytes(), "sha256").digest()
        if self.waiters >= self.max_waiters:
            raise RuntimeError("waiter bound exceeded")
        node = self._nodes.get(key)
        if node is None:
            if len(self._nodes) + len(self._retiring) >= self.max_nodes:
                raise RuntimeError("node bound exceeded")
            self._charge(producer + overlap)
            node = {"key": key, "refs": 0, "bound": payload_bound,
                    "producer_reservation": producer, "result": None}
            self._nodes[key] = node

            async def produce():
                self.executions += 1
                snapshot, logits = await execute()
                node["result"] = (snapshot, logits)
                basis = snapshot.state_basis
                expected_execution = None if basis is None else hashlib.sha256(
                    b"pllm.decoder.execution-basis.v1\0" + basis.native_phase_digest.encode()
                    + np.arange(len(values), dtype="<i8").tobytes()
                    + values.astype("<i8").tobytes()).hexdigest()
                if (snapshot.position != len(values) or snapshot.shared_kv
                        or snapshot.semantic_binding_digest != self.binding
                        or basis is None or not basis.completed_prefill()
                        or basis.source_binding_digest != self.binding
                        or basis.execution_digest != expected_execution
                        or basis.verification_failure_bits != 0
                        or basis.position != len(values) or basis.prefill_extent != len(values)
                        or logits.dtype != np.float32 or logits.ndim != 1
                        or not np.all(np.isfinite(logits)) or not snapshot.caches
                        or any(c.length != len(values) or c.key.dtype != np.float32
                               or c.value.dtype != np.float32 or c.key.shape != c.value.shape
                               or c.key.shape[0] != len(values) or not np.all(np.isfinite(c.key))
                               or not np.all(np.isfinite(c.value)) for c in snapshot.caches)
                        or inflight_payload(snapshot, logits) > payload_bound):
                    raise ValueError("prefix result provenance/shape/payload mismatch")
                node["commit"] = self._commit(snapshot, logits)
                return snapshot, logits

            node["task"] = asyncio.create_task(produce())
        else:
            if node["bound"] != payload_bound or node["producer_reservation"] != producer:
                raise ValueError("prefix reservation substitution")
            self._charge(overlap)
        node["refs"] += 1
        self.waiters += 1
        start = time.perf_counter()
        try:
            snapshot, logits = await asyncio.shield(node["task"])
            self.wait_seconds += time.perf_counter() - start
            if not hmac.compare_digest(node["commit"], self._commit(snapshot, logits)):
                raise ValueError("prefix result substitution")
            cpu = time.process_time()
            saved, scores = copy.deepcopy(snapshot), logits.copy()
            self.copy_cpu_seconds += time.process_time() - cpu
            self.copy_bytes += inflight_payload(saved, scores)
            return InflightLease(self, node, saved, scores, overlap)
        except BaseException:
            self._drop(node, overlap)
            if node["refs"] == 0:
                try:
                    await node["task"]
                except BaseException:
                    pass
            raise

    def _drop(self, node, reservation):
        node["refs"] -= 1
        self.waiters -= 1
        self.bytes -= reservation
        if node["refs"] == 0:
            self._nodes.pop(node["key"], None)
            self.retired += 1
            if not node["task"].done():
                self.cancelled_work += 1
                self._retiring[id(node)] = node
                node["task"].cancel()
                # Executors must be cancellation-cooperative; no detached threads.
                node["task"].add_done_callback(lambda task: self._finish_retirement(node))
            else:
                self._finish_retirement(node)

    def _finish_retirement(self, node):
        self._retiring.pop(id(node), None)
        self.bytes -= node["producer_reservation"]
        if node["result"] is not None:
            inflight_erase(*node["result"])
        if not node["task"].cancelled():
            node["task"].exception()


def inflight_freeze():
    """Freeze source/control and installed dependency/native bytes before worker."""
    root = Path(__file__).resolve().parents[1]
    paths = {Path(__file__).resolve(), root / "scripts/frontier_probe_support.py",
             root / "scripts/decoder_probe_support.py", root / "tests/test_inflight_prefix_probe.py",
             INFLIGHT_CONTROL, INFLIGHT_GEOMETRY}
    paths.update((root / "python/pllm").rglob("*.py"))
    for entry in sys.path:
        if "site-packages" in entry and Path(entry).is_dir():
            paths.update(Path(entry).glob("*.py"))
    for dist in importlib.metadata.distributions():
        for item in dist.files or ():
            if str(item).endswith((".py", ".so", ".dylib")):
                path = Path(dist.locate_file(item)).resolve()
                if path.is_file():
                    paths.add(path)
    return {str(p.resolve()): inflight_hash(p)
            for p in sorted(paths)}


def inflight_verify(frozen):
    for path, expected in frozen.items():
        with open(path, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise RuntimeError(f"frozen source changed: {path}")


def inflight_imports(frozen):
    result = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or name == "__main__":
            continue
        path = str(Path(path).resolve())
        if path.endswith((".py", ".so", ".dylib")) and "site-packages" in path:
            if path not in frozen:
                raise RuntimeError(f"unfrozen imported dependency: {name}")
            result[name] = {"path": path, "sha256": frozen[path]}
    return result


def inflight_generated(root, family, tokens, outputs):
    from pllm import Model, lower_model
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import MaskedLinear
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    from decoder_probe_support import stage_output
    path = create_tiny_llama_checkpoint(root, model_type=family, seed=719,
        hidden_size=16, intermediate_size=32, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_hidden_layers=2,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    checkpoint = Model.path(str(path), model_id="inflight-generated")
    source = resolve_model(checkpoint)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1,
                                     prepared_output_encoding="row_residues")
    asyncio.run(engine.load(source.manifest))
    model = engine.models[source.manifest.id]
    bundle = ClientBundle.unpack(engine.client_bundle(source.manifest.id))
    plan = lower_model((path / "config.json").read_bytes(), batch=1,
                       max_input_tokens=tokens, max_new_tokens=outputs)
    composition = MaskedLinearCpu(checkpoint,
        linear=MaskedLinear(prefill_pruning="terminal", request_encoding="stage_packed", output_encoding="row_residues"),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    class Fixture:
        def remote(self, stage_id, activation):
            return stage_output(model.stages[stage_id], activation, 8)
        def lock(self):
            return {"checkpoint_digest": source.checkpoint_digest,
                    "binding_digest": compiled.digest, "composition_digest": composition.digest(),
                    "plan_digest": plan.digest, "schedule_digest": compiled.runtime_schedule_digest}
    fixture = Fixture()
    fixture.compiled, fixture.bundle, fixture.model = compiled, bundle, model
    fixture.plan, fixture.composition = plan, composition
    return fixture


def inflight_stage_ledger(rows, stages):
    """Explicit arithmetic, NOT packed/residue/HTTP body measurements."""
    result = {}
    for stage_id, count in rows.items():
        spec = stages[stage_id].spec
        result[stage_id] = {"rows": count, "macs": count * spec.in_features * spec.out_features,
            "clear_f32_input_output_bytes": 4 * count * (spec.in_features + spec.out_features),
            "unframed_i64_input_output_correction_projection_bytes":
                8 * count * (spec.in_features + 2 * spec.out_features)}
    return result


def inflight_packed_callback(stage, rows, *, chunk_rows=16):
    """Tensor payload projection with imported committed W8 row bounds only."""
    widths = bytes(stage.output_residue_bits)
    if (stage.spec.activation_bits != 8 or len(widths) != stage.spec.out_features
            or not widths or min(widths) < 1 or max(widths) > stage.seeded_profile.wire_bits):
        raise ValueError("invalid imported W8 residue row bounds")
    limit = min(chunk_rows, 4_000_000 // max(stage.spec.in_features, stage.spec.out_features))
    if rows < 1 or limit < 1:
        raise ValueError("invalid packed callback row bound")
    chunks = [min(limit, rows - first) for first in range(0, rows, limit)]
    upload = sum((n * stage.spec.in_features * max(widths) + 7) // 8 for n in chunks)
    residue = sum((n * sum(widths) + 7) // 8 for n in chunks)
    return {"rows": rows, "chunks": chunks, "input_bits": max(widths),
            "output_bits_sum": sum(widths), "client_inference_tensor_bytes": upload,
            "inference_client_tensor_bytes": residue,
            "fresh_preparation_inference_correction_tensor_bytes": residue,
            "total_tensor_payload_projection_bytes": upload + 2 * residue}


def inflight_worker(spec):
    try:
        with ExitStack() as cleanup:
            return inflight_worker_body(spec, cleanup)
    finally:
        inflight_verify_checkpoint(spec.get("checkpoint_pin"))
        inflight_verify(spec["frozen"])


def inflight_worker_body(spec, cleanup):
    import tempfile
    from collections import Counter
    from pllm.quantization import SymmetricPerRow
    from pllm.protocols import MaskedLinear
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.prefill_cache import ExactPrefillCache
    from pllm.runtime.benchmark_memory import process_memory
    from decoder_probe_support import DecoderFixture
    frozen = spec["frozen"]
    inflight_verify(frozen)
    pin = spec.get("checkpoint_pin")
    if spec["family"] == "real" and pin is None:
        raise ValueError("real worker requires pre-worker checkpoint pin")
    inflight_verify_checkpoint(pin)
    tokens, prefix, branches, outputs = (spec[k] for k in ("tokens", "prefix", "branches", "outputs"))
    with tempfile.TemporaryDirectory(prefix="pllm-inflight-") as folder:
        load_start, load_cpu = time.perf_counter(), time.process_time()
        if spec["family"] == "real":
            fixture = DecoderFixture(inputs=tokens, outputs=outputs)
            fixture.engine.prepared_output_encoding = "row_residues"
            fixture.bundle = ClientBundle.unpack(fixture.engine.client_bundle(fixture.source.manifest.id))
            fixture.composition = fixture.composition.with_params(
                quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
                linear=MaskedLinear(prefill_pruning="terminal", request_encoding="stage_packed", output_encoding="row_residues"))
            fixture.compiled = compile_runtime_model(fixture.plan, fixture.bundle, composition=fixture.composition)
        else:
            fixture = inflight_generated(Path(folder) / "model", spec["family"], tokens, outputs)
        load_metrics = {"wall_seconds": time.perf_counter() - load_start,
                        "cpu_seconds": time.process_time() - load_cpu,
                        "combined_process_memory": process_memory()}
        lock = fixture.lock()
        control_admission = inflight_control_admission(fixture, pin) if pin else None
        state_binding = fixture.compiled.runtime(fixture.remote)._state_binding_digest
        lock["state_binding_digest"] = state_binding
        client = object()
        # Public deterministic controls only; do not emit tokens, logits or KV.
        vocab = int(fixture.bundle.cfg["vocab_size"])
        common = [(i * 31 + 11) % vocab for i in range(prefix)]
        suffixes = [[(i * 7 + branch + 19) % vocab for i in range(tokens - prefix)]
                    for branch in range(branches)]
        cases = {}
        reference = {}
        configured_prefix = prefix
        for scenario in ("shared", "identical", "no_match"):
            prefix = tokens if scenario == "identical" else configured_prefix
            scenario_suffixes = [[] for _ in range(branches)] if scenario == "identical" else suffixes
            mode_records = {}
            modes = ["cold_independent", "inflight", "completed_cache",
                     "completed_cache_hit", "inflight_cache_hit"]
            if scenario == "shared":
                modes.append("inflight_failed_branch")
            for mode in modes:
                rows = Counter()
                phase_rows = {}
                clear_cpu = {}
                construction_cpu = {}
                phase = "setup"
                inject_failure = False
                failed_rows = {}
                callbacks = []
                def remote(stage_id, activation):
                    nonlocal inject_failure
                    count = int(np.asarray(activation).size // fixture.model.stages[stage_id].spec.in_features)
                    rows[stage_id] += count
                    phase_rows.setdefault(phase, Counter())[stage_id] += count
                    callbacks.append({"stage": stage_id, "phase": phase,
                        **inflight_packed_callback(fixture.model.stages[stage_id], count)})
                    clear_start = time.process_time()
                    try:
                        value = fixture.remote(stage_id, activation)
                    finally:
                        clear_cpu[phase] = clear_cpu.get(phase, 0.0) + time.process_time() - clear_start
                    if inject_failure:
                        inject_failure = False
                        raise RuntimeError("inflight injected branch result failure")
                    return value
                def runtime():
                    construct_start = time.process_time()
                    r = fixture.compiled.runtime(remote)
                    cleanup.callback(r.reset)
                    r.install_continuation(fixture.plan.continuation_schedule(fixture.composition))
                    construction_cpu[phase] = construction_cpu.get(phase, 0.0) + time.process_time() - construct_start
                    return r
                prefixes = [common + suffixes[0] if scenario == "identical" else common if scenario == "shared" else
                    [(t + b * 3) % vocab for t in common] for b in range(branches)]
                cfg = fixture.bundle.cfg
                kv_row = int(cfg["num_hidden_layers"]) * 2 * int(cfg["num_key_value_heads"]) * int(cfg["head_dim"]) * 4
                payload = prefix * kv_row + vocab * 4
                # Restore may allocate compiled maximum capacity, not active rows.
                branch_overlap = payload + 2 * (tokens + outputs) * kv_row + vocab * 4
                producer_overlap = branch_overlap
                owner = InflightPrefixOwner(client, lock["checkpoint_digest"], lock["composition_digest"],
                    state_binding, max_waiters=branches, max_nodes=branches,
                    max_bytes=min(256 << 20, producer_overlap * branches + branch_overlap * branches))
                cache = ExactPrefillCache(64 << 20, causal_reduction="prefix_f32")
                cleanup.callback(cache.clear)
                def produce(ids):
                    r = runtime()
                    try:
                        _, logits, _ = r.prepare_ids(ids)
                        return r.snapshot(), logits.copy()
                    finally:
                        r.reset()
                start, cpu = time.perf_counter(), time.process_time()
                leases = []
                if mode in {"inflight", "inflight_failed_branch"}:
                    async def admit():
                        gate = asyncio.Event()
                        async def acquire(ids):
                            async def execute():
                                await gate.wait()
                                return produce(ids)
                            lease = await owner.acquire(client, lock["checkpoint_digest"],
                                lock["composition_digest"], state_binding, ids, execute,
                                payload_bound=payload, branch_overlap=branch_overlap,
                                producer_overlap=producer_overlap)
                            cleanup.callback(lease.close)
                            return lease
                        tasks = [asyncio.create_task(acquire(ids)) for ids in prefixes]
                        await asyncio.sleep(0)
                        gate.set()
                        return await asyncio.gather(*tasks)
                    leases = asyncio.run(admit())
                elif mode in {"completed_cache", "completed_cache_hit", "inflight_cache_hit"}:
                    published = set()
                    for b, ids in enumerate(prefixes):
                        key = "shared" if scenario != "no_match" else str(b)
                        if key not in published:
                            saved, logits = produce(ids)
                            cleanup.callback(inflight_erase, saved, logits)
                            if not cache.put(key, saved, logits):
                                raise RuntimeError("completed cache admission failed")
                            inflight_erase(saved, logits)
                            published.add(key)
                setup_seconds, setup_cpu = time.perf_counter() - start, time.process_time() - cpu
                stage_rows_setup = dict(rows)
                parity = []
                branch_cpu = 0.0
                for b, (ids, suffix) in enumerate(zip(prefixes, scenario_suffixes)):
                    phase = "branch"
                    r = runtime()
                    begin = time.process_time()
                    if mode == "cold_independent":
                        _, logits, _ = r.prepare_ids(ids + suffix)
                    else:
                        if mode in {"inflight", "inflight_failed_branch"}:
                            saved, logits = leases[b].snapshot, leases[b].logits
                        else:
                            saved, logits = cache.get("shared" if scenario != "no_match" else str(b),
                                                      position=prefix, layers=r.layers)
                            cleanup.callback(inflight_erase, saved, logits)
                        saved_logits = logits
                        r.restore(saved)
                        if suffix:
                            inject_failure = mode == "inflight_failed_branch" and b == 0
                            before_failure = rows.copy()
                            try:
                                logits = r.continue_ids(suffix)[-1]
                            except RuntimeError as error:
                                if str(error) != "inflight injected branch result failure":
                                    raise
                                failed_rows = dict(rows - before_failure)
                                branch_cpu += time.process_time() - begin
                                leases[b].close()
                                r.reset()
                                parity.append({"injected_failed_branch": True,
                                    "private_branch_retired": True, "other_waiters_alive": owner.waiters > 0})
                                continue
                        if mode in {"completed_cache", "completed_cache_hit", "inflight_cache_hit"}:
                            logits = logits.copy()
                            inflight_erase(saved, saved_logits)
                    vectors, selected = [logits.copy()], [int(np.argmax(logits))]
                    for _ in range(outputs - 1):
                        logits = r.forward_ids([selected[-1]])[-1]
                        vectors.append(logits.copy())
                        selected.append(int(np.argmax(logits)))
                    branch_cpu += time.process_time() - begin
                    # Last sampled token remains pending, never included in KV.
                    snapshot = r.snapshot()
                    cleanup.callback(inflight_erase, snapshot, logits.copy())
                    expected_position = tokens + outputs - 1
                    if r.position != expected_position:
                        raise AssertionError("pending token promoted")
                    current = (vectors, selected, [(c.key.copy(), c.value.copy()) for c in snapshot.caches])
                    key = (scenario, b)
                    if mode == "cold_independent":
                        reference[key] = current
                    golden = reference[key]
                    equal = (selected == golden[1] and all(np.array_equal(x.view(np.uint32), y.view(np.uint32))
                        for x, y in zip(vectors, golden[0], strict=True)) and all(
                        np.array_equal(x.view(np.uint32), y.view(np.uint32))
                        for pair, old in zip(current[2], golden[2], strict=True)
                        for x, y in zip(pair, old, strict=True)))
                    if not equal:
                        raise AssertionError("complete logits/KV branch parity failed")
                    parity.append({"exact_logits_kv": equal, "position": r.position,
                                   "pending_token_not_executed": True})
                    inflight_erase(snapshot, logits.copy())
                    r.reset()
                for lease in leases:
                    lease.close()
                cache.clear()
                ledger = inflight_stage_ledger(rows, fixture.model.stages)
                body_macs = sum(item["macs"] for sid, item in ledger.items()
                    if fixture.model.stages[sid].spec.role not in {"token_lookup", "lm_head"})
                mode_records[mode] = {"observed_stage_ledger": ledger,
                    "stage_packed_row_residue_tensor_projection": {
                        "scope": "unframed tensor payload arithmetic; no masks, serialization, frames or wire measured",
                        "callback_row_sizes_and_rounding": callbacks,
                        "total_tensor_payload_projection_bytes": sum(c["total_tensor_payload_projection_bytes"] for c in callbacks),
                        "fresh_correction_tensor_projection_bytes": sum(c["fresh_preparation_inference_correction_tensor_bytes"] for c in callbacks),
                        "setup_tensor_payload_projection_bytes": sum(c["total_tensor_payload_projection_bytes"] for c in callbacks if c["phase"] == "setup"),
                        "response_tensor_payload_projection_bytes": sum(c["total_tensor_payload_projection_bytes"] for c in callbacks if c["phase"] == "branch")},
                    "setup_observed_rows": stage_rows_setup,
                    "response_only_stage_ledger": inflight_stage_ledger(
                        phase_rows.get("branch", {}), fixture.model.stages),
                    "cache_already_populated": mode in {"completed_cache_hit", "inflight_cache_hit"},
                    "cache_setup_charged_in_horizon": True,
                    "phase_rows": {k: dict(v) for k, v in phase_rows.items()},
                    "setup_wall_seconds": setup_seconds, "setup_cpu_seconds": setup_cpu,
                    "branch_combined_cpu_seconds": branch_cpu,
                    "branch_runtime_cpu_excluding_clear_executor_seconds": branch_cpu - clear_cpu.get("branch", 0.0),
                    "clear_executor_cpu_seconds_by_phase": clear_cpu,
                    "runtime_construction_cpu_seconds_by_phase": construction_cpu,
                    "parity": parity,
                    "combined_clear_executor_process_memory": process_memory(),
                    "client_only_process_memory": None,
                    "owner": {k: getattr(owner, k) for k in ("executions", "peak_bytes", "copy_bytes",
                        "copy_cpu_seconds", "wait_seconds", "retired", "bytes", "waiters")},
                    "proposed_remote_body_macs": body_macs,
                    "proposed_remote_body_fraction": 1.0 if body_macs else None,
                    "measured_remote_body_fraction": 0.0,
                    "whole_response_traffic_bytes": None, "bundle_delivery_bytes": None,
                    "proposed_fresh_material_per_stage_rows": sum(rows.values()),
                    "failed_branch_executed_stage_rows": failed_rows,
                    "failed_branch_material_demand_units_not_reusable": sum(failed_rows.values()),
                    "material_scope": "arithmetic demand units only; no provider masks/tickets issued",
                    "fresh_material_required_per_operation": True}
                cleanup.close()
            if mode_records["inflight"]["observed_stage_ledger"] != mode_records["completed_cache"]["observed_stage_ledger"]:
                raise AssertionError("inflight invented extra saving against completed cache")
            if (mode_records["inflight_cache_hit"]["response_only_stage_ledger"] !=
                    mode_records["completed_cache_hit"]["response_only_stage_ledger"]
                    or mode_records["inflight_cache_hit"]["owner"]["executions"] != 0):
                raise AssertionError("completed cache hit must bypass in-flight work")
            cases[scenario] = mode_records
        inflight_verify(frozen)
        if pin:
            inflight_control_admission(fixture, pin)
        return {"schema": "pllm.inflight_prefix.v1", "status": "local_clear_screen",
            "source": lock, "family": spec["family"], "workload": {k: spec[k] for k in ("tokens", "prefix", "branches", "outputs")},
            "source_construction": load_metrics,
            "historical_control_admission": control_admission,
            "checkpoint_pin": pin,
            "execution_contract": {"prefill_pruning": "terminal", "request_encoding": "stage_packed", "output_encoding": "row_residues"},
            "cases": cases, "imported_sources": inflight_imports(frozen),
            "dependency_versions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
            "completed_cache_incremental_saving": 0,
            "live_provider_or_sdk_claim": False, "frozen_sources": frozen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--family", choices=("qwen2", "qwen3", "llama", "real"), default="qwen2")
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--prefix", type=int, default=8)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--outputs", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(inflight_worker(json.load(sys.stdin))))
        return
    if not (0 < args.prefix < args.tokens <= 150 and 1 <= args.branches <= 8 and 1 <= args.outputs <= 8):
        parser.error("bounded cohort requires prefix < tokens <= 150, branches/outputs <= 8")
    if args.output is None:
        parser.error("--output required")
    with inflight_fresh_output(args.output) as output:
        from frontier_probe_support import fresh_workers
        spec = {k: getattr(args, k) for k in ("family", "tokens", "prefix", "branches", "outputs")}
        with fresh_workers(Path(__file__).resolve(), memory_bytes=(6 << 30) if args.family == "real" else (512 << 20),
                           timeout=1200 if args.family == "real" else 240) as (run, guard, host):
            frozen = inflight_freeze()
            pin = inflight_checkpoint_pin() if args.family == "real" else None
            spec.update(frozen=frozen, checkpoint_pin=pin)
            try:
                inflight_verify_checkpoint(pin)
                result = run(spec)
            finally:
                inflight_verify(frozen)
                inflight_verify_checkpoint(pin)
            result["build"] = run.build
            result["memory_guard"] = {"maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                                      "error": guard.error}
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
