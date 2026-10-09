"""Bounded synthetic state mechanics; never a decoder continuation credential."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

import numpy as np

from pllm._benchmark_contract import TargetUnavailable, canonical, digest
from pllm._benchmark_worker import peak_rss


class SealedStateFixture:
    """Test-local AEAD generations with a separately retained trusted revision.

    The caller owns the key and head. Neither goes into benchmark evidence. This
    tests storage mechanics, not persistent anti-rollback key management or a
    runtime snapshot seal. Private model state is deliberately not accepted.
    """

    def __init__(self, directory, key, context, layout):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self.root, self.context, self.layout = Path(directory), context, layout
        if len(key) != 32 or not layout or len(layout) > 2048:
            raise ValueError("invalid sealed fixture contract")
        self.sizes = [math.prod(row["shape"]) * 4 for row in layout]
        if max(self.sizes) > 64 << 20 or sum(self.sizes) > 256 << 20:
            raise ValueError("sealed fixture exceeds payload admission")
        self.aead = AESGCM(key)
        self.root.mkdir(parents=True, exist_ok=True)

    def _aad(self, revision, index):
        return canonical({"schema": "pllm.synthetic_state_fixture.v1", "context": self.context,
                          "revision": revision, "index": index, "layout": self.layout[index]})

    def _write_record(self, out, value, revision, index):
        nonce = secrets.token_bytes(12)
        sealed = nonce + self.aead.encrypt(nonce, value.tobytes(), self._aad(revision, index))
        out.write(sealed)
        return hashlib.sha256(sealed).hexdigest()

    def _read_record(self, source, revision, index, expected):
        sealed = source.read(self.sizes[index] + 29)
        if (len(sealed) != self.sizes[index] + 28
                or hashlib.sha256(sealed).hexdigest() != expected):
            raise ValueError("sealed state frame differs")
        value = self.aead.decrypt(sealed[:12], sealed[12:], self._aad(revision, index))
        return np.frombuffer(value, dtype=np.float32).reshape(self.layout[index]["shape"])

    def commit(self, values, revision, *, fail_after=None):
        if type(revision) is not int or revision < 0:
            raise ValueError("invalid revision")
        pending = self.root / f"pending-{secrets.token_hex(8)}"
        destination = self.root / str(revision)
        pending.mkdir(mode=0o700)
        try:
            hashes = []
            for index, value in enumerate(values):
                if index >= len(self.layout) or value.dtype != np.float32 or list(value.shape) != self.layout[index]["shape"]:
                    raise ValueError("state layout differs")
                with (pending / str(index)).open("xb") as out:
                    hashes.append(self._write_record(out, value, revision, index))
                    out.flush()
                    os.fsync(out.fileno())
                if fail_after == index:
                    raise ValueError("injected transaction failure")
            if len(hashes) != len(self.layout):
                raise ValueError("incomplete state generation")
            manifest = canonical({"revision": revision, "context": self.context, "hashes": hashes})
            with (pending / "manifest").open("xb") as out:
                out.write(manifest)
                out.flush()
                os.fsync(out.fileno())
            if destination.exists():
                raise ValueError("state revision already exists")
            os.rename(pending, destination)
            return {"revision": revision, "manifest": hashlib.sha256(manifest).hexdigest()}
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    def read(self, head):
        revision = head["revision"]
        if type(revision) is not int or revision < 0:
            raise ValueError("invalid trusted revision")
        root = self.root / str(revision)
        with (root / "manifest").open("rb") as source:
            raw = source.read((1 << 20) + 1)
        if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != head["manifest"]:
            raise ValueError("state manifest differs from trusted head")
        document = json.loads(raw)
        if (set(document) != {"revision", "context", "hashes"} or document["revision"] != revision
                or document["context"] != self.context or len(document["hashes"]) != len(self.layout)):
            raise ValueError("state binding differs")
        for index, row in enumerate(self.layout):
            with (root / str(index)).open("rb") as source:
                value = self._read_record(source, revision, index, document["hashes"][index])
            yield value


def _values(layout, seed):
    # Per-record seeds let streaming and resident layouts construct the same bits.
    for index, row in enumerate(layout):
        yield np.random.default_rng(seed + index).standard_normal(row["shape"], dtype=np.float32)


def _consume(values, attention=False):
    from .semantic_attention import semantic_attention_scores, semantic_attention_values
    state = hashlib.sha256()
    iterator = iter(values)
    for keys in iterator:
        values = next(iterator)
        if attention:
            query = keys[..., -1:, :].copy()
            scores = semantic_attention_scores(query, keys, group=1, layout="batch_kv_heads_sequence_feature")
            scores *= np.float32(keys.shape[-1] ** -0.5)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            result = semantic_attention_values(probabilities, values, group=1,
                                               layout="batch_kv_heads_sequence_feature")
            state.update(result.tobytes())
        else:
            state.update(keys.tobytes())
            state.update(values.tobytes())
        del keys, values
    return state.hexdigest()


def _projection_values(p, context):
    from .benchmark_targets import state_layout
    a = p["assumptions"]
    if p["construction"] == "layer_kv":
        layout = state_layout(context[2], context[0].budget.max_input_tokens)
        layers = {}
        for row in layout:
            layers[row["layer"]] = layers.get(row["layer"], 0) + math.prod(row["shape"]) * 4
        if a["live_layers"] > len(layers):
            raise TargetUnavailable("live_layers_exceed_geometry")
        resident = a["fixed_bytes"] + sum(layers.values())
        candidate = a["fixed_bytes"] + sum(sorted(layers.values(), reverse=True)[:a["live_layers"]])
        details = dict(resident_bytes=resident, optimistic_candidate_bytes=candidate,
                       optimistic_ratio=resident / candidate, layers=len(layers), state_bytes=sum(layers.values()))
    else:
        resident = a["bootstrap_bytes"] + a["resumptions"] * a["control_bytes"]
        candidate = a["bootstrap_bytes"] + a["publication_bytes"] + a["resumptions"] * a["resumed_bytes"]
        delta = a["control_bytes"] - 10 * a["resumed_bytes"]
        numerator = 9 * a["bootstrap_bytes"] + 10 * a["publication_bytes"]
        details = dict(control_bytes=resident, candidate_bytes=candidate,
            optimistic_ratio=None if candidate == 0 else resident / candidate,
            minimum_tenfold_resumptions=None if delta <= 0 else (numerator + delta - 1) // delta)
    return candidate, details


def _projection(p, m):
    from .benchmark_targets import model_context

    def execute(_):
        context = model_context(p, m)
        return context, _projection_values(p, context)

    context, (candidate, details) = m.run(execute)[-1]
    m.details.update(details)
    m.metric("projection.candidate_bytes", candidate, kind="allocated", origin="estimate", scope="hypothetical_owners")
    m.fixture = {"kind": "projection", "digest": context[2].digest, "metadata": {"assumptions": p["assumptions"]}}
    m.oracle = {"kind": "native_geometry_and_public_arithmetic", "outcome": "passed", "checked_values": 1}
    m.comparison.update(workload_digest=digest(p), output_digest=digest(details),
                        measurement_digest=digest({"scope": "hypothetical_projection"}))
    m.limitations += ["Declared fixed costs exclude unspecified scratch and overlap; optimistic ratios are not measured RSS or serving speed.",
                     "Target timings measure geometry validation and cost calculation, not execution of the projected construction."]


def execute_state(target, p, m):
    from .benchmark_targets import model_context, state_layout
    if target.kind == "projection":
        return _projection(p, m)
    context = model_context(p, m)
    layout = sorted(state_layout(context[2], p["tokens"]), key=lambda row: (row["layer"], row["kind"]))
    size = sum(math.prod(row["shape"]) * 4 for row in layout)
    slab = max(math.prod(row["shape"]) * 4 for row in layout)
    if size > 256 << 20 or slab > 64 << 20:
        raise TargetUnavailable("state_fixture_payload_limit", status="blocked")
    if p["operation"] == "restart" and p["storage"] not in {"sealed_local", "sealed_stream"}:
        raise TargetUnavailable("restart_requires_local_sealed_fixture")
    m.admit((160 << 20) + size + 8 * slab)
    binding = digest({"source": m.comparison["source_digest"], "plan": context[2].digest, "layout": layout})
    start, cpu = time.perf_counter(), time.process_time()
    key = secrets.token_bytes(32)
    if p["storage"] == "sealed_stream":
        from .benchmark_sealed_state import StreamingSealedStateFixture
        store = StreamingSealedStateFixture(m.directory / "state", key, binding, layout)
    else:
        store = SealedStateFixture(m.directory / "state", key, binding, layout) if p["storage"] == "sealed_local" else None
    head = store.commit(_values(layout, p["seed"]), 0) if store else None
    resident = None if store else list(_values(layout, p["seed"]))
    m.details["prerequisites"] = {"wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu, "state_bytes": size,
        "written_file_bytes": sum(f.stat().st_size for f in store.root.rglob("*") if f.is_file()) if store else 0}
    children = []
    def execute(_):
        if p["operation"] == "restart":
            request = dict(directory=str(store.root), key=key.hex(), context=binding, layout=layout,
                           head=head, storage=p["storage"])
            process = subprocess.run([sys.executable, "-I", "-m", __name__], input=canonical(request),
                                     capture_output=True, check=True, timeout=m.sampling.timeout_seconds)
            child = json.loads(process.stdout)
            children.append(child)
            return child["digest"]
        return _consume(store.read(head) if store else resident, p["operation"] == "attention")
    values = m.run(execute)
    m.details["target_peak_rss_bytes"], m.details["target_cpu_from_birth_seconds"] = peak_rss(), time.process_time()
    if children:
        children = children[m.sampling.warmups:]
        m.details["timings"]["cpu_seconds"] = [cpu + child["cpu_seconds"]
            for cpu, child in zip(m.details["timings"]["cpu_seconds"], children, strict=True)]
        m.details["target_cpu_owners"] = ["client", "restore_child"]
        m.metric("restore_child.peak_rss", max(child["peak_rss_bytes"] for child in children),
                 kind="memory", role="restore_child", window="process_birth_through_restore",
                 scope="process_lifetime_high_water")
    expected = _consume(_values(layout, p["seed"]), p["operation"] == "attention")
    if any(value != expected for value in values):
        raise ValueError("synthetic state parity failed")
    if store:
        try:
            store.commit(_values(layout, p["seed"] + 1), 1, fail_after=0)
        except ValueError:
            pass
        if _consume(store.read(head)) != _consume(_values(layout, p["seed"])):
            raise ValueError("failed transaction changed prior generation")
    m.fixture = {"kind": "synthetic", "digest": digest({"layout": layout, "seed": p["seed"]}),
                 "metadata": {"plan_digest": context[2].digest, "tokens": p["tokens"]}}
    checked = (sum(row["shape"][0] * row["shape"][1] * row["shape"][3]
                   for row in layout if row["kind"] == "key") if p["operation"] == "attention" else size // 4)
    m.oracle = {"kind": "exact_synthetic_storage_or_same_kernel_attention", "outcome": "passed", "checked_values": checked * len(values)}
    m.comparison.update(state_digest=binding, workload_digest=digest({"tokens": p["tokens"], "operation": p["operation"], "seed": p["seed"]}),
        output_digest=m.commitment(expected), measurement_digest=digest({"scope": "synthetic_state", "operation": p["operation"]}))
    m.metric("state.payload_bytes", size, kind="allocated", origin="deterministic", scope="float32_kv_payload")
    m.metric("state.publication_cpu", m.details["prerequisites"]["cpu_seconds"], kind="cpu",
             window="fixture_publication", scope="synthetic_state_initialization")
    m.metric("state.written_bytes", m.details["prerequisites"]["written_file_bytes"], kind="allocated",
              origin="deterministic", scope="private_disk_payload")
    if p["storage"] == "sealed_stream":
        m.details["codec"] = {"encoding": "aes_gcm_existing_fixture_v1", "chunk_bytes": store._CHUNK,
            "maximum_record_bytes": slab, "plaintext_record_array_bound_bytes": 3 * (slab + 15),
            "scope": "codec arrays only; attention workspaces and caller-owned values are additional"}
    m.limitations += ["Synthetic full-KV storage mechanics; no real prefill provenance, logits, pending tokens, snapshot authorization or reusable protocol material.",
        "Trusted revision/key remain in the owner; this is not a persistent anti-rollback key service.",
        "Restore timing excludes separately reported initial writes; parent and restart-child peaks stay separate. Target CPU includes both processes.",
        "Attention parity uses the same compiled primitive on both storage layouts, not an independent floating-point or model-quality oracle."]


if __name__ == "__main__":
    request = json.loads(sys.stdin.buffer.read((1 << 20) + 1))
    cls = SealedStateFixture
    if request.get("storage") == "sealed_stream":
        from .benchmark_sealed_state import StreamingSealedStateFixture
        cls = StreamingSealedStateFixture
    elif request.get("storage", "sealed_local") != "sealed_local":
        raise ValueError("unsupported restart storage")
    store = cls(request["directory"], bytes.fromhex(request["key"]), request["context"], request["layout"])
    result = _consume(store.read(request["head"]))
    print(json.dumps({"digest": result, "cpu_seconds": time.process_time(), "peak_rss_bytes": peak_rss()}))
