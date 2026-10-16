#!/usr/bin/env python3
"""Bounded exact-reuse lifecycle simulator; no model imports or performance claims."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 128
VOCAB = 11
MAX_NODE_BYTES = 16384
MAX_NODES = 128
CONTEXT = ("source", "prefix_f32", "full-kv", "verification-none", "policy-v1")
V1_SHA256 = "fb1e62c8c2b579658f355aa75aa06a32dd0ccfbf67153291f8a05827836e56c2"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def validate_ids(ids, *, allow_empty=False):
    if (type(ids) not in (list, tuple) or not (0 if allow_empty else 1) <= len(ids) <= LIMIT
            or any(type(token) is not int or not 0 <= token < VOCAB for token in ids)):
        raise ValueError("token sequence bound or type")


@dataclass(frozen=True)
class State:
    ids: tuple[int, ...] = ()
    kv: tuple[tuple[int, int], ...] = ()
    logits: tuple[int, ...] = ()
    context: tuple[str, ...] = CONTEXT


def advance(state, token):
    """Incremental synthetic causal executor, with position-dependent keys."""
    validate_ids(state.ids, allow_empty=True)
    if type(token) is not int or not 0 <= token < VOCAB or len(state.ids) >= LIMIT:
        raise ValueError("token or capacity")
    pos = len(state.ids)
    pair = ((17 * token + 3 * pos) % 257, (token * token + pos + 1) % 263)
    kv = (*state.kv, pair)
    logits = tuple(1 + sum((k + j + 1) * v for k, v in kv) % 997 for j in range(VOCAB))
    return State((*state.ids, token), kv, logits, state.context)


def oracle(ids, context=CONTEXT):
    """Independent whole-history oracle: never calls advance or restores a node."""
    validate_ids(ids, allow_empty=True)
    keys = [(17 * token + 3 * p) % 257 for p, token in enumerate(ids)]
    vals = [(token * token + p + 1) % 263 for p, token in enumerate(ids)]
    logits = []
    for column in range(VOCAB):
        total = 0
        for index in range(len(ids)):
            total += (keys[index] + column + 1) * vals[index]
        logits.append(1 + total % 997)
    return State(tuple(ids), tuple(zip(keys, vals)), tuple(logits), context)


def sample(logits, rng):
    rng = (1664525 * rng + 1013904223) & 0xFFFFFFFF
    remaining = rng % sum(logits)
    for token, weight in enumerate(logits):
        if remaining < weight:
            return token, rng
        remaining -= weight
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Result:
    state: State
    output: tuple[int, ...]
    pending: int
    rng: int


@dataclass
class Ledger:
    links: dict[str, int] = field(default_factory=lambda: {
        name: 0 for name in ("artifact->client", "client->preparation", "preparation->client",
                            "preparation->inference",
                            "client->inference", "inference->client", "client->storage",
                            "storage->client", "client->application", "application->client")})
    issued: int = 0
    claimed: int = 0
    burned: int = 0
    sessions: int = 0
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    validation_bytes: int = 0
    disk_read_bytes: int = 0
    disk_written_bytes: int = 0
    retained_disk_bytes: int = 0
    peak_logical_state_bytes: int = 0

    def setup(self):
        # Stipulated tiny topology, identical cold artifact in both arms.
        self.links["artifact->client"] += 4096

    def execute(self, rows=1, burn=0):
        # A synthetic already-pruned stage-row costs 16 upload, 24 reply,
        # 24 correction bytes exclusively Preparation -> Inference. Four
        # stipulated 32-byte fixture controls are NOT actual runtime framing.
        if type(rows) is not int or rows < 1 or type(burn) is not int or burn < 0:
            raise ValueError("row counts must be plain nonnegative integers; rows positive")
        self.issued += rows + burn
        self.claimed += rows
        self.burned += burn
        self.sessions += 1
        self.links["client->preparation"] += 32
        self.links["preparation->client"] += 32
        self.links["preparation->inference"] += 24 * (rows + burn)
        self.links["client->inference"] += 32 + 16 * rows
        self.links["inference->client"] += 32 + 24 * rows

    def audit(self):
        if self.issued != self.claimed + self.burned:
            raise ValueError("row conservation")
        if any(v < 0 for v in self.links.values()):
            raise ValueError("negative directed traffic")
        expected = {
            "client->preparation": 32 * self.sessions,
            "preparation->client": 32 * self.sessions,
            "preparation->inference": 24 * self.issued,
            "client->inference": 32 * self.sessions + 16 * self.claimed,
            "inference->client": 32 * self.sessions + 24 * self.claimed,
        }
        if any(self.links.get(link) != size for link, size in expected.items()):
            raise ValueError("directed fixture conservation: correction q belongs exclusively to inference")

    def report(self):
        self.audit()
        return {**asdict(self), "all_link_bytes": sum(self.links.values()),
                "traffic_scope": "declared fixture payloads and stipulated controls only; not complete runtime traffic",
                "actual_runtime_controls_bytes": None,
                "unmodeled_runtime_directed_bytes": {"inference->preparation_acknowledgement": None},
                "complete_runtime_ledger": False,
                "client_cpu_seconds": None, "validation_cpu_seconds": None,
                "rss_bytes": None, "physical_disk_bytes": None,
                "whole_client_256_mib_qualified": False}


class Store:
    """Trusted-local immutable HMAC fixture. Not production encrypted storage."""
    def __init__(self, ledger, max_nodes=128):
        if type(max_nodes) is not int or not 1 <= max_nodes <= MAX_NODES:
            raise ValueError("node bound")
        self.ledger = ledger
        self.nodes = {}
        self.key = b"synthetic fixture key only"
        self.max_nodes = max_nodes
        self.head = 0

    def reopen(self, trusted_head):
        if trusted_head != self.head:
            raise ValueError("trusted head differs; no disk-selected latest")
        # Synthetic trusted-index read, independent of numeric page lookup.
        size = len(encode({"head": self.head, "paths": list(self.nodes)}))
        self.ledger.disk_read_bytes += size
        self.ledger.validation_bytes += size
        other = Store(self.ledger, self.max_nodes)
        other.nodes, other.head = self.nodes.copy(), self.head
        return other

    def publish(self, state, *, acknowledged=True):
        if type(state) is not State:
            raise ValueError("exact State required")
        validate_ids(state.ids)
        if (type(state.context) is not tuple or len(state.context) > 8
                or any(type(value) is not str or len(value) > 128 for value in state.context)):
            raise ValueError("context bound")
        if not acknowledged:
            raise ValueError("unretired provider material")
        if len(self.nodes) >= self.max_nodes and state.ids not in self.nodes:
            raise ValueError("node bound")
        if state != oracle(state.ids, state.context):
            raise ValueError("invalid exact state")
        raw = encode(asdict(state))
        if len(raw) > MAX_NODE_BYTES:
            raise ValueError("manifest bound")
        tag = hmac.digest(self.key, raw, "sha256")
        old = self.nodes.get(state.ids)
        if old is None:
            self.nodes[state.ids] = (state, raw, tag)
            # Fixture manifest carries an inspection copy of numeric state;
            # separate logical KV pages are charged as well (no free pages).
            size = len(raw) + len(tag) + len(state.ids) * 8
            self.ledger.disk_written_bytes += size
            self.ledger.retained_disk_bytes += size
            self.head += 1

    def lookup(self, ids, *, parent=None, delta=False, trusted_head=None):
        validate_ids(ids)
        self.ledger.lookups += 1
        if trusted_head is not None and trusted_head != self.head:
            raise ValueError("trusted head differs; no disk-selected latest")
        node = self.nodes.get(tuple(ids))
        if node is None:
            self.ledger.misses += 1
            return None
        state, raw, tag = node
        # Node manifest authenticated even on delta hits. HMAC fixture represents
        # trusted immutable-page binding, not a proof of model execution.
        manifest = len(raw) + len(tag)
        self.ledger.disk_read_bytes += manifest
        self.ledger.validation_bytes += manifest
        if not hmac.compare_digest(hmac.digest(self.key, raw, "sha256"), tag):
            raise ValueError("node authentication")
        if encode(asdict(state)) != raw or state.context != CONTEXT:
            raise ValueError("source/numeric/KV/verification/policy binding")
        row_bytes = 8  # Two synthetic uint32 words per KV row.
        if delta and parent is not None:
            if (state.ids[:-1] != parent.ids or parent.context != state.context
                    or state.kv[:-1] != parent.kv or parent != oracle(parent.ids)):
                raise ValueError("parent seal/state mismatch")
            page_bytes = row_bytes
        else:
            page_bytes = len(state.ids) * row_bytes
        self.ledger.disk_read_bytes += page_bytes
        self.ledger.validation_bytes += page_bytes
        self.ledger.hits += 1
        self.ledger.peak_logical_state_bytes = max(
            self.ledger.peak_logical_state_bytes,
            2 * (len(state.ids) * row_bytes + 4 * VOCAB) + manifest)
        return state


def installed_positions(total, budget_rows):
    """Mirror put_prefixes selection/limited-copy preflight, not a new cut API."""
    positions = {total - 1, *(1 << bit for bit in range(total.bit_length())
                             if 1 << bit < total), *range(8, total, 8)}
    limit = min(total, budget_rows)
    if limit < 1 or total < 2:
        return (), limit
    if limit < total:
        positions = {p for p in positions if p <= limit}
        positions.add(limit)
    return tuple(sorted(positions)), limit


class ResidentPrefixes:
    """Synthetic installed eight-row shared-block/LRU cache, identical in both arms.

    Keys represent source-bound exact paths; immutable block contents represent
    full-KV identity. Prefix logits are optional, as in the actual cache.
    """
    def __init__(self, max_bytes=32768, max_entries=4096):
        if type(max_bytes) is not int or not 1 <= max_bytes <= 256 << 20:
            raise ValueError("prefix byte bound")
        if type(max_entries) is not int or not 1 <= max_entries <= 4096:
            raise ValueError("prefix entry bound")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries, self.blocks = OrderedDict(), {}
        self.bytes = self.peak_bytes = self.publications = self.capture_bytes = 0
        self.immutable_copy_bytes = self.restore_copy_bytes = 0

    def remove(self, path):
        refs, logits, _ = self.entries.pop(path)
        self.bytes -= 0 if logits is None else 4 * VOCAB
        for block, _ in refs:
            self.blocks[block][1] -= 1
            if self.blocks[block][1] == 0:
                self.bytes -= self.blocks.pop(block)[0]

    def put(self, state, position, prepared, *, logits=None):
        refs = tuple((block, min(8, position - offset))
                     for offset, block in zip(range(0, position, 8), prepared))
        path = state.ids[:position]
        old = self.entries.get(path)
        if (logits is None and old is not None and old[2] == len(state.ids)
                and tuple(row for block, length in old[0] for row in block[:length])
                == tuple(row for block, length in refs for row in block[:length])):
            logits = old[1]
        payload = 0 if logits is None else 4 * VOCAB
        self.immutable_copy_bytes += payload
        if payload + sum(len(block) * 8 for block, _ in refs) > self.max_bytes:
            return False
        if path in self.entries:
            self.remove(path)
        while (self.bytes + payload + sum(len(block) * 8 for block, _ in refs
                                         if block not in self.blocks) > self.max_bytes
               or len(self.entries) >= self.max_entries):
            self.remove(next(iter(self.entries)))
        for block, _ in refs:
            if block not in self.blocks:
                self.blocks[block] = [len(block) * 8, 0]
                self.bytes += len(block) * 8
                self.immutable_copy_bytes += len(block) * 8
            self.blocks[block][1] += 1
        self.entries[path] = (refs, logits, len(state.ids))
        self.bytes += payload
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.publications += 1
        return True

    def publish_prefill(self, state, extra_cut=None):
        positions, limit = installed_positions(len(state.ids), self.max_bytes // 8)
        prepared = tuple(state.kv[start:min(start + 8, limit)] for start in range(0, limit, 8))
        self.capture_bytes += limit * 8  # Conservative per-call candidate-copy charge.
        for position in positions:
            self.put(state, position, prepared)
        if extra_cut is not None and extra_cut not in positions and extra_cut <= limit:
            # Same existing _prefix lineage/optional-logit mechanics. Extra
            # publication metadata is separate, not a new native-cut ability.
            self.put(state, extra_cut, prepared)
        if len(state.ids) * 8 + 4 * VOCAB <= self.max_bytes:
            full = tuple(state.kv[start:start + 8] for start in range(0, len(state.ids), 8))
            self.capture_bytes += len(state.ids) * 8
            self.put(state, len(state.ids), full, logits=state.logits)

    def lookup(self, ids):
        exact = tuple(ids)
        # Actual get removes an exact entry lacking required root logits before
        # longest_prefix probes strictly shorter entries.
        if exact in self.entries and self.entries[exact][1] is None:
            self.remove(exact)
        for position in range(len(ids), 0, -1):
            path = tuple(ids[:position])
            entry = self.entries.get(path)
            if entry is None or position == len(ids) and entry[1] is None:
                continue
            self.entries.move_to_end(path)
            rows = tuple(row for block, length in entry[0] for row in block[:length])
            checked = oracle(path)
            assert rows == checked.kv
            self.restore_copy_bytes += position * 8 + (4 * VOCAB if entry[1] is not None else 0)
            # Toy oracle supplies a comparison witness, not free production
            # logits. For proper prefixes suffix execution ignores these logits.
            return checked
        return None


def run(prompt, seed, outputs, ledger, store=None, *, cut=None, delta=False, ordinary=False,
        prefixes=None):
    """Optimized control has completed-prefix and exact decode hits already."""
    validate_ids(prompt)
    if (type(outputs) is not int or not 1 <= outputs <= LIMIT
            or len(prompt) + outputs - 1 > LIMIT
            or type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF
            or (cut is not None and (type(cut) is not int or not 1 <= cut <= len(prompt)))):
        raise ValueError("lifecycle bound")
    state = State()
    if prefixes is not None:
        resident = prefixes.lookup(prompt)
        if resident is not None:
            state = resident
    if store is not None:
        # Installed resident longest-prefix behavior is part of the control.
        for size in range(len(prompt), 0, -1):
            if tuple(prompt[:size]) in store.nodes:
                if size >= len(state.ids):
                    state = store.lookup(prompt[:size])
                break
    missing = len(prompt) - len(state.ids)
    if ordinary and missing + outputs - 1:
        ledger.execute(missing + outputs - 1)
    elif missing:
        ledger.execute(missing)
    for token in prompt[len(state.ids):]:
        state = advance(state, token)
    if prefixes is not None and missing:
        prefixes.publish_prefill(state, extra_cut=cut)
    if store is not None:
        store.publish(state)
    rng, emitted = seed, []
    for selection in range(outputs):
        token, rng = sample(state.logits, rng)
        emitted.append(token)
        if selection == outputs - 1:
            break  # Last sampled selection is pending, not executed KV.
        hit = (None if store is None or ordinary else
               store.lookup((*state.ids, token), parent=state, delta=delta))
        if hit is None:
            if not ordinary:
                ledger.execute()
            state = advance(state, token)
            if store is not None:
                store.publish(state)
        else:
            state = hit
    return Result(state, tuple(emitted), emitted[-1], rng)


def reference(prompt, seed, outputs):
    ids, emitted, rng = list(prompt), [], seed
    for index in range(outputs):
        weights = oracle(ids).logits
        rng = (rng * 1664525 + 1013904223) % (2 ** 32)
        draw, cumulative = rng % sum(weights), 0
        for token in range(VOCAB):
            cumulative += weights[token]
            if cumulative > draw:
                break
        emitted.append(token)
        if index < outputs - 1:
            ids.append(token)
    return Result(oracle(ids), tuple(emitted), emitted[-1], rng)


class Receipts:
    """Application idempotency transaction, never a fresh-generation cache."""
    def __init__(self, ledger):
        self.ledger, self.results = ledger, {}

    def call(self, operation, request, execute, *, fresh=False):
        request_bytes = encode(request)
        if len(request_bytes) > MAX_NODE_BYTES or not isinstance(operation, str) or len(operation) > 128:
            raise ValueError("receipt request bound")
        self.ledger.links["application->client"] += len(request_bytes)
        old = self.results.get(operation)
        if old is not None:
            if fresh or old[0] != digest(request):
                raise ValueError("operation conflict; fresh invocation needs new identity")
            raw = encode(asdict(old[1]))
            self.ledger.disk_read_bytes += len(raw)
            self.ledger.validation_bytes += len(raw)
            result = old[1]
        else:
            if len(self.results) >= 32:
                raise ValueError("receipt bound")
            result = execute()
            raw = encode(asdict(result))
            if len(raw) > MAX_NODE_BYTES:
                raise ValueError("receipt result bound")
            self.ledger.disk_written_bytes += len(raw)
            self.ledger.retained_disk_bytes += len(raw)
            self.results[operation] = (digest(request), result)
        self.ledger.links["client->application"] += len(encode(asdict(result)))
        return result


def workload(name, candidate, *, ordinary=False):
    ledger = Ledger()
    ledger.setup()
    store = Store(ledger)
    prefixes = ResidentPrefixes()
    prefix = [1, 2, 3, 4, 5, 6, 7, 8]
    prompts = {
        "fresh": [prefix + [1, 1]],
        "repeat": [prefix + [1, 1]] * 3,
        "branch": [prefix + [1, 1], prefix + [2, 2], prefix + [3, 3]],
        "restart": [prefix + [1, 1]] * 2,
        "unrelated": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    }[name]
    results = []
    for index, prompt in enumerate(prompts):
        # Durable installed control already survives restart. No replay strawman.
        if name == "restart" and index:
            store = store.reopen(store.head)
            prefixes = ResidentPrefixes()
        result = run(prompt, 7, 4, ledger, store,
                     delta=candidate, ordinary=ordinary, prefixes=prefixes)
        assert result == reference(prompt, 7, 4)
        assert len(result.state.ids) == len(prompt) + 3
        results.append(digest(asdict(result)))
    return {"ledger": ledger.report(), "result_commitments": results,
            "prefix_owner": {"payload_bytes": prefixes.bytes, "peak_payload_bytes": prefixes.peak_bytes,
                             "capture_copy_bytes": prefixes.capture_bytes,
                             "immutable_copy_bytes": prefixes.immutable_copy_bytes,
                             "restore_copy_bytes": prefixes.restore_copy_bytes,
                             "publications": prefixes.publications},
            "policy": "ordinary-full-inventory" if ordinary else "installed-one-selection-cache"}


def best_workload(name, candidate):
    alternatives = [workload(name, candidate, ordinary=p) for p in (False, True)]
    best = min(alternatives, key=lambda a: (a["ledger"]["all_link_bytes"],
                                          a["ledger"]["validation_bytes"]))
    return {**best, "policy_selection": "oracle lower-byte envelope; not a deployable predictor",
            "all_policy_ledgers": {a["policy"]: a["ledger"] for a in alternatives}}


def report():
    workloads = {}
    for name in ("fresh", "repeat", "branch", "restart", "unrelated"):
        control, candidate = best_workload(name, False), best_workload(name, True)
        assert control["result_commitments"] == candidate["result_commitments"]
        a, b = control["ledger"], candidate["ledger"]
        workloads[name] = {"optimized_control": control, "candidate": candidate,
                           "incremental_all_link_bytes_saved": a["all_link_bytes"] - b["all_link_bytes"],
                           "incremental_validation_bytes_saved": a["validation_bytes"] - b["validation_bytes"]}
    receipt_ledger = Ledger()
    receipt_ledger.setup()
    receipts = Receipts(receipt_ledger)
    request = {"tokens": [1, 2, 3], "seed_fixture": 7, "outputs": 4, "context": CONTEXT}
    calls = []
    for _ in range(3):
        calls.append(receipts.call("same-logical-operation", request,
                     lambda: run([1, 2, 3], 7, 4, receipt_ledger, ordinary=True)))
    assert calls[0] == calls[1] == calls[2]
    receipt_controls = []
    for ordinary in (False, True):
        control_ledger = Ledger()
        control_ledger.setup()
        control_store = Store(control_ledger)
        for index in range(3):
            control_ledger.links["application->client"] += len(encode(request))
            # An optimistic extra lower bound lets the control publish an
            # ordinary cold trajectory then use zero-issuance hits. Do not
            # attribute the installed fixed-one cold penalty to receipts.
            result = run([1, 2, 3], 7, 4, control_ledger, control_store,
                         ordinary=ordinary or index == 0)
            control_ledger.links["client->application"] += len(encode(asdict(result)))
            assert result == calls[0]
        receipt_controls.append(control_ledger.report())
    receipt_control = min(receipt_controls, key=lambda x: x["all_link_bytes"])
    scaling = []
    for count in (2, 4, 8, 16):
        arms = []
        for candidate in (False, True):
            costs = []
            for ordinary in (False, True):
                ledger = Ledger()
                ledger.setup()
                store = Store(ledger)
                prefixes = ResidentPrefixes()
                for branch in range(count):
                    prompt = [1, 2, 3, 4] * 12 + [branch // VOCAB, branch % VOCAB]
                    result = run(prompt, 7, 4, ledger, store,
                                 delta=candidate, ordinary=ordinary, prefixes=prefixes)
                    assert result == reference(prompt, 7, 4)
                costs.append(ledger.report())
            arms.append(min(costs, key=lambda x: x["all_link_bytes"]))
        saved = arms[0]["all_link_bytes"] - arms[1]["all_link_bytes"]
        scaling.append({"siblings": count, "shared_prompt_rows": 48,
                        "control_all_link_bytes": arms[0]["all_link_bytes"],
                        "candidate_all_link_bytes": arms[1]["all_link_bytes"],
                        "bytes_saved": saved,
                        "fraction_saved": saved / arms[0]["all_link_bytes"]})
    geometry_path = ROOT / "docs/evidence/research-narrow-ingress.json"
    geometry = json.loads(geometry_path.read_text())["checkpoint_geometry"]
    summary_path = ROOT / "docs/evidence/decode-cache-sdk-qwen25-2026-10-06-summary.json"
    summary = json.loads(summary_path.read_text())
    v1_path = ROOT / "docs/evidence/exact-reuse-frontier-reference.json"
    v1_bytes = v1_path.read_bytes()
    assert hashlib.sha256(v1_bytes).hexdigest() == V1_SHA256, "superseded v1 must stay byte-identical"
    v1 = json.loads(v1_bytes)
    mismatches = {p: {"v1_recorded_sha256": h,
                      "current_sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                  for p, h in v1["source_sha256"].items()
                  if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h}
    boundary_screen = []
    for prefix, total in ((8, 10), (48, 50), (45, 47)):
        positions, _ = installed_positions(total, 32768 // 8)
        nearest = max(p for p in positions if p <= prefix)
        boundary_screen.append({"shared_rows": prefix, "prompt_rows": total,
                                "installed_nearest_retained_rows": nearest,
                                "gross_additional_rows_per_sibling_ceiling": prefix - nearest,
                                "candidate_extra_capture_and_metadata_cpu": None,
                                "explicit_prefix_control_extra_cost": None,
                                "net_runtime_all_link_bytes_saved": None})
    locks = ["scripts/probe_exact_reuse_frontier.py", "python/pllm/runtime/client.py",
             "tests/test_exact_reuse_frontier.py",
             "docs/research/decode-cache-sdk-reference.md",
             "docs/research/exact-reuse-frontier-reference.md",
              "python/pllm/runtime/decode_store.py", "python/pllm/runtime/prefill_cache.py",
              "python/pllm/runtime/transformer_client.py",
             "python/pllm/runtime/conversation.py", "python/pllm/runtime/model_binding.py",
             "docs/evidence/research-narrow-ingress.json",
             "docs/evidence/decode-cache-sdk-qwen25-2026-10-06-summary.json"]
    return {
        "schema": "pllm.exact_reuse_frontier.v2", "evidence_class": "bounded synthetic simulator; corrected installed-prefix control",
        "superseded_v1": {
            "path": str(v1_path.relative_to(ROOT)), "sha256": V1_SHA256,
            "status": "rejected baseline; retained byte-identically, not a v2 replay target",
            "current_source_mismatches": mismatches,
            "observed_revision_start": "on-disk v1 already routed corrections to inference and matched pre-v2 source hashes; earlier auditor snapshot not recovered",
            "v1_source_reproduction": "not asserted; original code bundle not archived",
            "withdrawn_claims": ["1024 incremental fixture bytes", "27.59–76.27% incremental sibling savings", "new native cut ability required"]},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in locks},
        "synthetic_topology": {"cold_artifact_bytes": 4096, "row_upload_bytes": 16,
                                "row_reply_bytes": 24, "row_correction_bytes": 24,
                               "row_correction_link": "preparation->inference",
                               "fixture_controls_bytes_per_session": 128,
                               "fixture_control_bytes_by_link_per_session": {
                                   "client->preparation": 32, "preparation->client": 32,
                                   "client->inference": 32, "inference->client": 32},
                               "actual_runtime_control_and_framing_bytes": None,
                               "inference_to_preparation_acknowledgement_bytes": None,
                               "protected_remote_lookup_bytes": 0,
                               "remote_lookup_enabled": False, "max_tokens": LIMIT,
                               "max_nodes": MAX_NODES, "max_receipts": 32,
                               "restart": "synthetic index rebind; public artifact cache persists; no real process spawned",
                               "encryption_executed": False},
        "workloads": workloads,
        "application_receipt_retry": {"calls": 3, "logical_operations": 1,
                                      "ledger": receipt_ledger.report(),
                                      "best_installed_control_ledger": receipt_control,
                                      "control_scope": "optimistic cold ordinary + warm exact-hit lower bound; includes application links",
                                      "incremental_all_link_bytes_saved": receipt_control["all_link_bytes"] - sum(receipt_ledger.links.values()),
                                      "new_invocation_rng_replay_allowed": False},
        "conditional_sibling_scaling_not_measurements": scaling,
        "boundary_screen": boundary_screen,
        "prefix_policy": {"same_in_both_arms": True, "cache_payload_budget_bytes": 32768,
                          "block_rows": 8, "max_entries": 4096,
                          "selection": "powers of two, every eight rows, total-1, budget-limited cut",
                          "retention": "shared full blocks, clipped references, LRU eviction; full logits optional for proper prefixes",
                          "admission": "fixture assumes compiled prefix_f32 full-KV completed owner-free lineage with verification-none; real compiled admission not executed",
                          "capture_cost_scope": "prepare/immutable-copy/restore payloads and publications counted separately in both arms; upstream runtime.snapshot overlap, CPU and Python/native allocation unknown",
                          "concurrent_control": "wait for completed producer prefill/response under existing serial prepared queue; same reusable cuts, no additional bytes attributed to earlier publication",
                          "public_boundary_disclosure": "length and scheduling are public; all token/path matching remains caller-local"},
        "logical_ownership": {
            "max_numeric_state_payload_bytes": LIMIT * 8 + 4 * VOCAB,
            "max_store_node_serialization_bytes": MAX_NODE_BYTES,
            "max_decode_fixture_state_owner_bytes":
                128 * (MAX_NODE_BYTES + 32 + LIMIT * 192 + 4096)
                + 2 * (LIMIT * 192 + 4096) + MAX_NODE_BYTES,
            "scope": "conservative fixture raw/HMAC/tuple-int/metadata and two snapshot envelope; excludes interpreter, application receipts, source/model/runtime and allocator/RSS",
            "receipt_owner_bound_bytes": 32 * (MAX_NODE_BYTES + LIMIT * 192 + 4096),
            "prefix_owner_additional_envelope_bytes":
                4096 * (4096 + LIMIT * 96) + 32 * 32768 + 2 * (LIMIT * 192 + 4096),
            "prefix_owner_scope": "conservative synthetic shared-block/reference/path metadata and copy envelope; additional to decode/receipt/interpreter/runtime, not measured RSS",
            "production_largest_admissible_whole_client_bytes": None},
        "real_geometry": {"lock": geometry["lock"], "stage_count": len(geometry["stages"]),
                          "kv_row_bytes": 24576, "logit_bytes": 607744,
                          "geometry_source": "decode-cache-sdk-reference.md: 24 layers, 2 KV heads, dimension 64, vocabulary 151936",
                          "all_stage_u32_row_ceiling_by_link": {
                              "client->inference": sum(4 * s["shape"][1] for s in geometry["stages"]),
                              "inference->client": sum(4 * s["shape"][0] for s in geometry["stages"]),
                              "preparation->inference": sum(4 * s["shape"][0] for s in geometry["stages"])},
                          "ceiling_scope": "unpruned u32 geometry only, NOT optimized serving traffic or a saving estimate",
                          "optimized_checkpoint_bytes_saved": None},
        "installed_evidence_separate_not_pooled": {
            "largest_observed_client_rss_bytes": summary["checks"]["largest_client_peak_by_policy"],
            "new_candidate_rss_bytes": None,
            "new_candidate_256_mib_qualification": False},
        "thresholds": {
            "branch": "scheduled boundaries: incremental network ceiling zero; unscheduled gross <=7 causal rows per subsequent sibling if nearest scheduled cut survives; subtract candidate and explicit-control costs",
            "delta": "network ceiling=0 locally; hit disk saving=(parent_rows*8) in fixture; metadata unchanged",
            "receipt": "retry_probability > publication_cost/(avoided_retry_cost - lookup_cost), if denominator positive; compare optimized full-hit decode, not replay",
            "first_unique_request": "zero incremental network gain established, including scheduled-boundary sibling batch; cold setup always charged"},
        "unknown": ["net benefit of unscheduled boundary selection", "real hit rates", "new CPU/RSS/timing",
                    "actual runtime controls/framing including inference->preparation acknowledgements",
                    "checkpoint all-link incremental bytes", "physical wire/filesystem amplification",
                    "whole-client admission", "remote-MAC fraction measurement"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = report()
    with args.output.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
