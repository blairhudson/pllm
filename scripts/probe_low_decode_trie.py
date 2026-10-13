"""Bounded research-only exact decode trie; no model/provider imports.

Integer attention fixture is not the PLLM decoder. Disk format is private to this
probe, not compatible with SDK checkpoints or production state-basis seals.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = b"pllm.research.decode-trie.toy.v1\0"
MAX_NODES = 128
MAX_DEPTH = 32
MAX_FRAME = 16384
MAX_DISK = 2 << 20


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class State:
    history: tuple[int, ...]
    rows: tuple[tuple[int, int], ...]
    logits: tuple[int, ...]


def execute(state, token):
    """One exact integer causal-attention step; returns state after execution."""
    if type(token) is not int or not 0 <= token < 3:
        raise ValueError("invalid token")
    position = len(state.history)
    if position >= MAX_DEPTH:
        raise ValueError("depth bound")
    rows = (*state.rows, (token + position + 1, (token + 1) ** 2))
    query = token + 2
    attended = sum((query * key + 1) * value for key, value in rows)
    logits = tuple(1 + (attended + (index + 1) * query + position) % 7
                   for index in range(3))
    return State((*state.history, token), rows, logits)


def sample(logits, rng, greedy=False):
    if greedy:
        return max(range(len(logits)), key=lambda index: logits[index])
    # Exactly one draw on both cache-hit and cache-miss paths.
    draw = rng.randrange(sum(logits))
    for token, weight in enumerate(logits):
        draw -= weight
        if draw < 0:
            return token
    raise AssertionError("invalid sampler weights")


class DecodeTrie:
    """Single-writer, caller-trusted bounded index; immutable AEAD pages.

    All keys/history are private local probe inputs. No auto-latest disk lookup.
    Heads and key stay caller-owned; close does not delete durable files.
    """

    def __init__(self, directory, key, binding="toy-source:numeric-v1:full-kv"):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("fresh private directory required")
        self.aead = AESGCM(key)
        self.binding = binding
        self.heads = {}
        self.pages = {}
        self.revision = 0
        self.closed = False
        self.read_bytes = self.write_bytes = self.orphan_bytes = 0
        self.page_writes = self.manifest_writes = 0
        self.row_plaintext_bytes = self.manifest_plaintext_bytes = 0

    def _aad(self, kind, identity):
        return DOMAIN + encode([self.binding, kind, identity])

    def _write(self, kind, identity, payload):
        raw = encode(payload)
        if len(raw) + 28 > MAX_FRAME:
            raise ValueError("frame bound")
        if self.write_bytes + len(raw) + 28 > MAX_DISK:
            raise ValueError("disk bound")
        nonce = os.urandom(12)
        frame = nonce + self.aead.encrypt(nonce, raw, self._aad(kind, identity))
        commitment = digest(frame)
        path = self.directory / commitment
        # New random nonce per immutable publication; never overwrite accepted page.
        with path.open("xb") as stream:
            os.chmod(path, 0o600)
            stream.write(frame)
            stream.flush()
            os.fsync(stream.fileno())
        self.write_bytes += len(frame)
        if kind == "row":
            self.page_writes += 1
            self.row_plaintext_bytes += len(raw)
        else:
            self.manifest_writes += 1
            self.manifest_plaintext_bytes += len(raw)
        return commitment, len(frame)

    def _read(self, kind, identity, record):
        commitment, size = record
        if not 28 <= size <= MAX_FRAME:
            raise ValueError("frame bound")
        with (self.directory / commitment).open("rb") as stream:
            frame = stream.read(size + 1)
        self.read_bytes += len(frame)
        if len(frame) != size or digest(frame) != commitment:
            raise ValueError("frame commitment")
        return json.loads(self.aead.decrypt(frame[:12], frame[12:], self._aad(kind, identity)))

    def put(self, state, *, fail_before_head=False):
        if self.closed:
            raise ValueError("closed")
        history = state.history
        if (not 0 < len(history) <= MAX_DEPTH or len(state.rows) != len(history)
                or len(state.logits) != 3):
            raise ValueError("executed state required; pending selection is not KV")
        if history in self.heads:
            return
        if len(self.heads) >= MAX_NODES:
            raise ValueError("node bound")
        before = self.write_bytes
        records = []
        for length, row in enumerate(state.rows, 1):
            prefix = history[:length]
            if prefix not in self.pages:
                self.pages[prefix] = self._write("row", prefix, list(row))
            records.append(self.pages[prefix])
        revision = self.revision + 1
        manifest = {"history": history, "records": records, "logits": state.logits,
                    "revision": revision, "position": len(history)}
        head = self._write("node", [history, revision], manifest)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if fail_before_head:
            self.orphan_bytes += self.write_bytes - before
            raise OSError("injected publication failure")
        # Publish trusted owner only after durable numeric and manifest frames.
        self.heads[history] = (revision, head)
        self.revision = revision

    def get(self, history):
        if self.closed:
            raise ValueError("closed")
        history = tuple(history)
        accepted = self.heads.get(history)
        if accepted is None:
            return None
        revision, head = accepted
        doc = self._read("node", [history, revision], head)
        if (set(doc) != {"history", "records", "logits", "revision", "position"}
                or doc["history"] != list(history) or doc["position"] != len(history)
                or doc["revision"] != revision or len(doc["records"]) != len(history)
                or len(doc["logits"]) != 3
                or any(type(value) is not int or value < 1 for value in doc["logits"])):
            raise ValueError("executed state identity")
        rows = tuple(tuple(self._read("row", history[:length], record))
                     for length, record in enumerate(doc["records"], 1))
        if any(len(row) != 2 or any(type(value) is not int for value in row) for row in rows):
            raise ValueError("row geometry")
        return State(history, rows, tuple(doc["logits"]))

    def ledger(self):
        files = list(self.directory.iterdir())
        return {"nodes": len(self.heads), "unique_row_pages": len(self.pages),
                "logical_encrypted_reads": self.read_bytes,
                "logical_encrypted_writes": self.write_bytes,
                "orphan_publication_bytes": self.orphan_bytes,
                "retained_file_bytes": sum(path.stat().st_size for path in files),
                "allocated_disk_bytes": sum(path.stat().st_blocks * 512 for path in files),
                "files": len(files), "page_writes": self.page_writes,
                "manifest_writes": self.manifest_writes,
                "row_plaintext_written_bytes": self.row_plaintext_bytes,
                "manifest_plaintext_written_bytes": self.manifest_plaintext_bytes,
                "nonce_tag_written_bytes": 28 * (self.page_writes + self.manifest_writes),
                "trusted_index_serialized_bytes_not_rss": len(encode([
                    [history, revision, record] for history, (revision, record) in self.heads.items()
                ]))}

    def close(self):
        self.heads.clear()
        self.pages.clear()
        self.aead = None
        self.closed = True


def generate(root, *, seed, outputs=5, trie=None, greedy=False):
    if not 1 <= outputs <= MAX_DEPTH - len(root.history) + 1:
        raise ValueError("output bound")
    rng = random.Random(seed)
    state = root
    selected = []
    states = []
    hits = misses = 0
    for index in range(outputs):
        token = sample(state.logits, rng, greedy)
        selected.append(token)
        if index == outputs - 1:
            break  # Last selected token remains pending, not executed.
        cached = trie.get((*state.history, token)) if trie is not None else None
        if cached is None:
            misses += 1  # One fresh toy execution owner; not a provider measurement.
            state = execute(state, token)
            if trie is not None:
                trie.put(state)
        else:
            hits += 1
            state = cached
        states.append(state)
    return {"selected": tuple(selected), "state": state, "states": states,
            "pending": selected[-1], "hits": hits, "misses": misses,
            "rng_state": rng.getstate()}


def break_even(*, hits, per_execution_all_link_bytes, extra_all_link_bytes):
    """Conditional traffic arithmetic. Unknown real costs remain unknown."""
    if per_execution_all_link_bytes is None or extra_all_link_bytes is None:
        return None
    if min(hits, per_execution_all_link_bytes, extra_all_link_bytes) < 0:
        raise ValueError("negative cost")
    return hits * per_execution_all_link_bytes - extra_all_link_bytes


def screen():
    root = execute(execute(State((), (), ()), 0), 1)
    reports = {}
    for name, seeds, greedy in (
        ("greedy_repeated", [0] * 24, True),
        ("stochastic_same_seed", [19] * 24, False),
        ("stochastic_distinct_seeds", list(range(24)), False),
    ):
        with tempfile.TemporaryDirectory(prefix="low-decode-trie-") as directory:
            trie = DecodeTrie(directory, b"r" * 32)
            trie.put(root)
            hits = misses = checked_states = checked_numeric_words = 0
            for seed in seeds:
                control = generate(root, seed=seed, greedy=greedy)
                candidate = generate(root, seed=seed, trie=trie, greedy=greedy)
                assert all(candidate[field] == control[field]
                           for field in ("selected", "state", "states", "pending", "rng_state"))
                hits += candidate["hits"]
                misses += candidate["misses"]
                checked_states += len(candidate["states"])
                checked_numeric_words += sum(2 * len(state.rows) + len(state.logits)
                                             for state in candidate["states"])
            ledger = trie.ledger()
            reports[name] = {
                "generations": len(seeds), "output_tokens_per_generation": 5,
                "initial_prefix_tokens": 2, "baseline_executed_decode_rows": 96,
                "candidate_executed_decode_rows": misses, "cached_executed_rows": hits,
                "full_toy_state_checks": checked_states,
                "checked_toy_kv_logit_integer_words": checked_numeric_words,
                "initial_setup_executed_rows_both": 2,
                "sampler_calls_both": 120, "disk": ledger,
                "measured_all_link_bytes": None, "whole_client_peak_rss_bytes": None,
                "client_cpu_seconds": None, "authentication_cpu_seconds": None,
                "physical_disk_io_bytes": None, "remote_body_fraction": None,
                "extra_real_all_link_bytes": None, "real_per_decode_all_link_bytes": None,
                "traffic_break_even_bytes": break_even(hits=hits,
                    per_execution_all_link_bytes=None, extra_all_link_bytes=None),
            }
            trie.close()
    paths = ["scripts/probe_low_decode_trie.py", "tests/test_low_decode_trie_probe.py",
             "docs/research/low-decode-trie-reference.md",
             "python/pllm/runtime/client.py", "python/pllm/runtime/prefill_cache.py",
             "python/pllm/runtime/transformer_client.py", "python/pllm/runtime/context_graph.py",
             "python/pllm/runtime/conversation_checkpoint.py",
             "docs/research/sdk-continuation-reference.md",
             "docs/research/conversation-restore-reference.md",
             "docs/research/context-graph-reference.md"]
    return {"schema": "pllm.research.low-decode-trie.v1", "decision": "open",
            "qualification": "not promoted; synthetic exact-state screen only",
            "native_decoder_gate": "not performed",
            "rss_limit_bytes": 268435456, "required_remote_body_fraction": 0.9,
            "bounds": {"nodes": MAX_NODES, "depth": MAX_DEPTH,
                       "encrypted_frame_bytes": MAX_FRAME, "logical_disk_bytes": MAX_DISK},
            "source_implementation_sha256": {path: digest((ROOT / path).read_bytes()) for path in paths},
            "reproduction": "target/frontier-sdk-env/bin/python scripts/probe_low_decode_trie.py "
                            "--output docs/evidence/research-low-decode-trie.json",
            "workloads": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(screen(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
