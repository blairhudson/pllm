"""Fixed-size two-server PIR batches; query keys/replies stay unchanged."""
from dataclasses import dataclass
import hashlib
import time


@dataclass(frozen=True, slots=True)
class BatchedPrivateLookupProbe:
    records: int = 256
    record_bytes: int = 64
    batch_size: int = 8

    def __post_init__(self):
        if any(type(v) is not int or not 1 <= v <= bound for v, bound in
               ((self.records, 262144), (self.record_bytes, 65536), (self.batch_size, 64))) or self.records * self.record_bytes > 512 << 20:
            raise ValueError("private lookup batch exceeds bounded dimensions")

    def run(self, table: bytes | None = None) -> dict:
        import numpy as np
        from pllm import _native
        if table is None:
            table = np.random.default_rng(93014).integers(0, 256, self.records * self.record_bytes, dtype=np.uint8).tobytes()
        if type(table) is not bytes or len(table) != self.records * self.record_bytes:
            raise ValueError("public table bytes differ from layout")
        servers = [_native.PrivatePageServer(table, self.record_bytes, i) for i in (0, 1)]
        descriptor = servers[0].descriptor()
        modes = []
        for batched in (False, True):
            indices = [(i * 101 + 13) % self.records for i in range(self.batch_size)]
            started = time.process_time_ns()
            queries = [_native.private_page_issue(*descriptor, i) for i in indices]
            client_cpu = time.process_time_ns() - started
            replies, worker_cpu = [], []
            for i, server in enumerate(servers):
                keys = [q[i] for q in queries]
                started = time.process_time_ns()
                replies.append(server.evaluate_batch(keys) if batched else [server.evaluate(key) for key in keys])
                worker_cpu.append((time.process_time_ns() - started) / 1e9)
            body = sum(len(q[0]) + len(q[1]) for q in queries) + sum(len(v) for row in replies for v in row)
            started = time.process_time_ns()
            values = [q[2].decode(replies[0][j], replies[1][j]) for j, q in enumerate(queries)]
            client_cpu += time.process_time_ns() - started
            for i, value in zip(indices, values, strict=True):
                if value != table[i * self.record_bytes:(i + 1) * self.record_bytes]:
                    raise RuntimeError("batched PIR differs from the source oracle")
            modes.append({"one_table_traversal": batched, "query_and_reply_body_bytes": body,
                "client_cpu_seconds": client_cpu / 1e9, "worker_cpu_seconds": worker_cpu, "exact_records": True,
                "logical_table_traversal_bytes_both_workers": 2 * len(table) * (1 if batched else self.batch_size)})
        return {"schema": "pllm.batched_private_lookup_probe.v1", "source_digest": hashlib.sha256(table).hexdigest(),
                "records": self.records, "record_bytes": self.record_bytes, "batch_size": self.batch_size,
                "configurations": modes, "client_table_bytes_required": 0,
                "additional_worker_selection_bytes": self.batch_size * self.records,
                "scope": "in-process fixed public batch of independent one-use DPF queries",
                "whole_decoder_executable": False, "full_wire_bytes": None, "peak_client_memory_bytes": None,
                "limitations": ["Independent DPF expansion work remains per key; logical scans are not measured DRAM traffic",
                    "Fresh keys and record replies are unchanged; no body-byte reduction follows from batching",
                    "Private head use still needs a public fixed candidate-capacity certificate",
                    "Two non-colluding semi-honest parties; no transport, independent review or full decoder quality gate"]}
