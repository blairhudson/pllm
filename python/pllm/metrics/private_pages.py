"""Bounded public-page PIR measurement; native party state, no serving admission."""

from dataclasses import dataclass
import hashlib
import statistics
import time


@dataclass(frozen=True, slots=True)
class PrivatePageLookupProbe:
    """Measure native two-server retrieval of exact public records.

    A page contains a fixed public number of records. Only the client chooses
    the record within the returned page. Fixture/oracle data are test-local;
    the online client algorithm needs a descriptor, keys and one returned page.
    """

    records: int = 64
    record_bytes: int = 32
    page_records: int = 1
    queries: int = 3

    def __post_init__(self):
        for value, bound in (
            (self.records, 262144),
            (self.record_bytes, 65536),
            (self.queries, 16),
        ):
            if type(value) is not int or not 1 <= value <= bound:
                raise ValueError("private page probe exceeds bounded dimensions")
        if type(self.page_records) is not int or self.page_records not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("page_records must be a power of two from 1 through 64")
        pages = (self.records + self.page_records - 1) // self.page_records
        width = self.record_bytes * self.page_records
        if width > 65536 or pages * width > 512 * 1024 * 1024:
            raise ValueError("private page table exceeds 512 MiB or 64 KiB page policy")

    def run(self, table: bytes | None = None) -> dict:
        """Check exact bytes and measure separate client/worker CPU windows."""
        import numpy as np
        from pllm import _native

        if table is None:
            table = (
                np.random.default_rng(51703)
                .integers(0, 256, self.records * self.record_bytes, dtype=np.uint8)
                .tobytes()
            )
        if type(table) is not bytes or len(table) != self.records * self.record_bytes:
            raise ValueError("public fixture bytes differ from the declared table")
        width = self.page_records * self.record_bytes
        pages = (self.records + self.page_records - 1) // self.page_records
        source_digest = hashlib.sha256(table).hexdigest()
        padded = table + bytes(pages * width - len(table))
        started = time.process_time_ns()
        servers = tuple(_native.PrivatePageServer(padded, width, party) for party in (0, 1))
        setup_cpu = (time.process_time_ns() - started) / 1e9
        descriptor = servers[0].descriptor()
        assert servers[1].descriptor() == descriptor
        client_cpu, worker_cpu, bodies, live_payloads = [], [], [], []
        for ordinal in range(self.queries):
            # Public test fixtures. Neither indices nor key/reply contents enter reports.
            index = (ordinal * 1717 + self.records // 3) % self.records
            started = time.process_time_ns()
            a, b, decoder = _native.private_page_issue(*descriptor, index // self.page_records)
            client_ns = time.process_time_ns() - started
            replies, party_cpu = [], []
            for server, query in zip(servers, (a, b), strict=True):
                started = time.process_time_ns()
                replies.append(server.evaluate(query))
                party_cpu.append((time.process_time_ns() - started) / 1e9)
            started = time.process_time_ns()
            page = decoder.decode(*replies)
            offset = (index % self.page_records) * self.record_bytes
            record = page[offset : offset + self.record_bytes]
            client_ns += time.process_time_ns() - started
            if record != table[index * self.record_bytes : (index + 1) * self.record_bytes]:
                raise RuntimeError("private page bytes differ from the independent source oracle")
            client_cpu.append(client_ns / 1e9)
            worker_cpu.append(party_cpu)
            bodies.append(sum(map(len, (a, b, *replies))))
            live_payloads.append(bodies[-1] + len(page) + len(record))
        assert len(set(bodies)) == 1
        return {
            "schema": "pllm.private_page_lookup_probe.v1",
            "scope": "in-process native XOR-PIR; exact serialized query/reply bodies",
            "records": self.records,
            "record_bytes": self.record_bytes,
            "page_records": self.page_records,
            "queries": self.queries,
            "source_table_digest": source_digest,
            "layout_digest": descriptor[0].hex(),
            "body_bytes_per_query": bodies[0],
            "client_wire_and_returned_payload_bytes": max(live_payloads),
            "client_online_table_bytes_required": 0,
            "public_table_storage_bytes_per_worker": len(padded),
            "public_table_scan_bytes_per_query_both_workers": 2 * len(padded),
            "client_cpu_seconds_per_query_median": statistics.median(client_cpu),
            "worker_cpu_seconds_per_query_median": [
                statistics.median(row[i] for row in worker_cpu) for i in (0, 1)
            ],
            "two_worker_table_admission_cpu_seconds": setup_cpu,
            "exact_byte_parity": True,
            "whole_decoder_executable": False,
            "full_wire_bytes": None,
            "peak_client_memory_bytes": None,
            "limitations": [
                "Two semi-honest non-colluding servers, AES-based DPF assumption, no independent cryptographic review",
                "Process-local replay ledger; no authenticated transport or distributed restart contract",
                "Payload counts exclude Python/Rust object overhead and are not measured peak memory",
                "Fixture owner holds source bytes for validation; online client CPU excludes fixture and provider setup",
                "Local lookup currently has zero online bytes; PIR adds these bodies in exchange for optional client table removal",
                "A tied local output head still needs the table until a separate head boundary is implemented",
                "Source distribution, complete decoder compute and generation quality are unmeasured",
            ],
        }
