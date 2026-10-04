"""Exact masked-row chunking plus an explicitly hypothetical duplex schedule."""
from dataclasses import dataclass
import hashlib
import math
import secrets
import statistics
import time


def _finish(frames, up, down, delay):
    uploaded = computed = downloaded = 0.0
    for request, response, cpu in frames:
        uploaded += request / up
        computed = max(computed, uploaded + delay) + cpu
        downloaded = max(downloaded, computed) + response / down
    return downloaded + delay


@dataclass(frozen=True, slots=True)
class PreparedDuplexProbe:
    rows: int = 39
    chunk_rows: tuple[int, ...] = (1, 4, 8, 16)
    repetitions: int = 3
    download_mbps: float = 100
    upload_mbps: float = 40
    latency_ms: float = 20

    def __post_init__(self):
        if (type(self.rows) is not int or not 1 <= self.rows <= 128
                or type(self.repetitions) is not int or not 1 <= self.repetitions <= 8
                or type(self.chunk_rows) is not tuple or not 1 <= len(self.chunk_rows) <= 8
                or len(set(self.chunk_rows)) != len(self.chunk_rows)
                or any(type(n) is not int or not 1 <= n <= 128 for n in self.chunk_rows)):
            raise ValueError("duplex probe exceeds bounded row/repetition choices")
        for name in ("download_mbps", "upload_mbps", "latency_ms"):
            value = getattr(self, name)
            if (type(value) not in (float, int) or not math.isfinite(value)
                    or not (0 if name == "latency_ms" else 0.001) <= value <= 1000):
                raise ValueError("duplex probe requires bounded decimal Mbps and one-way delay")

    def run(self, weights=None, *, dense_wire_bits: int | None = None) -> dict:
        import msgpack
        import numpy as np
        from pllm import _native
        from pllm.native import MaskedGEMM
        from pllm.runtime.native import mask_prepared_input
        from pllm.runtime.residue_codec import PREPARED_ROW_MAGIC, unpack_row_response
        from pllm.runtime.stage_protocol import MaskedStageRequest, PreparedStageBatchRequest

        rng = np.random.default_rng(6021)
        w = rng.integers(-63, 64, (128, 64), dtype=np.int8) if weights is None else np.asarray(weights)
        if (w.dtype != np.int8 or w.ndim != 2 or not 1 <= w.size <= 32 << 20
                or min(w.shape) < 1 or max(w.shape) * self.rows > 4_000_000):
            raise ValueError("duplex probe requires bounded signed-i8 matrix weights")
        widths = _native.offset_row_bits(w.tobytes(), w.shape[1], 127)
        bits = next(v for v in (16, 24, 32) if v >= max(widths)) if dense_wire_bits is None else dense_wire_bits
        if type(bits) is not int or bits not in (16, 24, 32) or bits < max(widths):
            raise ValueError("duplex ring cannot represent the public output bound")
        x = rng.integers(-127, 128, (self.rows, w.shape[1]), dtype=np.int8)
        expected = x.astype(np.int64) @ w.astype(np.int64).T
        kernel = MaskedGEMM(threads=1).compile(w)
        if kernel.owner.backend != "rust":
            raise RuntimeError("duplex probe requires the native integer kernel")
        context = hashlib.sha256(b"pllm.prepared-duplex.v1\0" + w.tobytes()).digest()
        weight_digest = hashlib.sha256(w.tobytes()).hexdigest()
        bound = int(np.max(np.sum(np.abs(w.astype(np.int64)), axis=1))) * 127
        configurations = []
        rate = min(self.upload_mbps, self.download_mbps) * 1_000_000 / 8
        for chunk in sorted({self.rows, *(min(self.rows, n) for n in self.chunk_rows)}, reverse=True):
            clients, providers, estimates = [], [], []
            uploads = downloads = 0
            for _ in range(self.repetitions):
                # Each candidate/run uses fresh masks; row slices are disjoint.
                seed = secrets.token_bytes(32)
                r = np.frombuffer(_native.offset_seeded_share(seed, context, x.size, bits), "<u4").reshape(x.shape)
                s = np.frombuffer(_native.offset_seeded_share(seed, hashlib.sha256(context).digest(),
                    self.rows * w.shape[0], bits), "<u4").reshape(self.rows, w.shape[0])
                wr = kernel.wrap32(r)
                frames, results = [], []
                client_cpu = provider_cpu = 0.0
                for start in range(0, self.rows, chunk):
                    end = min(self.rows, start + chunk)
                    n, ticket = end - start, secrets.token_hex(16)
                    mask = s[start:end].tobytes()
                    correction = _native.prepared_pack_correction(wr[start:end].astype("<u4", copy=False).tobytes(),
                        mask, widths, n)
                    began = time.process_time_ns()
                    masked = mask_prepared_input(x[start:end], r[start:end], bits)
                    if n > 1:
                        request = PreparedStageBatchRequest(batch_id=ticket,
                            correlation_ids=tuple(secrets.token_hex(16) for _ in range(n)),
                            masked_input=masked, wire_bits=bits).pack()
                    else:
                        request = MaskedStageRequest(model="duplex-probe", stage_id="duplex-probe",
                            correlation_id=ticket, masked_input=masked, activation_scales=np.ones(1, dtype=np.float32),
                            modulus=1 << bits, wire_bits=bits, ring=f"u{bits}", body_fingerprint=context.hex(),
                            weight_digest=weight_digest, weight_bits=8, activation_bits=8,
                            session_id="duplex-probe", out_features=w.shape[0], signed_output_bound=bound).pack()
                    client_cpu += (time.process_time_ns() - began) / 1e9
                    began = time.process_time_ns()
                    wx = kernel.wrap32(masked).astype("<u4", copy=False).tobytes()
                    packed = _native.prepared_pack_output(wx, correction, widths, n)
                    response = PREPARED_ROW_MAGIC + msgpack.packb([ticket, "duplex-probe", n, bits,
                        hashlib.sha256(widths).digest(), 0, packed], use_bin_type=True)
                    compute = (time.process_time_ns() - began) / 1e9
                    provider_cpu += compute
                    began = time.process_time_ns()
                    parsed, _ = unpack_row_response(response, ticket=ticket, stage="duplex-probe", widths=widths,
                        rows=n, bits=bits, namespace="prepared")
                    result = np.frombuffer(_native.prepared_unpack_output(parsed, mask, widths, n), "<i8").reshape(n, w.shape[0])
                    results.append(result)
                    client_cpu += (time.process_time_ns() - began) / 1e9
                    frames.append((len(request), len(response), compute))
                began = time.process_time_ns()
                result = results[0] if len(results) == 1 else np.concatenate(results)
                client_cpu += (time.process_time_ns() - began) / 1e9
                if not np.array_equal(result, expected):
                    raise RuntimeError("duplex chunking changed exact integer output")
                clients.append(client_cpu)
                providers.append(provider_cpu)
                estimates.append(_finish(frames, rate, rate, self.latency_ms / 1000) + client_cpu)
                uploads = sum(f[0] for f in frames)
                downloads = sum(f[1] for f in frames)
            configurations.append({"chunk_rows": chunk, "chunks": math.ceil(self.rows / chunk),
                "request_body_bytes": uploads, "response_body_bytes": downloads,
                "client_cpu_seconds_median": statistics.median(clients),
                "provider_cpu_seconds_median": statistics.median(providers),
                "hypothetical_stage_seconds_median": statistics.median(estimates),
                "exact_integer_output": True})
        return {"schema": "pllm.prepared_duplex_probe.v1", "weight_digest": hashlib.sha256(w.tobytes()).hexdigest(),
            "shape": list(w.shape), "rows": self.rows, "dense_wire_bits": bits,
            "per_party_download_mbps": self.download_mbps, "per_party_upload_mbps": self.upload_mbps,
            "one_way_delay_ms": self.latency_ms, "configurations": configurations,
            "client_body_weights_required": 0, "whole_decoder_executable": False,
            "scope": "measured native integer kernels/codecs; hypothetical full-duplex stage timing",
            "limitations": ["No pipelined transport, one-use streaming session or whole-response execution is implemented",
                "One RTT assumed per stage, unlimited peer read/write progress, symmetric party access",
                "Fresh mask expansion/offline issuance, TCP/TLS, Python scheduling and peak memory are not priced",
                "One-row decode cannot use row overlap; no decoder TPS claim follows"]}
