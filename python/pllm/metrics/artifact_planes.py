"""Lossless public-byte codec measurement, independent of private activation data."""
from dataclasses import dataclass
import hashlib
import statistics
import time
import zlib


@dataclass(frozen=True, slots=True)
class ArtifactPlaneProbe:
    repetitions: int = 3

    def __post_init__(self):
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 8:
            raise ValueError("artifact codec probe repetitions must be 1..8")

    def run(self, payload: bytes | None = None) -> dict:
        from pllm import _native
        if payload is None:
            import numpy as np
            payload = np.clip(np.random.default_rng(11234).normal(0, 15, 1 << 20), -127, 127).astype(np.int8).tobytes()
        if type(payload) is not bytes or not 0 < len(payload) <= 256 << 20:
            raise ValueError("artifact probe requires 1..256 MiB public bytes")
        chunk = 1 << 20
        records = []
        for mode in ("zlib", "bitplanes-zlib"):
            enc_cpu, dec_cpu, sizes = [], [], []
            for _ in range(self.repetitions):
                encoded = []
                started = time.process_time_ns()
                for offset in range(0, len(payload), chunk):
                    block = payload[offset:offset + chunk]
                    if mode != "zlib":
                        block = _native.public_bitplanes(block, False)
                    encoded.append(zlib.compress(block, level=1))
                enc_cpu.append((time.process_time_ns() - started) / 1e9)
                sizes.append(sum(len(block) + 8 for block in encoded))
                # Hashing and fixture comparisons are outside the codec timer.
                elapsed = 0
                digest = hashlib.sha256()
                for block in encoded:
                    started = time.process_time_ns()
                    raw = zlib.decompress(block)
                    if mode != "zlib":
                        raw = _native.public_bitplanes(raw, True)
                    elapsed += time.process_time_ns() - started
                    digest.update(raw)
                dec_cpu.append(elapsed / 1e9)
                if digest.hexdigest() != hashlib.sha256(payload).hexdigest():
                    raise RuntimeError("artifact codec changed source bytes")
            records.append({"encoding": mode, "framed_body_bytes": sizes[0], "exact_raw_hash": True,
                "provider_encode_cpu_seconds_median": statistics.median(enc_cpu),
                "client_decode_cpu_seconds_median": statistics.median(dec_cpu)})
        return {"schema": "pllm.artifact_plane_probe.v1", "source_digest": hashlib.sha256(payload).hexdigest(),
                "raw_bytes": len(payload), "frame_raw_bytes_bound": chunk, "configurations": records,
                "scope": "public object codec only; source fixture/working array is not a client peak sample",
                "raw_cache_identity_preserved": True, "peak_client_memory_bytes": None,
                "online_stage_body_savings": 0}
