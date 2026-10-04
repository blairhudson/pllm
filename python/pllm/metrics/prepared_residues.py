"""Exact public-width compression of prepared corrections and masked results."""
from dataclasses import dataclass
import hashlib
import secrets
import statistics
import time


@dataclass(frozen=True, slots=True)
class PreparedResidueProbe:
    rows: int = 8
    repetitions: int = 3

    def __post_init__(self):
        if type(self.rows) is not int or not 1 <= self.rows <= 128 or type(self.repetitions) is not int or not 1 <= self.repetitions <= 16:
            raise ValueError("prepared residue probe exceeds row/repetition bounds")

    def run(self, weights=None, *, dense_wire_bits: int | None = None) -> dict:
        import numpy as np
        from pllm import _native
        from pllm.native import MaskedGEMM
        from pllm.runtime.native import mask_prepared_input, unmask_prepared_output
        rng = np.random.default_rng(7206)
        w = rng.integers(-63, 64, (128, 64), dtype=np.int8) if weights is None else np.asarray(weights)
        if w.ndim != 2 or w.dtype != np.int8 or not 1 <= w.size <= 32 << 20 or min(w.shape) < 1:
            raise ValueError("residue probe requires bounded signed-i8 weights")
        if self.rows * max(w.shape) > 4_000_000:
            raise ValueError("residue probe exceeds element bound")
        widths = _native.offset_row_bits(w.tobytes(), w.shape[1], 127)
        bits = next(v for v in (16, 24, 32) if v >= max(widths)) if dense_wire_bits is None else dense_wire_bits
        if type(bits) is not int or bits not in (16, 24, 32) or bits < max(widths):
            raise ValueError("dense wire ring cannot represent public output bound")
        dense_widths = bytes([bits]) * w.shape[0]
        kernel = MaskedGEMM(threads=1).compile(w)
        if kernel.owner.backend != "rust":
            raise RuntimeError("native kernel required")
        x = rng.integers(-127, 128, (self.rows, w.shape[1]), dtype=np.int8)
        expected = x.astype(np.int64) @ w.astype(np.int64).T
        context = hashlib.sha256(b"pllm.prepared-residue.probe.v1\0" + w.tobytes()).digest()
        output_context = hashlib.sha256(context + b"output").digest()
        timings = {"dense_numpy": [], "dense_native": [], "public_row_residues": []}
        input_timings = {name: [] for name in ("numpy", "native")}
        counts = {}
        for _ in range(self.repetitions):
            seed = secrets.token_bytes(32)
            r = _native.offset_seeded_share(seed, context, x.size, bits)
            s = _native.offset_seeded_share(seed, output_context, self.rows * w.shape[0], bits)
            r_array = np.frombuffer(r, "<u4").reshape(x.shape)
            start = time.process_time_ns()
            masked_numpy = ((x.astype(np.int32).astype(np.int64) % (1 << bits) - r_array.astype(np.int64)) % (1 << bits)).astype(np.uint32)
            input_timings["numpy"].append((time.process_time_ns() - start) / 1e9)
            start = time.process_time_ns()
            masked_array = mask_prepared_input(x, r_array, bits)
            input_timings["native"].append((time.process_time_ns() - start) / 1e9)
            if not np.array_equal(masked_numpy, masked_array):
                raise RuntimeError("native input masking differs")
            wr = kernel.wrap32(np.frombuffer(r, "<u4").reshape(x.shape)).astype("<u4").tobytes()
            wx = kernel.wrap32(masked_array).astype("<u4").tobytes()
            s_array = np.frombuffer(s, "<u4").reshape(expected.shape)
            for name, layout in (("dense_numpy", dense_widths), ("dense_native", dense_widths), ("public_row_residues", widths)):
                correction = _native.prepared_pack_correction(wr, s, layout, self.rows)
                online = _native.prepared_pack_output(wx, correction, layout, self.rows)
                started = time.process_time_ns()
                if name == "public_row_residues":
                    result = np.frombuffer(_native.prepared_unpack_output(online, s, layout, self.rows), "<i8").reshape(expected.shape)
                else:
                    array = np.frombuffer(_native.unpack_unsigned(online, bits // 8), "<u4").reshape(expected.shape)
                    if name == "dense_native":
                        result = unmask_prepared_output(array, s_array, bits)
                    else:
                        combined = (array.astype(np.int64) + s_array.astype(np.int64)) % (1 << bits)
                        if bits == 16:
                            result = combined.astype(np.uint16).view(np.int16).astype(np.int64)
                        elif bits == 24:
                            result = np.where(combined >= 1 << 23, combined - (1 << 24), combined)
                        else:
                            result = combined.astype(np.uint32).view(np.int32).astype(np.int64)
                timings[name].append((time.process_time_ns() - started) / 1e9)
                if not np.array_equal(result, expected):
                    raise RuntimeError("prepared residue protocol changed exact integer results")
                counts[name] = {"input_masked_body_bytes": x.size * (bits // 8),
                    "offline_correction_body_bytes": len(correction), "online_output_body_bytes": len(online),
                    "width_metadata_bytes": len(widths) if name == "public_row_residues" else 0}
        configurations = [{"encoding": name, **counts[name],
            "client_decode_cpu_seconds_median": statistics.median(timings[name]),
            "client_input_mask_cpu_seconds_median": statistics.median(input_timings["numpy" if name == "dense_numpy" else "native"]),
            "exact_integer_output": True}
            for name in timings]
        return {"schema": "pllm.prepared_residue_probe.v1", "configurations": configurations,
                "weight_digest": hashlib.sha256(w.tobytes()).hexdigest(), "shape": list(w.shape), "rows": self.rows,
                "public_width_bits_range": [min(widths), max(widths)], "dense_wire_bits": bits,
                "client_body_weights_required": 0, "scope": "in-process seeded-mask arithmetic-body codec reference",
                "whole_decoder_executable": False, "peak_client_memory_bytes": None,
                "limitations": ["Input masks remain full-width and independent; only public output bounds select widths",
                "This isolated probe does not measure the selectable MaskedLinear row-residue runtime",
                    "Counts omit tickets, root-seed/control frames and full wire; numeric dequantization is unchanged"]}
