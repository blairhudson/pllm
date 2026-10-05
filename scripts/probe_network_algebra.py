#!/usr/bin/env python3
"""Five public-weight algebra screens on a locked Qwen checkpoint.

Run: uv run --no-sync python scripts/probe_network_algebra.py --output PATH
No prompt values or live correlation material are recorded. Body counts below
are geometry projections, not HTTP, model delivery, or whole-response measures.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from decoder_probe_support import DecoderFixture
from network_algebra_hypotheses import (
    annihilator_widths, displacement_rank_bound, moment_bypass, packed_size,
    radix_pack, radix_size, radix_unpack, recurrence_leakage_witness,
    row_widths, signed_row_dictionary,
)


def timed(fn, repeats=3):
    samples = []
    for _ in range(repeats):
        started = time.process_time()
        fn()
        samples.append(time.process_time() - started)
    return float(np.median(samples))


def run(output: Path):
    from pllm import _native
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.stage_protocol import pack_residues

    if psutil.virtual_memory().available < 4 << 30:
        raise RuntimeError("public-weight screen requires 4 GiB physical headroom")
    before_swap = psutil.swap_memory().used
    fixture = DecoderFixture(inputs=64, outputs=8)
    native = MaskedGEMM(threads=2)
    rng, rows = np.random.default_rng(81781), 46
    entries = []
    result = {"schema": "pllm.network_algebra_hypotheses.v1", "complete": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "source": fixture.lock(),
              "platform": platform.platform(), "input_rows": 39, "generated_outputs": 8,
              "executed_rows_per_stage": rows, "stages": entries,
              "scope": "public source geometry and synthetic integer probes; no live protocol",
              "full_wire_bytes": None, "whole_client_peak_rss_bytes": None}
    for index, (key, stage) in enumerate(fixture.body.items()):
        w = stage.weight.values
        out_bits = row_widths(w)
        assert bytes(out_bits) == stage.output_residue_bits
        in_bits = annihilator_widths(w)
        uniform = np.full(w.shape[1], int(out_bits.max()), np.uint8)
        profile = stage.seeded_profile
        x = rng.integers(-127, 128, (3, w.shape[1]), dtype=np.int8)
        r = rng.integers(0, 2**profile.wire_bits, x.shape, dtype=np.uint32)
        s = rng.integers(0, 2**profile.wire_bits, (len(x), len(w)), dtype=np.uint32)
        a = ((x.astype(np.int64) - r) % 2**profile.wire_bits).astype("<u4")
        wr = stage.compiled_weight.wrap32(r)
        correction = _native.prepared_pack_correction(wr.astype("<u4").tobytes(),
            s.astype("<u4").tobytes(), bytes(out_bits), len(x))
        expected = stage.compiled_weight.clear(x).astype(np.int64)
        for widths in (in_bits, uniform):
            encoded = _native.offset_pack_rows(a.tobytes(), bytes(widths), len(x))
            reduced = np.frombuffer(_native.unpack_residue_rows(encoded, bytes(widths), len(x)), "<u4").reshape(a.shape)
            product = stage.compiled_weight.wrap32(reduced)
            response = _native.prepared_pack_output(product.astype("<u4").tobytes(), correction, bytes(out_bits), len(x))
            actual = np.frombuffer(_native.prepared_unpack_output(response, s.astype("<u4").tobytes(),
                bytes(out_bits), len(x)), "<i8").reshape(expected.shape)
            np.testing.assert_array_equal(actual, expected)
        out_bytes = packed_size(out_bits, rows)
        control_input = rows * w.shape[1] * (profile.wire_bits // 8)
        tight_input = packed_size(uniform, rows)
        record = {"stage": key, "weight_sha256": stage.weight_digest, "shape": list(w.shape),
                  "ring_bits": profile.wire_bits,
                  "control_arithmetic_bytes": control_input + 2 * out_bytes,
                  "annihilator": {"uniform_input_bits": int(out_bits.max()),
                      "coordinate_width_histogram": {str(int(k)): int((in_bits == k).sum()) for k in np.unique(in_bits)},
                      "uniform_arithmetic_bytes": tight_input + 2 * out_bytes,
                      "coordinate_arithmetic_bytes": packed_size(in_bits, rows) + 2 * out_bytes,
                      "integer_parity": True}, "moment_bypasses": []}
        # Odd rings need fresh uniform issuance, not reduction of existing 2^k masks.
        modulus = 2 * stage.signed_output_bound + 1
        radix_input = rng.integers(0, modulus, a.size, dtype=np.uint32)
        packed = radix_pack(radix_input, modulus)
        np.testing.assert_array_equal(radix_unpack(packed, len(radix_input), modulus), radix_input)
        odd_r = rng.integers(0, modulus, x.shape, dtype=np.uint32)
        odd_s = rng.integers(0, modulus, expected.shape, dtype=np.uint32)
        odd_a = ((x.astype(np.int64) - odd_r) % modulus).astype(np.uint32)
        odd_out = (stage.compiled_weight.modular(odd_a, modulus).astype(np.int64)
                   + stage.compiled_weight.modular(odd_r, modulus).astype(np.int64)
                   - odd_s.astype(np.int64)) % modulus
        restored = (odd_out + odd_s) % modulus
        restored = np.where(restored > modulus // 2, restored - modulus, restored)
        np.testing.assert_array_equal(restored, expected)
        record["odd_radix"] = {"modulus": modulus, "integer_parity": True,
            "arithmetic_bytes": radix_size(rows * w.shape[1], modulus) + 2 * radix_size(rows * len(w), modulus)}
        for groups in (1, 4, 16):
            residual, center = moment_bypass(w, groups)
            residual_bits = row_widths(residual)
            client_matrix = native.compile(center)
            server_matrix = native.compile(residual)
            sums = x.astype(np.int64).reshape(len(x), groups, -1).sum(axis=2).astype(np.uint32)
            partial = client_matrix.wrap32(sums).view(np.int32).astype(np.int64)
            np.testing.assert_array_equal(server_matrix.clear(x).astype(np.int64) + partial, expected)
            record["moment_bypasses"].append({"groups": groups, "integer_parity": True,
                "coefficient_bytes": center.nbytes, "client_coefficient_and_native_snapshot_bytes": 2 * center.nbytes,
                "client_integer_adds_and_macs": rows * (w.shape[1] + groups * len(w)),
                "arithmetic_bytes": (rows * w.shape[1] * int(residual_bits.max()) + 7) // 8
                                    + 2 * packed_size(residual_bits, rows)})
            del residual, center, client_matrix, server_matrix
        selected, mapping, signs = signed_row_dictionary(w)
        restored = np.zeros(expected.shape, np.int64)
        valid = mapping >= 0
        restored[:, valid] = expected[:, selected][:, mapping[valid]] * signs[valid]
        np.testing.assert_array_equal(restored, expected)
        record["row_dictionary"] = {"distinct_signed_rows": len(selected),
            "omitted_rows": len(w) - len(selected), "zero_rows": int((mapping < 0).sum()),
            "decoder_map_bytes": int(mapping.nbytes + signs.nbytes), "integer_parity": True,
            "arithmetic_bytes": tight_input + 2 * packed_size(out_bits[selected], rows)}
        rank, minor = displacement_rank_bound(w)
        record["shift_recurrence"] = {"gf2_minor_size": minor, "displacement_rank_lower_bound": rank,
            "optimistic_rank_coefficients_per_update": rank * (w.shape[1] + len(w)),
            "full_weight_coefficients": w.size}
        if index in (0, 1, 2, 3):
            # Conversion-inclusive client encoding / provider decode on 39-row geometry.
            sample = rng.integers(0, 2**profile.wire_bits, (39, w.shape[1]), dtype=np.uint32)
            packed_input = _native.offset_pack_rows(sample.astype("<u4").tobytes(), bytes(uniform), 39)
            record["codec_cpu"] = {
                "rows": 39,
                "raw_pack_seconds": timed(lambda: pack_residues(sample, profile.wire_bits)),
                "tight_pack_seconds": timed(lambda: _native.offset_pack_rows(sample.astype("<u4").tobytes(), bytes(uniform), 39)),
                "tight_unpack_seconds": timed(lambda: _native.unpack_residue_rows(packed_input, bytes(uniform), 39)),
                "radix_pack_three_rows_seconds": timed(lambda: radix_pack(radix_input, modulus)),
                "native_wrap_three_rows_seconds": timed(lambda: stage.compiled_weight.wrap32(a)),
                "native_odd_ring_three_rows_seconds": timed(lambda: stage.compiled_weight.modular(odd_a, modulus))}
        entries.append(record)
        if (index + 1) % 24 == 0:
            print(f"Checked {index + 1}/{len(fixture.body)} public stages", flush=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            gc.collect()
        if psutil.swap_memory().used > before_swap + (64 << 20) or psutil.virtual_memory().available < 2 << 30:
            raise RuntimeError("public-weight screen stopped on host memory pressure")
    summary = {"control_arithmetic_bytes": sum(s["control_arithmetic_bytes"] for s in entries),
        "stage_packed_arithmetic_bytes": sum(s["annihilator"]["uniform_arithmetic_bytes"] for s in entries),
        "coordinate_packed_arithmetic_bytes": sum(s["annihilator"]["coordinate_arithmetic_bytes"] for s in entries),
        "odd_radix_arithmetic_bytes": sum(s["odd_radix"]["arithmetic_bytes"] for s in entries),
        "signed_dictionary_omitted_rows": sum(s["row_dictionary"]["omitted_rows"] for s in entries),
        "signed_dictionary_map_bytes": sum(s["row_dictionary"]["decoder_map_bytes"] for s in entries),
        "minimum_displacement_rank_lower_bound": min(s["shift_recurrence"]["displacement_rank_lower_bound"] for s in entries),
        "moment_bypasses": [{"groups": g, **{k: sum(next(v for v in s["moment_bypasses"] if v["groups"] == g)[k]
            for s in entries) for k in ("coefficient_bytes", "client_coefficient_and_native_snapshot_bytes",
                                       "client_integer_adds_and_macs", "arithmetic_bytes")}} for g in (1, 4, 16)]}
    result.update(complete=True, summary=summary, recurrence_leakage=recurrence_leakage_witness(),
        new_swap_bytes=max(0, psutil.swap_memory().used - before_swap),
        research_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
        (1 if platform.system() == "Darwin" else 1024))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
