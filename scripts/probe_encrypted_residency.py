"""Finite encrypted residual-state boundary; serial versus interleaved SIMD lanes.

This is a synthetic polynomial-state probe, not a trained decoder or private
token feedback implementation. Only the final public head is decrypted.
All packed lanes belong to one client and one key context, not multiple clients.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import tempfile
import time


WIDTH = 8
ROWS = 2
TOLERANCE = 0.0002


def peak_rss_bytes(value, platform=sys.platform):
    return int(value) if platform == "darwin" else int(value * 1024)


def source_snapshot(paths):
    return {name: {"path": str(Path(path).resolve()),
                   "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for name, path in sorted(paths.items())}


def assert_sources_unchanged(expected, actual):
    if expected != actual:
        raise AssertionError("probe/helper/imported runtime sources changed during cohort")


def imported_source_paths():
    from pllm.runtime import ckks_reference, seal_frames, benchmark_memory, shutdown
    from pllm.metrics import BuildRef
    import frontier_probe_support
    modules = (ckks_reference, seal_frames, benchmark_memory, shutdown,
               frontier_probe_support, sys.modules[BuildRef.__module__])
    return {"probe_encrypted_residency": Path(__file__),
            **{module.__name__: Path(module.__file__) for module in modules}}


def schedule(steps=2, lanes=8):
    if type(steps) is not int or not 1 <= steps <= 3:
        raise ValueError("finite schedule admits one to three transitions, no refresh")
    if type(lanes) is not int or lanes not in (1, 2, 4, 8, 16):
        raise ValueError("lane count must be a power of two from one to sixteen")
    return {"width": WIDTH, "output_rows": ROWS, "steps": steps,
            "lanes": lanes, "depth": 2 * steps + 1, "live_ciphertexts": 16}


def fixture(lanes):
    # Public synthetic oracle fixture. No fixture values or decoded values are
    # persisted in evidence; this is not a private-prompt benchmark.
    weights = [[((r * 13 + c * 7) % 17 - 8) / 128
                for c in range(WIDTH)] for r in range(WIDTH)]
    head = [[((r * 5 + c * 11) % 13 - 6) / 64
             for c in range(WIDTH)] for r in range(ROWS)]
    inputs = [[((q * 19 + c * 3) % 23 - 11) / 32
               for c in range(WIDTH)] for q in range(lanes)]
    return weights, head, inputs


def oracle(weights, head, inputs, steps):
    output = []
    for vector in inputs:
        h = list(vector)
        for _ in range(steps):
            z = [math.fsum(w * x for w, x in zip(row, h, strict=True))
                 for row in weights]
            h = [x + y * y for x, y in zip(h, z, strict=True)]
        output.append([math.fsum(w * x for w, x in zip(row, h, strict=True))
                       for row in head])
    return output


def pack(inputs, slots):
    lanes = len(inputs)
    schedule(lanes=lanes)
    if slots % (WIDTH * lanes) or any(len(x) != WIDTH for x in inputs):
        raise ValueError("slot geometry cannot repeat complete interleaved periods")
    period = [inputs[q][c] for c in range(WIDTH) for q in range(lanes)]
    return period * (slots // len(period))


def diagonals(matrix):
    if not 1 <= len(matrix) <= WIDTH or any(len(row) != WIDTH for row in matrix):
        raise ValueError("public matrix shape differs")
    if any(not math.isfinite(x) or abs(x) > 1 for row in matrix for x in row):
        raise ValueError("public weights outside finite [-1,1] contract")
    return [[matrix[r][(r + shift) % WIDTH] if r < len(matrix) else 0.0
             for r in range(WIDTH)] for shift in range(WIDTH)]


def linear(arithmetic, x, layout, lanes):
    # One key rotates by lane count, not one slot. Coordinate-major interleave
    # keeps each lane independent, including period wraparound.
    slots = arithmetic.encoder.slot_count()
    if slots % (WIDTH * lanes):
        raise ValueError("incomplete SIMD period")
    shifted, result = x, None
    for index, diagonal in enumerate(layout):
        if index:
            shifted = arithmetic.rotate(shifted, lanes)
        coefficients = [v for v in diagonal for _ in range(lanes)]
        term = arithmetic.multiply_plain(shifted, coefficients * (slots // len(coefficients)))
        result = term if result is None else arithmetic.add(result, term)
    return result


def resident_boundary(arithmetic, x, weights, head, *, steps, lanes):
    schedule(steps, lanes)
    h = x
    for _ in range(steps):
        z = linear(arithmetic, h, weights, lanes)
        # h remains ciphertext-owned by evaluator. Adding lower-level z^2
        # explicitly modulus-switches residual h; no decrypt/re-encrypt refresh.
        h = arithmetic.add(h, arithmetic.multiply(z, z))
    return linear(arithmetic, h, head, lanes)


def preflight(steps, lanes):
    from pllm import _native
    spec = schedule(steps, lanes)
    metadata = json.loads(_native.ckks_parameters(spec["depth"], 1, True, spec["live_ciphertexts"]))
    slots = metadata["parameters"]["degree"] // 2
    pack(fixture(lanes)[2], slots)
    return spec, metadata


def resource_gate(spec, parameters):
    """Executable padded-frame schedule, not observed crypto/wire evidence."""
    setup_bytes = (512 + parameters["public_key_frame_bytes"]
                   + 2 * (parameters["switching_key_frame_bytes"] + 4096))
    request_bytes = parameters["ciphertext_frame_bytes"] + 81
    final_reply_bytes = 2 * parameters["degree"] * 8 + 4096 + 81
    packed_online = request_bytes + final_reply_bytes
    serial_online = spec["lanes"] * packed_online
    return {"scope": "public padded-frame prediction; no keys or ciphertexts generated",
            "setup_delivery_bytes_each_mode": setup_bytes,
            "serial_online_frame_bytes": serial_online,
            "packed_online_frame_bytes": packed_online,
            "online_ratio": serial_online / packed_online,
            "setup_inclusive_ratio": (setup_bytes + serial_online) / (setup_bytes + packed_online),
            "physical_wire_bytes": None}


def worker(payload):
    from pllm import _native
    from pllm.runtime import ckks_reference as ckks
    import _sealapi_cpp as seal
    source_paths = imported_source_paths()
    frozen_sources = source_snapshot(source_paths)
    assert_sources_unchanged(payload["frozen_sources"], frozen_sources)
    # Binding uses file-only serialization. Count actual unpadded payload file
    # writes/reads, not physical filesystem traffic or padded peer frame bytes.
    disk = {"serialized_file_write_bytes": 0, "serialized_file_read_bytes": 0}
    save, load = ckks.save_frame, ckks.load_frame

    def charged_save(*args, **kwargs):
        frame = save(*args, **kwargs)
        length = struct.unpack_from("<Q", frame)[0]
        disk["serialized_file_write_bytes"] += length
        disk["serialized_file_read_bytes"] += length
        return frame

    def charged_load(obj, context, frame, path, capacity):
        result = load(obj, context, frame, path, capacity)
        length = struct.unpack_from("<Q", frame)[0]
        disk["serialized_file_write_bytes"] += length
        disk["serialized_file_read_bytes"] += length
        return result

    ckks.save_frame, ckks.load_frame = charged_save, charged_load
    mode, steps, total = payload["mode"], payload["steps"], payload["lanes"]
    if mode not in ("serial", "packed"):
        raise ValueError("unknown boundary mode")
    spec, admitted = preflight(steps, total)
    lanes = 1 if mode == "serial" else total
    weights, head, inputs = fixture(total)
    expected = oracle(weights, head, inputs, steps)
    started = time.perf_counter()
    w_layout, h_layout = diagonals(weights), diagonals(head)
    public_seconds = time.perf_counter() - started
    binding = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).digest()
    setup_cpu = time.process_time()
    with tempfile.TemporaryDirectory(prefix="pllm-encrypted-residency-") as directory:
        client, provider, metadata = ckks.setup(seal, _native, Path(directory),
            depth=spec["depth"], rotation_steps=(lanes,), relinearization=True,
            live_ciphertexts=spec["live_ciphertexts"], queries=total if mode == "serial" else 1,
            binding=binding)
        setup_cpu = time.process_time() - setup_cpu
        process_cpu, wall = time.process_time(), time.perf_counter()
        client_cpu = provider_cpu = 0.0
        validation_cpu = validation_wall = 0.0
        frames, errors, padded_errors = [], [], []
        try:
            groups = [[x] for x in inputs] if mode == "serial" else [inputs]
            for index, group in enumerate(groups):
                started = time.process_time()
                request = client.request(pack(group, client.encoder.slot_count()))
                client_cpu += time.process_time() - started
                started = time.process_time()
                reply = provider.evaluate(request, lambda a, x: resident_boundary(a, x,
                    w_layout, h_layout, steps=steps, lanes=lanes))
                provider_cpu += time.process_time() - started
                started = time.process_time()
                decoded = client.receive(reply, spec["depth"])
                client_cpu += time.process_time() - started
                check_cpu, check_wall = time.process_time(), time.perf_counter()
                targets = expected[index:index + 1] if mode == "serial" else expected
                period = WIDTH * lanes
                for base in range(0, len(decoded), period):
                    for row in range(WIDTH):
                        for q in range(lanes):
                            target = targets[q][row] if row < ROWS else 0.0
                            error = abs(decoded[base + row * lanes + q] - target)
                            (errors if row < ROWS else padded_errors).append(error)
                validation_cpu += time.process_time() - check_cpu
                validation_wall += time.perf_counter() - check_wall
                frames.append({"request": len(request), "reply": len(reply)})
            online_wall = time.perf_counter() - wall - validation_wall
            online_cpu = time.process_time() - process_cpu - validation_cpu
            if max(errors + padded_errors) > TOLERANCE:
                raise RuntimeError("finite encrypted boundary exceeds independent float64 oracle")
            counts = dict(provider.arithmetic.counts)
            remaining = provider.arithmetic.minimum_remaining_depth
            if remaining != 0:
                raise AssertionError("public finite depth schedule differs")
        finally:
            client.retire()
            provider.retire()
        setup_frames = metadata["setup_frame_bytes"]
        setup_bytes = sum(setup_frames.values())
        online_bytes = sum(sum(x.values()) for x in frames)
        gate = resource_gate(spec, metadata["parameters"])
        if (setup_bytes != gate["setup_delivery_bytes_each_mode"]
                or online_bytes != gate[f"{mode}_online_frame_bytes"]):
            raise AssertionError("actual frames differ from public resource gate")
        assert_sources_unchanged(frozen_sources, source_snapshot(source_paths))
        # These are payload/accounting bounds, not measured per-party RSS.
        return {"mode": mode, "schedule": spec,
            "source_identity": frozen_sources,
            "public_execution": {"rotation_steps": [lanes],
                "ciphertext_queries": total if mode == "serial" else 1,
                "client_key_contexts": 1, "cross_client_multikey_batching": False},
            "native_implementation_digest": admitted["native_implementation_digest"],
            "parameters": metadata["parameters"], "setup_frames": setup_frames,
            "frames": frames, "setup_bytes": setup_bytes, "online_bytes": online_bytes,
            "total_delivery_bytes": setup_bytes + online_bytes,
            "owners": {"client_secret_polynomial_bytes": metadata["secret_polynomial_bytes"],
                "client_public_key_frame_bytes": setup_frames["public-key"],
                "provider_public_evaluation_key_frame_bytes": setup_bytes - setup_frames["parameters"],
                "public_matrix_float64_bytes": 8 * (WIDTH * WIDTH + ROWS * WIDTH),
                "public_diagonal_float64_bytes": 2 * WIDTH * WIDTH * 8,
                "combined_native_allocation_estimate_bytes": metadata["parameters"]["allocation_bytes_estimate"],
                "live_ciphertext_admission_count": spec["live_ciphertexts"],
                "contexts": {"client_secret_context": 1, "provider_public_only_context": 1},
                "serialization_file_payload_io": disk, "physical_disk_io_bytes": None,
                "physical_wire_bytes": None},
            "quality": {"checked_output_slots": len(errors), "checked_zero_padding_slots": len(padded_errors),
                "unique_output_values": total * ROWS, "maximum_absolute_error": max(errors),
                "maximum_zero_padding_error": max(padded_errors), "tolerance": TOLERANCE,
                "exact_integer_outputs": 0},
            "operations": counts, "remaining_depth": remaining,
            "timings": {"public_layout_wall_seconds": public_seconds,
                "setup_wall_seconds": metadata["setup_seconds"], "setup_process_cpu_seconds": setup_cpu,
                "online_wall_seconds": online_wall, "online_process_cpu_seconds": online_cpu,
                "oracle_validation_cpu_seconds": validation_cpu,
                "oracle_validation_wall_seconds": validation_wall,
                "client_online_cpu_seconds": client_cpu, "provider_online_cpu_seconds": provider_cpu},
            "whole_process_peak_rss_bytes": peak_rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "whole_response_metrics": {k: None for k in ("client_rss", "client_cpu", "preparation_cpu",
                "preparation_rss", "ttft", "n_minus_1_decode_tps", "all_link_traffic")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare", action="store_true", help="public admission only; no key generation")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--processes", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.prepare:
        spec, metadata = preflight(args.steps, args.lanes)
        print(json.dumps({"schedule": spec, "admission": metadata,
                          "resource_gate": resource_gate(spec, metadata["parameters"])}, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("new --output path required")
    spec, _ = preflight(args.steps, args.lanes)
    from frontier_probe_support import fresh_workers
    source_paths = imported_source_paths()
    frozen_sources = source_snapshot(source_paths)
    pairs = []
    with fresh_workers(__file__, memory_bytes=512 << 20, timeout=240) as (run, guard, host):
        for repeat in range(args.processes):
            pair = {}
            order = ("serial", "packed") if repeat % 2 == 0 else ("packed", "serial")
            for mode in order:
                assert_sources_unchanged(frozen_sources, source_snapshot(source_paths))
                pair[mode] = run({"mode": mode, "steps": args.steps, "lanes": args.lanes,
                                  "frozen_sources": frozen_sources})
                assert_sources_unchanged(frozen_sources, pair[mode]["source_identity"])
            for key in ("schedule", "native_implementation_digest", "parameters", "setup_frames", "source_identity"):
                if pair["serial"][key] != pair["packed"][key]:
                    raise AssertionError(f"unmatched public identity: {key}")
            pair["ratios_control_over_candidate"] = {
                "online_frame_bytes": pair["serial"]["online_bytes"] / pair["packed"]["online_bytes"],
                "setup_plus_online_frame_bytes": pair["serial"]["total_delivery_bytes"] / pair["packed"]["total_delivery_bytes"],
                "client_online_cpu": pair["serial"]["timings"]["client_online_cpu_seconds"] / pair["packed"]["timings"]["client_online_cpu_seconds"]}
            pairs.append(pair)
    assert_sources_unchanged(frozen_sources, source_snapshot(source_paths))
    report = {"schema": "pllm.encrypted_residency_probe.v1", "scope": __doc__, "schedule": spec,
        "samples": pairs, "host": asdict(host), "build": run.build,
        "source_identity_frozen_before_workers": frozen_sources,
        "source_identity_verified_unchanged_after_workers": True,
        "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
        "limitations": ["synthetic finite polynomial recurrence, not checkpoint decoder execution",
            "no token selection, feedback, encrypted KV, bootstrapping or malicious-result proof",
            "head zeros hide other logical coordinates only within observed CKKS tolerance; no circuit privacy",
            "same key reused across independent randomized inputs, no one-use garbling material",
            "all packed lanes belong to one client/key context; cross-client multi-key batching not established",
            "local serialization file payload bytes charged; physical disk and transport traffic unknown",
            "process RSS includes both roles; per-party RSS and whole-response serving metrics unknown"]}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": len(pairs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
