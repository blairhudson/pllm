"""Bounded synthetic MLP conveyor and conservative public event simulation.

This does not execute the SDK or measure serving performance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import weakref

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/probe_mlp_conveyor.py", "tests/test_mlp_conveyor_probe.py",
    "python/pllm/runtime/mlp_tiling.py", "python/pllm/runtime/semantic_executor.py",
    "python/pllm/runtime/transformer_client.py", "python/pllm/runtime/client.py",
    "python/pllm/runtime/server.py", "python/pllm/runtime/transformer_engine.py",
    "python/pllm/runtime/protocol.py", "crates/pllm-compiler/src/mlp_tiling.rs",
    "docs/research/mlp-working-set-reference.md",
    "docs/research/prepared-prefill-reference.md",
    "docs/research/sota-combinations-reference.md",
    "docs/research/mlp-conveyor-reference.md",
)


@dataclass(frozen=True)
class Contract:
    rows: int
    hidden: int = 896
    intermediate: int = 4864
    tile: int = 64
    chunk: int = 16
    depth: int = 2
    input_bits: int = 32
    output_bits: int = 32
    frame_overhead: int = 1024
    frame_cap: int = 16 << 20

    def __post_init__(self):
        for name, bound in (("rows", 384), ("hidden", 8192), ("intermediate", 32768),
                            ("tile", 256), ("chunk", 32), ("depth", 2)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= bound:
                raise ValueError(f"invalid public {name}")
        if self.depth not in (1, 2) or self.chunk not in (4, 8, 16, 32):
            raise ValueError("unsupported public wavefront")
        if (self.input_bits, self.output_bits, self.frame_overhead, self.frame_cap) != (32, 32, 1024, 16 << 20):
            raise ValueError("frozen conservative codec envelope changed")
        if max(self.frame_bytes("G", self.chunk) + self.frame_bytes("D", self.chunk)) > self.frame_cap:
            raise ValueError("frame exceeds public cap")
        t = min(self.rows, self.tile)
        frames = (t + self.chunk - 1) // self.chunk
        if t * max(self.hidden, 2 * self.intermediate) * 4 + frames * 1024 > self.frame_cap:
            raise ValueError("stage aggregate exceeds public cap")

    def frame_bytes(self, stage, rows):
        inputs, outputs = ((self.hidden, 2 * self.intermediate) if stage == "G"
                           else (self.intermediate, self.hidden))
        return (rows * inputs * 4 + self.frame_overhead,
                rows * outputs * 4 + self.frame_overhead)

    def tiles(self):
        return [(start, min(self.rows, start + self.tile)) for start in range(0, self.rows, self.tile)]


def public_order(contract):
    """Fixed two-slot order; each stage's row claims remain ascending."""
    count = len(contract.tiles())
    if contract.depth == 1:
        return [(stage, tile) for tile in range(count) for stage in ("G", "D")]
    result = [("G", 0)]
    for tile in range(count):
        if tile + 1 < count:
            result.append(("G", tile + 1))
        result.append(("D", tile))
    return result


class Lease:
    """Independent finite one-use stage-row and ordered-frame oracle."""
    def __init__(self, rows):
        self.rows = rows
        self.next = {"G": 0, "D": 0}
        self.claimed = set()
        self.closed = False
        self.sequence = 0

    def claim(self, stage, start, end, sequence):
        if (self.closed or stage not in self.next or start != self.next[stage]
                or not start < end <= self.rows or sequence != self.sequence):
            raise ValueError("closed, replayed, or out-of-order claim")
        tickets = {(stage, row) for row in range(start, end)}
        if self.claimed & tickets:
            raise ValueError("one-use row replay")
        self.claimed |= tickets
        self.next[stage] = end
        self.sequence += 1

    def close(self):
        self.closed = True
        return 2 * self.rows - len(self.claimed)


def local_product(gate, up):
    # Fixed float32 evaluation; no reassociation or feature reduction splitting.
    return (gate / (np.float32(1) + np.exp(-gate))) * up


def project(value, weights, scalar=False):
    scales = np.maximum(np.max(np.abs(value), axis=1), np.float32(1e-6)) / np.float32(127)
    quantized = np.clip(np.rint(value / scales[:, None]), -127, 127).astype(np.int64)
    if scalar:
        integer = np.array([[sum(int(x) * int(w) for x, w in zip(row, column))
                             for column in weights] for row in quantized], dtype=np.int64)
    else:
        integer = quantized @ weights.T.astype(np.int64)
    return integer.astype(np.float32) * scales[:, None] * np.float32(0.01)


def numeric_probe(rows, tile=7, depth=2, cancel_after=None):
    """Tiny generated body only; scalar integer projection reference, no checkpoint."""
    c = Contract(rows, hidden=8, intermediate=13, tile=tile, depth=depth)
    rng = np.random.default_rng(4101 + rows)
    source = rng.normal(size=(rows, 8)).astype(np.float32)
    gate = rng.integers(-5, 6, size=(26, 8), dtype=np.int8)
    down = rng.integers(-5, 6, size=(8, 13), dtype=np.int8)
    complete = project(source, gate, scalar=True)
    expected = project(local_product(complete[:, :13], complete[:, 13:]), down, scalar=True)
    result = np.empty_like(source)
    live = {}
    products = []
    lease = Lease(rows)
    peak = 0
    executed = 0
    canceled = False
    try:
        for stage, index in public_order(c):
            start, end = c.tiles()[index]
            lease.claim(stage, start, end, lease.sequence)
            if stage == "G":
                both = project(source[start:end], gate)
                live[index] = local_product(both[:, :13], both[:, 13:])
                products.append(weakref.ref(live[index]))
                del both
                peak = max(peak, len(live))
                if peak > depth:
                    raise AssertionError("owner credit exceeded")
            else:
                result[start:end] = project(live.pop(index), down)
                if products[index]() is not None:
                    raise AssertionError("completed product owner retained")
            executed += 1
            if executed == cancel_after:
                canceled = True
                break
    finally:
        burned = lease.close()
        live.clear()
        if any(reference() is not None for reference in products):
            raise AssertionError("terminal product owner retained")
    try:
        lease.claim("G", 0, 1, 0)
    except ValueError:
        replay_rejected = True
    else:
        raise AssertionError("terminal lease accepted replay")
    equal = None if canceled else bool(np.array_equal(result.view(np.uint32), expected.view(np.uint32)))
    if equal is False:
        raise AssertionError("synthetic complete float32 result changed")
    return {"rows": rows, "tile": tile, "depth": depth, "exact_bits": equal,
            "canceled": canceled, "claimed": len(lease.claimed), "burned": burned,
            "peak_product_owners": peak, "terminal_product_owners": len(live),
            "weakref_retirement": True,
            "replay_rejected": replay_rejected}


def simulate(c, *, local_seconds_per_row=0.00002, provider_seconds_per_mac=1e-11,
             upload_mbps=40, download_mbps=40, local_stall_seconds=0):
    """Resource DAG, not a timer. Provider receive/compute/send is serialized.

    One queued request frame allowed; two active tile credits. Transmission is
    store-and-forward, conservative versus byte-stream TCP overlap. Both party
    upload limits are 40 Mbps, hence effective download is also at most 40.
    """
    parameters = (local_seconds_per_row, provider_seconds_per_mac, local_stall_seconds)
    if any(not np.isfinite(v) or v < 0 for v in parameters):
        raise ValueError("invalid modeled duration")
    if not 0 < upload_mbps <= 40 or not 0 < download_mbps <= 40:
        raise ValueError("shared per-party link cap exceeded")
    upload = provider = local = 0.0
    previous_provider_start = 0.0
    done = {}
    events = []
    owner_intervals = {}
    totals = [0, 0]
    for stage, index in public_order(c):
        start, end = c.tiles()[index]
        release = (done.get(("D", index - c.depth), 0) if stage == "G"
                   else done[("L", index)])
        stage_end = 0.0
        for offset in range(start, end, c.chunk):
            count = min(c.chunk, end - offset)
            tx, rx = c.frame_bytes(stage, count)
            # Next request cannot finish uploading before previous request was
            # received. This freezes the queue to one whole request frame.
            duration = tx * 8 / (upload_mbps * 1e6)
            u0 = max(upload, release, previous_provider_start - duration)
            u1 = u0 + duration
            p0 = max(provider, u1)
            macs = count * c.hidden * c.intermediate * (2 if stage == "G" else 1)
            p1 = p0 + macs * provider_seconds_per_mac
            r1 = p1 + rx * 8 / (download_mbps * 1e6)
            events.append({"stage": stage, "tile": index, "row": offset, "rows": count,
                           "upload_start": u0, "upload_end": u1,
                           "compute_start": p0, "compute_end": p1, "reply_end": r1,
                           "tx_bytes": tx, "rx_bytes": rx})
            if stage == "G" and offset == start:
                owner_intervals[index] = [u0, None]
            upload, provider, previous_provider_start = u1, r1, p0
            stage_end = r1
            totals[0] += tx
            totals[1] += rx
        done[(stage, index)] = stage_end
        if stage == "G":
            l0 = max(local, stage_end)
            local = l0 + (end - start) * local_seconds_per_row + local_stall_seconds
            done[("L", index)] = local
        else:
            owner_intervals[index][1] = stage_end
    points = sorted([(a, 1) for a, b in owner_intervals.values()] +
                    [(b, -1) for a, b in owner_intervals.values()])
    active = peak = 0
    for _, delta in points:
        active += delta
        peak = max(peak, active)
    if peak > c.depth or active:
        raise AssertionError("modeled owner credit violated")
    return {"seconds": provider, "tx_bytes": totals[0], "rx_bytes": totals[1],
            "frames": len(events), "peak_tile_owners": peak, "events": events,
            "provider_mac_seconds": c.rows * 3 * c.hidden * c.intermediate * provider_seconds_per_mac,
            "local_seconds": c.rows * local_seconds_per_row + len(c.tiles()) * local_stall_seconds}


def ownership(c, original_client_bytes=None):
    """Conservative allocation ledger, not RSS/admission. Includes alias copies."""
    t, h, m, d = min(c.rows, c.tile), c.hidden, c.intermediate, c.depth
    # Each slot gets worst-case simultaneous buffers, even where lifetimes differ.
    slot = {"gate_up_f32": t * 2 * m * 4, "silu_product_scratch_f32": t * 5 * m * 4,
            "activation_quantized_i64": t * max(h, m) * 8,
            "activation_scales_f32": t * 4, "input_mask_u32": t * max(h, m) * 4,
            "output_correction_u32": t * max(2 * m, h) * 4,
            "decoded_parts_plus_concatenation_f32": t * max(2 * m, h) * 8,
            "down_output_f32": t * h * 4,
            "request_list_and_envelopes": t * max(h, m) * 8 + ((t + c.chunk - 1) // c.chunk) * 2048,
            "reply_list_and_envelopes": t * max(2 * m, h) * 8 + ((t + c.chunk - 1) // c.chunk) * 2048}
    max_tx = max(c.frame_bytes(s, c.chunk)[0] for s in ("G", "D"))
    max_rx = max(c.frame_bytes(s, c.chunk)[1] for s in ("G", "D"))
    shared = {"original_full_input_f32": c.rows * h * 4,
              "original_full_output_f32": c.rows * h * 4,
              "original_residual_f32": c.rows * h * 4,
              "transport_tx_copy": max_tx, "transport_rx_copy": max_rx}
    subtotal = d * sum(slot.values()) + sum(shared.values())
    n = min(c.rows, c.chunk)
    provider = {"one_queued_request_frame": max_tx,
                "executing_request_frame": max_tx,
                "decoded_input_plus_combined_u32": n * max(h, m) * 8,
                "kernel_output_u32": n * max(2 * m, h) * 4,
                "correction_parts_plus_concatenation_u32": n * max(2 * m, h) * 8,
                "modulo_and_cast_scratch_i64": n * max(2 * m, h) * 32,
                "prepared_reply_u32": n * max(2 * m, h) * 4,
                "reply_frame_and_transport_copy": 2 * max_rx}
    whole = None if original_client_bytes is None else original_client_bytes + subtotal
    return {"slot_count": d, "per_slot": slot, "shared": shared,
            "priced_region_bytes": subtotal, "original_nonregion_client_bytes": original_client_bytes,
            "provider_inflight": provider, "provider_inflight_bytes": sum(provider.values()),
            "provider_original_weights_inventory_runtime_bytes": None,
            "whole_client_bytes": whole,
            "low_256_mib_fits": None if whole is None else whole <= 256 << 20,
            "medium_2_gib_fits": None if whole is None else whole <= 2 << 30,
            "unpriced_original_owners": ["KV/state", "attention", "all other full outputs",
                "reserved inventory/paged corrections", "import weights and metadata",
                "interpreter/native/allocator", "dashboard", "WAN relay/socket buffers"]}


def report():
    controls = [numeric_probe(rows, tile, depth) for rows in (1, 7, 17, 150, 288)
                for tile in (7, 64) for depth in (1, 2)]
    controls += [numeric_probe(17, 7, 2, cancel_after=n) for n in (1, 2, 3, 5)]
    costs = []
    for rows in (150, 288):
        for tile in (16, 64):
            for local_rate in (0, 0.00002, 0.001):
                baseline = Contract(rows, tile=tile, depth=1)
                candidate = Contract(rows, tile=tile, depth=2)
                a = simulate(baseline, local_seconds_per_row=local_rate)
                b = simulate(candidate, local_seconds_per_row=local_rate)
                costs.append({"input_rows": rows, "output_allowance": 8, "tile_rows": tile,
                    "local_seconds_per_row_assumed": local_rate,
                    "serial_region_seconds_modeled": a["seconds"],
                    "conveyor_region_seconds_modeled": b["seconds"],
                    "saved_seconds_modeled": a["seconds"] - b["seconds"],
                    "same_frames_and_bytes": (a["frames"], a["tx_bytes"], a["rx_bytes"]) ==
                                             (b["frames"], b["tx_bytes"], b["rx_bytes"]),
                    "conveyor_peak_tile_owners": b["peak_tile_owners"],
                    "frames_per_region": b["frames"], "tx_bytes_per_region": b["tx_bytes"],
                    "rx_bytes_per_region": b["rx_bytes"],
                    "serial_ownership": ownership(baseline), "conveyor_ownership": ownership(candidate)})
    return {"schema": "pllm.mlp_conveyor_probe.v1", "scope": "synthetic correctness and simulation only",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES},
            "contract": asdict(Contract(288)), "numeric_controls": controls, "cost_screen": costs,
            "existing_model_exactness": None, "installed_execution": False,
            "remote_body_mac_fraction": None, "whole_client_qualification": False,
            "privacy_gate": "unpadded private-dependent readiness/timing adaptation rejected",
            "verdict": "native cross-stage gate missing; no serving or Pareto claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path,
                        help="Require the exact source hashes from an earlier probe report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh evidence path")
    if args.source_lock is not None:
        locked = json.loads(args.source_lock.read_text())["source_sha256"]
        current = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}
        if locked != current:
            parser.error("probe source lock changed; create a new research screen")
    result = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
