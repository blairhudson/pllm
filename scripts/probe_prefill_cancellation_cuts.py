"""Bounded hypothetical graph cuts. No SDK, model loading, timing, or security claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
LOW = 256 << 20
SOURCES = (
    "docs/research/epoch-prefix-sdk-reference.md",
    "docs/research/prepared-frontier.md",
    "docs/plans/low-tier-all-link-hypotheses.md",
    "crates/pllm-compiler/src/decoder_continuation.rs",
    "crates/pllm-compiler/src/decoder_runtime_schedule.rs",
    "python/pllm/runtime/client.py",
    "python/pllm/runtime/transformer_client.py",
    "python/pllm/runtime/prepared_accounting.py",
    "crates/pllm-core/src/prepared_stage.rs",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Node:
    name: str
    inputs: tuple[str, ...]
    kind: str
    persistent: bool = False


def graph(layers=2):
    if type(layers) is not int or not 1 <= layers <= 4:
        raise ValueError("bounded layer count")
    result = []
    h = "input"
    for layer in range(layers):
        kv, mix, up, down, out = (f"{layer}.{n}" for n in ("kv", "mix", "up", "down", "out"))
        result.extend((Node(kv, (h,), "remote", True), Node(mix, (h, kv), "add"),
                       Node(up, (mix,), "remote"), Node(down, (up,), "remote"),
                       Node(out, (h, down), "add")))
        h = out
    return tuple(result)


def validate(nodes, cuts):
    known = {"input"}
    for node in nodes:
        if node.name in known or node.kind not in {"remote", "add"} or not set(node.inputs) <= known:
            raise ValueError("dependency/order violation")
        known.add(node.name)
    if (not cuts or any(type(c) is not int for c in cuts)
            or tuple(sorted(set(cuts))) != tuple(cuts) or cuts[0] < 1 or cuts[-1] != len(nodes)):
        raise ValueError("public cuts must be increasing and finish graph")


def live_ids(nodes, pc):
    produced = {"input"} | {n.name for n in nodes[:pc]}
    needed = {i for n in nodes[pc:] for i in n.inputs}
    needed |= {n.name for n in nodes[:pc] if n.persistent}
    if pc == len(nodes):
        needed.add(nodes[-1].name)
    return tuple(sorted(produced & needed))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def execute(nodes, values, begin, end):
    """Toy i8 quantize -> integer linear -> original f32 boundary; not model numerics."""
    for node in nodes[begin:end]:
        rows = [values[i] for i in node.inputs]
        if node.kind == "remote":
            values[node.name] = tuple(f32(max(-127, min(127, round(x * 8))) * 3 / 8)
                                      for x in rows[0])
        else:
            values[node.name] = tuple(f32(x + y) for x, y in zip(*rows, strict=True))


def export_state(nodes, values, pc, *, generation=0):
    live = live_ids(nodes, pc)
    return {"schema": "toy.prefill_cut.v1", "graph_digest": digest([asdict(n) for n in nodes]),
            "numeric": "toy_i8_round_even_f32_v1", "pc": pc, "generation": generation,
            "tensors": {i: {"dtype": "f32le", "shape": [len(values[i])],
                            "hex": struct.pack(f"<{len(values[i])}f", *values[i]).hex(),
                            "sha256": hashlib.sha256(struct.pack(f"<{len(values[i])}f", *values[i])).hexdigest()}
                        for i in live}}


def restore_state(nodes, state, pc, rows, generation):
    if type(rows) is not int or not 1 <= rows <= 16:
        raise ValueError("bounded row geometry required")
    if (set(state) != {"schema", "graph_digest", "numeric", "pc", "generation", "tensors"}
            or state["schema"] != "toy.prefill_cut.v1"
            or state["graph_digest"] != digest([asdict(n) for n in nodes])
            or state["numeric"] != "toy_i8_round_even_f32_v1"
            or state["pc"] != pc or state["generation"] != generation
            or set(state["tensors"]) != set(live_ids(nodes, pc))):
        raise ValueError("cut binding/live manifest mismatch")
    result = {}
    for name, tensor in state["tensors"].items():
        if set(tensor) != {"dtype", "shape", "hex", "sha256"} or tensor["dtype"] != "f32le" or tensor["shape"] != [rows]:
            raise ValueError("typed geometry mismatch")
        if not isinstance(tensor["hex"], str) or len(tensor["hex"]) != rows * 8:
            raise ValueError("payload bound mismatch")
        raw = bytes.fromhex(tensor["hex"])
        if len(raw) != rows * 4 or hashlib.sha256(raw).hexdigest() != tensor["sha256"]:
            raise ValueError("payload bound mismatch")
        result[name] = struct.unpack(f"<{rows}f", raw)
    return result


@dataclass
class Ledger:
    issued: int = 0
    correction_acked: int = 0
    correction_inflight: int = 0
    reserved: int = 0
    claimed: int = 0
    online_acked: int = 0
    online_inflight: int = 0
    burned_unused: int = 0
    discarded: int = 0
    active: bool = False
    blocked: bool = False
    generation: int = 0

    def prepare(self, rows):
        if self.active or self.blocked or rows <= 0:
            raise ValueError("Preparation requires confirmed retirement")
        self.issued += rows
        self.correction_inflight += rows

    def seal(self):
        if self.active or self.blocked or not self.correction_inflight:
            raise ValueError("seal state")
        self.correction_acked += self.correction_inflight
        self.reserved += self.correction_inflight
        self.correction_inflight = 0
        self.active = True

    def claim(self, rows, ack=True):
        if not self.active or rows <= 0 or self.claimed + self.burned_unused + rows > self.reserved:
            raise ValueError("claim/replay/exhaustion")
        self.claimed += rows
        if ack:
            self.online_acked += rows
        else:
            self.online_inflight += rows

    def retire(self, authenticated=True):
        # Claims burn even if reply never arrived. Never make masks reusable.
        self.burned_unused = self.reserved - self.claimed
        self.discarded += self.correction_inflight
        self.correction_inflight = 0
        self.online_inflight = 0
        self.active = False
        self.blocked = not authenticated
        self.generation += 1

    def audit(self):
        if self.active or self.correction_inflight:
            raise ValueError("audit requires terminal state")
        assert self.issued == self.reserved + self.discarded
        assert self.reserved == self.claimed + self.burned_unused
        return {**asdict(self), "burn_on_claim": self.claimed,
                "burn_total_reserved": self.claimed + self.burned_unused}


@dataclass(frozen=True)
class Prices:
    rows: int = 4
    correction: int = 96
    ingress: int = 64
    reply: int = 96
    session_control: int = 1024
    cut_control: int = 512
    cold_public: int = 4096
    epoch_rows: int = 12


def material_bytes(rows, prices):
    """Overlapping lifecycle gauges, never added together as network totals."""
    return {"correction_issued": rows["issued"] * prices.correction,
            "correction_acked": rows["correction_acked"] * prices.correction,
            "correction_inflight_upper": rows["correction_inflight"] * prices.correction,
            "reserved_correction": rows["reserved"] * prices.correction,
            "claimed_burn_correction": rows["claimed"] * prices.correction,
            "unused_burn_correction": rows["burned_unused"] * prices.correction,
            "discarded_correction": rows["discarded"] * prices.correction,
            "online_acked_reply": rows["online_acked"] * prices.reply,
            "online_inflight_reply_upper": rows["online_inflight"] * prices.reply}


def simulate(cuts, *, stop=None, point="cut", prices=Prices()):
    """Defined observations: preparing, sealed, online_inflight, cut; stop is public pc."""
    nodes = graph()
    validate(nodes, cuts)
    if point not in {"preparing", "sealed", "online_inflight", "acknowledged", "cut"}:
        raise ValueError("unknown observable point")
    if stop is not None and (stop not in cuts if point not in {"online_inflight", "acknowledged"} else
                             not 1 <= stop <= len(nodes) or
                             (point == "online_inflight" and nodes[stop - 1].kind != "remote")):
        raise ValueError("stop not observable")
    ledger = Ledger()
    control, sessions, transitions, online, returned = 0, 0, 0, 0, 0
    snapshots = []
    begin = 0
    for end in cuts:
        segment_rows = prices.rows * sum(n.kind == "remote" for n in nodes[begin:end])
        if end == len(nodes):
            segment_rows += prices.epoch_rows
        ledger.prepare(segment_rows)
        if stop == end and point == "preparing":
            snapshots.append(asdict(ledger))
            ledger.retire()
            break
        ledger.seal()
        sessions += 1
        control += prices.session_control
        if stop == end and point == "sealed":
            ledger.retire()
            break
        interrupted = False
        for pc in range(begin + 1, end + 1):
            if nodes[pc - 1].kind != "remote":
                if point == "acknowledged" and stop == pc:
                    interrupted = True
                    break
                continue
            inflight = point == "online_inflight" and stop == pc
            ledger.claim(prices.rows, ack=not inflight)
            online += prices.rows * prices.ingress
            # Conservative bound: failed in-flight reply can still arrive in full.
            returned += prices.rows * prices.reply
            if inflight:
                snapshots.append(asdict(ledger))
                interrupted = True
                break
            if point == "acknowledged" and stop == pc:
                interrupted = True
                break
        if not interrupted and end == len(nodes) and stop is None:
            ledger.claim(prices.epoch_rows)
            online += prices.epoch_rows * prices.ingress
            returned += prices.epoch_rows * prices.reply
        if not interrupted:
            control += prices.cut_control if end != len(nodes) else 0
        ledger.retire()
        if interrupted or stop == end:
            break
        if end != len(nodes):
            transitions += 1
        begin = end
    rows = ledger.audit()
    correction = ledger.issued * prices.correction
    total = prices.cold_public + correction + online + returned + control
    return {"cuts": cuts, "stop": stop, "point": point, "rows": rows,
            "material_byte_gauges": material_bytes(rows, prices),
            "inflight_observations": [{"rows": s, "bytes": material_bytes(s, prices)} for s in snapshots],
            "sessions": sessions, "pauses": transitions,
            "directed_body_upper_bounds": {"publisher_to_client": prices.cold_public,
                "preparation_to_inference": correction, "client_to_inference": online,
                "inference_to_client": returned, "aggregate_controls": control},
            "total_body_upper_bound": total, "cpu_seconds": None, "pause_seconds": None,
            "physical_wire_upper_bound": None}


def oracle():
    nodes = graph()
    rows = 4
    initial = {"input": tuple(f32(x / 8) for x in (-3, 2, 5, 7))}
    serial = dict(initial)
    execute(nodes, serial, 0, len(nodes))
    checks = []
    # Every graph frontier, not just proposed complete-layer cuts.
    for pc in range(1, len(nodes)):
        values = dict(initial)
        execute(nodes, values, 0, pc)
        state = export_state(nodes, values, pc, generation=pc)
        restored = restore_state(nodes, state, pc, rows, pc)
        execute(nodes, restored, pc, len(nodes))
        final = export_state(nodes, restored, len(nodes))
        assert final == export_state(nodes, serial, len(nodes))
        checks.append({"pc": pc, "live": live_ids(nodes, pc),
                       "payload_bytes": sum(len(bytes.fromhex(t["hex"])) for t in state["tensors"].values()),
                       "exact_toy_final_bits": True})
    return checks


def admission(*, source_digest, state_manifest, base_bytes, scratch_bytes, correction_bytes,
              export_copy=True):
    if len(source_digest) != 64 or any(c not in "0123456789abcdef" for c in source_digest):
        raise ValueError("exact source digest required")
    owners = (base_bytes, scratch_bytes, correction_bytes, *state_manifest.values())
    if any(type(n) is not int or n < 0 for n in owners):
        raise ValueError("invalid owned bound")
    live = sum(state_manifest.values())
    total = base_bytes + scratch_bytes + correction_bytes + live * (2 if export_copy else 1)
    return {"source_digest": source_digest, "live_hidden_and_partial_kv_bytes": live,
            "export_copy_bytes": live if export_copy else 0, "required_owned_bytes": total,
            "owned_bound_fits": total <= LOW, "whole_client_low_qualified": False,
            "rss_and_source_native_admission": "unknown; allocation bound alone insufficient"}


def report():
    control = simulate((10,))
    success = simulate((5, 10))
    canceled = simulate((5, 10), stop=5)
    # Same executed/acknowledged prefix and fixed output epoch allowance.
    whole_cancel = simulate((10,), stop=5, point="acknowledged")
    saved = whole_cancel["total_body_upper_bound"] - canceled["total_body_upper_bound"]
    penalty = success["total_body_upper_bound"] - control["total_body_upper_bound"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES + (
        "scripts/probe_prefill_cancellation_cuts.py", "tests/test_prefill_cancellation_cuts_probe.py",
        "docs/research/prefill-cancellation-cuts-reference.md")}
    return {"schema": "pllm.prefill_cancellation_cuts_screen.v1", "deterministic": True,
            "verdict": "current contract cannot express stage cuts; extension-only conditional cost win",
            "scope": "toy graph and explicitly supplied bounded bodies; not deployment or model execution",
            "source_and_own_file_sha256": hashes, "oracle": oracle(),
            "synthetic_prices": asdict(Prices()), "successful_whole_prefill_fixed_epoch": control,
            "successful_staged_fixed_epoch": success, "canceled_staged": canceled,
            "canceled_whole_prefill_fixed_epoch": whole_cancel,
            "bounded_cancellation_traces": [simulate((5, 10), stop=pc, point=point)
                for point, pcs in (("preparing", (5, 10)), ("sealed", (5, 10)),
                                   ("online_inflight", (1, 3, 4, 6, 8, 9)), ("cut", (5, 10)))
                for pc in pcs],
            "break_even": {"canceled_body_saving": saved, "successful_body_penalty": penalty,
                "cancellation_fraction_strictly_greater_than": f"{penalty}/{penalty + saved}",
                "equation": "p*S > (1-p)*H; identical common cold/online bytes remain in denominator",
                "deployment_cancellation_fraction": None},
            "admission_example": admission(source_digest=hashes[SOURCES[3]],
                state_manifest={"residual": 64, "partial_kv": 128}, base_bytes=192 << 20,
                scratch_bytes=16 << 20, correction_bytes=1 << 20),
            "remote_body_mac_fraction": {"toy": 1.0, "source_prefill": None, "source_decode": None},
            "measured_latency_cpu_rss": None, "sdk_promotion": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(report(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
