"""Bounded, offline public gate/up ownership algebra and historical cost screen.

No serving execution, downloads, checkpoint loading, or RSS qualification. The
native compiler is inspected with configuration only. Existing integer/scaling
contracts are exercised on tiny synthetic matrices with an independent oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "docs/evidence/research-all-link-frontier.json"
GEOMETRY = ROOT / "docs/evidence/network-algebra-qwen25-2026-10-06.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_rows(channels, selected):
    selected = tuple(selected)
    if channels < 1 or any(type(i) is not int or not 0 <= i < channels for i in selected):
        raise ValueError("invalid channel selection")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate channel")
    return np.array((*selected, *(channels + i for i in selected)), dtype=np.int64)


def split_integer(x, weight, selected):
    """Integer row partition, not input partition; no quantizer is called here."""
    if x.dtype != np.int8 or weight.dtype != np.int8 or weight.ndim != 2:
        raise ValueError("existing i8 tensors required")
    if weight.shape[0] % 2 or x.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("incompatible paired matrix")
    local = paired_rows(weight.shape[0] // 2, selected)
    remote = np.array([i for i in range(weight.shape[0]) if i not in set(local)], dtype=np.int64)
    # int64 is the oracle accumulator, not a proposed production precision change.
    output = np.empty((len(x), len(weight)), dtype=np.int64)
    for indices in (local, remote):
        output[:, indices] = x.astype(np.int64) @ weight[indices].astype(np.int64).T
    return output


def masked_remote_integer(x, weight, r, s, bits):
    """Tiny independent ring algebra only; no production privacy implementation."""
    if bits not in (16, 24, 32):
        raise ValueError("unsupported ring")
    modulus = 1 << bits
    x, weight, r, s = (np.asarray(v, dtype=np.int64) for v in (x, weight, r, s))
    masked = (x - r) % modulus
    correction = (r @ weight.T - s) % modulus
    restored = (masked @ weight.T + correction + s) % modulus
    return np.where(restored >= modulus // 2, restored - modulus, restored)


def amortization(c0, recurring, bootstrap, saved, overhead, target):
    """C0 + R*T0 versus C0 + B + R*(T0-S+H), both equally warm."""
    if min(c0, recurring, bootstrap, saved, overhead) < 0 or not 0 < target < 1:
        raise ValueError("invalid cost")
    margin = saved - overhead - target * recurring
    if margin <= 0:
        return None
    return max(1, math.ceil((bootstrap + target * c0) / margin))


def compile_geometry(config, pipeline, control, geometry, input_rows=150):
    from pllm import lower_model
    from pllm.runtime.semantic_stages import scheduled_stage_specs, semantic_stage_role

    if not geometry["complete"] or geometry["source"]["body_fingerprint"] != control["body_fingerprint"]:
        raise ValueError("geometry body identity mismatch")
    if sum(control["classes"].values()) != control["pair_bytes"]:
        raise ValueError("control ledger does not conserve")
    plan = lower_model(config, batch=1, max_input_tokens=input_rows, max_new_tokens=8)
    schedule = plan.runtime_schedule(pipeline).to_dict()
    operations = {op["id"]: op for op in plan.prefill["operations"]}
    steps = {(s["layer"], semantic_stage_role(s, operations)): s
             for s in schedule["prefill"]["steps"] if s["weight_ids"] and s["layer"] is not None}
    measured = {s["stage"]: s for s in geometry["stages"]}
    items = []
    for stage in scheduled_stage_specs(plan, pipeline):
        if stage.layer_index is None:
            continue
        if [stage.out_features, stage.in_features] != measured[stage.id]["shape"]:
            raise ValueError("compiled shape differs from locked geometry")
        step = steps[stage.layer_index, stage.role]
        if step["executor"] != "remote_stage":
            raise ValueError("baseline must not already own body groups")
        rows = 1 if step.get("terminal_row_only", False) else input_rows
        traffic = control["stages"][stage.id]
        items.append({"id": stage.id, "layer": stage.layer_index, "role": stage.role,
                      "rows": stage.out_features, "columns": stage.in_features,
                      "prefill_rows": rows, "decode_macs": stage.out_features * stage.in_features,
                      "prefill_macs": rows * stage.out_features * stage.in_features,
                      "weight_bytes": stage.out_features * stage.in_features,
                      "scale_bytes": 4 * stage.out_features,
                      "output_correction_bytes": sum(traffic.get(k, 0) for k in
                          ("preparation->inference", "inference->client")),
                      "ingress_bytes": traffic.get("client->inference", 0),
                      "weight_sha256": measured[stage.id]["weight_sha256"]})
    if {i["id"] for i in items} != set(control["stages"]) or set(measured) != set(control["stages"]):
        raise ValueError("stage sets differ")
    return items, schedule


def price_selection(items, counts, *, loss=0.01, frame_per_row=64):
    totals = {key: sum(i[key] for i in items) for key in ("prefill_macs", "decode_macs")}
    selected = []
    weight = scales = maps = frames = gross = ingress = overhead = 0
    local_prefill = local_decode = 0
    for item in items:
        count = counts.get(item["id"], 0)
        channels = item["rows"] // 2
        if not count:
            continue
        if item["role"] != "mlp_gate_up" or not 0 <= count <= channels:
            raise ValueError("only complete gate/up pairs may be selected")
        fraction = count / channels
        local_decode += 2 * count * item["columns"]
        local_prefill += 2 * count * item["columns"] * item["prefill_rows"]
        w, s = 2 * count * item["columns"], 8 * count
        weight += w
        scales += s
        maps += 4 * count  # Explicit uint32 channels; contiguous-range maps can be smaller.
        frames += 128 + 64 * math.ceil((w + s) / (1 << 20))
        gross += fraction * item["output_correction_bytes"]
        if count == channels:
            ingress += item["ingress_bytes"]
        else:
            # Conservative incremental framing sensitivity per executed row.
            overhead += frame_per_row * (item["prefill_rows"] + 14)
        selected.append({"id": item["id"], "paired_channels": count,
                         "channel_rule": "public contiguous prefix [0,count)",
                         "fraction": fraction})
    return {"selected": selected, "weight_bytes": weight, "scale_bytes": scales,
            "map_bytes": maps, "artifact_header_auth_bytes": frames,
            "bootstrap_bytes": weight + scales + maps + frames,
            "gross_correction_reply_bytes_per_pair": gross,
            "removed_ingress_bytes_per_pair": ingress,
            "retained_ingress_bytes_per_pair": sum(i["ingress_bytes"] for i in items) - ingress,
            "framing_loss_bytes_per_pair": gross * loss,
            "incremental_frame_bytes_per_pair": overhead,
            "saved_bytes_per_pair": gross * (1 - loss) + ingress,
            "prefill_remote_fraction": 1 - local_prefill / totals["prefill_macs"],
            "decode_remote_fraction": 1 - local_decode / totals["decode_macs"]}


def choose_pairs(items, remote_minimum, horizon):
    """Exact bounded prefix-count search for this equal-input-width geometry.

    All ordinary stages cost P times decode; one terminal gate/up stage costs 1
    times decode. Enumerate its count and greedily fill other equal-cost channels.
    This maximizes raw linear byte benefit before discrete frame/auth overheads;
    those overheads are charged afterward, not claimed jointly optimal.
    """
    gates = [i for i in items if i["role"] == "mlp_gate_up"]
    if len({i["columns"] for i in gates}) != 1:
        raise ValueError("screen requires equal input widths")
    terminals = [i for i in gates if i["prefill_rows"] == 1]
    regular = [i for i in gates if i["prefill_rows"] != 1]
    if len(terminals) != 1 or len({i["prefill_rows"] for i in regular}) != 1:
        raise ValueError("screen requires one terminal expansion")
    step = 2 * gates[0]["columns"]
    p = regular[0]["prefill_rows"]
    dp = math.floor((1 - remote_minimum + 1e-12) * sum(i["decode_macs"] for i in items) / step)
    pp = math.floor((1 - remote_minimum + 1e-12) * sum(i["prefill_macs"] for i in items) / step)
    regular.sort(key=lambda i: (-i["output_correction_bytes"] / (i["rows"] // 2), i["id"]))
    prefix = [0.0]
    owners = []
    for item in regular:
        value = horizon * item["output_correction_bytes"] / (item["rows"] // 2) - 2 * (item["columns"] + 4)
        if value <= 0:
            continue
        for _ in range(item["rows"] // 2):
            prefix.append(prefix[-1] + value)
            owners.append(item["id"])
    terminal = terminals[0]
    value = horizon * terminal["output_correction_bytes"] / (terminal["rows"] // 2) - 2 * (terminal["columns"] + 4)
    best = (0.0, 0, 0)
    for n in range(min(dp, terminal["rows"] // 2) + 1):
        regular_count = min(len(owners), dp - n, (pp - n) // p)
        if regular_count < 0:
            continue
        candidate = (prefix[regular_count] + n * value, n, regular_count)
        if candidate[0] > best[0]:
            best = candidate
    counts = {}
    if best[1]:
        counts[terminal["id"]] = best[1]
    for owner in owners[:best[2]]:
        counts[owner] = counts.get(owner, 0) + 1
    return counts


def whole_groups(items, remote_minimum, horizon):
    """Exact same-budget whole-group comparator, exploiting terminal geometry."""
    unit = math.gcd(*(i["decode_macs"] for i in items))
    decode_budget = math.floor((1 - remote_minimum + 1e-12) * sum(i["decode_macs"] for i in items))
    prefill_budget = math.floor((1 - remote_minimum + 1e-12) * sum(i["prefill_macs"] for i in items))
    terminal = [i for i in items if i["prefill_rows"] == 1]
    regular = [i for i in items if i["prefill_rows"] != 1]
    if len(terminal) > 12 or len({i["prefill_rows"] for i in regular}) != 1:
        raise ValueError("unsupported whole-group terminal geometry")
    p = regular[0]["prefill_rows"]
    states = {0: (0, ())}
    for item in regular:
        cost = item["decode_macs"] // unit
        value = horizon * (item["output_correction_bytes"] + item["ingress_bytes"]) - item["weight_bytes"] - item["scale_bytes"]
        for used, (saved, chosen) in list(states.items()):
            nxt = used + cost
            if nxt * unit <= decode_budget and (saved + value) > states.get(nxt, (-1, ()))[0]:
                states[nxt] = saved + value, (*chosen, item["id"])
    best = (0, ())
    for mask in itertools.product((False, True), repeat=len(terminal)):
        subset = [i for i, yes in zip(terminal, mask, strict=True) if yes]
        dm = sum(i["decode_macs"] for i in subset)
        value = sum(horizon * (i["output_correction_bytes"] + i["ingress_bytes"]) - i["weight_bytes"] - i["scale_bytes"] for i in subset)
        for used, (saved, chosen) in states.items():
            if used * unit + dm <= decode_budget and used * unit * p + dm <= prefill_budget and saved + value > best[0]:
                best = saved + value, (*chosen, *(i["id"] for i in subset))
    selected = [i for i in items if i["id"] in best[1]]
    return {"selected": list(best[1]), "bootstrap_raw_payload_bytes": sum(i["weight_bytes"] + i["scale_bytes"] for i in selected),
            "saved_bytes_per_pair": sum(i["output_correction_bytes"] + i["ingress_bytes"] for i in selected),
            "net_bytes_at_horizon": best[0], "scope": "optimistic raw whole-group projection; no added manifests/framing",
            "prefill_remote_fraction": 1 - sum(i["prefill_macs"] for i in selected) / sum(i["prefill_macs"] for i in items),
            "decode_remote_fraction": 1 - sum(i["decode_macs"] for i in selected) / sum(i["decode_macs"] for i in items)}


def add_horizons(row, control):
    scenarios = []
    for recurring_common in (0, 390_000):
        c0 = control["classes"]["public_bundle"] - recurring_common
        t0 = control["pair_bytes"] - c0
        for bootstrap_multiplier in (0.8, 1.0, 1.1):
            b = row["bootstrap_bytes"] * bootstrap_multiplier
            s, h = row["saved_bytes_per_pair"], row["incremental_frame_bytes_per_pair"]
            scenarios.append({"recurring_common_bytes_per_pair": recurring_common,
                              "bootstrap_multiplier": bootstrap_multiplier,
                              "C0": c0, "T0": t0, "B": b, "S": s, "H": h,
                              "asymptotic_fraction": (s - h) / t0,
                              "break_even_pairs": {str(t): amortization(c0, t0, b, s, h, t) for t in (0.2, 0.35, 0.5)},
                              "horizons": [{"pairs": r, "control_bytes": c0 + r * t0,
                                            "candidate_bytes": c0 + b + r * (t0 - s + h),
                                            "fractional_saving": (r * (s - h) - b) / (c0 + r * t0)}
                                           for r in (1, 4, 5, 16, 64)]})
    row["amortization_sensitivities"] = scenarios
    return row


def tiny_algebra():
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row

    rng = np.random.default_rng(612)
    weight = rng.integers(-128, 128, size=(22, 17), dtype=np.int8)
    activation = quantize_activation_per_row(rng.normal(size=(3, 17)).astype(np.float32), bits=8)
    scales = np.linspace(0.001, 0.021, 22, dtype=np.float32)
    reference = np.array([[sum(int(a) * int(b) for a, b in zip(x, w, strict=True))
                           for w in weight] for x in activation.values], dtype=np.int64)
    cases = [(), (0,), (1, 3, 5, 9), tuple(range(11)), tuple(reversed(range(11)))]
    for selected in cases:
        split = split_integer(activation.values, weight, selected)
        if not np.array_equal(split, reference):
            raise AssertionError("integer split changed output")
        remote = np.array([i for i in range(len(weight)) if i not in set(paired_rows(11, selected))], dtype=np.int64)
        r = rng.integers(0, 1 << 24, size=activation.values.shape, dtype=np.int64)
        s = rng.integers(0, 1 << 24, size=(3, len(remote)), dtype=np.int64)
        if not np.array_equal(masked_remote_integer(activation.values, weight[remote], r, s, 24), reference[:, remote]):
            raise AssertionError("fresh masked remote lanes changed integers")
        left = dequantize_matmul(split, activation.scales, scales)
        right = dequantize_matmul(reference, activation.scales, scales)
        for value in (left, right):
            gate, up = np.split(value, 2, axis=1)
            result = (gate / (1 + np.exp(-gate))) * up
            if value is left:
                candidate = result
            elif candidate.tobytes() != result.tobytes():
                raise AssertionError("post-stitch SiLU/multiply changed bits")
    return {"scope": "synthetic 3x17 input, 22x17 i8 weight; independent Python integer oracle",
            "partitions_checked": len(cases), "integer_dequant_silu_multiply_bit_identity": True,
            "fresh_masked_remote_wrap24_identity": True,
            "input_sha256": hashlib.sha256(activation.values.tobytes()).hexdigest(),
            "weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
            "full_logits_KV_parity": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="existing public config.json; no checkpoint tensors read")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "target/prepared-frontier-artifacts/indexed.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from pllm import Experiment
    from pllm.runtime._native_support import extension

    prior = json.loads(PRIOR.read_text())
    if not prior["complete"]:
        raise ValueError("incomplete historical control")
    control = prior["control"]
    config = json.loads(args.config.read_text())
    pipeline = Experiment.from_spec(json.loads(args.pipeline.read_text())).pipeline
    paths = [Path(__file__), ROOT / "tests/test_bootstrap_placement_probe.py", PRIOR, GEOMETRY,
             args.config, args.pipeline, ROOT / "python/pllm/runtime/semantic_stages.py",
             ROOT / "python/pllm/runtime/quantization.py", ROOT / "python/pllm/roles/__init__.py",
             ROOT / "crates/pllm-compiler/src/decoder_runtime_schedule.rs",
             ROOT / "crates/pllm-compiler/src/mlp_tiling.rs", Path(extension().__file__)]
    frozen = {str(p): digest(p) for p in paths}
    items, schedule = compile_geometry(config, pipeline, control, json.loads(GEOMETRY.read_text()))
    gates = [i for i in items if i["role"] == "mlp_gate_up"]
    uniform = [add_horizons(price_selection(items, {i["id"]: math.floor(i["rows"] // 2 * f) for i in gates}), control)
               | {"requested_uniform_channel_fraction": f} for f in (0.1, 0.17, 0.25, 0.34, 0.5, 0.75, 1.0)]
    frontier = []
    for minimum in (0.9, 0.8, 0.7):
        for horizon in (1, 4, 16, 64):
            row = add_horizons(price_selection(items, choose_pairs(items, minimum, horizon)), control)
            if min(row["prefill_remote_fraction"], row["decode_remote_fraction"]) + 1e-12 < minimum:
                raise AssertionError("separate phase budget failed")
            comparator = whole_groups(items, minimum, horizon)
            row.update(remote_minimum=minimum, selection_horizon_pairs=horizon,
                       whole_group_comparator=comparator,
                       net_at_selection_horizon=horizon * (row["saved_bytes_per_pair"] - row["incremental_frame_bytes_per_pair"]) - row["bootstrap_bytes"],
                       comparable_raw_partial_net=horizon * (row["gross_correction_reply_bytes_per_pair"] + row["removed_ingress_bytes_per_pair"]) - row["weight_bytes"] - row["scale_bytes"])
            frontier.append(row)
    result = {"schema": "pllm.bootstrap_placement_screen.v1", "complete": True,
              "scope": "offline algebra/config compiler inspection and historical pair cost projections, not serving",
              "source_identity": frozen, "control": control,
              "compiler_schedule_sha256": hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest(),
              "geometry": items, "uniform_frontier": uniform, "remote_budget_frontier": frontier,
              "algebra": tiny_algebra(),
              "assumptions": ["150 input + 8 generated tokens cold/exact-repeat; 14 decode executions across pair, not 16",
                  "each new pair uses distinct prefix then exact RAM repeat; cache shared across all pairs in both arms",
                  "body MAC denominator excludes head, attention/nonlinear CPU; prefill terminal schedule priced separately",
                  "historical correction/reply bodies prorated by selected lanes; exact arithmetic/frame attribution unknown",
                  "1% body framing loss plus 64 incremental bytes per executed row of a split group are sensitivities, not measured",
                  "raw i8/float32 scales; explicit uint32 channel maps; 128-byte stage descriptor and 64-byte 1MiB page auth/header allowance",
                  "bootstrap x0.8/x1/x1.1 is hypothetical compression/framing sensitivity, no public weight codec measured",
                  "public_bundle may contain recurring manifests: 0 or 390000 bytes/pair sensitivity; attribution unknown",
                  "fresh private masks for every remaining remote lane; public row ownership is static, not activation-adaptive"],
              "memory": {"whole_client_limit_bytes": 256 << 20, "qualified": False,
                  "page_payload_cap_bytes": 1 << 20, "hypothetical_two_page_buffers_bytes": 2 << 20,
                  "integer_stitch_workspace_16_rows_bytes": 16 * max(i["rows"] for i in gates) * 4,
                  "native_weight_snapshot_quantization_workspace": "unknown; may defeat page bounds",
                  "disk": "selected weight+scale+map artifacts, plus common cache and authentication metadata",
                  "OS_file_cache": "additional owner, not zero and not equated to whole-client RSS",
                  "RSS": "not measured; existing base, KV, activations, runtime, Python/native transients remain charged"},
              "next_gate": "model-neutral paired-output subset compiler/provider/cache binding; bounded generated decoder complete logits/KV parity, then measured fresh-client admission and all-link cohort"}
    if frozen != {str(p): digest(p) for p in paths}:
        raise RuntimeError("source changed during probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation retains immutable evidence; reruns require a new path.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
