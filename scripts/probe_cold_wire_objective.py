"""Public synthetic, bounded integer training against prepared protocol byte cliffs.

No checkpoint, provider, timing cohort, private calibration, or SDK mutation.
Independent integer codec/size oracle is checked against installed serializers.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def row_width(l1: int, activation_bound: int = 127) -> int:
    if l1 < 0 or not 1 <= activation_bound <= 127:
        raise ValueError("invalid public bound")
    width = max(1, (int(l1) * activation_bound).bit_length() + 1)
    if width > 32:
        raise ValueError("signed wrap32 exceeded")
    return width


def l1_cap(width: int, activation_bound: int = 127) -> int:
    if not 1 <= width <= 32 or not 1 <= activation_bound <= 127:
        raise ValueError("invalid public width/bound")
    return ((1 << (width - 1)) - 1) // activation_bound


def widths_for(q, activation_bound=127):
    return bytes(row_width(int(v), activation_bound)
                 for v in np.abs(np.asarray(q, dtype=np.int64)).sum(axis=1))


def pack_integer(values, widths):
    """Independent LSB-first row-major oracle; modulo public row rings."""
    if not widths or any(not 1 <= w <= 32 for w in widths):
        raise ValueError("invalid widths")
    flat = np.asarray(values).reshape(-1)
    if len(flat) % len(widths):
        raise ValueError("shape mismatch")
    result = bytearray()
    buffer = used = 0
    for i, value in enumerate(flat):
        width = widths[i % len(widths)]
        buffer |= (int(value) & ((1 << width) - 1)) << used
        used += width
        while used >= 8:
            result.append(buffer & 255)
            buffer >>= 8
            used -= 8
    if used:
        result.append(buffer)
    return bytes(result)


def unpack_integer(data, widths, rows):
    bits = rows * sum(widths)
    if len(data) != (bits + 7) // 8 or (bits % 8 and data[-1] >> (bits % 8)):
        raise ValueError("length or padding mismatch")
    number = int.from_bytes(data, "little")
    out = []
    for width in list(widths) * rows:
        value = number & ((1 << width) - 1)
        out.append(value - (1 << width) if value >= 1 << (width - 1) else value)
        number >>= width
    return np.asarray(out, dtype=np.int64).reshape(rows, len(widths))


def layout_record(widths, binary=False):
    encoded = base64.b64encode(zlib.compress(widths, 1))
    return {"schema": "pllm.residue_layout.v1", "count": len(widths),
            "sha256": hashlib.sha256(widths).hexdigest(),
            "zlib_base64": encoded if binary else encoded.decode("ascii")}


def mp(value):
    return msgpack.packb(value, use_bin_type=True)


def bin_size(n):
    return n + (2 if n <= 255 else 3 if n <= 65535 else 5)


def framed_sizes(columns, widths, rows, bound, *, wire_bits=None, stage="public.linear", activation_bits=8):
    """Exact specified serializer bodies, sizes only: no large payload allocation.

    Identifiers/durations frozen here. Real envelopes/ACKs/registration excluded.
    Correction includes compressed inline public layout, not just tensor bytes.
    """
    wire_bits = wire_bits or next(b for b in (16, 24, 32) if bound < 1 << (b - 1))
    ticket = "01" * 16
    input_n = (rows * columns * max(widths) + 7) // 8
    output_n = (rows * sum(widths) + 7) // 8
    # MessagePack size difference replaces empty binary field with actual bytes.
    request = len(b"PLLMPSB3") + 4 + len(mp([
        bytes.fromhex(ticket), bytes(16 * rows), columns, wire_bits, max(widths), b""
    ])) - bin_size(0) + bin_size(input_n)
    response = 8 + len(mp([ticket, stage, rows, wire_bits,
        hashlib.sha256(widths).digest(), 0, b""])) - bin_size(0) + bin_size(output_n)
    correction = {"v": 3, "a": ticket, "h": "public-session", "m": "synthetic-wire-v1",
        "b": "b" * 64, "t": stage, "w": "c" * 64, "r": rows, "n": columns,
        "o": len(widths), "wb": 8, "ab": activation_bits, "q": bound,
        "k": f"u{wire_bits}", "p": 1 << wire_bits, "x": wire_bits,
        "e": 0, "d": b"", "y": layout_record(widths, binary=True)}
    correction_n = len(mp(correction)) - bin_size(0) + bin_size(output_n)
    prep_n = len(mp({"v": 3, "a": bytes.fromhex(ticket), "h": "public-session",
                    "r": rows, "z": bytes(32)}))
    return {"client_to_preparation": prep_n, "preparation_to_inference": correction_n,
            "client_to_inference": request, "inference_to_client": response,
            "input_tensor_bytes": input_n, "output_tensor_bytes": output_n,
            "covered_stage_bodies": prep_n + correction_n + request + response}


def train_budget(q0, x, target, cap, passes=2):
    """Greedy integer constrained distillation; exact SSE marginal, no STE.

    First shrink to public row budget; then bounded integer coordinate descent.
    Training only sees public train rows. Held-out data never enters selection.
    """
    q = np.asarray(q0, dtype=np.int64).copy()
    x = np.asarray(x, dtype=np.int64)
    if cap < 0 or q.ndim != 2 or x.ndim != 2 or q.shape[1] != x.shape[1] or np.max(np.abs(q)) > 127:
        raise ValueError("invalid bounded training inputs")
    norms = (x * x).sum(axis=0)
    edits = 0
    for row in range(len(q)):
        residual = x @ q[row] - target[:, row]
        while int(np.abs(q[row]).sum()) > cap:
            delta = -np.sign(q[row])
            costs = 2 * delta * (x.T @ residual) + norms * delta * delta
            costs[delta == 0] = np.iinfo(np.int64).max
            index = int(np.argmin(costs))
            q[row, index] += delta[index]
            residual += x[:, index] * delta[index]
            edits += 1
        for _ in range(passes):
            for index in range(q.shape[1]):
                best_cost, best_delta = 0, 0
                for delta in (-1, 1):
                    value = q[row, index] + delta
                    new_l1 = int(np.abs(q[row]).sum()) - abs(q[row, index]) + abs(value)
                    if abs(value) > 127 or new_l1 > cap:
                        continue
                    cost = int(2 * delta * (x[:, index] @ residual) + norms[index])
                    if cost < best_cost:
                        best_cost, best_delta = cost, delta
                if best_delta:
                    q[row, index] += best_delta
                    residual += x[:, index] * best_delta
                    edits += 1
    return q, edits


def quality(prediction, target):
    error = prediction.astype(np.float64) - target
    return {"relative_mse": float(np.sum(error * error) / max(1, np.sum(target.astype(float) ** 2))),
            "max_absolute_integer_teacher_error": float(np.max(np.abs(error))),
            "argmax_agreement": float(np.mean(np.argmax(prediction, axis=1) == np.argmax(target, axis=1)))}


def check_installed(q, x):
    """Checks installed actual serializers, native layouts and modular correction."""
    from pllm import _native
    from pllm.runtime.preparation_protocol import CorrectionPush, PreparationRequest
    from pllm.runtime.stage_protocol import PreparedStageBatchRequest, MaskedStageResponse
    from pllm.runtime.residue_codec import encode_layout, pack_row_response

    widths = widths_for(q)
    columns = q.shape[1]
    bound = int(np.abs(q).sum(axis=1).max()) * 127
    bits = next(b for b in (16, 24, 32) if bound < 1 << (b - 1))
    raw = q.astype(np.int8).tobytes()
    native_widths = _native.offset_row_bits(raw, columns, 127)
    assert native_widths == widths
    assert encode_layout(widths) == layout_record(widths)
    comparisons = []
    for rows in (1, 3, 16):
        tickets = tuple(f"{i + 1:032x}" for i in range(rows))
        # Public deterministic masks only in this oracle, never production randomness.
        r = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns) * 7919
        s = np.arange(rows * len(q), dtype=np.int64).reshape(rows, len(q)) * 104729
        activation = np.resize(x[:1], (rows, columns))
        masked = activation - r
        correction = r @ q.T - s
        product = masked @ q.T
        packed = pack_integer(product + correction, widths)
        centered = unpack_integer(pack_integer(activation @ q.T, widths), widths, rows)
        assert np.array_equal(centered, activation @ q.T)
        # Add s before centered reconstruction: sum modulo each public row ring.
        residue_sum = (product + correction) + s
        assert np.array_equal(unpack_integer(pack_integer(residue_sum, widths), widths, rows), activation @ q.T)
        assert _native.offset_pack_rows((product + correction).astype("<u4").tobytes(), widths, rows) == packed
        input_packed = pack_integer(masked, bytes([max(widths)]) * columns)
        input_residues = unpack_integer(input_packed, bytes([max(widths)]) * columns, rows)
        assert pack_integer(input_residues @ q.T + correction + s, widths) == pack_integer(activation @ q.T, widths)
        reconstructed = _native.offset_reconstruct_rows(packed, pack_integer(s, widths), widths, rows)
        assert np.array_equal(np.frombuffer(reconstructed, "<i8").reshape(rows, len(q)), activation @ q.T)
        request = PreparedStageBatchRequest("01" * 16, tickets, masked.astype(np.uint32), bits, max(widths)).pack()
        request_oracle = b"PLLMPSB3" + struct.pack(">I", rows) + mp([
            bytes.fromhex("01" * 16), b"".join(bytes.fromhex(t) for t in tickets),
            columns, bits, max(widths), input_packed])
        assert request == request_oracle
        common = dict(attempt_id="01" * 16, session_id="public-session", model="synthetic-wire-v1",
            body_fingerprint="b" * 64, stage_id="public.linear", weight_digest="c" * 64,
            rows=rows, in_features=columns, out_features=len(q), weight_bits=8, activation_bits=8,
            signed_output_bound=bound, ring=f"u{bits}", modulus=1 << bits, wire_bits=bits)
        push = CorrectionPush(**common, correction=correction.astype(np.uint32), output_residue_bits=widths).pack()
        prep = PreparationRequest(**common, seed=bytes(32)).pack()
        response = MaskedStageResponse(correlation_id="01" * 16, stage_id="public.linear",
            masked_output=(product + correction).astype(np.uint32), wire_bits=bits, server_ns=0,
            modulus=1 << bits)
        response_body = pack_row_response(response, widths, namespace="prepared")
        sizes = framed_sizes(columns, widths, rows, bound)
        assert [len(prep), len(push), len(request), len(response_body)] == [sizes[k] for k in (
            "client_to_preparation", "preparation_to_inference", "client_to_inference", "inference_to_client")]
        comparisons.append({"rows": rows, "native_widths_and_packing": True,
                            "packed_input_matmul_and_native_unmask_exact": True,
                            "exact_request_bytes": True, "all_four_body_sizes": True})
    return comparisons


def geometry():
    """Hypothetical Qwen2.5-0.5B-shaped body; not a checkpoint observation."""
    shapes = [(896, 896), (896, 128), (896, 128), (896, 896),
              (896, 4864), (896, 4864), (4864, 896)] * 24
    parameters = sum(n * o for n, o in shapes)
    output_rows = sum(o for _, o in shapes)
    records = []
    # Terminal pruning assumed: 39 prompt rows and 7 one-row decode steps for 8 outputs.
    # Uniform widths deliberately hypothetical; parent owns real weight scan.
    for width in (24, 23, 22, 20, 16):
        input_tensor = correction_tensor = response_tensor = frames = prep = 0
        for i, (n, o) in enumerate(shapes):
            widths = bytes([width]) * o
            bound = (1 << (width - 1)) - 1
            # Preparation issued once per stage for all 46 consumed rows.
            p = framed_sizes(n, widths, 46, bound, stage=f"layer.stage.{i}")
            correction_tensor += p["output_tensor_bytes"]
            prep += p["client_to_preparation"] + p["preparation_to_inference"]
            for rows in (16, 16, 7, 1, 1, 1, 1, 1, 1, 1):
                f = framed_sizes(n, widths, rows, bound, stage=f"layer.stage.{i}")
                input_tensor += f["input_tensor_bytes"]
                response_tensor += f["output_tensor_bytes"]
                frames += f["client_to_inference"] + f["inference_to_client"]
        delivery = {"publisher_to_inference_raw_signed_i8_weights": parameters,
                    "publisher_to_preparation_raw_signed_i8_weights": parameters,
                    "publisher_to_client_fp32_weight_scales": 4 * output_rows,
                    "publisher_to_client_axis_scales": 0,
                    "new_training_checkpoint_inference_delivery_extra": 0}
        records.append({"uniform_width": width, "delivery": delivery,
            "preparation_two_directed_bodies": prep, "online_two_directed_bodies": frames,
            "input_tensor_bytes": input_tensor, "correction_tensor_bytes": correction_tensor,
            "response_tensor_bytes": response_tensor,
            "covered_hypothetical_total": sum(delivery.values()) + prep + frames,
            "full_request_bytes": None})
    return {"scope": "hypothetical body only, 24 layers, h896/i4864, GQA k/v128; 39+7 rows; lm_head/client embeddings excluded",
            "body_parameters": parameters, "output_scale_count": output_rows,
            "training_fp32_master_plus_gradient_plus_two_adam_slots_bytes": 16 * parameters,
            "fp32_publisher_checkpoint_bytes": 4 * parameters,
            "records": records}


def run():
    rng = np.random.default_rng(6103)
    teacher = rng.integers(-48, 49, size=(8, 32), dtype=np.int64)
    teacher[:, 0] = 127  # Max-normalized W8 baseline, useful blocker/outlier.
    train = rng.integers(-127, 128, size=(128, 32), dtype=np.int64)
    heldout = rng.integers(-127, 128, size=(96, 32), dtype=np.int64)
    adverse = np.stack([127 * np.sign(row) for row in teacher])
    targets = [x @ teacher.T for x in (train, heldout, adverse)]
    base_width = max(widths_for(teacher))
    candidates = []

    def record(name, q, scale=1, edits=0, activation_bound=127):
        widths = widths_for(q, activation_bound)
        from pllm import _native
        assert _native.offset_row_bits(q.astype(np.int8).tobytes(), q.shape[1], activation_bound) == widths
        bound = int(np.abs(q).sum(axis=1).max()) * activation_bound
        sizes = framed_sizes(32, widths, 16, bound, activation_bits=4 if activation_bound == 7 else 8)
        metrics = []
        for x, target in zip((train, heldout, adverse), targets):
            if activation_bound != 127:
                x = np.rint(x * activation_bound / 127).astype(np.int64)
                prediction = (x @ q.T) * (127 / activation_bound) * scale
            else:
                prediction = (x @ q.T) * scale
            metrics.append(quality(prediction, target))
        candidates.append({"name": name, "public_activation_bound": activation_bound,
            "integer_weights_sha256": hashlib.sha256(q.astype(np.int8).tobytes()).hexdigest(),
            "fixed_weight_scale": scale, "widths": list(widths), "max_input_width": max(widths),
            "row_l1": np.abs(q).sum(axis=1).tolist(), "training_integer_edits": edits,
            "train": metrics[0], "heldout": metrics[1], "adverse_sign_aligned": metrics[2],
            "stage_16_rows": sizes,
            "delivery_raw_weight_bytes_per_provider": q.size,
            "delivery_fp32_weight_scales_client_bytes": 4 * len(q),
            "delivery_axis_scales_bytes": 0,
            "cold_known_raw_payload_plus_stage_bodies": sizes["covered_stage_bodies"] + 2 * q.size + 4 * len(q),
            "publisher_new_fp32_checkpoint_bytes": 4 * q.size,
            "whole_request_bytes": None})

    record("unchanged_W8A8", teacher)
    for drop in (1, 2):
        cap = l1_cap(base_width - drop)
        trained, edits = train_budget(teacher, train, targets[0], cap)
        record(f"integer_trained_width_minus_{drop}", trained, edits=edits)
        naive = np.rint(teacher * min(1, cap / int(np.abs(teacher).sum(axis=1).max()))).astype(np.int64)
        record(f"naive_shrink_width_minus_{drop}", naive)
    for factor in (2, 4):
        coarse = np.rint(teacher / factor).astype(np.int64)
        record(f"coarser_fixed_scale_{factor}", coarse, scale=factor)
        trained, edits = train_budget(coarse, train * factor, targets[0], l1_cap(base_width - int(np.log2(factor))))
        record(f"coarser_scale_{factor}_integer_trained", trained, scale=factor, edits=edits)
    record("public_A4_changed_numeric_identity", teacher, activation_bound=7)
    # Positive scalar shrink then max-normalized quantizer: normalized integers unchanged.
    f = teacher.astype(np.float64)
    requant = np.rint((f * .25) / (np.max(np.abs(f * .25), axis=1)[:, None] / 127)).astype(np.int64)
    assert np.array_equal(requant, teacher)
    thresholds = [{"width": b, "activation_bound": a, "row_l1_cap": l1_cap(b, a),
        "cap_width": row_width(l1_cap(b, a), a),
        "next_l1_width": row_width(l1_cap(b, a) + 1, a)}
        for a in (7, 127) for b in (16, 20, 21, 22, 23, 24)]
    installed = check_installed(teacher, train)
    even = 2 * rng.integers(-63, 64, size=(8, 32), dtype=np.int64)
    divided = even // 2
    assert np.array_equal(train @ even.T, 2 * (train @ divided.T))
    exact_reparameterization = {"scope": "constructed divisible integer weights only; not Qwen",
        "before_widths": list(widths_for(even)), "after_widths": list(widths_for(divided)),
        "before_stage": framed_sizes(32, widths_for(even), 16, int(np.abs(even).sum(axis=1).max()) * 127),
        "after_stage": framed_sizes(32, widths_for(divided), 16, int(np.abs(divided).sum(axis=1).max()) * 127),
        "integer_product_times_public_scale_exact": True,
        "extra_axis_scale_bytes": 0, "row_scale_bytes_unchanged": 32,
        "importer_must_preserve_fixed_quantized_values_and_scales": True}
    paths = ["scripts/probe_cold_wire_objective.py", "crates/pllm-core/src/offset_transport.rs",
             "python/pllm/runtime/stage_protocol.py", "python/pllm/runtime/preparation_protocol.py",
             "python/pllm/runtime/residue_codec.py", "python/pllm/runtime/quantization.py"]
    import pllm.runtime.stage_protocol as installed_stage
    import pllm.runtime.preparation_protocol as installed_prep
    import pllm.runtime.residue_codec as installed_residue
    from pllm import _native
    installed_paths = [Path(m.__file__) for m in (installed_stage, installed_prep, installed_residue)]
    native_path = Path(_native.__file__)
    if native_path.suffix == ".py":
        from pllm.runtime._native_support import extension
        native_path = Path(extension().__file__)
    return {"schema": "pllm.cold_wire_objective.synthetic.v1", "seed": 6103,
        "scope": "public synthetic constrained integer distillation; no real checkpoint/transport/timing",
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "installed_serializer_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in installed_paths},
        "installed_native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "installed_serializer_matches_checkout": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            == hashlib.sha256((ROOT / "python/pllm/runtime" / p.name).read_bytes()).hexdigest()
            for p in installed_paths},
        "numpy_version": np.__version__, "msgpack_version": msgpack.__version__,
        "synthetic_data_sha256": hashlib.sha256(b"".join(x.astype("<i8").tobytes() for x in (teacher, train, heldout, adverse))).hexdigest(),
        "data_shapes": {"teacher": [8, 32], "train": [128, 32], "heldout": [96, 32], "adverse": [8, 32]},
        "thresholds": thresholds, "candidates": candidates, "installed_checks": installed,
        "scalar_max_normalized_requantization_same_integers": True,
        "exact_divisible_weight_reparameterization": exact_reparameterization,
        "hypothetical_geometry": geometry(),
        "unpriced": ["client manifest and embeddings/head/nonlinear artifacts", "publisher/provider scales and wrappers",
            "registration, reservations, inventories, ticket issue/ACK, refill/lifecycle controls",
            "transport authentication, HTTP/TLS framing/retries", "compression/checkpoint architecture metadata",
            "new model source license/task quality"],
        "real_quality": None, "rss": None, "full_transport": None, "latency": None,
        "extra_array_design_upper_bound_bytes": 8 * 1024 * 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Replay and compare, do not overwrite evidence")
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.output.read_text() != text:
            raise SystemExit("deterministic evidence mismatch")
        print("deterministic replay matches")
    else:
        args.output.write_text(text)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
