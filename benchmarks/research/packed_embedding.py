"""FastQuery packing costs, frozen public quantizers, and embedding/head ownership."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRECISIONS = ("int8", "int4", "mixed_4_3_3")
# Fixed before fitting or measurement; source-only fitting uses no prompt data.
PUBLIC_HELDOUT = (
    "Why does a bicycle need reflectors when ridden after sunset?",
    "Explain why dry sand can flow through a narrow hourglass.",
    "Seven boxes each contain six pencils. How many pencils are there?",
    "How does a compass help a hiker choose a direction?",
    "Why does steam form above a saucepan of boiling water?",
    "Describe how a bird's feathers help it stay warm.",
    "What causes the length of a shadow to change during the day?",
    "Explain why an ice cube floats in a glass of water.",
    "A shop sells five bags with nine marbles each. How many marbles is that?",
    "Why should an experiment be repeated more than once?",
    "How does a magnifying glass make small letters easier to see?",
    "What happens to the air inside a pump when its handle is pushed?",
)
SOURCES = (
    "benchmarks/research/packed_embedding.py", "scripts/decoder_probe_support.py",
    "crates/pllm-core/src/packed_embedding.rs", "crates/pllm-garble/src/packed_embedding.rs",
    "crates/pllm-python/src/packed_embedding.rs", "python/pllm/metrics/packed_embedding.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def projection(report, rows=151936, columns=896, queries=70):
    """Concrete unrepacked layout only; full-vocabulary execution is not admitted."""
    from pllm.metrics.packed_embedding import _FRAME_BYTES, _WIDTHS
    widths = _WIDTHS[report["configuration"]["component"]["params"]["precision"]]
    groups = math.ceil(columns / len(widths))
    measured_groups = math.ceil(report["configuration"]["lanes"] / len(widths))
    operations = dict((name, (left, right)) for name, left, right in report["resources"]["key_payload_bytes_by_operation"])
    conversion = operations["packed-share-conversion"]
    if any(k % measured_groups for k in conversion):
        raise ValueError("nonintegral per-coefficient conversion cost")
    return {
        "vocabulary": rows, "hidden_width": columns, "lookup_rows": queries,
        "encrypted_query_polynomials_per_row": math.ceil(rows / 4096),
        "response_ciphertexts_per_row": groups,
        "he_query_reply_frame_bytes": queries * (math.ceil(rows / 4096) + groups) * _FRAME_BYTES,
        "conversion_key_bytes_by_party": [queries * groups * k // measured_groups for k in conversion],
        "conversion_boundary_label_bytes_estimate": queries * groups * report["resources"]["client_entry_exit_payload_estimate_bytes"] // measured_groups,
        "provider_signed_code_bytes": rows * columns,
        "provider_packed_coefficient_bytes": math.ceil(rows / 4096) * 4096 * groups * 2,
        "tied_head_released_client_weight_bytes": 0,
        "scope": "symbolic blockwise vocabulary sum with full ciphertext replies, 39+32 executed lookup rows; no HE-key reuse or whole-decoder claim",
        "native_admitted": False, "pack_lwes": False, "full_wire_bytes": None,
    }


def costs():
    reports = []
    for rows, columns in ((17, 7), (256, 64), (4096, 64)):
        for precision in PRECISIONS:
            raw = subprocess.run([sys.executable, "-m", "benchmarks.research.packed_embedding",
                "--worker", precision, str(rows), str(columns)], check=True, capture_output=True,
                text=True, timeout=300, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            reports.append(json.loads(raw.stdout))
    return {"schema": "pllm.packed_embedding_cost.v1", "fresh_processes": len(reports),
        "reports": reports, "checked_outputs": sum(r["quality"]["checked_outputs"] for r in reports),
        "projections": [projection(r) for r in reports if r["configuration"]["lanes"] == 64
                        and r["configuration"]["component"]["params"]["vocabulary_size"] == 4096],
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in SOURCES},
        "scope": "bounded local encrypted operators and separate unadmitted symbolic full-width layout",
        "decoder_executable": False, "full_wire_bytes": None}


class ChannelMapping:
    def __init__(self, precision, maxima, sums, source):
        self.precision, self.checked = precision, 0
        columns = len(maxima)
        self.widths = np.full(columns, 8 if precision == "int8" else 4, dtype=np.uint8)
        if precision == "mixed_4_3_3":
            salient = np.argsort(-sums, kind="stable")[:math.ceil(columns / 3)]
            selected = set(salient)
            remaining = np.asarray([i for i in range(columns) if i not in selected], dtype=np.int64)
            self.widths[:] = 3
            self.widths[salient] = 4
            order = []
            for i, column in enumerate(salient):
                order.extend([int(column), *remaining[2*i:2*i+2].tolist()])
            self.order = np.asarray(order)
        else:
            self.order = np.arange(columns)
        qmax = (1 << (self.widths.astype(np.int64) - 1)) - 1
        # Explicit local contract: nearest power of two (ties even in log2),
        # then symmetric integer rounding/clipping. No STE/fine-tuning.
        scales = np.maximum(maxima.astype(np.float64) / qmax, np.finfo(np.float32).tiny)
        self.scales = np.exp2(np.rint(np.log2(scales))).astype("<f4")
        self.profile = {"schema": "pllm.embedding.channel_quantizer.v1", "precision": precision,
            "source": source, "widths": self.widths.tolist(), "scales": self.scales.tolist(),
            "packing_permutation": self.order.tolist(), "source_values": "compiled W8 per-token table reconstructed in float32",
            "scale_rounding": "nearest power of two, log2 ties even", "integer_rounding": "ties even, symmetric saturation",
            "salience": "complete public source table mean absolute value; stable channel-order ties",
            "runtime_permutation": "restore original channel order before the first normalization; body and head unchanged",
            "fine_tuning": False}
        self.digest = sha(json.dumps(self.profile, sort_keys=True, separators=(",", ":")).encode())

    def evaluate(self, values):
        from pllm import _native
        values = np.asarray(values, dtype="<f4")
        actual = np.frombuffer(_native.packed_embedding_quantize(
            values.tobytes(), self.widths.tobytes(), self.scales.tobytes()), dtype="i1").reshape(values.shape)
        qmax = (1 << (self.widths.astype(np.int64) - 1)) - 1
        oracle = np.clip(np.rint(values.astype(np.float64) / self.scales), -qmax, qmax)
        if not np.array_equal(actual, oracle):
            raise ValueError("native channel quantizer differs from independent integer oracle")
        self.checked += actual.size
        return actual.astype(np.float32) * self.scales


def fit(fixture, check):
    from pllm import _native
    stage = fixture.bundle.stages["token_lookup"]
    if stage.client_weight_layout != "embedding" or stage.client_aux_weight is not None:
        raise ValueError("quality fixture requires the explicit tied-table row orientation")
    codes, scales = stage.client_weight, stage.client_weight_scales
    rows, columns = codes.shape
    maxima, sums = np.zeros(columns, np.float32), np.zeros(columns, np.float64)
    for start in range(0, rows, 512):
        check()
        block = codes[start:start+512].astype(np.float32) * scales[start:start+512, None]
        stats = json.loads(_native.packed_embedding_statistics(columns, block.astype("<f4").tobytes()))
        if not np.array_equal(stats["maxima"], np.max(np.abs(block), axis=0)):
            raise ValueError("native channel maxima differ")
        if not np.array_equal(stats["sums"], np.sum(np.abs(block).astype(np.float64), axis=0)):
            raise ValueError("native channel sums differ")
        maxima = np.maximum(maxima, stats["maxima"])
        sums += stats["sums"]
    return {p: ChannelMapping(p, maxima, sums, fixture.lock()) for p in PRECISIONS}


def screen(fixture, mappings, check):
    from .piecewise_gated_reference import snapshot, state_digest
    trials, generations = [], []
    for text in PUBLIC_HELDOUT:
        check()
        ids = fixture.tokens(text)
        reference, reference_generation = [], None
        for name, mapping in [("reference", None), ("control", None), *mappings.items()]:
            def new_runtime():
                runtime = fixture.compiled.runtime(fixture.remote)
                original = runtime._token_lookup
                def lookup(token_ids):
                    check()
                    primary, auxiliary = original(token_ids)
                    if auxiliary is not None:
                        raise ValueError("quality screen does not admit auxiliary token tables")
                    return (primary if mapping is None else mapping.evaluate(primary)), None
                runtime._token_lookup = lookup
                return runtime
            runtime, observations = new_runtime(), []
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for position in range(4):
                    if position:
                        logits = runtime.forward_ids([reference[position-1][2]])[-1]
                    states, selected = snapshot(runtime), int(np.argmax(logits))
                    if name == "reference":
                        reference.append((logits.copy(), states, selected))
                    else:
                        expected, kv, token = reference[position]
                        observations.append({"position": position, "selection_match": selected == token,
                            "all_logits_bit_exact": np.array_equal(logits.view(np.uint32), expected.view(np.uint32)),
                            "kv_bit_exact": len(states) == len(kv) and all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(states, kv)),
                            "worst_abs_logit_error": float(np.max(np.abs(logits-expected))),
                            "kv_sha256": state_digest(states), "logits_sha256": sha(logits.tobytes())})
            finally:
                del runtime._token_lookup
                del runtime
            runtime, generated = new_runtime(), []
            try:
                _, logits, _ = runtime.prepare_ids(ids)
                for position in range(8):
                    generated.append(int(np.argmax(logits)))
                    if position < 7:
                        logits = runtime.forward_ids([generated[-1]])[-1]
                if name == "reference":
                    reference_generation = generated
                else:
                    generations.append({"candidate": name, "input_tokens": len(ids),
                        "exact_trajectory": generated == reference_generation,
                        "output_sha256": sha(np.asarray(generated, dtype="<u4").tobytes())})
            finally:
                del runtime._token_lookup
                del runtime
            if name != "reference":
                trials.append({"candidate": name, "input_tokens": len(ids), "positions": observations})
    summary = {}
    for name in ["control", *mappings]:
        checked = [p for row in trials if row["candidate"] == name for p in row["positions"]]
        summary[name] = {"prefill_matches": sum(p["selection_match"] for p in checked if p["position"] == 0),
            "decode_matches": sum(p["selection_match"] for p in checked if p["position"] > 0),
            "bit_exact_positions": sum(p["all_logits_bit_exact"] for p in checked),
            "exact_kv_positions": sum(p["kv_bit_exact"] for p in checked),
            "worst_abs_logit_error": max(p["worst_abs_logit_error"] for p in checked),
            "exact_trajectories": sum(r["exact_trajectory"] for r in generations if r["candidate"] == name)}
    if any(summary["control"][k] != v for k, v in {"bit_exact_positions": 48, "exact_kv_positions": 48, "exact_trajectories": 12}.items()):
        raise ValueError("fresh-runtime quality control differs")
    return {"summary": summary, "trials": trials, "generations": generations}


def quality():
    from pllm import _native
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture
    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("quality screen needs 4 GiB beyond host reserve")
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}},
        lambda _: os.kill(os.getpid(), signal.SIGTERM))
    def check():
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)
    guard.start()
    try:
        fixture = DecoderFixture(inputs=96, outputs=8)
        mappings = fit(fixture, check)
        result = screen(fixture, mappings, check)
        token = fixture.bundle.stages["token_lookup"]
        head = fixture.bundle.stages["lm_head"]
        if not np.shares_memory(token.client_weight, head.client_weight):
            raise ValueError("pinned source no longer shares its token/head matrix")
        check()
    finally:
        guard.close()
    return {"schema": "pllm.packed_embedding_quality.v1", "source": fixture.lock(),
        "configuration": {"maximum_input_tokens": 96, "same_token_positions": 4, "greedy_output_cap": 8},
        "results": result, "profiles": {p: m.profile for p, m in mappings.items()},
        "profile_digests": {p: m.digest for p, m in mappings.items()},
        "independent_checked_elements": {p: m.checked for p, m in mappings.items()},
        "ownership": {"token_table_bytes": int(token.client_weight.nbytes),
            "token_scale_bytes": int(token.client_weight_scales.nbytes),
            "token_head_shared_storage": bool(np.shares_memory(token.client_weight, head.client_weight)),
            "embedding_only_released_client_weight_bytes": 0,
            "scope": "existing tied client head still owns the same matrix; raw payload, not measured peak RSS"},
        "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in SOURCES},
        "installed_native_sha256": sha(Path(_native.__file__).read_bytes()),
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "scope": "local clear W8A8 embedding re-quantization screen with original body/head; no STE, upstream-float reference or protected checkpoint execution",
        "protected_checkpoint_execution": False, "full_wire_bytes": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cost", "quality"), default="cost")
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        from pllm.metrics import benchmark_operator
        from pllm.protocols import PrivateEmbeddingLookup
        precision, rows, columns = args.worker
        print(benchmark_operator(PrivateEmbeddingLookup(precision=precision, vocabulary_size=int(rows)),
                                 lanes=int(columns), samples=3).to_json())
        return
    if args.output is None or args.output.exists():
        parser.error("choose a fresh evidence output path")
    report = costs() if args.mode == "cost" else quality()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["projections"] if args.mode == "cost" else report["results"]["summary"]))


if __name__ == "__main__":
    main()
