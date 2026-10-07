"""Complete Haar gated-block costs and a separately scoped fresh Qwen quality screen."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEVELS = (0, 2, 4)
# Fixed before the first measurement, independent of earlier Q7/Q12 screens.
PUBLIC_HELDOUT = (
    "Why does a bicycle tire become warm after a long ride?",
    "Explain why a straw appears bent in a glass of water.",
    "A farmer packs six apples into each of nine boxes. How many apples are packed?",
    "Why can an ice cube float in a glass of water?",
    "How does a compass help someone find a direction?",
    "What is the purpose of insulation around an electrical wire?",
    "Why does a shadow change length during the day?",
    "Explain why a lid helps a pot of water heat more quickly.",
    "A train covers sixty miles in two hours. What is its average speed?",
    "Why does a damp sponge weigh more than a dry sponge?",
    "How can a simple ramp make lifting a heavy box easier?",
    "What is one reason to record measurements during an experiment?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def even(values, denominator):
    quotient, remainder = np.divmod(values, denominator)
    return quotient + ((2 * remainder > denominator) | ((2 * remainder == denominator) & (quotient % 2 != 0)))


class NativeMapping:
    def __init__(self, levels):
        from pllm.metrics.wavelet_gated import _profile
        self.levels = levels
        table, self.digest = _profile(levels)
        self.table = np.asarray(table, dtype=np.int64)
        self.checked = 0

    def evaluate(self, gate, up):
        from pllm import _native
        if (gate.shape != up.shape or gate.dtype != np.float32 or up.dtype != np.float32
                or not 0 < gate.size <= 1 << 20):
            raise ValueError("numeric frame shape or representation differs")
        if (not np.all(np.isfinite(gate)) or not np.all(np.isfinite(up))
                or np.any(np.abs(gate) > 48) or np.any(np.abs(up) > 128)):
            raise ValueError("profile domain rejection")
        # Exact binary scaling of already-float32 values followed by ties-even.
        g = np.rint(gate.astype(np.float64) * (1 << 18)).astype("<i4")
        u = np.rint(up.astype(np.float64) * (1 << 18)).astype("<i4")
        actual = np.frombuffer(_native._wavelet_gated_clear(self.levels, g.tobytes(), u.tobytes()), dtype="<i4").reshape(g.shape)
        index = (g.astype(np.int64) + (64 << 18)) // (1 << (13 + self.levels))
        expected = even(self.table[index] * even(u.astype(np.int64), 512), 512)
        if not np.array_equal(actual, expected):
            raise ValueError("installed Haar mapping differs from independent integer oracle")
        self.checked += gate.size
        return actual.astype(np.float32) / np.float32(512)


def projected_cost(report, *, layers=24, width=4864, input_tokens=39, output_tokens=8):
    """Hypothetical complete MLP geometry; all non-MLP work remains outside scope."""
    if report["configuration"]["lanes"] != 256 or width % 256:
        raise ValueError("projection requires exact 256-lane chunks")
    chunks = layers * (input_tokens + output_tokens - 1) * (width // 256)
    resources = report["resources"]
    parts = resources["key_payload_bytes_by_operation"]
    if (sum(a for _, a, _ in parts) != resources["raw_key_payload_bytes_per_party"][0]
            or sum(b for _, _, b in parts) != resources["raw_key_payload_bytes_per_party"][1]):
        raise ValueError("complete key breakdown does not conserve material")
    return {
        "layers": layers, "width": width, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "executed_rows_per_layer": input_tokens + output_tokens - 1, "chunks": chunks,
        "combined_key_payload_bytes": chunks * sum(resources["raw_key_payload_bytes_per_party"]),
        "keys_by_operation_bytes": {name: chunks * (a + b) for name, a, b in parts},
        "peer_frame_bytes": chunks * sum(resources["peer_frame_bytes_by_direction"]),
        "client_boundary_payload_estimate_bytes": chunks * resources["client_entry_exit_payload_estimate_bytes"],
        "public_table_bytes_two_parties_once": 2 * resources["public_table_bytes_per_party"],
        "public_metadata_bytes_estimate": chunks * resources["public_metadata_bytes_estimate"],
        "full_wire_bytes": None, "protected_float_conversion": False, "executable_decoder": False,
    }


def costs():
    reports = []
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    for levels in LEVELS:
        for lanes in (1, 64, 256):
            raw = subprocess.run([sys.executable, "-m", "benchmarks.research.wavelet_gated", "--worker", str(levels), str(lanes)],
                                 env=environment, check=True, capture_output=True, text=True, timeout=120)
            reports.append(json.loads(raw.stdout))
    return {"schema": "pllm.wavelet_gated_cost.v1", "reports": reports,
            "fresh_processes": len(reports), "checked_outputs": sum(r["quality"]["checked_outputs"] for r in reports),
            "projections": [{"levels": r["configuration"]["component"]["params"]["levels"], **projected_cost(r)}
                            for r in reports if r["configuration"]["lanes"] == 256],
            "scope": "bounded in-process complete operators; projections are hypothetical MLP-only costs"}


def quality():
    from pllm import _native
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture
    from benchmarks.research.gated_numeric_screen import screen
    from benchmarks.research.gated_trajectories import generation

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
        bridges = {f"haar-{levels}": NativeMapping(levels) for levels in LEVELS}
        teacher = screen(fixture, bridges, PUBLIC_HELDOUT, check)
        greedy = generation(fixture, bridges, PUBLIC_HELDOUT, check)
        if (teacher["summary"]["control"]["bit_exact_positions"] != 48
                or teacher["summary"]["control"]["exact_kv_positions"] != 48
                or greedy["summary"]["control"]["exact_trajectories"] != 12):
            raise ValueError("fresh-runtime control is not exact")
        check()
    finally:
        guard.close()
    paths = ("benchmarks/research/wavelet_gated.py", "benchmarks/research/gated_numeric_screen.py",
             "benchmarks/research/gated_trajectories.py", "scripts/decoder_probe_support.py",
             "python/pllm/metrics/wavelet_gated.py", "crates/pllm-core/src/wavelet_gated_reference.rs")
    return {"schema": "pllm.wavelet_gated_quality.v1", "source": fixture.lock(),
            "configuration": {"levels": LEVELS, "maximum_input_tokens": 96, "same_token_positions": 4,
                "greedy_output_cap": 8, "client_input_conversion": "float32-to-signed-q18-ties-even-reject"},
            "numeric_contract_digests": {name: bridge.digest for name, bridge in bridges.items()},
            "independent_integer_checked_elements": {name: bridge.checked for name, bridge in bridges.items()},
            "installed_native_sha256": sha(Path(_native.__file__).read_bytes()),
            "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
            "teacher_forced": teacher, "greedy_generation": greedy,
            "source_sha256": {path: sha((ROOT / path).read_bytes()) for path in paths},
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
            "scope": "clear numeric replay on pinned compiled W8A8; no protected checkpoint execution",
            "protected_checkpoint_execution": False,
            "limitations": ["twelve public prompts are not representative quality", "reference is locked W8A8, not upstream floating point",
                            "full float-to-fixed private conversion and provider transport are absent", "profile levels have distinct numeric identities"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cost", "quality"), default="cost")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", nargs=2, type=int)
    args = parser.parse_args()
    if args.worker:
        from pllm.metrics import benchmark_operator
        from pllm.nonlinear import CurlWaveletGatedMultiplyQ9
        print(benchmark_operator(CurlWaveletGatedMultiplyQ9(levels=args.worker[0]), lanes=args.worker[1], samples=3).to_json())
        return
    if args.output is None or args.output.exists():
        parser.error("choose a new evidence output path")
    report = costs() if args.mode == "cost" else quality()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["projections"] if args.mode == "cost" else {
        "teacher": report["teacher_forced"]["summary"], "generation": report["greedy_generation"]["summary"]}))


if __name__ == "__main__":
    main()
