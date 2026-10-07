"""Complete Compact operator costs and frozen-profile, fresh Qwen quality."""
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
PROFILES = ((4, 2), (8, 3), (16, 3))
# Frozen before measurements; distinct from all earlier nonlinear cohorts.
PUBLIC_HELDOUT = (
    "Why do leaves collect dew on a cool morning?",
    "How does a pulley change the direction of a pulling force?",
    "A librarian puts eight books on each of seven shelves. How many books is that?",
    "What makes a metal spoon feel colder than a wooden spoon in the same room?",
    "Why is a map scale useful when planning a walk?",
    "Describe one way a seed can travel away from its parent plant.",
    "Why does an inflated balloon shrink when placed in a refrigerator?",
    "Explain how a prism separates white light into different colors.",
    "Four friends share twenty-four strawberries equally. How many does each get?",
    "What is the purpose of a control group in a simple experiment?",
    "Why can a tightly stretched string produce a higher musical note?",
    "How does a rain gauge help people compare rainfall across days?",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def even(values, divisor):
    quotient, remainder = np.divmod(values, divisor)
    return quotient + ((2 * remainder > divisor) | ((2 * remainder == divisor) & (quotient % 2 != 0)))


class NativeMapping:
    def __init__(self, pieces, degree):
        from pllm.metrics.compact_product import _profile
        from pllm.nonlinear import CompactGatedMultiplyQ12
        self.native, self.profile = _profile(CompactGatedMultiplyQ12(max_pieces=pieces, degree=degree))
        self.digest = self.profile["profile_digest"]
        self.checked = 0

    def evaluate(self, gate, up):
        if (gate.shape != up.shape or gate.dtype != np.float32 or up.dtype != np.float32
                or not 0 < gate.size <= 1 << 20):
            raise ValueError("numeric frame shape or representation differs")
        if (not np.all(np.isfinite(gate)) or not np.all(np.isfinite(up))
                or np.any(np.abs(gate) > 48) or np.any(np.abs(up) > 128)):
            raise ValueError("profile domain rejection")
        g = np.rint(gate.astype(np.float64) * (1 << 18)).astype("<i4")
        u = np.rint(up.astype(np.float64) * (1 << 18)).astype("<i4")
        actual = np.frombuffer(self.native.clear(g.tobytes(), u.tobytes()), dtype="<i4").reshape(g.shape)
        x = even(g.astype(np.int64), 64)
        activated = np.zeros(x.shape, dtype=np.int64)
        for piece in self.profile["pieces"]:
            mask = (x >= piece["lower"]) & (x < piece["upper"])
            t = (2 * x[mask] - piece["lower"] - piece["upper"]) * (1 << 20) // (piece["upper"] - piece["lower"])
            coefficients = piece["coefficients"][:self.profile["degree"] + 1]
            value = np.full(t.shape, coefficients[-1], dtype=np.int64)
            for coefficient in reversed(coefficients[:-1]):
                value = even(value * t, 1 << 20) + coefficient
            activated[mask] = even(value, 1 << 20)
        activated[x <= -(8 << 12)] = 0
        activated[x >= (8 << 12)] = x[x >= (8 << 12)]
        expected = even(activated * even(u.astype(np.int64), 64), 4096)
        if not np.array_equal(actual, expected):
            raise ValueError("installed Compact differs from independent vector integer oracle")
        self.checked += gate.size
        return actual.astype(np.float32) / np.float32(4096)


def costs():
    from .logrow_product import project
    reports = []
    for pieces, degree in PROFILES:
        for lanes in (1, 64, 256):
            raw = subprocess.run([sys.executable, "-m", "benchmarks.research.compact_product",
                "--worker", str(pieces), str(degree), str(lanes)], check=True, capture_output=True,
                text=True, timeout=900, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
            reports.append(json.loads(raw.stdout))
    return {"schema": "pllm.compact_product_cost.v1", "reports": reports,
        "fresh_processes": len(reports), "checked_outputs": sum(r["quality"]["checked_outputs"] for r in reports),
        "projections": [{"numeric_contract_digest": r["numeric_contract_digest"], **project(r)}
            for r in reports if r["configuration"]["lanes"] == 256],
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in
            ("benchmarks/research/compact_product.py", "benchmarks/research/logrow_product.py")},
        "scope": "bounded in-process complete operators; hypothetical MLP-only material, distinct numeric profiles"}


def quality():
    from pllm import _native
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture
    from .gated_numeric_screen import screen
    from .gated_trajectories import generation
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
        bridges = {f"pieces-{p}-degree-{d}": NativeMapping(p, d) for p, d in PROFILES}
        teacher = screen(fixture, bridges, PUBLIC_HELDOUT, check)
        greedy = generation(fixture, bridges, PUBLIC_HELDOUT, check)
        if (teacher["summary"]["control"]["bit_exact_positions"] != 48
                or teacher["summary"]["control"]["exact_kv_positions"] != 48
                or greedy["summary"]["control"]["exact_trajectories"] != 12):
            raise ValueError("fresh-runtime control is not exact")
        check()
    finally:
        guard.close()
    paths = ("benchmarks/research/compact_product.py", "benchmarks/research/gated_numeric_screen.py",
        "benchmarks/research/gated_trajectories.py", "scripts/decoder_probe_support.py",
        "python/pllm/metrics/compact_product.py", "crates/pllm-core/src/compact_product.rs")
    return {"schema": "pllm.compact_product_quality.v1", "source": fixture.lock(),
        "configuration": {"profiles": PROFILES, "public_calibration": "uniform 1024 bins over [-8,8)",
            "maximum_input_tokens": 96, "same_token_positions": 4, "greedy_output_cap": 8,
            "client_input_conversion": "float32-to-signed-q18-ties-even-reject"},
        "numeric_contract_digests": {name: b.digest for name, b in bridges.items()},
        "public_profiles": {name: b.profile for name, b in bridges.items()},
        "independent_integer_checked_elements": {name: b.checked for name, b in bridges.items()},
        "installed_native_sha256": sha(Path(_native.__file__).read_bytes()),
        "public_heldout_sha256": sha(json.dumps(PUBLIC_HELDOUT).encode()),
        "teacher_forced": teacher, "greedy_generation": greedy,
        "source_sha256": {p: sha((ROOT / p).read_bytes()) for p in paths},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "memory_guard": {"host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes},
        "scope": "clear numeric replay on pinned compiled W8A8; no protected checkpoint execution",
        "protected_checkpoint_execution": False,
        "limitations": ["twelve public prompts are not representative quality", "reference is locked W8A8, not upstream float",
            "private float conversion and provider transport are absent", "frozen profiles are distinct numeric identities"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cost", "quality"), default="cost")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", nargs=3, type=int)
    args = parser.parse_args()
    if args.worker:
        from pllm.metrics import benchmark_operator
        from pllm.nonlinear import CompactGatedMultiplyQ12
        p, d, lanes = args.worker
        print(benchmark_operator(CompactGatedMultiplyQ12(max_pieces=p, degree=d), lanes=lanes, samples=3).to_json())
        return
    if args.output is None or args.output.exists():
        parser.error("choose a fresh evidence output path")
    report = costs() if args.mode == "cost" else quality()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["projections"] if args.mode == "cost" else {
        "teacher": report["teacher_forced"]["summary"], "generation": report["greedy_generation"]["summary"]}))


if __name__ == "__main__":
    main()
