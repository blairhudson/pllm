"""NEXUS Algorithm 5 clear-only column experiment; no native/HE execution.

Run: python3 examples/benchmarks/nexus_column_scores.py
Launcher must enforce a 30-second external wall timeout; this script supplies
a 31-second fallback wall alarm and a separate 30-second CPU limit.
Ordinary benchmark JSON, not a receipt or source/checkpoint authentication.
One process/thread, deterministic integer fixtures, no upstream runtime dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
import signal
import sys
import time


MEMORY_BYTES = 64 << 20
OUTPUT_BYTES = 1 << 20
OUTPUT = Path("target/nexus-column-scores.json")


def fixture(m: int, n: int) -> tuple[list[int], list[int]]:
    return (
        [(x * 7 + x // n * 3 + 2) % 17 - 8 for x in range(m * n)],
        [(x * 11 + x // n * 5 + 9) % 17 - 8 for x in range(m * n)],
    )


def dense_integer(q: list[int], k: list[int], m: int, n: int) -> list[int]:
    """Independent row-pair integer dot products: no diagonal/column packing."""
    return [
        sum(a * b for a, b in zip(q[row : row + n], k[col : col + n]))
        for row in range(0, m * n, n)
        for col in range(0, m * n, n)
    ]


def column_scores(q: list[int], k: list[int], m: int, n: int) -> list[float]:
    """Logical columns, not physical CKKS slots. Separate multiply and add."""
    query = [[float(q[t * n + j]) for t in range(m)] for j in range(n)]
    key = [[float(k[t * n + j]) for t in range(m)] for j in range(n)]
    result = [0.0] * (m * m)
    for diagonal in range(m):
        r = [0.0] * m
        for j in range(n):
            for t in range(m):
                product = query[j][t] * key[j][t]
                r[t] = r[t] + product
        for t in range(m):
            result[t * m + (t + diagonal) % m] = r[t]
        # Include final diagonal's rotations: key packing returns to original.
        for j in range(n):
            first = key[j][0]
            for t in range(m - 1):
                key[j][t] = key[j][t + 1]
            key[j][m - 1] = first
    assert all(key[j][t] == k[t * n + j] for j in range(n) for t in range(m))
    return result


def peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def main() -> None:
    # External tool timeout must also be 30000 ms. Alarm is a 31-second fallback.
    signal.alarm(31)
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_BYTES, OUTPUT_BYTES))
    # RLIMIT_RSS is advisory on macOS. Report/check actual process peak instead
    # of pretending a 64 MiB logical payload quote is a hard RSS sandbox.
    try:
        resource.setrlimit(resource.RLIMIT_RSS, (MEMORY_BYTES, MEMORY_BYTES))
        rss_limit = "advisory"
    except (OSError, ValueError):
        # macOS may refuse RLIMIT_RSS entirely; never hide that distinction.
        rss_limit = "unavailable"
    if sys.platform != "darwin":
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    results = []
    for m, n in ((1, 1), (4, 3), (128, 64)):
        q, k = fixture(m, n)
        originals = (tuple(q), tuple(k))
        start = time.perf_counter()
        expected = dense_integer(q, k, m, n)
        dense_seconds = time.perf_counter() - start
        start = time.perf_counter()
        actual = column_scores(q, k, m, n)
        column_seconds = time.perf_counter() - start
        assert actual == expected
        assert originals == (tuple(q), tuple(k))
        assert peak_bytes() <= MEMORY_BYTES
        results.append(
            {
                "m": m,
                "n": n,
                "exact_binary64_integer_parity": True,
                "multiplies": m * m * n,
                "adds": m * m * n,
                "logical_left_column_rotations_including_final": m * n,
                "scatter_writes": m * m,
                "dense_clear_seconds": dense_seconds,
                "column_clear_seconds": column_seconds,
            }
        )
    report = {
        "status": "CLEAR_REFERENCE_ONLY_NOT_FULL_NEXUS_OR_PROTECTED_ADOPTION",
        "fixtures": results,
        "processes": 1,
        "threads": 1,
        "required_external_wall_timeout_seconds": 30,
        "fallback_wall_alarm_seconds": 31,
        "cpu_limit_seconds": 30,
        "max_memory_bytes": MEMORY_BYTES,
        "memory_guard": "observed_peak_check",
        "rss_limit": rss_limit,
        "peak_rss_bytes": peak_bytes(),
        "max_output_bytes": OUTPUT_BYTES,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    assert len(encoded.encode()) < OUTPUT_BYTES
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(encoded)
    print(encoded, end="")
    signal.alarm(0)


if __name__ == "__main__":
    main()
