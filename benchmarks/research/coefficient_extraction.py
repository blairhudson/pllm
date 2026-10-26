"""Bounded public coefficient algebra, not ROSETTA security or HE execution.

Stdlib only. Restriction is ordered RNS limb selection, never rescaling.
Dense public polynomials; independent convolution constant uses its N contributing
index pairs, not an N-squared full convolution whose unused terms waste budget.

No internal hard-deadline claim. Supervise the ENTIRE ordinary process, including
JSON encoding, file publication, stdout and exit, with this standard caller:

    uv run --no-sync --offline --no-python-downloads python -c 'import subprocess,sys; subprocess.run([sys.executable,"benchmarks/research/coefficient_extraction.py"],timeout=30,check=True)'

Child directly publishes file and stdout within supervision; parent never reprints.
Only terminal success within that timeout counts. Timeout or nonzero exit invalidates
any partial artifact/summary; no success is inferred from a file or partial stdout.
Ordinary publication-timeout controls (not part of the experiment):

    uv run --no-sync --offline --no-python-downloads python -m unittest \
        benchmarks.research.coefficient_extraction.PublicationTimeoutTests
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import time
import tracemalloc
import unittest
from pathlib import Path

MAX_OPS = 1 << 20
MAX_OWNERS = 1 << 20
MAX_EXPERIMENT = 64 << 20
MAX_OUTPUT = 1 << 20


def descriptors():
    cases = []
    # Exhaustive degree-one ring Z/3, all b,a,s; 27 cases.
    for b in range(3):
        for a in range(3):
            for s in range(3):
                cases.append((1, (3,), 0, (b, a, s)))
    cases.extend(
        [
            (2, (17,), 0, None),
            (4, (17,), 0, None),
            (2048, (65537, 12289), 1, None),
            (2048, ((1 << 61) - 1,), 0, None),
        ]
    )
    return cases


def quote(cases):
    """Validate ALL case metadata before any polynomial factory."""
    if not 1 <= len(cases) <= 32:
        raise ValueError("case limit")
    ops, owners, outputs = 512, 32768, 0
    for n, moduli, selected, _ in cases:
        if not 1 <= n <= 4096 or n & (n - 1):
            raise ValueError("degree")
        if not 1 <= len(moduli) <= 2 or not 0 <= selected < len(moduli):
            raise ValueError("ordered limbs")
        if len(set(moduli)) != len(moduli):
            raise ValueError("duplicate modulus")
        if any(not 3 <= q < 1 << 61 or q % 2 == 0 for q in moduli):
            raise ValueError("modulus")
        inputs = 2 * n * len(moduli)
        owners += 8 * (inputs + n + 1 + n)
        outputs += 8 * (n + 1)
        ops += 4 * inputs + 48 * n + 512
    if ops > MAX_OPS or owners > MAX_OWNERS or outputs > MAX_OUTPUT:
        raise ValueError("aggregate admission")
    return {"word_ops_upper_bound": ops, "kernel_owner_bytes": owners}


def vectors(n, moduli, selected, tiny):
    backing = []
    for q in moduli:
        backing.extend((q - 1 - (37 * i % q)) % q for i in range(2 * n))
    q = moduli[selected]
    secret = [(q - 1 - (11 * i % q)) % q for i in range(n)]
    if tiny is not None:
        backing[:] = tiny[:2]
        secret[:] = tiny[2:]
    return backing, secret


def convolution_constant(b, a, secret, q):
    """Independent dense negacyclic constant: enumerate contributing diagonals.

    i+j=0 contributes positively; i+j=N contributes negatively. Reduce every
    product immediately, matching safe fixed-width modular implementation.
    """
    value = b[0]
    products = 0
    for degree, sign in ((0, 1), (len(a), -1)):
        lo = max(0, degree - len(secret) + 1)
        hi = min(len(a) - 1, degree)
        for i in range(lo, hi + 1):
            j = degree - i
            value = (value + sign * ((a[i] * secret[j]) % q)) % q
            products += 1
    return value, products


def candidate(backing, n, selected, q):
    at = 2 * n * selected
    return [backing[at], backing[at + n]] + [(-backing[at + 2 * n - j]) % q for j in range(1, n)]


def experiment():
    started = time.perf_counter()
    cases = descriptors()
    admitted = quote(cases)
    tracemalloc.start()
    rows = []
    total_products = 0
    for index, (n, moduli, selected, tiny) in enumerate(cases):
        backing, secret = vectors(n, moduli, selected, tiny)
        q = moduli[selected]
        if len(backing) != 2 * n * len(moduli) or len(secret) != n:
            raise ValueError("complete backing length")
        for limb, modulus in enumerate(moduli):
            if any(not 0 <= x < modulus for x in backing[2 * n * limb : 2 * n * (limb + 1)]):
                raise ValueError("noncanonical limb")
        if any(not 0 <= x < q for x in secret):
            raise ValueError("noncanonical secret")
        at = 2 * n * selected
        tick = time.perf_counter()
        direct, products = convolution_constant(
            backing[at : at + n], backing[at + n : at + 2 * n], secret, q
        )
        oracle_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        extracted = candidate(backing, n, selected, q)
        dot = extracted[0]
        for a, s in zip(extracted[1:], secret, strict=True):
            dot = (dot + (a * s) % q) % q
        candidate_seconds = time.perf_counter() - tick
        if direct != dot:
            raise AssertionError("independent public algebra mismatch")
        total_products += products + n
        rows.append(
            {
                "case": index,
                "degree": n,
                "ordered_moduli": moduli,
                "selected": selected,
                "secret_dimension": n,
                "phase_constant": direct,
                "extracted_dot": dot,
                "parity": True,
                "modular_products": products + n,
                "oracle_clear_seconds": oracle_seconds,
                "candidate_clear_seconds": candidate_seconds,
            }
        )
        # Ordinary buffers; no claimed secret erasure or cryptographic lifecycle.
        del backing, secret, extracted
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if peak > MAX_EXPERIMENT:
        raise RuntimeError("experiment traced allocation limit")
    report = {
        "scope": "public deterministic algebra only; not ROSETTA roundtrip/security",
        "cases": len(rows),
        "all_parity": all(row["parity"] for row in rows),
        **admitted,
        "modular_products": total_products,
        "traced_python_peak_bytes": peak,
        "traced_peak_is_not_process_rss": True,
        "clear_seconds": time.perf_counter() - started,
        "restriction_counterexample": {
            "integer": 20,
            "q0": 17,
            "dropped_prime": 5,
            "restriction": 3,
            "rounded_rescale": 4,
            "modular_division": 4,
        },
        "results": rows,
    }
    return report


def main():
    # Ordinary single process. External supervisor must cover this entire function
    # through exit, not merely experiment(). Existing/partial output is not success.
    report = experiment()
    encoded = json.dumps(report, indent=2) + "\n"
    if len(encoded.encode()) > MAX_OUTPUT:
        raise RuntimeError("output byte limit")
    path = Path("target/rosetta-coefficient-extraction.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(encoded)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            sort_keys=True,
        )
    )


class PublicationTimeoutTests(unittest.TestCase):
    """Standard subprocess timeout controls with test-only publication delays."""

    def _timeout_during_publication(self, stage):
        # Patch in the CHILD, not the supervisor: computation completes immediately,
        # then ordinary main() stalls during write or after printing its summary.
        child = textwrap.dedent(
            """
            import builtins
            import importlib.util
            import sys
            import time
            from pathlib import Path
            from unittest.mock import patch

            spec = importlib.util.spec_from_file_location("publication_control", sys.argv[1])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            stage = sys.argv[2]
            original_write = Path.write_text
            original_print = builtins.print

            def delay_write(path, *args, **kwargs):
                result = original_write(path, *args, **kwargs)
                if stage == "write":
                    sys.stderr.write("publication-written\\n")
                    sys.stderr.flush()
                    time.sleep(60)
                return result

            def delay_summary(*args, **kwargs):
                result = original_print(*args, **kwargs)
                sys.stdout.flush()
                if stage == "summary":
                    sys.stderr.write("summary-printed\\n")
                    sys.stderr.flush()
                    time.sleep(60)
                return result

            with patch.object(module, "experiment", return_value={"all_parity": True, "results": []}), \\
                 patch.object(Path, "write_text", delay_write), \\
                 patch.object(builtins, "print", delay_summary):
                module.main()
            """
        )
        scratch = Path(__file__).resolve().parents[2] / "target"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            with self.assertRaises(subprocess.TimeoutExpired) as refused:
                subprocess.run(
                    [sys.executable, "-c", child, str(Path(__file__).resolve()), stage],
                    cwd=directory,
                    timeout=2,
                    check=True,
                    capture_output=True,
                )
            # Proves the timeout happened AFTER computation, inside publication.
            marker = b"publication-written" if stage == "write" else b"summary-printed"
            self.assertIn(marker, refused.exception.stderr or b"")
            self.assertTrue(
                (Path(directory) / "target/rosetta-coefficient-extraction.json").exists()
            )
            if stage == "summary":
                self.assertIn(b'"all_parity": true', refused.exception.stdout or b"")
            else:
                self.assertFalse(refused.exception.stdout)
            # A file, even alongside a printed parity summary, is NOT terminal
            # success: check=True invocation raised instead of returning success.

    def test_timeout_during_publication_is_not_success(self):
        self._timeout_during_publication("write")

    def test_timeout_after_summary_is_not_terminal_success(self):
        self._timeout_during_publication("summary")


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit("no arguments supported")
    main()
