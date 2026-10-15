"""Small exactness/dispatch tests; never run full Qwen geometry here."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/release/examples/boundary_head_dotprod"


class BoundaryHeadDotprodProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, str(ROOT / "scripts/probe_boundary_head_dotprod.py"),
                        "--build", "--rows", "1", "--cols", "1", "--repeats", "1",
                        "--processes", "1"], cwd=ROOT, check=True, capture_output=True)

    def run_fixture(self, cols, mode="auto", rows=7, seed=1234567):
        p = subprocess.run([str(BINARY), str(rows), str(cols), "1", str(seed), mode, "emit"],
                           check=True, capture_output=True, text=True)
        return json.loads(p.stdout)

    def test_exact_independent_python_integer_oracle_and_tails(self):
        for cols in (1, 3, 4, 15, 16, 17, 31, 32, 33, 127, 896):
            for mode in ("auto", "scalar"):
                with self.subTest(cols=cols, mode=mode):
                    a = self.run_fixture(cols, mode)
                    expected = [sum(w * x for w, x in zip(
                        a["weights"][r*cols:(r+1)*cols], a["input"])) & 0xffffffff
                        for r in range(a["rows"])]
                    self.assertEqual(a["output"], expected)
                    self.assertEqual(a["baseline_output"], expected)
                    self.assertIn(-128, a["weights"])
                    self.assertEqual(a["owned_buffers"]["extra_persistent_weight_bytes"], 0)
                    selected = "arm-sdot-inline-asm" if mode == "auto" and a["dotprod_available"] else "scalar-balanced-limbs"
                    self.assertEqual(a["selected"], selected)

    def test_carry_edges_and_wraparound(self):
        a = self.run_fixture(896, rows=3)
        self.assertTrue({0, 1, 127, 128, 255, 256, 0xffffffff, 0x80000000,
                         0x7fffffff, 0x80808080, 0xffffff80, 0x8000}.issubset(a["input"]))
        totals = [sum(w*x for w, x in zip(a["weights"][r*896:(r+1)*896], a["input"]))
                  for r in range(3)]
        self.assertTrue(any(t < 0 or t > 0xffffffff for t in totals))
        self.assertEqual(a["output"], [t & 0xffffffff for t in totals])

    def test_reject_geometry_before_allocation(self):
        for rows, cols in ((0, 1), (1, 0), (1, 131072), (999999999, 896), (100000000, 1)):
            p = subprocess.run([str(BINARY), str(rows), str(cols), "1", "1", "auto", "quiet"],
                               capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)

    def test_largest_safe_accumulator_geometry(self):
        p = subprocess.run([str(BINARY), "1", "131071", "1", "987654321", "auto", "quiet"],
                           check=True, capture_output=True, text=True)
        self.assertTrue(json.loads(p.stdout)["exact"])

    def test_fresh_worker_rss(self):
        p = subprocess.run([sys.executable, str(ROOT / "scripts/probe_boundary_head_dotprod.py"),
                            "--rows", "3", "--cols", "17", "--repeats", "1", "--processes", "1"],
                           check=True, capture_output=True, text=True)
        a = json.loads(p.stdout)
        self.assertGreater(a["fresh_process"]["peak_rss_bytes"], 0)
        for key in ("setup", "packing", "kernel", "baseline"):
            self.assertGreaterEqual(a[key]["cpu_seconds"], 0)
            self.assertGreater(a[key]["elapsed_seconds"], 0)

    def test_both_order_offsets_and_fresh_process_balance(self):
        for offset in (0, 1):
            p = subprocess.run([sys.executable, str(ROOT / "scripts/probe_boundary_head_dotprod.py"),
                                "--rows", "3", "--cols", "17", "--repeats", "3",
                                "--processes", "2", "--order-offset", str(offset), "--emit"],
                               check=True, capture_output=True, text=True)
            records = [json.loads(line) for line in p.stdout.splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["provenance"]["binary_sha256"],
                             records[1]["provenance"]["binary_sha256"])
            for index, a in enumerate(records):
                chosen = (offset + index) % 2
                self.assertEqual(a["order_offset"], chosen)
                self.assertTrue(a["process_schedule"]["paired_start_balance"])
                self.assertEqual(a["process_schedule"]["index"], index)
                self.assertIn("rustc 1.", a["build_provenance"]["rustc_verbose"])
                self.assertEqual(a["compiled_source_sha256"], a["provenance"]["source_sha256"][
                    "crates/pllm-core/examples/boundary_head_dotprod.rs"])
                expected = [sum(w*x for w, x in zip(a["weights"][r*17:(r+1)*17], a["input"])) & 0xffffffff
                            for r in range(3)]
                self.assertEqual(a["output"], expected)
                self.assertEqual(a["baseline_output"], expected)
                for j, sample in enumerate(a["samples"]):
                    self.assertEqual(sample["order"], "candidate-first" if (j+chosen) % 2 == 0 else "baseline-first")
                    self.assertTrue(sample["exact"])
                    for clock in ("elapsed_seconds", "cpu_seconds"):
                        self.assertAlmostEqual(sample["candidate_including_packing"][clock],
                                               sample["packing"][clock] + sample["kernel"][clock])


if __name__ == "__main__":
    unittest.main()
