"""Focused non-cryptographic tests; no native binary or upstream code executes."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import runpy
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[4]
Q = runpy.run_path(str(ROOT / "scripts/qualify_rust_ckks_refresh.py"))


class AdmissionTests(unittest.TestCase):
    def test_eager_keys_alone_exceed_entire_owner_cap(self):
        c = Q["contract"]()
        b = Q["key_lower_bound"](c)
        # A payload lower bound suffices to reject, never to admit. These
        # independent source arithmetic values include 32 rotations + conjugate.
        self.assertEqual(b["prepared_automorphism_payload_bytes"], 31_973_179_392)
        self.assertEqual(b["automorphism_preparation_overlap_payload_bytes"], 39_966_474_240)
        self.assertGreater(b["prepared_automorphism_payload_bytes"], 8 * (1 << 30))
        self.assertIn("prepared_automorphism_payload_exceeds_8_gib", Q["admission_blockers"](c))

    def test_drift_and_omitted_keys_do_not_lower_admission(self):
        c = Q["contract"]()
        c["keys"]["rotation_shifts"].pop()
        with self.assertRaises(ValueError):
            Q["key_lower_bound"](c)
        c = Q["contract"]()
        c["parameters"]["base2k"] = 28
        with self.assertRaises(ValueError):
            Q["key_lower_bound"](c)

    def test_filling_evidence_fields_cannot_enable_missing_pipeline(self):
        c = Q["contract"]()
        c["resources"]["full_owner_bound_bytes"] = 1
        pq = c["parameters"]["parameter_qualification"]
        pq["lattice_estimator_receipt"] = pq["whole_circuit_failure_probability_receipt"] = "pretend"
        self.assertIn("encrypted_pipeline_not_implemented_for_rejected_profile", Q["admission_blockers"](c))
        self.assertIn("prepared_automorphism_payload_exceeds_8_gib", Q["admission_blockers"](c))


class OracleTests(unittest.TestCase):
    def test_binary_rational_composition_and_wrap(self):
        for j in (0, 118, 126, 127, 32758, 32759, 32767):
            a, b = Q["oracle_slot"]("affine_1", (j + 8) % 32768)
            independently_composed = (2 * a - Fraction(1, 32), -2 * b + Fraction(1, 64))
            self.assertEqual(Q["oracle_slot"]("refresh_2", j), independently_composed)
        self.assertEqual(Q["oracle_slot"]("refresh_2", 1000), (0, Fraction(1, 32)))

    def test_all_padding_checked_and_nonfinite_rejected(self):
        values = [Q["oracle_slot"]("refresh_2", j) for j in range(32768)]
        real, imag = [float(v[0]) for v in values], [float(v[1]) for v in values]
        self.assertEqual(Q["check_slots"]("refresh_2", real, imag, 2**-12)["components_checked"], 65536)
        # Slot 32758 maps to original padding, immediately before wrap to valid.
        imag[32758] += 0.01
        with self.assertRaisesRegex(ValueError, "slot 32758"):
            Q["check_slots"]("refresh_2", real, imag, 2**-12)
        imag[32758] -= 0.01
        real[17] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            Q["check_slots"]("refresh_2", real, imag, 2**-12)
        with self.assertRaises(ValueError):
            Q["check_slots"]("refresh_2", real[:-1], imag, 2**-12)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.host = {"physical_bytes": 32 << 30, "page_bytes": 16384,
                     "free_speculative_bytes": 16 << 30, "swap_used_bytes": 3 << 30,
                     "swapout_pages": 100}

    def test_parser_does_not_count_inactive_pages_as_free(self):
        vm = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
              'Pages free: 10.\nPages speculative: 2.\nPages inactive: 999999.\nSwapouts: 100.\n')
        h = Q["parse_host"](vm, 'vm.swapusage: total = 4096.00M used = 3072.00M free = 1024.00M', 32 << 30)
        self.assertEqual(h["free_speculative_bytes"], 12 * 16384)
        self.assertEqual(h["swap_used_bytes"], 3 << 30)
        with self.assertRaises(ValueError):
            Q["parse_host"](vm.replace('Swapouts: 100.', ''), 'used = 1.0M', 32 << 30)

    def test_swapout_detected_even_if_current_swap_usage_declines(self):
        sample = {"host": deepcopy(self.host), "owner_rss_bytes": 1 << 20}
        sample["host"]["swapout_pages"] += 1
        sample["host"]["swap_used_bytes"] -= 1 << 20
        self.assertEqual(Q["resource_violation"](sample, self.host, 1 << 30, 4 << 30, 1, 600), "additional_swapout")
        sample["host"] = deepcopy(self.host)
        sample["owner_rss_bytes"] = 8 * (1 << 30) + 1
        self.assertEqual(Q["resource_violation"](sample, self.host, 8 << 30, 4 << 30, 1, 600), "aggregate_owner_rss_limit")

    def test_insufficient_headroom_launches_nothing(self):
        host = dict(self.host, free_speculative_bytes=1)
        with tempfile.TemporaryDirectory(prefix="poulpy-guard-", dir=Q["PROBE"]) as d:
            result = Q["guarded"](["must-not-execute"], d, seconds=1, owner_cap=1 << 30,
                                   reserve=4 << 30, disk_required=0, sample_host=lambda: host)
            self.assertEqual(result["status"], "blocked_prelaunch")
            self.assertNotIn("pid", result)

    @unittest.skipUnless(sys.platform == "darwin", "macOS process-group observer")
    def test_final_observation_cannot_hide_exceeded_deadline(self):
        calls = 0

        def delayed_host():
            nonlocal calls
            calls += 1
            if calls == 2:
                # The child exits during its first observation. Previously the
                # pre-read clock let this successful exit bypass the deadline.
                time.sleep(0.25)
            return self.host

        with tempfile.TemporaryDirectory(prefix="poulpy-guard-", dir=Q["PROBE"]) as d:
            result = Q["guarded"]([sys.executable, "-I", "-B", "-c", "pass"], d,
                                   seconds=0.1, owner_cap=512 << 20, reserve=0,
                                   disk_required=0, sample_host=delayed_host)
            self.assertEqual(result["status"], "failed_guard", result)
            self.assertEqual(result["reason"], "wall_clock_limit", result)
            self.assertFalse(result["trace_complete"], result)
            self.assertTrue(result["cleanup"]["complete"], result)

    @unittest.skipUnless(sys.platform == "darwin", "macOS process-group observer")
    def test_timeout_cleans_leader_and_grandchild(self):
        code = ('import subprocess,sys,time; '
                'subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); time.sleep(30)')
        with tempfile.TemporaryDirectory(prefix="poulpy-guard-", dir=Q["PROBE"]) as d:
            result = Q["guarded"]([sys.executable, "-c", code], d, seconds=0.3,
                                   owner_cap=512 << 20, reserve=0, disk_required=0,
                                   sample_host=lambda: self.host, interval=0.05)
            self.assertEqual(result["status"], "failed_guard", result)
            self.assertEqual(result["reason"], "wall_clock_limit", result)
            self.assertTrue(result["cleanup"]["complete"], result)
            self.assertEqual(Q["group_members"](result["pgid"]), [])
            self.assertTrue((Path(d) / "resources.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
