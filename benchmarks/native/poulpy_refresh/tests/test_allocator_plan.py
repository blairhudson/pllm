"""Integer budget tests and tiny independent macOS VM observations, not HE."""

import ctypes
import json
from pathlib import Path
import runpy
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
P = runpy.run_path(str(HERE / "allocator_plan.py"))


class AllocationPlanTests(unittest.TestCase):
    def test_independent_alignment_envelope_covers_all_page_residues(self):
        page = P["PAGE"]
        for exponent in range(0, 22):
            alignment = 1 << exponent
            for size in (1, 63, 64, page - 64, page, 3 * page + 1):
                g = P["allocation_geometry"](size, alignment)
                for base in range(page, max(2 * page, alignment + page), page):
                    user = ((base + 64 + max(alignment, 16) - 1) // max(alignment, 16)) * max(alignment, 16)
                    self.assertEqual(user % alignment, 0)
                    self.assertGreaterEqual(user - base, 64)
                    self.assertLessEqual(user - base + size, g["mapped_bytes"])
                    self.assertEqual(g["charge_bytes"], g["mapped_bytes"] + 2 * page)

    def test_zero_invalid_layout_and_charge_overflow_fail_closed(self):
        for size, align in ((0, 1), (-1, 1), (1, 0), (1, 3), (True, 1), ((1 << 63) - 1, 1), (1, 1 << 63)):
            with self.assertRaises(ValueError):
                P["allocation_geometry"](size, align)
        with self.assertRaises(ValueError):
            P["allocation_geometry"](1, 1, ticket=(1 << 64) - 1)

    def test_per_owner_rounding_and_conditional_fit_do_not_imply_admission(self):
        plan = P["plan"]()
        self.assertGreater(plan["conditional_factor_2_margin_bytes"], 0)
        self.assertGreater(plan["conditional_uniform_dft_margin_bytes"], 0)
        self.assertEqual(plan["uniform_dft_known_charge_bytes"] - plan["factor_2_evaluation_known_charge_bytes"], 4 * 65536 * 2 * 32)
        self.assertFalse(plan["admitted"])
        self.assertFalse(plan["budget_sufficiency_proven"])
        self.assertFalse(plan["conditional_total_is_complete_owner_bound"])
        self.assertIsNone(plan["attributable_kernel_io_upper_bound_bytes"])
        self.assertFalse(plan["bookkeeping_ticket_is_proven_kernel_bound"])
        self.assertIsNone(plan["full_owner_completion_bound_bytes"])
        # Aggregating tiny diagonals before page rounding would undercharge.
        self.assertGreater(P["buffer_charge"](512, 15), P["buffer_charge"](512 * 15))
        self.assertLess(plan["factor_2_loading_known_charge_bytes"], plan["factor_2_evaluation_known_charge_bytes"])

    def test_limits_and_original_profile_agree(self):
        c = json.loads((HERE / "bounded-allocator-contract.json").read_text())
        original = json.loads((HERE / "contract.json").read_text())
        self.assertEqual(c["owner_ceiling_bytes"], original["resources"]["owner_envelope_ceiling_bytes"])
        self.assertEqual(c["poulpy_commit"], original["dependency"]["commit"])
        self.assertEqual(c["preset"], original["parameters"]["preset"])
        self.assertEqual(c["cpu_execution_threads"], 1)
        self.assertEqual(c["maximum_refreshes"], 2)
        a = c["allocator"]
        self.assertEqual(a["charged_ceiling_bytes"], P["CHARGE_CEILING"])
        self.assertEqual(a["charged_ceiling_bytes"] + c["guard_reserve_bytes"] + a["non_rust_runtime_allowance_bytes"] + a["page_table_allowance_bytes"], c["owner_ceiling_bytes"])
        sources = json.loads((HERE / "staged_admission.sources.json").read_text())["supplementary_sources"]
        upstream_lock = next(s for s in sources if s["path"] == "Cargo.lock")
        self.assertEqual(c["solver_dependency_plan"]["upstream_lock_sha256"], upstream_lock["sha256"])
        self.assertFalse(c["hard_cap_enforcement_proves_completion"])

    def test_explicit_leaf_phases_match_aggregate_and_stay_partial(self):
        plan = P["plan"]()
        phases = plan["candidate_dft_phases"]
        self.assertEqual(phases["27_limb_loading"]["known_charge_bytes"], plan["factor_2_loading_known_charge_bytes"])
        self.assertEqual(phases["27_limb_evaluation"]["known_charge_bytes"], plan["factor_2_evaluation_known_charge_bytes"])
        self.assertEqual(phases["28_limb_evaluation"]["known_charge_bytes"], plan["uniform_dft_known_charge_bytes"])
        for name, phase in phases.items():
            self.assertIsNone(phase["complete_upper_bound_bytes"])
            self.assertEqual(phase["known_mapping_count"], 342 if name.endswith("loading") else 340)
            leaves = {leaf["name"] for leaf in phase["leaf_owners"]}
            self.assertEqual("vmp_private/0" in leaves, name.endswith("loading"))
            self.assertEqual("matrix_scratch/0" in leaves, name.endswith("evaluation"))
        self.assertEqual(plan["conditional_uniform_dft_total_bytes"], 8_552_267_776)
        self.assertEqual(plan["conditional_uniform_dft_margin_bytes"], 37_666_816)
        self.assertEqual(plan["host_reserve_bytes"], 4 << 30)
        self.assertTrue(plan["unclosed_phases"])

    def test_capacity_retention_and_inclusive_cap(self):
        charge = P["buffer_charge"](P["PAGE"])
        ledger = P["MappingLedger"](ceiling=charge)
        ledger.allocate("original_capacity", P["PAGE"])
        self.assertEqual(ledger.charged, charge)
        # Logical truncation/drop cannot replace the original Layout capacity.
        ledger.retire("original_capacity")
        with self.assertRaisesRegex(ValueError, "rejected"):
            ledger.allocate("tiny_replacement", 1)
        self.assertEqual(ledger.charged, charge)
        ledger.unmap_result("original_capacity", succeeded=True)
        ledger.allocate("tiny_replacement", 1)
        self.assertLess(ledger.charged, charge)
        with self.assertRaisesRegex(ValueError, "still live"):
            ledger.unmap_result("tiny_replacement", succeeded=True)
        with self.assertRaises(ValueError):
            P["MappingLedger"](ceiling=charge - 1).allocate("too_big", P["PAGE"])
        with self.assertRaises(ValueError):
            P["buffer_charge"](1, -1)

    def test_realloc_overlap_shrink_and_failed_unmap_stay_charged(self):
        charge = P["buffer_charge"](128)
        ledger = P["MappingLedger"](ceiling=charge)
        ledger.allocate("old", 128)
        for capacity in (64, 256):
            with self.assertRaisesRegex(ValueError, "rejected"):
                ledger.realloc("old", "new", capacity, old_unmap_succeeded=True)
            self.assertFalse(ledger.mappings["old"].retired)
            self.assertEqual(ledger.charged, charge)
        successful = P["MappingLedger"](ceiling=2 * charge)
        successful.allocate("old", 128)
        successful.realloc("old", "grown", 256, old_unmap_succeeded=True)
        successful.realloc("grown", "shrunk", 64, old_unmap_succeeded=True)
        self.assertEqual(set(successful.mappings), {"shrunk"})
        self.assertEqual(successful.mappings["shrunk"].capacity, 64)
        self.assertEqual(successful.charged, charge)
        self.assertEqual(successful.peak_charged, 2 * charge)
        ledger = P["MappingLedger"](ceiling=2 * charge)
        ledger.allocate("old", 128)
        self.assertEqual(ledger.realloc("old", "new", 64, old_unmap_succeeded=False), "new")
        self.assertTrue(ledger.poisoned)
        self.assertEqual(ledger.charged, 2 * charge)
        self.assertEqual(ledger.peak_charged, 2 * charge)
        with self.assertRaisesRegex(ValueError, "rejected"):
            ledger.realloc("new", "same", 64, old_unmap_succeeded=True)
        with self.assertRaisesRegex(ValueError, "no production retry"):
            ledger.unmap_result("old", succeeded=True)
        ledger.retire("new")
        ledger.unmap_result("new", succeeded=True)
        self.assertEqual(ledger.charged, charge)
        with self.assertRaisesRegex(ValueError, "rejected"):
            ledger.allocate("after_poison", 1)

    def test_mapping_count_failure_and_missing_unmap_outcome_fail_closed(self):
        ledger = P["MappingLedger"](max_mappings=1)
        ledger.allocate("one", 1)
        with self.assertRaisesRegex(ValueError, "rejected"):
            ledger.allocate("two", 1)
        ledger.retire("one")
        with self.assertRaises(ValueError):
            ledger.unmap_result("one", succeeded=None)
        ledger.unmap_result("one", succeeded=False)
        self.assertEqual(len(ledger.mappings), 1)
        self.assertTrue(ledger.poisoned)

    def test_dft_transition_requires_raw_and_old_baby_unmapping(self):
        ledger = P["MappingLedger"]()
        for name, capacity, alignment in P["dft_phase_owners"](28, loading=True):
            ledger.allocate(name, capacity, alignment)
        scratch = P["S"]["matrix_query_certificate"]()["matrix_query_bytes"]
        # Loading plus evaluation scratch cannot coexist even without unknowns.
        with self.assertRaisesRegex(ValueError, "rejected"):
            ledger.allocate("evaluation", scratch)
        for name in ("raw_import/0", "prepare_scratch/0", "vmp_private/0"):
            ledger.retire(name)
            ledger.unmap_result(name, succeeded=True)
        ledger.allocate("evaluation", scratch)
        for i in range(4):
            ledger.retire(f"baby/{i}")
        # One next-cache leaf fits the residual; the complete cache does not.
        ledger.allocate("next_baby/0", P["S"]["d"](2, 27))
        with self.assertRaisesRegex(ValueError, "rejected"):
            ledger.allocate("next_baby/1", P["S"]["d"](2, 27))
        ledger.retire("next_baby/0")
        ledger.unmap_result("next_baby/0", succeeded=True)
        for i in range(4):
            ledger.unmap_result(f"baby/{i}", succeeded=True)
        for i in range(4):
            ledger.allocate(f"next_baby/{i}", P["S"]["d"](2, 27))
        self.assertEqual(ledger.charged, P["plan"]()["factor_2_evaluation_known_charge_bytes"])

    def test_isolated_commands_cannot_build_dependencies(self):
        runner = runpy.run_path(str(HERE / "allocator_probe.py"))
        commands = runner["commands"](HERE / "target/unused-test-path")
        self.assertEqual(len(commands["compile"]), 2)
        for command in commands["compile"]:
            self.assertEqual(command[:4], ["rustup", "run", "1.85.1", "rustc"])
            self.assertNotIn("cargo", command)
            self.assertNotIn("--install", command)
        self.assertIn("--test-threads=1", commands["run"][0])
        # These fixtures are not Cargo integration-test entry points. The unit
        # file needs its allocator parent module; only direct rustc includes it.
        self.assertTrue((HERE / "tests/allocator/unit.rs").is_file())
        self.assertTrue((HERE / "tests/allocator/smoke.rs").is_file())
        self.assertFalse((HERE / "tests/allocator/main.rs").exists())
        self.assertFalse((HERE / "tests/bounded_allocator_unit.rs").exists())
        self.assertFalse((HERE / "tests/allocator_global_smoke.rs").exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS mapping API")
    def test_os_anonymous_zeroing_and_successful_unmap(self):
        # Independent OS observation only; it does not execute the Rust adapter.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.getpagesize.restype = ctypes.c_int
        page = libc.getpagesize()
        self.assertEqual(page, P["PAGE"])
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        libc.mmap.restype = ctypes.c_void_p
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        length = 2 * page
        address = libc.mmap(None, length, 3, 0x1002, -1, 0)
        self.assertNotEqual(address, ctypes.c_void_p(-1).value)
        self.assertIsNotNone(address)
        try:
            self.assertEqual(address % page, 0)
            data = (ctypes.c_ubyte * length).from_address(address)
            self.assertFalse(any(data))
            data[0] = data[length - 1] = 123
        finally:
            self.assertEqual(libc.munmap(address, length), 0)
        # macOS mincore(2) does not promise ENOMEM on an unmapped range. Its
        # residency bits are not a mapping-lifetime or RSS-release certificate.


if __name__ == "__main__":
    unittest.main()
