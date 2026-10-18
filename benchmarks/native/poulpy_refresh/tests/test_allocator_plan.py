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
