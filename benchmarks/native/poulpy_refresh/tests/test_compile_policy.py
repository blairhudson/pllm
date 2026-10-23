"""Compile-only approval must not relax the encrypted-execution contract."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CompilePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((ROOT / "contract.json").read_text())

    def test_framework_admission_keeps_eight_gib_and_single_job(self):
        build = self.contract["build"]
        self.assertEqual(build["jobs"], 1)
        self.assertEqual(build["compile_minimum_available_bytes"], 8 << 30)
        self.assertEqual(build["compile_required_pressure"], "normal")
        self.assertEqual(
            build["compile_host_memory_source"],
            "pllm.runtime.benchmark_memory.host_memory",
        )
        self.assertEqual(
            build["compile_owner_ceiling_bytes"] + build["host_reserve_bytes"],
            build["compile_minimum_available_bytes"],
        )

    def test_qualification_and_execution_remain_blocked(self):
        self.assertEqual(self.contract["status"], "blocked_before_keygen")
        for key in (
            "source_method_mapping",
            "production_security_qualified",
            "paper_numerics_qualified",
            "model_inference_qualified",
        ):
            self.assertIs(self.contract[key], False)
        self.assertIn("--no-run", self.contract["build"]["tests"])
        self.assertFalse(self.contract["build"]["global_toolchain_changes"])
        self.assertFalse(self.contract["build"]["root_workspace_changes"])

    def test_resource_and_dependency_limits_stay_frozen(self):
        build = self.contract["build"]
        self.assertEqual(build["compile_owner_ceiling_bytes"], 4 << 30)
        self.assertEqual(build["host_reserve_bytes"], 4 << 30)
        self.assertEqual(build["disk_reserve_bytes"], 8 << 30)
        self.assertEqual(build["additional_swap_bytes"], 0)
        self.assertEqual(build["additional_swapout_pages"], 0)
        self.assertEqual(self.contract["dependency"]["version"], "0.9.0")
        self.assertEqual(
            self.contract["dependency"]["nightly"], "nightly-2026-05-14"
        )


if __name__ == "__main__":
    unittest.main()
