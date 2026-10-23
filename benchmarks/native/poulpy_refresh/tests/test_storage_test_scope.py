"""The tiny storage-test exception does not authorize HE execution."""

import json
from pathlib import Path
import unittest


class StorageTestScope(unittest.TestCase):
    def test_only_existing_serial_storage_target_is_authorized(self):
        contract = json.loads(
            (Path(__file__).resolve().parents[1] / "contract.json").read_text()
        )
        build = contract["build"]
        scope = build["storage_test_scope"]
        self.assertEqual(scope["decision"], "D35")
        self.assertEqual(scope["target"], "staged_storage")
        self.assertEqual(scope["expected_tests"], 31)
        self.assertEqual(scope["test_threads"], 1)
        self.assertIn("fresh D33 admission", scope["host_admission"])
        self.assertIs(scope["he_keys_ciphertexts_solver_refresh"], False)
        self.assertIs(scope["rng_pin_changes_or_backend_adoption"], False)
        self.assertEqual(build["jobs"], 1)
        self.assertEqual(build["compile_minimum_available_bytes"], 8 << 30)
        self.assertEqual(build["compile_required_pressure"], "normal")
        self.assertEqual(build["additional_swap_bytes"], 0)
        self.assertEqual(build["additional_swapout_pages"], 0)


if __name__ == "__main__":
    unittest.main()
