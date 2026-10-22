"""Source arithmetic and lifetime failures; no HE, native build, or large buffers."""

from pathlib import Path
import hashlib
import json
import runpy
import unittest
from unittest.mock import patch


S = runpy.run_path(str(Path(__file__).resolve().parents[1] / "staged_admission.py"))


class StagedAdmissionTests(unittest.TestCase):
    def test_solver_source_frontier_never_claims_effective_closure(self):
        audit = S["solver_dependency_audit"]()
        self.assertEqual(audit["reviewed_source_packages"], ["cfg-if@1.0.4"])
        self.assertEqual(len(audit["unreviewed_package_sources"]), 6)
        self.assertIn("dashu-float@0.6.2", audit["unreviewed_package_sources"])
        self.assertEqual(audit["verified_source_hash_count"], 0)
        self.assertFalse(audit["effective_consumer_closure_verified"])
        self.assertIsNone(audit["arithmetic_allocation_upper_bound_bytes"])
        self.assertEqual(S["report"]()["solver_dependency_audit"], audit)
        self.assertTrue(any("optional cfg-if core" in r for r in audit["remaining_requirements"]))

    def test_dependency_hash_verification_is_explicit_and_fails_closed(self):
        # Tiny synthetic bytes test verification mechanics, not real cache state.
        manifest = json.loads((Path(__file__).resolve().parents[1] /
                               "staged_admission.sources.json").read_text())
        fixture = b"small synthetic source\n"
        for source in manifest["solver_dependency_source_audit"]["reviewed_source_leaves"][0]["sources"]:
            source["sha256"] = hashlib.sha256(fixture).hexdigest()
        with patch.object(Path, "read_text", return_value=json.dumps(manifest)):
            with patch.object(Path, "read_bytes", return_value=fixture) as read:
                audit = S["solver_dependency_audit"]("existing-registry")
                self.assertEqual(read.call_count, 2)
                self.assertEqual(audit["verified_source_hash_count"], 2)
                self.assertFalse(audit["effective_consumer_closure_verified"])
                self.assertIsNone(audit["arithmetic_allocation_upper_bound_bytes"])
            with patch.object(Path, "read_bytes", return_value=b"drift"):
                with self.assertRaisesRegex(ValueError, "dependency source drift"):
                    S["solver_dependency_audit"]("existing-registry")
            with patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError):
                    S["solver_dependency_audit"]("missing-registry")

    def test_cfg_if_review_matches_recorded_lock_identity_not_caller_bounds(self):
        here = Path(__file__).resolve().parents[1]
        manifest = json.loads((here / "staged_admission.sources.json").read_text())
        contract = json.loads((here / "bounded-allocator-contract.json").read_text())
        leaf, = manifest["solver_dependency_source_audit"]["reviewed_source_leaves"]
        locked = next(p for p in contract["solver_dependency_plan"]["upstream_solver_closure"]
                      if p["name"] == leaf["name"])
        self.assertEqual(leaf["version"], locked["version"])
        self.assertEqual(leaf["registry_checksum"], locked["checksum"])
        self.assertEqual(leaf["runtime_buffer_owners_defined_by_macro_itself"], 0)
        self.assertEqual(len(leaf["feature_obligations"]), 3)
        self.assertFalse(manifest["solver_dependency_source_audit"]["effective_consumer_closure_verified"])

    def test_matrix_query_covers_streamed_rhs_and_all_siblings(self):
        q = S["matrix_query_certificate"]()
        self.assertEqual(q["prepared_core_query_bytes"], 729_314_880)
        self.assertLess(max(q["other_branch_bounds_bytes"].values()), q["prepared_core_query_bytes"])
        # Previously only the lazy branch plus assign copy was accounted for.
        self.assertEqual(q["matrix_query_bytes"] - 758_675_008, 6_291_456)
        self.assertEqual(q["matrix_query_bytes"], 764_966_464)

    def test_retaining_owners_is_not_a_solution_to_reclamation(self):
        r = S["report"]()
        self.assertEqual(r["matrix_policy_subset_bytes"], 8_324_411_712)
        self.assertEqual(r["residual_after_raw_upper_bytes"], 23_301_824)
        self.assertEqual(r["live_first_baby_cache_policy_excess_bytes"], 204_239_168)
        self.assertEqual(r["live_separate_tensor_policy_excess_bytes"], 703_361_344)
        self.assertGreater(r["distinct_role_persistent_keys_plus_guard_bytes"], 8 << 30)
        # Enough headroom for at most eleven 2-MiB blocks, even with every other
        # omitted owner set to zero. This is a necessary condition, not admission.
        self.assertEqual(r["maximum_retained_vmp_blocks_after_raw_before_other_owners"], 11)
        self.assertFalse(r["admitted"])
        self.assertIsNone(r["complete_owner_upper_bound_bytes"])
        self.assertIsNone(r["complete_setup_upper_bound_bytes"])
        self.assertIsNone(r["measured_peak_bytes"])
        self.assertTrue(any("candidate exists" in blocker for blocker in r["blockers"]))
        self.assertFalse(any("no public automorphism-to-tensor" in blocker for blocker in r["blockers"]))

    def test_private_vmp_peak_is_not_cumulative_charge(self):
        ledger = S["RetentionLedger"]()
        # 462 tiny Python records stand for 462 sequential 2-MiB allocations;
        # no byte arrays of those sizes are constructed.
        for index in range(7 * 66):
            ledger.allocate(index, "vmp_private_u32", 2 << 20)
            ledger.retire(index)
        self.assertEqual(ledger.live_bytes, 0)
        self.assertEqual(ledger.peak_live_bytes, 2 << 20)
        self.assertEqual(ledger.charged_bytes, 968_884_224)

    def test_same_role_refill_preserves_payload_and_requires_finished_borrows(self):
        ledger = S["RetentionLedger"]()
        ledger.allocate("key0", "prepared_automorphism", 968_884_224)
        with ledger.borrow("key0"):
            with self.assertRaisesRegex(ValueError, "outstanding key views"):
                ledger.refill("key0", "prepared_automorphism", 968_884_224)
            with self.assertRaisesRegex(ValueError, "outstanding key views"):
                ledger.retire("key0")
        ledger.refill("key0", "prepared_automorphism", 968_884_224)
        self.assertEqual(ledger.charged_bytes, 968_884_224)
        with self.assertRaisesRegex(ValueError, "same-role"):
            ledger.refill("key0", "prepared_tensor", 968_884_224)
        with self.assertRaisesRegex(ValueError, "same-shape"):
            ledger.refill("key0", "prepared_automorphism", 12_582_912)
        ledger.retire("key0")
        with self.assertRaisesRegex(ValueError, "retired"):
            ledger.refill("key0", "prepared_automorphism", 968_884_224)
        self.assertEqual(ledger.charged_bytes, 968_884_224)

    def test_unwinding_borrow_and_invalid_sizes_cannot_create_release_credit(self):
        ledger = S["RetentionLedger"]()
        for size in (0, -1, True, 0.5, None):
            with self.assertRaises(ValueError):
                ledger.allocate("bad", "raw", size)
        ledger.allocate("one", "raw", 10)
        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            with ledger.borrow("one"):
                raise RuntimeError("operation failed")
        ledger.retire("one")
        ledger.allocate("two", "raw", 10)
        self.assertEqual(ledger.live_bytes, 10)
        self.assertEqual(ledger.charged_bytes, 20)
        with self.assertRaises(ValueError):
            ledger.allocate("one", "raw", 10)


if __name__ == "__main__":
    unittest.main()
