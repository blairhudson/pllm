"""Borrow-safe mapped-owner admission; integer models only, never HE execution."""

from pathlib import Path
import runpy
import unittest


P = runpy.run_path(str(Path(__file__).resolve().parents[1] / "allocator_plan.py"))


class MappingLifetimeTests(unittest.TestCase):
    def test_borrowed_mapping_cannot_retire_or_spend_release_credit(self):
        charge = P["buffer_charge"](128)
        ledger = P["MappingLedger"](ceiling=charge)
        ledger.allocate("key", 128)
        with ledger.borrow("key"):
            with self.assertRaisesRegex(ValueError, "outstanding key views"):
                ledger.retire("key")
            with self.assertRaisesRegex(ValueError, "still live"):
                ledger.unmap_result("key", succeeded=True)
            with self.assertRaisesRegex(ValueError, "rejected"):
                ledger.allocate("replacement", 128)
            self.assertFalse(ledger.mappings["key"].retired)
            self.assertEqual(ledger.charged, charge)
        ledger.retire("key")
        ledger.unmap_result("key", succeeded=True)
        ledger.allocate("replacement", 128)
        self.assertEqual(ledger.charged, charge)

    def test_every_realloc_requires_finished_views_even_same_size(self):
        charge = P["buffer_charge"](128)
        ledger = P["MappingLedger"](ceiling=2 * charge)
        ledger.allocate("key", 128)
        with ledger.borrow("key"):
            for size in (64, 128, 256):
                with self.subTest(size=size):
                    with self.assertRaisesRegex(ValueError, "outstanding key views"):
                        ledger.realloc("key", "replacement", size, old_unmap_succeeded=True)
                    self.assertEqual(set(ledger.mappings), {"key"})
                    self.assertFalse(ledger.mappings["key"].retired)
                    self.assertEqual((ledger.charged, ledger.peak_charged), (charge, charge))
        self.assertEqual(ledger.realloc("key", "same", 128, old_unmap_succeeded=True), "key")
        ledger.realloc("key", "replacement", 256, old_unmap_succeeded=True)
        self.assertEqual((ledger.charged, ledger.peak_charged), (charge, 2 * charge))

    def test_nested_views_and_exception_cleanup_share_retention_contract(self):
        for kind in ("mapped", "retained"):
            with self.subTest(kind=kind):
                if kind == "mapped":
                    ledger = P["MappingLedger"]()
                    ledger.allocate("key", 128)
                else:
                    ledger = P["S"]["RetentionLedger"]()
                    ledger.allocate("key", "prepared_automorphism", 128)
                with ledger.borrow("key"):
                    with self.assertRaisesRegex(RuntimeError, "phase failed"):
                        with ledger.borrow("key"):
                            raise RuntimeError("phase failed")
                    # Exiting inner view must not end the outer view.
                    with self.assertRaisesRegex(ValueError, "outstanding key views"):
                        ledger.retire("key")
                ledger.retire("key")
                with self.assertRaisesRegex(ValueError, "retired"):
                    with ledger.borrow("key"):
                        self.fail("retired storage exposed a new view")

    def test_full_candidate_cache_cannot_retire_while_one_leaf_is_borrowed(self):
        ledger = P["MappingLedger"]()
        for name, capacity, alignment in P["dft_phase_owners"](28):
            ledger.allocate(name, capacity, alignment)
        before = ledger.charged
        with ledger.borrow("baby/0"):
            with self.assertRaisesRegex(ValueError, "outstanding key views"):
                ledger.retire("baby/0")
            self.assertEqual(ledger.charged, before)
        # Lifetime closure alone earns no mapping-release credit.
        ledger.retire("baby/0")
        self.assertEqual(ledger.charged, before)
        charge = ledger.mappings["baby/0"].charge
        ledger.unmap_result("baby/0", succeeded=True)
        self.assertEqual(ledger.charged, before - charge)
        self.assertFalse(P["plan"]()["admitted"])
        self.assertIsNone(P["plan"]()["full_owner_completion_bound_bytes"])

    def test_deferred_view_cannot_rebind_to_replacement_mapping(self):
        ledger = P["MappingLedger"]()
        ledger.allocate("key", 128)
        pending_view = ledger.borrow("key")
        ledger.retire("key")
        ledger.unmap_result("key", succeeded=True)
        ledger.allocate("key", 256)
        with self.assertRaisesRegex(ValueError, "retired"):
            with pending_view:
                self.fail("old view rebound to new storage")
        # The stale view neither borrows nor retires the replacement owner.
        with ledger.borrow("key"):
            with self.assertRaisesRegex(ValueError, "outstanding key views"):
                ledger.retire("key")
        ledger.retire("key")
        ledger.unmap_result("key", succeeded=True)
        self.assertEqual((ledger.charged, len(ledger.mappings)), (0, 0))


if __name__ == "__main__":
    unittest.main()
