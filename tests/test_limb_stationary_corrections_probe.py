"""Independent small correctness, deterministic work and one-use ownership tests."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/probe_limb_stationary_corrections.py'
SPEC = importlib.util.spec_from_file_location('limb_stationary_corrections', PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class LimbStationaryTests(unittest.TestCase):
    def test_all_rings_tails_pages_against_python_oracle(self):
        evidence = probe.correctness()
        self.assertEqual(evidence['cases_passed'], 15)
        self.assertEqual(evidence['coherent_adversarial_cases_passed'], 9)

    def test_work_counts_and_extra_slab_cost(self):
        base = probe.preflight(32, 4096, 1024, 32, stationary=False)
        stationary = probe.preflight(32, 4096, 1024, 32)
        self.assertEqual(base['mask_pack_elements'], 4 * stationary['mask_pack_elements'])
        self.assertEqual(stationary['mask_pack_elements'], 32 * 1024 * 4)
        self.assertEqual(stationary['native_bytes'] - base['native_bytes'], 4 * 3 * 32 * 1023)
        for key in ('weight_pack_elements', 'gemm_calls', 'float_mac_products',
                    'integer_reconstruction_elements'):
            self.assertEqual(base[key], stationary[key])

    def test_page_tile_reuse_not_global_cache(self):
        base = probe.preflight(2, 2051, 17, 24, stationary=False, page_rows=1025)
        stationary = probe.preflight(2, 2051, 17, 24, page_rows=1025)
        self.assertEqual(stationary['pages'], 3)
        self.assertEqual(base['tiles'], 5)
        self.assertEqual(stationary['mask_pack_elements'], 2 * 17 * 3 * 3)
        self.assertEqual(base['mask_pack_elements'], 2 * 17 * 3 * 5)
        one_tile = probe.preflight(2, 2051, 17, 24, page_rows=512)
        self.assertEqual(one_tile['pages'], one_tile['tiles'])
        self.assertFalse(one_tile['useful_within_page_reuse'])
        baseline = probe.preflight(2, 2051, 17, 24, page_rows=512, stationary=False)
        self.assertEqual(one_tile['mask_pack_elements'], baseline['mask_pack_elements'])
        self.assertGreater(one_tile['native_bytes'], baseline['native_bytes'])

    def test_one_use_and_failure_spends_mask(self):
        w, x = probe.synthetic_case(2, 3, 1, 32)
        ticket = probe.OneUseMask(x)
        probe.reference(w, ticket, 32, stationary=True)
        with self.assertRaisesRegex(ValueError, 'already consumed'):
            probe.reference(w, ticket, 32, stationary=True)
        ticket = probe.OneUseMask(x)
        with patch.object(probe.np, 'zeros', side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                probe.reference(w, ticket, 32, stationary=True)
        self.assertTrue(ticket.spent)

    def test_invalid_domain_rejected_before_output(self):
        w, x = probe.synthetic_case(2, 3, 1, 16)
        x[0, 0] = 65536
        ticket = probe.OneUseMask(x)
        with patch.object(probe.np, 'zeros', side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError, 'outside ring'):
                probe.reference(w, ticket, 16, stationary=True)
        self.assertFalse(ticket.spent)
        for args in [(0, 2, 3, 32), (1, 2, 3, 17), (4097, 1, 1, 32),
                     (2, 3, probe.MAX_ELEMENTS, 32)]:
            with self.assertRaises(ValueError):
                probe.preflight(*args)

    def test_existing_limits_not_relaxed(self):
        # Valid original one-slab shape; extra digit slabs must trigger fallback.
        self.assertLessEqual(probe.preflight(4096, 1024, 1023, 32,
                                            stationary=False)['native_bytes'], probe.NATIVE_CAP)
        with self.assertRaisesRegex(ValueError, '64 MiB'):
            probe.preflight(4096, 1024, 1023, 32)
        with self.assertRaisesRegex(ValueError, '224 MiB'):
            probe.preflight(1, 1, 1, 32, page_transient_bytes=200 << 20)

    def test_random_signed_weights_and_masks(self):
        rng = np.random.default_rng(6006)
        for bits in (16, 24, 32):
            w = rng.integers(-128, 128, (13, 29), dtype=np.int8)
            x = rng.integers(0, 1 << bits, (3, 29), dtype=np.uint32)
            expected = probe.oracle(w, x, bits)
            for stationary in (False, True):
                actual, _ = probe.reference(w, probe.OneUseMask(x), bits, stationary=stationary)
                self.assertEqual(actual.tolist(), expected)


if __name__ == '__main__':
    unittest.main()
