"""Exact algebra, bit-rounding and cost controls; no native/model dependencies."""

from __future__ import annotations

import itertools
import unittest
from collections import Counter
from dataclasses import replace
from fractions import Fraction

from scripts.probe_exact_adapter_frontier import (
    STATES, bits, certified_dot, compile_quotient, controls, decode_share, evaluate_compiled, f32, geometry,
    integer_tail_veto, lift_step, observables, ordered_block, ordered_block_oracle, quotient,
    quotient_ledger, rational_f32_bits, rounding_certificate, transition,
    source_program_oracle, transition_table,
)


class ExactRoundingTests(unittest.TestCase):
    def test_independent_rational_oracle_at_ties_and_subnormals(self):
        cases = [Fraction(0), Fraction(1), Fraction(-3, 2), Fraction(1, 2**149),
                 Fraction(1, 2**150), Fraction(3, 2**150), Fraction(-1, 2**150)]
        for n in (2**24, 2**25, 2**30):
            cases.extend(Fraction(s * (n + i)) for i in range(-5, 6) for s in (-1, 1))
        for x in cases:
            self.assertEqual(rational_f32_bits(x), bits(float(x)), x)

    def test_certificate_is_universal_over_tail_not_observed_token(self):
        for p in (0, 1, -1, 2**24, 2**24 + 1, 2**25 + 4, -2**25 - 4):
            for bound in range(9):
                word = rounding_certificate(p, bound)
                expected = {rational_f32_bits(Fraction(p + t)) for t in range(-bound, bound + 1)}
                self.assertEqual(word is not None, len(expected) == 1)
                if word is not None:
                    self.assertEqual(expected, {word})

    def test_skip_nonzero_tail_and_preserve_tie_fallback(self):
        self.assertEqual(certified_dot((2**25 + 4, 1), (1, -1), 1, 1),
                         (rational_f32_bits(Fraction(2**25 + 3)), True))
        for tail in (-1, 0, 1):
            word, hit = certified_dot((2**24, 1), (1, tail), 1, 1)
            self.assertFalse(hit)
            self.assertEqual(word, rational_f32_bits(Fraction(2**24 + tail)))

    def test_integer_dot_controls_exhaustive_small_values(self):
        for weights in itertools.product((-2, 0, 3), repeat=3):
            for x in itertools.product((-1, 0, 1), repeat=3):
                expected = rational_f32_bits(Fraction(sum(w * a for w, a in zip(weights, x))))
                for prefix in range(4):
                    self.assertEqual(certified_dot(weights, x, prefix, 1)[0], expected)

    def test_invalid_certificate_and_range_fail_closed(self):
        for args in ((1, -1), (1.5, 0), (2**54, 1)):
            with self.assertRaises(ValueError):
                rounding_certificate(*args)
        for args in (((1,), (2,), 0, 1), ((1,), (1,), 2, 1), ((), (), 0, 1)):
            with self.assertRaises(ValueError):
                certified_dot(*args)

    def test_integer_cast_certificate_cannot_certify_ordered_float_dot(self):
        # An exact integer sum of 1 is not the source's stepwise float sum of 0.
        a = float(2**24)
        stepwise = f32(f32(a + 1.0) - a)
        self.assertEqual(stepwise, 0.0)
        self.assertEqual(rounding_certificate(1, 0), bits(1.0))
        self.assertNotEqual(bits(stepwise), rounding_certificate(1, 0))

    def test_qwen_width_veto_follows_public_bounds_not_tensor_samples(self):
        for width, cell in ((896, 1), (4864, 8)):
            veto = integer_tail_veto(width)
            self.assertEqual(veto["max_rounding_cell_width_integer_units"], cell)
            self.assertEqual(veto["minimum_nonzero_tail_interval_width"], 254)
            self.assertTrue(veto["any_nonzero_tail_vetoed"])
        # Wide enough synthetic geometry can make the same bound inconclusive.
        self.assertFalse(integer_tail_veto(1 << 24)["any_nonzero_tail_vetoed"])


class OrderedQuotientTests(unittest.TestCase):
    def test_source_compiler_never_groups_from_runtime_activations(self):
        gates = ((1.0, 0.25), (-0.5, 2.0), (1.0, 0.25))
        ups = ((2.0, -0.25), (1.0, 0.5), (2.0, -0.25))
        down = (1.0, 0.5, -2.0)
        compiled = compile_quotient(gates, ups, down)
        self.assertEqual(compiled.representatives, (0, 1))
        self.assertEqual(compiled.inverse, (0, 1, 0))
        for inputs in itertools.product((-2.0, -0.25, 1.0, 4.0), repeat=2):
            self.assertEqual(evaluate_compiled(compiled, gates, ups, down, inputs),
                             source_program_oracle(gates, ups, down, inputs))
        # At zero input all outputs coincide, but source-unequal row stays separate.
        self.assertEqual(len(compiled.representatives), 2)

    def test_source_binding_and_forged_certificate_rejected(self):
        gates, ups, down = ((1.0,), (1.0,)), ((2.0,), (2.0,)), (1.0, 2.0)
        compiled = compile_quotient(gates, ups, down)
        with self.assertRaisesRegex(ValueError, "source binding"):
            evaluate_compiled(compiled, ((1.0,), (2.0,)), ups, down, (1.0,))
        with self.assertRaisesRegex(ValueError, "invalid quotient"):
            evaluate_compiled(replace(compiled, inverse=(1, 0)), gates, ups, down, (1.0,))

    def test_exact_bit_fanout_against_independent_rounding_oracle(self):
        for g, u in itertools.product((0.25, -1.5, 2**-60), (0.5, -2.0)):
            gates, ups, weights = (g, g, 1.0, g), (u, u, 0.25, u), (1.0, 2**24, -0.5, -2**24)
            expected = ordered_block_oracle(gates, ups, weights)
            self.assertEqual(ordered_block(gates, ups, weights, compact=True), expected)
            self.assertEqual(ordered_block(gates, ups, weights, compact=False), expected)

    def test_map_inverse_and_original_channel_order(self):
        signatures = ((7, 9), (1, 3), (7, 9), (7, 9), (1, 4))
        reps, inverse = quotient(signatures)
        self.assertEqual(reps, (0, 1, 4))
        self.assertEqual(tuple(signatures[reps[j]] for j in inverse), signatures)

    def test_position_scale_and_signed_zero_are_not_equal_signatures(self):
        # Synthetic DAG signature fields: row, scale bits, position dependency.
        signatures = ((5, bits(1.0), 0), (5, bits(1.0), 1), (5, bits(2.0), 0),
                      (5, bits(0.0), 0), (5, bits(-0.0), 0))
        self.assertEqual(len(quotient(signatures)[0]), 5)

    def test_rank_one_does_not_authorize_linear_fusion(self):
        # Original rank-one W2(W1*x), then algebraically fused rank-one product.
        gates, ups = (1.0,) * 3, (1.0,) * 3
        weights = (float(2**24), 1.0, -float(2**24))
        original = ordered_block(gates, ups, weights, compact=True)
        self.assertEqual(original, bits(0.0))
        self.assertNotEqual(original, bits(1.0))

    def test_no_duplicate_has_no_component_savings(self):
        ledger = quotient_ledger(original=8, compact=8, rows=157, one_use_bytes_per_lane=32)
        self.assertEqual(ledger["gross_saving_per_request_bytes"], 0)
        self.assertLess(ledger["component_net_saving_bytes"], 0)


class FiniteLiftTests(unittest.TestCase):
    def test_full_state_observable_map_is_injective(self):
        self.assertEqual(len({observables(s)[:4] for s in STATES}), len(STATES))
        table = transition_table()
        self.assertEqual(tuple(STATES[i] for i in table), tuple(transition(s) for s in STATES))

    def test_scatter_collision_requires_add_not_overwrite(self):
        self.assertEqual(lift_step([2, 3, 4], (1, 1, 0), 7), [4, 5, 0])

    def test_share_linearity_and_decoding_full_state(self):
        table = transition_table()
        for j in range(len(STATES)):
            left = [(3 * i + j) % 257 for i in range(len(STATES))]
            right = [(int(i == j) - a) % 257 for i, a in enumerate(left)]
            left, right = lift_step(left, table), lift_step(right, table)
            decoded = tuple((a + b) % 257 for a, b in zip(decode_share(left), decode_share(right)))
            self.assertEqual(decoded, observables(transition(STATES[j])))

    def test_one_worker_full_share_distribution_is_input_independent(self):
        # Exhaust full ideal mask distribution, not merely one marginal coordinate.
        masks = list(itertools.product(range(5), repeat=3))
        views = []
        for secret in range(3):
            views.append(Counter(tuple((int(i == secret) - mask[i]) % 5 for i in range(3))
                                 for mask in masks))
        self.assertTrue(all(view == views[0] for view in views))

    def test_omitting_position_or_rng_does_not_define_equivalent_state(self):
        self.assertNotEqual(transition((0, 0, 0, 0)), transition((0, 0, 1, 0)))
        self.assertNotEqual(observables((1, 0, 0, 0)), observables((1, 0, 0, 3)))


class LedgerTests(unittest.TestCase):
    def test_n_minus_one_decode_and_terminal_prefill_pruning(self):
        for n in (1, 8, 64):
            g = geometry(150, n)
            self.assertEqual(g["unpruned_gated_elements"], 24 * 4864 * (150 + n - 1))
            self.assertEqual(g["terminal_pruned_gated_elements"], 4864 * (23 * (150 + n - 1) + n))
            self.assertEqual(g["cut_payload_6_ring32_words_per_element_bytes"],
                             24 * g["terminal_pruned_gated_elements"])

    def test_both_workers_preprocessing_artifacts_and_amortization(self):
        one = quotient_ledger(original=8, compact=2, rows=157, one_use_bytes_per_lane=32)
        two = quotient_ledger(original=8, compact=2, rows=157, one_use_bytes_per_lane=32, requests=2)
        self.assertEqual(sum(one["setup_links_bytes"].values()), 68)
        self.assertEqual(one["repeated_links_bytes"]["preparation_to_both_workers_material"], 157 * 2 * 32)
        self.assertEqual(one["component_net_saving_bytes"], 157 * 6 * (24 + 32) - 68)
        self.assertEqual(two["component_net_saving_bytes"], 2 * 157 * 6 * 56 - 68)
        self.assertFalse(two["whole_response_promotion"])
        self.assertTrue(two["unpriced"])

    def test_all_controls_and_no_generic_positive_claim(self):
        report = controls()
        self.assertEqual(report["lift_steps_full_state_logits_selection_rng_exact"], len(STATES) * 6)
        self.assertTrue(report["nonzero_tail_skipped"])
        self.assertTrue(report["boundary_forces_exact_fallback"])


if __name__ == "__main__":
    unittest.main()
