"""Bounded, offline, standard-library conformance; only PUBLIC fixtures."""
import hashlib
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from aes_reference import encrypt_block, expanded_key, source_fill
from fixtures import build_vectors, summary
from oracle import (Cursor, Interval, MAX_AES_WORK, MAX_LANES, MAX_OPERATIONS,
                    PROFILE, Q18, Rejected, TRUNC_ONLY_PROFILE, U64_MAX,
                    admit_product, canonical_bytes, digest, distribution,
                    literal_signed, local_mul, multiply_batch, product_hull,
                    rounded_hull, schedule, trunc_source, zero_share_schedule)

HERE = Path(__file__).resolve().parent
GAP = 1 << 62


class Intervals(unittest.TestCase):
    def test_cartesian_interval_iff_exhaustive_2025_boxes(self):
        intervals = [Interval(lo, hi) for lo in range(-4, 5) for hi in range(lo, 5)]
        count = 0
        for x, y in itertools.product(intervals, repeat=2):
            products = [a*b for a in range(x.lo, x.hi + 1) for b in range(y.lo, y.hi + 1)]
            exact = Interval(min(products), max(products))
            self.assertEqual(product_hull(x, y, 4), exact)
            safe = all(-4 <= p < 4 for p in products)
            if safe:
                self.assertEqual(admit_product(x, y, 4), exact)
                outputs = [(p + r) // 4 for p in products for r in range(4)]
                self.assertEqual(rounded_hull(exact, 4, 2), Interval(min(outputs), max(outputs)))
            else:
                with self.assertRaises(Rejected):
                    admit_product(x, y, 4)
            count += 1
        self.assertEqual(count, 2025)

    def test_asymmetric_endpoints_and_full_signed_input_domain(self):
        self.assertEqual(admit_product(Interval(1 << 31, 1 << 31), Interval(-(1 << 31), -(1 << 31))), Interval(-GAP, -GAP))
        with self.assertRaises(Rejected):
            admit_product(Interval(1 << 31, 1 << 31), Interval(1 << 31, 1 << 31))
        full = Interval(-(1 << 63), (1 << 63) - 1)
        self.assertEqual(admit_product(Interval(0, 0), full), Interval(0, 0))
        self.assertEqual(multiply_batch([0, 0], [full.lo, full.hi], [Interval(0, 0)]*2, [full]*2, [0, U64_MAX]), (0, 0))
        self.assertEqual(rounded_hull(Interval(-GAP, GAP-1)), Interval(-(1 << 44), 1 << 44))

    def test_reject_unwrapped_overflow_even_when_residue_looks_safe(self):
        for x, y in [(1 << 32, 1 << 32), (-(1 << 32), 1 << 32), ((1 << 63)-1, 4), (-(1 << 63), 2)]:
            wrapped = x*y % (1 << 64)
            signed = wrapped - (1 << 64) if wrapped >= 1 << 63 else wrapped
            self.assertTrue(-GAP <= signed < GAP)  # the incorrect post-wrap test would pass
            with self.assertRaises(Rejected):
                multiply_batch([x], [y], [Interval(x, x)], [Interval(y, y)], [0])
        for p in (-GAP-1, GAP):
            with self.assertRaises(Rejected):
                trunc_source(p, 0)

    def test_whole_batch_public_admission_precedes_any_truncation(self):
        with patch("oracle.trunc_source", side_effect=AssertionError("evaluated before admission")):
            with self.assertRaises(Rejected):
                multiply_batch([1, 0], [1, 0], [Interval(1, 1), Interval(0, 1 << 32)],
                               [Interval(1, 1), Interval(0, 1 << 32)], [0, 0])

    def test_correlation_and_accumulator_are_separate_proof_obligations(self):
        # A relational proof Y=-X could admit these actual points; an independent
        # public box contains +GAP as well and cannot establish that relation.
        interval = Interval(-(1 << 31), 1 << 31)
        with self.assertRaises(Rejected):
            admit_product(interval, interval)
        self.assertEqual(literal_signed(-(1 << 31)*(1 << 31), 0), -(1 << 44))
        admit_product(Interval(1, 1), Interval(GAP//2, GAP//2))
        with self.assertRaises(Rejected):
            trunc_source(GAP//2 + GAP//2, 0)  # individually safe products do not certify a sum

    def test_bounded_batch_and_invalid_fixture_metadata(self):
        one = Interval(1, 1)
        self.assertEqual(len(multiply_batch([1]*MAX_LANES, [1]*MAX_LANES, [one]*MAX_LANES,
                                           [one]*MAX_LANES, [0]*MAX_LANES)), MAX_LANES)
        for count in (0, MAX_LANES + 1):
            with self.assertRaises(Rejected):
                multiply_batch([1]*count, [1]*count, [one]*count, [one]*count, [0]*count)
        for args in [([1], [], [one], [one], [0]), ([2], [1], [one], [one], [0]),
                     ([1], [1], [one], [one], [-1])]:
            with self.assertRaises(Rejected):
                multiply_batch(*args)
        with self.assertRaises(Rejected):
            product_hull(Interval(0, 1 << 63), one)
        with self.assertRaises(Rejected):
            Interval(2, 1)


class Arithmetic(unittest.TestCase):
    def test_all_247552_small_ring_mask_equations_and_exact_laws(self):
        count = 0
        for width in range(4, 9):
            gap = 1 << (width - 2)
            for shift in range(1, width - 1):
                for value in range(-gap, gap):
                    floor, upper, numerator, denominator = distribution(value, width, shift)
                    uppers = 0
                    for mask in range(1 << width):
                        got = trunc_source(value, mask, width, shift)
                        self.assertEqual(got, (value + mask % (1 << shift)) // (1 << shift))
                        self.assertIn(got, (floor, upper))
                        uppers += got > floor
                        count += 1
                    self.assertEqual(uppers, numerator * ((1 << width) // denominator))
        self.assertEqual(count, 247552)

    def test_all_q18_low_residues_with_high_mask_wrap(self):
        upper_count = 0
        for low in range(Q18):
            mask = (U64_MAX ^ (Q18 - 1)) | low
            result = trunc_source(-9, mask)
            self.assertEqual(result, literal_signed(-9, mask))
            upper_count += result == 0
        self.assertEqual(upper_count, Q18 - 9)

    def test_literal_rounding_controls_and_q18_endpoints(self):
        # Literals do not come from the transcript generator or source formula.
        for product, mask, expected in [(-9, 0, -1), (-9, 9, 0),
                                        (GAP-1, 0, (1 << 44)-1), (GAP-1, 1, 1 << 44),
                                        (-GAP, U64_MAX, -(1 << 44)),
                                        (Q18*Q18, U64_MAX, Q18)]:
            self.assertEqual(trunc_source(product, mask), expected)
            self.assertEqual(literal_signed(product, mask), expected)
        with self.assertRaises(Rejected):
            trunc_source(0, 1 << 64)

    def test_zero_sharing_and_all_mod8_input_share_splits(self):
        triples = tuple(itertools.product(range(8), repeat=3))
        for edges in triples:
            self.assertEqual(sum((edges[i] - edges[(i+1) % 3]) % 8 for i in range(3)) % 8, 0)
        count = 0
        edges = (1, 5, 7)
        for xs in triples:
            for ys in triples:
                total = sum(local_mul((xs[i], xs[(i+1) % 3]), (ys[i], ys[(i+1) % 3]),
                                      edges[i], edges[(i+1) % 3], 3) for i in range(3)) % 8
                self.assertEqual(total, sum(xs)*sum(ys) % 8)
                count += 1
        self.assertEqual(count, 262144)
        # Omitting the cross terms is demonstrably wrong.
        xs, ys = (1, 0, 0), (0, 1, 0)
        self.assertNotEqual(sum(a*b for a, b in zip(xs, ys)) % 8, sum(xs)*sum(ys) % 8)


class SourceSchedule(unittest.TestCase):
    def test_all_lane_sizes_and_roles_exact_work_routes_and_offsets(self):
        self.assertNotEqual(PROFILE, TRUNC_ONLY_PROFILE)
        for lanes in range(1, MAX_LANES + 1):
            q = (lanes + 1) // 2
            for pivot in range(3):
                before = Cursor(7, 11, (3, 4, 5), (19, 23, 29), 2)
                plan, after = schedule(before, lanes, pivot)
                self.assertEqual([plan["mul_zero"]["start"], plan["trunc_mask_start"], plan["trunc_output_start"], after.pair],
                                 [11, 11+q, 11+2*q, 11+3*q])
                self.assertEqual(plan["mul_routes"], [[0, 2], [1, 0], [2, 1]])
                self.assertEqual(after.public, 8)
                for role in range(3):
                    is_helper = role == (pivot + 2) % 3
                    # Count actual AES invocations, not just the shared cursor.
                    counts = [q, q, 1, q, q] + ([q, q, lanes] if is_helper else [])
                    self.assertEqual(after.work[role] - before.work[role], sum(counts))
                    self.assertEqual(after.private[role] - before.private[role], lanes if is_helper else 0)
                self.assertEqual(plan["wire_payload_words"], 11*lanes)

    def test_exhaustion_invalid_pivot_and_no_hidden_zero_fill(self):
        for lanes in (0, MAX_LANES + 1):
            with self.assertRaises(Rejected):
                schedule(Cursor(), lanes, 0)
        for cursor in (Cursor(public=U64_MAX), Cursor(pair=U64_MAX-2),
                       Cursor(private=(0, 0, U64_MAX)), Cursor(work=(MAX_AES_WORK, 0, 0)),
                       Cursor(sequence=MAX_OPERATIONS)):
            with self.assertRaises(Rejected):
                schedule(cursor, 1, 0)
        for pivot in (U64_MAX, U64_MAX-1):
            with self.assertRaises(Rejected):
                schedule(Cursor(), 1, pivot)
        self.assertEqual(schedule(Cursor(), 1, U64_MAX-2)[0]["roles"]["a"], 1)
        zero = zero_share_schedule(3, 0)
        canonical, _ = schedule(Cursor(), 3, 0)
        with_extra_zero, _ = schedule(Cursor(pair=zero["end"]), 3, 0)
        self.assertEqual(canonical["mul_zero"]["end"], 2)
        self.assertEqual(canonical["pair_end"], 6)
        self.assertEqual(with_extra_zero["pair_end"], 8)  # different explicit graph, never silently inserted
        self.assertNotEqual(digest(canonical), digest(with_extra_zero))

    def test_aes_fips_kat_reset_discard_and_within_fill_carry(self):
        self.assertEqual(encrypt_block(bytes.fromhex("00112233445566778899aabbccddeeff"),
                                       expanded_key(bytes(range(16)))).hex(), "69c4e0d86a7b0430d8cdb78070b4c55a")
        self.assertEqual(source_fill(bytes(16), 0, 2), (0x3b2c8aefd44be966, 0x2e2b34ca59fa4c88))
        self.assertEqual(source_fill(bytes(16), 1, 1), (0xf06f1de916187147,))
        long_fill = source_fill(bytes(16), 0, 516)
        self.assertEqual(hashlib.sha256(b"".join(w.to_bytes(8, "little") for w in long_fill)).hexdigest(),
                         "b844e6a17d006cd080e7adcdf6d82be099aead7190912154e5358900dc3bdc04")
        for cursor in (255, 256):
            self.assertNotEqual(long_fill[2*cursor:2*cursor+2], source_fill(bytes(16), cursor, 2))
        for start, words in [(U64_MAX, 1), (0, 0), (0, 2*MAX_LANES+1)]:
            with self.assertRaises(Rejected):
                source_fill(bytes(16), start, words)

    def test_frozen_transcripts_and_actual_replicated_direction(self):
        vectors = build_vectors()
        frozen = json.loads((HERE / "vectors.json").read_text())
        self.assertEqual(frozen, json.loads(canonical_bytes(summary(vectors))))
        for case in vectors["cases"]:
            outputs, components = case["output_pairs_hex"], case["mul_components_hex"]
            for i in range(3):
                self.assertEqual(outputs[i][1], outputs[(i+1) % 3][0])
            frames = case["frames"]
            self.assertEqual(len(frames), 9)
            self.assertEqual(sum(len(f["words_hex"]) for f in frames), 11*case["plan"]["lanes"])
            rotated = [None]*3
            for frame in frames[:3]:
                rotated[frame["to"]] = (components[frame["to"]], frame["words_hex"])
            for i in range(3):
                self.assertEqual(rotated[i][1], rotated[(i+1) % 3][0])
            self.assertTrue(any(components[(i+2) % 3] != components[(i+1) % 3] for i in range(3)))


if __name__ == "__main__":
    unittest.main()
