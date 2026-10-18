"""Source-specific scalar conformance only. All code under test is independent.

Run from repository root with Python's standard-library unittest discovery.
Author files are read as inert hash evidence, never imported or executed.
"""

from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import freeze
import math_oracle as oracle
import reference as ref

FROZEN = json.loads((ref.HERE / "vectors.json").read_bytes())


def toy(bits, boundaries=None, booleans=()):
    boundaries = (0, 1 << bits) if boundaries is None else tuple(boundaries)
    return ref.Descriptor(ref.ArithmeticType(bits, 0), boundaries,
                          tuple(ref.AffinePiece((5 * i + 2) % (1 << bits), (3 * i + 1) % (1 << bits), booleans)
                                for i in range(len(boundaries) - 1)))


class SourceAndTable(unittest.TestCase):
    def test_frozen_vectors_and_derived_descriptor_bytes(self):
        self.assertEqual(freeze.vectors(), FROZEN)
        descriptor = ref.gelu_descriptor()
        self.assertEqual(ref.parse_descriptor(descriptor.canonical_bytes()), descriptor)

    def test_table_floors_have_rational_enclosures_and_independent_erfc_check(self):
        frozen = ref.load_table()
        self.assertEqual(oracle.gelu_table(128), frozen)
        self.assertEqual(oracle.gelu_table(160), frozen)
        # A separate libm formula is a cross-check, not the authoritative floor.
        alternate = tuple(math.floor(2048 * (i / 64) * math.erfc((i / 64) / math.sqrt(2))) for i in range(256))
        self.assertEqual(alternate, frozen)
        self.assertEqual(ref.table_digest(frozen), FROZEN["table_sha256_u64le"])

    def test_local_pdf_source_hashes(self):
        root = ref.HERE.parents[2]
        lock = json.loads((ref.HERE / "source-lock.json").read_bytes())
        for source in lock["sources"]:
            if "path" not in source:
                continue  # remote source was independently fetched/hashed at freeze, not executed
            path = root / source["path"]
            if not path.exists():
                self.skipTest("ignored source PDF unavailable; fetch and verify the pinned bytes separately")
            data = path.read_bytes()
            self.assertEqual(len(data), source["bytes"])
            self.assertEqual(sha256(data).hexdigest(), source["sha256"])

    def test_approved_decision_section_snapshots(self):
        root = ref.HERE.parents[2]
        lock = json.loads((ref.HERE / "source-lock.json").read_bytes())
        for item in lock["decision_snapshots"]:
            text = (root / item["path"]).read_text()
            section = re.search(r"^## " + item["heading"] + r"\..*?(?=^## |\Z)", text, re.M | re.S).group(0)
            self.assertEqual(sha256(section.encode()).hexdigest(), item["sha256_section_utf8"])


class DescriptorAndEncoding(unittest.TestCase):
    def test_signed_external_ties_and_neighbors(self):
        word = ref.ArithmeticType(50, 12)
        for item in FROZEN["external_encoding_ties"]:
            value = Fraction(item["numerator"], item["denominator"])
            self.assertEqual(ref.signed(ref.encode_external(value, word), 50), item["expected_signed_q12"])
        for numerator in range(-25, 26):
            for epsilon in (-1, 0, 1):
                value = Fraction(numerator, 8192) + Fraction(epsilon, 1 << 30)
                self.assertEqual(ref.signed(ref.encode_external(value, word), 50), round(value * 4096))
        self.assertEqual(ref.encode_external(Fraction(-1, 8192), word), 0)
        self.assertEqual((-Fraction(1, 8192) * 4096).__floor__(), -1)  # external floor would differ

    def test_external_format_overflow_and_bounded_rationals(self):
        word = ref.ArithmeticType(50, 12)
        for invalid in (0.5, float("nan"), float("inf"), 1, True, "0.5"):
            with self.assertRaises(ref.Invalid):
                ref.encode_external(invalid, word)
        for invalid in (Fraction(1 << 37), Fraction(-(1 << 37)) - Fraction(1, 4096), Fraction(1 << 4097)):
            with self.assertRaises(ref.Invalid):
                ref.encode_external(invalid, word)
        self.assertEqual(ref.encode_external(Fraction(-(1 << 37)), word), 1 << 49)
        with self.assertRaises(ref.Invalid):
            ref.encode_external(Fraction((1 << 50) - 1, 8192), word)  # tie rounds above signed maximum
        self.assertEqual(ref.encode_external(Fraction(-(1 << 50) - 1, 8192), word), 1 << 49)

    def test_descriptor_substitution_same_label_and_shape_is_rejected(self):
        original = ref.gelu_descriptor()
        # Profile names and byte lengths are not sufficient identities.
        pieces = list(original.pieces)
        pieces[0] = replace(pieces[0], constant=1)
        other = replace(original, pieces=tuple(pieces))
        self.assertEqual(len(original.canonical_bytes()), len(other.canonical_bytes()))
        self.assertNotEqual(original.digest(), other.digest())
        changed = ref.compile_descriptor(other, 0)
        self.assertEqual(ref.interpret_compiled(changed, 0, other.digest())[0], 1)
        self.assertEqual(oracle.interpret_unfused(original, 0)[0], 0)
        with self.assertRaisesRegex(ref.Invalid, "substitution"):
            ref.interpret_compiled(changed, 0, original.digest())
        for changed_descriptor in (replace(original, source_lock_sha256="1" * 64), replace(toy(3), gate="another-gate")):
            self.assertNotEqual(changed_descriptor.digest(), original.digest())

    def test_type_order_scale_and_phi_metadata_are_canonical(self):
        original = ref.gelu_descriptor()
        for key, replacement in [("degree", 2), ("phi", "ars12"), ("coefficient_fractional", [12, 12]),
                                 ("coefficient_order", ["linear", "constant"]), ("external_encoding", "floor"),
                                 ("internal_rounding", "nearest-even"), ("selector_semantics", "abs-before-floor")]:
            obj = original.object()
            obj[key] = replacement
            with self.assertRaises(ref.Invalid):
                ref.parse_descriptor(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
        for expression in (("const", True), ("const", 2), ("b2a", ("const", 1)),
                           ("lt", 9), ("low_lt", 0, 1), ("msb", 8), ("slt", 5)):
            with self.assertRaises(ref.Invalid):
                ref.admission(toy(3, booleans=(expression,)))
        with self.assertRaises(ref.Invalid):
            ref.admission(replace(toy(3), pieces=(ref.AffinePiece(True, 1),)))

    def test_bounded_parsing_and_early_geometry_rejection(self):
        for encoded in (b"{}", b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1.0}', b'{"a":' + b'9' * 100 + b'}',
                        b'[' * 2000 + b'0' + b']' * 2000, b' ' * (ref.MAX_DESCRIPTOR_BYTES + 1)):
            with self.assertRaises(ref.Invalid):
                ref.parse_descriptor(encoded)
        bad = replace(toy(10), boundaries=tuple(range(513)) + (1024,), pieces=(ref.AffinePiece(0, 1),) * 513)
        # No mask-dependent sorting/translation may precede descriptor admission.
        with patch.object(ref, "sorted", side_effect=AssertionError("late admission"), create=True):
            with self.assertRaises(ref.Invalid):
                ref.compile_descriptor(bad, 0)
        for boundaries in ((0, 0, 8), (1, 8), (0, 7), (0, True, 8)):
            with self.assertRaises(ref.Invalid):
                ref.admission(toy(3, boundaries))
        expression = ("const", 1)
        for _ in range(14):
            expression = ("not", expression)
        with self.assertRaises(ref.Invalid):
            ref.admission(toy(3, booleans=(expression,)))
        wide = replace(toy(10), boundaries=tuple(range(512)) + (1024,),
                       pieces=(ref.AffinePiece(0, 1, (("not", ("not", ("not", ("not", ("const", 1))))),) * 4),) * 512)
        with self.assertRaises(ref.Invalid):
            ref.admission(wide)  # 10,240 expression nodes before normalization


class ProtocolThree(unittest.TestCase):
    def test_all_small_partitions_masks_and_inputs(self):
        cases = 0
        for subset in range(128):
            boundaries = (0, *(i for i in range(1, 8) if subset >> (i - 1) & 1), 8)
            pieces = tuple(ref.AffinePiece((5 * i + 2) % 8, (3 * i + 1) % 8,
                                          (("xor", ("low_lt", 2, i % 5), ("msb", i)),))
                           for i in range(len(boundaries) - 1))
            descriptor = ref.Descriptor(ref.ArithmeticType(3, 0), boundaries, pieces)
            digest = descriptor.digest()
            shapes, lengths = set(), set()
            for mask in range(8):
                compiled = ref.compile_descriptor(descriptor, mask)
                shapes.add(compiled.shape)
                lengths.add(len(compiled.canonical_bytes()))
                self.assertEqual(len(compiled.ordinals), min(len(pieces) + 1, 8))
                self.assertTrue(all(a < b for a, b in zip(compiled.boundaries, compiled.boundaries[1:])))
                for x in range(8):
                    self.assertEqual(ref.interpret_compiled(compiled, (x + mask) % 8, digest),
                                     oracle.interpret_unfused(descriptor, x))
                    cases += 1
            self.assertEqual(len(shapes), 1)
            self.assertEqual(len(lengths), 1)
        self.assertEqual(cases, 8192)

    def test_fixed_first_split_full_domain_and_maximum_capacity(self):
        descriptor = toy(3, (0, 3, 8))
        for case in FROZEN["partition_0_3_8"]:
            compiled = ref.compile_descriptor(descriptor, case["mask"])
            self.assertEqual(list(compiled.boundaries), case["boundaries"])
            self.assertEqual(list(compiled.ordinals), case["ordinals"])
        full = toy(3, range(9))
        self.assertEqual(ref.compile_descriptor(full, 5).shape.padded_intervals, 8)
        maximum = toy(10, (*range(512), 1024))
        self.assertEqual(ref.compile_descriptor(maximum, 0).shape.padded_intervals, 513)
        self.assertEqual(ref.compile_descriptor(maximum, 0).boundaries[-2:], (512, 1024))
        # N=256 needs nine bits: a one-byte sentinel would silently become zero.
        eight = ref.compile_descriptor(toy(8), 0)
        self.assertEqual(len(eight.canonical_bytes()), eight.shape.reference_bytes)
        self.assertEqual(eight.boundaries[-1], 256)

    def test_all_small_predicates_shifts_signed_and_low_views(self):
        for bits in range(1, 5):
            modulus = 1 << bits
            expressions = [("lt", t) for t in range(modulus + 1)]
            expressions += [("slt", t) for t in range(-modulus // 2, modulus // 2 + 1)]
            expressions += [("msb", c) for c in range(modulus)]
            expressions += [("low_lt", k, t) for k in range(1, bits + 1) for t in range((1 << k) + 1)]
            expressions += [("or", ("not", ("lt", 1)), ("and", ("const", 1), ("slt", 0)))]
            for at in range(0, len(expressions), 4):
                descriptor = toy(bits, booleans=tuple(expressions[at:at + 4]))
                digest = descriptor.digest()
                for mask in range(modulus):
                    compiled = ref.compile_descriptor(descriptor, mask)
                    for x in range(modulus):
                        self.assertEqual(ref.interpret_compiled(compiled, (x + mask) % modulus, digest),
                                         oracle.interpret_unfused(descriptor, x))

    def test_mask_derived_zero_threshold_keeps_query_slots_and_bytes(self):
        descriptor = toy(3, booleans=(("lt", 3),))
        ordinary = ref.compile_descriptor(descriptor, 0)
        sampled_zero = ref.compile_descriptor(descriptor, 5)
        self.assertEqual(sampled_zero.queries[0].threshold, 0)
        self.assertEqual(len(sampled_zero.queries), 2)
        self.assertEqual(ordinary.shape, sampled_zero.shape)
        self.assertEqual(len(ordinary.canonical_bytes()), len(sampled_zero.canonical_bytes()))
        for sentinel in (0, 8):
            self.assertEqual(ref.admission(toy(3, booleans=(("lt", sentinel),))).queries, 0)

    def test_original_x_horner_counterexample_and_fixed_work(self):
        descriptor = ref.Descriptor(ref.ArithmeticType(3, 0), (0, 8), (ref.AffinePiece(1, 3),))
        compiled = ref.compile_descriptor(descriptor, 4)
        self.assertEqual(ref.interpret_compiled(compiled, 6, descriptor.digest())[0], 7)
        self.assertEqual((3 * 6 + 1) % 8, 3)  # wrong: use public xhat in Horner
        self.assertEqual(compiled.shape.horner_beaver_products, 1)
        constant = replace(descriptor, pieces=(ref.AffinePiece(1, 0),))
        self.assertEqual(ref.admission(constant).horner_beaver_products, 1)  # fixed degree-one padding


class GeluAndNegativeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = ref.load_table()
        cls.descriptor = ref.gelu_descriptor()
        cls.digest = cls.descriptor.digest()

    def test_every_boundary_neighbor_and_full_ring_wrap(self):
        modulus = 1 << 50
        words = {((b + delta) % modulus) for b in self.descriptor.boundaries for delta in (-1, 0, 1)}
        for mask in FROZEN["masks"]:
            compiled = ref.compile_descriptor(self.descriptor, mask)
            self.assertEqual(len(compiled.canonical_bytes()), 11846)
            for x in words:
                expected = oracle.gelu_signed(ref.signed(x, 50), self.table) % modulus
                self.assertEqual(oracle.interpret_unfused(self.descriptor, x), (expected, ()))
                self.assertEqual(ref.interpret_compiled(compiled, (x + mask) % modulus, self.digest), (expected, ()))

    def test_every_central_signed_code_and_frozen_tails(self):
        modulus = 1 << 50
        compiled = ref.compile_descriptor(self.descriptor, modulus - 1)
        for x in range(-16385, 16386):
            expected = oracle.gelu_signed(x, self.table) % modulus
            self.assertEqual(ref.interpret_compiled(compiled, (x - 1) % modulus, self.digest), (expected, ()))
        for x, expected in zip(FROZEN["gelu_inputs_signed_q12"], FROZEN["gelu_expected_signed_q12"]):
            actual, _ = ref.interpret_compiled(compiled, (x - 1) % modulus, self.digest)
            self.assertEqual(ref.signed(actual, 50), expected)
        self.assertEqual(self.descriptor.object()["source_schedule"]["early_tr_output"], [32, 6])

    def test_floor_abs_and_extra_rescale_negative_controls(self):
        x = -1
        correct = oracle.gelu_signed(x, self.table)
        wrong = max(x, 0) - self.table[min(abs(x) // 64, 255)]
        self.assertEqual((correct, wrong), (-31, 0))
        self.assertEqual((oracle.gelu_signed(64, self.table), oracle.gelu_signed(64, self.table) // 4096), (33, 0))

    def test_wrap_rescale_type_and_order_negative_controls(self):
        modulus = 1 << 50
        a = b = 1 << 25
        self.assertEqual(ref.ars((a * b) % modulus, 50, 12), 0)
        self.assertEqual((a * b) // 4096, 1 << 38)
        self.assertEqual(((63 + 1) // 64, 63 // 64 + 1 // 64), (1, 0))
        reduced, same_width = ref.tr(modulus - 1, 50, 6), ref.ars(modulus - 1, 50, 6)
        self.assertEqual((reduced, same_width), ((1 << 44) - 1, modulus - 1))
        self.assertEqual(ref.signed(reduced, 44), ref.signed(same_width, 50))
        self.assertNotEqual(ref.signed(reduced, 50), -1)  # wrong zero-extension/type

    def test_xor_and_low_mask_negative_controls(self):
        p, q, carry = int(0 < 5), int(0 < 3), 0
        self.assertEqual((p ^ q ^ carry, p + q + carry), (0, 2))
        # x=0, r=6, low width=2, gamma=3: r_low=2, theta=1, carry=1.
        self.assertEqual(int(2 < 1) ^ int(2 < 2) ^ 1, 1)
        self.assertEqual(int(2 < 1) ^ int(2 < 6) ^ 1, 0)

    def test_source_effective_width_is_a_real_precondition(self):
        x = (1 << 37) + 64
        full = oracle.gelu_signed(x, self.table)
        narrowed = ref.signed(x % (1 << 38), 38) // 64
        wrong = (x if narrowed >= 0 else 0) - self.table[min(abs(narrowed), 255)]
        self.assertEqual((full, wrong), (137438953536, 0))
        bounds = self.descriptor.object()["source_schedule"]["required_signed_input_range"]
        self.assertFalse(bounds[0] <= x < bounds[1])


if __name__ == "__main__":
    unittest.main()
