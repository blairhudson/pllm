"""Tiny independent scalar fixtures; no model, native dependency or payload reads."""
from __future__ import annotations

from fractions import Fraction as F
import importlib
import importlib.abc
from pathlib import Path
import sys
import unittest


class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "pllm", "torch", "numpy", "transformers", "tokenizers", "safetensors",
        }:
            raise AssertionError(f"oracle imported runtime dependency: {fullname}")
        return None


class MaverickNumericEncodingOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = NoRuntimeImports()
        sys.meta_path.insert(0, cls.guard)
        try:
            cls.o = importlib.import_module("benchmarks.research.maverick_numeric_encoding_oracle")
        except BaseException:
            sys.meta_path.remove(cls.guard)
            raise
        expected = Path(__file__).resolve().parents[1] / "benchmarks/research/maverick_numeric_encoding_oracle.py"
        if Path(cls.o.__file__).resolve() != expected:
            sys.meta_path.remove(cls.guard)
            raise AssertionError("oracle resolved outside selected tree")

    @classmethod
    def tearDownClass(cls):
        sys.meta_path.remove(cls.guard)

    def test_all_256_e4m3fn_patterns_have_exact_finite_or_nan_classification(self):
        values = []
        for byte in range(256):
            with self.subTest(byte=byte):
                if byte in (0x7f, 0xff):
                    with self.assertRaisesRegex(ValueError, "NaN"):
                        self.o.decode_e4m3fn(byte)
                else:
                    decoded = self.o.decode_e4m3fn(byte)
                    self.assertIs(type(decoded.value), F)
                    self.assertLessEqual(abs(decoded.value), 448)
                    self.assertEqual(decoded.negative_zero, byte == 0x80)
                    values.append(decoded.value)
        self.assertEqual(len(values), 254)
        self.assertEqual(len(set(values)), 253)  # Fraction coalesces signed zeros.
        self.assertEqual(min(abs(v) for v in values if v), F(1, 512))

    def test_e4m3fn_literal_values_not_signed_i8_or_fnuz(self):
        for byte, value in {
            0x00: 0, 0x80: 0, 0x01: F(1, 512), 0x07: F(7, 512),
            0x08: F(1, 64), 0x38: 1, 0x3c: F(3, 2), 0x78: 256,
            0x7e: 448, 0x81: -F(1, 512), 0xfe: -448,
        }.items():
            with self.subTest(byte=byte):
                self.assertEqual(self.o.decode_e4m3fn(byte).value, value)

    def test_finite_bf16_promotion_preserves_value_and_zero_sign(self):
        for word, expected in {
            0x0000: F(0), 0x8000: F(0), 0x0001: F(1, 1 << 133),
            0x007f: F(127, 1 << 133), 0x0080: F(1, 1 << 126),
            0x3f80: F(1), 0xbf80: F(-1), 0x5600: F(1 << 45),
            0x7f7f: F(255 << 120), 0xff7f: F(-(255 << 120)),
        }.items():
            with self.subTest(word=word):
                promoted = self.o.promote_bf16_to_f32_bits(word)
                self.assertEqual(promoted, word << 16)
                self.assertEqual(self.o.decode_bf16(word).value, expected)
                self.assertEqual(self.o.decode_f32(promoted).value, expected)
                self.assertEqual(self.o.decode_f32(promoted).negative_zero, word == 0x8000)

    def test_bf16_nonfinite_rejects_including_both_nan_signs(self):
        for word in (0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xff81):
            with self.subTest(word=word), self.assertRaises(ValueError):
                self.o.promote_bf16_to_f32_bits(word)

    def test_scale_gate_has_exact_positive_finite_2pow45_premise(self):
        for word in (0x0001, 0x007f, 0x0080, 0x3f80, 0x55ff, 0x5600):
            with self.subTest(word=word):
                self.assertGreater(self.o.validate_scale_bf16(word).value, 0)
        for word in (0, 0x8000, 0x8001, 0xbf80, 0x5601, 0x7f7f, 0x7f80, 0xff80, 0x7fc0):
            with self.subTest(word=word), self.assertRaises(ValueError):
                self.o.validate_scale_bf16(word)

    def test_signed_zero_rounding_and_projection(self):
        for sign in (False, True):
            self.assertEqual(self.o.round_bf16_bits(0, negative_zero=sign), int(sign) << 15)
            self.assertEqual(self.o.round_f32_bits(0, negative_zero=sign), int(sign) << 31)
        self.assertEqual(self.o.projection_f32_bits(0x80, 0x3f80), 0x80000000)
        self.assertEqual(self.o.projection_f32_bits(0x00, 0x3f80), 0)
        self.assertEqual(self.o.projection_q8(0x80, 0x3f80), 0)
        self.assertEqual(self.o.centered_reply_to_bf16_bits(0), 0)

    def test_gradual_underflow_ties_and_subnormal_to_normal_boundary(self):
        for round_, precision, sign_bit in (
            (self.o.round_bf16_bits, 7, 1 << 15), (self.o.round_f32_bits, 23, 1 << 31),
        ):
            quantum = F(1, 1 << (126 + precision))
            self.assertEqual(round_(quantum / 2), 0)
            self.assertEqual(round_(-quantum / 2), sign_bit)
            self.assertEqual(round_(3 * quantum / 2), 2)
            self.assertEqual(round_(-3 * quantum / 2), sign_bit | 2)
            midpoint = ((1 << precision) - F(1, 2)) * quantum
            self.assertEqual(round_(midpoint), 1 << precision)

    def test_normal_ties_even_parity_and_binade_carry(self):
        for round_, precision, sign_bit in (
            (self.o.round_bf16_bits, 7, 1 << 15), (self.o.round_f32_bits, 23, 1 << 31),
        ):
            for lower in ((127 << precision), (127 << precision) + 1, (128 << precision) - 1):
                decode = self.o.decode_bf16 if precision == 7 else self.o.decode_f32
                middle = (decode(lower).value + decode(lower + 1).value) / 2
                expected = lower if lower % 2 == 0 else lower + 1
                self.assertEqual(round_(middle), expected)
                self.assertEqual(round_(-middle), sign_bit | expected)

    def test_finite_overflow_midpoint_rejects_without_saturation(self):
        for round_, midpoint, maximum in (
            (self.o.round_bf16_bits, F(511 << 119), 0x7f7f),
            (self.o.round_f32_bits, F(((1 << 25) - 1) << 103), 0x7f7fffff),
        ):
            self.assertEqual(round_(midpoint - 1), maximum)
            for value in (midpoint, -midpoint, 2 * midpoint):
                with self.subTest(value=value), self.assertRaises(OverflowError):
                    round_(value)

    def test_q8_nearest_even_has_no_i8_clamp_or_f32_intermediate(self):
        for numerator, expected in ((1, 0), (3, 2), (5, 2), (7, 4), (-1, 0), (-3, -2), (-5, -2)):
            self.assertEqual(self.o.q8_i64(F(numerator, 512)), expected)
        self.assertEqual(self.o.q8_i64(448), 114688)
        # Exact Q8 tie + a rational epsilon must not be lost to an F32 cast.
        self.assertEqual(self.o.q8_i64(F(1, 512) + F(1, 1 << 100)), 1)
        self.assertEqual(self.o.q8_i64(-F(1, 512) - F(1, 1 << 100)), -1)

    def test_q8_i64_asymmetric_endpoint_ties_and_rejection(self):
        lo, hi = -(1 << 63), (1 << 63) - 1
        self.assertEqual(self.o.q8_i64(F(hi, 256)), hi)
        self.assertEqual(self.o.q8_i64(F(lo, 256)), lo)
        self.assertEqual(self.o.q8_i64(F(hi, 256) + F(1, 1024)), hi)
        self.assertEqual(self.o.q8_i64(F(lo, 256) - F(1, 512)), lo)
        for value in (F(hi, 256) + F(1, 512), F(lo, 256) - F(3, 1024)):
            with self.assertRaises(OverflowError):
                self.o.q8_i64(value)

    def test_projection_multiplies_scale_and_keeps_smallest_f32_subnormal_product(self):
        for byte, scale, f32_bits, integer in (
            (0x38, 0x4000, 0x40000000, 512),
            (0x38, 0x3f00, 0x3f000000, 128),
            (0x01, 0x0001, 0x00000080, 0),
            (0x81, 0x0001, 0x80000080, 0),
            (0x7e, 0x5600, 0x5a600000, 7 << 59),
            (0xfe, 0x5600, 0xda600000, -(7 << 59)),
        ):
            with self.subTest(byte=byte, scale=scale):
                self.assertEqual(self.o.projection_f32_bits(byte, scale), f32_bits)
                self.assertEqual(self.o.projection_q8(byte, scale), integer)

    def test_head_and_activation_q8_use_exact_bf16_without_synthetic_fp8_scale(self):
        for word, expected in ((0x3f80, 256), (0xbf80, -256), (0x0001, 0), (0x8000, 0), (0x3b00, 0)):
            self.assertEqual(self.o.q8_i64(self.o.decode_bf16(word).value), expected)

    def test_canonical_babybear_centering_and_literal_bf16_endpoints(self):
        p, half = 2_013_265_921, 1_006_632_960
        for residue, centered, bits in (
            (0, 0, 0), (1, 1, 0x3780), (65536, 65536, 0x3f80),
            (half, half, 0x4670), (half + 1, -half, 0xc670), (p - 1, -1, 0xb780),
        ):
            with self.subTest(residue=residue):
                self.assertEqual(self.o.centered_reply(residue), centered)
                self.assertEqual(self.o.centered_reply_to_bf16_bits(residue), bits)
        for residue in (-1, p, 2 * p):
            with self.assertRaises(ValueError):
                self.o.centered_reply_to_bf16_bits(residue)

    def test_direct_bf16_signed_double_rounding_counterexamples_and_neighbors(self):
        n, p = (1 << 24) + (1 << 16) + 1, 2_013_265_921
        for sign in (1, -1):
            sign_bit = 0 if sign == 1 else 0x8000
            for offset, expected in ((-2, 0x4380), (-1, 0x4380), (0, 0x4381), (1, 0x4381)):
                value = sign * (n + offset)
                self.assertEqual(self.o.centered_reply_to_bf16_bits(value % p), sign_bit | expected)
            direct = self.o.centered_reply_to_bf16_bits((sign * n) % p)
            f32 = self.o.round_f32_bits(F(sign * n, 65536))
            via_f32 = self.o.round_bf16_bits(self.o.decode_f32(f32).value)
            self.assertEqual(direct, sign_bit | 0x4381)
            self.assertEqual(via_f32, sign_bit | 0x4380)
            self.assertEqual(self.o.decode_bf16(direct).value, sign * 258)
            self.assertEqual(self.o.decode_bf16(via_f32).value, sign * 256)

    def test_small_field_dot_composition_is_not_a_model_or_no_wrap_proof(self):
        # Two scalar imports, two BF16 inputs. Distinct multiplication directions
        # and a negative coefficient exercise the exact map end-to-end.
        weights = [self.o.projection_q8(0x38, 0x4000), self.o.projection_q8(0xb8, 0x3f00)]
        inputs = [self.o.q8_i64(self.o.decode_bf16(w).value) for w in (0x3f80, 0x4000)]
        self.assertEqual(weights, [512, -128])
        self.assertEqual(inputs, [256, 512])
        reply = sum(w * x for w, x in zip(weights, inputs)) % 2_013_265_921
        self.assertEqual(reply, 65536)
        self.assertEqual(self.o.centered_reply_to_bf16_bits(reply), 0x3f80)
        # A wrapping dot returns its field representative, NOT an asserted true
        # integer sum. The oracle must not silently add a no-wrap certificate.
        true_sum = 2_013_265_922
        self.assertNotEqual(self.o.centered_reply(true_sum % 2_013_265_921), true_sum)

    def test_complete_fp8_global_scale_axes(self):
        result = self.o.complete_linear_oracle(
            rows=256, columns=256, activation_bf16=[0x3f80] * 256,
            weight_encoding="d32-fp8-bf16-scales", weights=[0x38] * (256 * 256),
            scales_bf16=[0x3f80, 0x4000, 0x4040, 0x4080],
        )
        self.assertEqual(result.weight_encoding, "d32-fp8-bf16-scales")
        self.assertEqual(result.integer_dots, (25_165_824,) * 128 + (58_720_256,) * 128)
        self.assertEqual(result.residues, (25_165_824,) * 128 + (58_720_256,) * 128)
        self.assertEqual(result.bf16_bits, (0x43c0,) * 128 + (0x4460,) * 128)
        self.assertEqual(result.signed_no_wrap, (True,) * 256)
        for field in (result.integer_dots, result.signed_no_wrap, result.residues, result.bf16_bits):
            self.assertIs(type(field), tuple)
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            result.weight_encoding = "original-bf16-head"
        for byte, expected in zip((0x01, 0x03, 0x05, 0x81, 0x83, 0x85), (0, 2, 2, 0, -2, -2)):
            self.assertEqual(self.o.projection_q8(byte, 0x3f80), expected)
        self.assertEqual(self.o.projection_q8(0x7e, 0x5600), 7 << 59)
        self.assertEqual(self.o.projection_q8(0x7e, 0x0001), 0)
        # Both allowed extreme scales also pass the complete API, not just scalars.
        for scale in (0x0001, 0x5600):
            weights = [0] * (128 * 128)
            weights[0] = 0x7e
            extreme = self.o.complete_linear_oracle(
                rows=128, columns=128, activation_bf16=[0] * 128,
                weight_encoding="d32-fp8-bf16-scales", weights=weights, scales_bf16=[scale],
            )
            self.assertEqual(extreme.integer_dots, (0,) * 128)
            self.assertEqual(extreme.residues, (0,) * 128)
            self.assertEqual(extreme.bf16_bits, (0,) * 128)
            self.assertEqual(extreme.signed_no_wrap, (True,) * 128)

    def test_complete_original_head_signed_ties(self):
        result = self.o.complete_linear_oracle(
            rows=128, columns=128, activation_bf16=[0x3bc0] * 128,
            weight_encoding="original-bf16-head",
            weights=[0x3f80] * (64 * 128) + [0xbf80] * (64 * 128),
        )
        self.assertEqual(result.weight_encoding, "original-bf16-head")
        self.assertEqual(result.integer_dots, (65_536,) * 64 + (-65_536,) * 64)
        self.assertEqual(result.residues, (65_536,) * 64 + (2_013_200_385,) * 64)
        self.assertEqual(result.bf16_bits, (0x3f80,) * 64 + (0xbf80,) * 64)
        self.assertEqual(result.signed_no_wrap, (True,) * 128)
        for word, expected in zip(
            (0x3b00, 0x3bc0, 0x3c20, 0xbb00, 0xbbc0, 0xbc20), (0, 2, 2, 0, -2, -2),
        ):
            self.assertEqual(self.o.q8_i64(self.o.decode_bf16(word).value), expected)

    def test_complete_dot_wrap_and_cancellation(self):
        for prefix, dot, residue, bits, no_wrap in (
            ([0x4a80, 0xca80, 0x4380], 65_536, 65_536, 0x3f80, True),
            ([0x4a80], 1_073_741_824, 1_073_741_824, 0xc660, False),
        ):
            result = self.o.complete_linear_oracle(
                rows=128, columns=128, activation_bf16=[0x3b80] * 128,
                weight_encoding="original-bf16-head",
                weights=(prefix + [0] * (128 - len(prefix))) * 128,
            )
            self.assertEqual(result.integer_dots, (dot,) * 128)
            self.assertEqual(result.residues, (residue,) * 128)
            self.assertEqual(result.bf16_bits, (bits,) * 128)
            self.assertEqual(result.signed_no_wrap, (no_wrap,) * 128)
        self.assertEqual(self.o.centered_reply(1_073_741_824), -939_524_097)
        self.assertNotEqual(self.o.round_bf16_bits(F(1_073_741_824, 65536)), 0xc660)

    def test_complete_dot_rejects_partial_rounding(self):
        result = self.o.complete_linear_oracle(
            rows=128, columns=128, activation_bf16=[0x3f80] * 128,
            weight_encoding="original-bf16-head",
            weights=([0x3f80, 0x3b80, 0x3b80] + [0] * 125) * 128,
        )
        self.assertEqual(result.integer_dots, (66_048,) * 128)
        self.assertEqual(result.residues, (66_048,) * 128)
        self.assertEqual(result.bf16_bits, (0x3f81,) * 128)
        self.assertEqual(result.signed_no_wrap, (True,) * 128)
        partial = F(0)
        for contribution in (F(1), F(1, 256), F(1, 256)):
            partial = self.o.decode_bf16(self.o.round_bf16_bits(partial + contribution)).value
        self.assertEqual(self.o.round_bf16_bits(partial), 0x3f80)
        # Literal complete signed examples retain the forbidden F32-bridge witness.
        for prefix, dot, residue, bits, wrong in (
            ([0x4780, 0x4380, 0x3b80], 16_842_753, 16_842_753, 0x4381, 0x4380),
            ([0xc780, 0xc380, 0xbb80], -16_842_753, 1_996_423_168, 0xc381, 0xc380),
        ):
            complete = self.o.complete_linear_oracle(
                rows=128, columns=128, activation_bf16=[0x3b80] * 128,
                weight_encoding="original-bf16-head", weights=(prefix + [0] * 125) * 128,
            )
            self.assertEqual(complete.integer_dots, (dot,) * 128)
            self.assertEqual(complete.residues, (residue,) * 128)
            self.assertEqual(complete.bf16_bits, (bits,) * 128)
            bridge = self.o.decode_f32(self.o.round_f32_bits(F(dot, 65536))).value
            self.assertEqual(self.o.round_bf16_bits(bridge), wrong)

    def test_complete_matrix_rejects_malformed_inputs(self):
        base = dict(rows=128, columns=128, activation_bf16=[0x3f80] * 128,
                    weight_encoding="original-bf16-head", weights=[0] * (128 * 128))
        for changes, error in (
            ({"rows": True}, TypeError), ({"columns": False}, TypeError),
            ({"rows": 128.0}, TypeError), ({"columns": "128"}, TypeError),
            ({"rows": 0}, ValueError), ({"columns": -128}, ValueError),
            ({"rows": 127}, ValueError), ({"columns": 129}, ValueError),
            ({"activation_bf16": []}, ValueError), ({"weights": []}, ValueError),
            ({"activation_bf16": [0] * 129}, ValueError),
            ({"weights": [0] * (128 * 128 + 1)}, ValueError),
            ({"weight_encoding": "fp8"}, ValueError),
            ({"weight_encoding": None}, ValueError),
            ({"scales_bf16": []}, ValueError), ({"scales_bf16": [0x3f80]}, ValueError),
        ):
            with self.subTest(changes=changes), self.assertRaises(error):
                self.o.complete_linear_oracle(**(base | changes))
        for field, length in (("activation_bf16", 128), ("weights", 128 * 128)):
            for bad, error in ((True, TypeError), (1.0, TypeError), (-1, ValueError),
                               (65536, ValueError), (0x7f80, ValueError),
                               (0xff80, ValueError), (0x7fc1, ValueError),
                               (0xffc1, ValueError), (0x5b00, OverflowError),
                               (0xdb01, OverflowError)):
                words = [0] * length
                words[-1] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(error):
                    self.o.complete_linear_oracle(**(base | {field: words}))
        fp8 = base | dict(weight_encoding="d32-fp8-bf16-scales", scales_bf16=[0x3f80])
        for scales, error in ((None, ValueError), ([], ValueError),
                              ([0x3f80, 0x3f80], ValueError), ([True], TypeError)):
            with self.subTest(scales=scales), self.assertRaises(error):
                self.o.complete_linear_oracle(**(fp8 | {"scales_bf16": scales}))
        for bad in (0x5601, 0, 0x8000, 0x8001, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0xffc1, -1, 65536):
            with self.subTest(scale=bad), self.assertRaises(ValueError):
                self.o.complete_linear_oracle(**(fp8 | {"scales_bf16": [bad]}))
        for bad, error in ((True, TypeError), (-1, ValueError), (256, ValueError),
                           (0x7f, ValueError), (0xff, ValueError)):
            words = [0] * (128 * 128)
            words[-1] = bad
            with self.subTest(byte=bad), self.assertRaises(error):
                self.o.complete_linear_oracle(**(fp8 | {"weights": words}))

    def test_rejects_nonexact_inputs_and_noncanonical_storage(self):
        for bad in (True, 1.0, "1"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                self.o.q8_i64(bad)
            with self.assertRaises(TypeError):
                self.o.centered_reply(bad)
        for byte in (-1, 256):
            with self.assertRaises(ValueError):
                self.o.decode_e4m3fn(byte)
        for word in (-1, 65536):
            with self.assertRaises(ValueError):
                self.o.decode_bf16(word)
        with self.assertRaises(ValueError):
            self.o.round_bf16_bits(1, negative_zero=True)


if __name__ == "__main__":
    unittest.main()
