use super::*;
use num_bigint::{BigInt, Sign};

const Q8_MOD_TEST_MODULI: [u64; 10] = [
    2,
    3,
    4,
    6,
    256,
    65_536,
    CODED_LINEAR_FIELD as u64,
    (1_u64 << 61) - 1,
    u64::MAX - 1,
    u64::MAX,
];

// Independent oracle: decode bits onto the universal 2^-149 integer grid,
// then divide by 2^141 for Q8. No production decode or rounding helper reuse.
fn q8_mod_bigint_oracle(bits: u32, modulus: u64) -> Result<u64, Error> {
    if modulus < 2 {
        return Err(Error::Geometry);
    }
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude / 0x0080_0000;
    let fraction = magnitude % 0x0080_0000;
    if exponent == 255 {
        return Err(Error::NonFinite);
    }
    let grid = if exponent == 0 {
        BigInt::from(fraction)
    } else {
        BigInt::from(0x0080_0000 + fraction) << (exponent - 1) as usize
    };
    let denominator = BigInt::from(1_u8) << 141_usize;
    let mut rounded = &grid / &denominator;
    let remainder = &grid % &denominator;
    let twice_remainder = remainder << 1_usize;
    if twice_remainder > denominator
        || (twice_remainder == denominator && &rounded % 2_u8 == BigInt::from(1_u8))
    {
        rounded += 1_u8;
    }
    if bits >> 31 != 0 {
        rounded = -rounded;
    }
    let modulus = BigInt::from(modulus);
    let mut residue = rounded % &modulus;
    if residue.sign() == Sign::Minus {
        residue += &modulus;
    }
    let (sign, digits) = residue.to_u64_digits();
    assert_ne!(sign, Sign::Minus);
    assert!(digits.len() <= 1);
    Ok(digits.first().copied().unwrap_or(0))
}

fn check_q8_mod_bits(bits: u32) {
    for modulus in Q8_MOD_TEST_MODULI {
        let expected = q8_mod_bigint_oracle(bits, modulus);
        assert_eq!(
            quantize_q8_mod(f32::from_bits(bits), modulus),
            expected,
            "bits={bits:08x}, modulus={modulus}"
        );
        if let Ok(legacy) = quantize_q8(f32::from_bits(bits)) {
            assert_eq!(
                expected,
                Ok(i128::from(legacy).rem_euclid(i128::from(modulus)) as u64),
                "legacy agreement: bits={bits:08x}, modulus={modulus}"
            );
        }
    }
}

#[test]
fn q8_mod_every_finite_exponent_selected_mantissas_both_signs() {
    for exponent in 0..255_u32 {
        for fraction in [
            0, 1, 2, 0x001f_ffff, 0x003f_ffff, 0x0040_0000, 0x007f_fffe, 0x007f_ffff,
        ] {
            for sign in [0, 0x8000_0000] {
                check_q8_mod_bits(sign | (exponent << 23) | fraction);
            }
        }
    }
}

#[test]
fn q8_mod_subnormal_boundaries_and_bounded_samples() {
    // Boundary neighborhoods and bounded deterministic samples, not exhaustive.
    for magnitude in [
        0, 1, 2, 3, 0x003f_ffff, 0x0040_0000, 0x007f_fffd, 0x007f_fffe, 0x007f_ffff,
        0x0080_0000, 0x0080_0001,
    ] {
        for sign in [0, 0x8000_0000] {
            check_q8_mod_bits(sign | magnitude);
        }
    }
    for sample in 0..1024_u32 {
        let fraction = sample.wrapping_mul(0x0001_9e37) & 0x007f_ffff;
        for sign in [0, 0x8000_0000] {
            check_q8_mod_bits(sign | fraction);
        }
    }
}

#[test]
fn q8_mod_half_neighbors_even_and_odd_ties() {
    // Exact x=(q+1/2)/256, constructed from bits; q=0 includes s=24.
    for (tie, rounded) in [
        (118_u32 << 23, 0_i128),
        ((119_u32 << 23) | 0x0040_0000, 2),
        ((120_u32 << 23) | 0x0020_0000, 2),
        ((120_u32 << 23) | 0x0060_0000, 4),
    ] {
        for sign in [0, 0x8000_0000] {
            for magnitude in [tie - 1, tie, tie + 1] {
                check_q8_mod_bits(sign | magnitude);
            }
            for modulus in Q8_MOD_TEST_MODULI {
                let signed = if sign == 0 { rounded } else { -rounded };
                assert_eq!(
                    quantize_q8_mod(f32::from_bits(sign | tie), modulus),
                    Ok(signed.rem_euclid(i128::from(modulus)) as u64)
                );
            }
        }
    }
    // Below/equal/above 1/2 at s=24 must give 0/0/1, not all zero.
    let half = 118_u32 << 23;
    for (bits, rounded) in [(half - 1, 0), (half, 0), (half + 1, 1)] {
        assert_eq!(quantize_q8_mod(f32::from_bits(bits), u64::MAX), Ok(rounded));
    }
}

#[test]
fn q8_mod_error_precedence_nonfinite_and_signed_zero() {
    for bits in [
        0, 0x8000_0000, 0x3f80_0000, 0x7f7f_ffff, 0x7f80_0000, 0xff80_0000,
        0x7fc0_0000, 0x7f80_0001, 0xffc0_1234, 0xffff_ffff,
    ] {
        for modulus in [0, 1] {
            assert_eq!(
                quantize_q8_mod(f32::from_bits(bits), modulus),
                Err(Error::Geometry)
            );
        }
        check_q8_mod_bits(bits);
    }
    for modulus in Q8_MOD_TEST_MODULI {
        for bits in [0, 0x8000_0000] {
            assert_eq!(quantize_q8_mod(f32::from_bits(bits), modulus), Ok(0));
        }
        for bits in [
            0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001, 0xffc0_1234, 0xffff_ffff,
        ] {
            assert_eq!(
                quantize_q8_mod(f32::from_bits(bits), modulus),
                Err(Error::NonFinite)
            );
        }
    }
}

#[test]
fn q8_mod_large_powers_extrema_and_composite_zero_residues() {
    let p61 = (1_u64 << 61) - 1;
    for (exponent, residue) in [(88_u32, 1_u64 << 35), (127, 1_u64 << 13)] {
        let bits = (exponent + 127) << 23;
        assert_eq!(quantize_q8_mod(f32::from_bits(bits), p61), Ok(residue));
        assert_eq!(
            quantize_q8_mod(f32::from_bits(bits | 0x8000_0000), p61),
            Ok(p61 - residue)
        );
        check_q8_mod_bits(bits);
        check_q8_mod_bits(bits | 0x8000_0000);
    }
    for bits in [0x7f7f_fffe, 0x7f7f_ffff, 0xff7f_fffe, 0xff7f_ffff] {
        check_q8_mod_bits(bits);
    }
    for modulus in [2, 4, 256, 65_536] {
        // x=2^16 gives Q8=2^24, divisible by every listed composite/power of two.
        for sign in [0, 0x8000_0000] {
            assert_eq!(
                quantize_q8_mod(f32::from_bits(sign | (143_u32 << 23)), modulus),
                Ok(0)
            );
        }
    }
}

#[test]
fn q8_mod_legacy_i64_boundary_agreement_and_refusals() {
    let boundary = 182_u32 << 23; // x=2^55, Q8=2^63.
    for bits in [
        boundary - 1,
        0x8000_0000 | (boundary - 1),
        0x8000_0000 | boundary,
    ] {
        assert!(quantize_q8(f32::from_bits(bits)).is_ok());
        check_q8_mod_bits(bits);
    }
    assert_eq!(
        quantize_q8(f32::from_bits(0x8000_0000 | boundary)),
        Ok(i64::MIN)
    );
    for bits in [
        boundary, boundary + 1, 0x8000_0000 | (boundary + 1), 0x7f7f_ffff, 0xff7f_ffff,
    ] {
        assert_eq!(quantize_q8(f32::from_bits(bits)), Err(Error::IntegerRange));
        check_q8_mod_bits(bits);
    }
}

fn hex(value: &serde_json::Value) -> u32 {
    u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()
}

#[test]
fn independent_shared_scalar_vectors() {
    let vectors: serde_json::Value = serde_json::from_str(include_str!("vectors.json")).unwrap();
    for row in vectors["fp8"].as_array().unwrap() {
        let result = decode_e4m3fn(hex(&row[0]) as u8).map(f32::to_bits);
        assert_eq!(result.ok(), row[1].as_str().map(|_| hex(&row[1])));
    }
    for row in vectors["dequant"].as_array().unwrap() {
        let result = dequantize_e4m3fn(hex(&row[0]) as u8, hex(&row[1]) as u16).map(f32::to_bits);
        assert_eq!(result.ok(), row[2].as_str().map(|_| hex(&row[2])));
    }
    for row in vectors["q8"].as_array().unwrap() {
        assert_eq!(
            quantize_q8(f32::from_bits(hex(&row[0]))).ok(),
            row[1].as_i64()
        );
    }
    for row in vectors["bf16"].as_array().unwrap() {
        assert_eq!(
            f32_to_bf16(f32::from_bits(hex(&row[0]))).ok(),
            row[1].as_str().map(|_| hex(&row[1]) as u16)
        );
    }
    for row in vectors["reply"].as_array().unwrap() {
        assert_eq!(
            decode_reply_bf16(row[0].as_u64().unwrap() as u32).ok(),
            row[1].as_str().map(|_| hex(&row[1]) as u16)
        );
    }
}

#[test]
fn every_fp8_byte_matches_independent_exact_binary64_formula() {
    for byte in 0..=255_u8 {
        let exponent = (byte >> 3) & 15;
        let fraction = byte & 7;
        if exponent == 15 && fraction == 7 {
            assert_eq!(decode_e4m3fn(byte), Err(Error::NonFinite));
            continue;
        }
        // All these dyadics are exactly representable in both binary64 and F32.
        let magnitude = if exponent == 0 {
            f64::from(fraction) / 512.0
        } else {
            (1.0 + f64::from(fraction) / 8.0) * 2_f64.powi(i32::from(exponent) - 7)
        };
        let expected = (if byte & 128 != 0 {
            -magnitude
        } else {
            magnitude
        }) as f32;
        assert_eq!(decode_e4m3fn(byte).unwrap().to_bits(), expected.to_bits());
    }
}

#[test]
fn every_bf16_logical_word_promotes_without_rounding() {
    for word in 0..=u16::MAX {
        if word & 0x7f80 == 0x7f80 {
            assert_eq!(promote_bf16(word), Err(Error::NonFinite));
        } else {
            assert_eq!(promote_bf16(word).unwrap().to_bits(), u32::from(word) << 16);
        }
    }
}

#[test]
fn source_dtype_and_positive_scale_gates_are_not_repairs() {
    for word in [0, 0x8000, 0xbf80, 0x8001] {
        assert_eq!(positive_bf16_scale(word), Err(Error::NonPositiveScale));
    }
    for word in [0x7f80, 0xff80, 0x7fc1] {
        assert_eq!(positive_bf16_scale(word), Err(Error::NonFinite));
    }
    assert_eq!(positive_bf16_scale(1).unwrap().to_bits(), 1 << 16);
    // 2^45 is a separately stated sufficient proof premise, not an importer cap.
    assert!(positive_bf16_scale(0x5601).is_ok());
    assert!(dequantize_e4m3fn(0x01, 0x5601).is_ok());
}

#[test]
fn bf16_coefficients_and_ties_do_not_use_fp8_or_float_equality() {
    assert_eq!(
        import_bf16_q8(&[0x3f80, 0xbf80, 0x0001, 0x8000]),
        Ok(vec![256, -256, 0, 0])
    );
    assert_eq!(import_bf16_q8(&[0x7fc1]), Err(Error::NonFinite));
    assert_eq!(import_bf16_q8(&[0x5b00]), Err(Error::IntegerRange));
    assert_eq!(
        validate_bf16_tie(&[0x3f80, 0x8000], &[0x3f80, 0x8000]),
        Ok(())
    );
    assert_eq!(validate_bf16_tie(&[0], &[0x8000]), Err(Error::TieMismatch));
    assert_eq!(
        validate_bf16_tie(&[0x7fc1], &[0x7fc1]),
        Err(Error::NonFinite)
    );
    assert_eq!(validate_bf16_tie(&[], &[]), Err(Error::Length));
    assert_eq!(validate_bf16_tie(&[0], &[]), Err(Error::Length));
}

#[test]
fn direct_reply_conversion_rejects_the_double_rounding_substitute() {
    let numerator = 16_842_753;
    assert_eq!(decode_reply_bf16(numerator), Ok(0x4381));
    let wrong_bridge = numerator as f32 / 65536.0;
    assert_eq!(f32_to_bf16(wrong_bridge), Ok(0x4380));
    assert_eq!(
        decode_reply_bf16(CODED_LINEAR_FIELD - numerator),
        Ok(0xc381)
    );
}

#[test]
fn block_orientation_crosses_both_128_axes() {
    let geometry = Block128Geometry::new(256, 256).unwrap();
    let mut weights = vec![0x38; 256 * 256]; // exact +1
    weights[0] = 0xb8; // exact -1
    let result = geometry
        .import_q8(&weights, &[0x3f80, 0x4000, 0x4040, 0x4080])
        .unwrap();
    for (row, col, expected) in [
        (0, 0, -256),
        (0, 127, 256),
        (0, 128, 512),
        (127, 255, 512),
        (128, 0, 768),
        (255, 128, 1024),
    ] {
        assert_eq!(result[row * 256 + col], expected);
    }
    assert_eq!(
        geometry.import_q8(&weights[..100], &[0x3f80; 4]),
        Err(Error::Length)
    );
    assert_eq!(
        geometry.import_q8(&weights, &[0x3f80; 3]),
        Err(Error::Length)
    );
    assert_eq!(
        geometry.import_q8(&weights, &[0x3f80, 0x3f80, 0, 0x3f80]),
        Err(Error::NonPositiveScale)
    );
    weights[17] = 0xff;
    assert_eq!(
        geometry.import_q8(&weights, &[0x3f80; 4]),
        Err(Error::NonFinite)
    );
}

#[test]
fn geometry_never_invents_edge_padding() {
    for (rows, cols) in [
        (0, 128),
        (128, 0),
        (127, 128),
        (128, 129),
        (usize::MAX, 128),
    ] {
        assert_eq!(Block128Geometry::new(rows, cols), Err(Error::Geometry));
    }
    assert!(Block128Geometry::new(4096, 2560).is_ok());
    assert!(Block128Geometry::new(2560, 9728).is_ok());
}

#[test]
fn signed_i64_extremes_use_existing_babybear_arithmetic() {
    for value in [i64::MIN, i64::MAX, -1, 0, 1] {
        assert_eq!(
            q8_to_field(value),
            i128::from(value).rem_euclid(i128::from(CODED_LINEAR_FIELD)) as u32
        );
    }
    assert_eq!(
        linear_bf16(&[256, -256], 2, &[0x3f80]),
        Ok(vec![0x3f80, 0xbf80])
    );
    // Reduce each multiply/add. This does NOT assert no-wrap integer equivalence.
    let weights = [i64::MAX; 4];
    let x = [0x3f80; 4];
    let expected =
        (4 * i128::from(i64::MAX) * 256).rem_euclid(i128::from(CODED_LINEAR_FIELD)) as u32;
    assert_eq!(
        linear_bf16(&weights, 1, &x),
        Ok(vec![decode_reply_bf16(expected).unwrap()])
    );
    assert_eq!(linear_bf16(&[256], 1, &[0x7fc1]), Err(Error::NonFinite));
}

#[test]
fn projection_conformance_full_scale_success() {
    for (byte, scale, expected_bits, expected_q8) in [
        (0x01, 0x5680, 0x5200_0000, 1_i64 << 45),
        (0xb8, 0x5b00, 0xdb00_0000, i64::MIN),
        (0x00, 0x7f7f, 0x0000_0000, 0),
        (0x80, 0x7f7f, 0x8000_0000, 0),
        (0x01, 0x0001, 0x0000_0080, 0),
        (0x81, 0x0001, 0x8000_0080, 0),
    ] {
        assert_eq!(
            dequantize_e4m3fn(byte, scale).map(f32::to_bits),
            Ok(expected_bits),
            "byte={byte:02x}, scale={scale:04x}"
        );
        assert_eq!(
            fp8_q8(byte, scale),
            Ok(expected_q8),
            "byte={byte:02x}, scale={scale:04x}"
        );
    }
}

#[test]
fn projection_conformance_error_classes() {
    assert_eq!(
        dequantize_e4m3fn(0x38, 0x5b00).map(f32::to_bits),
        Ok(0x5b00_0000)
    );
    assert_eq!(fp8_q8(0x38, 0x5b00), Err(Error::IntegerRange));
    assert_eq!(
        dequantize_e4m3fn(0x7e, 0x7f7f).map(f32::to_bits),
        Err(Error::NonFinite)
    );
    assert_eq!(fp8_q8(0x7e, 0x7f7f), Err(Error::NonFinite));
    for byte in [0x00, 0x80] {
        for scale in [0x0000, 0x8000, 0x8001, 0xbf80] {
            assert_eq!(
                dequantize_e4m3fn(byte, scale).map(f32::to_bits),
                Err(Error::NonPositiveScale)
            );
            assert_eq!(fp8_q8(byte, scale), Err(Error::NonPositiveScale));
        }
        for scale in [0x7f80, 0xff80, 0x7fc1, 0xffc1] {
            assert_eq!(
                dequantize_e4m3fn(byte, scale).map(f32::to_bits),
                Err(Error::NonFinite)
            );
            assert_eq!(fp8_q8(byte, scale), Err(Error::NonFinite));
        }
    }
    // Preserve explicit NaN error classes; JSON .ok() comparisons mask them.
    for byte in [0x7f, 0xff] {
        assert_eq!(
            dequantize_e4m3fn(byte, 0x3f80).map(f32::to_bits),
            Err(Error::NonFinite)
        );
        assert_eq!(fp8_q8(byte, 0x3f80), Err(Error::NonFinite));
    }
}

#[test]
fn projection_conformance_four_block_composition() {
    let geometry = Block128Geometry::new(256, 256).unwrap();
    let weights = geometry
        .import_q8(&vec![0x01; 256 * 256], &[0x5680, 0x5700, 0x5780, 0x5800])
        .unwrap();
    let expected_blocks = [1_i64 << 45, 1_i64 << 46, 1_i64 << 47, 1_i64 << 48];
    // Independent Python integer_dots constants: raw integer activation one,
    // not BF16 1.0 (whose Q8 encoding is 256), and no field-window reduction.
    let activation_q8 = [1_i64; 256];
    for row in 0..256 {
        for col in 0..256 {
            assert_eq!(
                weights[row * 256 + col],
                expected_blocks[(row / 128) * 2 + col / 128],
                "row={row}, col={col}"
            );
        }
        let integer_dot: i128 = weights[row * 256..(row + 1) * 256]
            .iter()
            .zip(&activation_q8)
            .map(|(&weight, &activation)| i128::from(weight) * i128::from(activation))
            .sum();
        let expected_dot = if row < 128 {
            13_510_798_882_111_488_i128
        } else {
            54_043_195_528_445_952_i128
        };
        assert_eq!(integer_dot, expected_dot, "row={row}");
    }

    // Separate ordinary-scale composition intentionally uses field arithmetic
    // and the existing direct rational-to-BF16 reply conversion.
    let weights = geometry
        .import_q8(&vec![0x38; 256 * 256], &[0x3f80, 0x4000, 0x4040, 0x4080])
        .unwrap();
    let output = linear_bf16(&weights, 256, &[0x3f80; 256]).unwrap();
    assert_eq!(output.len(), 256);
    for (row, &word) in output.iter().enumerate() {
        assert_eq!(word, if row < 128 { 0x43c0 } else { 0x4460 }, "row={row}");
    }
}
