use super::*;

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
