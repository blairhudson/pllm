//! Bounded independent scalar/bit oracles, authored SOURCE-ONLY and unrun.
//! These tests do not establish admission, source custody or publication safety.
use super::*;

fn workspace() -> SourceDecodeWorkspace {
    SourceDecodeWorkspace::new()
}

#[test]
fn quote_exact_byte_visit_bounds_and_actual_workspace_inventory() {
    for (dtype, width) in [
        (RawScalarType::F32Le, 4),
        (RawScalarType::Bf16Le, 2),
        (RawScalarType::E4m3Fn, 1),
    ] {
        for elements in [1, 2, 7, 128, 256, 257] {
            let source_bytes = elements * width;
            let exact = source_bytes + size_of::<SourceDecodeWorkspace>();
            // No workspace construction before successful quote. This is a
            // quote arithmetic check, NOT a reservation/permit implementation.
            let q = quote(dtype, elements, exact, elements).unwrap();
            let storage = workspace();
            assert_eq!(q.source_bytes, source_bytes);
            assert_eq!(q.scalar_visits, elements);
            assert_eq!(q.workspace_bytes, std::mem::size_of_val(&storage));
            assert_eq!(q.workspace_bytes, size_of::<[u8; 4]>());
            assert_eq!(q.source_bytes + q.workspace_bytes, exact);
            assert_eq!(
                quote(dtype, elements, exact - 1, elements),
                Err("source codec byte capacity exceeded")
            );
            assert_eq!(
                quote(dtype, elements, exact, elements - 1),
                Err("source codec scalar visits exceeded")
            );
            assert!(quote(dtype, elements, source_bytes, elements).is_err());
        }
        assert_eq!(
            quote(dtype, 0, usize::MAX, usize::MAX),
            Err("invalid ordered source selection")
        );
        assert!(quote(dtype, usize::MAX, usize::MAX, usize::MAX).is_err());
    }
    assert_eq!(
        quote(
            RawScalarType::F32Le,
            usize::MAX / 4 + 1,
            usize::MAX,
            usize::MAX
        ),
        Err("source extent overflow")
    );
    assert_eq!(
        quote(
            RawScalarType::Bf16Le,
            usize::MAX / 2 + 1,
            usize::MAX,
            usize::MAX
        ),
        Err("source extent overflow")
    );
    assert_eq!(
        quote(RawScalarType::E4m3Fn, usize::MAX, usize::MAX, usize::MAX),
        Err("resource byte overflow")
    );
    // The raw seam takes explicit caps; existing selected-source caps stay put.
    assert_eq!(super::super::MAX_SELECTED_SOURCE_ELEMENTS, 256);
    assert_eq!(super::super::SELECTED_SOURCE_BYTE_CAPACITY, 1024);
}

#[test]
fn exact_width_rejects_truncation_and_trailing_bytes_before_mutation() {
    let bytes = [0x80, 0x3f, 0, 0, 0, 0];
    for (dtype, width) in [
        (RawScalarType::F32Le, 4),
        (RawScalarType::Bf16Le, 2),
        (RawScalarType::E4m3Fn, 1),
    ] {
        for len in 0..=bytes.len() {
            if len == width {
                continue;
            }
            let mut storage = workspace();
            storage.bytes = [0xa5; 4];
            let error: SourceError = decode_finite(dtype, &bytes[..len], &mut storage).unwrap_err();
            assert_eq!(error, "source scalar width");
            assert_eq!(storage.bytes, [0xa5; 4]);
        }
    }
    for len in [0, 1, 3, 4, 5, 6] {
        let mut storage = workspace();
        storage.bytes = [0x5a; 4];
        assert_eq!(
            positive_bf16_scale(&bytes[..len], &mut storage),
            Err("source scalar width")
        );
        assert_eq!(storage.bytes, [0x5a; 4]);
    }
}

#[test]
fn literal_f32_and_bf16_bits_signed_zero_subnormals_and_legacy_bridge() {
    let mut storage = workspace();
    for bits in [
        0x0000_0000u32,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3e80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ] {
        storage.bytes = [0xff; 4];
        let bytes = bits.to_le_bytes();
        let actual = decode_finite(RawScalarType::F32Le, &bytes, &mut storage).unwrap();
        assert_eq!(actual.to_bits(), bits);
        assert_eq!(storage.bytes, bytes);
        let old = super::super::checked_decode_with_workspace(
            super::super::SourceDtype::F32,
            &bytes,
            &mut storage,
        )
        .unwrap();
        assert_eq!(old.to_bits(), f64::from(f32::from_bits(bits)).to_bits());
    }
    for (word, bits) in [
        (0x0000u16, 0x0000_0000u32),
        (0x8000, 0x8000_0000),
        (0x0001, 0x0001_0000),
        (0x8001, 0x8001_0000),
        (0x007f, 0x007f_0000),
        (0x807f, 0x807f_0000),
        (0x0080, 0x0080_0000),
        (0x8080, 0x8080_0000),
        (0x3e80, 0x3e80_0000),
        (0xbf80, 0xbf80_0000),
        (0x7f7f, 0x7f7f_0000),
        (0xff7f, 0xff7f_0000),
    ] {
        storage.bytes = [0xff; 4];
        let bytes = word.to_le_bytes();
        let actual = decode_finite(RawScalarType::Bf16Le, &bytes, &mut storage).unwrap();
        assert_eq!(actual.to_bits(), bits);
        assert_eq!(storage.bytes, [bytes[0], bytes[1], 0, 0]);
        let old = super::super::checked_decode_with_workspace(
            super::super::SourceDtype::Bf16,
            &bytes,
            &mut storage,
        )
        .unwrap();
        assert_eq!(old.to_bits(), f64::from(f32::from_bits(bits)).to_bits());
    }
}

#[test]
fn generated_bf16_bit_oracle_and_literal_f32_nonfinite_rejections() {
    let mut storage = workspace();
    // Complete 16-bit domain; expected results are literal bit placement, not
    // calls to promote_bf16/positive_bf16_scale or another production decoder.
    for word in 0..=u16::MAX {
        let bytes = word.to_le_bytes();
        let actual = decode_finite(RawScalarType::Bf16Le, &bytes, &mut storage);
        if word & 0x7f80 == 0x7f80 {
            assert_eq!(actual, Err("nonfinite source scalar"));
        } else {
            assert_eq!(actual.unwrap().to_bits(), u32::from(word) * 65536);
        }
        assert_eq!(storage.bytes, [bytes[0], bytes[1], 0, 0]);
    }
    for bits in [
        0x7f80_0000u32,
        0xff80_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0000,
        0xffff_ffff,
    ] {
        let bytes = bits.to_le_bytes();
        assert_eq!(
            decode_finite(RawScalarType::F32Le, &bytes, &mut storage),
            Err("nonfinite source scalar")
        );
        assert_eq!(storage.bytes, bytes);
        assert_eq!(
            super::super::checked_decode_with_workspace(
                super::super::SourceDtype::F32,
                &bytes,
                &mut storage
            ),
            Err("nonfinite source scalar")
        );
    }
}

// Independent exact-f64 dyadic oracle. The whole valid product has <=12
// significant bits and exponent >=-142, so every nonoverflowing result is
// EXACTLY F32-representable. No shared integer encoder/FP8 helper is invoked.
// Powers of two are constructed in F64's exponent field, not pow/libm/FTZ.
fn two(power: i32) -> f64 {
    f64::from_bits(((power + 1023) as u64) << 52)
}
fn product_oracle(byte: u8, scale: u16) -> Option<u32> {
    let e = (byte / 8) % 16;
    let f = byte % 8;
    if e == 15 && f == 7 {
        return None;
    }
    if scale & 0x8000 != 0 || scale == 0 || scale & 0x7f80 == 0x7f80 {
        return None;
    }
    let magnitude = if e == 0 {
        f64::from(f) / 512.
    } else {
        (1. + f64::from(f) / 8.) * two(i32::from(e) - 7)
    };
    let se = (scale / 128) % 256;
    let sf = scale % 128;
    let factor = if se == 0 {
        f64::from(sf) * two(-133)
    } else {
        (1. + f64::from(sf) / 128.) * two(i32::from(se) - 127)
    };
    let product = magnitude * factor;
    if product > f64::from(f32::MAX) {
        return None;
    }
    let signed = if byte < 128 { product } else { -product };
    let expected = signed as f32;
    assert_eq!(f64::from(expected).to_bits(), signed.to_bits());
    Some(expected.to_bits())
}

#[test]
fn all_e4m3fn_bytes_and_scaled_products_match_independent_bit_oracle() {
    let mut storage = workspace();
    for byte in 0..=u8::MAX {
        let raw = decode_finite(RawScalarType::E4m3Fn, &[byte], &mut storage);
        match product_oracle(byte, 0x3f80) {
            Some(bits) => assert_eq!(raw.unwrap().to_bits(), bits),
            None => assert_eq!(raw, Err("nonfinite source scalar")),
        }
        assert_eq!(storage.bytes, [byte, 0, 0, 0]);
        for scale in [
            0x0001u16, 0x0002, 0x0040, 0x007f, 0x0080, 0x0081, 0x3a00, 0x3c00, 0x3e80, 0x3f80,
            0x3fc0, 0x4000, 0x4080, 0x7f00, 0x7f7f,
        ] {
            let actual = scaled_e4m3fn(byte, scale.to_le_bytes());
            match product_oracle(byte, scale) {
                Some(bits) => assert_eq!(actual.unwrap().to_bits(), bits),
                None => assert_eq!(actual, Err("nonfinite source scalar")),
            }
        }
    }
    for (byte, bits) in [
        (0x00, 0u32),
        (0x80, 0x8000_0000),
        (0x01, 0x3b00_0000),
        (0x81, 0xbb00_0000),
        (0x7e, 0x43e0_0000),
    ] {
        assert_eq!(
            decode_finite(RawScalarType::E4m3Fn, &[byte], &mut storage)
                .unwrap()
                .to_bits(),
            bits
        );
    }
    for byte in [0x7f, 0xff] {
        assert_eq!(
            decode_finite(RawScalarType::E4m3Fn, &[byte], &mut storage),
            Err("nonfinite source scalar")
        );
    }
}

#[test]
fn scales_require_positive_finite_bf16_not_signed_zero_or_negative() {
    let mut storage = workspace();
    assert_eq!(
        positive_bf16_scale(&[0x80, 0x3f], &mut storage)
            .unwrap()
            .to_bits(),
        0x3f80_0000
    );
    assert_eq!(
        positive_bf16_scale(&[0x3f, 0x80], &mut storage),
        Err("nonpositive source scale")
    );
    assert_eq!(
        scaled_e4m3fn(0x38, [0x80, 0x3f]).unwrap().to_bits(),
        0x3f80_0000
    );
    assert_eq!(
        scaled_e4m3fn(0x38, [0x3f, 0x80]),
        Err("nonpositive source scale")
    );
    for word in [0x0001u16, 0x007f, 0x0080, 0x3e80, 0x3f80, 0x7f7f] {
        let bytes = word.to_le_bytes();
        assert_eq!(
            positive_bf16_scale(&bytes, &mut storage).unwrap().to_bits(),
            u32::from(word) << 16
        );
        assert_eq!(storage.bytes, [bytes[0], bytes[1], 0, 0]);
    }
    for word in [0x0000u16, 0x8000, 0x8001, 0x807f, 0xbf80, 0xff7f] {
        let bytes = word.to_le_bytes();
        assert_eq!(
            positive_bf16_scale(&bytes, &mut storage),
            Err("nonpositive source scale")
        );
        assert_eq!(storage.bytes, [bytes[0], bytes[1], 0, 0]);
        // Even zero weights cannot bypass scale validation.
        for byte in [0x00, 0x80, 0x01, 0x7e] {
            assert_eq!(scaled_e4m3fn(byte, bytes), Err("nonpositive source scale"));
        }
    }
    for word in [0x7f80u16, 0xff80, 0x7f81, 0xff81, 0x7fc0, 0xffff] {
        let bytes = word.to_le_bytes();
        assert_eq!(
            positive_bf16_scale(&bytes, &mut storage),
            Err("nonfinite source scalar")
        );
        assert_eq!(scaled_e4m3fn(0, bytes), Err("nonfinite source scalar"));
    }
}

// Independent nearest-even reference in exact F64: F32 * 256 is exact here,
// unlike an extra F32 multiply. Cast only bounded <=2^64 integers to i128;
// checked i64 conversion then decides the asymmetric signed endpoints.
fn q8_oracle(value: f32) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    let scaled = f64::from(value) * 256.;
    let magnitude = scaled.abs();
    if magnitude > two(64) {
        return None;
    }
    let low = magnitude.floor();
    let remainder = magnitude - low;
    let rounded = if remainder > 0.5 || (remainder == 0.5 && low % 2. != 0.) {
        low + 1.
    } else {
        low
    };
    let integer = rounded as i128;
    i64::try_from(if scaled.is_sign_negative() {
        -integer
    } else {
        integer
    })
    .ok()
}

#[test]
fn q8_ties_neighbors_i64_endpoints_and_nonfinite_match_independent_oracle() {
    for n in 0..128 {
        let tie = (n as f32 + 0.5) / 256.;
        for bits in [tie.to_bits() - 1, tie.to_bits(), tie.to_bits() + 1] {
            for sign in [0, 0x8000_0000] {
                let value = f32::from_bits(bits | sign);
                assert_eq!(q8(value).ok(), q8_oracle(value));
            }
        }
    }
    for bits in [
        0u32,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x5aff_ffff,
        0x5b00_0000,
        0x5b00_0001,
        0xdaff_ffff,
        0xdb00_0000,
        0xdb00_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0xffc0_0000,
    ] {
        let value = f32::from_bits(bits);
        assert_eq!(q8(value).ok(), q8_oracle(value));
    }
    assert_eq!(q8(f32::from_bits(0xdb00_0000)), Ok(i64::MIN));
    assert_eq!(
        q8(f32::from_bits(0x5b00_0000)),
        Err("source Q8 integer range")
    );
    assert_eq!(
        q8(f32::from_bits(0xdb00_0001)),
        Err("source Q8 integer range")
    );
    assert_eq!(q8(f32::INFINITY), Err("nonfinite source scalar"));
    assert_eq!(q8(0.5 / 256.), Ok(0));
    assert_eq!(q8(1.5 / 256.), Ok(2));
    assert_eq!(q8(-2.5 / 256.), Ok(-2));
}

#[test]
fn scaled_e4m3fn_q8_composition_preserves_ties_and_gradual_underflow() {
    for byte in 0..=u8::MAX {
        for scale in [0x0001u16, 0x0080, 0x3b80, 0x3c00, 0x3f80, 0x5b00, 0x7f7f] {
            let actual = scaled_e4m3fn(byte, scale.to_le_bytes()).and_then(q8);
            match product_oracle(byte, scale).and_then(|bits| q8_oracle(f32::from_bits(bits))) {
                Some(integer) => assert_eq!(actual.unwrap(), integer),
                None => assert!(actual.is_err()),
            }
        }
    }
    // Smallest scale * smallest nonzero FP8: exact F32 subnormal, not FTZ.
    assert_eq!(scaled_e4m3fn(1, [1, 0]).unwrap().to_bits(), 0x0000_0080);
    assert_eq!(scaled_e4m3fn(0x81, [1, 0]).unwrap().to_bits(), 0x8000_0080);
}

#[test]
fn block128_output_input_axes_match_independent_tile_walk_not_transpose() {
    for rows in [128, 256, 384] {
        for columns in [128, 256, 640] {
            for row in [0, 1, 127, 128, 255, 256, rows - 1] {
                for column in [0, 1, 127, 128, 255, 256, columns - 1] {
                    if row >= rows || column >= columns {
                        continue;
                    }
                    let mut ordinal = 0;
                    let mut expected = None;
                    for top in (0..rows).step_by(128) {
                        for left in (0..columns).step_by(128) {
                            if (top..top + 128).contains(&row)
                                && (left..left + 128).contains(&column)
                            {
                                expected = Some(ordinal);
                            }
                            ordinal += 1;
                        }
                    }
                    assert_eq!(
                        block128_scale_index(rows, columns, row, column).unwrap(),
                        expected.unwrap()
                    );
                }
            }
        }
    }
    assert_eq!(block128_scale_index(256, 384, 128, 0), Ok(3));
    assert_eq!(block128_scale_index(384, 256, 0, 128), Ok(1));
}

#[test]
fn block128_rejects_edge_tiles_checked_overflow_and_out_of_range_coordinates() {
    let aligned_max = usize::MAX - usize::MAX % 128;
    for (rows, columns) in [
        (0, 128),
        (128, 0),
        (127, 128),
        (128, 129),
        (256, 255),
        (aligned_max, 128),
        (128, aligned_max),
        (usize::MAX, 128),
    ] {
        assert_eq!(
            block128_scale_index(rows, columns, 0, 0),
            Err("invalid source block geometry")
        );
    }
    for (row, column) in [(256, 0), (0, 384), (usize::MAX, 1), (1, usize::MAX)] {
        assert_eq!(
            block128_scale_index(256, 384, row, column),
            Err("source block coordinate out of bounds")
        );
    }
    // Largest safe full-column multiple for a 128-row matrix, no allocation.
    let columns = (usize::MAX / 128 / 128) * 128;
    assert_eq!(
        block128_scale_index(128, columns, 127, columns - 1),
        Ok(columns / 128 - 1)
    );
}
