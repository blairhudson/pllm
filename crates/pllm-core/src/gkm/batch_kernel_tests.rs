use super::*;

#[test]
fn shared_matrix_kernel_token_major_signed_wrap_and_feature_tile_tails() {
    for columns in [1, 63, 64, 65, 129] {
        let weights: Vec<_> = (0..3 * columns)
            .map(|index| (index as i16).wrapping_mul(257).wrapping_add(i16::MIN))
            .collect();
        let matrix = I16Matrix::from_encoded(3, columns, &weights, &vec![1; columns]).unwrap();
        for batch in [1, 2, 5] {
            let values = [0, 1, u64::MAX, 1 << 63, 0x1234_5678_9abc_def0];
            let shares: Vec<_> = (0..batch * columns)
                .map(|index| values[index % values.len()])
                .collect();
            let mut out = vec![42; batch * 3];
            let failure = payload::failure::after(0);
            matrix
                .multiply_shares_into(&shares, batch, &mut out)
                .unwrap();
            drop(failure);
            for token in 0..batch {
                for row in 0..3 {
                    let expected: i128 = (0..columns)
                        .map(|column| {
                            i128::from(weights[row * columns + column])
                                * i128::from(shares[token * columns + column])
                        })
                        .sum();
                    assert_eq!(out[token * 3 + row], expected as u64);
                }
                assert_eq!(
                    &out[token * 3..token * 3 + 3],
                    matrix
                        .multiply_share(&shares[token * columns..(token + 1) * columns])
                        .unwrap()
                        .as_slice()
                );
            }
            let saved = out.clone();
            assert!(matches!(
                matrix.multiply_shares_into(&shares[1..], batch, &mut out),
                Err(Error::Shape)
            ));
            assert_eq!(out, saved);
            let short = out.len() - 1;
            assert!(matches!(
                matrix.multiply_shares_into(&shares, batch, &mut out[..short]),
                Err(Error::Shape)
            ));
            assert_eq!(out, saved);
            assert!(matches!(
                matrix.multiply_shares_into(&shares, 0, &mut out),
                Err(Error::Shape)
            ));
            assert_eq!(out, saved);
            if columns > 1 {
                assert!(matches!(
                    matrix.multiply_shares_into(&shares, usize::MAX, &mut out),
                    Err(Error::Limit)
                ));
                assert_eq!(out, saved);
            }
        }
    }
}

#[test]
fn shared_input_bounds_accept_minimum_signed_value_only_when_exact_bound_allows_it() {
    let failure = payload::failure::after(0);
    numeric::validate_inputs(&[i64::MIN], 1, &[1u64 << 63], 1).unwrap();
    assert!(matches!(
        numeric::validate_inputs(&[i64::MIN], 1, &[(1u64 << 63) - 1], 1),
        Err(Error::Bound)
    ));
    drop(failure);
}

#[test]
fn shared_lookup_kernel_dense_raw_bits_nonbyte_vocabulary_and_byte_tile_tail() {
    let patterns = [0u32, 0x80000000, 1, 0x80000001, 0x3fa00000];
    let bytes: Vec<_> = (0..9 * 17)
        .flat_map(|index| patterns[index % patterns.len()].to_le_bytes())
        .collect();
    let table = FloatTable::from_le_bytes(9, 17, &bytes).unwrap();
    let selectors = [0xff, 1, 0, 1, 0x55, 0];
    let mut out = vec![42; 3 * 68];
    let failure = payload::failure::after(0);
    table.xor_selectors_into(&selectors, 3, &mut out).unwrap();
    drop(failure);
    for token in 0..3 {
        let mut expected = [0u8; 68];
        for row in 0..9 {
            if (selectors[token * 2 + row / 8] >> (row % 8)) & 1 != 0 {
                for index in 0..68 {
                    expected[index] ^= bytes[row * 68 + index];
                }
            }
        }
        assert_eq!(&out[token * 68..(token + 1) * 68], expected);
        assert_eq!(
            table
                .xor_rows(&selectors[token * 2..token * 2 + 2])
                .unwrap()
                .as_slice(),
            expected
        );
    }
    let saved = out.clone();
    let mut bad = selectors;
    bad[5] = 0x80;
    let failure = payload::failure::after(0);
    assert!(matches!(
        table.xor_selectors_into(&bad, 3, &mut out),
        Err(Error::Shape)
    ));
    assert!(matches!(
        table.xor_selectors_into(&selectors, usize::MAX, &mut out),
        Err(Error::Limit)
    ));
    drop(failure);
    assert_eq!(out, saved);
}
