use super::{
    codec,
    kernels::{Executor, Matrix},
};

fn reference(w: &[u8], x: &[u32], m: usize, n: usize, b: usize, p: u32) -> Vec<u32> {
    (0..b)
        .flat_map(|bi| {
            (0..m).map(move |j| {
                (0..n)
                    .map(|i| w[j * n + i] as i8 as i128 * x[bi * n + i] as i128)
                    .sum::<i128>()
                    .rem_euclid(p as i128) as u32
            })
        })
        .collect()
}
#[test]
fn modular_all_paths_and_tails() {
    for n in [1, 7, 8, 15, 16, 17, 31, 32, 33, 511, 4097] {
        for batch in [1, 3, 9] {
            let m = 5;
            let w: Vec<u8> = (0..m * n).map(|i| ((i * 17) % 256) as u8).collect();
            let matrix = Matrix::new(&w, m, n).unwrap();
            for p in [65537, 33554467, 2147483647] {
                let x: Vec<u32> = (0..batch * n)
                    .map(|i| ((i as u64 * 7919 + 2147000000) % p as u64) as u32)
                    .collect();
                let expected = reference(&w, &x, m, n, batch, p);
                for threads in [1, 2] {
                    for simd in [false, true] {
                        assert_eq!(
                            matrix
                                .modular(&Executor::new(threads, simd).unwrap(), &x, batch, p)
                                .unwrap(),
                            expected
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn regression_widen_before_multiply() {
    let m = Matrix::new(&[7; 8], 1, 8).unwrap();
    assert_eq!(
        m.modular(
            &Executor::new(1, true).unwrap(),
            &[2147483646; 8],
            1,
            2147483647
        )
        .unwrap(),
        vec![2147483591]
    );
}
#[test]
fn wrap_full_range() {
    for n in [1, 7, 8, 15, 16, 17, 31, 32, 33, 511, 4097] {
        let rows = 5;
        let batch = 3;
        let w: Vec<u8> = (0..rows * n)
            .map(|i| ((i * 29 + 128) % 256) as u8)
            .collect();
        let x: Vec<u32> = (0..batch * n)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_sub(1))
            .collect();
        let expected: Vec<u32> = (0..batch)
            .flat_map(|bi| {
                let w = &w;
                let x = &x;
                (0..rows).map(move |row| {
                    (0..n).fold(0u32, |acc, col| {
                        acc.wrapping_add(
                            (w[row * n + col] as i8 as i32 as u32).wrapping_mul(x[bi * n + col]),
                        )
                    })
                })
            })
            .collect();
        let matrix = Matrix::new(&w, rows, n).unwrap();
        for threads in [1, 2] {
            for simd in [false, true] {
                assert_eq!(
                    matrix
                        .wrap32(&Executor::new(threads, simd).unwrap(), &x, batch)
                        .unwrap(),
                    expected
                );
            }
        }
    }
}
#[test]
fn wrap64_full_range() {
    for n in [1, 7, 8, 17, 33, 511] {
        let rows = 5;
        let batch = 3;
        let w: Vec<u8> = (0..rows * n)
            .map(|i| ((i * 29 + 128) % 256) as u8)
            .collect();
        let x: Vec<u64> = (0..batch * n)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15).wrapping_sub(1))
            .collect();
        let expected: Vec<u64> = (0..batch)
            .flat_map(|batch_index| {
                let w = &w;
                let x = &x;
                (0..rows).map(move |row| {
                    (0..n).fold(0u64, |acc, column| {
                        acc.wrapping_add(
                            (w[row * n + column] as i8 as i64 as u64)
                                .wrapping_mul(x[batch_index * n + column]),
                        )
                    })
                })
            })
            .collect();
        let matrix = Matrix::new(&w, rows, n).unwrap();
        for threads in [1, 2] {
            for simd in [false, true] {
                assert_eq!(
                    matrix
                        .wrap64(&Executor::new(threads, simd).unwrap(), &x, batch)
                        .unwrap(),
                    expected
                );
            }
        }
        assert!(matrix
            .wrap64(&Executor::new(1, true).unwrap(), &x[..x.len() - 1], batch)
            .is_err());
    }
}
#[test]
fn clear_extrema() {
    let m = Matrix::new(&vec![128; 8193], 1, 8193).unwrap();
    let expected = 8193 * 16384;
    for simd in [true, false] {
        assert_eq!(
            m.clear(&Executor::new(1, simd).unwrap(), &vec![-128; 8193], 1)
                .unwrap(),
            vec![expected]
        );
    }
}
#[test]
fn coefficient_exact_u128_reference() {
    let (n, m, k) = (17, 3, 129);
    let q = (1u64 << 53) - 111;
    let w: Vec<u8> = (0..n * m).map(|v| ((v % 15) as i8 - 7) as u8).collect();
    let a: Vec<u64> = (0..n * k).map(|i| q - 1 - i as u64).collect();
    let mat = Matrix::new(&w, m, n).unwrap();
    let out = mat
        .coefficients(&Executor::new(2, true).unwrap(), &a, k, q)
        .unwrap();
    for j in 0..m {
        for c in 0..k {
            let expected = (0..n)
                .map(|i| w[j * n + i] as i8 as i128 * a[i * k + c] as i128)
                .sum::<i128>()
                .rem_euclid(q as i128) as u64;
            assert_eq!(out[j * k + c], expected);
        }
    }
}
#[test]
fn zero_rows_and_empty_batch() {
    let mat = Matrix::new(&[0; 32], 2, 16).unwrap();
    let e = Executor::new(1, true).unwrap();
    assert_eq!(mat.modular(&e, &[2; 16], 1, 65537).unwrap(), vec![0, 0]);
    assert!(mat.modular(&e, &[], 0, 65537).unwrap().is_empty());
}
#[test]
fn invalid_buffers_and_bounds() {
    assert!(Matrix::new(&[0], 2, 3).is_err());
    let mat = Matrix::new(&[1], 1, 1).unwrap();
    let e = Executor::new(1, true).unwrap();
    assert!(mat.modular(&e, &[65537], 1, 65537).is_err());
    assert!(mat.modular(&e, &[1], 1, 1).is_err());
    assert!(Executor::new(0, true).is_err());
    assert!(Executor::new(100, true).is_err());
    assert!(codec::u32s(&[1, 2]).is_err());
    assert!(codec::unpack(&[1, 2], 3).is_err());
}
#[test]
fn codecs_round_trip() {
    for width in [2, 3, 4] {
        let max = ((1u64 << (width * 8)) - 1) as u32;
        let values = [0, 1, 42, 255, 256, max];
        assert_eq!(
            codec::unpack(&codec::pack(&values, width).unwrap(), width).unwrap(),
            values
        );
    }
    assert!(codec::pack(&[1 << 24], 3).is_err());
}
#[test]
fn quantization_ties_and_zero() {
    let (q, s) = codec::quantize(&[0.5, 1.5, 2.5, -0.5, -1.5, -2.5], 1, 6, 4, Some(&[1.])).unwrap();
    assert_eq!(q, vec![0, 2, 2, 0, (-2i8) as u8, (-2i8) as u8]);
    assert_eq!(s, vec![1.]);
    assert_eq!(
        codec::quantize(&[0., 0.], 1, 2, 4, None).unwrap(),
        (vec![0, 0], vec![1.])
    );
    assert!(codec::quantize(&[f32::NAN], 1, 1, 4, None).is_err());
}
#[test]
fn mask_round_trip_all_rings() {
    let x = vec![(-128i8) as u8, (-7i8) as u8, 0, 7, 127];
    for p in [65536u64, 65537, 33554467, 1u64 << 32] {
        let r = codec::random_residues(p, 5).unwrap();
        let y = codec::mask(&x, &r, p).unwrap();
        assert_eq!(codec::unmask(&y, &r, p).unwrap(), vec![-128, -7, 0, 7, 127]);
    }
}
