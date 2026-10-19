use super::*;

fn check(bits: u8, output_bits: u8, r: u64, inputs: &[u64], boundaries: &[u64]) {
    let n = 1u64 << bits;
    let mask = (1u64 << output_bits) - 1;
    let cost = RingComparison::resources(bits, output_bits).unwrap();
    let [a, b] =
        RingComparison::issue(bits, output_bits, r, cost.combined_allocation_bytes).unwrap();
    assert_eq!(a.allocation(), cost);
    assert_eq!(b.allocation(), cost);
    for &p in boundaries {
        let c = RingComparison::lt_correction(bits, output_bits, r, p).unwrap();
        for &x in inputs {
            let opened = (x + r) & (n - 1);
            assert_eq!(
                a.eval_lt_difference(0, opened, p)
                    .unwrap()
                    .wrapping_add(b.eval_lt_difference(1, opened, p).unwrap())
                    .wrapping_add(c)
                    & mask,
                u64::from(x < p),
                "bits={bits} group={output_bits} r={r} p={p} x={x}"
            );
        }
    }
}

#[test]
fn all_small_masks_inputs_original_boundaries_and_sentinels() {
    for bits in 1..=4 {
        let n = 1u64 << bits;
        let inputs: Vec<_> = (0..n).collect();
        let boundaries: Vec<_> = (0..=n).collect();
        for output_bits in [1, 5, 50] {
            for r in 0..n {
                check(bits, output_bits, r, &inputs, &boundaries);
            }
        }
    }
}

#[test]
fn ring50_zero_mask_wraps_and_source_width_endpoints() {
    let n = 1u64 << 50;
    let points = [
        0,
        1,
        63,
        64,
        (1 << 32) - 1,
        1 << 32,
        (1 << 37) - 1,
        n / 2 - 1,
        n / 2,
        n - 2,
        n - 1,
    ];
    let mut boundaries = points.to_vec();
    boundaries.push(n);
    for r in points {
        check(50, 50, r, &points, &boundaries);
    }
}

#[test]
fn ring32_legacy_universal_outputs_and_invalid_words() {
    let n = 1u64 << 32;
    let points = [0, 1, n / 2, n - 1];
    for r in points {
        let [a, b] = Comparison::issue(32, r as u32).unwrap();
        let [c, d] = RingComparison::issue(32, 32, r, usize::MAX).unwrap();
        for u in points {
            for p in points {
                assert_eq!(
                    a.eval(0, u as u32, p as u32)
                        .wrapping_add(b.eval(1, u as u32, p as u32)),
                    c.eval(0, u, p)
                        .unwrap()
                        .wrapping_add(d.eval(1, u, p).unwrap()) as u32
                );
            }
        }
    }
    assert!(matches!(
        RingComparison::issue(50, 50, 1 << 50, usize::MAX),
        Err(Error::Domain)
    ));
    assert!(matches!(
        RingComparison::issue(50, 50, 0, 0),
        Err(Error::Resource)
    ));
    for (bits, output_bits, r, p) in [
        (0, 50, 0, 0),
        (51, 50, 0, 0),
        (50, 0, 0, 0),
        (50, 51, 0, 0),
        (3, 3, 8, 0),
        (3, 3, 0, 9),
    ] {
        assert_eq!(
            RingComparison::lt_correction(bits, output_bits, r, p),
            Err(Error::Domain)
        );
    }
    let [a, _] = RingComparison::issue(3, 3, 0, usize::MAX).unwrap();
    assert_eq!(a.eval(2, 0, 0), Err(Error::Domain));
    assert_eq!(a.eval(0, 8, 0), Err(Error::Domain));
    assert_eq!(a.eval(0, 0, 9), Err(Error::Domain));
}
