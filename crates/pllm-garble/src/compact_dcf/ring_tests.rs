use super::*;

#[test]
fn complete_small_rings_payloads_and_independent_group_widths() {
    for bits in 1..=5 {
        for output_bits in [1, 5, 50] {
            let mask = (1u64 << output_bits) - 1;
            for threshold in 0..1u64 << bits {
                for payload in [0, 1, mask, 1 << (output_bits - 1)] {
                    let expected = RingComparison::resources(bits, output_bits).unwrap();
                    let [a, b] = RingComparison::issue(
                        bits,
                        output_bits,
                        threshold,
                        payload,
                        expected.combined_allocation_bytes,
                    )
                    .unwrap();
                    assert_eq!(a.allocation(), expected);
                    assert_eq!(a.allocation(), b.allocation());
                    assert_eq!(expected.encoded_key_bytes, 24 + 26 * usize::from(bits));
                    for x in 0..1u64 << bits {
                        assert_eq!(
                            (a.eval(0, x).unwrap() + b.eval(1, x).unwrap()) & mask,
                            if x < threshold { payload } else { 0 }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn ring50_threshold_endpoints_and_payload_hiding_shapes() {
    let n = 1u64 << 50;
    let cost = RingComparison::resources(50, 50).unwrap();
    assert_eq!(cost.encoded_key_bytes, 1324);
    for threshold in [0, 1, (1 << 32) + 1, n / 2, n - 1] {
        for payload in [0, 1, (1 << 32) + 7, n - 1] {
            let [a, b] = RingComparison::issue(50, 50, threshold, payload, usize::MAX).unwrap();
            let [fresh, _] = RingComparison::issue(50, 50, threshold, payload, usize::MAX).unwrap();
            assert_ne!(*a.key.root, *b.key.root);
            assert_ne!(*a.key.root, *fresh.key.root);
            assert_eq!(a.allocation(), cost);
            for x in [
                0,
                1,
                threshold.wrapping_sub(1) & (n - 1),
                threshold,
                (threshold + 1) & (n - 1),
                n / 2,
                n - 1,
            ] {
                assert_eq!(
                    (a.eval(0, x).unwrap() + b.eval(1, x).unwrap()) & (n - 1),
                    if x < threshold { payload } else { 0 }
                );
            }
        }
    }
}

fn deterministic(bytes: &mut [u8]) -> Result<(), Error> {
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = i as u8 + 1;
    }
    Ok(())
}

#[test]
fn same_roots_preserve_each_legacy_u32_share_and_key_words() {
    for bits in [1, 5, 16, 32] {
        let max = (1u64 << bits) - 1;
        for threshold in [0, 1, max / 2, max] {
            for payload in [0, 1, u32::MAX, 1 << 31] {
                let old = generate(
                    bits,
                    threshold,
                    payload,
                    u64::from(u32::MAX),
                    std::mem::size_of::<Comparison>(),
                    usize::MAX,
                    deterministic,
                )
                .unwrap();
                let wide = RingComparison::issue_with(
                    bits,
                    32,
                    threshold,
                    u64::from(payload),
                    usize::MAX,
                    deterministic,
                )
                .unwrap();
                for party in 0..2 {
                    assert_eq!(*old[party].root, *wide[party].key.root);
                    assert_eq!(*old[party].final_word, *wide[party].key.final_word as u32);
                    for (a, b) in old[party].words.iter().zip(&wide[party].key.words) {
                        assert_eq!(a.seed, b.seed);
                        assert_eq!(a.control, b.control);
                        assert_eq!(a.value, b.value as u32);
                    }
                    for x in [0, threshold.wrapping_sub(1) & max, threshold, max] {
                        assert_eq!(
                            old[party].eval(party as u8, x as u32),
                            wide[party].eval(party as u8, x).unwrap() as u32
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_geometry_limits_and_entropy_failure_are_fail_closed() {
    let mut calls = 0;
    for (n, g, t, p, limit) in [
        (0, 50, 0, 0, usize::MAX),
        (51, 50, 0, 0, usize::MAX),
        (50, 0, 0, 0, usize::MAX),
        (50, 51, 0, 0, usize::MAX),
        (3, 3, 8, 0, usize::MAX),
        (3, 3, 0, 8, usize::MAX),
        (50, 50, 0, 1, 0),
    ] {
        assert!(RingComparison::issue_with(n, g, t, p, limit, |_| {
            calls += 1;
            Ok(())
        })
        .is_err());
    }
    assert_eq!(calls, 0);
    let limit = RingComparison::resources(50, 50)
        .unwrap()
        .combined_allocation_bytes;
    assert!(matches!(
        RingComparison::issue_with(50, 50, 0, 1, limit - 1, |_| panic!(
            "entropy requested before admission"
        )),
        Err(Error::Resource)
    ));
    assert!(matches!(
        RingComparison::issue_with(50, 50, 0, 1, limit, |_| {
            calls += 1;
            if calls == 2 {
                Err(Error::Randomness)
            } else {
                Ok(())
            }
        }),
        Err(Error::Randomness)
    ));
    assert_eq!(calls, 2);
    let [a, _] = RingComparison::issue(3, 3, 0, 1, usize::MAX).unwrap();
    assert_eq!(a.eval(2, 0), Err(Error::Domain));
    assert_eq!(a.eval(0, 8), Err(Error::Domain));
}
