use super::*;

fn oracle_sum(x: u64, y: u64, mask: u64) -> u64 {
    ((u128::from(x) + u128::from(y)) & u128::from(mask)) as u64
}
fn oracle_other(value: u64, share: u64, bits: u8) -> u64 {
    let n = 1u128 << bits;
    ((u128::from(value) + n - u128::from(share)) % n) as u64
}
fn oracle_product(x: u64, y: u64, mask: u64) -> u64 {
    ((u128::from(x) * u128::from(y)) & u128::from(mask)) as u64
}

#[test]
fn exhaustive_small_ring_values_and_all_input_splits() {
    for bits in 1..=4 {
        let mask = mask(bits).unwrap();
        for x in 0..=mask {
            for y in 0..=mask {
                for sx in 0..=mask {
                    for sy in 0..=mask {
                        // Fixed test-only entropy, varying across inputs/shares.
                        // No injectable entropy is exposed to crate callers.
                        let words = [x ^ sy, y ^ sx, sx, sy, (x + y) & mask];
                        let mut draws = words.into_iter();
                        let [a, b] = issue_with(bits, || Ok(draws.next().unwrap())).unwrap();
                        let left = a.open(sx, sy).unwrap();
                        let right = b
                            .open(oracle_other(x, sx, bits), oracle_other(y, sy, bits))
                            .unwrap();
                        let u = a.finish(0, &left, &right).unwrap();
                        let v = b.finish(1, &right, &left).unwrap();
                        assert_eq!(
                            oracle_sum(u, v, mask),
                            oracle_product(x, y, mask),
                            "bits={bits}, x={x}, y={y}, splits={sx}/{sy}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn ring50_ring64_signed_endpoints_and_carry_openings() {
    for bits in [50, 64] {
        let mask = mask(bits).unwrap();
        let half = 1u64 << (bits - 1);
        // Two's-complement signed extrema/-1 plus products crossing u64 limits.
        let values = [
            0,
            1,
            2,
            half - 1,
            half,
            half + 1,
            mask - 1,
            mask,
            (1u64 << 32) - 1,
            (1u64 << 32) + 1,
        ];
        for x in values {
            for y in values {
                for (sx, sy) in [(0, 0), (mask, mask), (mask, 1), (half, mask)] {
                    let [a, b] = issue(bits).unwrap();
                    let own = a.open(sx, sy).unwrap();
                    let peer = b
                        .open(oracle_other(x, sx, bits), oracle_other(y, sy, bits))
                        .unwrap();
                    let u = a.finish(0, &own, &peer).unwrap();
                    let v = b.finish(1, &peer, &own).unwrap();
                    assert!(u <= mask && v <= mask);
                    assert_eq!(oracle_sum(u, v, mask), oracle_product(x, y, mask));
                }
            }
        }
    }
}

#[test]
fn all_widths_os_random_correlation_and_payload_size() {
    assert_eq!(TRIPLE_BYTES, 24);
    assert_eq!(std::mem::size_of::<TripleShare>(), TRIPLE_BYTES + 8);
    for bits in 1..=64 {
        let mask = mask(bits).unwrap();
        let [left, right] = issue(bits).unwrap();
        assert!(left
            .words
            .iter()
            .chain(right.words.iter())
            .all(|x| *x <= mask));
        let a = oracle_sum(left.words[0], right.words[0], mask);
        let b = oracle_sum(left.words[1], right.words[1], mask);
        let c = oracle_sum(left.words[2], right.words[2], mask);
        assert_eq!(c, oracle_product(a, b, mask));
        let u = left.open(mask, 1).unwrap();
        let v = right.open(1, mask).unwrap();
        assert_eq!(
            oracle_sum(
                left.finish(0, &u, &v).unwrap(),
                right.finish(1, &v, &u).unwrap(),
                mask
            ),
            0
        );
    }
    let [a, b] = issue(50).unwrap();
    let [c, d] = issue(50).unwrap();
    assert!(a.words.as_ref() != c.words.as_ref());
    assert!(b.words.as_ref() != d.words.as_ref());
}

#[test]
fn geometry_precedes_rng_and_exact_legacy_draw_order() {
    for bits in [0, 65, 255] {
        let mut draws = 0;
        assert!(matches!(
            issue_with(bits, || {
                draws += 1;
                Ok(0)
            }),
            Err(Error::Domain)
        ));
        assert_eq!(draws, 0);
    }
    for bits in [1, 50, 64] {
        let words = [u64::MAX, 7, 11, 13, 17];
        let mut n = 0;
        let [a, b] = issue_with(bits, || {
            let word = words[n];
            n += 1;
            Ok(word)
        })
        .unwrap();
        let mask = mask(bits).unwrap();
        assert_eq!(n, 5);
        assert!(a.words.as_ref() == [11 & mask, 13 & mask, 17 & mask]);
        assert_eq!(oracle_sum(a.words[0], b.words[0], mask), words[0] & mask);
        assert_eq!(oracle_sum(a.words[1], b.words[1], mask), words[1] & mask);
        assert_eq!(
            oracle_sum(a.words[2], b.words[2], mask),
            oracle_product(words[0] & mask, words[1] & mask, mask)
        );
    }
    for fail_at in 0..5 {
        let mut draws = 0;
        assert!(matches!(
            issue_with(50, || {
                let index = draws;
                draws += 1;
                if index == fail_at {
                    Err(Error::Randomness)
                } else {
                    Ok(u64::MAX)
                }
            }),
            Err(Error::Randomness)
        ));
        assert_eq!(draws, fail_at + 1);
    }
}

#[test]
fn reject_noncanonical_operands_openings_and_invalid_party() {
    for bits in [1, 8, 50, 63] {
        let invalid = 1u64 << bits;
        let [a, b] = issue(bits).unwrap();
        assert!(matches!(a.open(invalid, 0), Err(Error::Domain)));
        assert!(matches!(a.open(0, invalid), Err(Error::Domain)));
        assert!(matches!(
            a.finish(0, &[0, 0], &[invalid, 0]),
            Err(Error::Domain)
        ));
        assert!(matches!(
            b.finish(1, &[0, invalid], &[0, 0]),
            Err(Error::Domain)
        ));
    }
    for party in [2, 255] {
        let [a, _] = issue(64).unwrap();
        assert!(matches!(
            a.finish(party, &[0, 0], &[0, 0]),
            Err(Error::Domain)
        ));
    }
}
