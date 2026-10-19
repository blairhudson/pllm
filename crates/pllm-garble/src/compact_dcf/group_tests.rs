use super::*;
use sha2::{Digest, Sha256};

fn deterministic() -> impl FnMut(&mut [u8]) -> Result<(), Error> {
    let mut next = 0u8;
    move |bytes| {
        for byte in bytes {
            *byte = next;
            next = next.wrapping_add(1);
        }
        Ok(())
    }
}

#[test]
fn frozen_scalar_key_identities() {
    // Public, deliberately insecure deterministic fixture roots. Never an
    // issuance API; freeze pre-vector-refactor private scalar key identities.
    let old = generate(
        8,
        137,
        0x12345678u32,
        u64::from(u32::MAX),
        std::mem::size_of::<Comparison>(),
        usize::MAX,
        deterministic(),
    )
    .unwrap();
    let wide = RingComparison::issue_with(
        50,
        50,
        (1 << 49) + 37,
        (1 << 48) + 99,
        usize::MAX,
        deterministic(),
    )
    .unwrap();
    let mut h = Sha256::new();
    for key in &old {
        h.update(key.root.as_slice());
        for word in &key.words {
            h.update(word.seed);
            h.update(word.value.to_le_bytes());
            h.update(word.control);
        }
        h.update(key.final_word.to_le_bytes());
    }
    assert_eq!(
        <[u8; 32]>::from(h.finalize()),
        [
            230, 57, 151, 19, 248, 214, 148, 218, 235, 176, 193, 130, 231, 6, 174, 197, 157, 174,
            138, 160, 212, 53, 137, 71, 181, 251, 200, 102, 33, 150, 166, 155
        ]
    );
    let mut h = Sha256::new();
    for owner in &wide {
        let key = &owner.key;
        h.update(key.root.as_slice());
        for word in &key.words {
            h.update(word.seed);
            h.update(word.value.to_le_bytes());
            h.update(word.control);
        }
        h.update(key.final_word.to_le_bytes());
    }
    assert_eq!(
        <[u8; 32]>::from(h.finalize()),
        [
            175, 96, 75, 107, 31, 133, 198, 149, 47, 105, 31, 95, 245, 146, 228, 89, 146, 222, 148,
            251, 223, 66, 36, 7, 35, 249, 88, 112, 50, 178, 71, 144
        ]
    );
}

#[test]
fn vector_group_exhaustive_small_inputs_and_independent_widths() {
    for input_bits in 1..=5 {
        for output_bits in [1, 5, 50] {
            let mask = (1u64 << output_bits) - 1;
            let cost = VectorComparison::resources(input_bits, output_bits).unwrap();
            for threshold in 0..1u64 << input_bits {
                for payload in [[0, 0], [1, mask], [mask, mask / 2]] {
                    let [a, b] = VectorComparison::issue_with_reservation(
                        input_bits,
                        output_bits,
                        threshold,
                        payload,
                        cost.combined_allocation_bytes,
                        |_| Ok(()),
                    )
                    .unwrap();
                    assert_eq!(a.allocation(), cost);
                    assert_eq!(b.allocation(), cost);
                    assert_eq!(cost.encoded_key_bytes, 32 + 34 * usize::from(input_bits));
                    for input in 0..1u64 << input_bits {
                        let aa = a.eval(0, input).unwrap();
                        let bb = b.eval(1, input).unwrap();
                        for j in 0..2 {
                            let actual =
                                ((u128::from(aa[j]) + u128::from(bb[j])) & u128::from(mask)) as u64;
                            assert_eq!(actual, if input < threshold { payload[j] } else { 0 });
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn vector_first_coordinate_matches_scalar_keys_shares_and_full_ring50() {
    for input_bits in [1, 5, 32, 50] {
        for output_bits in [1, 5, 50] {
            let input_mask = (1u64 << input_bits) - 1;
            let output_mask = (1u64 << output_bits) - 1;
            for threshold in [0, 1, input_mask / 2, input_mask] {
                let payload = [output_mask, output_mask / 2];
                let scalar = RingComparison::issue_with(
                    input_bits,
                    output_bits,
                    threshold,
                    payload[0],
                    usize::MAX,
                    deterministic(),
                )
                .unwrap();
                let vector = VectorComparison::issue_with(
                    input_bits,
                    output_bits,
                    threshold,
                    payload,
                    usize::MAX,
                    |_| Ok(()),
                    deterministic(),
                )
                .unwrap();
                for party in 0..2 {
                    assert_eq!(*scalar[party].key.root, *vector[party].key.root);
                    assert_eq!(
                        *scalar[party].key.final_word,
                        vector[party].key.final_word[0]
                    );
                    for (a, b) in scalar[party].key.words.iter().zip(&vector[party].key.words) {
                        assert_eq!(a.seed, b.seed);
                        assert_eq!(a.control, b.control);
                        assert_eq!(a.value, b.value[0]);
                    }
                }
                for input in [
                    0,
                    1,
                    threshold.wrapping_sub(1) & input_mask,
                    threshold,
                    (threshold + 1) & input_mask,
                    input_mask,
                ] {
                    let aa = vector[0].eval(0, input).unwrap();
                    let bb = vector[1].eval(1, input).unwrap();
                    for j in 0..2 {
                        assert_eq!(
                            (aa[j] + bb[j]) & output_mask,
                            if input < threshold { payload[j] } else { 0 }
                        );
                    }
                    for party in 0..2 {
                        assert_eq!(
                            scalar[party].eval(party as u8, input).unwrap(),
                            vector[party].eval(party as u8, input).unwrap()[0]
                        );
                    }
                }
            }
        }
    }
    let cost = VectorComparison::resources(50, 50).unwrap();
    assert_eq!(cost.encoded_key_bytes, 1732);
    assert_eq!(cost.key_owner_bytes, 64);
    assert_eq!(cost.key_capacity_bytes, 2000);
    println!("shared vector ring50: {cost:?}");
}

#[test]
fn vector_geometry_capacity_hooks_and_failures_precede_entropy() {
    use std::cell::Cell;
    let reservations = Cell::new(0);
    let entropy = Cell::new(0);
    for (n, g, t, p, limit) in [
        (0, 50, 0, [0, 0], usize::MAX),
        (51, 50, 0, [0, 0], usize::MAX),
        (50, 0, 0, [0, 0], usize::MAX),
        (50, 51, 0, [0, 0], usize::MAX),
        (3, 3, 8, [0, 0], usize::MAX),
        (3, 3, 0, [8, 0], usize::MAX),
        (3, 3, 0, [0, 8], usize::MAX),
        (50, 50, 0, [1, 1], 0),
    ] {
        assert!(VectorComparison::issue_with(
            n,
            g,
            t,
            p,
            limit,
            |_| {
                reservations.set(reservations.get() + 1);
                Ok(())
            },
            |_| {
                entropy.set(entropy.get() + 1);
                Ok(())
            }
        )
        .is_err());
    }
    assert_eq!(reservations.get(), 0);
    assert_eq!(entropy.get(), 0);
    let cost = VectorComparison::resources(50, 50).unwrap();
    for fail in 0..2 {
        reservations.set(0);
        assert!(matches!(
            VectorComparison::issue_with(
                50,
                50,
                0,
                [0, 1],
                cost.combined_allocation_bytes,
                |bytes| {
                    assert_eq!(bytes, cost.key_capacity_bytes);
                    let index = reservations.get();
                    reservations.set(index + 1);
                    if index == fail {
                        Err(Error::Resource)
                    } else {
                        Ok(())
                    }
                },
                |_| {
                    entropy.set(entropy.get() + 1);
                    Ok(())
                }
            ),
            Err(Error::Resource)
        ));
        assert_eq!(entropy.get(), 0);
        assert_eq!(reservations.get(), fail + 1);
    }
    for fail in 0..2 {
        reservations.set(0);
        entropy.set(0);
        assert!(matches!(
            VectorComparison::issue_with(
                50,
                50,
                0,
                [0, 1],
                cost.combined_allocation_bytes,
                |_| {
                    reservations.set(reservations.get() + 1);
                    Ok(())
                },
                |bytes| {
                    let index = entropy.get();
                    entropy.set(index + 1);
                    bytes.fill(0);
                    if index == fail {
                        Err(Error::Randomness)
                    } else {
                        Ok(())
                    }
                }
            ),
            Err(Error::Randomness)
        ));
        assert_eq!(reservations.get(), 2);
        assert_eq!(entropy.get(), fail + 1);
    }
    let [a, _] =
        VectorComparison::issue_with_reservation(3, 3, 0, [0, 1], usize::MAX, |_| Ok(())).unwrap();
    assert_eq!(a.eval(2, 0), Err(Error::Domain));
    assert_eq!(a.eval(0, 8), Err(Error::Domain));
}
