//! Independent integer-division oracle vs checked-product production proof.
use super::*;
use sha2::{Digest, Sha256};

#[test]
fn all_256_values_match_independent_u128_quotients_and_fixed_byte_identity() {
    let checked = verify().unwrap();
    for (z, &value) in checked.values().iter().enumerate() {
        let expected = if z == 0 { 0 } else { (1u128 << 18) / z as u128 };
        assert_eq!(u128::from(value), expected, "coordinate {z}");
    }
    let mut hash = Sha256::new();
    for value in checked.values() {
        hash.update(value.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "b03af89917afce4cad1e1923bc09064e7d0b19e7f5b906f7694c3a421c3e8f3d"
    );
}

#[test]
fn every_changed_coordinate_rejects_including_sentinel_and_last_entry() {
    for index in 0..ENTRY_COUNT {
        let mut changed = SELECTED;
        changed[index] += 1;
        assert!(matches!(check(&changed), Err(Error::Literal { index: bad, .. }) if bad == index));
        if SELECTED[index] != 0 {
            changed[index] = SELECTED[index] - 1;
            assert!(
                matches!(check(&changed), Err(Error::Literal { index: bad, .. }) if bad == index)
            );
        }
    }
    let mut changed = SELECTED;
    changed[255] = u64::MAX;
    assert!(matches!(
        check(&changed),
        Err(Error::Literal { index: 255, .. })
    ));
}

#[test]
fn complete_definition_and_global_max_do_not_mint_reachable_coordinate_range() {
    let d = definition();
    assert_eq!((d.entries, d.numerator, d.sentinel), (256, 262144, 0));
    assert_eq!((d.coordinate_bits, d.coordinate_fractional_bits), (8, 0));
    assert_eq!(d.denominator_fractional_bits, 6);
    assert_eq!((d.payload_bits, d.payload_fractional_bits), (50, 12));
    assert_eq!(d.global_payload_max, 262144);
    let checked = verify().unwrap();
    assert_eq!(checked.values().iter().copied().max(), Some(262144));
    for (z, expected) in [
        (0, 0),
        (1, 262144),
        (2, 131072),
        (63, 4161),
        (64, 4096),
        (192, 1365),
        (255, 1028),
    ] {
        assert_eq!(checked.values()[z], expected);
    }
    assert!(
        checked.values()[1] > 4096,
        "Verified alone cannot certify R<=4096"
    );
}

#[test]
fn verified_reuses_immutable_static_storage_without_a_second_table_owner_or_cache() {
    let a = verify().unwrap();
    let b = verify().unwrap();
    assert!(std::ptr::eq(a.values(), b.values()));
    assert_eq!(size_of::<Verified>(), 0);
}

#[test]
fn quote_names_every_fixed_arithmetic_check_and_static_backing() {
    let q = quote().unwrap();
    assert_eq!(
        (q.entries, q.sentinel_checks, q.quotient_checks),
        (256, 1, 255)
    );
    assert_eq!(
        (q.checked_u128_products, q.checked_u128_additions),
        (510, 255)
    );
    assert_eq!(q.validation_slots_bytes, size_of::<ValidationSlots>());
    assert_eq!(q.selected_literal_bytes, 256 * size_of::<u64>());
}
