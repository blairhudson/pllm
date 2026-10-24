//! Real native hookups; no outer-row, input-proof, privacy or quality claim.
use super::*;
use crate::shared_allocation::{Budget, OwnedBudgetContext};
use std::sync::atomic::Ordering;

fn context() -> source_lookup::Context {
    source_lookup::Context {
        plan: [1; 32],
        session: [2; 32],
        operation: [3; 32],
        input_wire: [4; 32],
        output_wire: [5; 32],
        arithmetic_wire: [6; 32],
        correction_wire: [7; 32],
        select_wire: [8; 32],
        numeric_contract: [9; 32],
        first_tensor_index: 17,
    }
}

// Independent fresh deterministic test setups only, no live owner reset/reuse.
fn rng(mut state: u64) -> impl FnMut(&mut [u8]) -> Result<(), NativeError> {
    move |bytes| {
        for byte in bytes {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state as u8;
        }
        Ok(())
    }
}

fn preparation_allowance(q: &Quote) -> usize {
    // Existing containing-owner logical hash-work allowance, not measured stack.
    q.native_query_phase_bytes + q.preparation_peak_bytes + q.issuance_adapter_bytes + (8 << 10)
}

#[test]
fn prepared_identity_is_actual_fixed_bytes_with_original_preparation_charge_retained() {
    let q = quote::<&Budget>(1).unwrap();
    let allowance = preparation_allowance(&q);
    let budget = Budget::new(allowance).unwrap();
    let original = budget.reserve(allowance).unwrap();
    let (checked, digest) = prepare().unwrap();
    assert_eq!(budget.used_bytes(), allowance);
    assert_eq!(digest, tables::SHA256_U64LE);
    assert_eq!(checked.values()[0], 0);
    assert_eq!(checked.values()[1], 262144);
    assert_eq!(checked.values()[255], 1028);
    assert_eq!(identity(&checked).unwrap(), digest);
    drop((checked, original));
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn quote_reuses_original_native_layout_and_has_one_pair_of_actual_table_copies() {
    fn check<C: AllocationContext>() {
        for lanes in [1, 256] {
            let q = quote::<C>(lanes).unwrap();
            let native = source_lookup::resources_for_context::<C>(spec(lanes)).unwrap();
            assert_eq!(q.lookup, native);
            assert_eq!(
                q.native_reservation_bytes,
                native.combined_reservation_bytes
            );
            assert_eq!(q.native_table_payload_bytes, 2 * 256 * size_of::<u64>());
            assert_eq!(q.verification.entries, 256);
            assert_eq!(q.verification.checked_u128_products, 510);
            assert_eq!(q.verification.checked_u128_additions, 255);
            assert_eq!(q.preparation_hash_bytes, 2048);
            assert_eq!(q.preparation_hash_blocks, 33);
            assert_eq!(q.issuance_pin_hash_bytes_per_lookup, 2048);
            assert_eq!(q.issuance_pin_hash_blocks_per_lookup, 33);
            assert!(q.preparation_peak_bytes >= q.verification.validation_slots_bytes);
            assert_eq!(
                q.issuance_adapter_bytes,
                size_of::<IssuanceSlots<C>>() + size_of::<IdentitySlots>()
            );
            assert!(q.native_query_phase_bytes > lookup_resource_calculation_bytes().unwrap());
        }
        for lanes in [0, 257, usize::MAX] {
            assert!(matches!(
                quote::<C>(lanes),
                Err(Error::Native(NativeError::Domain))
            ));
        }
    }
    check::<&Budget>();
    check::<OwnedBudgetContext>();
}

#[test]
fn verification_rejection_cannot_reach_native_allocation_or_entropy() {
    let q = quote::<&Budget>(1).unwrap();
    let scratch = preparation_allowance(&q);
    for rejection in [
        tables::Error::Overflow,
        tables::Error::Literal {
            index: 255,
            selected: 1029,
        },
    ] {
        let budget = Budget::new(scratch + q.native_reservation_bytes).unwrap();
        let mut original = budget
            .reserve(scratch + q.native_reservation_bytes)
            .unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        let result = after_verification(Err(rejection)).and_then(|(checked, _)| {
            issue(
                &checked,
                context(),
                &[0],
                original.split(q.native_reservation_bytes).unwrap(),
                |_| panic!("verification rejection reached operation entropy"),
            )
        });
        assert!(matches!(result, Err(Error::Verification(e)) if e == rejection));
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(budget.used_bytes(), scratch + q.native_reservation_bytes);
        drop(original);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn nonexact_native_reservation_rejects_before_entropy_and_allocation() {
    let q = quote::<&Budget>(1).unwrap();
    let scratch = preparation_allowance(&q);
    for adjustment in [-1isize, 1] {
        let bytes = q
            .native_reservation_bytes
            .checked_add_signed(adjustment)
            .unwrap();
        let budget = Budget::new(scratch + bytes).unwrap();
        let mut original = budget.reserve(scratch + bytes).unwrap();
        let (checked, _) = prepare().unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        let result = issue(
            &checked,
            context(),
            &[0],
            original.split(bytes).unwrap(),
            |_| panic!("nonexact native reservation reached entropy"),
        );
        assert!(matches!(result, Err(Error::Native(NativeError::Resource))));
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
        assert_eq!(budget.used_bytes(), scratch);
        drop(original);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn original_context_numeric_binding_lease_and_native_copies_are_not_replaced() {
    let q = quote::<&Budget>(1).unwrap();
    let scratch = preparation_allowance(&q);
    let budget = Budget::new(scratch + q.native_reservation_bytes).unwrap();
    let mut original = budget
        .reserve(scratch + q.native_reservation_bytes)
        .unwrap();
    let (checked, _) = prepare().unwrap();
    let (masks, parties) = issue(
        &checked,
        context(),
        &[17],
        original.split(q.native_reservation_bytes).unwrap(),
        rng(7919),
    )
    .unwrap();
    let reference_budget = Budget::new(scratch + q.native_reservation_bytes).unwrap();
    let mut reference_original = reference_budget
        .reserve(scratch + q.native_reservation_bytes)
        .unwrap();
    let (reference_masks, reference_parties) = source_lookup::issue_reserved_with(
        spec(1),
        context(),
        checked.values(),
        &[17],
        reference_original
            .split(q.native_reservation_bytes)
            .unwrap(),
        rng(7919),
    )
    .unwrap();
    assert_eq!(masks.values(), reference_masks.values());
    for role in 0..2 {
        assert_eq!(
            parties[role].internal_binding(),
            reference_parties[role].internal_binding()
        );
    }
    assert_eq!(budget.used_bytes(), reference_budget.used_bytes());
    let actual_binding = parties[0].internal_binding();
    drop((masks, parties, reference_masks, reference_parties));
    assert_eq!(budget.used_bytes(), scratch);
    assert_eq!(reference_budget.used_bytes(), scratch);
    let mut changed = context();
    changed.numeric_contract[31] ^= 1;
    let (other_masks, other_parties) = issue(
        &checked,
        changed,
        &[17],
        budget.reserve(q.native_reservation_bytes).unwrap(),
        rng(7919),
    )
    .unwrap();
    assert_ne!(actual_binding, other_parties[0].internal_binding());
    drop((other_masks, other_parties, original, reference_original));
    assert_eq!(budget.used_bytes(), 0);
    assert_eq!(reference_budget.used_bytes(), 0);
}

fn frame<C: AllocationContext>(step: source_lookup::StepFor<C>) -> source_lookup::FrameFor<C> {
    match step {
        source_lookup::StepFor::Frame(frame) => frame,
        _ => panic!("missing lookup Select exchange"),
    }
}
fn output<C: AllocationContext>(
    step: source_lookup::StepFor<C>,
) -> source_lookup::MaskedOutputFor<C> {
    match step {
        source_lookup::StepFor::Reconstructed(out) => out,
        _ => panic!("missing original lookup output"),
    }
}

#[test]
fn sentinel_global_max_and_reachable_edges_use_original_native_table_backing() {
    // Intentional full-table diagnostics. Only the OUTER owner may admit a
    // secret reachable denominator; these tests do not grant that authority.
    let indices = [0u64, 1, 63, 64, 192, 255];
    let expected = [0, 262144, 4161, 4096, 1365, 1028];
    let input_masks = [1, 2, 3, 4, 5, 6];
    let masked = std::array::from_fn::<_, 6, _>(|i| (indices[i] + input_masks[i]) & 255);
    let q = quote::<&Budget>(6).unwrap();
    let scratch = preparation_allowance(&q);
    let budget = Budget::new(scratch + q.native_reservation_bytes).unwrap();
    let mut original = budget
        .reserve(scratch + q.native_reservation_bytes)
        .unwrap();
    let (masks, [mut a, mut b]) = {
        let (checked, _) = prepare().unwrap();
        issue(
            &checked,
            context(),
            &input_masks,
            original.split(q.native_reservation_bytes).unwrap(),
            rng(4243),
        )
        .unwrap()
    };
    // Actual party copies retain all bytes after the arithmetic token drops.
    let first = [a.begin(&masked).unwrap(), b.begin(&masked).unwrap()];
    let second = [
        frame(a.advance(first[1].as_bytes()).unwrap()),
        frame(b.advance(first[0].as_bytes()).unwrap()),
    ];
    let outputs = [
        output(a.advance(second[1].as_bytes()).unwrap()),
        output(b.advance(second[0].as_bytes()).unwrap()),
    ];
    assert_eq!(outputs[0].values(), outputs[1].values());
    for (i, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            outputs[0].values()[i].wrapping_sub(masks.values()[i]) & ((1u64 << 50) - 1),
            expected
        );
    }
    drop((a, b, masks, outputs));
    assert!(
        budget.used_bytes() > scratch,
        "retained native frames keep ORIGINAL charges"
    );
    drop((first, second));
    assert_eq!(budget.used_bytes(), scratch);
    drop(original);
    assert_eq!(budget.used_bytes(), 0);
}
