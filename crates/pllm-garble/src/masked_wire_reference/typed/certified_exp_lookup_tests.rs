//! Real existing-lookup hookups, not full nExp/source/range/security evidence.
use super::*;
use crate::exp_floor;
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

// Independent fresh deterministic setups only; never reuse/reset a live owner.
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

#[test]
fn preparation_reuses_same_static_owner_and_returns_both_actual_pinned_digests() {
    let mut work = Workspace::default();
    let (checked, observed) = prepare(&mut work).unwrap();
    let (other, again) = prepare(&mut work).unwrap();
    assert_eq!(observed, again);
    assert_eq!(size_of::<Verified>(), 0);
    for (index, table) in [Table::LowByte, Table::HighByte].into_iter().enumerate() {
        assert!(std::ptr::eq(checked.values(table), other.values(table)));
        assert!(matches_hex(
            &observed[index],
            table.definition().sha256_u64le
        ));
        assert_eq!(checked.values(table).len(), 256);
        assert_eq!(table.definition().payload_scale, 4096);
        assert_eq!((spec(1).index_bits, spec(1).output_bits), (8, 50));
        assert_eq!(spec(1).boundary, source_lookup::Boundary::Reconstructed);
    }
    assert_ne!(observed[0], observed[1]);
    assert_eq!(
        tables::SOURCE_COMMIT,
        "aa0f2f860266fb79901405ef5e9965b8c9fe2e88"
    );
    assert_eq!(
        tables::PAPER_SHA256,
        "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf"
    );
    let mut changed_digest = observed[0];
    changed_digest[31] ^= 1;
    assert!(!matches_hex(
        &changed_digest,
        Table::LowByte.definition().sha256_u64le
    ));
    assert!(!matches_hex(&observed[0], "00"));
}

#[test]
fn quote_reuses_native_layouts_and_prices_public_work_without_budget_authority() {
    fn check<C: AllocationContext>() {
        for lanes in [1, 256] {
            let q = quote::<C>(lanes).unwrap();
            let native = source_lookup::resources_for_context::<C>(spec(lanes)).unwrap();
            assert_eq!(q.lookup, native);
            assert_eq!(q.floors.proof.entries, 512);
            assert_eq!(q.floors.proof.taylor_steps, 512 * 24);
            assert_eq!(q.floors.selected_literal_bytes, 4096);
            assert_eq!(q.preparation_hash_bytes, 4096);
            assert_eq!(q.issuance_pin_hash_bytes_per_lookup, 2048);
            assert_eq!(
                q.native_table_payload_bytes_per_pair,
                4 * 256 * size_of::<u64>()
            );
            assert_eq!(
                q.paired_native_reservation_bytes,
                2 * native.combined_reservation_bytes
            );
            assert!(q.preparation_peak_bytes >= q.floors.proof.workspace_and_call_bytes);
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
fn unresolved_and_other_floor_failures_cannot_reach_private_allocation_or_entropy() {
    let cost = quote::<&Budget>(1)
        .unwrap()
        .lookup
        .combined_reservation_bytes;
    for rejection in [
        exp_floor::Error::Domain,
        exp_floor::Error::Overflow,
        exp_floor::Error::Unresolved {
            lower_floor: 1,
            upper_floor: 2,
        },
    ] {
        let budget = Budget::new(cost).unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        // Inject only a typed rejection into the SAME post-floor boundary used
        // by prepare. No valid certificate/Verified can be supplied this way.
        let result = after_floors(Err(tables::Error::Floor(rejection))).and_then(|(checked, _)| {
            issue(
                &checked,
                Table::LowByte,
                context(),
                &[0],
                budget.reserve(cost).unwrap(),
                |_| panic!("floor rejection reached entropy"),
            )
        });
        assert!(matches!(result, Err(Error::Floor(tables::Error::Floor(e))) if e == rejection));
        assert_eq!(budget.used_bytes(), 0);
        assert_eq!(budget.fail_after.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn nonexact_original_lease_rejects_before_native_allocation_or_entropy() {
    let (checked, _) = prepare(&mut Workspace::default()).unwrap();
    let cost = quote::<&Budget>(1)
        .unwrap()
        .lookup
        .combined_reservation_bytes;
    for bytes in [cost - 1, cost + 1] {
        let budget = Budget::new(cost + 1).unwrap();
        budget.fail_after.store(0, Ordering::SeqCst);
        let result = issue(
            &checked,
            Table::HighByte,
            context(),
            &[0],
            budget.reserve(bytes).unwrap(),
            |_| panic!("nonexact lease reached entropy"),
        );
        assert!(matches!(result, Err(Error::Native(NativeError::Resource))));
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn hookup_is_exact_native_context_and_material_not_another_table_copy_owner() {
    let (checked, _) = prepare(&mut Workspace::default()).unwrap();
    let cost = quote::<&Budget>(1)
        .unwrap()
        .lookup
        .combined_reservation_bytes;
    let mut bindings = Vec::new();
    for table in [Table::LowByte, Table::HighByte] {
        let budget = Budget::new(cost + 97).unwrap();
        let guard = budget.reserve(97).unwrap();
        budget.fail_after.store(1000, Ordering::SeqCst);
        let (masks, parties) = issue(
            &checked,
            table,
            context(),
            &[17],
            budget.reserve(cost).unwrap(),
            rng(7919),
        )
        .unwrap();
        let reference_budget = Budget::new(cost).unwrap();
        reference_budget.fail_after.store(1000, Ordering::SeqCst);
        let (reference_masks, reference_parties) = source_lookup::issue_reserved_with(
            spec(1),
            context(),
            checked.values(table),
            &[17],
            reference_budget.reserve(cost).unwrap(),
            rng(7919),
        )
        .unwrap();
        assert_eq!(masks.values(), reference_masks.values());
        assert_eq!(
            parties[0].internal_binding(),
            reference_parties[0].internal_binding()
        );
        assert_eq!(
            parties[1].internal_binding(),
            reference_parties[1].internal_binding()
        );
        assert_eq!(
            budget.fail_after.load(Ordering::SeqCst),
            reference_budget.fail_after.load(Ordering::SeqCst)
        );
        assert_eq!(budget.used_bytes(), reference_budget.used_bytes() + 97);
        bindings.push(parties[0].internal_binding());
        drop((masks, parties, reference_masks, reference_parties));
        assert_eq!(budget.used_bytes(), 97);
        assert_eq!(reference_budget.used_bytes(), 0);
        drop(guard);
        assert_eq!(budget.used_bytes(), 0);
    }
    assert_ne!(
        bindings[0], bindings[1],
        "actual selected bytes must bind the invocation"
    );
}

fn frame<C: AllocationContext>(step: source_lookup::StepFor<C>) -> source_lookup::FrameFor<C> {
    match step {
        source_lookup::StepFor::Frame(frame) => frame,
        _ => panic!("missing unchanged reconstructed Select exchange"),
    }
}
fn output<C: AllocationContext>(
    step: source_lookup::StepFor<C>,
) -> source_lookup::MaskedOutputFor<C> {
    match step {
        source_lookup::StepFor::Reconstructed(output) => output,
        _ => panic!("must return original reconstructed lookup owner"),
    }
}

#[test]
fn full_byte_coordinates_and_zero_tail_use_actual_original_lookup_backing() {
    for table in [Table::LowByte, Table::HighByte] {
        let indices = [0usize, 1, 133, 134, 255];
        let input_masks = [1, 2, 3, 4, 5];
        let masked = std::array::from_fn::<_, 5, _>(|i| (indices[i] as u64 + input_masks[i]) & 255);
        let cost = quote::<&Budget>(5)
            .unwrap()
            .lookup
            .combined_reservation_bytes;
        let budget = Budget::new(cost).unwrap();
        let (expected, (masks, [mut a, mut b])) = {
            let (checked, _) = prepare(&mut Workspace::default()).unwrap();
            let expected = indices.map(|index| checked.values(table)[index]);
            let issued = issue(
                &checked,
                table,
                context(),
                &input_masks,
                budget.reserve(cost).unwrap(),
                rng(4243),
            )
            .unwrap();
            (expected, issued)
        };
        // The actual native party copies, not the arithmetic token's lifetime,
        // carry table ownership/charges through all later protocol reads.
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
        for (i, &expected) in expected.iter().enumerate() {
            assert_eq!(
                outputs[0].values()[i].wrapping_sub(masks.values()[i]) & ((1u64 << 50) - 1),
                expected
            );
        }
        drop((masks, a, b, outputs));
        assert!(
            budget.used_bytes() > 0,
            "retained native frames must keep original leases"
        );
        drop((first, second));
        assert_eq!(budget.used_bytes(), 0);
    }
}
