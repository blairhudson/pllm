use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn coefficients() -> [Affine; 2] {
    [
        Affine { left: 2, right: 0, constant: 1 },
        Affine { left: -1, right: 0, constant: 3 },
    ]
}

fn builder() -> Builder {
    Builder::new(2, Interval::new(-2, 2).unwrap()).unwrap()
}

fn program(owner: &ResidentBuilder) -> &Program {
    &owner.builder.as_ref().unwrap().0
}

// Semantic fields, not raw object representation or padding. Heap identity is
// asserted separately against the original Vec, never a Program stack address.
fn assert_program_eq(actual: &Program, expected: &Program) {
    assert_eq!(actual.nodes, expected.nodes);
    assert_eq!(actual.len, expected.len);
    assert_eq!(actual.lanes, expected.lanes);
    assert_eq!(actual.digest, expected.digest);
    assert_eq!(actual.numeric_profile, expected.numeric_profile);
    assert_eq!(actual.checked_validity, expected.checked_validity);
    assert_eq!(actual.last_use, expected.last_use);
    assert_eq!(actual.slots, expected.slots);
    assert_eq!(actual.register_count, expected.register_count);
    assert_eq!(actual.bit_widths, expected.bit_widths);
    assert_eq!(actual.public_affines, expected.public_affines);
    assert_eq!(actual.public_affine_count, expected.public_affine_count);
    assert_eq!(actual.work, expected.work);
}

fn check_affine_error(original: Builder, input: Value, coefficients: &[Affine], error: Error) {
    let before = original.0.clone();
    let pointers = original.0.public_affines.each_ref().map(|v| (v.as_ptr(), v.capacity()));
    let mut legacy = Builder(original.0.clone());
    let mut owner = ResidentBuilder::from_builder(original);
    assert_eq!(legacy.affine_lanes(input, coefficients), Err(error));
    assert_eq!(owner.affine_lanes(input, coefficients), Err(error));
    assert_eq!(owner.state, ResidentState::Failed);
    assert_program_eq(program(&owner), &before);
    assert_program_eq(&legacy.0, &before);
    assert_eq!(
        program(&owner).public_affines.each_ref().map(|v| (v.as_ptr(), v.capacity())),
        pointers
    );
    assert_eq!(owner.pending_affine.len(), 0);
    assert_eq!(owner.pending_affine.capacity(), 0);
    assert_eq!(owner.affine_lanes(input, coefficients), Err(Error::Phase));
    assert_eq!(owner.finish(input), Err(Error::Phase));
    assert_eq!(owner.finish_checked(input, input), Err(Error::Phase));
    let mut out = None;
    assert_eq!(owner.transfer_complete(&mut out), Err(Error::Phase));
    assert!(out.is_none());
}

fn scheduled_fixture() -> (Builder, Value) {
    let mut b = builder();
    let table = b.affine_lanes(Value(0), &coefficients()).unwrap();
    let identity = Affine { left: 1, right: 0, constant: 0 };
    let intermediate = b.affine(table, None, identity).unwrap();
    let output = b.affine(intermediate, None, identity).unwrap();
    (b, output)
}

fn checked_fixture(compact: bool) -> (Builder, Value, Value) {
    let mut b = builder();
    if compact {
        b.0.numeric_profile = COMPACT_NONLINEAR_PROFILE;
    }
    let valid = b.affine(Value(0), None, Affine { left: 0, right: 0, constant: 1 }).unwrap();
    let output = b.affine_lanes(Value(0), &coefficients()).unwrap();
    (b, output, valid)
}

#[test]
fn entering_rejects_all_mutations_without_changing_resident_owners() {
    let (b, output, valid) = checked_fixture(false);
    let mut owner = ResidentBuilder::from_builder(b);
    owner.state = ResidentState::Entering; // Private lifecycle oracle only.
    // One table slot remains free; input/output/validity and coefficients are
    // otherwise valid. Clone expectations only, never replace resident owners.
    assert_eq!(program(&owner).public_affine_count, 1);
    let before = program(&owner).clone();
    let tables = program(&owner).public_affines.each_ref()
        .map(|v| (v.as_ptr(), v.len(), v.capacity()));
    let pending = (
        owner.pending_affine.as_ptr(),
        owner.pending_affine.len(),
        owner.pending_affine.capacity(),
    );
    let pending_content = owner.pending_affine.clone();
    let builder_present = owner.builder.is_some();
    let assert_unchanged = |owner: &ResidentBuilder| {
        assert_eq!(owner.state, ResidentState::Entering);
        assert_eq!(owner.builder.is_some(), builder_present);
        assert_program_eq(program(owner), &before);
        assert_eq!(program(owner).work, before.work);
        assert_eq!(
            program(owner).public_affines.each_ref()
                .map(|v| (v.as_ptr(), v.len(), v.capacity())),
            tables
        );
        assert_eq!(
            (owner.pending_affine.as_ptr(), owner.pending_affine.len(), owner.pending_affine.capacity()),
            pending
        );
        assert_eq!(owner.pending_affine, pending_content);
    };
    assert_eq!(owner.affine_lanes(Value(0), &coefficients()), Err(Error::Phase));
    assert_unchanged(&owner);
    assert_eq!(owner.finish(output), Err(Error::Phase));
    assert_unchanged(&owner);
    assert_eq!(owner.finish_checked(output, valid), Err(Error::Phase));
    assert_unchanged(&owner);
}

#[test]
fn small_affine_fixture_has_explicit_numeric_work_and_schedule() {
    let mut owner = ResidentBuilder::from_builder(builder());
    assert_eq!(owner.affine_lanes(Value(0), &coefficients()).unwrap(), Value(1));
    assert_eq!(owner.affine_lanes(Value(0), &coefficients()).unwrap(), Value(2));
    owner.finish(Value(2)).unwrap();
    assert_eq!(owner.state, ResidentState::Complete);
    let actual = program(&owner);
    assert_eq!(actual.len, 3);
    assert_eq!(actual.lanes, 2);
    assert_eq!(actual.instructions(), 2); // Two local instructions, no RNG.
    assert_eq!(actual.public_affine_count, 2);
    assert_eq!(actual.checked_validity, None);
    // Independent small algebra: 2*x+1 gives [-3,5], -x+3 gives [1,5].
    // Both nodes read the original [-2,2] input, not each other's output.
    assert_eq!(actual.nodes[..3], [
        Some(Node {
            instruction: Instruction::Input,
            bounds: Interval::new(-2, 2).unwrap(),
            bit: false,
            lanes: 2,
        }),
        Some(Node {
            instruction: Instruction::LaneAffine(Value(0), 0),
            bounds: Interval::new(-3, 5).unwrap(),
            bit: false,
            lanes: 2,
        }),
        Some(Node {
            instruction: Instruction::LaneAffine(Value(0), 1),
            bounds: Interval::new(-3, 5).unwrap(),
            bit: false,
            lanes: 2,
        }),
    ]);
    let expected_table = [
        Affine { left: 2, right: 0, constant: 1 },
        Affine { left: -1, right: 0, constant: 3 },
    ];
    assert_eq!(actual.public_affines[0], expected_table);
    assert_eq!(actual.public_affines[1], expected_table);
    assert_eq!(actual.output_bounds(), Interval::new(-3, 5).unwrap());
    assert_eq!(actual.output_lanes(), 2);
    assert_eq!(actual.last_use[..3], [2, 0, 3]);
    assert_eq!(actual.slots[..3], [0, 1, 1]);
    assert_eq!(actual.register_count, 2);
    assert!(actual.nodes[3..].iter().all(Option::is_none));
    assert!(actual.last_use[3..].iter().all(|&v| v == 0));
    assert!(actual.slots[3..].iter().all(|&v| v == 0));
    assert!(actual.bit_widths.iter().all(|&v| v == 0));
    assert_eq!(actual.work.operations(), 2);
    let cursors = actual.work.cursors();
    assert_eq!(cursors.public_blocks, 0);
    assert_eq!(cursors.pair_blocks, 0);
    assert_eq!(cursors.private_blocks, 0);
    assert_eq!(cursors.aes_work, 0);
    // No invented digest literal: wrapper parity tests below share production
    // helpers and are not an independent digest oracle. This numeric/work/
    // schedule oracle covers this fixture, not all operators or funded retirement.
}

#[test]
fn from_builder_moves_original_tables_without_rewriting_program() {
    let (b, _) = scheduled_fixture();
    let before = b.0.clone();
    let pointer = b.0.public_affines[0].as_ptr();
    let capacity = b.0.public_affines[0].capacity();
    let owner = ResidentBuilder::from_builder(b);
    assert_eq!(owner.state, ResidentState::Ready);
    assert_program_eq(program(&owner), &before);
    assert_eq!(program(&owner).public_affines[0].as_ptr(), pointer);
    assert_eq!(program(&owner).public_affines[0].capacity(), capacity);
    assert_eq!(owner.pending_affine.len(), 0);
    assert_eq!(owner.pending_affine.capacity(), 0);
}

#[test]
fn resident_affine_matches_legacy_instructions_bounds_and_clears_pending_on_success() {
    let mut legacy = builder();
    let mut owner = ResidentBuilder::from_builder(builder());
    let mut input = Value(0);
    for index in 0..2 {
        let expected = legacy.affine_lanes(input, &coefficients()).unwrap();
        let actual = owner.affine_lanes(input, &coefficients()).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(owner.state, ResidentState::Ready);
        assert_eq!(owner.pending_affine.len(), 0);
        assert_eq!(owner.pending_affine.capacity(), 0);
        assert_eq!(program(&owner).public_affines[index].len(), 2);
        assert_eq!(program(&owner).public_affines[index].capacity(), 2);
        assert_program_eq(program(&owner), &legacy.0);
        input = actual;
    }
    let pointers = program(&owner).public_affines.each_ref().map(|v| (v.as_ptr(), v.capacity()));
    owner.finish(input).unwrap();
    let expected = legacy.finish(input).unwrap();
    let mut out = None;
    owner.transfer_complete(&mut out).unwrap();
    let actual = out.as_ref().unwrap();
    assert_program_eq(actual, &expected);
    assert_eq!(actual.public_affines.each_ref().map(|v| (v.as_ptr(), v.capacity())), pointers);
}

#[test]
fn affine_original_node_length_and_coefficient_error_precedence() {
    // Binding versus Bounds is observable. Competing Bounds cases below (also
    // table/node limits versus bad coefficients) preserve retention and source-
    // reviewed static validation order, not an observably distinct branch/order
    // proof: the same Bounds result alone cannot distinguish those failures.
    let bad = [Affine { left: 1, right: 1, constant: 0 }; 2];
    check_affine_error(builder(), Value(255), &bad[..1], Error::Binding);
    check_affine_error(builder(), Value(0), &bad[..1], Error::Bounds);
    check_affine_error(builder(), Value(0), &bad, Error::Bounds);
}

#[test]
fn affine_table_limit_retains_original_tables_and_precedes_coefficient_validation() {
    let mut b = builder();
    b.affine_lanes(Value(0), &coefficients()).unwrap();
    b.affine_lanes(Value(0), &coefficients()).unwrap();
    let bad = [Affine { left: 1, right: 1, constant: 0 }; 2];
    check_affine_error(b, Value(0), &bad, Error::Bounds);
}

#[test]
fn affine_node_limit_preserves_legacy_bounds_rejection_without_reservation() {
    let mut b = builder();
    let identity = Affine { left: 1, right: 0, constant: 0 };
    for _ in 1..MAX_NODES {
        b.affine(Value(0), None, identity).unwrap();
    }
    let bad = [Affine { left: 1, right: 1, constant: 0 }; 2];
    check_affine_error(b, Value(0), &bad, Error::Bounds);
}

#[test]
fn affine_bound_overflow_matches_legacy_before_reservation() {
    let b = Builder::new(2, Interval::new(i64::MAX, i64::MAX).unwrap()).unwrap();
    let overflow = [Affine { left: 2, right: 0, constant: 0 }; 2];
    check_affine_error(b, Value(0), &overflow, Error::Bounds);
}

#[test]
fn after_reserve_unwind_retains_exact_pending_allocation_and_latches_terminal_failure() {
    let mut b = builder();
    b.affine_lanes(Value(0), &coefficients()).unwrap();
    let before = b.0.clone();
    let original = (b.0.public_affines[0].as_ptr(), b.0.public_affines[0].capacity());
    // Both the actual resident owner and caller coefficients outlive the frame.
    let mut owner = ResidentBuilder::from_builder(b);
    let coefficients = coefficients();
    PANIC_AFTER_AFFINE_RESERVE.with(|flag| flag.set(true));
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let _ = owner.affine_lanes(Value(0), &coefficients);
    }));
    assert!(caught.is_err());
    assert_eq!(owner.state, ResidentState::Failed);
    assert_program_eq(program(&owner), &before);
    assert_eq!(
        (program(&owner).public_affines[0].as_ptr(), program(&owner).public_affines[0].capacity()),
        original
    );
    let observed = RESERVED_AFFINE.with(|observed| observed.get());
    assert!(!observed.0.is_null());
    assert!(observed.1 >= coefficients.len());
    assert_eq!(owner.pending_affine.len(), 0);
    assert_eq!((owner.pending_affine.as_ptr(), owner.pending_affine.capacity()), observed);
    assert_eq!(owner.affine_lanes(Value(0), &coefficients), Err(Error::Phase));
    assert_eq!(owner.finish(Value(1)), Err(Error::Phase));
    assert_eq!((owner.pending_affine.as_ptr(), owner.pending_affine.capacity()), observed);
    PANIC_AFTER_AFFINE_RESERVE.with(|flag| assert!(!flag.get()));
}

#[test]
fn ordinary_finish_success_and_exact_once_transfer_preserve_original_table_backing() {
    let (b, output) = scheduled_fixture();
    let expected = Builder(b.0.clone()).finish(output).unwrap();
    let pointer = b.0.public_affines[0].as_ptr();
    let capacity = b.0.public_affines[0].capacity();
    let mut owner = ResidentBuilder::from_builder(b);
    let mut out = None;
    assert_eq!(owner.transfer_complete(&mut out), Err(Error::Phase));
    assert!(out.is_none());
    owner.finish(output).unwrap();
    assert_eq!(owner.state, ResidentState::Complete);
    assert_program_eq(program(&owner), &expected);
    owner.transfer_complete(&mut out).unwrap();
    assert_eq!(owner.state, ResidentState::Transferred);
    assert!(owner.builder.is_none());
    let actual = out.as_ref().unwrap();
    assert_program_eq(actual, &expected);
    assert_eq!((actual.public_affines[0].as_ptr(), actual.public_affines[0].capacity()), (pointer, capacity));
    assert_eq!(owner.transfer_complete(&mut out), Err(Error::Phase));
    let mut empty = None;
    assert_eq!(owner.transfer_complete(&mut empty), Err(Error::Phase));
    assert!(empty.is_none());
    assert_eq!(owner.finish(output), Err(Error::Phase));
    assert_eq!(owner.finish_checked(output, Value(1)), Err(Error::Phase));
    assert_program_eq(out.as_ref().unwrap(), &expected);
}

#[test]
fn ordinary_finish_error_keeps_original_program_and_matches_consuming_legacy() {
    for output in [Value(255), Value(0), Value(1)] {
        let (b, _) = scheduled_fixture();
        let expected = Builder(b.0.clone()).finish(output).err();
        assert_eq!(expected, Some(Error::Binding));
        let before = b.0.clone();
        let pointer = b.0.public_affines[0].as_ptr();
        let mut owner = ResidentBuilder::from_builder(b);
        assert_eq!(owner.finish(output).err(), expected);
        assert_eq!(owner.state, ResidentState::Failed);
        assert_program_eq(program(&owner), &before);
        assert_eq!(program(&owner).public_affines[0].as_ptr(), pointer);
        assert_eq!(owner.finish(output), Err(Error::Phase));
    }
}

#[test]
fn partial_schedule_unwind_retains_digest_last_use_slots_registers_and_table() {
    let (b, output) = scheduled_fixture();
    let expected = Builder(b.0.clone()).finish(output).unwrap();
    let pointer = b.0.public_affines[0].as_ptr();
    let capacity = b.0.public_affines[0].capacity();
    let mut owner = ResidentBuilder::from_builder(b);
    let mut out = None;
    PANIC_AFTER_PARTIAL_SCHEDULE.with(|flag| flag.set(true));
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let _ = owner.finish(output);
    }));
    assert!(caught.is_err());
    assert_eq!(owner.state, ResidentState::Failed);
    let retained = program(&owner);
    assert_eq!(retained.digest, expected.digest);
    assert_ne!(retained.digest, [0; 32]);
    assert_eq!(retained.last_use, expected.last_use);
    assert_eq!(retained.slots[0], expected.slots[0]);
    assert_eq!(retained.slots[1], expected.slots[1]);
    assert_eq!(retained.slots[1], 1);
    assert_eq!(retained.slots[2], 0); // Unvisited, not rolled back or completed.
    assert_eq!(retained.register_count, 2);
    assert_eq!(retained.work, WorkBudget::default());
    assert_ne!(expected.work, retained.work);
    assert_eq!((retained.public_affines[0].as_ptr(), retained.public_affines[0].capacity()), (pointer, capacity));
    assert_eq!(owner.transfer_complete(&mut out), Err(Error::Phase));
    assert!(out.is_none());
    assert_eq!(owner.finish(output), Err(Error::Phase));
    assert_eq!(program(&owner).digest, expected.digest);
    PANIC_AFTER_PARTIAL_SCHEDULE.with(|flag| assert!(!flag.get()));
}

#[test]
fn checked_finish_matches_legacy_digest_work_and_both_profile_rules() {
    for compact in [false, true] {
        let (b, output, valid) = checked_fixture(compact);
        let expected = Builder(b.0.clone()).finish_checked(output, valid).unwrap();
        let pointer = b.0.public_affines[0].as_ptr();
        let mut owner = ResidentBuilder::from_builder(b);
        owner.finish_checked(output, valid).unwrap();
        assert_eq!(owner.state, ResidentState::Complete);
        assert_program_eq(program(&owner), &expected);
        assert_eq!(program(&owner).checked_validity, Some(valid));
        assert_eq!(program(&owner).numeric_profile, if compact { COMPACT_NONLINEAR_PROFILE } else { CHECKED_RANGE_PROFILE });
        let mut out = None;
        owner.transfer_complete(&mut out).unwrap();
        assert_program_eq(out.as_ref().unwrap(), &expected);
        assert_eq!(out.as_ref().unwrap().public_affines[0].as_ptr(), pointer);
    }
}

#[test]
fn checked_finish_precedence_matches_legacy_before_profile_mutation() {
    for (output, valid, error) in [
        (Value(255), Value(255), Error::Binding),
        (Value(255), Value(0), Error::Bounds),
        (Value(0), Value(255), Error::Binding),
    ] {
        let (b, _, _) = checked_fixture(false);
        assert_eq!(Builder(b.0.clone()).finish_checked(output, valid).err(), Some(error));
        let before = b.0.clone();
        let mut owner = ResidentBuilder::from_builder(b);
        assert_eq!(owner.finish_checked(output, valid), Err(error));
        assert_eq!(owner.state, ResidentState::Failed);
        assert_program_eq(program(&owner), &before);
    }
}

#[test]
fn checked_finish_retains_profile_and_validity_when_later_ordinary_validation_fails() {
    let (b, _, valid) = checked_fixture(false);
    assert_eq!(Builder(b.0.clone()).finish_checked(Value(0), valid).err(), Some(Error::Binding));
    let mut expected = b.0.clone();
    expected.checked_validity = Some(valid);
    expected.numeric_profile = CHECKED_RANGE_PROFILE;
    let mut owner = ResidentBuilder::from_builder(b);
    assert_eq!(owner.finish_checked(Value(0), valid), Err(Error::Binding));
    assert_eq!(owner.state, ResidentState::Failed);
    assert_program_eq(program(&owner), &expected);
    assert_eq!(owner.finish_checked(Value(2), valid), Err(Error::Phase));
}

#[test]
fn checked_finish_unwind_retains_checked_mutations_and_partial_schedule() {
    let (b, output, valid) = checked_fixture(false);
    let expected = Builder(b.0.clone()).finish_checked(output, valid).unwrap();
    let mut owner = ResidentBuilder::from_builder(b);
    PANIC_AFTER_PARTIAL_SCHEDULE.with(|flag| flag.set(true));
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let _ = owner.finish_checked(output, valid);
    }));
    assert!(caught.is_err());
    assert_eq!(owner.state, ResidentState::Failed);
    let retained = program(&owner);
    assert_eq!(retained.checked_validity, Some(valid));
    assert_eq!(retained.numeric_profile, CHECKED_RANGE_PROFILE);
    assert_eq!(retained.digest, expected.digest);
    assert_eq!(retained.last_use, expected.last_use);
    assert_eq!(retained.register_count, 2);
    assert_eq!(retained.work, WorkBudget::default());
    assert_eq!(owner.finish_checked(output, valid), Err(Error::Phase));
}

#[test]
fn transfer_refusals_keep_destination_builder_pending_and_state_unchanged() {
    for state in [ResidentState::Ready, ResidentState::Entering, ResidentState::Failed, ResidentState::Complete, ResidentState::Transferred] {
        let (b, output) = scheduled_fixture();
        let destination = Builder(b.0.clone()).finish(output).unwrap();
        let mut owner = ResidentBuilder::from_builder(b);
        owner.state = state; // Private lifecycle oracle, not a public transition.
        owner.pending_affine = coefficients().to_vec();
        let before = program(&owner).clone();
        let pending = (owner.pending_affine.as_ptr(), owner.pending_affine.capacity());
        let mut out = Some(destination.clone());
        let destination_pointer = out.as_ref().unwrap().public_affines[0].as_ptr();
        assert_eq!(owner.transfer_complete(&mut out), Err(Error::Phase));
        assert_eq!(owner.state, state);
        assert_program_eq(program(&owner), &before);
        assert_program_eq(out.as_ref().unwrap(), &destination);
        assert_eq!(out.as_ref().unwrap().public_affines[0].as_ptr(), destination_pointer);
        assert_eq!((owner.pending_affine.as_ptr(), owner.pending_affine.capacity()), pending);
        assert_eq!(owner.pending_affine, coefficients());
        if state != ResidentState::Complete {
            let mut empty = None;
            assert_eq!(owner.transfer_complete(&mut empty), Err(Error::Phase));
            assert!(empty.is_none());
            assert_eq!(owner.state, state);
            assert_program_eq(program(&owner), &before);
        }
    }
}

#[test]
fn missing_builder_errors_latch_on_entry_but_transfer_refusal_changes_nothing() {
    for checked in [false, true] {
        let mut owner = ResidentBuilder::from_builder(builder());
        owner.builder = None; // Invariant-failure oracle only.
        let result = if checked { owner.finish_checked(Value(1), Value(0)) } else { owner.finish(Value(1)) };
        assert_eq!(result, Err(Error::Phase));
        assert_eq!(owner.state, ResidentState::Failed);
    }
    let mut owner = ResidentBuilder::from_builder(builder());
    owner.builder = None;
    assert_eq!(owner.affine_lanes(Value(0), &coefficients()), Err(Error::Phase));
    assert_eq!(owner.state, ResidentState::Failed);
    owner.state = ResidentState::Complete;
    let mut out = None;
    assert_eq!(owner.transfer_complete(&mut out), Err(Error::Phase));
    assert_eq!(owner.state, ResidentState::Complete);
    assert!(owner.builder.is_none());
    assert!(out.is_none());
}

#[test]
fn complete_owner_rejects_mutation_and_repeat_finish_without_changing_completed_program() {
    let (b, output) = scheduled_fixture();
    let mut owner = ResidentBuilder::from_builder(b);
    owner.finish(output).unwrap();
    let before = program(&owner).clone();
    assert_eq!(owner.affine_lanes(Value(0), &coefficients()), Err(Error::Phase));
    assert_eq!(owner.finish(output), Err(Error::Phase));
    assert_eq!(owner.finish_checked(output, Value(0)), Err(Error::Phase));
    assert_eq!(owner.state, ResidentState::Complete);
    assert_program_eq(program(&owner), &before);
}

#[test]
fn consuming_legacy_affine_remains_reusable_after_error() {
    let mut b = builder();
    assert_eq!(b.affine_lanes(Value(255), &[]), Err(Error::Binding));
    let output = b.affine_lanes(Value(0), &coefficients()).unwrap();
    let finished = b.finish(output).unwrap();
    assert_eq!(finished.public_affine_count, 1);
    assert_eq!(finished.public_affines[0], coefficients());
    assert_ne!(finished.digest(), [0; 32]);
}
