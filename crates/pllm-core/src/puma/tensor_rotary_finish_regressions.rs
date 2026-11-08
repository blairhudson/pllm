//! AUTHOR-ONLY / UNRUN. Shared-helper parity is not independent numeric, PRSS or
//! runtime evidence. Hold schedule, outputs and ledger outside unwind closures.
use super::*;
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

const ONE: i64 = 262_144;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tables {
    backing: [(*const Affine, usize, usize); 2],
    contents: [[Affine; 4]; 2],
}

fn tables(p: &Program) -> Tables {
    assert_eq!(p.public_affine_count, 2);
    Tables {
        backing: std::array::from_fn(|i| {
            let table = &p.public_affines[i];
            assert_eq!(table.len(), 4);
            let extent = crate::exact_buffer::layout_bytes::<Affine>(table.len()).unwrap();
            assert_eq!(extent, 4 * std::mem::size_of::<Affine>());
            (table.as_ptr(), table.len(), extent)
        }),
        contents: std::array::from_fn(|i| (&*p.public_affines[i]).try_into().unwrap()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Observation {
    installs: usize,
    tables: [Option<Tables>; 2],
}

thread_local! {
    static OBSERVE: RefCell<Option<Observation>> = const { RefCell::new(None) };
}

// Called only AFTER the actual recipe Builder is installed in the schedule's
// inline ResidentBuilder, never from preparation or the infallible commit gap.
pub(super) fn installed(resident: &ResidentBuilder) {
    OBSERVE.with(|slot| {
        if let Some(observed) = &mut *slot.borrow_mut() {
            assert_eq!(resident.state, ResidentState::Ready);
            assert!(resident.pending_affine.is_empty());
            let p = &resident.builder.as_ref().unwrap().0;
            observed.tables[observed.installs] = Some(tables(p));
            observed.installs += 1;
        }
    });
}

fn observe() {
    OBSERVE.with(|slot| assert!(slot.replace(Some(Observation {
        installs: 0, tables: [None, None],
    })).is_none()));
}

fn observed() -> Observation {
    OBSERVE.with(|slot| slot.borrow_mut().take().unwrap())
}

fn input() -> Interval { Interval::new(-32, 32).unwrap() }

fn source(tail: bool) -> ResidentFactorySource<'static> {
    let mut cosine = Vec::with_capacity(if tail { 7 } else { 9 });
    let mut sine = Vec::with_capacity(if tail { 8 } else { 10 });
    cosine.extend_from_slice(if tail { &[0, ONE] } else { &[ONE, ONE / 2] });
    sine.extend_from_slice(if tail { &[ONE, 0] } else { &[0, -ONE / 2] });
    Construction::RotaryHalf { rows: 1, width: 4, cosine_len: 2, sine_len: 2 }
        .resident_rotary_parameters(input(), cosine, sine)
}

#[derive(Debug, PartialEq, Eq)]
struct BodySnapshot {
    geometry: (usize, usize, usize, usize),
    input: Interval,
    backing: [(*const i64, usize, usize); 2],
    values: [[i64; 2]; 2],
}

fn body(body: &FactoryBody<'_>) -> BodySnapshot {
    let geometry = match body.construction {
        Construction::RotaryHalf { rows, width, cosine_len, sine_len } =>
            (rows, width, cosine_len, sine_len),
        _ => panic!("rotary fixture required"),
    };
    match &body.parameters {
        ParameterStore::Rotary { input, cosine, sine } => BodySnapshot {
            geometry, input: *input,
            backing: [(cosine.as_ptr(), cosine.len(), cosine.capacity()),
                (sine.as_ptr(), sine.len(), sine.capacity())],
            values: [cosine.as_slice().try_into().unwrap(), sine.as_slice().try_into().unwrap()],
        },
        _ => panic!("owned fixture required"),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct NativeSnapshot {
    nodes: [Option<Node>; MAX_NODES],
    len: usize,
    lanes: usize,
    profile: &'static str,
    digest: [u8; 32],
    last_use: [u8; MAX_NODES],
    slots: [u8; MAX_NODES],
    registers: usize,
    bits: [u8; MAX_NODES],
    validity: Option<Value>,
    work: WorkBudget,
    tables: Tables,
}

fn native(p: &Program) -> NativeSnapshot {
    NativeSnapshot {
        nodes: p.nodes, len: p.len, lanes: p.lanes, profile: p.numeric_profile,
        digest: p.digest, last_use: p.last_use, slots: p.slots,
        registers: p.register_count, bits: p.bit_widths, validity: p.checked_validity,
        work: p.work, tables: tables(p),
    }
}

fn slot_snapshot(slot: &NativeFinishSlot) -> (Option<(ResidentState, Option<NativeSnapshot>, usize, usize, *const Affine)>, Option<NativeSnapshot>) {
    (slot.builder.as_ref().map(|b| (b.state, b.builder.as_ref().map(|b| native(&b.0)),
        b.pending_affine.len(), b.pending_affine.extent(), b.pending_affine.as_ptr())),
        slot.program.as_ref().map(native))
}

fn output(out: &Option<FactoryOutput>) -> Option<(NativeSnapshot, ProducedCertificate)> {
    out.as_ref().map(|o| (native(o.program()), o.certificate()))
}

fn ready(tail: bool, total: &mut WorkBudget) -> ResidentNativeSchedule<'static> {
    let mut full = Some(source(false));
    let mut tail_source = tail.then(|| source(true));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, tail.then_some(&mut tail_source), 3).unwrap();
    owner.bind().unwrap();
    let attempts = rotary_work::factory_attempts();
    owner.preflight(&mut WorkPreflightWorkspace::empty()).unwrap();
    assert_eq!(rotary_work::factory_attempts(), attempts);
    owner.admit(&mut CallerWorkLedger::new(total, u64::MAX, u64::MAX).unwrap()).unwrap();
    owner
}

fn literal_tables(p: &Program, tail: bool) {
    let cosine = if tail { [0, ONE, 0, ONE] } else { [ONE, ONE / 2, ONE, ONE / 2] };
    let sine = if tail { [-ONE, 0, ONE, 0] } else { [0, ONE / 2, 0, -ONE / 2] };
    // Independent literal sign/half/order coefficients, not numeric execution.
    assert_eq!(tables(p).contents, [cosine, sine].map(|row|
        row.map(|left| Affine { left, right: 0, constant: 0 })));
}

fn parity(actual: &FactoryOutput, expected: &FactoryOutput) {
    let a = actual.program();
    let e = expected.program();
    assert_eq!(a.nodes, e.nodes); // instruction order and every node's bounds
    assert_eq!((a.len, a.lanes), (e.len, e.lanes));
    assert_eq!(a.numeric_profile(), e.numeric_profile());
    assert_eq!(a.digest(), e.digest());
    assert_eq!(a.output_bounds(), e.output_bounds());
    assert_eq!(a.last_use, e.last_use);
    assert_eq!(a.slots, e.slots);
    assert_eq!(a.register_count, e.register_count);
    assert_eq!(a.bit_widths, e.bit_widths);
    assert_eq!(a.checked_validity, e.checked_validity);
    assert_eq!(a.public_affine_count, e.public_affine_count);
    assert_eq!(a.public_affines, e.public_affines);
    assert_eq!(a.work_budget(), e.work_budget());
    assert_eq!(actual.certificate(), expected.certificate());
}

fn legacy(body: &FactoryBody<'_>) -> FactoryOutput {
    let lens = body.lens();
    build_core(lens.construction, &lens.parameters).unwrap()
}

#[test]
fn admitted_rotary_finish_observes_both_resident_tables_and_transfers_identity() {
    let mut total = WorkBudget::local();
    let prior = total;
    let mut owner = ready(false, &mut total);
    let original = body(owner.full.as_ref().unwrap());
    let expected = legacy(owner.full.as_ref().unwrap());
    assert_eq!(total, prior.checked_add(owner.summary().unwrap().work).unwrap());
    let charged = total;
    let attempts = rotary_work::factory_attempts();
    let hooks = resident_builder::finish_hook_counts();
    let (mut full, mut tail) = (None, None);
    observe();
    owner.build_into(&mut full, &mut tail).unwrap();
    let seen = observed();
    assert_eq!(seen.installs, 1);
    assert_eq!(resident_builder::finish_hook_counts(), (hooks.0 + 1, hooks.1));
    assert_eq!(rotary_work::factory_attempts(), attempts + 1);
    assert_eq!(tables(full.as_ref().unwrap().program()), seen.tables[0].unwrap());
    literal_tables(full.as_ref().unwrap().program(), false);
    parity(full.as_ref().unwrap(), &expected);
    assert_eq!(full.as_ref().unwrap().certificate(), ProducedCertificate::None);
    assert!(tail.is_none());
    assert_eq!(owner.phase, Phase::Complete);
    assert_eq!(owner.full_finish.builder.as_ref().unwrap().state, ResidentState::Transferred);
    assert!(owner.full_finish.builder.as_ref().unwrap().builder.is_none());
    assert!(owner.full_finish.program.is_none());
    assert_eq!(body(owner.full.as_ref().unwrap()), original);
    assert_eq!(owner.count, 3);
    assert_eq!(total, charged);
}

#[test]
fn admitted_rotary_panic_inside_finish_retains_both_tables_failed_and_burned() {
    let mut total = WorkBudget::default();
    let mut owner = ready(false, &mut total);
    let original = body(owner.full.as_ref().unwrap());
    let charged = total;
    let hooks = resident_builder::finish_hook_counts();
    let (mut full, mut tail) = (None, None);
    observe();
    resident_builder::arm_finish_panic(1);
    assert!(catch_unwind(AssertUnwindSafe(|| owner.build_into(&mut full, &mut tail))).is_err());
    let seen = observed();
    assert_eq!(seen.installs, 1);
    assert_eq!(resident_builder::finish_hook_counts(), (hooks.0 + 1, hooks.1 + 1));
    let b = owner.full_finish.builder.as_ref().unwrap();
    assert_eq!(b.state, ResidentState::Failed);
    let p = &b.builder.as_ref().unwrap().0;
    assert_eq!(tables(p), seen.tables[0].unwrap());
    literal_tables(p, false);
    assert!(p.register_count > 0); // real partial scheduling, not factory entry
    assert_eq!(owner.phase, Phase::Burned);
    assert!(owner.full_finish.program.is_none());
    assert!(full.is_none() && tail.is_none());
    assert_eq!(body(owner.full.as_ref().unwrap()), original);
    assert_eq!(total, charged);
}

#[test]
fn private_malformed_finish_value_error_retains_both_owners_not_admitted_reachability() {
    // Deliberately private finish oracle. Value(0) is NOT the valid admitted
    // recipe output; no public path or allocator-failure reachability is claimed.
    let (builder, valid) = numeric::prepare_rotary_half(1, 4, input(), &[ONE, ONE / 2], &[0, -ONE / 2]).unwrap();
    assert_ne!(valid, Value(0));
    let before = native(&builder.0);
    let mut slot = NativeFinishSlot::empty();
    slot.builder = Some(ResidentBuilder::from_builder(builder));
    let mut out = None;
    let hooks = resident_builder::finish_hook_counts();
    assert_eq!(slot.finish_into(Value(0), &mut out), Err(Error::Binding));
    assert_eq!(resident_builder::finish_hook_counts(), hooks);
    let b = slot.builder.as_ref().unwrap();
    assert_eq!(b.state, ResidentState::Failed);
    assert_eq!(native(&b.builder.as_ref().unwrap().0), before);
    assert!(slot.program.is_none() && out.is_none());
    assert_eq!(slot.finish_into(valid, &mut out), Err(Error::Phase));
    assert_eq!(native(&slot.builder.as_ref().unwrap().builder.as_ref().unwrap().0), before);
}

#[test]
fn failed_and_completed_rotary_all_schedule_reentries_preserve_every_owner_and_ledger() {
    for fail in [false, true] {
        let mut total = WorkBudget::local();
        let mut owner = ready(true, &mut total);
        let (mut full, mut tail) = (None, None);
        if fail {
            resident_builder::arm_finish_panic(1);
            assert!(catch_unwind(AssertUnwindSafe(|| owner.build_into(&mut full, &mut tail))).is_err());
        } else {
            owner.build_into(&mut full, &mut tail).unwrap();
        }
        assert_eq!(owner.phase, if fail { Phase::Burned } else { Phase::Complete });
        let mut supplied_full = Some(source(true));
        let mut supplied_tail = Some(source(false));
        let mut workspace = WorkPreflightWorkspace::empty();
        let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
        macro_rules! snapshot {
            () => { (
                (body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap()), owner.count, owner.phase, owner.summary()),
                (slot_snapshot(&owner.full_finish), slot_snapshot(&owner.tail_finish)),
                (output(&full), output(&tail)),
                (body(&supplied_full.as_ref().unwrap().body), body(&supplied_tail.as_ref().unwrap().body)),
                ledger.total(), workspace.is_poisoned(), rotary_work::factory_attempts(), resident_builder::finish_hook_counts(),
            ) };
        }
        let before = snapshot!();
        let phase = WorkPreflightError::Native(Error::Phase);
        assert_eq!(owner.install(&mut supplied_full, Some(&mut supplied_tail), 77), Err(phase));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.bind(), Err(phase));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.preflight(&mut workspace), Err(phase));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.admit(&mut ledger), Err(phase));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.build_into(&mut full, &mut tail), Err(Error::Phase));
        assert_eq!(snapshot!(), before);
        // Also supply empty outputs after completion, retaining real outputs in
        // caller storage; phase refusal must not depend on occupied outputs.
        let (mut empty_full, mut empty_tail) = (None, None);
        assert_eq!(owner.build_into(&mut empty_full, &mut empty_tail), Err(Error::Phase));
        assert!(empty_full.is_none() && empty_tail.is_none());
        assert_eq!(snapshot!(), before);
    }
}

#[test]
fn occupied_output_burns_before_any_native_entry_preserving_source_destination_and_slots() {
    for occupied_tail in [false, true] {
        let mut total = WorkBudget::default();
        let mut owner = ready(true, &mut total);
        let existing = legacy(owner.full.as_ref().unwrap());
        let (mut full, mut tail) = if occupied_tail { (None, Some(existing)) } else { (Some(existing), None) };
        let originals = (body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap()));
        let outputs = (output(&full), output(&tail));
        let charged = total;
        let hooks = resident_builder::finish_hook_counts();
        let attempts = rotary_work::factory_attempts();
        observe();
        assert_eq!(owner.build_into(&mut full, &mut tail), Err(Error::Binding));
        assert_eq!(observed().installs, 0);
        assert_eq!(owner.phase, Phase::Burned);
        assert_eq!(rotary_work::factory_attempts(), attempts);
        assert_eq!(resident_builder::finish_hook_counts(), hooks);
        assert!(owner.full_finish.builder.is_none() && owner.full_finish.program.is_none());
        assert!(owner.tail_finish.builder.is_none() && owner.tail_finish.program.is_none());
        assert_eq!((output(&full), output(&tail)), outputs);
        assert_eq!((body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap())), originals);
        assert_eq!(owner.count, 3);
        assert_eq!(total, charged);
    }
}

#[test]
fn tail_second_finish_panic_preserves_completed_full_and_resident_tail_tables() {
    let mut total = WorkBudget::default();
    let mut owner = ready(true, &mut total);
    let expected = legacy(owner.full.as_ref().unwrap());
    let originals = (body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap()));
    let charged = total;
    let hooks = resident_builder::finish_hook_counts();
    let attempts = rotary_work::factory_attempts();
    let (mut full, mut tail) = (None, None);
    observe();
    resident_builder::arm_finish_panic(2); // SECOND actual finish, NOT factory entry
    assert!(catch_unwind(AssertUnwindSafe(|| owner.build_into(&mut full, &mut tail))).is_err());
    let seen = observed();
    assert_eq!(seen.installs, 2);
    assert_eq!(resident_builder::finish_hook_counts(), (hooks.0 + 2, hooks.1 + 1));
    assert_eq!(rotary_work::factory_attempts(), attempts + 2);
    assert_eq!(tables(full.as_ref().unwrap().program()), seen.tables[0].unwrap());
    parity(full.as_ref().unwrap(), &expected);
    literal_tables(full.as_ref().unwrap().program(), false);
    assert_eq!(owner.full_finish.builder.as_ref().unwrap().state, ResidentState::Transferred);
    let b = owner.tail_finish.builder.as_ref().unwrap();
    assert_eq!(b.state, ResidentState::Failed);
    assert_eq!(tables(&b.builder.as_ref().unwrap().0), seen.tables[1].unwrap());
    literal_tables(&b.builder.as_ref().unwrap().0, true);
    assert!(b.builder.as_ref().unwrap().0.register_count > 0);
    assert!(tail.is_none());
    assert_eq!(owner.phase, Phase::Burned);
    assert_eq!((body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap())), originals);
    let full_before = output(&full);
    let tail_before = slot_snapshot(&owner.tail_finish);
    assert_eq!(owner.build_into(&mut full, &mut tail), Err(Error::Phase));
    assert_eq!(output(&full), full_before);
    assert_eq!(slot_snapshot(&owner.tail_finish), tail_before);
    assert_eq!(total, charged);
}

#[test]
fn complete_transfer_occupied_destination_preserves_both_then_empty_destination_moves_identity() {
    let (builder, value) = numeric::prepare_rotary_half(1, 4, input(), &[ONE, ONE / 2], &[0, -ONE / 2]).unwrap();
    let original = tables(&builder.0);
    let mut resident = ResidentBuilder::from_builder(builder);
    resident.finish(value).unwrap();
    let before = native(&resident.builder.as_ref().unwrap().0);
    let mut occupied = Some(Program::rotary_half(1, 4, input(), &[0, ONE], &[ONE, 0]).unwrap());
    let destination = native(occupied.as_ref().unwrap());
    assert_eq!(resident.transfer_complete(&mut occupied), Err(Error::Phase));
    assert_eq!(resident.state, ResidentState::Complete);
    assert_eq!(native(&resident.builder.as_ref().unwrap().0), before);
    assert_eq!(native(occupied.as_ref().unwrap()), destination);
    let mut empty = None;
    resident.transfer_complete(&mut empty).unwrap();
    assert_eq!(native(empty.as_ref().unwrap()), before);
    assert_eq!(tables(empty.as_ref().unwrap()), original);
    assert_eq!(resident.state, ResidentState::Transferred);
    assert!(resident.builder.is_none());
    assert_eq!(native(occupied.as_ref().unwrap()), destination);
}

#[test]
fn legacy_rotary_prepare_error_precedence_and_factory_entry_instrumentation_unchanged() {
    // Literal original winners, NOT geometry-only quote error precedence.
    for (rows, width, cosine, sine, expected) in [
        (usize::MAX, 2, &[][..], &[][..], Error::Bounds),
        (0, 0, &[ONE + 1][..], &[][..], Error::Shape),
        (1, 3, &[ONE + 1][..], &[0][..], Error::Shape),
        (1, 4, &[ONE + 1][..], &[0][..], Error::Shape),
        (1, 4, &[ONE + 1, 0][..], &[0, 0][..], Error::Bounds),
    ] {
        let attempts = rotary_work::factory_attempts();
        assert!(matches!(Program::rotary_half(rows, width, input(), cosine, sine), Err(e) if e == expected));
        assert_eq!(rotary_work::factory_attempts(), attempts + 1);
        assert!(matches!(numeric::prepare_rotary_half(rows, width, input(), cosine, sine), Err(e) if e == expected));
        assert_eq!(rotary_work::factory_attempts(), attempts + 2);
    }
    // Tile limit wins before domain validation for shape-valid geometry.
    let width = MAX_TILE_LANES + 2;
    let coefficients = vec![ONE + 1; width / 2];
    assert!(matches!(Program::rotary_half(1, width, input(), &coefficients, &coefficients), Err(Error::Bounds)));
}

#[test]
fn full_tail_success_preserves_legacy_metadata_aggregation_and_immutable_binding() {
    let mut total = WorkBudget::local();
    let prior = total;
    let mut owner = ready(true, &mut total);
    let originals = (body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap()));
    let expected_full = legacy(owner.full.as_ref().unwrap());
    let expected_tail = legacy(owner.tail.as_ref().unwrap());
    let summary = owner.summary().unwrap();
    assert_eq!(summary.full.invocations, 3);
    assert_eq!(summary.tail.unwrap().invocations, 1);
    assert_eq!(summary.full.unit_work, expected_full.program().work_budget());
    assert_eq!(summary.tail.unwrap().unit_work, expected_tail.program().work_budget());
    assert_eq!(summary.work, summary.full.unit_work.checked_mul(3).unwrap()
        .checked_add(summary.tail.unwrap().unit_work).unwrap());
    assert_eq!(total, prior.checked_add(summary.work).unwrap());
    let hooks = resident_builder::finish_hook_counts();
    let (mut full, mut tail) = (None, None);
    observe();
    owner.build_into(&mut full, &mut tail).unwrap();
    let seen = observed();
    assert_eq!(seen.installs, 2);
    assert_eq!(resident_builder::finish_hook_counts(), (hooks.0 + 2, hooks.1));
    parity(full.as_ref().unwrap(), &expected_full);
    parity(tail.as_ref().unwrap(), &expected_tail);
    assert_eq!(tables(full.as_ref().unwrap().program()), seen.tables[0].unwrap());
    assert_eq!(tables(tail.as_ref().unwrap().program()), seen.tables[1].unwrap());
    literal_tables(full.as_ref().unwrap().program(), false);
    literal_tables(tail.as_ref().unwrap().program(), true);
    assert_eq!((body(owner.full.as_ref().unwrap()), body(owner.tail.as_ref().unwrap())), originals);
    assert_eq!(owner.count, 3);
    assert_eq!(owner.summary(), Some(summary));
    assert_eq!(total, prior.checked_add(summary.work).unwrap());
}

#[test]
fn nonrotary_full_tail_and_both_mixed_orders_remain_successful() {
    for rotary in [[false, false], [true, false], [false, true]] {
        let make = |is_rotary, tail| if is_rotary { source(tail) } else {
            Construction::Negative { lanes: 4 }.resident_parameters(FactoryParameters::Negative { input: input() })
        };
        let mut full_source = Some(make(rotary[0], false));
        let mut tail_source = Some(make(rotary[1], true));
        let mut owner = WorkPreflightWorkspace::resident_schedule();
        owner.install(&mut full_source, Some(&mut tail_source), 2).unwrap();
        owner.bind().unwrap();
        owner.preflight(&mut WorkPreflightWorkspace::empty()).unwrap();
        let expected_full = legacy(owner.full.as_ref().unwrap());
        let expected_tail = legacy(owner.tail.as_ref().unwrap());
        let mut total = WorkBudget::default();
        owner.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
        let charged = total;
        let attempts = rotary_work::factory_attempts();
        let (mut full, mut tail) = (None, None);
        observe();
        owner.build_into(&mut full, &mut tail).unwrap();
        let count = usize::from(rotary[0]) + usize::from(rotary[1]);
        assert_eq!(observed().installs, count);
        assert_eq!(rotary_work::factory_attempts(), attempts + count);
        parity(full.as_ref().unwrap(), &expected_full);
        parity(tail.as_ref().unwrap(), &expected_tail);
        assert_eq!(owner.full_finish.builder.is_some(), rotary[0]);
        assert_eq!(owner.tail_finish.builder.is_some(), rotary[1]);
        assert_eq!(owner.phase, Phase::Complete);
        assert_eq!(owner.count, 2);
        assert_eq!(total, charged);
    }
}
