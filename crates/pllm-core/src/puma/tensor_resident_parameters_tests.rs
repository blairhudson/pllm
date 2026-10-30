use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

const ONE: i64 = 1 << 18;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Backing {
    allocation: (*const i64, usize, usize, *const i64, usize, usize),
    cosine: [i64; 2],
    sine: [i64; 2],
    input: Interval,
    construction: (u8, [usize; 4]),
}

fn input() -> Interval {
    Interval::new(-32, 32).unwrap()
}

fn source(rows: usize) -> ResidentFactorySource<'static> {
    let mut cosine = Vec::with_capacity(7);
    let mut sine = Vec::with_capacity(9);
    cosine.extend_from_slice(&[ONE, ONE / 2]);
    sine.extend_from_slice(&[0, -ONE / 2]);
    Construction::RotaryHalf { rows, width: 4, cosine_len: 2, sine_len: 2 }
        .resident_rotary_parameters(input(), cosine, sine)
}

fn distinct_tail() -> ResidentFactorySource<'static> {
    let mut cosine = Vec::with_capacity(5);
    let mut sine = Vec::with_capacity(6);
    cosine.extend_from_slice(&[0, ONE]);
    sine.extend_from_slice(&[ONE, 0]);
    Construction::RotaryHalf { rows: 1, width: 4, cosine_len: 2, sine_len: 2 }
        .resident_rotary_parameters(input(), cosine, sine)
}

fn backing(body: &FactoryBody<'_>) -> Backing {
    // Independent owned scalar copies, NOT references or another owner of the
    // original backing. These bounded fixtures have at most two coefficients.
    let construction = match body.construction {
        Construction::RotaryHalf { rows, width, cosine_len, sine_len } =>
            (0, [rows, width, cosine_len, sine_len]),
        Construction::Negative { lanes } => (1, [lanes, 0, 0, 0]),
        Construction::GuardedSilu { lanes } => (2, [lanes, 0, 0, 0]),
        _ => panic!("unexpected construction in rotary-owned fixture"),
    };
    match &body.parameters {
        ParameterStore::Rotary { input, cosine, sine } => {
            assert!(cosine.len() <= 2 && sine.len() <= 2);
            Backing {
                allocation: (
                    cosine.as_ptr(), cosine.len(), cosine.capacity(),
                    sine.as_ptr(), sine.len(), sine.capacity(),
                ),
                cosine: [cosine.first().copied().unwrap_or(0), cosine.get(1).copied().unwrap_or(0)],
                sine: [sine.first().copied().unwrap_or(0), sine.get(1).copied().unwrap_or(0)],
                input: *input,
                construction,
            }
        }
        _ => panic!("expected original owned rotary rows"),
    }
}

fn backings(owner: &ResidentNativeSchedule<'_>) -> (Backing, Option<Backing>) {
    (backing(owner.full.as_ref().unwrap()), owner.tail.as_ref().map(backing))
}

fn installed(rows: usize, count: u64) -> ResidentNativeSchedule<'static> {
    let mut source = Some(source(rows));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut source, None, count).unwrap();
    assert!(source.is_none());
    owner
}

fn priced() -> ResidentNativeSchedule<'static> {
    let mut owner = installed(1, 3);
    owner.bind().unwrap();
    owner.preflight(&mut WorkPreflightWorkspace::empty()).unwrap();
    owner
}

fn phase_error() -> WorkPreflightError {
    WorkPreflightError::Native(Error::Phase)
}

fn panic_at(stage: TestStage, invocation: usize) {
    PANIC_STAGE.with(|slot| {
        assert!(slot.replace(Some((stage, invocation))).is_none());
    });
}

fn same_program(actual: &Program, expected: &Program) {
    assert_eq!(actual.nodes, expected.nodes);
    assert_eq!(actual.len, expected.len);
    assert_eq!(actual.lanes, expected.lanes);
    assert_eq!(actual.digest(), expected.digest());
    assert_eq!(actual.work_budget(), expected.work_budget());
    assert_eq!(actual.numeric_profile(), expected.numeric_profile());
    assert_eq!(actual.public_affines, expected.public_affines);
    assert_eq!(actual.public_affine_count, expected.public_affine_count);
    assert_eq!(actual.last_use, expected.last_use);
    assert_eq!(actual.slots, expected.slots);
    assert_eq!(actual.register_count, expected.register_count);
    assert_eq!(actual.checked_validity, expected.checked_validity);
    assert_eq!(actual.bit_widths, expected.bit_widths);
    assert_eq!(actual.output_bounds(), expected.output_bounds());
}

fn same_output(actual: &FactoryOutput, expected: &FactoryOutput) {
    // Shared legacy parity ONLY, not independent numeric or digest proof.
    same_program(actual.program(), expected.program());
    assert_eq!(actual.certificate(), expected.certificate());
}

#[derive(Debug, PartialEq, Eq)]
struct OutputSnapshot {
    nodes: [Option<Node>; MAX_NODES],
    len: usize,
    lanes: usize,
    digest: [u8; 32],
    numeric_profile: &'static str,
    checked_validity: Option<Value>,
    last_use: [u8; MAX_NODES],
    slots: [u8; MAX_NODES],
    register_count: usize,
    bit_widths: [u8; MAX_NODES],
    public_affines: [Vec<Affine>; 2],
    public_affine_count: usize,
    table_backing: [Option<(*const Affine, usize)>; 2],
    work: WorkBudget,
    certificate: ProducedCertificate,
}

fn output_snapshot(output: &FactoryOutput) -> OutputSnapshot {
    let p = output.program();
    OutputSnapshot {
        nodes: p.nodes,
        len: p.len,
        lanes: p.lanes,
        digest: p.digest,
        numeric_profile: p.numeric_profile,
        checked_validity: p.checked_validity,
        last_use: p.last_use,
        slots: p.slots,
        register_count: p.register_count,
        bit_widths: p.bit_widths,
        // Copies of coefficients, not Program clones or owners of its tables.
        public_affines: std::array::from_fn(|i| p.public_affines[i].to_vec()),
        public_affine_count: p.public_affine_count,
        table_backing: std::array::from_fn(|i| {
            let table = &p.public_affines[i];
            (!table.is_empty()).then_some((table.as_ptr(), table.capacity()))
        }),
        work: p.work,
        certificate: output.certificate(),
    }
}

fn assert_rotary_tables(output: &FactoryOutput, cosine: [i64; 4], sine: [i64; 4]) {
    // Independent literal row/half/sign oracle. No audited bind/price/build
    // helper constructs these expected coefficients. Not an execution oracle.
    let p = output.program();
    assert_eq!(p.lanes, 4);
    assert_eq!(p.public_affine_count, 2);
    assert_eq!(p.public_affines[0].as_slice(), &cosine.map(|left| Affine { left, right: 0, constant: 0 }));
    assert_eq!(p.public_affines[1].as_slice(), &sine.map(|left| Affine { left, right: 0, constant: 0 }));
    assert_eq!(output.certificate(), ProducedCertificate::None);
}

fn assert_full_tail_aggregation(summary: WorkSummary) {
    assert_eq!(summary.full.invocations, 2);
    let tail = summary.tail.unwrap();
    assert_eq!(tail.invocations, 1);
    let f = summary.full.unit_work;
    let t = tail.unit_work;
    // Independent small-fixture aggregate arithmetic across ALL five fields.
    // Unit-price/numeric correctness below remains shared legacy parity only.
    assert_eq!(summary.work.operations(), 2 * f.operations() + t.operations());
    let actual = summary.work.cursors();
    let f = f.cursors();
    let t = t.cursors();
    assert_eq!(actual.aes_work, 2 * f.aes_work + t.aes_work);
    assert_eq!(actual.public_blocks, 2 * f.public_blocks + t.public_blocks);
    assert_eq!(actual.pair_blocks, 2 * f.pair_blocks + t.pair_blocks);
    assert_eq!(actual.private_blocks, 2 * f.private_blocks + t.private_blocks);
}

#[test]
fn parameter_install_moves_complete_original_full_tail_backings_once() {
    let mut full = Some(source(1));
    let mut tail = Some(source(1));
    let original = (backing(&full.as_ref().unwrap().body), Some(backing(&tail.as_ref().unwrap().body)));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 7).unwrap();
    assert!(full.is_none() && tail.is_none());
    assert_eq!(backings(&owner), original);
    assert_eq!(owner.count, 7);
    assert_eq!(owner.phase, Phase::Installed);
    assert!(owner.summary().is_none());
    assert_eq!(owner.install(&mut full, Some(&mut tail), 8), Err(phase_error()));
    assert_eq!(backings(&owner), original);
    assert_eq!(owner.count, 7);
}

#[test]
fn parameter_install_missing_full_or_tail_refuses_before_either_take() {
    for missing_full in [true, false] {
        let mut full = (!missing_full).then(|| source(1));
        let mut tail = missing_full.then(|| source(1));
        let slots_before = (full.as_ref().map(|source| backing(&source.body)),
            tail.as_ref().map(|source| backing(&source.body)));
        let original = full.as_ref().or(tail.as_ref()).map(|source| backing(&source.body)).unwrap();
        let mut owner = WorkPreflightWorkspace::resident_schedule();
        assert_eq!(owner.install(&mut full, Some(&mut tail), 3), Err(phase_error()));
        assert_eq!(owner.phase, Phase::Empty);
        assert_eq!(owner.count, 0);
        assert!(owner.full.is_none() && owner.tail.is_none() && owner.priced.is_none());
        assert_eq!(full.as_ref().or(tail.as_ref()).map(|source| backing(&source.body)), Some(original));
        assert_eq!((full.as_ref().map(|source| backing(&source.body)),
            tail.as_ref().map(|source| backing(&source.body))), slots_before);
    }
}

#[test]
fn parameter_install_occupied_resident_preserves_all_three_original_owners() {
    let mut owner = installed(1, 2);
    let resident_before = backings(&owner);
    let mut full = Some(source(1));
    let mut tail = Some(source(1));
    let full_before = backing(&full.as_ref().unwrap().body);
    let tail_before = backing(&tail.as_ref().unwrap().body);
    assert_eq!(owner.install(&mut full, Some(&mut tail), 9), Err(phase_error()));
    assert_eq!(backings(&owner), resident_before);
    assert_eq!(backing(&full.as_ref().unwrap().body), full_before);
    assert_eq!(backing(&tail.as_ref().unwrap().body), tail_before);
    assert_eq!(owner.count, 2);
}

#[test]
fn parameter_binding_preserves_legacy_length_mismatch_and_unsupported_precedence() {
    for (construction, expected) in [
        (Construction::RotaryHalf { rows: 0, width: 0, cosine_len: 3, sine_len: 2 }, WorkPreflightError::ParameterBinding),
        (Construction::Negative { lanes: 0 }, WorkPreflightError::ParameterBinding),
        (Construction::GuardedSilu { lanes: 0 }, WorkPreflightError::Unsupported),
    ] {
        let cosine = [ONE, ONE / 2];
        let sine = [0, -ONE / 2];
        let legacy = construction.bind(FactoryParameters::RotaryHalf { input: input(), cosine: &cosine, sine: &sine });
        assert!(matches!(legacy, Err(error) if error == expected));
        let mut full = Some(construction.resident_rotary_parameters(input(), cosine.to_vec(), sine.to_vec()));
        let before = backing(&full.as_ref().unwrap().body);
        let mut owner = WorkPreflightWorkspace::resident_schedule();
        owner.install(&mut full, None, 0).unwrap();
        assert_eq!(owner.bind(), Err(expected));
        assert_eq!(backings(&owner).0, before);
        assert_eq!(owner.phase, Phase::BindingEntered);
        assert_eq!(owner.bind(), Err(phase_error()));
        assert_eq!(backings(&owner).0, before);
        assert_eq!(owner.preflight(&mut WorkPreflightWorkspace::empty()), Err(phase_error()));
        assert_eq!(backings(&owner).0, before);
    }
}

#[test]
fn parameter_binding_unwind_keeps_both_original_backings_terminal() {
    let mut full = Some(source(1));
    let mut tail = Some(source(1));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 2).unwrap();
    let before = backings(&owner);
    panic_at(TestStage::Bind, 1);
    assert!(catch_unwind(AssertUnwindSafe(|| owner.bind())).is_err());
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.phase, Phase::BindingEntered);
    assert_eq!(owner.bind(), Err(phase_error()));
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_tail_binding_error_keeps_both_original_backings_without_pricing() {
    let mut full = Some(source(1));
    let mut tail = Some(Construction::Negative { lanes: 0 }
        .resident_rotary_parameters(input(), vec![ONE], vec![0]));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 2).unwrap();
    let before = backings(&owner);
    assert_eq!(owner.bind(), Err(WorkPreflightError::ParameterBinding));
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.phase, Phase::BindingEntered);
    assert!(owner.summary().is_none());
    assert_eq!(owner.bind(), Err(phase_error()));
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_preflight_native_full_error_precedes_zero_count_and_poisons_only_on_err() {
    let mut owner = installed(0, 0);
    let before = backings(&owner);
    owner.bind().unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    assert_eq!(owner.preflight(&mut workspace), Err(WorkPreflightError::Native(Error::Shape)));
    assert!(workspace.is_poisoned());
    assert_eq!(owner.phase, Phase::PreflightEntered);
    assert_eq!(backings(&owner), before);
    workspace.reset();
    assert_eq!(owner.preflight(&mut workspace), Err(phase_error()));
    assert!(!workspace.is_poisoned());
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_preflight_valid_full_zero_count_matches_consuming_legacy() {
    let mut owner = installed(1, 0);
    let before = backings(&owner);
    owner.bind().unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    assert_eq!(owner.preflight(&mut workspace), Err(WorkPreflightError::EmptySchedule));
    assert!(workspace.is_poisoned());
    assert_eq!(backings(&owner), before);
    let lens = owner.full.as_ref().unwrap().lens();
    let mut legacy_workspace = WorkPreflightWorkspace::empty();
    assert!(matches!(lens.construction.bind(lens.parameters).unwrap().preflight(0, &mut legacy_workspace), Err(WorkPreflightError::EmptySchedule)));
    assert!(legacy_workspace.is_poisoned());
}

#[test]
fn parameter_preflight_multiplication_precedes_invalid_tail_price() {
    let mut full = Some(source(1));
    let mut tail = Some(source(0));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), u64::MAX).unwrap();
    let before = backings(&owner);
    owner.bind().unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    assert_eq!(owner.preflight(&mut workspace), Err(WorkPreflightError::Native(Error::Bounds)));
    assert!(workspace.is_poisoned());
    assert_eq!(backings(&owner), before);
    assert!(owner.summary().is_none());
}

#[test]
fn parameter_preflight_invalid_tail_retains_both_owners_and_no_summary() {
    let mut full = Some(source(1));
    let mut tail = Some(source(0));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 1).unwrap();
    let before = backings(&owner);
    owner.bind().unwrap();
    let mut workspace = WorkPreflightWorkspace::empty();
    assert_eq!(owner.preflight(&mut workspace), Err(WorkPreflightError::Native(Error::Shape)));
    assert!(workspace.is_poisoned());
    assert_eq!(backings(&owner), before);
    assert!(owner.summary().is_none());
}

#[test]
fn parameter_preflight_price_unwind_does_not_newly_poison_workspace_or_allow_retry() {
    let mut owner = installed(1, 2);
    owner.bind().unwrap();
    let before = backings(&owner);
    let mut workspace = WorkPreflightWorkspace::empty();
    panic_at(TestStage::Price, 1); // after actual native work query, before quote
    assert!(catch_unwind(AssertUnwindSafe(|| owner.preflight(&mut workspace))).is_err());
    assert!(!workspace.is_poisoned());
    assert_eq!(owner.phase, Phase::PreflightEntered);
    assert_eq!(backings(&owner), before);
    workspace.reset();
    assert_eq!(owner.preflight(&mut workspace), Err(phase_error()));
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_preflight_poisoned_workspace_wins_and_owner_remains_terminal_after_reset() {
    let mut owner = installed(1, 0);
    owner.bind().unwrap();
    let before = backings(&owner);
    let mut workspace = WorkPreflightWorkspace::empty();
    workspace.poisoned = true;
    assert_eq!(owner.preflight(&mut workspace), Err(WorkPreflightError::WorkspacePoisoned));
    assert_eq!(backings(&owner), before);
    workspace.reset();
    assert_eq!(owner.preflight(&mut workspace), Err(phase_error()));
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_admission_debits_same_immutable_body_count_all_five_fields_once() {
    let mut owner = priced();
    let before = backings(&owner);
    let summary = owner.summary().unwrap();
    assert_eq!(owner.count, 3);
    assert_eq!(summary.full.invocations, 3);
    assert_eq!(summary.work, summary.full.unit_work.checked_mul(3).unwrap());
    let prior = WorkBudget::rescale(1).unwrap();
    let mut total = prior;
    let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
    owner.admit(&mut ledger).unwrap();
    let expected = prior.checked_add(summary.work).unwrap();
    assert_eq!(ledger.total(), expected); // Eq covers operations and all four cursors
    assert_eq!(owner.phase, Phase::Admitted);
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.admit(&mut ledger), Err(phase_error()));
    assert_eq!(ledger.total(), expected);
    assert_eq!(owner.count, 3);
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_admission_denial_or_overflow_keeps_ledger_and_original_backings() {
    let aes_unit = WorkBudget::rescale(1).unwrap();
    let near_aes = aes_unit.checked_mul(u64::MAX / aes_unit.cursors().aes_work).unwrap();
    for (prior, operations, aes) in [
        (WorkBudget::default(), 0, u64::MAX),
        (WorkBudget::default(), u64::MAX, 0),
        (WorkBudget::local().checked_mul(u64::MAX).unwrap(), u64::MAX, u64::MAX),
        (near_aes, u64::MAX, u64::MAX),
    ] {
        let mut owner = priced();
        let before = backings(&owner);
        let mut total = prior;
        let mut ledger = CallerWorkLedger::new(&mut total, operations, aes).unwrap();
        assert_eq!(owner.admit(&mut ledger), Err(WorkPreflightError::Native(Error::Bounds)));
        assert_eq!(ledger.total(), prior);
        assert_eq!(backings(&owner), before);
        assert_eq!(owner.phase, Phase::AdmissionEntered);
        assert_eq!(owner.admit(&mut ledger), Err(phase_error()));
        assert_eq!(ledger.total(), prior);
        assert_eq!(backings(&owner), before);
    }
}

#[test]
fn parameter_admission_unwind_before_debit_keeps_owners_and_actual_ledger() {
    let mut owner = priced();
    let before = backings(&owner);
    let mut total = WorkBudget::local();
    let prior = total;
    let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
    panic_at(TestStage::Admit, 1);
    assert!(catch_unwind(AssertUnwindSafe(|| owner.admit(&mut ledger))).is_err());
    assert_eq!(ledger.total(), prior);
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.phase, Phase::AdmissionEntered);
    assert_eq!(owner.admit(&mut ledger), Err(phase_error()));
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_build_success_matches_legacy_full_tail_and_retains_original_backing() {
    let mut full = Some(source(1));
    let mut tail = Some(distinct_tail());
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 2).unwrap();
    let before = backings(&owner);
    owner.bind().unwrap();
    assert_eq!(backings(&owner), before);
    let mut workspace = WorkPreflightWorkspace::empty();
    owner.preflight(&mut workspace).unwrap();
    assert_eq!(backings(&owner), before);
    assert_full_tail_aggregation(owner.summary().unwrap());
    let full_lens = owner.full.as_ref().unwrap().lens();
    let tail_lens = owner.tail.as_ref().unwrap().lens();
    let mut legacy_workspace = WorkPreflightWorkspace::empty();
    let legacy = full_lens.construction.bind(full_lens.parameters).unwrap()
        .preflight_full_tail(2, tail_lens.construction.bind(tail_lens.parameters).unwrap(), &mut legacy_workspace).unwrap();
    assert_eq!(owner.summary(), Some(legacy.summary()));
    let mut legacy_total = WorkBudget::default();
    let legacy = CallerWorkLedger::new(&mut legacy_total, u64::MAX, u64::MAX).unwrap().admit(legacy).unwrap();
    let (legacy_full, legacy_tail) = legacy.into_parts();
    let expected_full = legacy_full.build().unwrap();
    let expected_tail = legacy_tail.unwrap().build().unwrap();
    // Separate reference outputs, shared legacy parity (NOT independent
    // numeric/digest proof). Literal coefficient oracles follow below.
    let mut total = WorkBudget::default();
    owner.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
    assert_eq!(total, legacy_total);
    assert_eq!(total, owner.summary().unwrap().work);
    assert_eq!(owner.count, 2);
    assert_eq!(backings(&owner), before);
    let mut full_out = None;
    let mut tail_out = None;
    owner.build_into(&mut full_out, &mut tail_out).unwrap();
    same_output(full_out.as_ref().unwrap(), &expected_full);
    same_output(tail_out.as_ref().unwrap(), &expected_tail);
    assert_rotary_tables(full_out.as_ref().unwrap(),
        [ONE, ONE / 2, ONE, ONE / 2], [0, ONE / 2, 0, -ONE / 2]);
    assert_rotary_tables(tail_out.as_ref().unwrap(),
        [0, ONE, 0, ONE], [-ONE, 0, ONE, 0]);
    assert_ne!(full_out.as_ref().unwrap().program().public_affines,
        tail_out.as_ref().unwrap().program().public_affines);
    let full_before = output_snapshot(full_out.as_ref().unwrap());
    let tail_before = output_snapshot(tail_out.as_ref().unwrap());
    assert!(full_before.table_backing.iter().all(Option::is_some));
    assert!(tail_before.table_backing.iter().all(Option::is_some));
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.phase, Phase::Complete);
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Phase));
    assert_eq!(backings(&owner), before);
    assert_eq!(output_snapshot(full_out.as_ref().unwrap()), full_before);
    assert_eq!(output_snapshot(tail_out.as_ref().unwrap()), tail_before);
}

#[test]
fn parameter_build_tail_unwind_keeps_caller_resident_full_output_and_both_rows() {
    let mut full = Some(source(1));
    let mut tail = Some(distinct_tail());
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 2).unwrap();
    let before = backings(&owner);
    owner.bind().unwrap();
    assert_eq!(backings(&owner), before);
    owner.preflight(&mut WorkPreflightWorkspace::empty()).unwrap();
    assert_eq!(backings(&owner), before);
    assert_full_tail_aggregation(owner.summary().unwrap());
    let mut total = WorkBudget::default();
    owner.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
    let charged = total;
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.count, 2);
    assert_eq!(charged, owner.summary().unwrap().work);
    assert_eq!(before.1.unwrap().cosine, [0, ONE]);
    assert_eq!(before.1.unwrap().sine, [ONE, 0]);
    // Direct native reference is shared-native parity, not an independent
    // numeric/digest oracle. It does not own either original parameter Vec.
    let expected_full = FactoryOutput {
        program: Program::rotary_half(1, 4, input(), &[ONE, ONE / 2], &[0, -ONE / 2]).unwrap(),
        certificate: ProducedCertificate::None,
    };
    let mut full_out = None;
    let mut tail_out = None;
    panic_at(TestStage::Build, 2); // BEFORE tail construction, not partial tail work
    assert!(catch_unwind(AssertUnwindSafe(|| owner.build_into(&mut full_out, &mut tail_out))).is_err());
    assert!(full_out.is_some() && tail_out.is_none());
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.phase, Phase::Burned);
    assert_eq!(total, charged);
    same_output(full_out.as_ref().unwrap(), &expected_full);
    assert_rotary_tables(full_out.as_ref().unwrap(),
        [ONE, ONE / 2, ONE, ONE / 2], [0, ONE / 2, 0, -ONE / 2]);
    let full_before = output_snapshot(full_out.as_ref().unwrap());
    assert!(full_before.table_backing.iter().all(Option::is_some));
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Phase));
    assert_eq!(output_snapshot(full_out.as_ref().unwrap()), full_before);
    assert!(tail_out.is_none());
    assert_eq!(backings(&owner), before);
    assert_eq!(total, charged);
    // No claim about native Builder/temporary-result retention inside its call.
}

#[test]
fn parameter_private_native_error_oracle_is_not_valid_admitted_path_reachability() {
    let mut owner = installed(0, 1);
    let before = backings(&owner);
    // Deliberately private state oracle: invalid geometry cannot pass preflight.
    // This exercises the real native error through a BORROW, not a fake ticket,
    // public admission claim or deterministic allocator-failure simulation.
    owner.phase = Phase::Admitted;
    let mut full_out = None;
    let mut tail_out = None;
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Shape));
    assert_eq!(backings(&owner), before);
    assert!(full_out.is_none() && tail_out.is_none());
    assert_eq!(owner.phase, Phase::Burned);
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Phase));
    assert_eq!(backings(&owner), before);
    assert!(full_out.is_none() && tail_out.is_none());
}

#[test]
fn parameter_build_occupied_output_burns_once_without_replacing_output_or_rows() {
    let mut owner = priced();
    let before = backings(&owner);
    let mut total = WorkBudget::default();
    owner.admit(&mut CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap()).unwrap();
    let charged = total;
    let mut full_out = Some(FactoryOutput {
        program: Program::equal_zero(1, input()).unwrap(), certificate: ProducedCertificate::None,
    });
    let output_before = output_snapshot(full_out.as_ref().unwrap());
    let mut tail_out = None;
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Binding));
    assert_eq!(output_snapshot(full_out.as_ref().unwrap()), output_before);
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.phase, Phase::Burned);
    assert!(tail_out.is_none());
    assert_eq!(total, charged);
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Phase));
    assert_eq!(output_snapshot(full_out.as_ref().unwrap()), output_before);
    assert_eq!(backings(&owner), before);
    assert!(tail_out.is_none());
    assert_eq!(total, charged);
}

#[test]
fn parameter_rms_original_gamma_stays_borrowed_not_owned_or_self_referential() {
    let gamma = [ONE, ONE / 2];
    let mut source = Some(Construction::RmsNormCertified { rows: 1, width: 2, gamma_len: 2 }
        .resident_parameters(FactoryParameters::RmsNormCertified { input: input(), gamma: &gamma }));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut source, None, 1).unwrap();
    owner.bind().unwrap();
    match owner.full.as_ref().unwrap().lens().parameters {
        FactoryParameters::RmsNormCertified { gamma: original, .. } => {
            assert_eq!(original.as_ptr(), gamma.as_ptr());
            assert_eq!(original, gamma);
        }
        _ => panic!("expected original borrowed gamma"),
    }
}

#[test]
fn parameter_complete_sources_can_mix_owned_rotary_and_borrowed_gamma_lifetimes() {
    let gamma = [ONE, ONE / 2];
    let mut full = Some(Construction::RotaryHalf { rows: 1, width: 2, cosine_len: 1, sine_len: 1 }
        .resident_rotary_parameters(input(), vec![ONE], vec![0]));
    let original = backing(&full.as_ref().unwrap().body);
    let mut tail = Some(Construction::RmsNormCertified { rows: 1, width: 2, gamma_len: 2 }
        .resident_parameters(FactoryParameters::RmsNormCertified { input: input(), gamma: &gamma }));
    let mut owner = WorkPreflightWorkspace::resident_schedule();
    owner.install(&mut full, Some(&mut tail), 2).unwrap();
    owner.bind().unwrap();
    assert!(full.is_none() && tail.is_none());
    assert_eq!(backing(owner.full.as_ref().unwrap()), original);
    match owner.tail.as_ref().unwrap().lens().parameters {
        FactoryParameters::RmsNormCertified { gamma: original, .. } => {
            assert_eq!(original.as_ptr(), gamma.as_ptr());
        }
        _ => panic!("expected borrowed tail gamma"),
    }
}

#[test]
fn parameter_wrong_stage_has_no_numeric_permission_adoption_or_retry() {
    let mut owner = installed(1, 2);
    let before = backings(&owner);
    let mut total = WorkBudget::default();
    let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
    assert_eq!(owner.admit(&mut ledger), Err(phase_error()));
    assert_eq!(backings(&owner), before);
    assert_eq!(owner.preflight(&mut WorkPreflightWorkspace::empty()), Err(phase_error()));
    assert_eq!(backings(&owner), before);
    let mut full_out = None;
    let mut tail_out = None;
    assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Phase));
    assert_eq!(ledger.total(), WorkBudget::default());
    assert_eq!(owner.phase, Phase::Installed);
    assert_eq!(backings(&owner), before);
}

#[test]
fn parameter_terminal_refusal_matrix_preserves_all_resident_and_caller_state() {
    for target in [
        Phase::BindingEntered,
        Phase::PreflightEntered,
        Phase::AdmissionEntered,
        Phase::Burned,
        Phase::Complete,
    ] {
        // Valid, distinct complete sources, original ledger, workspace and all
        // output owners live OUTSIDE the catch closures. No direct phase edits.
        let mut full = Some(source(1));
        let mut tail = Some(distinct_tail());
        let mut owner = WorkPreflightWorkspace::resident_schedule();
        owner.install(&mut full, Some(&mut tail), 2).unwrap();
        let mut workspace = WorkPreflightWorkspace::empty();
        let mut total = WorkBudget::local();
        let mut ledger = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap();
        let mut full_out = None;
        let mut tail_out = None;
        let mut completed_outputs = (None, None);
        if target == Phase::BindingEntered {
            panic_at(TestStage::Bind, 1); // after entry, before validation
            assert!(catch_unwind(AssertUnwindSafe(|| owner.bind())).is_err());
        } else {
            owner.bind().unwrap();
            if target == Phase::PreflightEntered {
                panic_at(TestStage::Price, 1); // after actual native query
                assert!(catch_unwind(AssertUnwindSafe(|| owner.preflight(&mut workspace))).is_err());
            } else {
                owner.preflight(&mut workspace).unwrap();
                assert_full_tail_aggregation(owner.summary().unwrap());
                if target == Phase::AdmissionEntered {
                    panic_at(TestStage::Admit, 1); // before debit
                    assert!(catch_unwind(AssertUnwindSafe(|| owner.admit(&mut ledger))).is_err());
                } else {
                    owner.admit(&mut ledger).unwrap();
                    if target == Phase::Burned {
                        panic_at(TestStage::Build, 1); // before full construction
                        assert!(catch_unwind(AssertUnwindSafe(|| owner.build_into(&mut full_out, &mut tail_out))).is_err());
                    } else {
                        assert_eq!(target, Phase::Complete);
                        owner.build_into(&mut full_out, &mut tail_out).unwrap();
                        assert_rotary_tables(full_out.as_ref().unwrap(),
                            [ONE, ONE / 2, ONE, ONE / 2], [0, ONE / 2, 0, -ONE / 2]);
                        assert_rotary_tables(tail_out.as_ref().unwrap(),
                            [0, ONE, 0, ONE], [-ONE, 0, ONE, 0]);
                        // Preserve successful outputs while supplying otherwise
                        // valid EMPTY result slots for the refused build call.
                        completed_outputs = (full_out.take(), tail_out.take());
                    }
                }
            }
        }
        assert_eq!(owner.phase, target);
        assert_eq!(owner.count, 2);
        assert!(!workspace.is_poisoned());
        assert!(full_out.is_none() && tail_out.is_none());
        let mut supplied_full = Some(distinct_tail());
        let mut supplied_tail = Some(source(1));
        // Pure owned data snapshots do not borrow any resident field across an
        // attempted mutation. Table data copies do NOT retain original backing.
        macro_rules! snapshot {
            () => {
                (
                    (owner.full.as_ref().map(backing), owner.tail.as_ref().map(backing),
                        owner.count, owner.phase, owner.summary()),
                    (supplied_full.as_ref().map(|source| backing(&source.body)),
                        supplied_tail.as_ref().map(|source| backing(&source.body))),
                    workspace.is_poisoned(),
                    ledger.total(),
                    (full_out.as_ref().map(output_snapshot), tail_out.as_ref().map(output_snapshot)),
                    (completed_outputs.0.as_ref().map(output_snapshot),
                        completed_outputs.1.as_ref().map(output_snapshot)),
                )
            };
        }
        let before = snapshot!();
        // Check refusal and nonmutation with present full/tail, positive count,
        // unpoisoned workspace, ample ledger caps and empty result slots. These
        // assertions do not identify the rejecting guard: install also checks
        // occupied ownership, and early admission has no priced summary.
        // Guard attribution remains static-source evidence.
        assert_eq!(owner.install(&mut supplied_full, Some(&mut supplied_tail), 2), Err(phase_error()));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.bind(), Err(phase_error()));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.preflight(&mut workspace), Err(phase_error()));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.admit(&mut ledger), Err(phase_error()));
        assert_eq!(snapshot!(), before);
        assert_eq!(owner.build_into(&mut full_out, &mut tail_out), Err(Error::Phase));
        assert_eq!(snapshot!(), before);
    }
    // No finish API exists. No tail-partial-work, drop-order, abort/OOM or
    // double-panic claim: these hooks and snapshots cover parameters/refusals.
}
