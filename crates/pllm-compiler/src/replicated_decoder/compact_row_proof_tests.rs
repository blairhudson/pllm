use super::*;
use crate::replicated_decoder::runtime::body::{self, handoff, layer};
use crate::replicated_decoder::{Interval, WeightMode};

fn attempts() -> usize {
    OWNER_RETAIN_ATTEMPTS.with(std::cell::Cell::get)
}

// Pure resource-census fixtures, never executable owners or custody tickets.
fn export_phases() -> super::super::Phases {
    super::super::Phases {
        baseline: 101,
        construction: 211,
        copies: 223,
        execution: 227,
        restoration: 229,
    }
}

#[test]
fn original_export_quote_names_new_wrapper_result_move_and_restore_controls() {
    let common = size_of::<OriginalExportQuote>()
        + size_of::<OriginalExportQuote>()
        + size_of::<Result<OriginalExportQuote, Error>>()
        + size_of::<Result<(), Error>>()
        + size_of::<[usize; 3]>();
    let finish = common
        + size_of::<CoordinatorDrained<Invocation>>()
        + size_of::<Invocation>()
        + size_of::<PartyRuntime>()
        + size_of::<Values>()
        + size_of::<Result<(PartyRuntime, Values), Error>>()
        + size_of::<OriginalBridgeExport>()
        + size_of::<OriginalBridgeExport>()
        + size_of::<Result<OriginalBridgeExport, Error>>()
        + size_of::<Result<usize, Error>>();
    let retained = common + size_of::<OriginalBridgeExport>();
    let restoration = common
        + size_of::<OriginalBridgeExport>()
        + size_of::<OriginalBridgeExport>()
        + size_of::<Worker>()
        + size_of::<Result<Worker, Error>>()
        + size_of::<super::super::Binding>()
        + size_of::<[usize; 1]>()
        + size_of::<&OriginalRestoreOutputs>()
        + size_of::<Secret>()
        + size_of::<Result<Secret, Error>>()
        + size_of::<Result<(u64, u64), Error>>()
        + OriginalExportQuote::owned_restoration_controls().unwrap();
    assert_eq!(OriginalExportQuote::quote_controls().unwrap(), common);
    assert_eq!(OriginalExportQuote::finish_controls().unwrap(), finish);
    assert_eq!(OriginalExportQuote::retained_controls().unwrap(), retained);
    assert_eq!(
        OriginalExportQuote::restoration_controls().unwrap(),
        restoration
    );
    let quote = OriginalExportQuote::new(&export_phases(), 137).unwrap();
    let original = 137 + 101 + 229;
    assert_eq!(quote.original_bytes, original);
    assert_eq!(quote.finish_bytes, original + finish);
    assert_eq!(quote.retained_bytes, original + retained);
    assert_eq!(quote.restoration_bytes, original + restoration);
    assert_eq!(
        quote.admitted_bytes,
        original + finish.max(retained).max(restoration)
    );
    // No old Invocation slot or native fixed reserve offsets the new wrappers.
    assert!(quote.retained_bytes > original + size_of::<OriginalBridgeExport>());
}

#[test]
fn original_export_preflight_is_exact_one_short_and_keeps_existing_cap() {
    let quote = OriginalExportQuote::new(&export_phases(), 137).unwrap();
    assert_eq!(quote.preflight(quote.admitted_bytes), Ok(()));
    assert_eq!(
        quote.preflight(quote.admitted_bytes - 1),
        Err(Error::Bounds)
    );
    assert_eq!(quote.preflight(MAX_PROGRAM_BYTES), Ok(()));
    // The old phase ceiling can fit EXACTLY while the new transition cannot.
    assert_eq!(quote.preflight(quote.original_bytes), Err(Error::Bounds));
    let phases = super::super::Phases {
        baseline: MAX_PROGRAM_BYTES,
        construction: 0,
        copies: 0,
        execution: 0,
        restoration: 0,
    };
    assert_eq!(phases.admit(), Ok(()));
    let enlarged = OriginalExportQuote::new(&phases, 0).unwrap();
    assert!(enlarged.admitted_bytes > MAX_PROGRAM_BYTES);
    assert_eq!(enlarged.preflight(usize::MAX), Err(Error::Bounds));
}

#[test]
fn original_export_adds_baseline_to_each_phase_and_uses_phase_maximum() {
    for winner in 0..4 {
        let mut sizes = [11, 13, 17, 19];
        sizes[winner] = 503;
        let phases = super::super::Phases {
            baseline: 101,
            construction: sizes[0],
            copies: sizes[1],
            execution: sizes[2],
            restoration: sizes[3],
        };
        let zero = OriginalExportQuote::new(&phases, 0).unwrap();
        let added = OriginalExportQuote::new(&phases, 137).unwrap();
        assert_eq!(zero.original_bytes, 101 + 503);
        assert_eq!(added.original_bytes, zero.original_bytes + 137);
        assert_eq!(added.finish_bytes, zero.finish_bytes + 137);
        assert_eq!(added.retained_bytes, zero.retained_bytes + 137);
        assert_eq!(added.restoration_bytes, zero.restoration_bytes + 137);
        assert_eq!(added.admitted_bytes, zero.admitted_bytes + 137);
        assert_eq!(
            added.admitted_bytes,
            added
                .finish_bytes
                .max(added.retained_bytes)
                .max(added.restoration_bytes)
        );
    }
}

#[test]
fn executable_copy_reuses_complete_export_quote_without_losing_increment() {
    let quote = OriginalExportQuote::new(&export_phases(), 137).unwrap();
    let copy_controls = size_of::<ExecutableRowCopy<'_>>()
        + size_of::<ExecutableRowCopy<'_>>()
        + size_of::<Result<ExecutableRowCopy<'_>, Error>>()
        + size_of::<OriginalBridgeExport>();
    let zero = ExecutableRowCopy::retained_baseline(&quote, 0).unwrap();
    assert_eq!(zero, quote.admitted_bytes + copy_controls);
    assert_eq!(
        zero - (quote.original_bytes + copy_controls),
        quote.admitted_bytes - quote.original_bytes
    );
    assert_eq!(
        ExecutableRowCopy::retained_baseline(&quote, 239).unwrap(),
        zero + 239
    );
    let extra_export = OriginalExportQuote::new(&export_phases(), 137 + 307).unwrap();
    assert_eq!(
        ExecutableRowCopy::retained_baseline(&extra_export, 0).unwrap(),
        zero + 307
    );
}

#[test]
fn original_export_and_copy_quotes_reject_checked_addition_overflow() {
    assert!(matches!(
        OriginalExportQuote::new(&export_phases(), usize::MAX),
        Err(Error::Bounds)
    ));
    for overflow_baseline in [false, true] {
        let phases = super::super::Phases {
            baseline: if overflow_baseline { usize::MAX } else { 0 },
            construction: if overflow_baseline { 1 } else { usize::MAX },
            copies: 0,
            execution: 0,
            restoration: 0,
        };
        assert!(matches!(
            OriginalExportQuote::new(&phases, 0),
            Err(Error::Bounds)
        ));
    }
    let quote = OriginalExportQuote::new(&export_phases(), 0).unwrap();
    assert_eq!(
        ExecutableRowCopy::retained_baseline(&quote, usize::MAX),
        Err(Error::Bounds)
    );
}

#[test]
fn owned_restoration_quote_adds_guard_and_transition_slots_before_allocation() {
    let delta = size_of::<OriginalRestoreOutputs>()
        + handoff::owned_restore_inline_bytes::<OriginalRestoreOutputs>().unwrap()
        + size_of::<Result<usize, Error>>();
    assert_eq!(
        OriginalExportQuote::owned_restoration_controls().unwrap(),
        delta
    );
    // Guard controls alone include output-owner argument + destruction moves,
    // independently of the caller's construction slot. No old-slot refunds.
    assert!(delta > 3 * size_of::<OriginalRestoreOutputs>());
    let quote = OriginalExportQuote::new(&export_phases(), 137).unwrap();
    let old_restoration = OriginalExportQuote::restoration_controls().unwrap() - delta;
    let old_ceiling = quote.original_bytes
        + OriginalExportQuote::finish_controls()
            .unwrap()
            .max(OriginalExportQuote::retained_controls().unwrap())
            .max(old_restoration);
    assert_eq!(
        quote.restoration_bytes,
        quote.original_bytes + old_restoration + delta
    );
    assert_eq!(
        quote.admitted_bytes,
        old_ceiling.max(quote.restoration_bytes)
    );
    assert_eq!(quote.preflight(quote.admitted_bytes), Ok(()));
    assert_eq!(
        quote.preflight(quote.admitted_bytes - 1),
        Err(Error::Bounds)
    );
    assert_eq!(
        ExecutableRowCopy::retained_baseline(&quote, 239).unwrap(),
        quote.admitted_bytes
            + 239
            + size_of::<ExecutableRowCopy<'_>>()
            + size_of::<ExecutableRowCopy<'_>>()
            + size_of::<Result<ExecutableRowCopy<'_>, Error>>()
            + size_of::<OriginalBridgeExport>()
    );
}

// Drop probes exercise the SAME generic guard and ?/unwind paths used by
// restore_owned. They are not native identity, runtime, or proof fixtures. Only
// the trace handle is shared; execution-owner probes themselves are not Clone.
struct RestoreDropProbe {
    trace: std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>,
    name: &'static str,
    panic_on_drop: bool,
}

impl Drop for RestoreDropProbe {
    fn drop(&mut self) {
        self.trace.borrow_mut().push(self.name);
        if self.panic_on_drop {
            panic!("injected output-erasure unwind");
        }
    }
}

struct RestoreOutputProbes {
    _values: RestoreDropProbe,
    _compiled_source: RestoreDropProbe,
}

fn restore_drop_probes(
    trace: &std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>,
    panic_on_output_drop: bool,
) -> (RestoreOutputProbes, RestoreDropProbe, RestoreDropProbe) {
    let probe = |name, panic_on_drop| RestoreDropProbe {
        trace: std::rc::Rc::clone(trace),
        name,
        panic_on_drop,
    };
    (
        RestoreOutputProbes {
            _values: probe("values", panic_on_output_drop),
            _compiled_source: probe("compiled-source", false),
        },
        probe("runtime", false),
        probe("worker-source", false),
    )
}

#[test]
fn restoration_allocation_or_export_error_erases_outputs_before_execution_owners() {
    for allocation_error in [true, false] {
        let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let (outputs, runtime, suspended) = restore_drop_probes(&trace, false);
        let result = handoff::test_restore_owner_order(outputs, runtime, suspended, |_, _| {
            if allocation_error {
                Err(Error::Allocation)
            } else {
                Err(Error::Binding)
            }
        });
        if allocation_error {
            assert!(matches!(result, Err(Error::Allocation)));
        } else {
            assert!(matches!(result, Err(Error::Binding)));
        }
        assert_eq!(
            trace.borrow().as_slice(),
            &["values", "compiled-source", "runtime", "worker-source"]
        );
    }
}

#[test]
fn restoration_copy_or_erasure_unwind_keeps_execution_owners_until_outputs_drop() {
    for panic_on_output_drop in [false, true] {
        let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let (outputs, runtime, suspended) = restore_drop_probes(&trace, panic_on_output_drop);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handoff::test_restore_owner_order(outputs, runtime, suspended, |_, _| {
                if !panic_on_output_drop {
                    panic!("injected export-copy unwind");
                }
                Ok(())
            })
        }));
        assert!(result.is_err());
        assert_eq!(
            trace.borrow().as_slice(),
            &["values", "compiled-source", "runtime", "worker-source"]
        );
    }
}

#[test]
fn successful_restoration_erases_outputs_before_releasing_original_owners() {
    let trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let (outputs, runtime, suspended) = restore_drop_probes(&trace, false);
    let originals =
        handoff::test_restore_owner_order(outputs, runtime, suspended, |_, _| Ok(())).unwrap();
    assert_eq!(trace.borrow().as_slice(), &["values", "compiled-source"]);
    drop(originals);
    assert_eq!(
        trace.borrow().as_slice(),
        &["values", "compiled-source", "runtime", "worker-source"]
    );
}

// Actual native residual proof, not a test-only numeric insertion. No native
// tensor constructor, runtime, source import or checkpoint is needed.
fn producer() -> (Arc<Plan>, Layout, Layout) {
    let mut builder = layer::Builder::new(
        WeightMode::Public,
        &[
            body::Register {
                lanes: 6,
                bounds: Interval::new(-12, 12).unwrap(),
            },
            body::Register {
                lanes: 6,
                bounds: Interval::new(0, 0).unwrap(),
            },
        ],
    )
    .unwrap();
    let input = builder.input(0).unwrap();
    let zero = builder.input(1).unwrap();
    let output = builder.residual(&input, &zero, 3).unwrap();
    let other = builder.residual(&output, &zero, 3).unwrap();
    let plan = builder.finish(&[output.clone(), other]).unwrap();
    (
        plan,
        output.slice(0, 3).unwrap(),
        output.slice(3, 3).unwrap(),
    )
}

fn consumer() -> (Arc<Plan>, Layout) {
    let mut builder = layer::Builder::new(
        WeightMode::Public,
        &[
            body::Register {
                lanes: 6,
                bounds: Interval::new(-12, 12).unwrap(),
            },
            body::Register {
                lanes: 6,
                bounds: Interval::new(-12, 12).unwrap(),
            },
        ],
    )
    .unwrap();
    let input = builder.input(0).unwrap();
    let other = builder.input(1).unwrap();
    let output = builder.binary(&input, &other, false).unwrap();
    (
        builder.finish(&[output]).unwrap(),
        input.slice(0, 3).unwrap(),
    )
}

fn exported(plan: &Arc<Plan>, row: &Layout) -> ExportedRow {
    ExportedRow::from_export(plan, 0, 0, row, 0, MAX_PROGRAM_BYTES).unwrap()
}

#[test]
fn exact_export_keeps_original_arc_and_native_proof_without_numeric_changes() {
    let (plan, row, _) = producer();
    let digest = plan.digest();
    let work = plan.work_budget();
    let advances = plan.coordinator_advances();
    let resources = plan.resources().admitted_bytes;
    let refs = Arc::strong_count(&plan);
    let original = plan.complete_row(&row).unwrap();
    let proof = exported(&plan, &row);
    assert!(Arc::ptr_eq(&proof.producer, &plan));
    assert_eq!(Arc::strong_count(&plan), refs + 1);
    assert_eq!(proof.producer_proof().producer(), digest);
    assert_eq!(proof.producer_proof().key(), original.key());
    assert_eq!(proof.producer_proof().energy(), original.energy());
    assert_eq!(proof.producer_proof().mass(), original.mass());
    // Separately allocated ordered aliases are valid. Descriptor identity is
    // not required; original producer Arc identity is.
    let alias = Layout::concat(&[row.slice(0, 1).unwrap(), row.slice(1, 2).unwrap()]).unwrap();
    proof.check_export(&plan, 0, 0, &alias).unwrap();
    assert_eq!(plan.digest(), digest);
    assert_eq!(plan.work_budget(), work);
    assert_eq!(plan.coordinator_advances(), advances);
    assert_eq!(plan.resources().admitted_bytes, resources);
}

#[test]
fn wrong_original_owner_with_same_digest_is_not_a_replacement() {
    let (original, row, _) = producer();
    let (replacement, same_row, _) = producer();
    assert_eq!(original.digest(), replacement.digest());
    assert!(!Arc::ptr_eq(&original, &replacement));
    let proof = exported(&original, &row);
    assert_eq!(
        proof.check_export(&replacement, 0, 0, &same_row),
        Err(Error::Binding)
    );
    let retained = Arc::clone(&original);
    proof.check_export(&retained, 0, 0, &row).unwrap();
}

#[test]
fn reordered_partial_duplicate_strided_and_mixed_rows_never_qualify() {
    let (plan, row, next) = producer();
    let key = row.row_key(3).unwrap();
    let bad = [
        row.slice(0, 2).unwrap(),
        Layout::concat(&[row.slice(2, 1).unwrap(), row.slice(0, 2).unwrap()]).unwrap(),
        Layout::concat(&[row.slice(0, 1).unwrap(), row.slice(0, 2).unwrap()]).unwrap(),
        Layout::affine(key.owner, key.offset, &[3], &[2]).unwrap(),
        Layout::concat(&[row.slice(0, 2).unwrap(), next.slice(2, 1).unwrap()]).unwrap(),
        Layout::concat(&[row.clone(), next]).unwrap(),
    ];
    let before = attempts();
    let refs = Arc::strong_count(&plan);
    for candidate in bad {
        assert!(ExportedRow::from_export(&plan, 0, 0, &candidate, 0, MAX_PROGRAM_BYTES).is_err());
    }
    assert_eq!(attempts(), before);
    assert_eq!(Arc::strong_count(&plan), refs);
}

#[test]
fn internal_proof_and_correct_energy_do_not_authorize_wrong_export() {
    let (plan, row, next) = producer();
    assert!(plan.complete_row(&row).is_some());
    assert!(plan.complete_row(&next).is_some());
    // Native proof exists, but export 1 contains another owner.
    assert!(ExportedRow::from_export(&plan, 1, 0, &row, 0, MAX_PROGRAM_BYTES).is_err());
    assert!(ExportedRow::from_export(&plan, 0, 3, &row, 0, MAX_PROGRAM_BYTES).is_err());
    assert!(ExportedRow::from_export(&plan, 2, 0, &row, 0, MAX_PROGRAM_BYTES).is_err());
    assert!(ExportedRow::from_export(&plan, 0, usize::MAX, &row, 0, MAX_PROGRAM_BYTES).is_err());
    let proof = ExportedRow::from_export(&plan, 0, 3, &next, 0, MAX_PROGRAM_BYTES).unwrap();
    proof.check_export(&plan, 0, 3, &next).unwrap();
    assert!(proof.check_export(&plan, 0, 0, &row).is_err());
}

#[test]
fn consumer_endpoints_pin_original_arc_input_owner_offset_and_complete_width() {
    let (plan, row, _) = producer();
    let proof = exported(&plan, &row);
    let (consumer, input) = consumer();
    let (replacement, same_input) = self::consumer();
    assert_eq!(consumer.digest(), replacement.digest());
    let obligation =
        ConsumerObligation::new(&proof, &consumer, &input, 0, MAX_PROGRAM_BYTES).unwrap();
    obligation
        .check_endpoints(&plan, 0, 0, &row, &consumer, &input)
        .unwrap();
    assert!(
        obligation
            .check_endpoints(&plan, 0, 0, &row, &replacement, &same_input)
            .is_err()
    );
    for wrong in [
        Layout::affine(1, 0, &[3], &[1]).unwrap(),
        Layout::affine(0, 3, &[3], &[1]).unwrap(),
        input.slice(0, 2).unwrap(),
        Layout::affine(0, 0, &[3], &[0]).unwrap(),
    ] {
        assert!(
            obligation
                .check_endpoints(&plan, 0, 0, &row, &consumer, &wrong)
                .is_err()
        );
    }
    let wrong_width = input.slice(0, 2).unwrap();
    assert!(
        ConsumerObligation::new(&proof, &consumer, &wrong_width, 0, MAX_PROGRAM_BYTES).is_err()
    );
    // Structural endpoint checks deliberately cannot import producer energy.
    assert!(consumer.complete_row(&input).is_none());
}

#[test]
fn matching_endpoints_never_turn_caller_labels_into_consumer_authority() {
    let (plan, row, _) = producer();
    let proof = exported(&plan, &row);
    let (consumer, input) = consumer();
    let obligation =
        ConsumerObligation::new(&proof, &consumer, &input, 0, MAX_PROGRAM_BYTES).unwrap();
    obligation
        .check_endpoints(&plan, 0, 0, &row, &consumer, &input)
        .unwrap();
    assert!(proof.producer_proof().energy().is_some());
    assert!(consumer.complete_row(&input).is_none());
    // No destination proof, authenticated-edge constructor, general energy
    // accessor or ledger-writing API is supplied by ConsumerObligation.
}

#[test]
fn source_and_attention_numbers_are_not_general_complete_row_provenance() {
    let (plan, input) = consumer();
    // Source/block and attention energies may numerically equal a genuine row
    // bound. No such scalar is an accepted argument to ExportedRow.
    let (producer, row, _) = producer();
    let energy = producer.complete_row(&row).unwrap().energy().unwrap();
    let mut scan = crate::replicated_decoder::source_norms::BlockScan::new(1, 3, 1).unwrap();
    scan.append(0, &[energy.isqrt() as i64, 0, 0]).unwrap();
    let source = scan.finish().unwrap();
    assert_eq!(source.block(0, 1).unwrap().frobenius_squared, energy);
    let attention_only = crate::replicated_decoder::norm_bounds::linear_energy(
        1 << 36,
        source.block(0, 1).unwrap(),
        18,
    )
    .unwrap();
    assert!(attention_only >= energy);
    assert!(plan.complete_row(&input).is_none());
    assert!(ExportedRow::from_export(&plan, 0, 0, &input, 0, MAX_PROGRAM_BYTES).is_err());
    assert!(plan.complete_row(plan.output_layout(0).unwrap()).is_none());
    assert!(
        ExportedRow::from_export(
            &plan,
            0,
            0,
            plan.output_layout(0).unwrap(),
            0,
            MAX_PROGRAM_BYTES
        )
        .is_err()
    );
}

#[test]
fn caller_created_source_descriptor_is_not_general_row_authority() {
    let bounds = Interval::new(-1, 1).unwrap();
    let mut builder =
        layer::Builder::new(WeightMode::Public, &[body::Register { lanes: 3, bounds }]).unwrap();
    let input = builder.input(0).unwrap();
    // Source::new is explicitly unchecked descriptor construction, not the
    // authenticated source importer. This test never executes or imports files.
    let source = layer::Source::new(0, "weight".into(), [7; 32], 9, bounds).unwrap();
    let output = builder
        .public_matrix(&input, source, 1, 3, 3, false, false)
        .unwrap();
    let plan = builder.finish(&[output.clone()]).unwrap();
    assert!(plan.complete_row(&output).is_none());
    assert!(ExportedRow::from_export(&plan, 0, 0, &output, 0, MAX_PROGRAM_BYTES).is_err());
}

#[test]
fn destination_quote_pins_retained_producer_but_never_authenticates_consumer_import() {
    let (plan, row, _) = producer();
    let (replacement, same_row, _) = producer();
    assert_eq!(plan.digest(), replacement.digest());
    let exported = exported(&plan, &row);
    let mut destination = layer::Builder::new(
        WeightMode::Public,
        &[body::Register {
            lanes: 3,
            bounds: Interval::new(-12, 12).unwrap(),
        }],
    )
    .unwrap();
    let input = destination.input(0).unwrap();
    let quote = exported
        .destination_import_quote(&destination, &plan, 0, 0, &row, &input, 137, 19)
        .unwrap();
    assert_eq!(
        quote,
        destination
            .complete_row_import_quote(
                &plan,
                &plan,
                0,
                0,
                &row,
                &input,
                137 + size_of::<ExportedRow>(),
                19,
            )
            .unwrap()
    );
    assert!(
        exported
            .destination_import_quote(&destination, &replacement, 0, 0, &same_row, &input, 137, 19,)
            .is_err()
    );
    assert!(
        exported
            .destination_import_quote(&destination, &plan, 1, 0, &row, &input, 137, 19)
            .is_err()
    );
    assert!(matches!(
        destination
            .bind_complete_row_import(&plan, &plan, 0, 0, &row, &input, 137, 19)
            .err(),
        Some(Error::Binding)
    ));
    let output = destination.binary(&input, &input, false).unwrap();
    let consumer = destination.finish(&[output]).unwrap();
    assert!(consumer.complete_row(&input).is_none());
}

#[test]
fn owner_quotes_precede_retention_include_baseline_and_refuse_overflow() {
    let (plan, row, _) = producer();
    let baseline = 137;
    let quote = ExportedRow::quote(&plan, &row, baseline).unwrap();
    assert_eq!(
        quote,
        ExportedRow::quote(&plan, &row, 0).unwrap() + baseline
    );
    assert!(quote > baseline + plan.resources().admitted_bytes + row.backing_bytes().unwrap());
    let before = attempts();
    let refs = Arc::strong_count(&plan);
    assert!(matches!(
        ExportedRow::from_export(&plan, 0, 0, &row, baseline, quote - 1),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        ExportedRow::quote(&plan, &row, usize::MAX),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        ExportedRow::from_export(&plan, 0, 0, &row, MAX_PROGRAM_BYTES, usize::MAX),
        Err(Error::Bounds)
    ));
    assert_eq!(attempts(), before);
    assert_eq!(Arc::strong_count(&plan), refs);
    let proof = ExportedRow::from_export(&plan, 0, 0, &row, baseline, quote).unwrap();
    assert_eq!(attempts(), before + 1);
    let (consumer, input) = consumer();
    let quote = ConsumerObligation::quote(&proof, &consumer, &input, baseline).unwrap();
    assert!(
        quote > baseline + plan.resources().admitted_bytes + consumer.resources().admitted_bytes
    );
    let before = attempts();
    let refs = (Arc::strong_count(&plan), Arc::strong_count(&consumer));
    assert!(matches!(
        ConsumerObligation::new(&proof, &consumer, &input, baseline, quote - 1),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        ConsumerObligation::quote(&proof, &consumer, &input, usize::MAX),
        Err(Error::Bounds)
    ));
    assert_eq!(attempts(), before);
    assert_eq!(
        (Arc::strong_count(&plan), Arc::strong_count(&consumer)),
        refs
    );
    let _obligation = ConsumerObligation::new(&proof, &consumer, &input, baseline, quote).unwrap();
    assert_eq!(attempts(), before + 2);
}

fn native_prepared_producer() -> (Arc<Plan>, Layout, Arc<tensor::Program>) {
    let mut b = layer::Builder::new(
        WeightMode::Public,
        &[body::Register {
            lanes: 3,
            bounds: Interval::new(-12, 12).unwrap(),
        }],
    )
    .unwrap();
    let input = b.input(0).unwrap();
    let (output, valid) = b
        .rms(&input, 3, &[crate::replicated_decoder::ONE; 3], None, None)
        .unwrap();
    assert!(valid.is_none());
    let plan = b.finish(&[output.clone()]).unwrap();
    let program = Arc::clone(plan.complete_row_prepared_owner(&output).unwrap());
    (plan, output, program)
}

fn native_copy_destination(prepared: &tensor::Program) -> (Builder, Layout) {
    let mut b = Builder::new(
        WeightMode::Public,
        &[body::Register {
            lanes: 3,
            bounds: prepared.output_bounds(),
        }],
    )
    .unwrap();
    let input = b.input(0).unwrap();
    (b, input)
}

#[test]
fn executable_recipe_uses_immutable_order_and_both_components_once() {
    let (producer, row, prepared) = native_prepared_producer();
    let (mut destination, input) = native_copy_destination(&prepared);
    let mut checked = CheckedRowCopy::new(
        &mut destination,
        &producer,
        &prepared,
        &prepared,
        0,
        0,
        &row,
        &input,
        0,
        0,
    )
    .unwrap();
    checked.check_owner(&producer, &prepared).unwrap();
    checked.check_destination(&*checked.destination).unwrap();
    let mut output = Secret::new(3).unwrap();
    let words = [(1, 11), (2, 22), (3, 33)];
    let mut reads = Vec::new();
    checked
        .recipe
        .copy_with(&mut output, |export, lane| {
            reads.push((export, lane));
            Ok(words[lane])
        })
        .unwrap();
    assert_eq!(reads, [(0, 0), (0, 1), (0, 2)]);
    assert_eq!(output.first.as_slice(), &[1, 2, 3]);
    assert_eq!(output.second.as_slice(), &[11, 22, 33]);
    assert_eq!(
        checked
            .recipe
            .copy_with(&mut output, |_, _| panic!("reused reader")),
        Err(Error::Retired)
    );
    // This private recipe test is not an authenticated bridge fixture and has
    // no path to Builder::import_complete_row or a destination certificate.
}

#[test]
fn executable_copy_rejects_original_owner_substitution_and_foreign_builder() {
    let (producer, row, prepared) = native_prepared_producer();
    let (replacement, _, other_prepared) = native_prepared_producer();
    assert_eq!(producer.digest(), replacement.digest());
    assert_eq!(prepared.digest(), other_prepared.digest());
    let (mut destination, input) = native_copy_destination(&prepared);
    let (mut foreign, foreign_input) = native_copy_destination(&prepared);
    let checked = CheckedRowCopy::new(
        &mut destination,
        &producer,
        &prepared,
        &prepared,
        0,
        0,
        &row,
        &input,
        0,
        0,
    )
    .unwrap();
    assert_eq!(
        checked.check_owner(&replacement, &prepared),
        Err(Error::Binding)
    );
    assert_eq!(
        checked.check_owner(&producer, &other_prepared),
        Err(Error::Binding)
    );
    assert_eq!(checked.check_destination(&foreign), Err(Error::Binding));
    checked.check_owner(&producer, &prepared).unwrap();
    checked.check_destination(&*checked.destination).unwrap();
    drop(checked);
    let foreign_output = foreign
        .binary(&foreign_input, &foreign_input, false)
        .unwrap();
    let foreign_plan = foreign.finish(&[foreign_output]).unwrap();
    assert!(foreign_plan.complete_row(&foreign_input).is_none());
    assert!(
        CheckedRowCopy::new(
            &mut destination,
            &producer,
            &prepared,
            &other_prepared,
            0,
            0,
            &row,
            &input,
            0,
            0,
        )
        .is_err()
    );
}

#[test]
fn executable_copy_reorder_partial_and_wrong_export_reject_before_retention() {
    let (producer, row, prepared) = native_prepared_producer();
    let (mut destination, input) = native_copy_destination(&prepared);
    let reordered = Layout::concat(&[row.slice(2, 1).unwrap(), row.slice(0, 2).unwrap()]).unwrap();
    let before = attempts();
    for candidate in [row.slice(0, 2).unwrap(), reordered] {
        assert!(
            CheckedRowCopy::new(
                &mut destination,
                &producer,
                &prepared,
                &prepared,
                0,
                0,
                &candidate,
                &input,
                0,
                0,
            )
            .is_err()
        );
    }
    for (export, start) in [(1, 0), (0, 1)] {
        assert!(
            CheckedRowCopy::new(
                &mut destination,
                &producer,
                &prepared,
                &prepared,
                export,
                start,
                &row,
                &input,
                0,
                0,
            )
            .is_err()
        );
    }
    assert_eq!(attempts(), before);
}

#[test]
fn executable_copy_shape_and_partial_failure_burn_recipe_without_retry() {
    let (producer, row, prepared) = native_prepared_producer();
    for wrong_shape in [false, true] {
        let (mut destination, input) = native_copy_destination(&prepared);
        let mut checked = CheckedRowCopy::new(
            &mut destination,
            &producer,
            &prepared,
            &prepared,
            0,
            0,
            &row,
            &input,
            0,
            0,
        )
        .unwrap();
        let mut output = Secret::new(if wrong_shape { 2 } else { 3 }).unwrap();
        let mut reads = 0;
        let error = checked.recipe.copy_with(&mut output, |_, lane| {
            reads += 1;
            if lane == 1 {
                Err(Error::Binding)
            } else {
                Ok((7, 8))
            }
        });
        assert_eq!(
            error,
            Err(if wrong_shape {
                Error::Shape
            } else {
                Error::Binding
            })
        );
        assert_eq!(reads, if wrong_shape { 0 } else { 2 });
        assert_eq!(
            checked
                .recipe
                .copy_with(&mut output, |_, _| panic!("retry reader")),
            Err(Error::Retired)
        );
    }
}
