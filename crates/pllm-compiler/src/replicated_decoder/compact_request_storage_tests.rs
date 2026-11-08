//! Real source-create/import/declaration funnel. The expanded compiler below is
//! only an independent post-issuance numeric/recipe oracle, never an entry seal.
use super::super::source_entry::SourceRequestImport;
use super::*;
use crate::replicated_decoder::runtime::BoundaryProgress;

#[test]
fn original_controller_lexical_loan_identity_survives_local_unwind() {
    use compact_storage::{ControllerAccounting, ControllerLoan};
    let (entry, _) = importer(3);
    let result = entry.consume_compact(|created, mut controller, _, _, _| {
        let plan = Arc::clone(&created.permit.history.origin.payload().source);
        let identity = std::ptr::from_ref(&controller) as usize;
        let payload = controller.payload_bytes();
        let charged = controller.charged_bytes();
        let used = plan.budget().used_bytes();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let loan = ControllerLoan::new(&mut controller);
            assert_eq!(loan.controller_identity(), identity);
            assert!(loan.is_for(plan.budget()));
            assert_eq!(loan.payload_bytes(), payload);
            assert_eq!(loan.charged_bytes(), charged);
            panic!("loan-only ordinary unwind");
        }));
        assert!(unwind.is_err());
        assert_eq!(std::ptr::from_ref(&controller) as usize, identity);
        assert_eq!(controller.payload_bytes(), payload);
        assert_eq!(controller.charged_bytes(), charged);
        assert_eq!(plan.budget().used_bytes(), used);
        let scope = compact_storage::EmissionScope::new(created, controller, MAX_PROGRAM_BYTES)?;
        scope.into_owned(|_, _, _| Ok(()))
    });
    assert!(result.is_ok());
}
#[test]
fn original_context_phase_overflow_preserves_work_charge_and_burns() {
    let (entry, _) = importer(3);
    let result = entry.consume_compact(|created, controller, _, _, _| {
        let mut scope = compact_storage::EmissionScope::new(created, controller, MAX_PROGRAM_BYTES)?;
        let (_, context, _) = scope.parts();
        let work = context.work();
        let payload = context.controller_payload_bytes();
        let identity = context.controller_identity();
        let result = context.admit_phase(usize::MAX, 0);
        assert!(matches!(&result, Err(Error::Bounds)));
        assert!(matches!(context.check_open(), Err(Error::Binding)));
        assert_eq!(context.work(), work);
        assert_eq!(context.controller_payload_bytes(), payload);
        assert_eq!(context.controller_identity(), identity);
        result
    });
    assert!(matches!(result, Err(Error::Bounds)));
}

fn importer(vocabulary: usize) -> (SourceRequestImport, Manifest) {
    importer_with_unsafe_row(vocabulary, false)
}

type WholeHistory = super::super::source_entry::compact_pending::SourceCustody<
    super::super::compact_storage::owning::OwnedEmission<
        super::super::compact_storage::compact_history::CompletionLoan,
    >,
>;
fn same_owned_runtime(original: &Worker, candidate: &Worker, program: &Program) {
    match (
        original.test_boundary_state(),
        candidate.test_boundary_state(),
    ) {
        (Ok((step, cursors, old)), Ok((at, material, new))) => {
            assert_eq!((step, cursors), (at, material));
            let old = old.collect::<Vec<_>>();
            let new = new.collect::<Vec<_>>();
            let mut live = vec![false; new.len()];
            for (logical, value) in old.iter().enumerate() {
                if let Some(value) = value {
                    let slot = program.value_slot(logical);
                    assert!(
                        !live[slot],
                        "two live original owners cannot alias one physical slot"
                    );
                    live[slot] = true;
                    assert_eq!(Some(*value), new[slot], "logical owner {logical}");
                }
            }
            for (slot, value) in new.iter().enumerate() {
                assert_eq!(
                    value.is_some(),
                    live[slot],
                    "unmatched live physical slot {slot}"
                );
            }
        }
        (Err(Error::Phase), Err(Error::Phase)) => (),
        other => panic!(
            "mismatched original/compact boundary: {} {}",
            other.0.is_ok(),
            other.1.is_ok()
        ),
    }
    // Physical-slot occupancy during an in-flight kernel is representation
    // specific. Every idle live slot/component is checked above; loop position
    // and material cursors retain their original logical meaning.
    assert_eq!(original.loop_test_state().0, candidate.loop_test_state().0);
}
fn reconstruct_owned(workers: &[Worker; 3], program: &Program, coordinate: (usize, usize)) -> u64 {
    workers.iter().fold(0u64, |n, w| {
        let (_, _, mut values) = w.test_boundary_state().unwrap();
        let (first, _) = values
            .nth(program.value_slot(coordinate.0))
            .unwrap()
            .unwrap();
        n.wrapping_add(first[coordinate.1])
    })
}
fn paired_owned_stage(
    a: &mut [Worker; 3],
    b: &mut [Worker; 3],
    end: Option<usize>,
    program: &Program,
) -> usize {
    let (mut qa, mut qb) = (Vec::new(), Vec::new());
    let mut bytes = 0;
    let mut reached = [false; 3];
    for _ in 0..100_000 {
        if qa.is_empty() {
            if reached.iter().all(|v| *v) {
                assert!(qb.is_empty());
                return bytes;
            }
            for role in 0..3 {
                if !reached[role] {
                    let progress = |w: &mut Worker| match end {
                        Some(end) => w.advance_before(end),
                        None if w.complete() => Ok(BoundaryProgress::Reached),
                        None => w.advance().map(BoundaryProgress::Frames),
                    };
                    match (
                        progress(&mut a[role]).unwrap(),
                        progress(&mut b[role]).unwrap(),
                    ) {
                        (BoundaryProgress::Reached, BoundaryProgress::Reached) => {
                            reached[role] = true
                        }
                        (BoundaryProgress::Frames(x), BoundaryProgress::Frames(y)) => {
                            bytes += super::frames(a[role].party(), x, y, &mut qa, &mut qb)
                        }
                        _ => panic!("original/compact stage differs"),
                    }
                }
            }
        } else {
            assert_eq!(qa.len(), qb.len());
            for ((from, to, x), (other, other_to, y)) in std::mem::take(&mut qa)
                .into_iter()
                .zip(std::mem::take(&mut qb))
            {
                assert_eq!((from, to), (other, other_to));
                let x = a[to as usize].receive(x.as_bytes(), from).unwrap();
                let y = b[to as usize].receive(y.as_bytes(), from).unwrap();
                bytes += super::frames(to, x, y, &mut qa, &mut qb);
            }
        }
        for role in 0..3 {
            same_owned_runtime(&a[role], &b[role], program);
        }
    }
    panic!("original/compact router failed to finish");
}

#[test]
fn compact_sealed_same_worker_two_epochs_full_frames_outputs_and_early_late_sticky() {
    let (entry, manifest) = importer_geometry_with_original_row(3, true, 2, true);
    let program = entry.compile_compact().unwrap();
    assert!(program.nodes.is_empty(), "no expanded executable candidate");
    program
        .compact_storage()
        .unwrap()
        .check_completed()
        .unwrap();
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Program>();
    let public = Arc::clone(program.request_custody.as_ref().unwrap().public());
    // Only after the real compact Program exists is the independent control built.
    let control = Arc::new(
        Program::compile_with_norms(
            &manifest.config,
            4,
            2,
            &manifest.tensors,
            WeightMode::Public,
            manifest.execution,
            &manifest.norms,
        )
        .unwrap(),
    );
    assert_eq!(program.digest(), control.digest());
    assert_eq!(program.work_budget(), control.work_budget());
    assert!(program.value_slot_count() < control.value_slot_count());
    let (end, sticky) = program
        .compact_storage()
        .unwrap()
        .test_epoch_boundary(0)
        .unwrap();
    let sticky = (sticky.unwrap(), 0);
    for (case, tokens, expected, before) in [
        (0, TOKENS, [1, 0, 1], 1),
        (1, [0, 2, 1, 2], [0, 0, 0], 0),
        (2, [1, 2, 1, 1], [0, 0, 0], 1),
    ] {
        let mut a = super::fresh_input(
            Arc::clone(&control),
            &public,
            false,
            tokens,
            [93 + case; 32],
        );
        let mut b =
            super::fresh_input(Arc::clone(&program), &public, true, tokens, [93 + case; 32]);
        let bytes = paired_owned_stage(&mut a, &mut b, Some(end), &program);
        assert_eq!(
            reconstruct_owned(&a, &control, sticky),
            before,
            "first epoch sticky, case {case}"
        );
        assert_eq!(reconstruct_owned(&b, &program, sticky), before);
        let bytes = bytes + paired_owned_stage(&mut a, &mut b, None, &program);
        assert!(bytes > 0);
        for (coordinate, value) in program.outputs.iter().zip(expected) {
            assert_eq!(
                reconstruct_owned(&a, &control, *coordinate),
                value,
                "control output, case {case}"
            );
            assert_eq!(
                reconstruct_owned(&b, &program, *coordinate),
                value,
                "compact output, case {case}"
            );
        }
        for (a, b) in a.into_iter().zip(b) {
            same_owned_runtime(&a, &b, &program);
            assert_eq!(
                a.finish_frame().unwrap().as_slice(),
                b.finish_frame().unwrap().as_slice()
            );
        }
    }
}

#[test]
fn compact_sealed_program_publication_exact_and_one_short_burns_with_original_native_owners() {
    use super::super::source_entry::compact_pending::sealed::observe_publication_once;
    use schedule_finalization::BorrowedRecipe;
    for short in [false, true] {
        let (mut custody, _, origin) = genuine_history_fixture();
        let outside = std::rc::Rc::new(std::cell::RefCell::new(
            custody.outside_lease_for_tests().unwrap(),
        ));
        let sealed = custody.seal_compact().unwrap();
        let work = sealed.work();
        assert!(work.operations() > 0);
        let native = (0..sealed.logical_len())
            .find_map(
                |ordinal| match sealed.read_recipe(ordinal).unwrap().recipe {
                    BorrowedRecipe::Tensor { program, .. }
                    | BorrowedRecipe::CheckedRange { program, .. } => Some(Arc::downgrade(program)),
                    _ => None,
                },
            )
            .expect("actual original native invocation");
        let pressure = std::rc::Rc::clone(&outside);
        observe_publication_once(move |required| {
            // Real same-counter named outside lease. Only one byte separates
            // these actual Program-header/outputs publication admission cases.
            let bytes = MAX_PROGRAM_BYTES.checked_sub(required).unwrap() + usize::from(short);
            pressure.borrow_mut().resize_payload(bytes).unwrap();
        });
        let result = sealed.into_program();
        if short {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
        } else {
            let program = result.unwrap();
            assert_eq!(program.work_budget(), work);
            assert!(native.upgrade().is_some());
            assert_eq!(origin.payload().state.load(Ordering::Acquire), COMPLETE);
            drop(program);
        }
        assert!(
            native.upgrade().is_none(),
            "original native payload dies before outside lease refund"
        );
        assert!(outside.borrow().budget().used_bytes() >= outside.borrow().payload_bytes());
    }
}
pub(in crate::replicated_decoder::program) fn genuine_legacy_root_fixture() -> super::super::source_entry::compact_pending::sealed::LegacyRoot {
    genuine_history_fixture().0
}

fn genuine_history_fixture() -> (WholeHistory, Manifest, Arc<OwnedCharged<Origin>>) {
    use super::super::compact_storage::compact_history::PendingCompactHistory;
    let (entry, manifest) = importer_geometry_with_original_row(3, true, 2, true);
    let mut origin = None;
    let custody = entry
        .consume_compact(|created, guard, _, _, source| {
            let mut scope = super::super::compact_storage::EmissionScope::new(
                created,
                guard,
                MAX_PROGRAM_BYTES,
            )?;
            let first = {
                let (imports, context, _) = scope.parts();
                origin = Some(context.test_origin());
                let mut graph =
                    super::super::super::compact_lowering::CompactGraph::from_source_request(
                        imports,
                        context,
                        MAX_PROGRAM_BYTES,
                    )?;
                for _ in &context.plan().plan().prefill.operations {
                    graph.lower_next()?;
                }
                graph.freeze_owned(Vec::new(), source)?
            };
            let history = PendingCompactHistory::continue_original(scope, first, limits())
                .inspect_err(|e| eprintln!("continue: {e:?}"))?;
            let second = {
                let (imports, context, _) = history.checked_graph_parts()?;
                assert_eq!(context.initial_input().1, 5);
                let input = context
                    .history_input()
                    .expect("consuming original selected-coordinate map");
                let original = context.initial_input().0;
                assert_eq!(
                    imports.nodes[original].lanes, 4,
                    "no redeclared five-token input"
                );
                for lane in 0..4 {
                    assert_eq!(input.word(lane)?, (original, lane));
                }
                let selected = input.word(4)?;
                assert!(
                    selected.0 >= imports.nodes.len(),
                    "actual first-epoch producer, not a new Input"
                );
                assert!(
                    selected.0 < input.first(),
                    "producer remains in preceding epoch"
                );
                assert_eq!(
                    context.token_words_for_tests().iter().collect::<Vec<_>>(),
                    (0..5).map(|i| input.word(i).unwrap()).collect::<Vec<_>>()
                );
                assert!(
                    input.word(5).is_err(),
                    "final token never embedded a third time"
                );
                assert_eq!(context.plan().workload().max_input_tokens, 5);
                assert_eq!(context.plan().workload().max_new_tokens, 1);
                let original_nodes = imports.nodes.len();
                let mut graph =
                    super::super::super::compact_lowering::CompactGraph::from_pending_history(
                        &history,
                    )
                    .inspect_err(|e| eprintln!("epoch1 begin: {e:?}"))?;
                for (index, op) in context.plan().plan().prefill.operations.iter().enumerate() {
                    graph
                        .lower_next()
                        .inspect_err(|e| eprintln!("epoch1 op {index} {:?}: {e:?}", op.operator))?;
                }
                assert_eq!(
                    imports.nodes.len(),
                    original_nodes,
                    "imports-only Builder remains unchanged"
                );
                graph
                    .freeze_owned(Vec::new(), source)
                    .inspect_err(|e| eprintln!("epoch1 freeze: {e:?}"))?
            };
            history
                .finish(second)
                .inspect_err(|e| eprintln!("finish: {e:?}"))
        })
        .unwrap();
    (custody, manifest, origin.unwrap())
}

#[test]
fn compact_genuine_history_continues_four_to_five_before_final_acceptance() {
    let (custody, manifest, origin) = genuine_history_fixture();
    custody.produced().check_open().unwrap();
    assert_eq!(custody.produced().produced.epochs().len(), 2);
    let reader=crate::replicated_decoder::program::compact_lowering::frozen_region::owning::recipe::HistorySchedule::new(&custody).unwrap();
    let outputs = custody.produced().produced.outputs();
    let canonical =
        crate::replicated_decoder::program::schedule_finalization::quote(&reader, &outputs, 0)
            .unwrap();
    assert_eq!(
        canonical.work,
        custody.produced().work(),
        "whole canonical visitor, no second debit"
    );
    // Independent expanded control AFTER both candidate epochs and acceptance.
    let control = Program::compile_with_norms(
        &manifest.config,
        4,
        2,
        &manifest.tensors,
        WeightMode::Public,
        manifest.execution,
        &manifest.norms,
    )
    .unwrap();
    assert_eq!(custody.produced().work(), control.work_budget());
    use crate::replicated_decoder::program::schedule_finalization::BorrowedOrderedNodes;
    assert_eq!(reader.node_count(), control.nodes.len());
    let actual_prefix = custody.compact_canonical_prefix().unwrap();
    let mut actual_hash = actual_prefix;
    let mut expected_hash = custody.compact_canonical_prefix().unwrap();
    for ordinal in 0..reader.node_count() {
        crate::replicated_decoder::program::schedule_finalization::hash_node(
            &mut actual_hash,
            reader.node(ordinal).unwrap(),
        )
        .unwrap();
        crate::replicated_decoder::program::schedule_finalization::hash_node(
            &mut expected_hash,
            crate::replicated_decoder::program::schedule_finalization::BorrowedNode::from(
                &control.nodes[ordinal],
            ),
        )
        .unwrap();
    }
    assert_eq!(
        actual_hash.finalize(),
        expected_hash.finalize(),
        "all original recipes and ordered coordinates"
    );
    assert_eq!(
        custody.produced().produced.outputs().as_slice(),
        control.outputs.as_slice()
    );
    fn send_sync<T: Send + Sync>() {}
    send_sync::<
        super::super::source_entry::compact_pending::SourceCustody<
            super::super::compact_storage::owning::OwnedEmission<
                super::super::compact_storage::compact_history::CompletionLoan,
            >,
        >,
    >();
    drop(reader);
    assert_eq!(origin.payload().state.load(Ordering::Acquire), OPEN);
    let logical = custody.seal_compact().unwrap();
    // Requires real compact schedule-access delegation: never expanded nodes or
    // an accepted partial publication if that private adapter is not mounted.
    let program = logical.into_program().unwrap();
    let logical = program.compact_storage().unwrap();
    send_sync::<crate::replicated_decoder::program::source_entry::compact_pending::sealed::SealedOwnedStorage>();
    assert_eq!(logical.digest(), control.digest());
    assert_eq!(logical.work(), control.work_budget());
    assert_eq!(logical.outputs(), control.outputs.as_slice());
    assert_eq!(logical.input_count(), control.inputs);
    assert_eq!(
        logical.source_binding(),
        SourceRegistry::from_manifest(&manifest).unwrap().binding()
    );
    let mut persisted = sha2::Sha256::new();
    let mut persisted_control = sha2::Sha256::new();
    assert!(
        logical.value_slot_count() < logical.logical_len(),
        "actual retirement intervals issue reused physical slots"
    );
    for ordinal in 0..logical.logical_len() {
        let node = logical.read_recipe(ordinal).unwrap();
        crate::replicated_decoder::program::schedule_finalization::hash_node(&mut persisted, node)
            .unwrap();
        crate::replicated_decoder::program::schedule_finalization::hash_node(
            &mut persisted_control,
            crate::replicated_decoder::program::schedule_finalization::BorrowedNode::from(
                &control.nodes[ordinal],
            ),
        )
        .unwrap();
        assert_eq!(
            logical.retirement(ordinal).unwrap(),
            control.retire[ordinal].as_slice()
        );
        assert!(logical.value_slot(ordinal).unwrap() < logical.value_slot_count());
    }
    assert_eq!(
        persisted.finalize(),
        persisted_control.finalize(),
        "COMPLETE-only finalized root reuses pure recipe body"
    );
    assert_eq!(
        origin.payload().state.load(Ordering::Acquire),
        COMPLETE,
        "original acceptance ordering, not execution authority"
    );
    drop(program);
    assert_eq!(
        origin.payload().state.load(Ordering::Acquire),
        COMPLETE,
        "successful publication commits the original finish Attempt"
    );
}

#[test]
fn owned_epoch_moves_native_rows_and_source_history_without_context_borrows() {
    let (entry, _) = importer(3);
    let mut native = Vec::new();
    let mut source_addresses = Vec::new();
    let mut norm_addresses = Vec::new();
    let custody = entry
        .consume_compact(|created, guard, _, _, source| {
            let scope = super::super::compact_storage::EmissionScope::new(
                created,
                guard,
                MAX_PROGRAM_BYTES,
            )?;
            scope.into_owned(|imports, context, _| {
                source_addresses = context
                    .registry()
                    .source_records()
                    .values()
                    .map(|s| s as *const _ as usize)
                    .collect();
                norm_addresses = context
                    .registry()
                    .source_norms()
                    .values()
                    .map(|n| n as *const _ as usize)
                    .collect();
                let mut graph =
                    super::super::super::compact_lowering::CompactGraph::from_source_request(
                        imports,
                        context,
                        MAX_PROGRAM_BYTES,
                    )?;
                for _ in &context.plan().plan().prefill.operations {
                    graph.lower_next()?;
                }
                native = graph.test_softmax_programs();
                graph.freeze_owned(Vec::new(), source)
            })
        })
        .unwrap();
    let custody = Box::new(custody);
    custody.produced().check_open().unwrap();
    assert_eq!(
        custody.produced().work(),
        custody.produced().produced.work()
    );
    assert!(custody.produced().produced.recipe_count() > 0);
    let mut resolved = 0;
    for index in 0..custody.produced().produced.recipe_count() {
        let sources = custody.borrow_recipe_sources(index).unwrap();
        for source in [sources.primary(), sources.bias()].into_iter().flatten() {
            assert!(
                source_addresses.contains(&(source as *const _ as usize)),
                "same original Manifest tensor allocation after move"
            );
            resolved += 1;
        }
        for norm in [sources.norms(), sources.bias_norms()]
            .into_iter()
            .flatten()
        {
            assert!(
                norm_addresses.contains(&(norm as *const _ as usize)),
                "same original norm association after move"
            );
        }
    }
    assert!(resolved > 0);
    assert!(matches!(
        custody.borrow_recipe_sources(usize::MAX),
        Err(Error::Binding)
    ));
    assert!(custody.produced().produced.softmax_programs().count() > 0);
    let loan = custody.inspect_epoch().unwrap();
    assert!(std::ptr::eq(loan.epoch(), &custody.produced().produced));
    assert!(loan
        .imports()
        .nodes
        .iter()
        .all(|node| matches!(&node.code, Code::Input(_) | Code::PublicInput { .. })));
    let mut attempt = loan.attempt().unwrap();
    let scratch = loan.charge(1024, 1).unwrap();
    assert_eq!(scratch.payload_bytes(), 1024);
    drop(scratch);
    attempt.commit();
    let programs = custody
        .produced()
        .produced
        .softmax_programs()
        .collect::<Vec<_>>();
    assert_eq!(native.len(), programs.len());
    for (old, new) in native.iter().zip(programs) {
        assert!(Arc::ptr_eq(&old.upgrade().unwrap(), new));
    }
    fn send_sync<T: Send + Sync>() {}
    send_sync::<
        super::super::source_entry::compact_pending::SourceCustody<
            super::super::compact_storage::owning::OwnedEmission<
                super::super::super::compact_lowering::frozen_region::owning::OwnedEpoch,
            >,
        >,
    >();
}

#[test]
fn owned_epoch_unexpected_row_alias_refuses_and_burns_without_clone_fallback() {
    let (entry, _) = importer(3);
    let result = entry.consume_compact(|created, guard, _, _, source| {
        let scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
        scope.into_owned(|imports, context, _| {
            let mut graph = compact_lowering::CompactGraph::from_source_request(
                imports,
                context,
                MAX_PROGRAM_BYTES,
            )?;
            for _ in &context.plan().plan().prefill.operations {
                graph.lower_next()?;
            }
            let alias = graph.test_hold_softmax_alias()?;
            let work = context.work();
            let charged = context.reservation().payload_bytes();
            let result = graph.freeze_owned(Vec::new(), source);
            assert!(matches!(result, Err(Error::Binding)));
            assert!(
                matches!(context.check_open(), Err(Error::Binding)),
                "burn before scope drop"
            );
            assert_eq!(context.work(), work);
            assert!(context.reservation().payload_bytes() >= charged);
            drop(alias);
            result
        })
    });
    assert!(matches!(result, Err(Error::Binding)));
}

#[test]
fn owned_epoch_combined_live_conversion_exact_and_one_short_before_values_allocation() {
    use compact_lowering::frozen_region::owning::conversion_probe;
    for short in [false, true] {
        let (entry, _) = importer(3);
        let result = entry.consume_compact(|created, guard, _, _, source| {
            let scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            scope.into_owned(|imports, context, _| {
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    imports,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                for _ in &context.plan().plan().prefill.operations {
                    graph.lower_next()?;
                }
                let work = context.work();
                let before = context.reservation().payload_bytes();
                let public = Arc::downgrade(context.public());
                conversion_probe::start(short);
                let owned = graph.freeze_owned(Vec::new(), source);
                let (allocations, target, entered) = conversion_probe::finish();
                assert!(entered, "actual Values conversion reached");
                assert!(target < MAX_PROGRAM_BYTES);
                assert_eq!(
                    context.work(),
                    work,
                    "conversion cannot redebit native work"
                );
                assert!(
                    context.reservation().payload_bytes() >= before,
                    "no early refund"
                );
                assert!(Arc::ptr_eq(&public.upgrade().unwrap(), context.public()));
                assert_eq!(
                    imports.nodes.len(),
                    context.initial_input().0 + 1,
                    "imports-only Builder retained"
                );
                if short {
                    assert!(matches!(owned, Err(Error::Bounds)));
                    assert_eq!(
                        allocations, 0,
                        "one-short must refuse before destination allocation"
                    );
                    assert!(
                        matches!(context.check_open(), Err(Error::Binding)),
                        "synchronous burn before scope drop"
                    );
                } else {
                    assert!(owned.is_ok());
                    assert!(allocations > 0);
                    assert_eq!(
                        context.historical_peak(),
                        MAX_PROGRAM_BYTES,
                        "exact unchanged phase cap"
                    );
                    context.check_open()?;
                }
                owned
            })
        });
        assert_eq!(result.is_err(), short);
    }
}

#[test]
fn source_handles_resolve_after_custody_move_and_reject_copied_gamma_norm_and_foreign_instance() {
    use super::super::source_entry::compact_pending::handles;
    let (first, manifest) = importer(4);
    let (second, _) = importer(4);
    let gamma = manifest
        .tensors
        .iter()
        .find(|(_, source)| source.public_q18.is_some())
        .unwrap()
        .0;
    let file = manifest
        .tensors
        .iter()
        .find(|(_, source)| source.public_q18.is_none())
        .unwrap()
        .0;
    let norm = manifest.norms.keys().find(|key| *key != gamma).unwrap();
    handles::assert_genuine_consuming_handles(first, second, gamma, file, norm);
}

#[test]
fn source_handles_reject_equal_content_foreign_maps_and_empty_map_identity() {
    use super::super::source_entry::compact_pending::handles;
    let (entry, manifest) = importer(4);
    entry
        .consume_compact(|_, _, registry, _, _| {
            handles::assert_equal_content_foreign_map(registry, &manifest);
            Ok(())
        })
        .unwrap();
    handles::assert_empty_map_predicate(manifest);
}

#[test]
fn source_handles_reject_orphan_norm_through_original_ingress() {
    use super::super::source_entry::compact_pending::handles;
    let (control, mut manifest) = importer(4);
    drop(control.consume_compact(|_, _, _, _, _| Ok(())).unwrap());
    let norm = manifest.norms.values().next().unwrap().clone();
    manifest
        .norms
        .insert("orphan-source-not-in-plan".to_owned(), norm);
    // If original validation refuses first, preserve that earlier boundary.
    let mut entry = match SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap()) {
        Err(Error::Binding) => return,
        Err(other) => panic!("unrelated orphan refusal: {other:?}"),
        Ok(entry) => entry,
    };
    let (_, _, clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    for (key, source) in &manifest.tensors {
        if source.public_q18.is_none() {
            let values = &clear[key];
            let bytes = (0..source.shape.iter().product())
                .flat_map(|i| (values[i % values.len()] as f32 / ONE as f32).to_le_bytes())
                .collect::<Vec<_>>();
            entry.append(key, 0, &bytes).unwrap();
        }
    }
    handles::assert_orphan_norm_funnel_refused(entry);
}

#[test]
fn source_handles_payload_and_failed_issuance_retire_before_original_charges() {
    use super::super::source_entry::compact_pending::{self, handles};
    let (entry, _) = importer(4);
    let root = compact_pending::test_root(&entry);
    let baseline = compact_pending::test_retained_baseline();
    handles::assert_payload_before_guard(entry, &root, baseline);
    let (entry, _) = importer(4);
    let root = compact_pending::test_root(&entry);
    let result = entry.consume_compact(|_, _, registry, _, scope| -> Result<(), Error> {
        let actual = registry.source_records().values().next().unwrap();
        let copy = actual.clone();
        assert!(matches!(scope.tensor(&copy), Err(Error::Binding)));
        Err(Error::Binding)
    });
    assert!(matches!(result, Err(Error::Binding)));
    assert_eq!(root.budget().used_bytes(), baseline);
}
fn importer_with_unsafe_row(
    vocabulary: usize,
    unsafe_row: bool,
) -> (SourceRequestImport, Manifest) {
    importer_geometry(vocabulary, unsafe_row, 2)
}
fn importer_geometry(
    vocabulary: usize,
    unsafe_row: bool,
    width: usize,
) -> (SourceRequestImport, Manifest) {
    importer_geometry_with_original_row(vocabulary, unsafe_row, width, false)
}
pub(in crate::replicated_decoder::program) fn importer_geometry_with_original_row(
    vocabulary: usize,
    unsafe_row: bool,
    width: usize,
    original_history_row: bool,
) -> (SourceRequestImport, Manifest) {
    let (config, _, mut clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    if unsafe_row {
        clear.get_mut("model.embed_tokens.weight").unwrap()[0] = 3 * ONE / 2;
    }
    if original_history_row {
        // Exact original request_origin::Fixture: UNUSED row zero needs the
        // genuine checked guard; final ID zero is output, never embedded again.
        assert_eq!(width, 2);
        clear.get_mut("model.embed_tokens.weight").unwrap()[..2].fill(3 * ONE / 2);
    }
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["vocab_size"] = serde_json::json!(vocabulary);
    config["hidden_size"] = serde_json::json!(width);
    if config.get("head_dim").is_some() {
        config["head_dim"] = serde_json::json!(width);
    }
    let config = serde_json::to_vec(&config).unwrap();
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let mut publisher = Publisher::new_with_policy(
        &config,
        4,
        2,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        execution,
    )
    .unwrap();
    let shapes = publisher
        .shapes()
        .iter()
        .map(|(k, s)| (k.clone(), s.clone()))
        .collect::<Vec<_>>();
    let mut payloads = BTreeMap::new();
    for (key, shape) in shapes {
        let values = &clear[&key];
        let bytes = (0..shape.iter().product())
            .flat_map(|i| (values[i % values.len()] as f32 / ONE as f32).to_le_bytes())
            .collect::<Vec<_>>();
        publisher.append(&key, 0, &bytes).unwrap();
        payloads.insert(key, bytes);
    }
    publisher.seal(&TOKENS).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32]; // no precompiled identity authorizes construction
    let mut importer = SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap()).unwrap();
    for (key, bytes) in payloads {
        importer.append(&key, 0, &bytes).unwrap();
    }
    (importer, manifest)
}

#[test]
fn source_lookup_to_compact_certified_rms_retains_original_guard_and_native_owners() {
    for (width, guarded) in [(2, false), (2, true), (36, true)] {
        let (importer, manifest) = importer_geometry(4, guarded, width);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let prefix = builder.nodes.len();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                graph.lower_next()?;
                let output = graph.lower_next()?;
                assert_eq!(
                    builder.nodes.len(),
                    prefix,
                    "no expanded request executable"
                );
                let (program, gamma, input_energy, output_energy, guard_tiles) =
                    graph.test_rms_owner().unwrap();
                assert_eq!(guard_tiles > 0, guarded);
                assert!(
                    input_energy.is_none(),
                    "lookup must not invent GENERAL ancestry"
                );
                assert!(context
                    .registry()
                    .source_records()
                    .values()
                    .any(|source| std::ptr::eq(source, gamma)));
                assert!(output_energy > 0);
                context.check_final_work(graph.work_budget())?;
                // Independent original control AFTER candidate owner/certificate.
                let oracle = Program::compile_with_norms(
                    &manifest.config,
                    4,
                    2,
                    &manifest.tensors,
                    WeightMode::Public,
                    manifest.execution,
                    &manifest.norms,
                )
                .unwrap();
                let mut end = 0;
                for lane in 0..4 * width {
                    let (owner, index) = output.word(lane)?;
                    let Code::Tensor {
                        program: original, ..
                    } = &oracle.nodes[owner].code
                    else {
                        panic!("original RMS consumer geometry")
                    };
                    assert_eq!(original.digest(), program.digest());
                    assert_eq!(index, lane % width);
                    end = end.max(owner + 1);
                }
                let expected = oracle.nodes[prefix + 2..end]
                    .iter()
                    .try_fold(WorkBudget::default(), |work, node| {
                        work.checked_add(node.work_budget()?)
                    })?;
                assert_eq!(expected, context.work());
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn compact_request_first_instruction_constructs_original_lookup() {
    for vocabulary in [4, 128] {
        let (importer, manifest) = importer(vocabulary);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let prefix = builder.nodes.len();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                let output = graph.lower_next()?;
                let census = graph.token_lookup_census().next().unwrap();
                assert_eq!(census.first, prefix + 2);
                assert_eq!(census.geometry, [4, vocabulary, 2]);
                assert_eq!(builder.nodes.len(), prefix, "no expanded request graph");
                let mut work = census
                    .program
                    .work_budget()
                    .checked_mul((vocabulary / census.program.lanes()) as u64)?
                    .checked_mul(4)?;
                if let Some(tail) = census.tail {
                    work = work.checked_add(tail.work_budget().checked_mul(4)?)?;
                }
                assert_eq!(context.work(), work);
                assert_eq!(graph.container_census()?.partial_work, work);
                // Independent expanded control starts ONLY AFTER candidate issuance.
                let oracle = Program::compile_with_norms(
                    &manifest.config,
                    4,
                    2,
                    &manifest.tensors,
                    WeightMode::Public,
                    manifest.execution,
                    &manifest.norms,
                )
                .unwrap();
                let Code::TensorLoop {
                    program,
                    tail,
                    input,
                } = &oracle.nodes[prefix + 2].code
                else {
                    panic!("original lookup loop")
                };
                assert_eq!(program.digest(), census.program.digest());
                assert_eq!(
                    tail.as_ref().map(|p| p.digest()),
                    census.tail.map(|p| p.digest())
                );
                assert!(matches!(input,LoopInput::TokenOffsets{owner,index:0} if *owner==prefix-1));
                for lane in 0..8 {
                    assert_eq!(
                        output.word(lane)?,
                        (prefix + 2 + (lane / 2) * 3 + 2, lane % 2)
                    );
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn genuine_compact_qkv_reshape_rotary_prefix_has_original_work_and_coordinates() {
    let (importer, manifest) = importer(4);
    importer
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (builder, context, _) = scope.parts();
            let prefix = builder.nodes.len();
            let mut graph = compact_lowering::CompactGraph::from_source_request(
                builder,
                context,
                MAX_PROGRAM_BYTES,
            )?;
            let mut projections = 0;
            let mut rotaries = 0;
            let mut caches = 0;
            let mut score_export = None;
            let (mut scores, mut scales, mut softmaxes, mut values) = (0, 0, 0, 0);
            for op in &context.plan().plan().prefill.operations {
                let output = graph.lower_next()?;
                if op.operator == Op::AttentionScores {
                    scores += 1;
                    let first = graph
                        .descriptor_census()
                        .last()
                        .ok_or(Error::Binding)?
                        .virtual_start;
                    let end = graph
                        .descriptor_census()
                        .last()
                        .ok_or(Error::Binding)?
                        .virtual_end;
                    score_export = Some((first, end, output.clone()));
                    for start in (0..output.lanes()).step_by(128) {
                        graph.ordered_chunk_bounds(
                            &output,
                            start,
                            (output.lanes() - start).min(128),
                        )?;
                    }
                }
                if op.operator == Op::AttentionScale {
                    scales += 1;
                }
                if op.operator == Op::Softmax {
                    softmaxes += 1;
                }
                if op.operator == Op::AttentionValues {
                    values += 1;
                }
                if op.operator == Op::KvCacheAppend {
                    caches += 1;
                }
                if op.operator == Op::Linear {
                    projections += 1;
                }
                if op.operator == Op::RotaryEmbedding {
                    rotaries += 1;
                    for lane in 0..op.output_shape.iter().product::<u64>() as usize {
                        let _ = output.word(lane)?;
                    }
                }
                if projections == 4 {
                    break;
                } // original attention projection consumer
            }
            assert_eq!((projections, rotaries, caches), (4, 2, 2));
            assert_eq!((scores, scales, softmaxes, values), (1, 1, 1, 1));
            assert_eq!(builder.nodes.len(), prefix, "no expanded candidate");
            context.check_final_work(graph.work_budget())?;
            let end = graph.descriptor_census().last().unwrap().virtual_end;
            // Independent expanded control AFTER all candidate native issuance.
            let oracle = Program::compile_with_norms(
                &manifest.config,
                4,
                2,
                &manifest.tensors,
                WeightMode::Public,
                manifest.execution,
                &manifest.norms,
            )
            .map_err(|_| Error::Binding)?;
            let expected = oracle.nodes[prefix + 2..end]
                .iter()
                .try_fold(WorkBudget::default(), |w, n| {
                    w.checked_add(n.work_budget()?)
                })?;
            assert_eq!(expected, context.work());
            for program in graph.owned_native_programs() {
                assert!(oracle.nodes[prefix + 2..end]
                    .iter()
                    .any(|node| match &node.code {
                        Code::Tensor {
                            program: original, ..
                        } =>
                            original.digest() == program.digest()
                                && original.work_budget() == program.work_budget(),
                        Code::TensorLoop {
                            program: original,
                            tail,
                            ..
                        } => std::iter::once(original)
                            .chain(tail.iter())
                            .any(|original| original.digest() == program.digest()
                                && original.work_budget() == program.work_budget()),
                        _ => false,
                    }));
            }
            // Actual publication/frozen reader, not a direct child helper. All
            // original query/key gathers and private products remain ordinal-
            // and proof-exact, including causal zero suffix export coordinates.
            use schedule_finalization::{BorrowedOrderedNodes, BorrowedRecipe};
            let (first, score_end, output) = score_export.ok_or(Error::Binding)?;
            let coordinates = (0..output.lanes())
                .map(|i| output.word(i))
                .collect::<Result<Vec<_>, _>>()?;
            let frozen = graph.freeze(vec![output], MAX_PROGRAM_BYTES)?;
            let reader = frozen.borrow_schedule(MAX_PROGRAM_BYTES)?;
            for ordinal in first..score_end {
                let actual = reader.node(ordinal)?;
                let expected = &oracle.nodes[ordinal];
                assert_eq!(
                    (actual.lanes, actual.bounds),
                    (expected.lanes, expected.bounds)
                );
                match (actual.recipe, &expected.code) {
                    (BorrowedRecipe::Gather(words), Code::Gather(original)) => {
                        assert_eq!(words.len(), original.len());
                        for i in 0..words.len() {
                            assert_eq!(words.lane_ref(i)?, original.at(i));
                        }
                    }
                    (
                        BorrowedRecipe::Product {
                            left,
                            right,
                            shape,
                            rescale,
                            norm,
                        },
                        Code::Product {
                            left: l,
                            right: r,
                            shape: s,
                            rescale: t,
                            norm: n,
                        },
                    ) => {
                        assert_eq!((left, right, shape, rescale), (*l, *r, *s, *t));
                        assert_eq!(norm.map(|p| p.digest()), n.as_ref().map(|p| p.digest()));
                    }
                    _ => return Err(Error::Binding),
                }
            }
            assert!(!coordinates.is_empty());
            Ok(())
        })
        .unwrap();
}

#[test]
fn source_chain_temporary_retirement_preserves_original_admitted_peak() {
    let (importer, _) = importer(4);
    let (source, peak) = importer
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (_, context, _) = scope.parts();
            let current = context.plan().budget().used_bytes();
            let before = context.historical_peak();
            assert!(before >= current);
            Ok((Arc::clone(context.plan()), before))
        })
        .unwrap();
    assert!(
        source.budget().used_bytes() < peak,
        "real manifest/import/permit temporaries retired"
    );
    assert_eq!(source.budget().admitted_peak_bytes(), peak);
}

#[test]
fn consuming_pending_prebox_bounds_completes_actual_slots_before_controller() {
    use source_entry::{compact_pending as pending, Event, Trace};
    let (entry, _) = importer(4);
    // Retain ONLY the observation counter before installing separate pressure.
    let observation = pending::test_root(&entry);
    let identity = pending::test_controller_identity(&entry);
    let trace = Trace::new(observation.budget());
    assert_eq!(trace.next_ordinal(), 1);
    assert_eq!(trace.budget_id(), identity.0);
    let pressure = pending::test_hold_remaining(&entry);
    assert_eq!(pending::test_controller_identity(&entry), identity);
    assert_eq!(observation.budget().used_bytes(), MAX_PROGRAM_BYTES);
    let mut entered = false;
    let result = entry.consume_compact_observed(&trace, |_, _, _, _, _| {
        entered = true;
        Ok(())
    });
    assert!(matches!(result, Err(Error::Bounds)));
    assert!(!entered, "prebox refusal never invokes emit");
    let controller = trace.completion(Event::ControllerCharge);
    assert!(controller > 0);
    for event in [Event::DirectorySlot, Event::OriginalSlot, Event::ManifestSlot, Event::RawSlot] {
        let completed = trace.completion(event);
        assert!(completed > 0 && completed < controller);
    }
    let mut completed = [
        trace.completion(Event::DirectorySlot),
        trace.completion(Event::OriginalSlot),
        trace.completion(Event::ManifestSlot),
        trace.completion(Event::RawSlot),
        controller,
    ];
    completed.sort_unstable();
    assert_eq!(completed, [1, 2, 3, 4, 5]);
    assert_eq!(controller, 5);
    assert_eq!(trace.next_ordinal(), 6);
    // All five completion markers belong to actual moved slots, including the
    // non-Clone controller. None-directory proves only its Option slot.
    // SOURCE sensitivity, not an executed mutation: old early destructuring
    // declares controller after raw/manifest/original/directory, so reverse local
    // destruction reports controller first on this same prebox Bounds return.
    // Counter values establish pressure/identity, never destruction order.
    drop(pressure);
    drop(observation);
}

#[test]
fn emission_constructor_contrived_undersize_completes_actual_created_before_guard() {
    use source_entry::{Event, Trace};
    let (entry, _) = importer(4);
    entry.with_created(|created, mut guard| {
        // Deliberate precondition refusal, NOT a naturally reachable production
        // sizing defect. Move this exact incoming controller; mint no replacement.
        let identity = guard.budget() as *const _ as usize;
        let trace = Trace::new(guard.budget());
        assert_eq!(trace.next_ordinal(), 1);
        assert_eq!(trace.budget_id(), identity);
        guard.resize_payload(0).unwrap();
        assert_eq!(guard.budget() as *const _ as usize, identity);
        assert_eq!(guard.payload_bytes(), 0);
        let result = compact_storage::EmissionScope::new_observed(
            created, guard, MAX_PROGRAM_BYTES, &trace,
        );
        assert!(matches!(result, Err(Error::Bounds)));
        let created = trace.completion(Event::CreatedInputs);
        let controller = trace.completion(Event::ControllerCharge);
        assert!(created > 0 && controller > created);
        assert_eq!(created, 1);
        assert_eq!(controller, 2);
        assert_eq!(trace.next_ordinal(), 3);
        // SOURCE sensitivity only: separate created/guard parameters without
        // PreEmissionScope retire guard before created on this required-size
        // refusal. The shared constructor body, not a historical fake, is traced.
        Ok(())
    }).unwrap();
}

#[test]
fn consuming_pending_early_cap_refusal_retires_importer_before_original_charge() {
    use source_entry::compact_pending as pending;
    let (entry, _) = importer(4);
    let hold = pending::test_hold_remaining(&entry);
    pending::test_drop_record();
    let result = entry.consume_compact(|_, _, _, _, _| -> Result<(), Error> {
        panic!("factory entered after cap refusal")
    });
    assert!(matches!(result, Err(Error::Bounds)));
    let (dead, used, charge) = pending::test_drop_record().unwrap();
    assert!(dead && charge > 0 && used >= charge);
    assert!(hold.budget().used_bytes() < used);
}

#[test]
fn consuming_pending_early_unwind_retires_importer_before_original_charge() {
    use source_entry::compact_pending as pending;
    let (entry, _) = importer(4);
    pending::test_drop_record();
    pending::test_panic_before_box();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        entry.consume_compact(|_, _, _, _, _| -> Result<(), Error> {
            panic!("factory entered before unwind")
        })
    }));
    assert!(failed.is_err());
    let (dead, used, charge) = pending::test_drop_record().unwrap();
    assert!(dead && charge > 0 && used >= charge);
}

#[test]
fn outer_public_payload_retires_before_its_own_reservation_on_success_and_failure() {
    for fail in [false, true] {
        let (importer, _) = importer(4);
        let mut retained_plan = None;
        let result: Result<(), Error> = importer.with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (_, context, _) = scope.parts();
            retained_plan = Some(Arc::clone(context.plan()));
            if fail {
                Err(Error::Allocation)
            } else {
                Ok(())
            }
        });
        assert_eq!(result.is_err(), fail);
        let (last_public_payload_retired, before_refund, charge) =
            super::super::source_entry::test_public_owner_drop().unwrap();
        assert!(
            last_public_payload_retired,
            "outer source payload dies before its guard"
        );
        assert!(charge > size_of::<OwnedExternalCharge>());
        assert!(
            before_refund >= charge,
            "same-counter reservation still live at payload retirement"
        );
        assert!(retained_plan.unwrap().budget().used_bytes() <= before_refund - charge);
    }
}

#[test]
fn consuming_scope_failure_burns_original_invocation() {
    let (importer, _) = importer(4);
    importer
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (_, context, _) = scope.parts();
            let origin = context.test_origin();
            drop(scope);
            assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
            Ok(())
        })
        .unwrap();
}

#[test]
fn request_lookup_exact_and_one_short_before_factory() {
    let (source_import, _) = importer(128);
    let exact = source_import
        .with_created(|created, guard| {
            let mut scope = compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            let (builder, context, _) = scope.parts();
            let mut graph = compact_lowering::CompactGraph::from_source_request(
                builder,
                context,
                MAX_PROGRAM_BYTES,
            )?;
            graph.lower_next()?;
            Ok(graph.construction_peak_bytes())
        })
        .unwrap();
    for limit in [exact, exact - 1] {
        let (importer, _) = importer(128);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let origin = context.test_origin();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                graph.test_lower_limit(limit);
                let before = compact_lowering::test_lookup_attempts();
                let result = graph.lower_next();
                if limit == exact {
                    result?;
                    assert!(compact_lowering::test_lookup_attempts() > before);
                } else {
                    assert!(matches!(result, Err(Error::Bounds)));
                    assert_eq!(compact_lowering::test_lookup_attempts(), before);
                    assert_eq!(origin.payload().state.load(Ordering::Acquire), BURNED);
                    assert!(matches!(
                        compact_lowering::CompactGraph::from_source_request(
                            builder,
                            context,
                            MAX_PROGRAM_BYTES
                        ),
                        Err(Error::Binding)
                    ));
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn request_lookup_factory_and_publication_failure_burn_without_early_refund() {
    for stage in [1, 2] {
        let (importer, _) = importer(128);
        importer
            .with_created(|created, guard| {
                let mut scope =
                    compact_storage::EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
                let (builder, context, _) = scope.parts();
                let origin = context.test_origin();
                let mut graph = compact_lowering::CompactGraph::from_source_request(
                    builder,
                    context,
                    MAX_PROGRAM_BYTES,
                )?;
                let before = context.reservation().payload_bytes();
                let attempts = compact_lowering::test_lookup_attempts();
                compact_lowering::test_lookup_failure(stage);
                assert!(matches!(graph.lower_next(), Err(Error::Allocation)));
                assert_eq!(
                    origin.payload().state.load(Ordering::Acquire),
                    BURNED,
                    "scope still alive"
                );
                assert!(context.work().operations() > 0);
                assert!(compact_lowering::test_lookup_attempts() > attempts);
                let charged = context.reservation().payload_bytes();
                assert!(charged >= before);
                let native = graph
                    .token_lookup_census()
                    .next()
                    .map(|c| Arc::downgrade(c.program));
                assert_eq!(native.is_some(), stage == 2);
                if let Some(native) = &native {
                    assert!(native.upgrade().is_some());
                }
                assert!(matches!(
                    compact_lowering::CompactGraph::from_source_request(
                        builder,
                        context,
                        MAX_PROGRAM_BYTES
                    ),
                    Err(Error::Binding)
                ));
                drop(graph);
                if let Some(native) = native {
                    assert!(native.upgrade().is_none());
                }
                assert_eq!(
                    context.reservation().payload_bytes(),
                    charged,
                    "no refund on payload destruction/failure"
                );
                Ok(())
            })
            .unwrap();
    }
}
