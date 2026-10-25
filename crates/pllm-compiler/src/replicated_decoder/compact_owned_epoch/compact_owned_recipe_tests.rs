//! Registered genuine-ingress inspection tests; no into_test_payload shortcut.
//! These are source definitions until an explicit compiler-test slot is granted.
use super::*;
use crate::replicated_decoder::program::request_origin::compact_storage::{
    owning::OwnedEmission, EmissionScope,
};
use crate::replicated_decoder::program::source_entry::{
    compact_pending::SourceCustody, SourceRequestImport,
};
use crate::replicated_decoder::{ExecutionPolicy, Publisher, RangePolicy, WeightMode};
use std::collections::BTreeMap;

type Custody = SourceCustody<OwnedEmission<OwnedEpoch>>;
#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    digest: [u8; 32],
    work: WorkBudget,
    geometry: Vec<(usize, Interval)>,
    native: Vec<(usize, usize)>,
    retired: Vec<Vec<usize>>,
}
fn snapshot(schedule: &impl BorrowedOrderedNodes) -> Result<Snapshot, Error> {
    let result =
        schedule_finalization::finalize(schedule, &[], &Sha256::new(), 0, MAX_PROGRAM_BYTES)?;
    let mut geometry = Vec::new();
    let mut native = Vec::new();
    for ordinal in 0..schedule.node_count() {
        let node = schedule.node(ordinal)?;
        geometry.push((node.lanes, node.bounds));
        match node.recipe {
            BorrowedRecipe::Tensor { program, .. }
            | BorrowedRecipe::CheckedRange { program, .. } => {
                native.push((ordinal, Arc::as_ptr(program) as usize))
            }
            BorrowedRecipe::TensorLoop { program, tail, .. } => {
                native.push((ordinal, Arc::as_ptr(program) as usize));
                if let Some(tail) = tail {
                    native.push((ordinal, Arc::as_ptr(tail) as usize));
                }
            }
            _ => (),
        }
    }
    Ok(Snapshot {
        digest: result.digest,
        work: result.work,
        geometry,
        native,
        retired: result.retire,
    })
}

// Same ordinary generated fixture + source publisher as the existing owning
// epoch tests. No source hash/Instance/request owner is manufactured here.
fn entry(width: usize, unsafe_row: bool) -> SourceRequestImport {
    let (config, _, mut clear) = crate::replicated_decoder::tests::fixture_sources("qwen2", false);
    if unsafe_row {
        clear.get_mut("model.embed_tokens.weight").unwrap()[0] = 3 * ONE / 2;
    }
    let mut config: serde_json::Value = serde_json::from_slice(&config).unwrap();
    config["vocab_size"] = serde_json::json!(3);
    config["hidden_size"] = serde_json::json!(width);
    if config.get("head_dim").is_some() {
        config["head_dim"] = serde_json::json!(width);
    }
    let config = serde_json::to_vec(&config).unwrap();
    let mut publisher = Publisher::new_with_policy(
        &config,
        4,
        2,
        [1; 32],
        [2; 32],
        WeightMode::Public,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        },
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
    publisher.seal(&[1, 2, 1, 2]).unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&publisher.manifest().unwrap()).unwrap();
    manifest.program = [0; 32];
    let mut entry = SourceRequestImport::new(&serde_json::to_vec(&manifest).unwrap()).unwrap();
    for (key, bytes) in payloads {
        entry.append(&key, 0, &bytes).unwrap();
    }
    entry
}

#[derive(Clone, Copy)]
enum Corruption {
    None,
    Origin,
    Source,
    Mass,
    Checked,
    Energy,
    EqualDigest,
    MissingInvocation,
    QuoteOverflow,
}
fn build(width: usize, unsafe_row: bool, corrupt: Corruption) -> (Custody, Snapshot) {
    let mut before = None;
    let custody = entry(width, unsafe_row)
        .consume_compact(|created, guard, _, _, source| {
            let scope = EmissionScope::new(created, guard, MAX_PROGRAM_BYTES)?;
            scope.into_owned(|imports, context, _| {
                let mut graph =
                    CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
                for (step, operation) in context.plan().plan().prefill.operations.iter().enumerate() {
                    graph.lower_next().inspect_err(|e| { eprintln!("fixture lower {step} operator={:?} shape={:?}: {e:?}", operation.operator, operation.output_shape); })?;
                }
                // Exact production freeze_owned precharge, with an intervening
                // BORROWED inspection before the same frozen records move.
                let exports = Vec::new();
                let quote = graph.freeze_quote(&exports)?;
                let new = add_bytes(
                    quote.new_backing_bytes,
                    2 * size_of::<Result<FrozenRegion<'_>, Error>>(),
                )?;
                context.admit_phase(new, 0)?;
                context.retain(new)?;
                let frozen = graph.freeze(exports, MAX_PROGRAM_BYTES)?;
                {
                    let reader = frozen.borrow_schedule(MAX_PROGRAM_BYTES)?;
                    before = Some(snapshot(&reader).inspect_err(|e| { eprintln!("fixture borrowed snapshot: {e:?}"); })?);
                }
                let mut epoch = frozen.into_owned(context, source).inspect_err(|e| { eprintln!("fixture transfer: {e:?}"); })?;
                match corrupt {
                    Corruption::None => (),
                    Corruption::QuoteOverflow => epoch.next = usize::MAX,
                    Corruption::MissingInvocation => {
                        let values = epoch.ops.iter_mut().find_map(|op| match &mut op.kind {
                            Kind::Values(v) => Some(v), _ => None,
                        }).ok_or(Error::Binding)?;
                        let row = values.rows.first_mut().ok_or(Error::Binding)?;
                        assert!(row.probability.take().is_some(), "delete a REAL retained native association");
                    }
                    Corruption::EqualDigest => {
                        let weight = |op: &Op| match &op.kind {
                            Kind::Lookup { weight, .. } | Kind::Projection { weight, .. } => Some(*weight),
                            _ => None,
                        };
                        let mut pair = None;
                        'outer: for a in 0..epoch.ops.len() {
                            let Some(left) = weight(&epoch.ops[a]) else { continue; };
                            let left_node = &imports.nodes[left];
                            for b in a + 1..epoch.ops.len() {
                                let Some(right) = weight(&epoch.ops[b]) else { continue; };
                                let right_node = &imports.nodes[right];
                                if let (Code::PublicInput { key: ak, digest: ad, .. }, Code::PublicInput { key: bk, digest: bd, .. }) = (&left_node.code, &right_node.code) {
                                    if left != right && ak != bk && ad == bd && left_node.lanes == right_node.lanes {
                                        pair = Some((a, b, left, right)); break 'outer;
                                    }
                                }
                            }
                        }
                        let (a, b, left, right) = pair.expect("fixture must contain distinct equal-digest declarations; no fabricated digest");
                        for (slot, replacement) in [(a, right), (b, left)] {
                            match &mut epoch.ops[slot].kind {
                                Kind::Lookup { weight, .. } | Kind::Projection { weight, .. } => *weight = replacement,
                                _ => unreachable!(),
                            }
                        }
                    }
                    Corruption::Origin => epoch.ops[0].input.origin = Arc::new(()),
                    Corruption::Source => {
                        let op = epoch
                            .ops
                            .iter_mut()
                            .find(|op| matches!(&op.kind, Kind::Lookup { .. }))
                            .ok_or(Error::Binding)?;
                        if let Kind::Lookup { weight, .. } = &mut op.kind {
                            *weight = usize::MAX;
                        }
                    }
                    Corruption::Mass => {
                        let values = epoch
                            .ops
                            .iter_mut()
                            .find_map(|op| {
                                if let Kind::Values(v) = &mut op.kind {
                                    Some(v)
                                } else {
                                    None
                                }
                            })
                            .ok_or(Error::Binding)?;
                        assert!(values.rows.iter().any(|r| r.probability.is_some()));
                        let tile = values.tiles.first_mut().ok_or(Error::Binding)?;
                        tile.bounds = Interval::new(
                            tile.bounds.lower(),
                            tile.bounds.upper().checked_add(1).ok_or(Error::Bounds)?,
                        )?;
                    }
                    Corruption::Checked => {
                        let checked = epoch
                            .ops
                            .iter_mut()
                            .find_map(|op| {
                                if let Kind::Rms {
                                    checked: Some(rows),
                                    ..
                                } = &mut op.kind
                                {
                                    Some(rows)
                                } else {
                                    None
                                }
                            })
                            .ok_or(Error::Binding)?;
                        checked.first = checked.first.checked_add(1).ok_or(Error::Bounds)?;
                    }
                    Corruption::Energy => {
                        let tile = epoch
                            .ops
                            .iter_mut()
                            .find_map(|op| match &mut op.kind {
                                Kind::Scores(scores) => {
                                    scores.tiles.iter_mut().find(|t| t.proof.is_some())
                                }
                                _ => None,
                            })
                            .ok_or(Error::Binding)?;
                        tile.inputs.left = Interval::new(
                            tile.inputs.left.lower(),
                            tile.inputs
                                .left
                                .upper()
                                .checked_add(1)
                                .ok_or(Error::Bounds)?,
                        )?;
                    }
                }
                Ok(epoch)
            })
        })
        .unwrap();
    (custody, before.unwrap())
}

fn parity(width: usize, guarded: bool) {
    let (custody, before) = build(width, guarded, Corruption::None);
    let loan = custody.inspect_epoch().unwrap();
    let schedule = OwnedSchedule::new(loan).unwrap();
    let after = snapshot(&schedule).unwrap();
    assert_eq!(
        before, after,
        "all ordered canonical recipes, work, retirements and ORIGINAL native pointers"
    );
    assert_eq!(schedule.work(), after.work);
    assert!(!after.native.is_empty());
    assert!(!schedule.loan.epoch().ops.is_empty());
    assert!(schedule.loan.epoch().ops.iter().any(|op| matches!(
        &op.kind,
        Kind::CausalMask | Kind::Reshape | Kind::CacheSuffix
    )));
    assert!(schedule
        .loan
        .epoch()
        .ops
        .iter()
        .any(|op| matches!(&op.kind, Kind::Softmax)));
}
#[test]
fn owned_reader_all_recipes_match_before_and_after_transfer() {
    parity(2, false);
}
#[test]
fn owned_reader_checked_rows_match_original_guards_and_native_arcs() {
    parity(2, true);
}
#[test]
fn owned_reader_variable_causal_tail_coordinates_match_original() {
    parity(36, true);
}
#[test]
fn owned_reader_closed_original_attempt_cannot_be_reopened() {
    let (custody, _) = build(2, false, Corruption::None);
    {
        let loan = custody.inspect_epoch().unwrap();
        let _abort = loan.attempt().unwrap();
    }
    assert!(custody.inspect_epoch().is_err());
}
#[test]
fn owned_reader_foreign_origin_rejects_without_rebinding() {
    let (custody, _) = build(2, false, Corruption::Origin);
    let loan = custody.inspect_epoch().unwrap();
    assert!(OwnedSchedule::new(loan).is_err());
    assert!(
        custody.inspect_epoch().is_ok(),
        "structural reader failure does not complete/burn original history"
    );
}
#[test]
fn owned_reader_missing_source_import_rejects() {
    let (custody, _) = build(2, false, Corruption::Source);
    assert!(OwnedSchedule::new(custody.inspect_epoch().unwrap()).is_err());
}
#[test]
fn owned_reader_actual_mass_bound_mismatch_rejects() {
    let (custody, _) = build(2, false, Corruption::Mass);
    assert!(OwnedSchedule::new(custody.inspect_epoch().unwrap()).is_err());
}
#[test]
fn owned_reader_actual_checked_row_ordinals_mismatch_rejects() {
    let (custody, _) = build(2, true, Corruption::Checked);
    assert!(OwnedSchedule::new(custody.inspect_epoch().unwrap()).is_err());
}
#[test]
fn owned_reader_missing_invocation_is_not_a_uniform_row_fallback() {
    let (custody, _) = build(2, false, Corruption::MissingInvocation);
    assert!(OwnedSchedule::new(custody.inspect_epoch().unwrap()).is_err());
}

#[test]
fn owned_reader_distinct_equal_digest_import_swap_rejects() {
    let (custody, _) = build(2, false, Corruption::EqualDigest);
    assert!(OwnedSchedule::new(custody.inspect_epoch().unwrap()).is_err());
}

#[test]
fn owned_reader_quote_overflow_refuses_before_source_lens_allocation() {
    let (custody, _) = build(2, false, Corruption::QuoteOverflow);
    assert!(matches!(
        OwnedSchedule::new(custody.inspect_epoch().unwrap()),
        Err(Error::Bounds)
    ));
}

#[test]
fn owned_reader_actual_row_energy_proof_mismatch_rejects() {
    let (custody, _) = build(2, false, Corruption::Energy);
    assert!(OwnedSchedule::new(custody.inspect_epoch().unwrap()).is_err());
}
#[test]
fn owned_reader_generated_iota_is_original_owner_bound_recipe() {
    let (custody, before) = build(2, false, Corruption::None);
    let reader = OwnedSchedule::new(custody.inspect_epoch().unwrap()).unwrap();
    let mut iotas = 0;
    for (slot, op) in reader.loan.epoch().ops.iter().enumerate() {
        if matches!(&op.kind, Kind::Argmax(_)) {
            assert_eq!(
                reader.loan.epoch().argmax_lanes(slot).unwrap(),
                op.input.lanes
            );
            let node = reader.node(op.first).unwrap();
            assert!(matches!(
                node.recipe,
                BorrowedRecipe::GeneratedPublicValues(_)
            ));
            assert_eq!(
                node.bounds,
                Interval::new(0, op.input.lanes as i64 - 1).unwrap()
            );
            iotas += 1;
        }
    }
    assert!(
        iotas > 0,
        "fixture must genuinely lower argmax, not invent an iota source"
    );
    assert_eq!(before, snapshot(&reader).unwrap());
    // Inspection only. Existing invalid same-graph Argmax acceptance test is
    // untouched and remains unqualified; no history completion is issued here.
}
#[test]
fn owned_reader_distinct_origins_retain_their_own_source_lenses() {
    let (first, _) = build(2, false, Corruption::None);
    let (second, _) = build(2, false, Corruption::None);
    let first = OwnedSchedule::new(first.inspect_epoch().unwrap()).unwrap();
    let second = OwnedSchedule::new(second.inspect_epoch().unwrap()).unwrap();
    assert!(first.loan.epoch().instance != second.loan.epoch().instance);
    let mut compared = 0;
    for index in 0..first.loan.epoch().recipe_count() {
        let a = first.loan.sources(index).unwrap();
        let b = second.loan.sources(index).unwrap();
        if let (Some(a), Some(b)) = (a.primary(), b.primary()) {
            assert!(!std::ptr::eq(a, b));
            compared += 1;
        }
    }
    assert!(compared > 0);
}
