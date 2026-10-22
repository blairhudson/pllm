//! Original Builder publication contracts only; no finalizer/manifest/Worker
//! capture or source/row authority is supplied by these local declarations.
use super::*;
use crate::replicated_decoder::prepared_tensor_map::{OneLoopScope, PreparedTensorMap};

fn domain() -> Interval {
    Interval::new(-ONE, ONE).unwrap()
}

fn make(lanes: usize, bounds: Interval, silu: bool) -> tensor::Program {
    if silu {
        tensor::Program::activation_guarded(tensor::Activation::Silu, lanes, bounds).unwrap()
    } else {
        let mut b = tensor::Builder::new(lanes, bounds).unwrap();
        let out = b.rescale(b.input(), 18).unwrap();
        b.finish(out).unwrap()
    }
}

fn prepare(lanes: usize, silu: bool) -> PreparedTensorMap {
    if silu {
        PreparedTensorMap::guarded_silu(lanes, domain(), 0, |_| Ok(())).unwrap()
    } else {
        PreparedTensorMap::rescale(lanes, domain(), 18, 0, |_| Ok(())).unwrap()
    }
}

fn builder(lanes: usize, bounds: Interval) -> Builder {
    let mut b = Builder::default();
    assert_eq!(
        b.push(Code::Input("local".into()), lanes, bounds).unwrap(),
        0
    );
    b
}

// Exact TensorLoop portion of the existing canonical execution encoding,
// including output hull and input discriminator. Return bytes, not just hashes.
fn logical_record(node: &Node) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(node.lanes as u64).to_le_bytes());
    bytes.extend_from_slice(&node.bounds.lower().to_le_bytes());
    bytes.extend_from_slice(&node.bounds.upper().to_le_bytes());
    let Code::TensorLoop {
        input,
        program,
        tail,
    } = &node.code
    else {
        panic!("not a tensor loop")
    };
    bytes.push(8);
    match input {
        LoopInput::Slice(owner) => {
            bytes.push(0);
            bytes.extend_from_slice(&(*owner as u64).to_le_bytes());
        }
        LoopInput::TokenOffsets { owner, index } => {
            bytes.push(1);
            bytes.extend_from_slice(&(*owner as u64).to_le_bytes());
            bytes.extend_from_slice(&(*index as u64).to_le_bytes());
        }
    }
    bytes.extend_from_slice(&program.digest());
    bytes.push(u8::from(tail.is_some()));
    if let Some(tail) = tail {
        bytes.extend_from_slice(&tail.digest());
    }
    bytes
}

fn assert_originals(b: &Builder, id: usize, prepared: &PreparedTensorMap) {
    let Code::TensorLoop {
        input,
        program,
        tail,
    } = &b.nodes[id].code
    else {
        panic!("not a tensor loop")
    };
    assert!(matches!(input, LoopInput::Slice(0)));
    let stored: Vec<_> = std::iter::once(program).chain(tail).collect();
    let originals: Vec<_> = prepared.programs().collect();
    assert_eq!(stored.len(), originals.len());
    for (stored, original) in stored.into_iter().zip(originals) {
        assert!(Arc::ptr_eq(stored, original));
        assert!(Arc::ptr_eq(&b.tensors[&original.digest()], original));
    }
}

#[test]
fn prepared_publication_matches_closure_full_and_tail_without_owner_substitution() {
    for silu in [false, true] {
        for lanes in [1, 128, 129, 257] {
            let prepared = prepare(lanes, silu);
            let mut old = builder(lanes, domain());
            let mut new = builder(lanes, domain());
            let old_id = old
                .tensor_loop(LoopInput::Slice(0), lanes, |n| Ok(make(n, domain(), silu)))
                .unwrap();
            let new_id = new.prepared_tensor_map(0, &prepared).unwrap();
            assert_eq!(old_id, new_id);
            assert_eq!(old.nodes.len(), new.nodes.len());
            assert_eq!(old.nodes[old_id].inputs(), new.nodes[new_id].inputs());
            assert_eq!(
                logical_record(&old.nodes[old_id]),
                logical_record(&new.nodes[new_id])
            );
            assert_eq!(new.nodes[new_id].bounds, prepared.output_bounds().unwrap());
            assert_originals(&new, new_id, &prepared);
            assert_eq!(old.allocated, new.allocated);
            assert_eq!(old.value_payload_bytes, new.value_payload_bytes);
            assert_eq!(old.workspace_bytes, new.workspace_bytes);
            assert_eq!(old.tensors.len(), new.tensors.len());
            for (digest, original) in &new.tensors {
                assert_eq!(
                    old.tensors[digest].resources().unwrap(),
                    original.resources().unwrap()
                );
                assert_eq!(old.tensors[digest].work_budget(), original.work_budget());
            }
            assert_eq!(
                schedule_work(&old.nodes).unwrap(),
                schedule_work(&new.nodes).unwrap()
            );
            assert_eq!(
                schedule_work(&new.nodes).unwrap(),
                prepared.work_budget().unwrap()
            );
        }
    }
}

// Diagnostics may report an attempted admission; published nodes, accounting,
// template owners and schedule must not change on any rejection.
fn assert_rejected_unchanged(b: &mut Builder, owner: usize, prepared: &PreparedTensorMap) {
    assert_publication_rejected_unchanged(b, |b| b.prepared_tensor_map(owner, prepared));
}

fn assert_publication_rejected_unchanged(
    b: &mut Builder,
    publish: impl FnOnce(&mut Builder) -> Result<usize, Error>,
) -> Error {
    let nodes = b.nodes.len();
    let allocated = b.allocated;
    let payload = b.value_payload_bytes;
    let workspace = b.workspace_bytes;
    let work = schedule_work(&b.nodes).unwrap();
    let owners: Vec<_> = b
        .tensors
        .iter()
        .map(|(d, p)| (*d, Arc::as_ptr(p)))
        .collect();
    let error = publish(b).unwrap_err();
    assert_eq!(b.nodes.len(), nodes);
    assert_eq!(b.allocated, allocated);
    assert_eq!(b.value_payload_bytes, payload);
    assert_eq!(b.workspace_bytes, workspace);
    assert_eq!(schedule_work(&b.nodes).unwrap(), work);
    assert_eq!(
        b.tensors
            .iter()
            .map(|(d, p)| (*d, Arc::as_ptr(p)))
            .collect::<Vec<_>>(),
        owners
    );
    error
}

#[test]
fn shared_tensor_loop_publication_rejects_all_malformed_geometry_classes() {
    let full = Arc::new(make(128, domain(), false));
    let one = Arc::new(make(1, domain(), false));
    // Bounds must be rejected before any modulo or publication, even with
    // unrelated full/tail widths and a maximal usize lane count.
    for lanes in [0, MAX_VALUES + 1, usize::MAX] {
        for program in [&one, &full] {
            for tail in [None, Some(one.clone()), Some(full.clone())] {
                let mut b = builder(129, domain());
                let error = assert_publication_rejected_unchanged(&mut b, |b| {
                    b.publish_tensor_loop(LoopInput::Slice(0), lanes, program.clone(), tail)
                });
                assert_eq!(error, Error::Bounds);
            }
        }
    }

    // Every possible tile width requires NO tail for an exact single tile.
    // Also cover multiple full tiles; an extra tail is never silently ignored.
    for lanes in (1..=128).chain([256, 384]) {
        let program = Arc::new(make(lanes.min(128), domain(), false));
        let mut b = builder(lanes, domain());
        let error = assert_publication_rejected_unchanged(&mut b, |b| {
            b.publish_tensor_loop(LoopInput::Slice(0), lanes, program, Some(one.clone()))
        });
        assert_eq!(error, Error::Shape);
    }

    // Every nonzero remainder requires exactly that tail width, for both one
    // and multiple full tiles. Test missing, undersized and oversized tails.
    for remainder in 1..128 {
        for tiles in [1, 2] {
            let lanes = tiles * 128 + remainder;
            let wrong_width = if remainder == 1 { 2 } else { remainder - 1 };
            for tail in [
                None,
                Some(Arc::new(make(wrong_width, domain(), false))),
                Some(full.clone()),
            ] {
                let mut b = builder(lanes, domain());
                let error = assert_publication_rejected_unchanged(&mut b, |b| {
                    b.publish_tensor_loop(LoopInput::Slice(0), lanes, full.clone(), tail)
                });
                assert_eq!(error, Error::Shape);
            }
        }
    }

    // Full input width is checked independently, with and without a tail.
    for (lanes, program, tail) in [
        (1, full.clone(), None),
        (128, one.clone(), None),
        (129, one.clone(), Some(one.clone())),
    ] {
        let mut b = builder(lanes, domain());
        let error = assert_publication_rejected_unchanged(&mut b, |b| {
            b.publish_tensor_loop(LoopInput::Slice(0), lanes, program, tail)
        });
        assert_eq!(error, Error::Shape);
    }
}

#[test]
fn shared_tensor_loop_publication_rejects_full_and_tail_output_width_changes() {
    let reduce = |lanes| {
        let mut b = tensor::Builder::new(lanes, domain()).unwrap();
        let out = b
            .row_map(b.input(), lanes, multiply::arithmetic::RowMap::Sum)
            .unwrap();
        let program = Arc::new(b.finish(out).unwrap());
        assert_eq!(program.lanes(), lanes);
        assert_eq!(program.output_lanes(), 1);
        program
    };
    for (lanes, program, tail) in [
        (128, reduce(128), None),
        (130, reduce(128), Some(Arc::new(make(2, domain(), false)))),
        (130, Arc::new(make(128, domain(), false)), Some(reduce(2))),
        (130, reduce(128), Some(reduce(2))),
    ] {
        let mut b = builder(lanes, domain());
        let error = assert_publication_rejected_unchanged(&mut b, |b| {
            b.publish_tensor_loop(LoopInput::Slice(0), lanes, program, tail)
        });
        assert_eq!(error, Error::Shape);
    }
}

#[test]
fn prepared_publication_checks_local_owner_width_domain_and_caps() {
    for silu in [false, true] {
        let prepared = prepare(129, silu);
        assert_rejected_unchanged(&mut builder(129, domain()), 1, &prepared);
        assert_rejected_unchanged(&mut builder(128, domain()), 0, &prepared);
        assert_rejected_unchanged(
            &mut builder(129, Interval::new(-ONE - 1, ONE).unwrap()),
            0,
            &prepared,
        );
        assert_rejected_unchanged(
            &mut builder(129, Interval::new(-ONE, ONE + 1).unwrap()),
            0,
            &prepared,
        );
        let mut capped = builder(129, domain());
        capped.allocated = MAX_PROGRAM_BYTES;
        assert_rejected_unchanged(&mut capped, 0, &prepared);

        let mut narrower = builder(129, Interval::new(-ONE / 2, ONE / 2).unwrap());
        let id = narrower.prepared_tensor_map(0, &prepared).unwrap();
        // Keep captured factory output hull; do not re-infer tighter bounds.
        assert_eq!(narrower.nodes[id].bounds, prepared.output_bounds().unwrap());
        assert_originals(&narrower, id, &prepared);
    }
}

#[test]
fn prepared_publication_rejects_foreign_full_or_tail_without_substitution() {
    for silu in [false, true] {
        for lanes in [1, 128, 129, 257] {
            let prepared = prepare(lanes, silu);
            for original in prepared.programs() {
                let mut b = builder(lanes, domain());
                // Populate via the legacy path, not a synthetic allowance.
                let input = b
                    .push(Code::Input("template".into()), original.lanes(), domain())
                    .unwrap();
                b.tensor(input, make(original.lanes(), domain(), silu))
                    .unwrap();
                let foreign = &b.tensors[&original.digest()];
                assert_eq!(foreign.digest(), original.digest());
                assert!(!Arc::ptr_eq(foreign, original));
                assert_rejected_unchanged(&mut b, 0, &prepared);
                assert_eq!(
                    prepared.one_loop_scope(),
                    Some(if silu {
                        OneLoopScope::GuardedSilu
                    } else {
                        OneLoopScope::Rescale18
                    })
                );
            }
        }
    }
}

#[test]
fn prepared_publication_reuses_populated_original_owners_and_legacy_still_interns() {
    for silu in [false, true] {
        for lanes in [1, 128, 129, 257] {
            let prepared = prepare(lanes, silu);
            let mut b = builder(lanes, domain());
            let first = b.prepared_tensor_map(0, &prepared).unwrap();
            let allocated = b.allocated;
            let second = b.prepared_tensor_map(0, &prepared).unwrap();
            assert_originals(&b, first, &prepared);
            assert_originals(&b, second, &prepared);
            assert_eq!(
                b.allocated - allocated,
                4096 + std::mem::size_of::<Node>() + 16 * lanes
            );
            let legacy = b
                .tensor_loop(LoopInput::Slice(0), lanes, |n| Ok(make(n, domain(), silu)))
                .unwrap();
            assert_originals(&b, legacy, &prepared);
            assert_eq!(
                logical_record(&b.nodes[first]),
                logical_record(&b.nodes[legacy])
            );
        }
    }
}

#[test]
fn prepared_publication_exact_one_short_and_recovery_reuse_originals() {
    for silu in [false, true] {
        for lanes in [1, 129] {
            let prepared = prepare(lanes, silu);
            // Independent literal legacy publication charge, NOT a claim that
            // the pressure fixture physically owns MAX_PROGRAM_BYTES of heap.
            let required = prepared
                .programs()
                .map(|program| program.admitted_bytes() + 1024)
                .sum::<usize>()
                + 4096
                + std::mem::size_of::<Node>()
                + 16 * lanes;
            let before = crate::replicated_decoder::prepared_tensor_map::native_attempts();
            let mut exact = builder(lanes, domain());
            exact.allocated = MAX_PROGRAM_BYTES - required;
            let output = exact.prepared_tensor_map(0, &prepared).unwrap();
            assert_eq!(exact.allocated, MAX_PROGRAM_BYTES);
            assert_originals(&exact, output, &prepared);
            let mut short = builder(lanes, domain());
            short.allocated = MAX_PROGRAM_BYTES - required + 1;
            let error = assert_publication_rejected_unchanged(&mut short, |b| {
                b.prepared_tensor_map(0, &prepared)
            });
            assert_eq!(error, Error::Bounds);
            assert!(short.tensors.is_empty());
            assert_eq!(short.nodes.len(), 1);
            // Retry the SAME capability/owners after removing pressure. This
            // tests admission rollback, not unavailable allocator failpoints.
            short.allocated = MAX_PROGRAM_BYTES - required;
            let output = short.prepared_tensor_map(0, &prepared).unwrap();
            assert_eq!(short.allocated, MAX_PROGRAM_BYTES);
            assert_originals(&short, output, &prepared);
            assert_eq!(
                crate::replicated_decoder::prepared_tensor_map::native_attempts(),
                before
            );
        }
    }
}
