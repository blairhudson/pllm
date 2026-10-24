//! Staged expanded publication parity, not enabled decoder dispatch. Compare raw
//! logical records and schedules, then exercise the shared original finalizer.
//! The ten full-graph frozen snapshots remain independent.
use super::*;
use crate::replicated_decoder::{
    prepared_tensor_map::{OneLoopScope, PreparedTensorMap},
    runtime::body,
};

fn domain() -> Interval {
    Interval::new(-ONE, ONE).unwrap()
}

#[test]
fn protected_worker_variants_keep_original_templates_and_compiler_retirement() {
    for lanes in [1, 128, 129, 257] {
        for retained in [false, true] {
            for sticky in [false, true] {
                let (program, prepared, input, output, valid) =
                    prepared_rescale18_protected_worker_fixture(lanes, domain(), retained, sticky)
                        .unwrap();
                assert_eq!(input, 0);
                assert_eq!(valid.is_some(), sticky);
                assert_eq!(
                    program.nodes[output].bounds,
                    prepared.output_bounds().unwrap()
                );
                let Code::TensorLoop {
                    program: full,
                    tail,
                    input: owner,
                } = &program.nodes[output].code
                else {
                    panic!("original prepared TensorLoop required");
                };
                assert!(matches!(owner, LoopInput::Slice(0)));
                assert!(Arc::ptr_eq(full, prepared.program_at(0).unwrap()));
                if let Some(tail) = tail {
                    assert!(Arc::ptr_eq(
                        tail,
                        prepared.program_at(lanes - lanes % 128).unwrap()
                    ));
                }
                assert!(matches!(
                    program.nodes[output + 1].code,
                    Code::Product { .. }
                ));
                assert_eq!(program.retire[output].contains(&input), !retained);
                if let Some(valid) = valid {
                    assert!(!program.retire[output].contains(&valid));
                    assert!(program.outputs.contains(&(valid, 0)));
                }
                assert_ne!(program.work_budget(), prepared.work_budget().unwrap());
            }
        }
    }
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
fn worker_fixture_preserves_factory_owners_identity_and_original_retirement() {
    for lanes in [1, 128, 129, 257] {
        let (program, prepared, input, output) =
            prepared_rescale18_worker_fixture(lanes, domain()).unwrap();
        assert_eq!((input, output), (0, 1));
        assert_eq!(program.inputs, 1);
        assert_eq!(program.inputs().collect::<Vec<_>>(), vec![("local", lanes)]);
        assert_eq!(program.nodes.len(), output + 2);
        assert_eq!(prepared.one_loop_scope(), Some(OneLoopScope::Rescale18));
        prepared.validate().unwrap();

        let Code::TensorLoop {
            input: LoopInput::Slice(owner),
            program: full,
            tail,
        } = &program.nodes[output].code
        else {
            panic!("fixture must publish an original Slice TensorLoop")
        };
        assert_eq!(*owner, input);
        let originals: Vec<_> = prepared.programs().cloned().collect();
        let stored: Vec<_> = std::iter::once(full).chain(tail).collect();
        assert_eq!(stored.len(), originals.len());
        for (stored, original) in stored.into_iter().zip(&originals) {
            assert!(Arc::ptr_eq(stored, original));
        }
        assert_eq!(
            program.nodes[output].bounds,
            prepared.output_bounds().unwrap()
        );
        assert_eq!(program.work_budget(), prepared.work_budget().unwrap());
        assert_eq!(
            program.work_budget(),
            schedule_work(&program.nodes).unwrap()
        );

        // Compare actual canonical loop bytes against the original closure path,
        // independently of factory provenance and pointer-identity checks above.
        let mut legacy = builder(lanes, domain());
        let legacy_output = legacy
            .tensor_loop(LoopInput::Slice(input), lanes, |n| {
                Ok(make(n, domain(), false))
            })
            .unwrap();
        assert_eq!(
            logical_record(&program.nodes[output]),
            logical_record(&legacy.nodes[legacy_output])
        );
        assert_eq!(
            program.outputs,
            (0..lanes).map(|index| (output, index)).collect::<Vec<_>>()
        );
        // Source's last use is the whole loop. A real downstream original node
        // supplies its exact-end boundary without retaining source or dead result.
        assert!(matches!(
            program.nodes[output + 1].code,
            Code::Affine { left, right: None, .. } if left == output
        ));
        assert_eq!(program.retire, vec![vec![], vec![input], vec![output + 1]]);
        assert!(program.nodes[output + 1].inputs().contains(&output));
        assert!(program.payload_bytes() > legacy.allocated);

        // Independent builds have identical finalizer identities, NOT identical
        // allocations. Neither may substitute for the captured original owners.
        let (again, again_prepared, again_input, again_output) =
            prepared_rescale18_worker_fixture(lanes, domain()).unwrap();
        assert_eq!((again_input, again_output), (input, output));
        assert_eq!(program.digest(), again.digest());
        assert_eq!(program.source_binding, again.source_binding);
        assert_eq!(program.payload_bytes(), again.payload_bytes());
        assert!(!Arc::ptr_eq(&program, &again));
        for (original, other) in originals.iter().zip(again_prepared.programs()) {
            assert_eq!(original.digest(), other.digest());
            assert!(!Arc::ptr_eq(original, other));
        }

        // Moving the SAME capability into the semantic sink leaves expanded
        // ownership intact. No independently reconstructed template is accepted.
        let mut semantic = body::layer::Builder::new(
            WeightMode::Public,
            &[body::Register {
                lanes,
                bounds: domain(),
            }],
        )
        .unwrap();
        let semantic_input = semantic.input(0).unwrap();
        let semantic_output = semantic
            .prepared_tensor_map(&semantic_input, prepared, program.payload_bytes())
            .unwrap();
        let plan = semantic.finish(&[semantic_output]).unwrap();
        assert_eq!(plan.work_budget(), program.work_budget());
        for (stored, original) in std::iter::once(full).chain(tail).zip(originals) {
            assert!(Arc::ptr_eq(stored, &original));
        }
    }
}

#[test]
fn worker_fixture_rejects_unqualified_lane_counts_before_native_construction() {
    for lanes in [0, 2, 127, 130, 256, 258, MAX_VALUES + 1, usize::MAX] {
        let attempts = super::super::prepared_tensor_map::native_attempts();
        assert!(matches!(
            prepared_rescale18_worker_fixture(lanes, domain()),
            Err(Error::Bounds)
        ));
        assert_eq!(
            super::super::prepared_tensor_map::native_attempts(),
            attempts
        );
    }
}

#[test]
fn prepared_publication_matches_closure_full_and_tail_and_remains_consumable() {
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

            // The SAME capability remains movable into the existing semantic
            // sink, while expanded nodes continue to own its original Arcs.
            let originals: Vec<_> = prepared.programs().cloned().collect();
            let mut semantic = body::layer::Builder::new(
                WeightMode::Public,
                &[body::Register {
                    lanes,
                    bounds: domain(),
                }],
            )
            .unwrap();
            let input = semantic.input(0).unwrap();
            let output = semantic
                .prepared_tensor_map(&input, prepared, new.allocated)
                .unwrap();
            let _plan = semantic.finish(&[output]).unwrap();
            for original in originals {
                assert!(Arc::ptr_eq(&new.tensors[&original.digest()], &original));
            }
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
fn prepared_publication_authenticates_owner_width_domain_and_caps() {
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
