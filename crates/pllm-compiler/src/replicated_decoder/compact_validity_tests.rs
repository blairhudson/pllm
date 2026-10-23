//! Local custody qualification; no whole-request/producer-export issuer.
use super::*;
use crate::replicated_decoder::program::original_import::OriginalImportedView;
use crate::replicated_decoder::{
    program::schedule_finalization as shared, source_registry::SourceRegistry,
};
use compact_admission::{RequestOwners, RequestWork};
use shared::BorrowedOrderedNodes;

struct Fixture {
    plan: pllm_models::DecoderPlan,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
    rows: usize,
    width: usize,
}

impl Fixture {
    fn new(rows: usize, width: usize) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type":"qwen2", "hidden_size":width, "intermediate_size":width,
            "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":width, "vocab_size":3, "tie_word_embeddings":false,
            "hidden_act":"silu", "attention_bias":false, "max_position_embeddings":64,
            "rope_theta":10000, "rms_norm_eps":0.000001, "attention_dropout":0.0,
            "use_sliding_window":false, "use_cache":true, "max_window_layers":1
        }))
        .unwrap();
        let mut plan = lower_with_weights(&config, rows, 1, WeightMode::Public).unwrap();
        let mut prefix: Vec<_> = plan
            .prefill
            .operations
            .iter()
            .filter(|op| op.operator == Op::RmsNorm)
            .cloned()
            .collect();
        assert_eq!(prefix.len(), 3);
        // Original operator IDs/gamma sources, on a declared immutable local
        // state input. Declaration, not a retrospectively blessed numeric View.
        prefix[0].inputs = vec!["state.local".into()];
        prefix[1].inputs = vec!["state.local".into()];
        prefix[2].inputs = vec![prefix[0].id.clone()];
        for op in &mut prefix {
            op.output_shape = vec![1, rows as u64, width as u64];
        }
        prefix.extend(
            plan.prefill
                .operations
                .iter()
                .filter(|op| op.operator != Op::RmsNorm)
                .cloned(),
        );
        plan.prefill.operations = prefix;
        plan.prefill.output = plan.prefill.operations[2].id.clone();
        plan.prefill.state_inputs.push(pllm_models::StateTensor {
            id: "state.local".into(),
            layer: None,
            kind: pllm_models::StateKind::Recurrent,
            shape: vec![1, rows as u64, width as u64],
            maximum_sequence: rows as u64,
        });
        let sources = Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public)
            .unwrap()
            .into_iter()
            .map(|(key, shape)| {
                let public = key.contains("norm");
                let data = vec![if public { ONE / 2 } else { 0 }; shape.iter().product()];
                let mut hash = Sha256::new();
                for value in &data {
                    hash.update(value.to_le_bytes());
                }
                let source = SourceTensor {
                    shape,
                    lower: data[0],
                    upper: data[0],
                    digest: hash.finalize().into(),
                    public_q18: public.then_some(data),
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                };
                (key, source)
            })
            .collect();
        Self {
            plan,
            sources,
            norms: SourceNorms::new(),
            rows,
            width,
        }
    }

    fn policy(&self) -> ExecutionPolicy {
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        }
    }

    fn validated(&self) -> ValidatedSourceDescriptors<'_> {
        let shapes = self
            .sources
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        validate_source_records(
            &self.plan.prefill,
            &shapes,
            &self.sources,
            WeightMode::Public,
            self.policy(),
            &self.norms,
        )
        .unwrap()
    }

    fn imports(&self) -> Builder {
        let mut b = Builder {
            public_weights: true,
            checked_bound: self.policy().checked_bound(),
            ..Builder::default()
        };
        for (key, source) in self
            .sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
        {
            b.push(
                Code::PublicInput {
                    key: key.clone(),
                    digest: source.digest,
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                source.shape.iter().product(),
                Interval::new(source.lower, source.upper).unwrap(),
            )
            .unwrap();
        }
        b
    }

    fn declare<'a>(
        &'a self,
        b: &mut Builder,
        registry: &'a SourceRegistry<'a>,
        prior: &RequestWork,
        owners: &RequestOwners<'_>,
    ) -> OriginalImportedView<'a> {
        b.declare_original_input(
            registry,
            &self.plan,
            &self.plan.prefill,
            "state.local",
            Interval::new(i64::MIN, i64::MAX).unwrap(),
            prior,
            owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap()
    }

    fn original_input(&self, b: &mut Builder) -> View {
        let owner = b
            .push(
                Code::Input("state.local".into()),
                self.rows * self.width,
                Interval::new(i64::MIN, i64::MAX).unwrap(),
            )
            .unwrap();
        b.view(owner, vec![1, self.rows, self.width])
    }
}

fn owners(values: &Vec<Option<Secret>>) -> RequestOwners<'_> {
    crate::replicated_decoder::program::original_import::tests::owners(values)
}

// Literal original composition, deliberately not the compact/shared RMS sink.
fn original_row_call(
    b: &mut Builder,
    fixture: &Fixture,
    op: &pllm_models::ModelOperation,
    input: &View,
) -> View {
    let width = fixture.width;
    let safe = b.checked_rows(&input.words, width).unwrap();
    let gamma = &fixture.sources[op.attributes["weight"].as_str().unwrap()];
    let mut words = Words::new();
    for row in safe.chunks_exact(width) {
        let owner = b.rms_norm_row(&row, width, gamma).unwrap();
        words.extend((0..width).map(|lane| (owner, lane)));
    }
    View {
        words,
        shape: input.shape.clone(),
    }
}

fn original_hash(nodes: &[Node]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for node in nodes {
        hash_node_header(&mut hash, node.lanes, node.bounds);
        match &node.code {
            Code::Gather(words) => hash_gather(&mut hash, words.iter()),
            Code::Tensor { input, program } => hash_tensor(&mut hash, *input, program),
            Code::CheckedRange { input, program } => {
                hash.update([9]);
                hash.update((*input as u64).to_le_bytes());
                hash.update(program.digest());
            }
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => hash_product(&mut hash, *left, *right, *shape, *rescale, norm.as_deref()),
            _ => panic!("unexpected original RMS record"),
        }
    }
    hash.finalize().into()
}

fn attempts() -> (usize, usize, usize) {
    (
        FACTORIES.with(|n| n.get()),
        STORAGE.with(|n| n.get()),
        RMS_NATIVE_ATTEMPTS.with(|n| n.get()),
    )
}

#[test]
fn original_multitile_tail_guard_all_rows_and_crosscall_sticky_match_ids_hash_work() {
    for (rows, width) in [(1, 2), (3, 2), (1, 130), (3, 130)] {
        let fixture = Fixture::new(rows, width);
        let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
        let values = Vec::new();
        let outside = owners(&values);
        let prior = RequestWork::default();
        let mut imported = fixture.imports();
        let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
        let mut compact = CompactGraph::with_bound_inputs(
            &imported,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let mut reference = fixture.imports();
        let original = fixture.original_input(&mut reference);
        let first = reference.nodes.len();
        let mut frontier = BTreeMap::from([("state.local".into(), original)]);
        let mut previous = None;
        for op in &fixture.plan.prefill.operations[..3] {
            let input = frontier[&op.inputs[0]].clone();
            let expected = original_row_call(&mut reference, &fixture, op, &input);
            let actual = compact.lower_next().unwrap();
            assert!(actual.words().eq(expected.words.iter()));
            assert_eq!(compact.next, reference.nodes.len());
            assert_eq!(compact.validity.current(), reference.validity);
            assert_eq!(
                compact.work_budget(),
                schedule_work(&reference.nodes[first..]).unwrap()
            );
            let mut hash = Sha256::new();
            compact.hash_expanded_nodes(&mut hash).unwrap();
            assert_eq!(
                <[u8; 32]>::from(hash.finalize()),
                original_hash(&reference.nodes[first..])
            );
            let PublishedKind::RmsNorm {
                checked: Some(checked),
                program,
                ..
            } = &compact.ops.last().unwrap().kind
            else {
                panic!("local custody")
            };
            assert_eq!(checked.incoming, previous);
            assert_eq!(checked.outgoing, reference.validity);
            let mut last_guard = None;
            checked
                .visit(&mut |event| {
                    last_guard = Some(event.owner);
                    Ok(())
                })
                .unwrap();
            if let Some(last) = last_guard {
                assert!(last < checked.rows_first());
            }
            for row in 0..rows {
                let owner = checked.rows_first() + 2 * row + 1;
                let Code::Tensor {
                    program: original, ..
                } = &reference.nodes[owner].code
                else {
                    panic!("original RMS")
                };
                assert_eq!(program.digest(), original.digest());
                let proof = checked
                    .row(
                        &imported,
                        &compact.ops.last().unwrap().input,
                        row * width,
                        width,
                    )
                    .unwrap();
                assert_eq!(
                    proof.word(width - 1).unwrap(),
                    checked.word((row + 1) * width - 1).unwrap()
                );
            }
            previous = checked.outgoing;
            frontier.insert(op.id.clone(), expected);
        }
    }
}

#[test]
fn local_guard_full_tail_arcs_reused_without_reset_and_frozen_owner_lives() {
    let fixture = Fixture::new(3, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    compact.lower_next().unwrap();
    let before = FACTORIES.with(|n| n.get());
    let first_sticky = compact.validity.current();
    let output = compact.lower_next().unwrap();
    assert_eq!(FACTORIES.with(|n| n.get()), before);
    let (
        PublishedKind::RmsNorm {
            checked: Some(first),
            ..
        },
        PublishedKind::RmsNorm {
            checked: Some(second),
            ..
        },
    ) = (&compact.ops[0].kind, &compact.ops[1].kind)
    else {
        panic!("local transitions")
    };
    assert_eq!(first.runs.len(), 2); // 128 * 3 + six-lane tail, not 390 entries
    assert_eq!(first.runs[0].tiles, 3);
    assert_eq!(first.runs[1].lanes, 6);
    assert_eq!(second.incoming, first_sticky);
    assert_ne!(second.outgoing, first_sticky);
    let mut reset = RequestValidity::original(&imported, compact.origin.clone());
    assert!(matches!(reset.commit(second), Err(Error::Binding)));
    let mut other_request = RequestValidity::original(&imported, Arc::new(()));
    assert!(matches!(other_request.commit(first), Err(Error::Binding)));
    for (a, b) in first.runs.iter().zip(&second.runs) {
        assert!(Arc::ptr_eq(&a.checked, &b.checked));
        assert!(Arc::ptr_eq(&a.all_valid, &b.all_valid));
    }
    let weak: Vec<_> = compact
        .owned_native_programs()
        .map(Arc::downgrade)
        .collect();
    let sticky = compact.validity.current();
    let exports = vec![output];
    let quote = compact.freeze_quote(&exports).unwrap();
    let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
    assert_eq!(frozen.local_validity_owner(), sticky);
    assert!(weak.iter().all(|program| program.upgrade().is_some()));
    let recipe_quote = frozen.schedule_recipe_quote().unwrap();
    let schedule = frozen.borrow_schedule(recipe_quote.peak_bytes).unwrap();
    let n = schedule.node(sticky.unwrap()).unwrap();
    assert!(matches!(
        n.recipe,
        shared::BorrowedRecipe::Product { rescale: false, .. }
    ));
    drop(schedule);
    drop(frozen);
    assert!(weak.iter().all(|program| program.upgrade().is_none()));
}

#[test]
fn exact_cap_and_one_short_reject_before_guard_storage_or_any_native_factory() {
    let fixture = Fixture::new(3, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let make = || {
        CompactGraph::with_bound_inputs(
            &imported,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES,
        )
        .unwrap()
    };
    let mut measured = make();
    measured.lower_next().unwrap();
    let exact = measured.construction_peak_bytes();
    let mut accepted = make();
    accepted.limit = exact;
    accepted.lower_next().unwrap();
    assert_eq!(accepted.construction_peak_bytes(), exact);
    let mut short = make();
    short.limit = exact - 1;
    let before = attempts();
    let next = short.next;
    assert!(matches!(short.lower_next(), Err(Error::Bounds)));
    assert_eq!(attempts(), before);
    assert_eq!(short.next, next);
    assert!(short.ops.is_empty());
    assert!(short.validity.current().is_none());
    assert!(short.failed);
    assert_eq!(short.contraction_control_bytes.get(), 0);
    assert!(matches!(short.lower_next(), Err(Error::Retired)));
}

#[test]
fn aggregate_work_overflow_rejects_before_guard_storage_and_factories() {
    let fixture = Fixture::new(1, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    // Existing ledger is intentionally saturated with repeated ACTUAL native
    // recipe work. This fixture does not turn work numbers into source authority.
    let mut workspace = tensor::RmsWorkWorkspace::empty();
    let work = tensor::Program::checked_range_work(
        128,
        Interval::new(i64::MIN, i64::MAX).unwrap(),
        Interval::new(-ONE, ONE).unwrap(),
        0,
        &mut workspace,
    )
    .unwrap();
    let mut low = 0u64;
    let mut high = u64::MAX / work.operations().max(1);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if work
            .checked_mul(mid)
            .is_ok_and(|work| work.limits().is_ok())
        {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    compact.work = work.checked_mul(low).unwrap();
    compact.work.limits().unwrap();
    let before = attempts();
    let original = compact.work;
    assert!(matches!(compact.lower_next(), Err(Error::Bounds)));
    assert_eq!(attempts(), before);
    assert_eq!(compact.work, original);
    assert!(compact.validity.current().is_none());
    assert!(compact.ops.is_empty());
}

#[test]
fn foreign_identical_source_owner_and_prefix_cannot_mint_local_safe_rows() {
    let fixture = Fixture::new(1, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let mut foreign = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let other = vec![fixture.declare(&mut foreign, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    compact.lower_next().unwrap();
    let PublishedKind::RmsNorm {
        checked: Some(rows),
        ..
    } = &compact.ops[0].kind
    else {
        panic!("safe rows")
    };
    let input = &compact.ops[0].input;
    assert!(matches!(
        rows.row(&foreign, input, 0, 130),
        Err(Error::Binding)
    ));
    let mut reordered = input.clone();
    reordered.start = 1;
    assert!(matches!(
        rows.row(&imported, &reordered, 0, 130),
        Err(Error::Binding)
    ));
    assert!(matches!(
        rows.row(&imported, input, 0, 65),
        Err(Error::Binding)
    ));
    let before = attempts();
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &imported,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &other,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    // Source-only prefix skipped, even with identical original input bytes/IDs.
    let prefix = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        1,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    );
    match prefix {
        Err(Error::Binding) => (),
        Ok(mut prefix) => assert!(matches!(prefix.lower_next(), Err(Error::Binding))),
        Err(_) => panic!("wrong cross-prefix rejection"),
    }
    assert_eq!(attempts(), before);
}

#[test]
fn shared_driver_gamma_error_wins_before_guard_and_no_sticky_reset() {
    let mut fixture = Fixture::new(1, 130);
    fixture.plan.prefill.operations[0].attributes["weight_offset"] = serde_json::json!(1);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let before = attempts();
    assert!(matches!(compact.lower_next(), Err(Error::Binding)));
    assert_eq!(attempts(), before);
    assert!(compact.validity.current().is_none());
}

#[test]
fn frozen_schedule_canonical_liveness_and_work_match_independent_original_builder() {
    let fixture = Fixture::new(3, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let mut original = fixture.imports();
    let input = fixture.original_input(&mut original);
    let mut frontier = BTreeMap::from([("state.local".into(), input)]);
    let mut output = None;
    for op in &fixture.plan.prefill.operations[..3] {
        let input = frontier[&op.inputs[0]].clone();
        let expected = original_row_call(&mut original, &fixture, op, &input);
        output = Some(compact.lower_next().unwrap());
        frontier.insert(op.id.clone(), expected);
    }
    let output = output.unwrap();
    let mut selected: Vec<_> = output.words().collect();
    selected.push((original.validity.unwrap(), 0));
    let exports = vec![output];
    let quote = compact.freeze_quote(&exports).unwrap();
    let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
    let quote = frozen.schedule_recipe_quote().unwrap();
    let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    let baseline = schedule.resources().finalization_baseline_bytes().unwrap();
    let actual = shared::quote(&schedule, &selected, baseline).unwrap();
    let expected = shared::quote(original.nodes.as_slice(), &selected, 0).unwrap();
    let actual = shared::finalize(
        &schedule,
        &selected,
        &Sha256::new(),
        baseline,
        actual.peak_bytes,
    )
    .unwrap();
    let expected = shared::finalize(
        original.nodes.as_slice(),
        &selected,
        &Sha256::new(),
        0,
        expected.peak_bytes,
    )
    .unwrap();
    assert_eq!(actual.digest, expected.digest);
    assert_eq!(actual.retire, expected.retire);
    assert_eq!(actual.work, expected.work);
    assert_eq!(actual.work, frozen.work_budget());
}

#[test]
fn native_rejected_lanes_are_zero_safe_and_later_success_cannot_clear_sticky() {
    use crate::replicated_decoder::tests::{execute, workers};
    let fixture = Fixture::new(1, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    compact.lower_next().unwrap();
    compact.lower_next().unwrap();
    let output = compact.lower_next().unwrap();
    let first = match &compact.ops[0].kind {
        PublishedKind::RmsNorm {
            checked: Some(rows),
            ..
        } => rows,
        _ => unreachable!(),
    };
    let safe: Vec<_> = (0..130).map(|i| first.word(i).unwrap()).collect();
    let mut selected = safe;
    selected.extend(output.words());
    selected.push((compact.validity.current().unwrap(), 0));
    let exports = vec![output];
    let quote = compact.freeze_quote(&exports).unwrap();
    let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
    let quote = frozen.schedule_recipe_quote().unwrap();
    let schedule = frozen.borrow_schedule(quote.peak_bytes).unwrap();
    // Test-only executable oracle. Clone captured Arc handles, never rebuild a
    // native factory or require ExpandedCompiled in the production constructor.
    let nodes: Vec<_> = (0..schedule.node_count())
        .map(|id| {
            let node = schedule.node(id).unwrap();
            let code = match node.recipe {
                shared::BorrowedRecipe::Input(key) => Code::Input(key.into()),
                shared::BorrowedRecipe::PublicInput { key, digest } => Code::PublicInput {
                    key: key.into(),
                    digest: *digest,
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                shared::BorrowedRecipe::Gather(words) => Code::Gather(
                    (0..words.len())
                        .map(|i| words.lane_ref(i).unwrap())
                        .collect(),
                ),
                shared::BorrowedRecipe::CheckedRange { input, program } => Code::CheckedRange {
                    input,
                    program: program.clone(),
                },
                shared::BorrowedRecipe::Tensor { input, program } => Code::Tensor {
                    input,
                    program: program.clone(),
                },
                shared::BorrowedRecipe::Product {
                    left,
                    right,
                    shape,
                    rescale,
                    norm: None,
                } => Code::Product {
                    left,
                    right,
                    shape,
                    rescale,
                    norm: None,
                },
                _ => panic!("fixture recipe outside original local RMS vocabulary"),
            };
            Node {
                code,
                lanes: node.lanes,
                bounds: node.bounds,
            }
        })
        .collect();
    let quote = shared::quote(&schedule, &selected, 0).unwrap();
    let finalized =
        shared::finalize(&schedule, &selected, &Sha256::new(), 0, quote.peak_bytes).unwrap();
    let program = Arc::new(Program {
        nodes,
        retire: finalized.retire,
        // Public source declarations precede the genuine state input. Preserve
        // the full original import prefix, not a count of secret inputs alone.
        inputs: imported.nodes.len(),
        outputs: selected,
        digest: finalized.digest,
        source_binding: frozen.source_binding(),
        payload_bytes: quote.peak_bytes,
        work: finalized.work,
    });
    assert_eq!(
        program.inputs().collect::<Vec<_>>(),
        vec![("state.local", 130)]
    );
    for rejected in [false, true] {
        let mut input = vec![0; 130];
        if rejected {
            input[0] = i64::MIN;
            input[129] = i64::MAX;
        }
        let mut clear: BTreeMap<_, _> = fixture
            .sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
            .map(|(key, source)| (key.clone(), vec![0; source.shape.iter().product()]))
            .collect();
        clear.insert("state.local".into(), input);
        let result = execute(workers(program.clone(), &clear));
        assert!(result[..260].iter().all(|value| *value == 0));
        assert_eq!(result[260], u64::from(!rejected));
    }
}

#[test]
fn row_proof_rejects_head_width_change_and_restores_exact_inverse_alias() {
    let mut fixture = Fixture::new(3, 130);
    let mut heads = fixture.plan.prefill.operations[0].clone();
    heads.id = "local.heads".into();
    heads.operator = Op::Reshape;
    heads.inputs = vec![fixture.plan.prefill.operations[0].id.clone()];
    heads.output_shape = vec![1, 2, 3, 65];
    heads.attributes = serde_json::json!({"layout":"batch_heads_sequence_feature"});
    let mut inverse = heads.clone();
    inverse.id = "local.inverse".into();
    inverse.inputs = vec![heads.id.clone()];
    inverse.output_shape = vec![1, 3, 130];
    inverse.attributes = serde_json::json!({"layout":"batch_sequence_hidden"});
    fixture.plan.prefill.operations.insert(1, heads);
    fixture.plan.prefill.operations.insert(2, inverse);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let mut compact = CompactGraph::with_bound_inputs(
        &imported,
        fixture.validated(),
        &registry,
        &fixture.plan,
        &fixture.plan.prefill,
        0,
        &bindings,
        &prior,
        &outside,
        MAX_PROGRAM_BYTES,
    )
    .unwrap();
    let original = compact.lower_next().unwrap();
    let energy = row_energy(&compact, &original, 0).unwrap().unwrap();
    let node_count = compact.next;
    let work = compact.work_budget();
    let sticky = compact.validity.current();
    let heads = compact.lower_next().unwrap();
    for row in 0..6 {
        assert!(row_energy(&compact, &heads, row).unwrap().is_none());
    }
    let inverse = compact.lower_next().unwrap();
    for row in 0..3 {
        assert_eq!(row_energy(&compact, &inverse, row).unwrap(), Some(energy));
    }
    assert!(inverse.words().eq(original.words()));
    assert_eq!(compact.next, node_count);
    assert_eq!(compact.work_budget(), work);
    assert_eq!(compact.validity.current(), sticky);
    assert!(
        compact
            .attention_energy(&inverse, 0, 130)
            .unwrap()
            .is_none()
    );
}

#[test]
fn known_input_norm_aggregate_work_rejects_before_compact_container_factory() {
    let fixture = Fixture::new(1, 130);
    let registry = SourceRegistry::from_validated(fixture.validated()).unwrap();
    let values = Vec::new();
    let mut outside = owners(&values);
    let prior = RequestWork::default();
    let mut imported = fixture.imports();
    let bindings = vec![fixture.declare(&mut imported, &registry, &prior, &outside)];
    let pending =
        pending_contraction_work(&imported, &fixture.plan.prefill, 0, &fixture.sources).unwrap();
    let mut workspace = tensor::RmsWorkWorkspace::empty();
    let unit = tensor::Program::checked_range_work(
        128,
        Interval::new(i64::MIN, i64::MAX).unwrap(),
        Interval::new(-ONE, ONE).unwrap(),
        0,
        &mut workspace,
    )
    .unwrap();
    let mut low = 0u64;
    let mut high = u64::MAX / unit.operations().max(1);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if unit
            .checked_mul(mid)
            .and_then(|work| work.checked_add(pending))
            .is_ok_and(|work| work.limits().is_ok())
        {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    outside.other_work = unit.checked_mul(low).unwrap();
    outside
        .other_work
        .checked_add(pending)
        .unwrap()
        .limits()
        .unwrap();
    let before = attempts();
    let containers = input_owner::CONTAINER_FACTORIES.with(|n| n.get());
    assert!(matches!(
        CompactGraph::with_bound_inputs(
            &imported,
            fixture.validated(),
            &registry,
            &fixture.plan,
            &fixture.plan.prefill,
            0,
            &bindings,
            &prior,
            &outside,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Bounds)
    ));
    assert_eq!(attempts(), before);
    assert_eq!(
        input_owner::CONTAINER_FACTORIES.with(|n| n.get()),
        containers
    );
}
