//! Regression definitions only: original primitive parity and bounded publication,
//! not runtime/full-request admission or production promotion.
use super::*;

thread_local! {
    static LOWER_NEXT_FAULT: std::cell::Cell<Option<(bool, bool)>> = const { std::cell::Cell::new(None) };
    static WALK_HEADER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static WALK_DESCRIPTOR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn pending_walk_header() { WALK_HEADER.with(|n| n.set(n.get() + 1)); }
pub(super) fn pending_walk_descriptor() { WALK_DESCRIPTOR.with(|n| n.set(n.get() + 1)); }

#[test]
fn pending_walk_prices_full_extents_unowned_children_and_repeated_censuses() {
    // Cardinality arithmetic only; these values confer no funding or source
    // identity. T remains nine even when live GuardRuns coalesce to one.
    let counts = PendingWalkCounts { ops: 7, tickets: 9, specs: 4, rms: 1,
        rotary: 3, scale: 2, softmax: 4, runs: 1, argmax: 6 };
    assert_eq!(counts.positions().unwrap(), 6 * 9 + 3 + 3 * 4);
    assert_eq!(counts.children().unwrap(), 5 * 7 + 3 + 2 + 4 + 3 + 6);
    let search = 64 * (1 + 8 * 7 + counts.children().unwrap());
    let local = 64 * (1 + 9 + 2 * 4 + 2 * 7 + counts.positions().unwrap());
    assert_eq!(counts.body_visits().unwrap() as usize, local + counts.positions().unwrap() * search);
    let repeated = (0..12).try_fold(0u64, |sum, _| sum.checked_add(counts.header_visits().unwrap())
        .and_then(|n| n.checked_add(counts.body_visits().unwrap()))).unwrap();
    assert_eq!(repeated, 12 * (counts.header_visits().unwrap() + counts.body_visits().unwrap()));
    assert!(PendingWalkCounts { ops: usize::MAX, ..PendingWalkCounts::default() }.header_visits().is_err());
    assert!(PendingWalkCounts { tickets: usize::MAX, ..PendingWalkCounts::default() }.body_visits().is_err());
    // Larger allocated extent increases price even when no additional program
    // owns flag becomes true. Unowned argument candidates still affect K.
    let larger = PendingWalkCounts { tickets: 10, argmax: 8, ..counts };
    assert!(larger.body_visits().unwrap() > counts.body_visits().unwrap());
}

#[test]
fn native_owner_search_counts_eight_descriptor_passes_on_miss_and_bounded_hit() {
    // Existing non-request fixture: iterator mechanics, NOT authentic pending
    // custody/funding proof. Genuine-request downstream definitions stay ignored.
    let fixture = Fixture::new(4, 1, false, false);
    let mut graph = fixture.graph(0);
    graph.lower_next().unwrap();
    let target = Arc::as_ptr(graph.owned_native_programs().last().unwrap());
    let descriptors = graph.ops.len();
    WALK_DESCRIPTOR.with(|n| n.set(0));
    assert!(!graph.owned_native_programs().any(|p| std::ptr::eq(p.as_ref(), std::ptr::null())));
    WALK_DESCRIPTOR.with(|n| assert_eq!(n.get(), 8 * descriptors));
    WALK_DESCRIPTOR.with(|n| n.set(0));
    assert!(graph.owned_native_programs().any(|p| std::ptr::eq(p.as_ref(), target)));
    WALK_DESCRIPTOR.with(|n| assert!(n.get() > 0 && n.get() <= 8 * descriptors));
    for _ in 0..12 {
        WALK_DESCRIPTOR.with(|n| n.set(0));
        assert!(!graph.owned_native_programs().any(|p| std::ptr::eq(p.as_ref(), std::ptr::null())));
        WALK_DESCRIPTOR.with(|n| assert_eq!(n.get(), 8 * descriptors));
    }
}

#[test]
fn compiler_owner_vectors_count_attempts_cleanup_and_no_native_arc_double_charge() {
    let rms = compact_validity::request_rms::compiler_events(9, 36).unwrap();
    assert_eq!(rms.allocation_attempts, 2 * 9 + 4);
    assert!(rms.cleanup_dispatches > 0);
    let rotary = compact_rotary::compiler_events(4, 36).unwrap();
    assert_eq!(rotary.allocation_attempts, 4 * 3 + 3);
    assert!(compact_rotary::compiler_events(5, 36).unwrap().total().unwrap() > rotary.total().unwrap());
    assert!(compact_rotary::compiler_events(usize::MAX, usize::MAX).is_err());
    assert!(compact_validity::request_rms::compiler_events(usize::MAX, usize::MAX).is_err());
}

// Ordinary fault-injection tests at the live outer boundary. No synthetic source
// token/ledger/funding witness; non-request fixtures assert graph lifecycle only.
pub(super) fn lower_next_checkpoint(retirement: bool) -> Result<(), Error> {
    LOWER_NEXT_FAULT.with(|fault| {
        if let Some((wanted, unwind)) = fault.get() {
            if wanted == retirement {
                fault.set(None);
                if unwind { panic!("test outer compact lowering unwind"); }
                return Err(Error::Bounds);
            }
        }
        Ok(())
    })
}

#[test]
fn compact_norm_outer_error_and_unwind_prohibit_graph_reuse_including_retirement() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    for repeat in [false, true] {
        for retirement in [false, true] {
            for unwind in [false, true] {
                let fixture = Fixture::new(4, 1, false, repeat);
                let mut graph = fixture.graph(0);
                LOWER_NEXT_FAULT.with(|fault| assert!(fault.replace(Some((retirement, unwind))).is_none()));
                let result = catch_unwind(AssertUnwindSafe(|| graph.lower_next()));
                if unwind {
                    assert!(result.is_err());
                    assert!(graph.failure_site().is_none());
                } else {
                    assert!(matches!(result.unwrap(), Err(Error::Bounds)));
                    let site = graph.failure_site().unwrap();
                    assert_eq!(site.step, 0);
                    assert_eq!(site.operation, "final_norm");
                    assert_eq!(site.operator, fixture.region.operations[0].operator);
                    assert_eq!(site.next_virtual_owner, graph.next);
                }
                assert!(graph.failed);
                assert_eq!(graph.ops.len(), usize::from(retirement));
                assert_eq!(graph.cursor, usize::from(retirement));
                let cursor = graph.cursor;
                let work = graph.work_budget();
                let programs = graph.ops.len();
                let failure = graph.failure_site().map(|site| (
                    site.step, site.operation, site.operator, site.input_owners,
                    site.source_owner, site.next_virtual_owner,
                ));
                assert!(matches!(graph.lower_next(), Err(Error::Retired)));
                assert_eq!(graph.cursor, cursor);
                assert_eq!(graph.ops.len(), programs);
                assert_eq!(graph.work_budget(), work);
                assert!(graph.failed);
                assert_eq!(graph.failure_site().map(|site| (
                    site.step, site.operation, site.operator, site.input_owners,
                    site.source_owner, site.next_virtual_owner,
                )), failure);
                LOWER_NEXT_FAULT.with(|fault| assert!(fault.get().is_none()));
            }
        }
    }
}

#[test]
fn compact_norm_exhausted_sequence_has_no_failure_site() {
    let fixture = Fixture::new(4, 1, false, false);
    let mut graph = fixture.graph(0);
    graph.lower_next().unwrap();
    assert_eq!(graph.cursor, 1);
    assert!(graph.lower_next().is_err());
    assert!(graph.failed);
    assert!(graph.failure_site().is_none());
    assert!(matches!(graph.lower_next(), Err(Error::Retired)));
    assert!(graph.failure_site().is_none());
}

struct Fixture {
    full: DecoderGraph,
    region: DecoderGraph,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
    imports: Builder,
    input: View,
    execution: ExecutionPolicy,
}

impl Fixture {
    fn new(width: usize, rows: usize, checked: bool, repeat: bool) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type": "llama", "hidden_size": width, "intermediate_size": 2,
            "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": width, "vocab_size": 3, "tie_word_embeddings": false,
            "hidden_act": "silu", "attention_bias": false, "max_position_embeddings": 64,
            "rope_theta": 10000, "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
            "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
        }))
        .unwrap();
        let full = lower_with_weights(&config, rows, 1, WeightMode::Public)
            .unwrap()
            .prefill;
        let operation = full
            .operations
            .iter()
            .find(|op| op.id == "final_norm")
            .unwrap()
            .clone();
        let mut region = full.clone();
        region.operations = vec![operation.clone()];
        if repeat {
            let mut second = operation;
            second.id.push_str(".same_input");
            region.operations.push(second);
        }
        region.output = region.operations.last().unwrap().id.clone();
        let sources: BTreeMap<_, _> =
            Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public)
                .unwrap()
                .into_iter()
                .map(|(key, shape)| {
                    let public = key.contains("norm");
                    let values: Vec<i64> = (0..shape.iter().product())
                        .map(|i| {
                            if public {
                                if i % 2 == 0 { ONE } else { ONE / 2 }
                            } else {
                                ONE / 64
                            }
                        })
                        .collect();
                    let mut hash = Sha256::new();
                    for value in &values {
                        hash.update(value.to_le_bytes());
                    }
                    (
                        key,
                        SourceTensor {
                            shape,
                            lower: if public { ONE / 2 } else { -ONE / 64 },
                            upper: if public { ONE } else { ONE / 64 },
                            public_q18: public.then_some(values),
                            digest: hash.finalize().into(),
                            row_l1: None,
                            row_squared_norm: None,
                            row_l1_squared_sum: None,
                        },
                    )
                })
                .collect();
        let execution = ExecutionPolicy {
            range_policy: if checked {
                RangePolicy::Checked
            } else {
                RangePolicy::Static
            },
            max_abs_activation: if checked {
                1
            } else {
                policy::DEFAULT_MAX_ABS_ACTIVATION
            },
            ..ExecutionPolicy::default()
        };
        let mut imports = Builder {
            public_weights: true,
            checked_bound: execution.checked_bound(),
            ..Builder::default()
        };
        for (key, source) in sources
            .iter()
            .filter(|(_, source)| source.public_q18.is_none())
        {
            imports
                .push(
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
        let owner = imports
            .push(
                Code::Input("hidden".into()),
                width * rows,
                Interval::new(-ONE / 4, ONE / 4).unwrap(),
            )
            .unwrap();
        let input = imports.view(owner, vec![1, rows, width]);
        Self {
            full,
            region,
            sources,
            norms: SourceNorms::new(),
            imports,
            input,
            execution,
        }
    }

    fn graph(&self, outside: usize) -> CompactGraph<'_, '_> {
        let shapes = self
            .sources
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        let validated = validate_source_records(
            &self.full,
            &shapes,
            &self.sources,
            WeightMode::Public,
            self.execution,
            &self.norms,
        )
        .unwrap();
        let mut graph =
            CompactGraph::with_owner_baseline(&self.imports, validated, &self.region, 0, outside)
                .unwrap();
        graph
            .bind_import(&self.region.operations[0].inputs[0], &self.input)
            .unwrap();
        graph
    }
}

fn hash_original(nodes: &[Node]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for node in nodes {
        hash_node_header(&mut hash, node.lanes, node.bounds);
        match &node.code {
            Code::Gather(words) => hash_gather(&mut hash, words.iter()),
            Code::Tensor { input, program } => hash_tensor(&mut hash, *input, program),
            _ => panic!("not an original RMSNorm row"),
        }
    }
    hash.finalize().into()
}

#[test]
fn shared_driver_compact_norm_preserves_original_rows_encoding_work_gamma_and_arcs() {
    for (width, rows) in [(2, 1), (4, 3), (130, 2)] {
        let fixture = Fixture::new(width, rows, false, true);
        let original_fixture = Fixture::new(width, rows, false, true);
        let mut expanded = original_fixture.imports;
        let first = expanded.nodes.len();
        let mut graph = fixture.graph(37);
        let baseline = graph.baseline().unwrap();
        let gamma = &fixture.sources[fixture.region.operations[0].attributes["weight"]
            .as_str()
            .unwrap()];
        let mut expected = Vec::new();
        for _ in &fixture.region.operations {
            // Independent pre-driver composition. No compact sink/driver oracle.
            let mut words = Words::new();
            for row in original_fixture.input.words.chunks_exact(width) {
                let owner = expanded.rms_norm_row(&row, width, gamma).unwrap();
                words.extend((0..width).map(|lane| (owner, lane)));
            }
            let output = graph.lower_next().unwrap();
            assert!(output.words().eq(words.iter()));
            assert_eq!(output.bounds, expanded.bounds(&words).unwrap());
            assert_eq!(
                graph.row_norm(&output).unwrap(),
                expanded.row_norm(&words, width)
            );
            expected.extend(words.iter());
        }
        let census: Vec<_> = graph.native_row_census().collect();
        assert_eq!(census.len(), rows * 2);
        for (row, native) in census.iter().enumerate() {
            assert_eq!(native.owner, first + 2 * row + 1);
            assert_eq!(native.offset, 0);
            assert_eq!(native.width, width);
            assert!(std::ptr::eq(native.gamma, gamma));
            assert!(Arc::ptr_eq(native.program, census[0].program));
            assert_eq!(native.source_binding, graph.binding);
            assert_eq!(expected[row * width], (native.owner, 0));
        }
        assert!(matches!(
            graph.ops[0].kind,
            PublishedKind::RmsNorm {
                owns_program: true,
                input_energy: None,
                ..
            }
        ));
        assert!(matches!(
            graph.ops[1].kind,
            PublishedKind::RmsNorm {
                owns_program: false,
                ..
            }
        ));
        assert_eq!(graph.owned_native_programs().count(), 1);
        assert_eq!(
            graph.baseline().unwrap() - baseline,
            census[0]
                .program
                .resources()
                .unwrap()
                .immutable_template_bytes
        );
        assert!(fixture.imports.validity.is_none());
        assert!(fixture.imports.row_energies.is_empty());
        assert!(fixture.imports.attention_energies.is_empty());
        let mut hash = Sha256::new();
        graph.hash_expanded_nodes(&mut hash).unwrap();
        assert_eq!(
            <[u8; 32]>::from(hash.finalize()),
            hash_original(&expanded.nodes[first..])
        );
        assert_eq!(
            graph.work_budget(),
            schedule_work(&expanded.nodes[first..]).unwrap()
        );
    }
}

#[test]
fn compact_norm_exact_limit_and_one_short_reject_before_native_factory() {
    let fixture = Fixture::new(4, 3, false, false);
    let mut quote = fixture.graph(71);
    quote.lower_next().unwrap();
    let peak = quote.construction_peak_bytes();
    let mut exact = fixture.graph(71);
    exact.limit = peak;
    exact.lower_next().unwrap();
    assert_eq!(exact.construction_peak_bytes(), peak);
    let mut short = fixture.graph(71);
    short.limit = peak - 1;
    let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(short.lower_next(), Err(Error::Bounds)));
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    assert!(short.ops.is_empty());
    assert_eq!(short.next, fixture.imports.nodes.len());
    assert_eq!(short.work_budget(), WorkBudget::default());
    assert_eq!(short.failure_site().unwrap().operation, "final_norm");
    assert!(matches!(short.lower_next(), Err(Error::Retired)));
}

#[test]
fn compact_norm_retained_arc_overlap_is_not_free_on_reuse() {
    let fixture = Fixture::new(4, 2, false, true);
    let mut graph = fixture.graph(83);
    graph.lower_next().unwrap();
    let program = graph.owned_native_programs().next().unwrap().clone();
    let before = graph.baseline().unwrap();
    graph.limit = 83 + before;
    let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(graph.lower_next(), Err(Error::Bounds)));
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    assert_eq!(graph.ops.len(), 1);
    assert!(Arc::ptr_eq(
        graph.owned_native_programs().next().unwrap(),
        &program
    ));
}

#[test]
fn compact_norm_existing_native_row_energy_is_the_only_refinement_for_later_norm() {
    let mut fixture = Fixture::new(4, 3, false, true);
    fixture.region.operations[1].inputs = vec![fixture.region.operations[0].id.clone()];
    let original_fixture = Fixture::new(4, 3, false, false);
    let mut expanded = original_fixture.imports;
    let mut words = original_fixture.input.words;
    let first = expanded.nodes.len();
    let mut graph = fixture.graph(0);
    let gamma = &fixture.sources[fixture.region.operations[0].attributes["weight"]
        .as_str()
        .unwrap()];
    let mut prior_energy = None;
    for op in &fixture.region.operations {
        let mut output = Words::new();
        for row in words.chunks_exact(4) {
            assert_eq!(expanded.row_norm(&row, 4), prior_energy);
            let owner = expanded.rms_norm_row(&row, 4, gamma).unwrap();
            output.extend((0..4).map(|lane| (owner, lane)));
        }
        let compact = graph.lower_next().unwrap();
        assert!(compact.words().eq(output.iter()));
        let PublishedKind::RmsNorm {
            input_energy,
            output_energy,
            ..
        } = &graph.ops.last().unwrap().kind
        else {
            panic!("native RMSNorm")
        };
        assert_eq!(*input_energy, prior_energy);
        prior_energy = Some(*output_energy);
        // Native row custody is not a source projection/attention proof.
        assert!(graph.complete_row(&compact, 0).unwrap().is_none());
        assert!(graph.attention_energy(&compact, 0, 4).unwrap().is_none());
        assert_eq!(
            graph.ops.last().unwrap().output.lanes,
            op.output_shape.iter().product::<u64>() as usize
        );
        words = output;
    }
    assert!(prior_energy.is_some());
    let mut hash = Sha256::new();
    graph.hash_expanded_nodes(&mut hash).unwrap();
    assert_eq!(
        <[u8; 32]>::from(hash.finalize()),
        hash_original(&expanded.nodes[first..])
    );
    assert_eq!(
        graph.work_budget(),
        schedule_work(&expanded.nodes[first..]).unwrap()
    );
}

#[test]
fn compact_norm_alias_order_rejects_transpose_and_accepts_inverse() {
    let mut fixture = Fixture::new(4, 3, false, false);
    let mut transpose = fixture.region.operations[0].clone();
    transpose.id = "final_norm.heads".into();
    transpose.operator = Op::Reshape;
    transpose.layer = None;
    transpose.state_kind = None;
    transpose.inputs = vec![fixture.region.operations[0].id.clone()];
    transpose.output_shape = vec![1, 2, 3, 2];
    transpose.attributes = serde_json::json!({"layout": "batch_heads_sequence_feature"});
    let mut inverse = transpose.clone();
    inverse.id = "final_norm.inverse".into();
    inverse.inputs = vec![transpose.id.clone()];
    inverse.output_shape = vec![1, 3, 4];
    inverse.attributes = serde_json::json!({"layout": "batch_sequence_hidden"});
    fixture.region.output = inverse.id.clone();
    fixture.region.operations.extend([transpose, inverse]);

    let mut graph = fixture.graph(0);
    let rms = graph.lower_next().unwrap();
    let PublishedKind::RmsNorm { output_energy, .. } = &graph.ops[0].kind else {
        panic!("native RMSNorm")
    };
    let energy = *output_energy;
    assert_eq!(rms.lanes(), 12);
    assert_eq!(graph.row_norm(&rms).unwrap(), Some(energy));
    // Native RMS energy is retained, not a source ProjectionRow certificate.
    assert!(graph.complete_row(&rms, 0).unwrap().is_none());
    let next = graph.next;
    let work = graph.work_budget();

    let transpose = graph.lower_next().unwrap();
    assert_eq!(transpose.lanes(), 12);
    // Independent [sequence, head, feature] -> [head, sequence, feature]
    // permutation: hidden width four is two heads of two features each.
    let expected = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11];
    for (index, original) in expected.into_iter().enumerate() {
        assert_eq!(transpose.word(index).unwrap(), rms.word(original).unwrap());
    }
    assert_eq!(transpose.word(2).unwrap(), rms.word(4).unwrap());
    assert_ne!(transpose.word(2).unwrap(), rms.word(2).unwrap());
    assert!(graph.alias_range(&transpose, 0, 12).unwrap().is_none());
    assert_eq!(graph.row_norm(&transpose).unwrap(), None);

    let inverse = graph.lower_next().unwrap();
    assert_eq!(inverse.lanes(), 12);
    for index in 0..12 {
        assert_eq!(inverse.word(index).unwrap(), rms.word(index).unwrap());
    }
    let (original, offset) = graph.alias_range(&inverse, 0, 12).unwrap().unwrap();
    assert_eq!(original.slot, rms.slot);
    assert_eq!(original.slot, Some(0));
    assert_eq!(offset, 0);
    assert_eq!(graph.row_norm(&inverse).unwrap(), Some(energy));
    // Unchanged interval metadata alone does not establish ordered row custody.
    // These are symbolic owner/lane identities, not evaluated numeric values.
    assert_eq!(transpose.bounds, rms.bounds);
    assert_eq!(inverse.bounds, rms.bounds);
    assert_eq!(graph.next, next);
    assert_eq!(graph.work_budget(), work);
}

#[test]
fn compact_norm_quote_retains_original_gamma_capacity_not_only_logical_slice() {
    let fixture = Fixture::new(4, 2, false, false);
    let mut original = fixture.graph(29);
    original.lower_next().unwrap();
    let peak = original.construction_peak_bytes();
    let mut padded = Fixture::new(4, 2, false, false);
    let key = padded.region.operations[0].attributes["weight"]
        .as_str()
        .unwrap();
    let gamma = padded.sources.get_mut(key).unwrap();
    let old = gamma.shape.capacity() * std::mem::size_of::<usize>()
        + gamma.public_q18.as_ref().unwrap().capacity() * std::mem::size_of::<i64>();
    gamma.shape.reserve_exact(17);
    gamma.public_q18.as_mut().unwrap().reserve_exact(61);
    let extra = gamma.shape.capacity() * std::mem::size_of::<usize>()
        + gamma.public_q18.as_ref().unwrap().capacity() * std::mem::size_of::<i64>()
        - old;
    let mut graph = padded.graph(29);
    graph.lower_next().unwrap();
    assert_eq!(graph.construction_peak_bytes(), peak + extra);
    assert_eq!(
        graph.construction_peak_without_source.get(),
        original.construction_peak_without_source.get()
    );
}

#[test]
fn compact_norm_rejects_unmapped_checked_custody_even_for_contained_input() {
    let fixture = Fixture::new(4, 2, true, false);
    let mut graph = fixture.graph(0);
    assert!(
        fixture.imports.nodes.last().unwrap().bounds.upper()
            < fixture.imports.checked_bound.unwrap()
    );
    let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
    assert!(matches!(graph.lower_next(), Err(Error::Binding)));
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    assert!(graph.native_row_census().next().is_none());
    assert!(graph.ops.is_empty());
    assert!(fixture.imports.validity.is_none());
}

#[test]
fn compact_norm_foreign_and_mutated_coordinate_provenance_fail_before_factory() {
    let fixture = Fixture::new(4, 2, false, false);
    let mut foreign = fixture.graph(0);
    let foreign_view = foreign
        .frontier
        .inputs(&fixture.region.operations[0])
        .0
        .unwrap()
        .clone();
    // Same source digest and import ordinals do not grant another graph origin.
    foreign.lower_next().unwrap();
    let original = fixture.graph(0);
    let view = original
        .frontier
        .inputs(&fixture.region.operations[0])
        .0
        .unwrap()
        .clone();
    let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
    for mutation in 0..5 {
        let mut graph = fixture.graph(0);
        let mut changed = view.clone();
        changed.origin = graph.origin.clone();
        match mutation {
            0 => changed = foreign_view.clone(),
            1 => changed.start = 1,
            2 => changed.mapping = Mapping::Dense(0),
            3 => changed.bounds = Interval::new(-1, 1).unwrap(),
            _ => {
                changed.order = LaneOrder::HeadsSequence {
                    heads: 2,
                    rows: 1,
                    width: 4,
                    to_heads: true,
                    base_start: 0,
                }
            }
        }
        graph
            .frontier
            .bind(fixture.region.operations[0].inputs[0].clone(), changed);
        assert!(matches!(graph.lower_next(), Err(Error::Binding)));
        assert!(graph.ops.is_empty());
        assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    }
}

#[test]
fn compact_norm_bad_gamma_offset_and_shape_never_reach_native_factory() {
    for mutation in 0..3 {
        let mut fixture = Fixture::new(4, 2, false, false);
        match mutation {
            0 => fixture.region.operations[0].attributes["weight_offset"] = serde_json::json!(1),
            1 => fixture.region.operations[0].attributes["weight"] = serde_json::json!("not.gamma"),
            _ => fixture.region.operations[0].output_shape = vec![1, 1, 8],
        }
        let mut graph = fixture.graph(0);
        let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
        assert!(graph.lower_next().is_err());
        assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
        assert!(graph.ops.is_empty());
    }
}

#[test]
fn compact_norm_imported_general_or_attention_ledgers_do_not_grant_row_custody() {
    for attention in [false, true] {
        let mut fixture = Fixture::new(4, 2, false, false);
        let row = fixture.input.words.slice(..4);
        if attention {
            fixture
                .imports
                .record_attention_energy(&row, (ONE as u128).pow(2))
                .unwrap();
        } else {
            fixture
                .imports
                .record_row_energy(&row, (ONE as u128).pow(2))
                .unwrap();
        }
        let shapes = fixture
            .sources
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        let validated = validate_source_records(
            &fixture.full,
            &shapes,
            &fixture.sources,
            WeightMode::Public,
            fixture.execution,
            &fixture.norms,
        )
        .unwrap();
        let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
        assert!(matches!(
            CompactGraph::new(&fixture.imports, validated, &fixture.region, 0),
            Err(Error::Binding)
        ));
        assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    }
}

#[test]
fn compact_norm_changed_gamma_bytes_cannot_reuse_original_commitment() {
    let mut fixture = Fixture::new(4, 2, false, false);
    let key = fixture.region.operations[0].attributes["weight"]
        .as_str()
        .unwrap();
    fixture
        .sources
        .get_mut(key)
        .unwrap()
        .public_q18
        .as_mut()
        .unwrap()[0] = ONE / 2;
    let shapes = fixture
        .sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    assert!(
        validate_source_records(
            &fixture.full,
            &shapes,
            &fixture.sources,
            WeightMode::Public,
            fixture.execution,
            &fixture.norms,
        )
        .is_err()
    );
}

#[test]
fn compact_checked_norm_rejection_cannot_replace_original_multitile_sticky_custody() {
    // Two native tiles, including a partial tail. The oracle is the original
    // Builder composition, not compact lowering or a body-family substitute.
    let width = 4;
    let rows = tensor::MAX_TILE_LANES / width + 1;
    let mut fixture = Fixture::new(width, rows, true, false);
    let input_owner = fixture.input.words.at(0).0;
    fixture.imports.nodes[input_owner].bounds = Interval::new(-2 * ONE, 2 * ONE).unwrap();
    let mut original = Fixture::new(width, rows, true, false);
    original.imports.nodes[input_owner].bounds = Interval::new(-2 * ONE, 2 * ONE).unwrap();
    let mut expanded = original.imports;
    let gamma = &fixture.sources[fixture.region.operations[0].attributes["weight"]
        .as_str()
        .unwrap()];
    let mut previous = None;
    let mut guard_programs: Vec<Arc<tensor::Program>> = Vec::new();
    let mut validity_programs: Vec<Arc<tensor::Program>> = Vec::new();
    let mut first_sticky = None;
    for pass in 0..2 {
        let first = expanded.nodes.len();
        let safe = expanded.checked_rows(&original.input.words, width).unwrap();
        let guard_end = expanded.nodes.len();
        let mut next = first;
        for (tile, start) in (0..original.input.words.len())
            .step_by(tensor::MAX_TILE_LANES)
            .enumerate()
        {
            let lanes = (original.input.words.len() - start).min(tensor::MAX_TILE_LANES);
            let Code::Gather(input) = &expanded.nodes[next].code else {
                panic!("original checked input gather")
            };
            assert!(
                input
                    .iter()
                    .eq(original.input.words.slice(start..start + lanes).iter())
            );
            let Code::CheckedRange { input, program } = &expanded.nodes[next + 1].code else {
                panic!("original checked dual-output producer")
            };
            assert_eq!(*input, next);
            assert_eq!(expanded.nodes[next + 1].lanes, lanes * 2);
            assert_eq!(
                expanded.nodes[next + 1].bounds,
                Interval::new(-ONE, ONE).unwrap()
            );
            if pass == 0 {
                guard_programs.push(program.clone());
            } else {
                assert!(Arc::ptr_eq(program, &guard_programs[tile]));
            }
            for (offset, base) in [(2, 0), (3, lanes)] {
                let Code::Gather(words) = &expanded.nodes[next + offset].code else {
                    panic!("disjoint original safe/validity gather")
                };
                assert!(
                    words
                        .iter()
                        .eq((base..base + lanes).map(|lane| (next + 1, lane)))
                );
            }
            assert!(
                safe.slice(start..start + lanes)
                    .iter()
                    .eq((0..lanes).map(|lane| (next + 2, lane)))
            );
            let valid = next + 4;
            let Code::Tensor { input, program } = &expanded.nodes[valid].code else {
                panic!("original all-valid owner")
            };
            assert_eq!(*input, next + 3);
            if pass == 0 {
                validity_programs.push(program.clone());
            } else {
                assert!(Arc::ptr_eq(program, &validity_programs[tile]));
            }
            next += 5;
            previous = Some(if let Some(prior) = previous {
                let Code::Product {
                    left,
                    right,
                    shape,
                    rescale,
                    norm,
                } = &expanded.nodes[next].code
                else {
                    panic!("original request-wide sticky product")
                };
                assert_eq!((*left, *right), (prior, valid));
                assert!(shape.is_none() && !*rescale && norm.is_none());
                next += 1;
                next - 1
            } else {
                valid
            });
        }
        assert_eq!(next, guard_end);
        assert_eq!(expanded.validity, previous);
        if pass == 0 {
            first_sticky = previous;
        } else {
            assert_ne!(previous, first_sticky);
        }
        // General energy is absent on these raw inputs. Checked bounds and
        // validity alone must not manufacture a producer-row certificate.
        assert_eq!(expanded.row_norm(&safe, width), None);
        assert_eq!(expanded.attention_energy(&safe, width), None);
        for (row, words) in safe.chunks_exact(width).enumerate() {
            let owner = expanded.rms_norm_row(&words, width, gamma).unwrap();
            assert_eq!(owner, guard_end + row * 2 + 1);
        }
        assert!(
            expanded.nodes[guard_end..]
                .iter()
                .all(|node| !matches!(node.code, Code::CheckedRange { .. }))
        );
    }

    // Compact has neither an authenticated incoming sticky owner nor an
    // unchanged outgoing transition/export mapping. Reject before any RMS
    // factory rather than representing the two real guards as unchecked rows.
    let mut compact = fixture.graph(43);
    let attempts = RMS_NATIVE_ATTEMPTS.with(|count| count.get());
    let first = compact.next;
    assert!(matches!(compact.lower_next(), Err(Error::Binding)));
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
    assert_eq!(compact.next, first);
    assert!(compact.ops.is_empty());
    assert_eq!(compact.work_budget(), WorkBudget::default());
    assert!(matches!(compact.lower_next(), Err(Error::Retired)));

    // The actual original guard-bearing Builder cannot be re-imported as an
    // imports-only region. Resetting its validity field would not remove the
    // arithmetic/proof/source obligations and is not an integration seam.
    let shapes = fixture
        .sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    let validated = validate_source_records(
        &fixture.full,
        &shapes,
        &fixture.sources,
        WeightMode::Public,
        fixture.execution,
        &fixture.norms,
    )
    .unwrap();
    assert!(expanded.validity.is_some());
    assert!(matches!(
        CompactGraph::new(&expanded, validated, &fixture.region, 0),
        Err(Error::Binding)
    ));
    assert_eq!(RMS_NATIVE_ATTEMPTS.with(|count| count.get()), attempts);
}
