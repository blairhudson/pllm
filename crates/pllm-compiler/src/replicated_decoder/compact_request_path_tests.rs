//! Connected supported-region qualification, NOT a full decoder or runtime test.
//!
//! Mount as a child of `program::compact_lowering::compact_admission`: original
//! Builder/CompactGraph and RequestWork fields intentionally remain private to
//! their owners. No production visibility changes are needed for these tests.
use super::super::*;
use super::*;
use crate::replicated_decoder::source_registry::SourceRegistry;
use std::sync::Weak;

const TOKENS: usize = 3;
const WIDTH: usize = 4;
const VOCABULARY: usize = 3;
const CONFIG: &[u8] = br#"{"model_type":"llama","hidden_size":4,"intermediate_size":4,
    "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
    "head_dim":2,"vocab_size":3,"tie_word_embeddings":false,"hidden_act":"silu",
    "attention_bias":false,"max_position_embeddings":64,"rope_theta":10000,
    "rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}"#;

struct Fixture {
    full: DecoderGraph,
    sources: BTreeMap<String, SourceTensor>,
    norms: SourceNorms,
    public: PublicSource,
}

impl Fixture {
    fn new() -> Self {
        let full = lower_with_weights(CONFIG, TOKENS, 1, WeightMode::Public)
            .unwrap()
            .prefill;
        let shapes =
            Program::source_shapes_with_weights(CONFIG, TOKENS, 1, WeightMode::Public).unwrap();
        let sources: BTreeMap<_, _> = shapes
            .into_iter()
            .map(|(key, shape)| {
                let coefficient = full.operations.iter().any(|op| {
                    op.operator == Op::RmsNorm
                        && op.attributes["weight"].as_str() == Some(key.as_str())
                });
                let value = if coefficient { ONE } else { ONE / 64 };
                let lanes = shape.iter().product();
                let mut hash = Sha256::new();
                for _ in 0..lanes {
                    hash.update(value.to_le_bytes());
                }
                (
                    key,
                    SourceTensor {
                        shape,
                        lower: if coefficient { ONE } else { -value },
                        upper: value,
                        public_q18: coefficient.then(|| vec![ONE; lanes]),
                        digest: hash.finalize().into(),
                        row_l1: None,
                        row_squared_norm: None,
                        row_l1_squared_sum: None,
                    },
                )
            })
            .collect();
        let norms = SourceNorms::new();
        let registry = SourceRegistry::from_validated(validated(&full, &sources, &norms)).unwrap();
        // Authenticate the actual tiny fixture values through the original public
        // importer. No checkpoint, invented certificate or digest-only file owner.
        // This allowance is NOT a complete request/construction admission claim.
        let frame_bytes = sources
            .values()
            .map(|source| source.shape.iter().product::<usize>() * 4)
            .max()
            .unwrap();
        let outside = crate::replicated_decoder::source_registry::semantic_metadata_bytes(&full)
            .unwrap()
            + size_of::<SourceRegistry<'_>>()
            + frame_bytes
            + size_of::<Vec<u8>>();
        let mut importer =
            PublicSourceImport::new_checked(&registry, outside, MAX_PROGRAM_BYTES).unwrap();
        for (key, source) in &sources {
            let value = if source.public_q18.is_some() {
                1.0_f32
            } else {
                1.0_f32 / 64.0
            };
            let bytes: Vec<_> = (0..source.shape.iter().product::<usize>())
                .flat_map(|_| value.to_le_bytes())
                .collect();
            importer.append(key, 0, &bytes).unwrap();
        }
        let public = importer.finish().unwrap();
        public.check_registry(&registry).unwrap();
        Self {
            full,
            sources,
            norms,
            public,
        }
    }

    fn imports(&self) -> (Builder, View) {
        let registry = SourceRegistry::from_validated(self.validated()).unwrap();
        self.public.check_registry(&registry).unwrap();
        let mut builder = Builder {
            public_weights: true,
            ..Builder::default()
        };
        // Same source ordering and original PublicInput recipes as Program's
        // public import arm, checked against the authenticated source registry.
        for (expected, key, source, norm) in registry.records() {
            assert!(norm.is_none());
            let bounds = Interval::new(source.lower, source.upper).unwrap();
            let lanes = source.shape.iter().product();
            let owner = builder
                .push(
                    Code::PublicInput {
                        key: key.into(),
                        digest: source.digest,
                        row_l1: source.row_l1,
                        row_squared_norm: source.row_squared_norm,
                        row_l1_squared_sum: source.row_l1_squared_sum,
                    },
                    lanes,
                    bounds,
                )
                .unwrap();
            assert_eq!(owner, expected);
            registry
                .check_descriptor(owner, key, source.digest, lanes, bounds)
                .unwrap();
            assert_eq!(self.public.file(key).unwrap().digest(), source.digest);
        }
        let owner = builder
            .push(
                Code::Input("input.tokens".into()),
                TOKENS,
                Interval::new(0, VOCABULARY as i64 - 1).unwrap(),
            )
            .unwrap();
        let input = builder.view(owner, vec![1, TOKENS]);
        (builder, input)
    }

    fn validated(&self) -> ValidatedSourceDescriptors<'_> {
        validated(&self.full, &self.sources, &self.norms)
    }

    fn connected(&self) -> DecoderGraph {
        let mut semantic = self.full.clone();
        let find = |operator| {
            self.full
                .operations
                .iter()
                .find(|op| op.operator == operator)
                .unwrap()
                .clone()
        };
        let lookup = find(Op::TokenLookup);
        let mut heads = self
            .full
            .operations
            .iter()
            .find(|op| {
                op.operator == Op::Reshape
                    && op.attributes["layout"] == "batch_heads_sequence_feature"
            })
            .unwrap()
            .clone();
        heads.inputs = vec![lookup.id.clone()];
        let mut inverse = self
            .full
            .operations
            .iter()
            .find(|op| {
                op.operator == Op::Reshape && op.attributes["layout"] == "batch_sequence_hidden"
            })
            .unwrap()
            .clone();
        inverse.inputs = vec![heads.id.clone()];
        let mut norm = find(Op::RmsNorm);
        norm.inputs = vec![inverse.id.clone()];
        let start = self
            .full
            .operations
            .iter()
            .position(|op| op.id.ends_with(".gate_proj"))
            .unwrap();
        let mut tail = self.full.operations[start..].to_vec();
        tail[0].inputs = vec![norm.id.clone()];
        tail[1].inputs = vec![norm.id.clone()];
        assert_eq!(tail[5].operator, Op::ResidualAdd);
        tail[5].inputs = vec![inverse.id.clone(), tail[4].id.clone()];
        // Qualify the supported static last-row alias, not ragged valid-length
        // authority. No invented sequence_lengths import is supplied.
        assert_eq!(tail[7].operator, Op::LastToken);
        tail[7].inputs.truncate(1);
        semantic.operations = vec![lookup, heads, inverse, norm];
        semantic.operations.extend(tail);
        semantic.output = semantic.operations.last().unwrap().id.clone();
        semantic.state_inputs.clear();
        semantic.state_outputs.clear();
        assert_eq!(semantic.operations.len(), 15);
        assert!(semantic.operations.len() < self.full.operations.len());
        semantic
    }

    fn lower<'g, 'a>(
        &'a self,
        imports: &'a Builder,
        input: &View,
        semantic: &'g DecoderGraph,
    ) -> (CompactGraph<'g, 'a>, Vec<PublishedView>) {
        let mut compact = CompactGraph::new(imports, self.validated(), semantic, 0).unwrap();
        compact.bind_import("input.tokens", input).unwrap();
        let exports = semantic
            .operations
            .iter()
            .map(|_| compact.lower_next().unwrap())
            .collect();
        (compact, exports)
    }
}

fn validated<'a>(
    full: &DecoderGraph,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
) -> ValidatedSourceDescriptors<'a> {
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    validate_source_records(
        full,
        &shapes,
        sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        norms,
    )
    .unwrap()
}

/// Original graph recipes through original Builder primitives. In particular,
/// lookup uses raw public matrix multiplication, projection uses rescale18, and
/// rank permutations use the expanded graph's coordinate equations independently
/// of compact Mapping/LaneOrder. No oracle numeric shortcut or fusion.
fn expanded_op(
    builder: &mut Builder,
    sources: &BTreeMap<String, SourceTensor>,
    op: &pllm_models::ModelOperation,
    input: &View,
    right: Option<&View>,
) -> View {
    let shape: Vec<_> = op.output_shape.iter().map(|n| *n as usize).collect();
    let source = || {
        let key = op.attributes["weight"].as_str().unwrap();
        SourceView {
            owner: sources
                .iter()
                .filter(|(_, source)| source.public_q18.is_none())
                .position(|(name, _)| name == key)
                .unwrap(),
            shape: sources[key].shape.clone(),
        }
    };
    match op.operator {
        Op::TokenLookup => {
            let source = source();
            let bounds = builder.nodes[source.owner].bounds;
            let mut words = Words::new();
            for (owner, index) in input.words.iter() {
                let original = builder.nodes[owner].bounds;
                let offsets =
                    Interval::new(original.lower() - (VOCABULARY - 1) as i64, original.upper())
                        .unwrap();
                let equal = builder
                    .tensor_loop(LoopInput::TokenOffsets { owner, index }, VOCABULARY, |n| {
                        tensor::Program::equal_zero(n, offsets)
                    })
                    .unwrap();
                let selectors = builder.view(equal, vec![1, VOCABULARY]);
                let row = builder
                    .matrix(
                        &selectors,
                        MatrixView::source(&source, true).unwrap(),
                        false,
                    )
                    .unwrap();
                for (owner, _) in row.words.iter() {
                    builder.nodes[owner].bounds = bounds;
                }
                words.extend(row.words);
            }
            View { words, shape }
        }
        Op::Reshape => {
            let words = match op.attributes["layout"].as_str().unwrap() {
                "batch_heads_sequence_feature" => {
                    let (heads, rows, width) = (shape[1], shape[2], shape[3]);
                    (0..heads)
                        .flat_map(|h| {
                            (0..rows).flat_map(move |r| {
                                (0..width).map(move |d| input.words.at((r * heads + h) * width + d))
                            })
                        })
                        .collect()
                }
                "batch_sequence_hidden" => {
                    let (heads, rows, width) = (input.shape[1], input.shape[2], input.shape[3]);
                    (0..rows)
                        .flat_map(|r| {
                            (0..heads).flat_map(move |h| {
                                (0..width).map(move |d| input.words.at((h * rows + r) * width + d))
                            })
                        })
                        .collect()
                }
                _ => panic!("not an original rank-reshape recipe"),
            };
            View { words, shape }
        }
        Op::RmsNorm => {
            let gamma = &sources[op.attributes["weight"].as_str().unwrap()];
            let width = *input.shape.last().unwrap();
            let mut words = Words::new();
            for row in input.words.chunks_exact(width) {
                let output = builder.rms_norm_row(&row, width, gamma).unwrap();
                words.extend((0..width).map(|i| (output, i)));
            }
            View { words, shape }
        }
        Op::Linear | Op::OutputHead => {
            let source = source();
            let mut output = builder
                .matrix(input, MatrixView::source(&source, false).unwrap(), true)
                .unwrap();
            output.shape = shape;
            builder
                .projection_energies(input, &output, &source, None)
                .unwrap();
            output
        }
        Op::Silu => {
            let gathered = builder.gather(&input.words).unwrap();
            let bounds = builder.nodes[gathered].bounds;
            let output = builder
                .tensor_loop(LoopInput::Slice(gathered), input.words.len(), |n| {
                    tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
                })
                .unwrap();
            builder.view(output, shape)
        }
        Op::Multiply => View {
            words: builder
                .binary(&input.words, &right.unwrap().words, true)
                .unwrap(),
            shape,
        },
        Op::ResidualAdd => View {
            words: builder.residual(input, right.unwrap()).unwrap(),
            shape,
        },
        Op::LastToken => View {
            words: input.words.slice(input.words.len() - WIDTH..),
            shape,
        },
        Op::GreedyTokenSelection => builder.argmax_tree(input).unwrap(),
        Op::TokenFeedback => input.clone(),
        _ => panic!("not a supported connected-region recipe"),
    }
}

fn expanded_digest(nodes: &[Node]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for node in nodes {
        hash_node_header(&mut hash, node.lanes, node.bounds);
        match &node.code {
            Code::Gather(words) => hash_gather(&mut hash, words.iter()),
            Code::TensorLoop {
                input,
                program,
                tail,
            } => hash_tensor_loop(&mut hash, *input, program, tail.as_ref()),
            Code::Tensor { input, program } => hash_tensor(&mut hash, *input, program),
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => hash_public_product(
                &mut hash,
                *input,
                *weight,
                *rows,
                *inner,
                *columns,
                *transposed,
            ),
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => hash_product(&mut hash, *left, *right, *shape, *rescale, norm.as_deref()),
            Code::Affine {
                left,
                right,
                coefficients,
            } => hash_affine(&mut hash, *left, *right, *coefficients),
            Code::PublicValues(values) => {
                hash.update([7]);
                for value in values {
                    hash.update(value.to_le_bytes());
                }
            }
            _ => panic!("unexpected original connected-region node"),
        }
    }
    hash.finalize().into()
}

fn native_attempts() -> (usize, usize, usize, usize) {
    (
        LOOKUP_NATIVE_ATTEMPTS.with(|n| n.get()),
        RMS_NATIVE_ATTEMPTS.with(|n| n.get()),
        ARGMAX_NATIVE_ATTEMPTS.with(|n| n.get()),
        crate::replicated_decoder::prepared_tensor_map::native_attempts(),
    )
}

fn argmax_ids(compact: &CompactGraph<'_, '_>, descriptor: usize) -> Vec<*const tensor::Program> {
    let mut ids = Vec::new();
    compact
        .visit_argmax(descriptor, &mut |event| {
            if let ArgmaxCode::Tensor { program, .. } = event.code {
                ids.push(Arc::as_ptr(program));
            }
            Ok(())
        })
        .unwrap();
    ids
}

#[test]
fn connected_supported_region_matches_original_recipes_and_records_after_semantic_scope_exit() {
    let fixture = Fixture::new();
    let (imports, input) = fixture.imports();
    let imports_before = (
        imports.nodes.len(),
        imports.allocated,
        imports.tensors.len(),
    );
    let (
        frozen,
        words,
        names,
        original_native,
        lookup_id,
        prepared_ids,
        norm_ids,
        argmax,
        work,
        descriptors,
        attempts,
    ) = {
        let semantic = fixture.connected();
        let mut compact = CompactGraph::new(&imports, fixture.validated(), &semantic, 0).unwrap();
        compact.bind_import("input.tokens", &input).unwrap();
        let (mut expanded, original_input) = fixture.imports();
        let first = expanded.nodes.len();
        let mut values = BTreeMap::from([("input.tokens".to_owned(), original_input)]);
        let mut exports: Vec<PublishedView> = Vec::new();
        let mut words = Vec::new();
        for (index, op) in semantic.operations.iter().enumerate() {
            let expected = expanded_op(
                &mut expanded,
                &fixture.sources,
                op,
                &values[&op.inputs[0]],
                op.inputs.get(1).map(|id| &values[id]),
            );
            let before = (compact.next, compact.work_budget());
            let output = compact.lower_next().unwrap();
            assert!(output.words().eq(expected.words.iter()), "{}", op.id);
            assert_eq!(output.logical.dimensions(), expected.shape, "{}", op.id);
            assert_eq!(output.bounds, expanded.bounds(&expected.words).unwrap());
            assert_eq!(compact.next, expanded.nodes.len());
            assert_eq!(
                compact.work_budget(),
                schedule_work(&expanded.nodes[first..]).unwrap()
            );
            let mut digest = Sha256::new();
            compact.hash_expanded_nodes(&mut digest).unwrap();
            assert_eq!(
                <[u8; 32]>::from(digest.finalize()),
                expanded_digest(&expanded.nodes[first..])
            );
            if matches!(op.operator, Op::Reshape | Op::LastToken | Op::TokenFeedback) {
                assert_eq!(before, (compact.next, compact.work_budget()));
            }
            if index == 1 {
                assert_eq!(output.rank, 4);
                assert_ne!(
                    output.words().collect::<Vec<_>>(),
                    exports[0].words().collect::<Vec<_>>()
                );
                assert!(compact.complete_row(&output, 0).unwrap().is_none());
            }
            if index == 2 {
                assert!(output.words().eq(exports[0].words()));
                assert_eq!(output.rank, 3);
                assert!(compact.row_norm(&output).unwrap().is_none());
                assert!(compact
                    .attention_energy(&output, 0, WIDTH)
                    .unwrap()
                    .is_none());
            }
            if op.operator == Op::TokenFeedback {
                assert_eq!(output.rank, 1); // Preserve selected View, not declared [1, 1].
                assert!(output.words().eq(exports[index - 1].words()));
            }
            words.push(expected.words.iter().collect::<Vec<_>>());
            exports.push(output);
            values.insert(op.id.clone(), expected);
        }
        let census = CompactOwnerCensus::read(&compact).unwrap();
        assert_eq!(census.remaining_operations, 0);
        assert_eq!(
            census.body_descriptors, None,
            "metadata completion is not body mapping"
        );
        let retained_values = Vec::new();
        let quote = RequestQuote::new(
            &compact,
            RequestOwners {
                borrowed_external_bytes: 0,
                owned_source_bytes: fixture.public.resident_metadata_bytes().unwrap(),
                other_owned_metadata_bytes: 0,
                other_stored_descriptors: 0,
                other_work: WorkBudget::default(),
                retained_values: &retained_values,
                external_frame_bytes: 0,
                caller_control_bytes: 0,
                prior_request_peak_bytes: 0,
            },
        )
        .unwrap();
        assert!(matches!(
            quote.preflight_body_inputs(&[], &[]),
            Err(Error::Binding)
        ));
        #[allow(clippy::drop_non_drop)] // Explicit end of the borrowed request quote.
        drop(quote);
        for op in &semantic.operations[..semantic.operations.len() - 1] {
            assert!(compact.frontier.value(&op.id).is_none());
        }
        let original_native: Vec<_> = compact
            .owned_native_programs()
            .map(|program| {
                (
                    Arc::as_ptr(program),
                    Arc::downgrade(program),
                    Arc::strong_count(program),
                )
            })
            .collect();
        for (gate, up) in compact
            .kernel(4)
            .unwrap()
            .programs()
            .zip(compact.kernel(5).unwrap().programs())
        {
            assert!(
                Arc::ptr_eq(gate, up),
                "reuse the original prepared capability"
            );
        }
        let lookup = compact.token_lookup_census().next().unwrap();
        assert!(std::ptr::eq(
            lookup.source,
            &fixture.sources[semantic.operations[0].attributes["weight"]
                .as_str()
                .unwrap()]
        ));
        let lookup_id = Arc::as_ptr(lookup.program);
        let prepared_ids: Vec<_> = semantic
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op.operator, Op::Linear | Op::OutputHead | Op::Silu))
            .map(|(index, _)| {
                (
                    index,
                    compact
                        .kernel(index)
                        .unwrap()
                        .programs()
                        .map(Arc::as_ptr)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let norm_ids: Vec<_> = compact
            .native_row_census()
            .map(|row| (row.descriptor, Arc::as_ptr(row.program)))
            .collect();
        let argmax = argmax_ids(&compact, 13);
        assert!(!argmax.is_empty());
        let work = compact.work_budget();
        let descriptors = compact.stored_descriptors();
        let names: Vec<_> = semantic.operations.iter().map(|op| op.id.clone()).collect();
        let attempts = native_attempts();
        let freeze_quote = compact.freeze_quote(&exports).unwrap();
        let frozen = compact.freeze(exports, freeze_quote.peak_bytes).unwrap();
        assert_eq!(frozen.resources(), freeze_quote);
        (
            frozen,
            words,
            names,
            original_native,
            lookup_id,
            prepared_ids,
            norm_ids,
            argmax,
            work,
            descriptors,
            attempts,
        )
        // The only semantic graph used by CompactGraph is destroyed here.
    };
    assert_eq!(native_attempts(), attempts);
    assert_eq!(frozen.work_budget(), work);
    assert!(std::ptr::eq(
        frozen.borrowed_source_records(),
        &fixture.sources
    ));
    assert!(std::ptr::eq(frozen.borrowed_source_norms(), &fixture.norms));
    assert_eq!(
        frozen.source_binding(),
        source_binding_with_norms(&fixture.sources, &fixture.norms)
    );
    for (owner, original) in imports.nodes.iter().enumerate() {
        assert!(std::ptr::eq(
            frozen.imported_recipe(owner).unwrap(),
            original
        ));
    }
    assert_eq!(
        Arc::as_ptr(frozen.token_lookup_census().next().unwrap().program),
        lookup_id
    );
    for (index, ids) in prepared_ids {
        assert_eq!(
            frozen
                .prepared(index)
                .unwrap()
                .programs()
                .map(Arc::as_ptr)
                .collect::<Vec<_>>(),
            ids
        );
    }
    for (index, id) in norm_ids {
        assert_eq!(Arc::as_ptr(frozen.native_row_program(index).unwrap()), id);
    }
    let mut frozen_argmax = Vec::new();
    frozen
        .visit_argmax(13, &mut |event| {
            if let ArgmaxCode::Tensor { program, .. } = event.code {
                frozen_argmax.push(Arc::as_ptr(program));
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(frozen_argmax, argmax);
    for (index, (view, expected)) in frozen.exports().iter().zip(&words).enumerate() {
        assert_eq!(
            (0..expected.len())
                .map(|lane| frozen.word(view, lane).unwrap())
                .collect::<Vec<_>>(),
            *expected
        );
        assert_eq!(frozen.recipe_name(index), Some(names[index].as_str()));
    }
    assert!(
        frozen.named_view(&names[0]).is_none(),
        "retired semantic names stay retired"
    );
    assert_eq!(frozen.output().unwrap().word(0).unwrap(), words[14][0]);
    // This output belongs only to the explicitly rewired supported region. No
    // rotary, attention, KV-cache, final acceptance or executable request exists.
    let ordinary = WorkBudget::local();
    let mut request = RequestWork::default();
    request.record_work(ordinary).unwrap();
    for repetitions in 1..=2 {
        request.record_frozen(&frozen).unwrap();
        assert_eq!(
            request.work(),
            ordinary
                .checked_add(work.checked_mul(repetitions).unwrap())
                .unwrap()
        );
        assert_eq!(
            request.stored_descriptors,
            descriptors * repetitions as usize
        );
        assert_eq!(
            request.retained_graph_bytes,
            frozen.resources().retained_bytes * repetitions as usize
        );
        assert_eq!(request.regions, repetitions as usize);
    }
    assert_retained_boundary(&frozen, work, descriptors);
    for (id, weak, count) in &original_native {
        assert_eq!(weak.as_ptr(), *id);
        assert_eq!(
            weak.strong_count(),
            *count,
            "recording must not clone/rebuild native owners"
        );
    }
    assert_eq!(
        imports_before,
        (
            imports.nodes.len(),
            imports.allocated,
            imports.tensors.len()
        )
    );
    assert_eq!(native_attempts(), attempts);
    drop(frozen);
    assert!(original_native
        .iter()
        .all(|(_, weak, _)| weak.upgrade().is_none()));
}

fn assert_retained_boundary(
    frozen: &frozen_region::FrozenRegion<'_>,
    work: WorkBudget,
    descriptors: usize,
) {
    // Counter-boundary setup only; no forged recipe/source/certificate. Exercise
    // actual record_frozen census at exact retained cap and one byte over it.
    let retained = frozen.resources().retained_bytes;
    let mut exact = RequestWork {
        work: WorkBudget::default(),
        stored_descriptors: MAX_NODES - descriptors,
        retained_graph_bytes: MAX_PROGRAM_BYTES - retained,
        regions: 2,
    };
    exact.record_frozen(frozen).unwrap();
    assert_eq!(exact.work(), work);
    assert_eq!(exact.stored_descriptors, MAX_NODES);
    assert_eq!(exact.retained_graph_bytes, MAX_PROGRAM_BYTES);
    assert_eq!(exact.regions, 3);
    let mut short = RequestWork {
        work: WorkBudget::local(),
        stored_descriptors: 7,
        retained_graph_bytes: MAX_PROGRAM_BYTES - retained + 1,
        regions: 2,
    };
    let before = (
        short.work(),
        short.stored_descriptors,
        short.retained_graph_bytes,
        short.regions,
    );
    assert!(matches!(short.record_frozen(frozen), Err(Error::Bounds)));
    assert_eq!(
        (
            short.work(),
            short.stored_descriptors,
            short.retained_graph_bytes,
            short.regions
        ),
        before
    );
}

#[test]
fn connected_freeze_exact_fit_and_one_short_keep_original_factories_out_of_transition() {
    let fixture = Fixture::new();
    let semantic = fixture.connected();
    let (imports, input) = fixture.imports();
    let (compact, exports) = fixture.lower(&imports, &input, &semantic);
    let quote = compact.freeze_quote(&exports).unwrap();
    quote.admit(quote.peak_bytes).unwrap();
    assert_eq!(quote.admit(quote.peak_bytes - 1), Err(Error::Bounds));
    let owners: Vec<Weak<tensor::Program>> = compact
        .owned_native_programs()
        .map(Arc::downgrade)
        .collect();
    let attempts = native_attempts();
    assert!(matches!(
        compact.freeze(exports, quote.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    assert_eq!(native_attempts(), attempts);
    assert!(owners.iter().all(|weak| weak.upgrade().is_none()));
    let (compact, exports) = fixture.lower(&imports, &input, &semantic);
    let quote = compact.freeze_quote(&exports).unwrap();
    let attempts = native_attempts();
    let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
    assert_eq!(frozen.resources(), quote);
    assert_eq!(native_attempts(), attempts);
}

#[test]
fn rank_four_transition_and_original_decoder_barrier_reject_without_request_authority() {
    let fixture = Fixture::new();
    let (imports, input) = fixture.imports();
    let mut rank_four = fixture.connected();
    let mut linear = rank_four.operations[4].clone();
    linear.inputs = vec![rank_four.operations[1].id.clone()];
    rank_four.operations.truncate(2); // Original lookup -> genuine rank-four alias.
    rank_four.output = linear.id.clone();
    rank_four.operations.push(linear);
    for semantic in [&rank_four, &fixture.full] {
        let mut compact = CompactGraph::new(&imports, fixture.validated(), semantic, 0).unwrap();
        compact.bind_import("input.tokens", &input).unwrap();
        let mut exports = Vec::new();
        loop {
            let before = (compact.next, compact.work_budget(), compact.ops.len());
            let attempts = native_attempts();
            match compact.lower_next() {
                Ok(output) => exports.push(output),
                Err(error) => {
                    assert_eq!(error, Error::Shape);
                    assert_eq!(
                        (compact.next, compact.work_budget(), compact.ops.len()),
                        before
                    );
                    assert_eq!(native_attempts(), attempts);
                    break;
                }
            }
        }
        let failed = compact.failure_site().unwrap();
        assert_eq!(
            failed.operator,
            if std::ptr::eq(semantic, &rank_four) {
                Op::Linear
            } else {
                Op::RotaryEmbedding
            }
        );
        let mut request = RequestWork::default();
        assert!(matches!(
            request.record_graph(&compact),
            Err(Error::Retired)
        ));
        assert_eq!(request.work(), WorkBudget::default());
        assert_eq!(request.regions, 0);
        assert!(matches!(compact.lower_next(), Err(Error::Retired)));
        assert!(matches!(
            compact.freeze_quote(&exports),
            Err(Error::Retired)
        ));
        assert!(matches!(
            compact.freeze(exports, MAX_PROGRAM_BYTES),
            Err(Error::Retired)
        ));
    }
}
