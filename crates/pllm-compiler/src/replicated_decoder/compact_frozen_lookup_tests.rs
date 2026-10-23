//! Frozen original public PerToken owner retention, not execution authority.
use super::*;

fn fixture(
    vocabulary: usize,
    tokens: usize,
) -> (DecoderGraph, BTreeMap<String, SourceTensor>, Builder, View) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type": "llama", "hidden_size": 2, "intermediate_size": 2,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": vocabulary, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": false, "max_position_embeddings": 64,
        "rope_theta": 10000, "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap();
    let full = lower_with_weights(&config, tokens, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let sources: BTreeMap<_, _> =
        Program::source_shapes_with_weights(&config, tokens, 1, WeightMode::Public)
            .unwrap()
            .into_iter()
            .map(|(key, shape)| {
                let public = key.contains("norm");
                let value = if public { ONE } else { ONE / 64 };
                let lanes = shape.iter().product();
                let mut hash = Sha256::new();
                for _ in 0..lanes {
                    hash.update(value.to_le_bytes());
                }
                (
                    key,
                    SourceTensor {
                        shape,
                        lower: if public { ONE } else { -value },
                        upper: value,
                        public_q18: public.then(|| vec![ONE; lanes]),
                        digest: hash.finalize().into(),
                        row_l1: None,
                        row_squared_norm: None,
                        row_l1_squared_sum: None,
                    },
                )
            })
            .collect();
    let mut imports = Builder {
        public_weights: true,
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
                    row_l1: source.row_l1,
                    row_squared_norm: source.row_squared_norm,
                    row_l1_squared_sum: source.row_l1_squared_sum,
                },
                source.shape.iter().product(),
                Interval::new(source.lower, source.upper).unwrap(),
            )
            .unwrap();
    }
    let owner = imports
        .push(
            Code::Input("input.tokens".into()),
            tokens,
            Interval::new(0, vocabulary as i64 - 1).unwrap(),
        )
        .unwrap();
    let input = imports.view(owner, vec![1, tokens]);
    (full, sources, imports, input)
}

fn validated<'a>(
    full: &DecoderGraph,
    sources: &'a BTreeMap<String, SourceTensor>,
    norms: &'a SourceNorms,
    execution: ExecutionPolicy,
) -> ValidatedSourceDescriptors<'a> {
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    validate_source_records(full, &shapes, sources, WeightMode::Public, execution, norms).unwrap()
}

fn lookup_region(full: &DecoderGraph) -> DecoderGraph {
    let mut semantic = full.clone();
    let lookup = full
        .operations
        .iter()
        .find(|op| op.operator == Op::TokenLookup)
        .unwrap()
        .clone();
    semantic.output = lookup.id.clone();
    semantic.operations = vec![lookup];
    semantic
}

fn native_attempts() -> (usize, usize, usize, usize) {
    (
        LOOKUP_NATIVE_ATTEMPTS.with(|count| count.get()),
        super::super::super::super::prepared_tensor_map::native_attempts(),
        RMS_NATIVE_ATTEMPTS.with(|count| count.get()),
        ARGMAX_NATIVE_ATTEMPTS.with(|count| count.get()),
    )
}

struct KernelSnapshot {
    handle: *const Arc<tensor::Program>,
    owner: std::sync::Weak<tensor::Program>,
    strong_count: usize,
}

impl KernelSnapshot {
    fn new(program: &Arc<tensor::Program>) -> Self {
        Self {
            handle: program,
            owner: Arc::downgrade(program),
            strong_count: Arc::strong_count(program),
        }
    }

    fn assert_original(&self, program: &Arc<tensor::Program>) {
        assert_eq!(self.handle, program as *const _);
        assert_eq!(self.owner.as_ptr(), Arc::as_ptr(program));
        assert_eq!(self.strong_count, Arc::strong_count(program));
        assert!(Arc::ptr_eq(&self.owner.upgrade().unwrap(), program));
    }
}

struct LookupSnapshot {
    descriptor: usize,
    first: usize,
    geometry: [usize; 3],
    source_owner: usize,
    source: *const SourceTensor,
    binding: [u8; 32],
    full: KernelSnapshot,
    tail: Option<KernelSnapshot>,
}

impl LookupSnapshot {
    fn new(census: TokenLookupCensus<'_, '_>) -> Self {
        Self {
            descriptor: census.descriptor,
            first: census.first,
            geometry: census.geometry,
            source_owner: census.source_owner,
            source: census.source,
            binding: census.source_binding,
            full: KernelSnapshot::new(census.program),
            tail: census.tail.map(KernelSnapshot::new),
        }
    }

    fn assert_original(&self, census: TokenLookupCensus<'_, '_>) {
        assert_eq!(self.descriptor, census.descriptor);
        assert_eq!(self.first, census.first);
        assert_eq!(self.geometry, census.geometry);
        assert_eq!(self.source_owner, census.source_owner);
        assert_eq!(self.source, census.source as *const _);
        assert_eq!(self.binding, census.source_binding);
        self.full.assert_original(census.program);
        match (&self.tail, census.tail) {
            (Some(expected), Some(actual)) => expected.assert_original(actual),
            (None, None) => {}
            _ => panic!("original tail ownership changed"),
        }
    }
}

#[test]
fn frozen_lookup_census_borrows_moved_recipes_after_semantic_scope_exit() {
    for vocabulary in [1, 127, 128, 129, 256, 257] {
        for tokens in [1, 3] {
            let (full, sources, imports, input) = fixture(vocabulary, tokens);
            let norms = SourceNorms::new();
            let imported = (
                imports.nodes.len(),
                imports.allocated,
                imports.tensors.len(),
            );
            let (frozen, snapshots, words, work, next, recipes, exports, origin, quote, attempts) = {
                let mut semantic = lookup_region(&full);
                let mut last = full
                    .operations
                    .iter()
                    .find(|op| op.operator == Op::LastToken)
                    .unwrap()
                    .clone();
                last.inputs = vec![semantic.output.clone()];
                let mut again = semantic.operations[0].clone();
                again.id.push_str(".again");
                semantic.output = again.id.clone();
                semantic.operations.extend([last, again]);
                let mut compact = CompactGraph::new(
                    &imports,
                    validated(&full, &sources, &norms, ExecutionPolicy::default()),
                    &semantic,
                    0,
                )
                .unwrap();
                assert_eq!(compact.token_lookup_census().count(), 0);
                compact.bind_import("input.tokens", &input).unwrap();
                let mut exports = Vec::new();
                for _ in 0..3 {
                    exports.push(compact.lower_next().unwrap());
                }
                let attempts = native_attempts();
                let snapshots: Vec<_> = compact
                    .token_lookup_census()
                    .map(LookupSnapshot::new)
                    .collect();
                assert_eq!(snapshots.len(), 2);
                assert_eq!(snapshots[0].descriptor, 0);
                assert_eq!(snapshots[1].descriptor, 2);
                assert_eq!(
                    snapshots[0].full.owner.as_ptr(),
                    snapshots[1].full.owner.as_ptr()
                );
                assert_eq!(snapshots[0].full.strong_count, 2);
                assert_eq!(
                    snapshots[0].tail.is_some(),
                    vocabulary > 128 && vocabulary % 128 != 0
                );
                if let Some(tail) = &snapshots[0].tail {
                    assert_eq!(
                        tail.owner.as_ptr(),
                        snapshots[1].tail.as_ref().unwrap().owner.as_ptr()
                    );
                    assert_ne!(tail.owner.as_ptr(), snapshots[0].full.owner.as_ptr());
                    assert_eq!(tail.strong_count, 2);
                }
                let words: Vec<Vec<_>> =
                    exports.iter().map(|view| view.words().collect()).collect();
                let work = compact.work_budget();
                let next = compact.next;
                let recipes = compact.ops.as_ptr();
                let export_ptr = exports.as_ptr();
                let origin = Arc::downgrade(&compact.origin);
                let quote = compact.freeze_quote(&exports).unwrap();
                assert_eq!(
                    quote.native_owner_bytes,
                    compact
                        .owned_native_programs()
                        .map(|program| {
                            program.resources().unwrap().immutable_template_bytes
                                + 2 * size_of::<usize>()
                        })
                        .sum::<usize>()
                );
                let frozen = compact.freeze(exports, quote.peak_bytes).unwrap();
                (
                    frozen, snapshots, words, work, next, recipes, export_ptr, origin, quote,
                    attempts,
                )
                // The semantic region is destroyed here; no census borrows it.
            };
            assert_eq!(frozen.ops.as_ptr(), recipes);
            assert_eq!(frozen.exports().as_ptr(), exports);
            assert_eq!(frozen.resources(), quote);
            assert_eq!(frozen.work_budget(), work);
            assert_eq!(frozen.next_virtual_owner(), next);
            assert!(std::ptr::eq(frozen.borrowed_source_records(), &sources));
            assert_eq!(frozen.source_binding(), snapshots[0].binding);
            let census: Vec<_> = frozen.token_lookup_census().collect();
            assert_eq!(census.len(), snapshots.len());
            assert!(Arc::ptr_eq(census[0].program, census[1].program));
            for (expected, actual) in snapshots.iter().zip(census) {
                let source = actual.source;
                expected.assert_original(actual);
                assert_eq!(expected.geometry, [tokens, vocabulary, 2]);
                let Code::PublicInput { key, digest, .. } =
                    &frozen.imported_recipe(expected.source_owner).unwrap().code
                else {
                    panic!("source owner must remain the original public import");
                };
                assert!(std::ptr::eq(source, &sources[key]));
                assert_eq!(*digest, source.digest);
                assert_eq!(
                    expected.binding,
                    source_binding_with_norms(&sources, &norms)
                );
            }
            for (view, expected) in frozen.exports().iter().zip(&words) {
                assert_eq!(
                    (0..view.lanes)
                        .map(|lane| frozen.word(view, lane).unwrap())
                        .collect::<Vec<_>>(),
                    *expected
                );
                assert!(Arc::ptr_eq(&origin.upgrade().unwrap(), &view.origin));
            }
            assert_eq!(words[1], words[0][2 * (tokens - 1)..]);
            for snapshot in &snapshots {
                let view = &frozen.exports()[snapshot.descriptor];
                for token in 0..tokens {
                    for lane in 0..2 {
                        assert_eq!(
                            frozen.word(view, token * 2 + lane).unwrap(),
                            (snapshot.first + 3 * token + 2, lane)
                        );
                    }
                }
            }
            let mut foreign = frozen.exports()[0].clone();
            foreign.origin = Arc::new(());
            assert!(matches!(frozen.resolve(&foreign), Err(Error::Binding)));
            drop(foreign);
            assert_eq!(frozen.output().unwrap().word(0).unwrap(), words[2][0]);
            assert_eq!(
                native_attempts(),
                attempts,
                "freeze/census must not construct kernels"
            );
            assert_eq!(
                imported,
                (
                    imports.nodes.len(),
                    imports.allocated,
                    imports.tensors.len()
                )
            );
            drop(frozen);
            assert!(origin.upgrade().is_none());
            for snapshot in snapshots {
                assert!(snapshot.full.owner.upgrade().is_none());
                if let Some(tail) = snapshot.tail {
                    assert!(tail.owner.upgrade().is_none());
                }
            }
        }
    }
}

#[test]
fn frozen_lookup_does_not_authorize_unsupported_shapes_or_policies() {
    let (full, sources, imports, input) = fixture(129, 3);
    let norms = SourceNorms::new();
    for output_shape in [vec![1, 3, 3], vec![1, 6], vec![2, 3, 1]] {
        let mut semantic = lookup_region(&full);
        semantic.operations[0].output_shape = output_shape;
        let mut compact = CompactGraph::new(
            &imports,
            validated(&full, &sources, &norms, ExecutionPolicy::default()),
            &semantic,
            0,
        )
        .unwrap();
        compact.bind_import("input.tokens", &input).unwrap();
        let attempts = native_attempts();
        assert!(matches!(compact.lower_next(), Err(Error::Shape)));
        assert_eq!(compact.token_lookup_census().count(), 0);
        assert_eq!(compact.work_budget(), WorkBudget::default());
        assert!(matches!(
            compact.freeze(Vec::new(), MAX_PROGRAM_BYTES),
            Err(Error::Retired)
        ));
        assert_eq!(native_attempts(), attempts);
    }
    for lookup in [LookupPolicy::Tiled, LookupPolicy::Trie] {
        let (full, sources, mut imports, input) = fixture(129, 3);
        imports.lookup = lookup;
        let semantic = lookup_region(&full);
        let mut compact = CompactGraph::new(
            &imports,
            validated(
                &full,
                &sources,
                &norms,
                ExecutionPolicy {
                    lookup,
                    ..ExecutionPolicy::default()
                },
            ),
            &semantic,
            0,
        )
        .unwrap();
        compact.bind_import("input.tokens", &input).unwrap();
        let attempts = native_attempts();
        assert!(matches!(compact.lower_next(), Err(Error::Binding)));
        assert_eq!(compact.token_lookup_census().count(), 0);
        assert!(matches!(
            compact.freeze(Vec::new(), MAX_PROGRAM_BYTES),
            Err(Error::Retired)
        ));
        assert_eq!(native_attempts(), attempts);
    }
}
