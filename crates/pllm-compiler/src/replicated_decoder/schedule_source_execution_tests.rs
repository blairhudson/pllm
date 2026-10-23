//! Authenticated files + genuine original Builder custody, TEST ONLY.
//! Not a bounded producer, prefix handoff, manifest/request completion or drain
//! issuer. Full Manifest authentication and the closed prefix are distinct owners.
use super::super::*;
use super::*;
use crate::replicated_decoder::runtime::{frozen_execution_tests as custody, Worker};
use crate::replicated_decoder::source_registry::{self, SourceConstructionOwners, SourceRegistry};
use crate::replicated_decoder::{Manifest, PublicSource, PublicSourceImport};
use puma::{Context, Party};
use std::mem::size_of;

struct Fixture {
    manifest: Manifest,
    full: DecoderGraph,
    prefix: DecoderGraph,
    lookup: bool,
    vocabulary: usize,
}

fn coefficient(key: &str, index: usize) -> i64 {
    if key.contains("norm") {
        ONE
    } else if key == "model.embed_tokens.weight" {
        (index as i64 % 5 - 2) * (ONE / 256)
    } else if key == "lm_head.weight" {
        (index as i64 % 3 - 1) * (ONE / 128)
    } else if key.ends_with("q_proj.weight") && index % 3 == 0 {
        ONE / 64
    } else {
        0
    }
}

impl Fixture {
    fn new(lookup: bool, vocabulary: usize) -> Self {
        let config = serde_json::to_vec(&serde_json::json!({"model_type":"llama",
            "hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":2,
            "vocab_size":vocabulary,"tie_word_embeddings":false,"hidden_act":"silu",
            "attention_bias":false,"max_position_embeddings":64,"rope_theta":10000,
            "rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}))
        .unwrap();
        let rows = 2;
        let full = lower_with_weights(&config, rows, 1, WeightMode::Public)
            .unwrap()
            .prefill;
        let shapes =
            Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public).unwrap();
        let mut tensors = BTreeMap::new();
        for (key, shape) in shapes {
            let lanes = shape.iter().product::<usize>();
            let mut hash = Sha256::new();
            let (mut lower, mut upper) = (i64::MAX, i64::MIN);
            for i in 0..lanes {
                let value = coefficient(&key, i);
                lower = lower.min(value);
                upper = upper.max(value);
                hash.update(value.to_le_bytes());
            }
            let public_q18 = key.contains("norm").then(|| vec![ONE; lanes]);
            tensors.insert(
                key,
                SourceTensor {
                    shape,
                    lower,
                    upper,
                    public_q18,
                    digest: hash.finalize().into(),
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
            );
        }
        let mut manifest = Manifest {
            schema: "pllm.replicated-decoder.manifest.v1".into(),
            weights: WeightMode::Public,
            execution: ExecutionPolicy::default(),
            config,
            rows,
            outputs: 1,
            source: [31; 32],
            composition: [32; 32],
            session: [33; 32],
            tensors,
            norms: SourceNorms::new(),
            program: [0; 32],
        };
        // A REAL ordinary full-model manifest identity, not the prefix digest.
        // Existing compiler construction admission applies; that earlier phase
        // is not retroactively claimed as bounded frozen production admission.
        let mut original = Program::compile_with_norms(
            &manifest.config,
            rows,
            1,
            &manifest.tensors,
            manifest.weights,
            manifest.execution,
            &manifest.norms,
        )
        .unwrap();
        original.bind(manifest.source, manifest.composition);
        manifest.program = original.digest();
        drop(original);
        drop(manifest.compile().unwrap());
        let find = |operator| {
            full.operations
                .iter()
                .find(|op| op.operator == operator)
                .unwrap()
                .clone()
        };
        let mut operations = Vec::new();
        let mut input = "input.data".to_string();
        if lookup {
            let op = find(Op::TokenLookup);
            input = op.id.clone();
            operations.push(op);
        } else {
            // Two genuine affine descriptors force nonidentity physical slots
            // and nonzero offsets before the source-backed raw output.
            for id in ["sum", "sum.again"] {
                let mut op = find(Op::ResidualAdd);
                op.id = id.into();
                op.inputs = vec![input, "input.data".into()];
                op.output_shape = vec![1, rows as u64, 2];
                input = op.id.clone();
                operations.push(op);
            }
            let mut linear = full
                .operations
                .iter()
                .find(|op| {
                    op.operator == Op::Linear
                        && op.attributes["weight"]
                            .as_str()
                            .unwrap()
                            .ends_with("q_proj.weight")
                })
                .unwrap()
                .clone();
            linear.inputs = vec![input];
            linear.output_shape = vec![1, rows as u64, 2];
            input = linear.id.clone();
            operations.push(linear);
        }
        let mut last = find(Op::LastToken);
        last.inputs = vec![input];
        let mut head = find(Op::OutputHead);
        head.inputs = vec![last.id.clone()];
        operations.push(last);
        operations.push(head.clone());
        let mut prefix = full.clone();
        prefix.operations = operations;
        prefix.output = head.id;
        Self {
            manifest,
            full,
            prefix,
            lookup,
            vocabulary,
        }
    }
    fn key(&self) -> &'static str {
        if self.lookup {
            "input.tokens"
        } else {
            "input.data"
        }
    }
    fn input(&self, i: usize) -> i64 {
        if self.lookup {
            (i % self.vocabulary) as i64
        } else {
            (i as i64 % 3 - 1) * (ONE / 256)
        }
    }
    fn registry(&self) -> SourceRegistry<'_> {
        let shapes: BTreeMap<_, _> = self
            .manifest
            .tensors
            .iter()
            .map(|(key, source)| (key.clone(), source.shape.clone()))
            .collect();
        let outside = source_registry::semantic_metadata_bytes(&self.prefix).unwrap();
        let owners = SourceConstructionOwners {
            manifest: &self.manifest,
            incoming: None,
            public: None,
            retained_baseline_bytes: outside,
        };
        source_registry::descriptor_validation_quote(
            &self.full,
            &shapes,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap()
        .admit(MAX_PROGRAM_BYTES)
        .unwrap();
        let validated = validate_source_records(
            &self.full,
            &shapes,
            &self.manifest.tensors,
            self.manifest.weights,
            self.manifest.execution,
            &self.manifest.norms,
        )
        .unwrap();
        let shape_bytes =
            shapes
                .iter()
                .fold(size_of::<BTreeMap<String, Vec<usize>>>(), |n, (k, v)| {
                    n + source_registry::tree_entry_bytes::<(String, Vec<usize>)>().unwrap()
                        + k.capacity()
                        + v.capacity() * size_of::<usize>()
                });
        let retained =
            outside + shape_bytes + source_registry::semantic_metadata_bytes(&self.full).unwrap();
        let owners = SourceConstructionOwners {
            retained_baseline_bytes: retained,
            ..owners
        };
        SourceRegistry::from_validated_with_owners_checked(validated, &owners, MAX_PROGRAM_BYTES)
            .unwrap()
    }
    fn imports(&self) -> (Builder, View) {
        let mut b = Builder {
            public_weights: true,
            ..Builder::default()
        };
        for (key, s) in self
            .manifest
            .tensors
            .iter()
            .filter(|(_, s)| s.public_q18.is_none())
        {
            b.push(
                Code::PublicInput {
                    key: key.clone(),
                    digest: s.digest,
                    row_l1: s.row_l1,
                    row_squared_norm: s.row_squared_norm,
                    row_l1_squared_sum: s.row_l1_squared_sum,
                },
                s.shape.iter().product(),
                Interval::new(s.lower, s.upper).unwrap(),
            )
            .unwrap();
        }
        let lanes = if self.lookup { 2 } else { 4 };
        let bounds = if self.lookup {
            Interval::new(0, self.vocabulary as i64 - 1).unwrap()
        } else {
            Interval::new(-ONE / 256, ONE / 256).unwrap()
        };
        let id = b
            .push(Code::Input(self.key().into()), lanes, bounds)
            .unwrap();
        let shape = if self.lookup {
            vec![1, 2]
        } else {
            vec![1, 2, 2]
        };
        let view = b.view(id, shape);
        (b, view)
    }
    fn freeze<'a>(&'a self, imports: &'a Builder, input: &View) -> FrozenRegion<'a> {
        let shapes = self
            .manifest
            .tensors
            .iter()
            .map(|(k, s)| (k.clone(), s.shape.clone()))
            .collect();
        let token = validate_source_records(
            &self.full,
            &shapes,
            &self.manifest.tensors,
            self.manifest.weights,
            self.manifest.execution,
            &self.manifest.norms,
        )
        .unwrap();
        let mut compact =
            compact_lowering::CompactGraph::new(imports, token, &self.prefix, 0).unwrap();
        compact.bind_import(self.key(), input).unwrap();
        let mut out = None;
        for _ in &self.prefix.operations {
            out = Some(compact.lower_next().unwrap());
        }
        compact
            .freeze(vec![out.unwrap()], MAX_PROGRAM_BYTES)
            .unwrap()
    }
    fn original(&self, frozen: &FrozenRegion<'_>) -> Arc<Program> {
        let (mut b, mut current) = self.imports();
        let imports = b.nodes.len();
        // Original numeric constructors below retain the very same interner
        // owners. This copies Arc handles ONLY, never coefficients or recipes.
        let mut retain = |p: &Arc<tensor::Program>| {
            if let Some(original) = b.tensors.get(&p.digest()) {
                assert!(Arc::ptr_eq(original, p), "same-digest foreign native owner");
            } else {
                let bytes = p.resources().unwrap().immutable_template_bytes
                    + 2 * size_of::<usize>()
                    + source_registry::tree_entry_bytes::<([u8; 32], Arc<tensor::Program>)>()
                        .unwrap();
                b.charge_program_bytes(bytes).unwrap();
                b.tensors.insert(p.digest(), p.clone());
            }
        };
        for i in 0..frozen.descriptor_census().count() {
            if let Ok(map) = frozen.prepared(i) {
                for p in map.programs() {
                    retain(p);
                }
            }
        }
        for c in frozen.token_lookup_census() {
            retain(c.program);
            if let Some(t) = c.tail {
                retain(t);
            }
        }
        #[allow(clippy::drop_non_drop)] // Release the closure's exclusive Builder borrow.
        drop(retain);
        let source_view = |key: &str| SourceView {
            owner: self
                .manifest
                .tensors
                .iter()
                .filter(|(_, s)| s.public_q18.is_none())
                .position(|(name, _)| name == key)
                .unwrap(),
            shape: self.manifest.tensors[key].shape.clone(),
        };
        let original_input = current.clone();
        for op in &self.prefix.operations {
            let shape = op.output_shape.iter().map(|n| *n as usize).collect();
            current = match op.operator {
                Op::ResidualAdd => View {
                    words: b.residual(&current, &original_input).unwrap(),
                    shape,
                },
                Op::LastToken => View {
                    words: current.words.slice(current.words.len() - 2..),
                    shape,
                },
                Op::Linear | Op::OutputHead => {
                    let source = source_view(op.attributes["weight"].as_str().unwrap());
                    let mut output = b
                        .matrix(&current, MatrixView::source(&source, false).unwrap(), true)
                        .unwrap();
                    output.shape = shape;
                    b.projection_energies(&current, &output, &source, None)
                        .unwrap();
                    output
                }
                Op::TokenLookup => {
                    let key = op.attributes["weight"].as_str().unwrap();
                    let source = source_view(key);
                    let mut words = Words::new();
                    for (owner, index) in original_input.words.iter() {
                        let bounds = b.nodes[owner].bounds;
                        let difference = Interval::new(
                            bounds.lower() - (self.vocabulary - 1) as i64,
                            bounds.upper(),
                        )
                        .unwrap();
                        let selector = b
                            .tensor_loop(
                                LoopInput::TokenOffsets { owner, index },
                                self.vocabulary,
                                |n| tensor::Program::equal_zero(n, difference),
                            )
                            .unwrap();
                        let selectors = b.view(selector, vec![1, 1, self.vocabulary]);
                        let row = b
                            .matrix(
                                &selectors,
                                MatrixView::source(&source, true).unwrap(),
                                false,
                            )
                            .unwrap();
                        let bounds = Interval::new(
                            self.manifest.tensors[key].lower,
                            self.manifest.tensors[key].upper,
                        )
                        .unwrap();
                        for (owner, _) in row.words.iter() {
                            b.nodes[owner].bounds = bounds;
                        }
                        words.extend(row.words.iter());
                    }
                    View { words, shape }
                }
                _ => panic!("unsupported original closed prefix"),
            };
        }
        Arc::new(
            b.finish(
                imports,
                current.words.iter().collect(),
                &mut AdmissionSite::default(),
                |h| {
                    h.update(b"original-source-frozen-execution-test-only");
                    h.update(self.manifest.source);
                    h.update(self.manifest.composition);
                    Ok(())
                },
                || source_binding_with_norms(&self.manifest.tensors, &self.manifest.norms),
            )
            .unwrap(),
        )
    }
}

/// Source/registry factories are independently checked BEFORE file buffers and
/// party/native factories. Existing fixture/expanded construction is baseline,
/// not retrospectively certified as bounded production construction.
fn authenticated(f: &Fixture, registry: &SourceRegistry<'_>, retained: usize) -> PublicSource {
    let mut importer =
        PublicSourceImport::new_checked(registry, retained, MAX_PROGRAM_BYTES).unwrap();
    for (key, s) in &f.manifest.tensors {
        let lanes = s.shape.iter().product::<usize>();
        for start in (0..lanes).step_by(3) {
            let mut frame = [0u8; 12];
            let count = (lanes - start).min(3);
            for i in 0..count {
                let value = coefficient(key, start + i) as f32 / ONE as f32;
                frame[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
            }
            importer.append(key, start, &frame[..count * 4]).unwrap();
        }
    }
    let source = importer.finish().unwrap();
    source.check_registry(registry).unwrap();
    source
}

fn pair(
    f: &Fixture,
    registry: &SourceRegistry<'_>,
    frozen: &FrozenRegion<'_>,
    program: Arc<Program>,
) -> ([Worker; 3], [Worker; 3]) {
    let mut retained = frozen.resources().peak_bytes
        + 12 * program.frozen_oracle_metadata_bytes().unwrap()
        + 6 * size_of::<Worker>()
        + 12 * puma::MAX_ADMITTED_BYTES
        + 6 * program.logical_len() * size_of::<Option<Secret>>();
    let input_lanes: usize = program.inputs().map(|(_, n)| n).sum();
    // Six retained imported owners plus four transient input/component arrays.
    retained += (6 * 2 + 4) * input_lanes * size_of::<u64>();
    for (_, key, _, _) in registry.records() {
        retained += 6
            * (source_registry::tree_entry_bytes::<(String, pllm_core::public_file::PublicFile)>()
                .unwrap()
                + key.len()
                + size_of::<std::fs::File>()
                + 2 * size_of::<usize>());
    }
    retained += 6 * size_of::<PublicSource>();
    // Manifest config/schema inline/heap owner lies outside frozen source maps.
    retained += size_of::<Manifest>() + f.manifest.schema.capacity() + f.manifest.config.capacity();
    retained += source_registry::semantic_metadata_bytes(&f.full).unwrap()
        + source_registry::semantic_metadata_bytes(&f.prefix).unwrap();
    PublicSourceImport::construction_quote(registry, retained)
        .unwrap()
        .admit(MAX_PROGRAM_BYTES)
        .unwrap();
    let context = Context::new(f.manifest.session, program.digest(), 0).unwrap();
    let fresh = || {
        puma::prss::insecure_test_support::fresh_compound_parties(
            context,
            program.work_budget().limits().unwrap(),
        )
        .unwrap()
    };
    let make = |runtimes: [multiply::PartyRuntime; 3]| {
        let mut workers = runtimes.map(|runtime| {
            let source = authenticated(f, registry, retained);
            Worker::new_with_source(program.clone(), runtime, f.manifest.session, Some(source))
                .unwrap()
        });
        for (key, n) in program.inputs() {
            let a: Vec<_> = (0..n)
                .map(|i| 0x12345678_u64.wrapping_add(i as u64 * 17))
                .collect();
            let b: Vec<_> = (0..n)
                .map(|i| 0x98765432_u64.wrapping_add(i as u64 * 31))
                .collect();
            let c: Vec<_> = (0..n)
                .map(|i| (f.input(i) as u64).wrapping_sub(a[i]).wrapping_sub(b[i]))
                .collect();
            let shares = [a, b, c];
            for (i, w) in workers.iter_mut().enumerate() {
                w.import(key, &shares[i], &shares[(i + 1) % 3]).unwrap();
            }
        }
        workers
    };
    (make(fresh()), make(fresh()))
}

#[test]
fn authenticated_non_square_layouts_borrow_offsets_match_independent_ring_macs_and_wrong_source_rejects(
) {
    let f = Fixture::new(false, 3);
    let other = Fixture::new(false, 4);
    let registry = f.registry();
    let other_registry = other.registry();
    let mut outside = 2 * size_of::<Fixture>()
        + 2 * size_of::<SourceRegistry<'_>>()
        + pllm_core::streamed_matrix::READ_BUFFER_BYTES
        + size_of::<[[u64; 8]; 2]>()
        + size_of::<[[u64; 6]; 2]>()
        + 2 * size_of::<Secret>()
        + size_of::<Result<Secret, Error>>()
        + 12 * size_of::<u64>();
    for fixture in [&f, &other] {
        outside += size_of::<Manifest>()
            + fixture.manifest.schema.capacity()
            + fixture.manifest.config.capacity()
            + source_registry::semantic_metadata_bytes(&fixture.full).unwrap()
            + source_registry::semantic_metadata_bytes(&fixture.prefix).unwrap()
            + source_registry::source_metadata_bytes(
                &fixture.manifest.tensors,
                &fixture.manifest.norms,
            )
            .unwrap();
    }
    let public = authenticated(&f, &registry, outside);
    let wrong = authenticated(
        &other,
        &other_registry,
        outside + public.resident_metadata_bytes().unwrap(),
    );
    assert!(matches!(
        wrong.check_registry(&registry),
        Err(Error::Binding)
    ));
    drop(wrong);
    let storage = [
        [9u64, 3, u64::MAX - 7, 11, 17, 23, 31, 999],
        [7u64, 5, 13, u64::MAX - 11, 19, 29, 37, 888],
    ];
    for transposed in [false, true] {
        let (inner, columns) = if transposed { (3, 2) } else { (2, 3) };
        let inputs = [&storage[0][1..1 + 2 * inner], &storage[1][1..1 + 2 * inner]];
        let output = crate::replicated_decoder::public_source::multiply_components(
            public.file("lm_head.weight").unwrap(),
            inputs,
            2,
            inner,
            columns,
            transposed,
        )
        .unwrap();
        let mut expected = [[0u64; 6]; 2];
        for component in 0..2 {
            for row in 0..2 {
                for col in 0..columns {
                    for k in 0..inner {
                        let index = if transposed {
                            k * columns + col
                        } else {
                            col * inner + k
                        };
                        expected[component][row * columns + col] =
                            expected[component][row * columns + col].wrapping_add(
                                inputs[component][row * inner + k].wrapping_mul(coefficient(
                                    "lm_head.weight",
                                    index,
                                )
                                    as u64),
                            );
                    }
                }
            }
        }
        assert_eq!(output.first.as_slice(), &expected[0][..2 * columns]);
        assert_eq!(output.second.as_slice(), &expected[1][..2 * columns]);
    }
}

#[test]
fn authenticated_projection_head_and_per_token_lookup_match_full_wire_components_prss_and_files() {
    for (lookup, vocabulary) in [(false, 3), (false, 129), (true, 3), (true, 129)] {
        let f = Fixture::new(lookup, vocabulary);
        let registry = f.registry();
        let (imports, input) = f.imports();
        let frozen = f.freeze(&imports, &input);
        drop(input);
        let program = f.original(&frozen);
        let (a, b) = pair(&f, &registry, &frozen, program);
        custody::with_sources(
            a,
            b,
            &frozen,
            Some(&registry),
            MAX_PROGRAM_BYTES,
            |mut a, mut b, q| {
                let files = b.each_ref().map(|w| q.source_backing(w, 0).unwrap());
                if !lookup {
                    assert!(q.has_packed_offsets());
                }
                assert!(super::execution_tests::drive(&mut a, &mut b, q) > 0);
                for i in 0..3 {
                    q.assert_same_state(&a[i], &b[i]);
                    assert_eq!(q.source_backing(&b[i], 0).unwrap(), files[i]);
                }
                for (a, b) in a.into_iter().zip(b) {
                    assert_eq!(a.finish_frame()?.as_slice(), b.finish_frame()?.as_slice());
                }
                Ok(())
            },
        )
        .unwrap();
    }
}

#[test]
fn source_transition_exact_cap_and_one_short_precede_layout_or_dispatch() {
    let f = Fixture::new(false, 3);
    let registry = f.registry();
    let (imports, input) = f.imports();
    let frozen = f.freeze(&imports, &input);
    drop(input);
    let p = f.original(&frozen);
    let (a, b) = pair(&f, &registry, &frozen, p.clone());
    let exact = custody::source_quote(&a, &b, &frozen, &registry).unwrap();
    assert!(matches!(
        custody::with_sources::<()>(a, b, &frozen, Some(&registry), exact - 1, |_, _, _| panic!(
            "short source admission reached storage"
        )),
        Err(Error::Bounds)
    ));
    let (a, b) = pair(&f, &registry, &frozen, p);
    custody::with_sources(a, b, &frozen, Some(&registry), exact, |_, _, _| Ok(())).unwrap();
}

#[test]
fn equal_binding_foreign_registry_and_native_owners_never_substitute_for_original_custody() {
    let f = Fixture::new(true, 129);
    let foreign = Fixture::new(true, 129);
    let registry = f.registry();
    let foreign_registry = foreign.registry();
    assert_eq!(registry.binding(), foreign_registry.binding());
    let (imports, input) = f.imports();
    let frozen = f.freeze(&imports, &input);
    let other = f.freeze(&imports, &input);
    drop(input);
    let p = f.original(&frozen);
    for (region, registry_arg) in [(&frozen, &foreign_registry), (&other, &registry)] {
        let (a, b) = pair(&f, &registry, &frozen, p.clone());
        assert!(matches!(
            custody::with_sources(
                a,
                b,
                region,
                Some(registry_arg),
                MAX_PROGRAM_BYTES,
                |_, _, _| -> Result<(), Error> { panic!("foreign source/native owner qualified") }
            ),
            Err(Error::Binding)
        ));
    }
}

#[test]
fn malformed_original_matrix_geometry_and_tampered_authenticated_import_reject() {
    let f = Fixture::new(false, 3);
    let registry = f.registry();
    let (imports, input) = f.imports();
    let frozen = f.freeze(&imports, &input);
    drop(input);
    let mut p = f.original(&frozen);
    // Negative-only mutation of genuine original custody. Same digest/intervals
    // cannot bless wrong matrix geometry; never used as a positive issuer.
    let node = Arc::get_mut(&mut p)
        .unwrap()
        .nodes
        .iter_mut()
        .find(|n| matches!(n.code, Code::PublicProduct { .. }))
        .unwrap();
    let Code::PublicProduct { inner, .. } = &mut node.code else {
        unreachable!()
    };
    *inner += 1;
    let (a, b) = pair(&f, &registry, &frozen, p);
    assert!(matches!(
        custody::with_sources(
            a,
            b,
            &frozen,
            Some(&registry),
            MAX_PROGRAM_BYTES,
            |_, _, _| -> Result<(), Error> { panic!("wrong shape qualified") }
        ),
        Err(Error::Binding)
    ));
    let mut p = f.original(&frozen);
    let node = Arc::get_mut(&mut p)
        .unwrap()
        .nodes
        .iter_mut()
        .find(|n| matches!(n.code, Code::Gather(_)))
        .unwrap();
    // Negative-only public coefficient gather. Qualification must stop it before
    // any word read; a scalar-file fallback is forbidden even for tiny sources.
    node.code = Code::Gather(std::iter::repeat_n((0, 0), node.lanes).collect());
    let (a, b) = pair(&f, &registry, &frozen, p);
    assert!(matches!(
        custody::with_sources::<()>(
            a,
            b,
            &frozen,
            Some(&registry),
            MAX_PROGRAM_BYTES,
            |_, _, _| panic!("per-word coefficient path qualified")
        ),
        Err(Error::Binding)
    ));
    let retained = frozen.resources().peak_bytes
        + size_of::<Manifest>()
        + f.manifest.config.capacity()
        + f.manifest.schema.capacity()
        + source_registry::semantic_metadata_bytes(&f.full).unwrap()
        + source_registry::semantic_metadata_bytes(&f.prefix).unwrap();
    let quote = PublicSourceImport::construction_quote(&registry, retained).unwrap();
    assert!(matches!(
        PublicSourceImport::new_checked(&registry, retained, quote.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    let mut importer =
        PublicSourceImport::new_checked(&registry, retained, quote.peak_bytes).unwrap();
    let key = f.manifest.tensors.keys().next().unwrap();
    assert!(importer
        .append(key, 0, &f32::INFINITY.to_le_bytes())
        .is_err());
    assert!(importer.append(key, 0, &0f32.to_le_bytes()).is_err());
    // In-range but wrong bytes fail the real digest/file writer at record end,
    // not merely the float/range validator. Importer then remains burned.
    let mut importer =
        PublicSourceImport::new_checked(&registry, retained, quote.peak_bytes).unwrap();
    let count = f.manifest.tensors[key].shape.iter().product::<usize>();
    for i in 0..count - 1 {
        importer.append(key, i, &0f32.to_le_bytes()).unwrap();
    }
    assert!(importer
        .append(key, count - 1, &0f32.to_le_bytes())
        .is_err());
    assert!(importer.finish().is_err());
}

#[test]
fn source_cancellation_partial_tail_error_and_unwind_erase_secrets_before_original_files_drop() {
    let f = Fixture::new(true, 129);
    let registry = f.registry();
    let (imports, input) = f.imports();
    let frozen = f.freeze(&imports, &input);
    drop(input);
    let p = f.original(&frozen);
    let (a, b) = pair(&f, &registry, &frozen, p.clone());
    custody::with_sources(
        a,
        b,
        &frozen,
        Some(&registry),
        MAX_PROGRAM_BYTES,
        |mut a, mut b, q| {
            assert!(super::execution_tests::drive_with_stop(&mut a, &mut b, q, true) > 0);
            assert!(b.iter().any(|w| w.loop_test_state().0 == 128));
            for i in 0..3 {
                q.source_backing(&b[i], 0)?;
                a[i].cancel();
                b[i].cancel();
                q.assert_same_state(&a[i], &b[i]);
                assert!(matches!(q.advance(&mut b[i], None), Err(Error::Retired)));
            }
            Ok(())
        },
    )
    .unwrap();
    let (a, b) = pair(&f, &registry, &frozen, p.clone());
    custody::with_sources(
        a,
        b,
        &frozen,
        Some(&registry),
        MAX_PROGRAM_BYTES,
        |mut a, mut b, q| {
            let x = a[0].advance()?;
            let y = q.advance(&mut b[0], None)?;
            assert_eq!(x.len(), y.len());
            for (x, y) in x.iter().zip(&y) {
                assert_eq!(x.0, y.0);
                assert_eq!(x.1.as_bytes(), y.1.as_bytes());
            }
            let x = a[0].receive(&[0; 3], Party::P1).err().unwrap();
            let y = b[0].receive(&[0; 3], Party::P1).err().unwrap();
            assert_eq!(x, y);
            q.assert_same_state(&a[0], &b[0]);
            Ok(())
        },
    )
    .unwrap();
    let mut weak = None;
    let (a, b) = pair(&f, &registry, &frozen, p.clone());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<(), Error> = custody::with_sources(
            a,
            b,
            &frozen,
            Some(&registry),
            MAX_PROGRAM_BYTES,
            |_, _, q| {
                weak = Some(q.weak_layout());
                panic!("source custody unwind")
            },
        );
    }));
    assert!(panic.is_err());
    assert!(weak.unwrap().upgrade().is_none());
    assert_eq!(Arc::strong_count(&p), 1);
}
