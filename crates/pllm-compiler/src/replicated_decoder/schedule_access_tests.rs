//! Borrowed-reader parity only; no compact runtime/source admission is issued.
use super::super::*;
use super::*;
use std::collections::BTreeMap;

fn fixture(rows: usize, width: usize) -> (DecoderGraph, BTreeMap<String, SourceTensor>) {
    let config = serde_json::to_vec(&serde_json::json!({
        "model_type":"llama","hidden_size":2,"intermediate_size":width,
        "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":3,"tie_word_embeddings":false,"hidden_act":"silu",
        "attention_bias":false,"max_position_embeddings":64,"rope_theta":10000,
        "rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true
    }))
    .unwrap();
    let graph = lower_with_weights(&config, rows, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let sources = Program::source_shapes_with_weights(&config, rows, 1, WeightMode::Public)
        .unwrap()
        .into_iter()
        .map(|(key, shape)| {
            let public = key.contains("norm");
            let n = shape.iter().product();
            let value = if public { ONE } else { ONE / 64 };
            let mut hash = Sha256::new();
            for _ in 0..n {
                hash.update(value.to_le_bytes());
            }
            let row = (shape.len() == 2).then(|| shape[1] as u64 * (ONE / 64) as u64);
            let source = SourceTensor {
                lower: if public { ONE } else { -ONE / 64 },
                upper: value,
                row_l1: row,
                row_squared_norm: row
                    .map(|_| shape[1] as u128 * (ONE / 64) as u128 * (ONE / 64) as u128),
                row_l1_squared_sum: row.map(|r| shape[0] as u128 * u128::from(r).pow(2)),
                shape,
                public_q18: public.then(|| vec![ONE; n]),
                digest: hash.finalize().into(),
            };
            (key, source)
        })
        .collect();
    (graph, sources)
}
fn imports(sources: &BTreeMap<String, SourceTensor>, rows: usize) -> (Builder, View) {
    let mut b = Builder {
        public_weights: true,
        ..Builder::default()
    };
    for (key, s) in sources.iter().filter(|(_, s)| s.public_q18.is_none()) {
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
    let id = b
        .push(
            Code::Input("hidden".into()),
            2 * rows,
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
        .unwrap();
    let v = b.view(id, vec![1, rows, 2]);
    (b, v)
}
fn validated<'a>(
    g: &DecoderGraph,
    s: &'a BTreeMap<String, SourceTensor>,
    n: &'a SourceNorms,
) -> ValidatedSourceDescriptors<'a> {
    let shapes = s
        .iter()
        .map(|(k, s)| (k.clone(), s.shape.clone()))
        .collect();
    validate_source_records(
        g,
        &shapes,
        s,
        WeightMode::Public,
        ExecutionPolicy::default(),
        n,
    )
    .unwrap()
}

// Independent literal expanded owners: no reconstruction from the reader or
// compact descriptors, no hash/ID equality used as execution authority.
fn expanded(
    g: &DecoderGraph,
    s: &BTreeMap<String, SourceTensor>,
    rows: usize,
    start: usize,
) -> Program {
    let (mut b, input) = imports(s, rows);
    let count = b.nodes.len();
    let mut values = BTreeMap::from([(g.operations[start].inputs[0].clone(), input.clone())]);
    let residual = g.operations[start + 5]
        .inputs
        .iter()
        .find(|name| **name != g.operations[start + 4].id)
        .unwrap();
    values.insert(residual.clone(), input);
    let mut output = None;
    for op in &g.operations[start..start + 6] {
        let a = &values[&op.inputs[0]];
        let shape = op.output_shape.iter().map(|n| *n as usize).collect();
        let v = match op.operator {
            Op::Linear => {
                let key = op.attributes["weight"].as_str().unwrap();
                let owner = s
                    .iter()
                    .filter(|(_, s)| s.public_q18.is_none())
                    .position(|(k, _)| k == key)
                    .unwrap();
                let source = SourceView {
                    owner,
                    shape: s[key].shape.clone(),
                };
                let mut v = b
                    .matrix(a, MatrixView::source(&source, false).unwrap(), true)
                    .unwrap();
                v.shape = shape;
                b.projection_energies(a, &v, &source, None).unwrap();
                v
            }
            Op::Silu => {
                let id = b.gather(&a.words).unwrap();
                let bounds = b.nodes[id].bounds;
                let out = b
                    .tensor_loop(LoopInput::Slice(id), a.words.len(), |n| {
                        tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
                    })
                    .unwrap();
                b.view(out, shape)
            }
            Op::Multiply => View {
                words: b
                    .binary(&a.words, &values[&op.inputs[1]].words, true)
                    .unwrap(),
                shape,
            },
            Op::ResidualAdd => View {
                words: b.residual(a, &values[&op.inputs[1]]).unwrap(),
                shape,
            },
            _ => panic!("fixture changed original family order"),
        };
        output = Some(v.clone());
        values.insert(op.id.clone(), v);
    }
    b.finish(
        count,
        output.unwrap().words.iter().collect(),
        &mut AdmissionSite::default(),
        |h| {
            h.update(b"schedule-access.literal-fixture");
            Ok(())
        },
        || source_binding(s),
    )
    .unwrap()
}

fn read_canonical(schedule: PrivateScheduleRef<'_>) -> [u8; 32] {
    let mut h = Sha256::new();
    for ordinal in 0..schedule.len() {
        let node = schedule.node(ordinal).unwrap();
        hash_node_header(&mut h, node.lanes(), node.bounds());
        match node.kind() {
            Kind::Input => {
                let k = node.input_name().unwrap();
                h.update([0]);
                h.update((k.len() as u64).to_le_bytes());
                h.update(k);
            }
            Kind::PublicInput => {
                let (k, d) = node.public_input().unwrap();
                h.update([5]);
                h.update((k.len() as u64).to_le_bytes());
                h.update(k);
                h.update(d);
            }
            Kind::Gather | Kind::SourceGather => {
                hash_gather(&mut h, node.lane_refs().unwrap().map(Result::unwrap))
            }
            Kind::PublicProduct => {
                let (i, w, r, n, c, t) = node.public_product().unwrap();
                hash_public_product(&mut h, *i, *w, *r, *n, *c, *t);
            }
            Kind::TensorLoop => {
                let (p, t) = node.loop_programs().unwrap();
                hash_tensor_loop(&mut h, *node.loop_input().unwrap(), p, t);
            }
            Kind::Product => {
                let (l, r, s, z, n) = node.product().unwrap();
                hash_product(&mut h, *l, *r, *s, *z, *n);
            }
            Kind::Affine => {
                let (l, r, c) = node.affine().unwrap();
                hash_affine(&mut h, *l, *r, *c);
            }
            _ => panic!("unexpected literal fixture recipe"),
        }
    }
    h.finalize().into()
}

#[test]
fn independent_expanded_and_frozen_reads_match_canonical_dependencies_full_tail_and_exports() {
    for rows in [1, 2] {
        for width in [128, 129] {
            let (graph, sources) = fixture(rows, width);
            let norms = SourceNorms::new();
            let start = graph
                .operations
                .iter()
                .position(|op| op.id.ends_with(".gate_proj"))
                .unwrap();
            let original = expanded(&graph, &sources, rows, start);
            let (imports, input) = imports(&sources, rows);
            let mut graph_owner = compact_lowering::CompactGraph::new(
                &imports,
                validated(&graph, &sources, &norms),
                &graph,
                start,
            )
            .unwrap();
            graph_owner
                .bind_import(&graph.operations[start].inputs[0], &input)
                .unwrap();
            let residual = graph.operations[start + 5]
                .inputs
                .iter()
                .find(|name| **name != graph.operations[start + 4].id)
                .unwrap();
            graph_owner.bind_import(residual, &input).unwrap();
            let exports = (0..6).map(|_| graph_owner.lower_next().unwrap()).collect();
            let frozen = graph_owner.freeze(exports, MAX_PROGRAM_BYTES).unwrap();
            let recipes = frozen.borrow_schedule(MAX_PROGRAM_BYTES).unwrap();
            let expanded = original.schedule_access();
            let compact = PrivateScheduleRef::frozen(&recipes);
            let attempts = super::super::super::prepared_tensor_map::native_attempts();
            assert_eq!(expanded.len(), compact.len());
            assert_eq!(read_canonical(expanded), read_canonical(compact));
            for id in 0..expanded.len() {
                let a = expanded.node(id).unwrap();
                let b = compact.node(id).unwrap();
                assert_eq!(a.kind(), b.kind());
                assert_eq!(a.lanes(), b.lanes());
                assert_eq!(a.bounds(), b.bounds());
                assert_eq!(
                    a.inputs().collect::<Result<Vec<_>, _>>().unwrap(),
                    b.inputs().collect::<Result<Vec<_>, _>>().unwrap()
                );
                if a.kind() == Kind::TensorLoop {
                    let (full, tail) = b.loop_programs().unwrap();
                    let strong = Arc::strong_count(full);
                    assert!(Arc::ptr_eq(b.loop_tile(0).unwrap(), full));
                    let remainder = b.lanes() % full.lanes();
                    if remainder != 0 {
                        assert!(Arc::ptr_eq(
                            b.loop_tile(b.lanes() - remainder).unwrap(),
                            tail.unwrap()
                        ));
                    }
                    let owned = frozen.descriptor_census().enumerate().any(|(slot, _)| {
                        frozen
                            .prepared(slot)
                            .is_ok_and(|prepared| prepared.programs().any(|p| Arc::ptr_eq(p, full)))
                    });
                    assert!(owned);
                    assert_eq!(Arc::strong_count(full), strong);
                    assert!(matches!(b.loop_tile(b.lanes()), Err(Error::Phase)));
                    assert!(matches!(b.loop_tile(usize::MAX), Err(Error::Bounds)));
                }
            }
            let output = frozen.exports().last().unwrap();
            assert_eq!(
                FrozenExportRefs::new(&frozen, output)
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
                original.output_coordinates()
            );
            assert!(matches!(compact.value_slot(0), Err(Error::Binding)));
            assert!(matches!(compact.retirement(0), Err(Error::Binding)));
            assert!(matches!(compact.outputs(), Err(Error::Binding)));
            assert!(matches!(compact.source_binding(), Err(Error::Binding)));
            assert!(matches!(compact.input_count(), Err(Error::Binding)));
            assert!(matches!(compact.trie_owner(0), Err(Error::Binding)));
            assert!(matches!(compact.selection_owner(0), Err(Error::Binding)));
            for source in [expanded, compact] {
                assert!(matches!(source.node(source.len()), Err(Error::Binding)));
                assert!(matches!(source.node(usize::MAX), Err(Error::Binding)));
            }
            assert_eq!(
                super::super::super::prepared_tensor_map::native_attempts(),
                attempts
            );
        }
    }
}

#[test]
fn expanded_import_source_outputs_retirement_and_control_preflight_are_borrowed_exactly() {
    let (program, _, _, _) =
        prepared_rescale18_worker_fixture(129, Interval::new(-ONE, ONE).unwrap()).unwrap();
    let reader = program.schedule_access();
    assert_eq!(reader.input_count().unwrap(), program.inputs);
    assert_eq!(reader.public_input_count().unwrap(), 0);
    assert!(reader.public_prefix(0).unwrap());
    assert!(!reader.public_prefix(1).unwrap());
    assert!(std::ptr::eq(
        reader.outputs().unwrap(),
        program.outputs.as_slice()
    ));
    for id in 0..reader.len() {
        assert_eq!(reader.value_slot(id).unwrap(), id);
        assert!(std::ptr::eq(
            reader.retirement(id).unwrap(),
            program.retire[id].as_slice()
        ));
    }
    let attempts = super::super::super::prepared_tensor_map::native_attempts();
    assert_eq!(
        control_preflight(97, 97 + CONTROL_BYTES).unwrap(),
        97 + CONTROL_BYTES
    );
    assert!(matches!(
        control_preflight(97, 96 + CONTROL_BYTES),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        control_preflight(usize::MAX, usize::MAX),
        Err(Error::Bounds)
    ));
    assert_eq!(
        super::super::super::prepared_tensor_map::native_attempts(),
        attempts
    );
}

#[test]
fn lazy_coordinate_errors_and_original_missing_tail_error_order() {
    let mut factory = tensor::Builder::new(128, Interval::new(-ONE, ONE).unwrap()).unwrap();
    let scaled = factory.rescale(factory.input(), 18).unwrap();
    let native = Arc::new(factory.finish(scaled).unwrap());
    let node = ReadNode(BorrowedNode {
        lanes: 129,
        bounds: Interval::new(-1, 1).unwrap(),
        recipe: BorrowedRecipe::TensorLoop {
            input: LoopInput::Slice(0),
            program: &native,
            tail: None,
        },
    });
    assert!(Arc::ptr_eq(node.loop_tile(0).unwrap(), &native));
    assert!(matches!(node.loop_tile(128), Err(Error::Shape)));
    assert!(matches!(node.loop_tile(129), Err(Error::Phase)));
    assert!(matches!(node.loop_tile(130), Err(Error::Bounds)));
    let source = SourceGather {
        owner: 7,
        offset: 2,
        rows: 2,
        columns: 3,
        row_stride: 7,
        column_stride: 2,
    };
    let node = ReadNode(BorrowedNode {
        lanes: 6,
        bounds: Interval::new(-1, 1).unwrap(),
        recipe: BorrowedRecipe::SourceGather(source),
    });
    assert_eq!(
        node.lane_refs()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        vec![(7, 2), (7, 4), (7, 6), (7, 9), (7, 11), (7, 13)]
    );
    assert_eq!(
        node.inputs().collect::<Result<Vec<_>, _>>().unwrap(),
        vec![7]
    );
    let invalid = ReadNode(BorrowedNode {
        recipe: BorrowedRecipe::SourceGather(SourceGather {
            columns: 0,
            ..source
        }),
        ..node.0
    });
    assert!(matches!(invalid.lane_refs(), Err(Error::Shape)));
    let overflow = ReadNode(BorrowedNode {
        recipe: BorrowedRecipe::SourceGather(SourceGather {
            offset: usize::MAX,
            ..source
        }),
        ..node.0
    });
    let mut refs = overflow.lane_refs().unwrap();
    assert_eq!(refs.next().unwrap().unwrap(), (7, usize::MAX));
    assert!(matches!(refs.next(), Some(Err(Error::Bounds))));
    assert!(refs.next().is_none());
}

#[test]
fn sealed_frozen_iota_is_recognized_but_not_reissued_as_executable_public_values() {
    let (graph, sources) = fixture(1, 129);
    let norms = SourceNorms::new();
    let start = graph
        .operations
        .iter()
        .position(|op| op.operator == Op::GreedyTokenSelection)
        .unwrap();
    let (mut imports, _) = imports(&sources, 1);
    let id = imports
        .push(
            Code::Input("logits".into()),
            3,
            Interval::new(-ONE / 4, ONE / 4).unwrap(),
        )
        .unwrap();
    let logits = imports.view(id, vec![1, 3]);
    let mut compact = compact_lowering::CompactGraph::new(
        &imports,
        validated(&graph, &sources, &norms),
        &graph,
        start,
    )
    .unwrap();
    compact
        .bind_import(&graph.operations[start].inputs[0], &logits)
        .unwrap();
    let output = compact.lower_next().unwrap();
    let frozen = compact.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap();
    let recipes = frozen.borrow_schedule(MAX_PROGRAM_BYTES).unwrap();
    let reader = PrivateScheduleRef::frozen(&recipes);
    let iota = (0..reader.len())
        .map(|id| reader.node(id).unwrap())
        .find(|n| n.kind() == Kind::GeneratedPublicValues)
        .unwrap();
    assert_eq!(iota.lanes(), 3);
    assert_eq!(iota.bounds(), Interval::new(0, 2).unwrap());
    assert!(iota.inputs().next().is_none());
    assert!(matches!(iota.public_values(), Err(Error::Binding)));
    // Original expanded iota remains readable, not regenerated from a semantic
    // label, node geometry or an unqualified compact completion claim.
    let mut original = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let id = original
        .push(Code::Input("logits".into()), 3, logits_bounds())
        .unwrap();
    let input = original.view(id, vec![1, 3]);
    original.argmax_tree(&input).unwrap();
    let literal = original
        .nodes
        .iter()
        .find_map(|node| match &node.code {
            Code::PublicValues(values) => Some(values),
            _ => None,
        })
        .unwrap();
    assert_eq!(literal, &[0, 1, 2]);
    let loan = ReadNode(BorrowedNode::from(
        original
            .nodes
            .iter()
            .find(|node| matches!(node.code, Code::PublicValues(_)))
            .unwrap(),
    ));
    assert!(std::ptr::eq(
        loan.public_values().unwrap(),
        literal.as_slice()
    ));
}
fn logits_bounds() -> Interval {
    Interval::new(-ONE / 4, ONE / 4).unwrap()
}

#[test]
fn default_expanded_worker_keeps_original_full_tail_numeric_results_and_burns() {
    for lanes in [1, 128, 129, 257] {
        let (program, _, _, _) =
            prepared_rescale18_worker_fixture(lanes, Interval::new(-ONE, ONE).unwrap()).unwrap();
        // Exact integer multiples avoid probabilistic floor-error assumptions.
        let clear: Vec<_> = (0..lanes).map(|i| (i as i64 % 3 - 1) * ONE).collect();
        let expected: Vec<_> = clear.iter().map(|x| (x / ONE) as u64).collect();
        let input = BTreeMap::from([("local".to_owned(), clear)]);
        let workers = super::super::super::tests::workers(program.clone(), &input);
        assert_eq!(super::super::super::tests::execute(workers), expected);
        let mut workers = super::super::super::tests::workers(program, &input);
        for worker in &mut workers {
            // Already-imported source rejects another import before dispatch.
            assert!(matches!(
                worker.import("local", &[0], &[0]),
                Err(Error::Phase)
            ));
            assert!(worker.retired());
            assert!(matches!(worker.advance(), Err(Error::Retired)));
        }
    }
}
