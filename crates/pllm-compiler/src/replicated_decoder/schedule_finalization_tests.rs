//! Independent legacy-style encoding/retirement oracle. These are finalizer
//! metadata tests, not execution/source-provenance or complete-decoder parity.
use super::*;

fn with_generated_owner(test: impl FnOnce(&compact_lowering::frozen_region::FrozenRegion<'_>)) {
    let config = br#"{"model_type":"llama","hidden_size":2,"intermediate_size":2,
        "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":3,"tie_word_embeddings":false,"hidden_act":"silu",
        "attention_bias":false,"max_position_embeddings":64,"rope_theta":10000,
        "rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}"#;
    let full = lower_with_weights(config, 1, 1, WeightMode::Public)
        .unwrap()
        .prefill;
    let sources: BTreeMap<_, _> =
        Program::source_shapes_with_weights(config, 1, 1, WeightMode::Public)
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
                (
                    key,
                    SourceTensor {
                        shape,
                        lower: if public { ONE } else { -ONE / 64 },
                        upper: value,
                        public_q18: public.then(|| vec![ONE; n]),
                        digest: hash.finalize().into(),
                        row_l1: None,
                        row_squared_norm: None,
                        row_l1_squared_sum: None,
                    },
                )
            })
            .collect();
    let norms = SourceNorms::new();
    let shapes = sources
        .iter()
        .map(|(key, source)| (key.clone(), source.shape.clone()))
        .collect();
    let validated = validate_source_records(
        &full,
        &shapes,
        &sources,
        WeightMode::Public,
        ExecutionPolicy::default(),
        &norms,
    )
    .unwrap();
    let mut imported = Builder {
        public_weights: true,
        ..Builder::default()
    };
    for (key, source) in sources
        .iter()
        .filter(|(_, source)| source.public_q18.is_none())
    {
        imported
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
    let input = imported
        .push(Code::Input("local.logits".into()), 2, bounds())
        .unwrap();
    let input = imported.view(input, vec![1, 1, 2]);
    let mut semantic = full.clone();
    semantic
        .operations
        .retain(|op| op.operator == Op::GreedyTokenSelection);
    semantic.output = semantic.operations[0].id.clone();
    let mut compact =
        compact_lowering::CompactGraph::new(&imported, validated, &semantic, 0).unwrap();
    compact
        .bind_import(&semantic.operations[0].inputs[0], &input)
        .unwrap();
    let output = compact.lower_next().unwrap();
    let frozen = compact.freeze(vec![output], MAX_PROGRAM_BYTES).unwrap();
    test(&frozen);
}

struct GeneratedSyntaxSchedule<'a> {
    nodes: &'a [Node],
    generated: GeneratedPublicValues<'a>,
    lanes: usize,
    bounds: Interval,
}

impl BorrowedOrderedNodes for GeneratedSyntaxSchedule<'_> {
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        let mut node = BorrowedNode::from(self.nodes.get(ordinal).ok_or(Error::Binding)?);
        if ordinal == 8 {
            node.lanes = self.lanes;
            node.bounds = self.bounds;
            node.recipe = BorrowedRecipe::GeneratedPublicValues(self.generated);
        }
        Ok(node)
    }
}

#[test]
fn sealed_generated_iota_matches_original_public_values_encoding_work_and_retirement() {
    with_generated_owner(|owner| {
        let mut nodes = fixture();
        nodes[8].code = Code::PublicValues(vec![0, 1]);
        nodes[8].bounds = Interval::new(0, 1).unwrap();
        let schedule = GeneratedSyntaxSchedule {
            nodes: &nodes,
            generated: GeneratedPublicValues::argmax_indices(owner, 0).unwrap(),
            lanes: 2,
            bounds: Interval::new(0, 1).unwrap(),
        };
        let selected = [(8, 1), (11, 0), (8, 0)];
        let expected = legacy(&nodes, &selected, &prefix());
        reset_factories();
        let quote = quote(&schedule, &selected, 71).unwrap();
        assert_no_factories();
        assert_eq!(quote.work, expected.2);
        assert!(matches!(
            finalize(&schedule, &selected, &prefix(), 71, quote.peak_bytes - 1),
            Err(Error::Bounds)
        ));
        assert_no_factories();
        let actual = finalize(&schedule, &selected, &prefix(), 71, quote.peak_bytes).unwrap();
        assert_eq!(actual.digest, expected.0);
        assert_eq!(actual.retire, expected.1);
        assert_eq!(actual.work, expected.2);
        assert!(actual.retire.iter().all(|bucket| !bucket.contains(&8)));
        assert_eq!(FINALIZATION_FACTORY_ENTRIES.with(|n| n.get()), 1);
    });
}

#[test]
fn generated_iota_rejects_wrong_width_bounds_and_unbounded_geometry_before_factories() {
    with_generated_owner(|owner| {
        let nodes = fixture();
        reset_factories();
        for (generated, lanes, bounds) in [
            (0, 0, Interval::new(0, 0).unwrap()),
            (2, 1, Interval::new(0, 1).unwrap()),
            (2, 2, Interval::new(-1, 1).unwrap()),
            (MAX_VALUES + 1, MAX_VALUES + 1, Interval::new(0, 1).unwrap()),
            (usize::MAX, usize::MAX, Interval::new(0, 1).unwrap()),
        ] {
            let schedule = GeneratedSyntaxSchedule {
                nodes: &nodes,
                generated: public_constants::invalid_geometry_fixture(owner, generated),
                lanes,
                bounds,
            };
            assert!(matches!(quote(&schedule, &[], 0), Err(Error::Shape)));
            assert!(matches!(
                finalize(&schedule, &[], &prefix(), 0, MAX_PROGRAM_BYTES),
                Err(Error::Shape)
            ));
        }
        assert_no_factories();
    });
}

fn reset_factories() {
    FINALIZATION_FACTORY_ENTRIES.with(|n| n.set(0));
    FINALIZATION_RESERVATIONS.with(|n| n.set(0));
}

fn assert_no_factories() {
    assert_eq!(FINALIZATION_FACTORY_ENTRIES.with(|n| n.get()), 0);
    assert_eq!(FINALIZATION_RESERVATIONS.with(|n| n.get()), 0);
}

fn bounds() -> Interval {
    Interval::new(-4, 4).unwrap()
}

fn rescale(lanes: usize) -> Arc<tensor::Program> {
    let mut b = tensor::Builder::new(lanes, bounds()).unwrap();
    let output = b.rescale(b.input(), 18).unwrap();
    Arc::new(b.finish(output).unwrap())
}

fn node(code: Code, lanes: usize) -> Node {
    Node {
        code,
        lanes,
        bounds: bounds(),
    }
}

/// Every original canonical code tag, with real immutable native plan/template
/// owners. Public trie/proof route commitments below are only syntax fixtures;
/// finalization does NOT confer source, token-domain or row-proof authority.
fn fixture() -> Vec<Node> {
    let tokens = Interval::new(0, 1).unwrap();
    let full = rescale(2);
    let tail = rescale(1);
    let checked = Arc::new(
        tensor::Program::checked_range(2, bounds(), Interval::new(-1, 1).unwrap(), 0).unwrap(),
    );
    let lookup = Arc::new(runtime::lookup::Plan::new(1, 2, 2, tokens).unwrap());
    let selection = Arc::new(runtime::selection::Plan::new(2, bounds()).unwrap());
    let trie = Arc::new(
        runtime::trie::Plan::new(
            1,
            2,
            2,
            2,
            tokens,
            &[runtime::trie::DomainOrigin::ValidatedInitialIds([8; 32])],
            runtime::trie::SourceBinding {
                owner: 1,
                source_commitment: [7; 32],
                file_digest: [6; 32],
            },
        )
        .unwrap(),
    );
    let proof = Arc::new(
        puma::multiply::norm::RowEnergyProof::new(
            2,
            InputBounds {
                left: bounds(),
                right: bounds(),
            },
            32,
            32,
        )
        .unwrap(),
    );
    vec![
        node(Code::Input("input".into()), 4),
        node(
            Code::PublicInput {
                key: "weight".into(),
                digest: [6; 32],
                row_l1: Some(8),
                row_squared_norm: Some(32),
                row_l1_squared_sum: Some(128),
            },
            4,
        ),
        Node {
            code: Code::Input("tokens".into()),
            lanes: 1,
            bounds: tokens,
        },
        node(
            Code::Gather([(0, 3), (0, 0), (0, 3)].into_iter().collect()),
            3,
        ),
        node(
            Code::SourceGather(SourceGather {
                owner: 1,
                offset: 0,
                rows: 1,
                columns: 3,
                row_stride: 3,
                column_stride: 1,
            }),
            3,
        ),
        node(
            Code::Affine {
                left: 3,
                right: Some(4),
                coefficients: affine(1, -1, 2),
            },
            3,
        ),
        node(
            Code::Product {
                left: 3,
                right: 4,
                shape: None,
                rescale: true,
                norm: None,
            },
            3,
        ),
        node(
            Code::PublicProduct {
                input: 0,
                weight: 1,
                rows: 2,
                inner: 2,
                columns: 2,
                transposed: true,
            },
            4,
        ),
        node(Code::PublicValues(vec![1, -1]), 2),
        node(
            Code::Tensor {
                input: 8,
                program: full.clone(),
            },
            2,
        ),
        node(
            Code::CheckedRange {
                input: 8,
                program: checked,
            },
            // Native checked owner concatenates two safe + two validity words.
            // The old syntax-only fixture omitted its validity half.
            4,
        ),
        node(
            Code::TensorLoop {
                input: LoopInput::Slice(3),
                program: full,
                tail: Some(tail),
            },
            3,
        ),
        node(
            Code::TensorLoop {
                input: LoopInput::TokenOffsets { owner: 2, index: 0 },
                program: Arc::new(
                    tensor::Program::equal_zero(2, Interval::new(-1, 1).unwrap()).unwrap(),
                ),
                tail: None,
            },
            2,
        ),
        node(
            Code::PublicLookup {
                input: 2,
                weight: 1,
                plan: lookup,
            },
            2,
        ),
        node(
            Code::Selection {
                input: 8,
                plan: selection,
            },
            1,
        ),
        node(
            Code::TrieLookup {
                input: 2,
                weight: 1,
                plan: trie,
            },
            2,
        ),
        node(
            Code::Product {
                left: 0,
                right: 1,
                shape: Some(MatrixShape::new(2, 2, 2).unwrap()),
                rescale: false,
                norm: Some(proof),
            },
            4,
        ),
    ]
}

/// Does not use production canonical helpers or dependency traversal. Hash bytes
/// and input last uses follow the old Builder::finish structure independently.
fn legacy(
    nodes: &[Node],
    selected: &[(usize, usize)],
    prefix: &Sha256,
) -> ([u8; 32], Vec<Vec<usize>>, WorkBudget) {
    fn word(hash: &mut Sha256, value: usize) {
        hash.update((value as u64).to_le_bytes());
    }
    let mut hash = prefix.clone();
    let mut last: Vec<_> = (0..nodes.len()).collect();
    for (ordinal, node) in nodes.iter().enumerate() {
        for input in node.inputs() {
            last[input] = ordinal;
        }
        word(&mut hash, node.lanes);
        hash.update(node.bounds.lower().to_le_bytes());
        hash.update(node.bounds.upper().to_le_bytes());
        match &node.code {
            Code::Input(key) => {
                hash.update([0]);
                word(&mut hash, key.len());
                hash.update(key);
            }
            Code::PublicInput { key, digest, .. } => {
                hash.update([5]);
                word(&mut hash, key.len());
                hash.update(key);
                hash.update(digest);
            }
            Code::PublicValues(values) => {
                hash.update([7]);
                for value in values {
                    hash.update(value.to_le_bytes());
                }
            }
            Code::Gather(items) => {
                hash.update([1]);
                for (owner, index) in items.iter() {
                    word(&mut hash, owner);
                    word(&mut hash, index);
                }
            }
            Code::SourceGather(view) => {
                hash.update([1]);
                for row in 0..view.rows {
                    for column in 0..view.columns {
                        word(&mut hash, view.owner);
                        word(
                            &mut hash,
                            view.offset + row * view.row_stride + column * view.column_stride,
                        );
                    }
                }
            }
            Code::Affine {
                left,
                right,
                coefficients: c,
            } => {
                hash.update([2]);
                word(&mut hash, *left);
                hash.update(right.map_or(u64::MAX, |v| v as u64).to_le_bytes());
                for value in [c.left, c.right, c.constant] {
                    hash.update(value.to_le_bytes());
                }
            }
            Code::Product {
                left,
                right,
                shape,
                rescale,
                norm,
            } => {
                hash.update([if norm.is_some() { 13 } else { 3 }, u8::from(*rescale)]);
                word(&mut hash, *left);
                word(&mut hash, *right);
                for value in shape.map_or([0, 0, 0], MatrixShape::dimensions) {
                    word(&mut hash, value);
                }
                if let Some(norm) = norm {
                    hash.update(norm.digest());
                }
            }
            Code::PublicProduct {
                input,
                weight,
                rows,
                inner,
                columns,
                transposed,
            } => {
                hash.update([6, u8::from(*transposed)]);
                for value in [*input, *weight, *rows, *inner, *columns] {
                    word(&mut hash, value);
                }
            }
            Code::Tensor { input, program } => {
                hash.update([4]);
                word(&mut hash, *input);
                hash.update(program.digest());
            }
            Code::CheckedRange { input, program } => {
                hash.update([9]);
                word(&mut hash, *input);
                hash.update(program.digest());
            }
            Code::TensorLoop {
                input,
                program,
                tail,
            } => {
                hash.update([8]);
                match input {
                    LoopInput::Slice(owner) => {
                        hash.update([0]);
                        word(&mut hash, *owner);
                    }
                    LoopInput::TokenOffsets { owner, index } => {
                        hash.update([1]);
                        word(&mut hash, *owner);
                        word(&mut hash, *index);
                    }
                }
                hash.update(program.digest());
                hash.update([u8::from(tail.is_some())]);
                if let Some(tail) = tail {
                    hash.update(tail.digest());
                }
            }
            Code::PublicLookup {
                input,
                weight,
                plan,
            } => {
                hash.update([10]);
                word(&mut hash, *input);
                word(&mut hash, *weight);
                hash.update(plan.digest());
            }
            Code::Selection { input, plan } => {
                hash.update([11]);
                word(&mut hash, *input);
                hash.update(plan.digest());
            }
            Code::TrieLookup {
                input,
                weight,
                plan,
            } => {
                hash.update([12]);
                word(&mut hash, *input);
                word(&mut hash, *weight);
                hash.update(plan.digest());
            }
        }
    }
    for &(owner, index) in selected {
        last[owner] = nodes.len();
        word(&mut hash, owner);
        word(&mut hash, index);
    }
    let mut retire = vec![vec![]; nodes.len()];
    for (owner, end) in last.into_iter().enumerate() {
        if end < nodes.len() {
            retire[end].push(owner);
        }
    }
    (
        hash.finalize().into(),
        retire,
        schedule_work(nodes).unwrap(),
    )
}

fn prefix() -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(b"original-source-policy-config-prefix");
    hash
}

#[test]
fn original_codes_match_independent_hash_last_use_work_and_exact_bucket_capacities() {
    let nodes = fixture();
    let selected = [(0, 1), (11, 1), (11, 0), (15, 0), (0, 1)];
    let prefix = prefix();
    let original_prefix: [u8; 32] = prefix.clone().finalize().into();
    let expected = legacy(&nodes, &selected, &prefix);
    let quote = quote(nodes.as_slice(), &selected, 113).unwrap();
    assert_eq!(quote.retained_output_owners, 3);
    assert_eq!(
        quote.retirement_entries_bytes,
        (nodes.len() - 3) * size_of::<usize>()
    );
    let actual = finalize(nodes.as_slice(), &selected, &prefix, 113, quote.peak_bytes).unwrap();
    assert_eq!(actual.digest, expected.0);
    assert_eq!(actual.retire, expected.1);
    assert_eq!(actual.work, expected.2);
    assert_eq!(actual.retire.capacity(), nodes.len());
    for bucket in &actual.retire {
        assert_eq!(bucket.capacity(), bucket.len());
    }
    for owner in [0, 11, 15] {
        assert!(actual.retire.iter().all(|bucket| !bucket.contains(&owner)));
    }
    assert!(actual.retire.last().unwrap().contains(&(nodes.len() - 1)));
    let after: [u8; 32] = prefix.clone().finalize().into();
    assert_eq!(after, original_prefix);
}

#[test]
fn exact_fit_one_short_overflow_and_fixed_cap_precede_first_owned_factory() {
    let nodes = fixture();
    let selected = [(11, 0)];
    reset_factories();
    let quote = quote(nodes.as_slice(), &selected, 79).unwrap();
    assert_no_factories();
    assert!(matches!(
        finalize(
            nodes.as_slice(),
            &selected,
            &prefix(),
            79,
            quote.peak_bytes - 1
        ),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        finalize(
            nodes.as_slice(),
            &selected,
            &prefix(),
            usize::MAX,
            usize::MAX
        ),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        finalize(
            nodes.as_slice(),
            &selected,
            &prefix(),
            MAX_PROGRAM_BYTES,
            usize::MAX
        ),
        Err(Error::Bounds)
    ));
    assert_no_factories();
    let result = finalize(nodes.as_slice(), &selected, &prefix(), 79, quote.peak_bytes).unwrap();
    assert_eq!(FINALIZATION_FACTORY_ENTRIES.with(|n| n.get()), 1);
    assert!(FINALIZATION_RESERVATIONS.with(|n| n.get()) >= 3);
    assert_eq!(result.work, quote.work);
    assert!(matches!(
        storage_quote(usize::MAX, 0, 0, WorkBudget::default()),
        Err(Error::Bounds)
    ));
    assert!(matches!(
        storage_quote(1, 2, 0, WorkBudget::default()),
        Err(Error::Binding)
    ));
    let expected = quote.retained_baseline_bytes
        + quote.last_use_bytes
        + quote.retirement_count_bytes
        + quote.retirement_container_bytes
        + quote.retirement_entries_bytes
        + quote.workspace_inline_bytes
        + quote.hash_finish_bytes
        + quote.result_slot_bytes;
    assert_eq!(quote.peak_bytes, expected);
}

#[test]
fn malformed_dependencies_coordinates_and_loop_tails_fail_before_owned_factory() {
    reset_factories();
    for kind in 0..8 {
        let mut nodes = fixture();
        match kind {
            0 => {
                nodes[5].code = Code::Affine {
                    left: 5,
                    right: None,
                    coefficients: affine(1, 0, 0),
                }
            }
            1 => {
                nodes[5].code = Code::Affine {
                    left: usize::MAX,
                    right: None,
                    coefficients: affine(1, 0, 0),
                }
            }
            2 => nodes[3].code = Code::Gather([(0, 4), (0, 0), (0, 1)].into_iter().collect()),
            3 => {
                if let Code::SourceGather(view) = &mut nodes[4].code {
                    view.offset = usize::MAX;
                }
            }
            4 => {
                if let Code::TensorLoop { tail, .. } = &mut nodes[11].code {
                    *tail = None;
                }
            }
            5 => {
                if let Code::TensorLoop { tail, .. } = &mut nodes[12].code {
                    *tail = Some(rescale(1));
                }
            }
            6 => {
                if let Code::TensorLoop { input, .. } = &mut nodes[12].code {
                    *input = LoopInput::TokenOffsets { owner: 2, index: 1 };
                }
            }
            _ => nodes[3].lanes = 2,
        }
        assert!(
            finalize(nodes.as_slice(), &[], &prefix(), 0, MAX_PROGRAM_BYTES).is_err(),
            "kind={kind}"
        );
    }
    let nodes = fixture();
    for selected in [[(usize::MAX, 0)], [(0, 4)]] {
        assert!(finalize(nodes.as_slice(), &selected, &prefix(), 0, MAX_PROGRAM_BYTES).is_err());
    }
    let local = [node(Code::Input("no-work".into()), 1)];
    assert!(matches!(
        quote(local.as_slice(), &[], 0),
        Err(Error::Bounds)
    ));
    assert_no_factories();
}

#[test]
fn complete_schedule_work_overflow_rejects_before_owned_factory() {
    // Metadata only: no huge lane array is allocated. Per-node work is still
    // derived from the actual original native template and full-loop geometry.
    let nodes = [
        node(Code::Input("large-owner".into()), usize::MAX),
        node(
            Code::TensorLoop {
                input: LoopInput::Slice(0),
                program: rescale(1),
                tail: None,
            },
            usize::MAX,
        ),
    ];
    reset_factories();
    assert!(finalize(nodes.as_slice(), &[], &prefix(), 0, MAX_PROGRAM_BYTES).is_err());
    assert_no_factories();
}

#[test]
fn complete_work_limit_rejection_precedes_helper_allocations() {
    let lanes = puma::MAX_PROGRAM_OPERATIONS as usize + 1;
    let nodes = [
        node(Code::Input("large-owner".into()), lanes),
        node(
            Code::TensorLoop {
                input: LoopInput::Slice(0),
                program: rescale(1),
                tail: None,
            },
            lanes,
        ),
    ];
    let work = schedule_work(&nodes).unwrap();
    assert!(work.operations() > puma::MAX_PROGRAM_OPERATIONS);
    reset_factories();
    assert!(matches!(
        finalize(nodes.as_slice(), &[], &prefix(), 0, MAX_PROGRAM_BYTES),
        Err(Error::Bounds)
    ));
    assert_no_factories();
}

#[test]
fn original_failure_sites_and_identity_source_binding_order_remain_authoritative() {
    use std::cell::Cell;

    // Metadata-only fixtures: no huge native program or lane payload is built.
    // Original ordering differs from the candidate helper: identity precedes
    // zero-work/complete-work-limits failures, but follows work-sum overflow.
    let over_limit = puma::MAX_PROGRAM_OPERATIONS as usize + 1;
    for (lanes, fail_identity) in [
        (0, false),
        (1, true),
        (usize::MAX, false),
        (over_limit, false),
    ] {
        let mut nodes = vec![node(Code::Input("original".into()), lanes.max(1))];
        if lanes != 0 {
            nodes.push(node(
                Code::TensorLoop {
                    input: LoopInput::Slice(0),
                    program: rescale(1),
                    tail: None,
                },
                lanes,
            ));
        }
        let mut builder = Builder {
            nodes,
            admission_stage: "before_finish",
            admission_limit: Some("unchanged"),
            requested_bytes: Some(17),
            requested_lanes: Some(23),
            ..Builder::default()
        };
        let original_nodes = builder.nodes.as_ptr();
        let original_len = builder.nodes.len();
        let mut site = AdmissionSite {
            selection: 7,
            step: Some(3),
            operator: Some(Op::Linear),
            layer: Some(2),
            inputs: [Some((-4, 4)); 2],
            complete_work: Some((91, 92)),
        };
        let identities = Cell::new(0);
        let bindings = Cell::new(0);
        reset_factories();
        let result = builder.finish(
            1,
            vec![],
            &mut site,
            |_| {
                identities.set(identities.get() + 1);
                if fail_identity {
                    Err(Error::Binding)
                } else {
                    Ok(())
                }
            },
            || {
                bindings.set(bindings.get() + 1);
                [3; 32]
            },
        );
        if fail_identity {
            assert!(matches!(result, Err(Error::Binding)));
        } else {
            assert!(matches!(result, Err(Error::Bounds)));
        }
        assert_eq!(identities.get(), usize::from(lanes != usize::MAX));
        assert_eq!(bindings.get(), 0);
        assert_eq!(site.selection, 7);
        assert!(site.step.is_none());
        assert!(site.operator.is_none());
        assert!(site.layer.is_none());
        assert_eq!(site.inputs, [None; 2]);
        if lanes == over_limit {
            let work = schedule_work(&builder.nodes).unwrap();
            assert_eq!(
                site.complete_work,
                Some((work.operations(), work.cursors().aes_work))
            );
            assert_eq!(builder.admission_stage, "complete_work_limits");
        } else {
            assert_eq!(site.complete_work, Some((91, 92)));
            assert_eq!(builder.admission_stage, "complete_schedule_work");
        }
        assert_eq!(builder.admission_limit, Some("unchanged"));
        assert_eq!(builder.requested_bytes, Some(17));
        assert_eq!(builder.requested_lanes, Some(23));
        assert_eq!(builder.nodes.as_ptr(), original_nodes);
        assert_eq!(builder.nodes.len(), original_len);
        assert_no_factories();
    }
}

#[test]
fn original_success_transfers_same_nodes_selected_capacity_and_source_binding() {
    use std::cell::Cell;

    let nodes = fixture();
    let original_nodes = nodes.as_ptr();
    let original_capacity = nodes.capacity();
    let mut selected = Vec::with_capacity(19);
    selected.extend([(0, 1), (11, 1), (11, 0), (15, 0), (0, 1)]);
    let original_selected = selected.as_ptr();
    let selected_capacity = selected.capacity();
    let expected = legacy(&nodes, &selected, &prefix());
    let template = match &nodes[11].code {
        Code::TensorLoop { program, .. } => Arc::as_ptr(program),
        _ => unreachable!(),
    };
    let owners = match &nodes[11].code {
        Code::TensorLoop { program, .. } => Arc::strong_count(program),
        _ => unreachable!(),
    };
    let mut builder = Builder {
        nodes,
        allocated: 137,
        ..Builder::default()
    };
    let mut site = AdmissionSite {
        selection: 9,
        ..AdmissionSite::default()
    };
    let order = Cell::new(0);
    reset_factories();
    let program = builder
        .finish(
            3,
            selected,
            &mut site,
            |hash| {
                assert_eq!(order.replace(1), 0);
                *hash = prefix();
                Ok(())
            },
            || {
                assert_eq!(order.replace(2), 1);
                [42; 32]
            },
        )
        .unwrap();
    assert_eq!(order.get(), 2);
    assert_eq!(program.nodes.as_ptr(), original_nodes);
    assert_eq!(program.nodes.capacity(), original_capacity);
    assert_eq!(program.outputs.as_ptr(), original_selected);
    assert_eq!(program.outputs.capacity(), selected_capacity);
    assert_eq!(program.outputs, [(0, 1), (11, 1), (11, 0), (15, 0), (0, 1)]);
    assert_eq!(program.inputs, 3);
    assert_eq!(program.source_binding, [42; 32]);
    assert_eq!(program.payload_bytes(), 137);
    assert_eq!((program.digest(), program.retire, program.work), expected);
    assert!(builder.nodes.is_empty());
    assert_eq!(site.selection, 9);
    assert_eq!(
        site.complete_work,
        Some((expected.2.operations(), expected.2.cursors().aes_work))
    );
    assert_eq!(builder.admission_stage, "complete_work_limits");
    if let Code::TensorLoop { program, .. } = &program.nodes[11].code {
        assert_eq!(Arc::as_ptr(program), template);
        assert_eq!(Arc::strong_count(program), owners);
    } else {
        unreachable!();
    }
    // Prove original finish actually reached the test-only comparison seam.
    assert_eq!(FINALIZATION_FACTORY_ENTRIES.with(|n| n.get()), 1);
}

#[test]
fn borrowed_comparison_replays_independently_frozen_full_prefix_identities() {
    // Same installed-wheel identities frozen in replicated_decoder/tests.rs,
    // not newly captured helper output. Compilation checks metadata only here;
    // execution, numeric lowering and default expanded routing stay unchanged.
    for (family, active, expected) in [
        (
            "qwen2",
            false,
            "333aa62a92ec54bdfb694e6bc356d3b4d873629d0cbee5bbbbaa3fb447d28c6d",
        ),
        (
            "qwen2",
            true,
            "4529a304429056c986676ea4b8da1adf4cf864e23e41f40a0778a32268b671a1",
        ),
        (
            "qwen3",
            false,
            "83886ad38965f9244ba95b403855fb56c6298f44392c44e022b400b19118f7d2",
        ),
        (
            "qwen3",
            true,
            "98044e61b32ee4cde96ee63a0c0accc70a776b51b689ef763930353aea60d45e",
        ),
        (
            "llama",
            false,
            "f9b5f1dcc3c06ac87013adca91751f7acae25e5993fae0137b0b90ca98c08eca",
        ),
        (
            "llama",
            true,
            "9e9593392a6058d6dac04ecb442ec83a31d6e017291e5007144ac91731e4ce8c",
        ),
    ] {
        let config = serde_json::to_vec(&serde_json::json!({
            "model_type": family, "hidden_size": 2, "intermediate_size": 2,
            "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
            "head_dim": 2, "vocab_size": 3, "tie_word_embeddings": false,
            "hidden_act": "silu", "attention_bias": family == "qwen2",
            "max_position_embeddings": 64, "rope_theta": 10000,
            "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
            "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
        }))
        .unwrap();
        let mut sources = BTreeMap::new();
        for (key, shape) in Program::source_shapes(&config, 2, 2).unwrap() {
            let values: Vec<i64> = if active && key == "model.embed_tokens.weight" {
                vec![ONE / 4, ONE / 4, -ONE / 4, -ONE / 4, 0, 0]
            } else if key == "model.embed_tokens.weight" {
                vec![ONE, ONE, ONE, 0, -ONE, 0]
            } else if key == "lm_head.weight" {
                vec![ONE, 0, -ONE, 0, 0, 0]
            } else if key.contains("norm") {
                vec![ONE; shape.iter().product()]
            } else if active && key.ends_with("v_proj.weight") {
                vec![ONE, 0, 0, ONE]
            } else if active && key.ends_with("o_proj.weight") {
                vec![-ONE / 2, 0, 0, -ONE / 2]
            } else if active && key.contains("mlp.") {
                vec![ONE / 64; shape.iter().product()]
            } else {
                vec![0; shape.iter().product()]
            };
            let mut hash = Sha256::new();
            for value in &values {
                hash.update(value.to_le_bytes());
            }
            sources.insert(
                key.clone(),
                SourceTensor {
                    shape,
                    lower: *values.iter().min().unwrap(),
                    upper: *values.iter().max().unwrap(),
                    public_q18: key.contains("norm").then_some(values),
                    digest: hash.finalize().into(),
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
            );
        }
        reset_factories();
        let before_compile = FINALIZATION_FACTORY_ENTRIES.with(|n| n.get());
        assert_eq!(before_compile, 0);
        let mut program = Program::compile(&config, 2, 2, &sources).unwrap();
        // Default compilation deliberately keeps its original inline finalizer;
        // Builder::finish already runs the test-only borrowed comparison.
        // This counter counts admitted helper entries, not lifetime finalization callbacks.
        let after_compile = FINALIZATION_FACTORY_ENTRIES.with(|n| n.get());
        assert_eq!(after_compile, before_compile + 1);
        let mut prefix = Sha256::new();
        prefix.update(b"pllm.replicated-decoder.source-q18.full-prefix-final-ids.v1");
        prefix.update(&config);
        prefix.update(2_u64.to_le_bytes());
        prefix.update(2_u64.to_le_bytes());
        prefix.update(serde_json::to_vec(&sources).unwrap());
        let literal = legacy(&program.nodes, &program.outputs, &prefix);
        assert_eq!(literal.0, program.digest());
        assert_eq!(literal.1, program.retire);
        assert_eq!(literal.2, program.work_budget());
        let quote = quote(program.nodes.as_slice(), &program.outputs, 0).unwrap();
        assert_eq!(FINALIZATION_FACTORY_ENTRIES.with(|n| n.get()), after_compile);
        let replay = finalize(
            program.nodes.as_slice(),
            &program.outputs,
            &prefix,
            0,
            quote.peak_bytes,
        )
        .unwrap();
        assert_eq!(replay.digest, literal.0);
        assert_eq!(replay.retire, literal.1);
        assert_eq!(replay.work, literal.2);
        assert_eq!(FINALIZATION_FACTORY_ENTRIES.with(|n| n.get()), after_compile + 1);
        assert_eq!(program.source_binding, source_binding(&sources));
        program.bind([7; 32], [9; 32]);
        use std::fmt::Write as _;
        let mut actual = String::new();
        for v in program.digest() {
            write!(&mut actual, "{v:02x}").unwrap();
        }
        assert_eq!(actual, expected, "{family}, active={active}");
    }
}

#[test]
fn genuine_original_compiler_fixture_and_arc_owners_remain_unchanged() {
    for lanes in [1, 128, 129, 257] {
        let (program, prepared, _, output) =
            prepared_rescale18_worker_fixture(lanes, bounds()).unwrap();
        let counts: Vec<_> = prepared.programs().map(Arc::strong_count).collect();
        let pointers: Vec<_> = prepared.programs().map(Arc::as_ptr).collect();
        let mut identity = Sha256::new();
        identity.update(b"pllm.replicated-decoder.prepared-rescale18-worker-fixture.v1");
        // Helper-only comparison; payload_bytes is not a global owner census.
        let quote = quote(program.nodes.as_slice(), &program.outputs, 0).unwrap();
        let actual = finalize(
            program.nodes.as_slice(),
            &program.outputs,
            &identity,
            0,
            quote.peak_bytes,
        )
        .unwrap();
        assert_eq!(actual.digest, program.digest());
        assert_eq!(actual.retire, program.retire);
        assert_eq!(actual.work, program.work_budget());
        assert_eq!(
            prepared
                .programs()
                .map(Arc::strong_count)
                .collect::<Vec<_>>(),
            counts
        );
        assert_eq!(
            prepared.programs().map(Arc::as_ptr).collect::<Vec<_>>(),
            pointers
        );
        assert!(actual.retire.iter().all(|bucket| !bucket.contains(&output)));
    }
}

#[test]
fn borrowed_segment_seam_reuses_original_nodes_without_arc_clones() {
    struct Segments<'a> {
        first: &'a [Node],
        second: &'a [Node],
    }
    impl BorrowedOrderedNodes for Segments<'_> {
        fn node_count(&self) -> usize {
            self.first.len() + self.second.len()
        }
        fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
            if ordinal < self.first.len() {
                self.first.get(ordinal)
            } else {
                self.second.get(ordinal - self.first.len())
            }
            .map(BorrowedNode::from)
            .ok_or(Error::Binding)
        }
    }
    let nodes = fixture();
    let split = Segments {
        first: &nodes[..8],
        second: &nodes[8..],
    };
    let selected = [(11, 2)];
    let prefix = prefix();
    let expected = legacy(&nodes, &selected, &prefix);
    let quote = quote(&split, &selected, 0).unwrap();
    let actual = finalize(&split, &selected, &prefix, 0, quote.peak_bytes).unwrap();
    assert_eq!((actual.digest, actual.retire, actual.work), expected);
}

/// Retains only borrowed recipe descriptors; no expanded Node adapter on lookup.
struct Recipes<'a>(&'a [BorrowedNode<'a>]);

impl BorrowedOrderedNodes for Recipes<'_> {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
        self.0.get(ordinal).copied().ok_or(Error::Binding)
    }
}

#[test]
fn borrowed_recipe_views_match_every_expanded_code_and_exact_short_admission() {
    let nodes = fixture();
    let views: Vec<_> = nodes.iter().map(BorrowedNode::from).collect();
    let recipes = Recipes(&views);
    let selected = [(0, 1), (11, 1), (11, 0), (15, 0), (0, 1)];
    // This baseline is fixture-owned descriptors only, not request admission.
    let baseline = views.capacity() * size_of::<BorrowedNode<'_>>();
    let expected = legacy(&nodes, &selected, &prefix());
    let expanded_quote = quote(nodes.as_slice(), &selected, baseline).unwrap();
    let recipe_quote = quote(&recipes, &selected, baseline).unwrap();
    assert_eq!(recipe_quote, expanded_quote);
    for (original, view) in nodes.iter().zip(&views) {
        assert_eq!(original.work_budget().unwrap(), view.work_budget().unwrap());
        if let (
            Code::TensorLoop {
                program: original, ..
            },
            BorrowedRecipe::TensorLoop {
                program: borrowed, ..
            },
        ) = (&original.code, view.recipe)
        {
            assert!(std::ptr::eq(original, borrowed));
        }
    }
    reset_factories();
    assert!(matches!(
        finalize(
            &recipes,
            &selected,
            &prefix(),
            baseline,
            recipe_quote.peak_bytes - 1
        ),
        Err(Error::Bounds)
    ));
    assert_no_factories();
    let actual = finalize(
        &recipes,
        &selected,
        &prefix(),
        baseline,
        recipe_quote.peak_bytes,
    )
    .unwrap();
    assert_eq!((actual.digest, actual.retire, actual.work), expected);
}

/// A retained arithmetic recipe, not Words, Node, a Vec of lanes or hash bytes.
struct ReverseRefs {
    owner: usize,
    len: usize,
    fail_at: Option<usize>,
}

impl BorrowedLaneRefs for ReverseRefs {
    fn len(&self) -> usize {
        self.len
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if self.fail_at == Some(index) || index >= self.len {
            return Err(Error::Binding);
        }
        Ok((self.owner, self.len - index - 1))
    }
}

fn generated_views<'a>(
    refs: &'a dyn BorrowedLaneRefs,
    program: &'a Arc<tensor::Program>,
) -> [BorrowedNode<'a>; 3] {
    [
        BorrowedNode {
            lanes: 2,
            bounds: bounds(),
            recipe: BorrowedRecipe::Input("input"),
        },
        BorrowedNode {
            lanes: 2,
            bounds: bounds(),
            recipe: BorrowedRecipe::Gather(refs),
        },
        BorrowedNode {
            lanes: 2,
            bounds: program.output_bounds(),
            recipe: BorrowedRecipe::Tensor { input: 1, program },
        },
    ]
}

#[test]
fn generated_refs_and_native_recipe_borrow_original_encoder_without_expansion() {
    let refs = ReverseRefs {
        owner: 0,
        len: 2,
        fail_at: None,
    };
    let program = rescale(2);
    let owners = Arc::strong_count(&program);
    let views = generated_views(&refs, &program);
    let recipes = Recipes(&views);
    // Independent expanded oracle exists ONLY in this test, not the recipe path.
    let expanded = [
        node(Code::Input("input".into()), 2),
        node(Code::Gather([(0, 1), (0, 0)].into_iter().collect()), 2),
        Node {
            code: Code::Tensor {
                input: 1,
                program: program.clone(),
            },
            lanes: 2,
            bounds: program.output_bounds(),
        },
    ];
    for selected in [vec![(2, 1), (2, 0), (2, 1)], vec![(2, 0), (2, 1), (2, 1)]] {
        let expected = legacy(&expanded, &selected, &prefix());
        let quote = quote(&recipes, &selected, 0).unwrap();
        reset_factories();
        assert!(matches!(
            finalize(&recipes, &selected, &prefix(), 0, quote.peak_bytes - 1),
            Err(Error::Bounds)
        ));
        assert_no_factories();
        let actual = finalize(&recipes, &selected, &prefix(), 0, quote.peak_bytes).unwrap();
        assert_eq!((actual.digest, actual.retire, actual.work), expected);
    }
    assert_eq!(Arc::strong_count(&program), owners + 1); // expanded oracle only
    let selected_a = [(2, 1), (2, 0)];
    let selected_b = [(2, 0), (2, 1)];
    let q = quote(&recipes, &selected_a, 0).unwrap();
    let a = finalize(&recipes, &selected_a, &prefix(), 0, q.peak_bytes).unwrap();
    let b = finalize(&recipes, &selected_b, &prefix(), 0, q.peak_bytes).unwrap();
    assert_ne!(a.digest, b.digest);
    assert_eq!(a.retire, b.retire);
}

#[test]
fn generated_invalid_refs_coordinates_order_and_outputs_fail_before_factory() {
    let program = rescale(2);
    reset_factories();
    for (refs, lanes, expect_shape) in [
        (
            ReverseRefs {
                owner: 1,
                len: 2,
                fail_at: None,
            },
            2,
            false,
        ), // self
        (
            ReverseRefs {
                owner: 2,
                len: 2,
                fail_at: None,
            },
            2,
            false,
        ), // forward
        (
            ReverseRefs {
                owner: usize::MAX,
                len: 2,
                fail_at: None,
            },
            2,
            false,
        ),
        (
            ReverseRefs {
                owner: 0,
                len: 3,
                fail_at: None,
            },
            3,
            true,
        ), // coordinate
        (
            ReverseRefs {
                owner: 0,
                len: 2,
                fail_at: Some(1),
            },
            2,
            false,
        ),
        // Shape precedes even an invalid owner or failing reference getter.
        (
            ReverseRefs {
                owner: usize::MAX,
                len: 2,
                fail_at: Some(0),
            },
            1,
            true,
        ),
    ] {
        let mut views = generated_views(&refs, &program);
        views[1].lanes = lanes;
        let result = finalize(&Recipes(&views), &[], &prefix(), 0, MAX_PROGRAM_BYTES);
        if expect_shape {
            assert!(matches!(result, Err(Error::Shape)));
        } else {
            assert!(matches!(result, Err(Error::Binding)));
        }
        assert_no_factories();
    }
    let refs = ReverseRefs {
        owner: 0,
        len: 2,
        fail_at: None,
    };
    let mut views = generated_views(&refs, &program);
    assert!(matches!(
        finalize(&Recipes(&views), &[(3, 0)], &prefix(), 0, MAX_PROGRAM_BYTES),
        Err(Error::Binding)
    ));
    assert!(matches!(
        finalize(&Recipes(&views), &[(2, 2)], &prefix(), 0, MAX_PROGRAM_BYTES),
        Err(Error::Shape)
    ));
    views.swap(1, 2); // tensor now refers to itself; schedule order is not sorted
    assert!(matches!(
        finalize(&Recipes(&views), &[], &prefix(), 0, MAX_PROGRAM_BYTES),
        Err(Error::Binding)
    ));
    assert_no_factories();
}

#[test]
fn missing_recipe_ordinal_and_fallible_canonical_refs_return_errors_not_partial_digest() {
    struct MissingLast<'a>(Recipes<'a>);
    impl BorrowedOrderedNodes for MissingLast<'_> {
        fn node_count(&self) -> usize {
            self.0.node_count() + 1
        }
        fn node(&self, ordinal: usize) -> Result<BorrowedNode<'_>, Error> {
            self.0.node(ordinal)
        }
    }
    let program = rescale(2);
    let refs = ReverseRefs {
        owner: 0,
        len: 2,
        fail_at: None,
    };
    let views = generated_views(&refs, &program);
    reset_factories();
    assert!(matches!(
        finalize(
            &MissingLast(Recipes(&views)),
            &[],
            &prefix(),
            0,
            MAX_PROGRAM_BYTES
        ),
        Err(Error::Binding)
    ));
    assert_no_factories();

    // Exercise the private fallible encoder itself: a failed coordinate cannot
    // silently turn a gather into a shorter, successfully finalized identity.
    let refs = ReverseRefs {
        owner: 0,
        len: 2,
        fail_at: Some(1),
    };
    let view = generated_views(&refs, &program)[1];
    let prefix = prefix();
    let original: [u8; 32] = prefix.clone().finalize().into();
    let mut private_hash = prefix.clone();
    assert!(matches!(
        hash_node(&mut private_hash, view),
        Err(Error::Binding)
    ));
    let unchanged: [u8; 32] = prefix.finalize().into();
    assert_eq!(unchanged, original);
    assert_no_factories();
}

#[test]
fn generated_recipe_error_order_matches_expanded_adapter_and_checks_loop_tails() {
    let nodes = fixture();
    for kind in 0..4 {
        let tail = rescale(1);
        let mut views: Vec<_> = nodes.iter().map(BorrowedNode::from).collect();
        views[11].recipe = match kind {
            0 => BorrowedRecipe::TensorLoop {
                input: LoopInput::Slice(3),
                program: match views[11].recipe {
                    BorrowedRecipe::TensorLoop { program, .. } => program,
                    _ => unreachable!(),
                },
                tail: None,
            },
            1 => BorrowedRecipe::TensorLoop {
                input: LoopInput::TokenOffsets { owner: 2, index: 1 },
                program: &tail,
                tail: None,
            },
            2 => BorrowedRecipe::TensorLoop {
                input: LoopInput::Slice(11),
                program: &tail,
                tail: Some(&tail),
            },
            _ => BorrowedRecipe::TensorLoop {
                input: LoopInput::Slice(3),
                program: &tail,
                tail: Some(&tail),
            },
        };
        reset_factories();
        // Invalid output must not obscure the earlier recipe/reference error.
        let result = finalize(
            &Recipes(&views),
            &[(usize::MAX, 0)],
            &prefix(),
            0,
            MAX_PROGRAM_BYTES,
        );
        if kind == 2 {
            assert!(matches!(result, Err(Error::Binding)));
        } else {
            assert!(matches!(result, Err(Error::Shape)));
        }
        assert_no_factories();
    }
    let mut invalid = fixture();
    invalid[3].lanes = 2; // gather length failure precedes later invalid owner
    invalid[5].code = Code::Affine {
        left: usize::MAX,
        right: None,
        coefficients: affine(1, 0, 0),
    };
    let views: Vec<_> = invalid.iter().map(BorrowedNode::from).collect();
    assert!(matches!(
        quote(invalid.as_slice(), &[], 0),
        Err(Error::Shape)
    ));
    assert!(matches!(quote(&Recipes(&views), &[], 0), Err(Error::Shape)));
}

#[test]
fn native_tensor_and_checked_range_geometry_reject_before_work_or_owned_factory() {
    fn rejects<S: BorrowedOrderedNodes + ?Sized>(schedule: &S, prefix: &Sha256) {
        assert!(matches!(quote(schedule, &[], 0), Err(Error::Shape)));
        // Geometry precedes invalid selected owner and an insufficient cap.
        assert!(matches!(
            finalize(schedule, &[(usize::MAX, 0)], prefix, 0, 0),
            Err(Error::Shape)
        ));
    }
    let one = rescale(1);
    let checked = Arc::new(
        tensor::Program::checked_range(1, bounds(), Interval::new(-1, 1).unwrap(), 0).unwrap(),
    );
    // (checked recipe, native owner, dependency lanes, declared output lanes).
    // First case is the concrete two-lane node / one-lane template counterexample.
    for (is_checked, program, input_lanes, output_lanes) in [
        (false, &one, 1, 2),
        (false, &one, 2, 1),
        (true, &checked, 1, 1), // omitted validity half
        (true, &checked, 2, 2), // correct combined output, wrong input width
        (true, &one, 1, 1),     // ordinary program has no protected validity output
    ] {
        let recipe = if is_checked {
            BorrowedRecipe::CheckedRange { input: 0, program }
        } else {
            BorrowedRecipe::Tensor { input: 0, program }
        };
        let views = [
            BorrowedNode {
                lanes: input_lanes,
                bounds: bounds(),
                recipe: BorrowedRecipe::Input("input"),
            },
            BorrowedNode {
                lanes: output_lanes,
                bounds: bounds(),
                recipe,
            },
        ];
        let expanded = [
            node(Code::Input("input".into()), input_lanes),
            node(
                if is_checked {
                    Code::CheckedRange {
                        input: 0,
                        program: Arc::clone(program),
                    }
                } else {
                    Code::Tensor {
                        input: 0,
                        program: Arc::clone(program),
                    }
                },
                output_lanes,
            ),
        ];
        let prefix = prefix();
        let before: [u8; 32] = prefix.clone().finalize().into();
        reset_factories();
        rejects(&Recipes(&views), &prefix);
        rejects(expanded.as_slice(), &prefix);
        let after: [u8; 32] = prefix.finalize().into();
        assert_eq!(before, after);
        assert_no_factories();
    }
}

#[test]
fn native_reduction_and_combined_checked_outputs_keep_original_canonical_identity() {
    let reduction = Arc::new(tensor::Program::all_valid(2).unwrap());
    let checked = Arc::new(
        tensor::Program::checked_range(2, bounds(), Interval::new(-1, 1).unwrap(), 0).unwrap(),
    );
    assert_eq!(reduction.lanes(), 2);
    assert_eq!(reduction.output_lanes(), 1);
    assert_eq!(checked.output_lanes(), 2);
    assert_eq!(checked.validity_lanes(), Some(2));
    let nodes = [
        Node {
            code: Code::Input("bits".into()),
            lanes: 2,
            bounds: Interval::new(0, 1).unwrap(),
        },
        Node {
            code: Code::Tensor {
                input: 0,
                program: reduction,
            },
            lanes: 1,
            bounds: Interval::new(0, 1).unwrap(),
        },
        node(Code::Input("guarded".into()), 2),
        node(
            Code::CheckedRange {
                input: 2,
                program: checked,
            },
            4,
        ),
    ];
    // Include validity coordinates, not only the safe-output half.
    let selected = [(1, 0), (3, 2), (3, 3)];
    let views: Vec<_> = nodes.iter().map(BorrowedNode::from).collect();
    let expected = legacy(&nodes, &selected, &prefix());
    let recipes = Recipes(&views);
    let q = quote(&recipes, &selected, 0).unwrap();
    let actual = finalize(&recipes, &selected, &prefix(), 0, q.peak_bytes).unwrap();
    assert_eq!((actual.digest, actual.retire, actual.work), expected);
}

#[test]
fn lazy_gather_order_changes_only_identity_not_work_or_retirement() {
    struct OrderedRefs {
        reversed: bool,
    }
    impl BorrowedLaneRefs for OrderedRefs {
        fn len(&self) -> usize {
            2
        }
        fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
            if index >= 2 {
                return Err(Error::Binding);
            }
            Ok((0, if self.reversed { 1 - index } else { index }))
        }
    }
    let forward = OrderedRefs { reversed: false };
    let reverse = OrderedRefs { reversed: true };
    let program = rescale(2);
    let a_views = generated_views(&forward, &program);
    let b_views = generated_views(&reverse, &program);
    let a = Recipes(&a_views);
    let b = Recipes(&b_views);
    let selected = [(2, 0), (2, 1)];
    let q = quote(&a, &selected, 0).unwrap();
    assert_eq!(q, quote(&b, &selected, 0).unwrap());
    let first = finalize(&a, &selected, &prefix(), 0, q.peak_bytes).unwrap();
    let second = finalize(&b, &selected, &prefix(), 0, q.peak_bytes).unwrap();
    assert_ne!(first.digest, second.digest);
    assert_eq!(first.work, second.work);
    assert_eq!(first.retire, second.retire);
    for (refs, actual) in [(&forward, first), (&reverse, second)] {
        let nodes = [
            node(Code::Input("input".into()), 2),
            node(
                Code::Gather((0..2).map(|i| refs.lane_ref(i).unwrap()).collect()),
                2,
            ),
            Node {
                code: Code::Tensor {
                    input: 1,
                    program: program.clone(),
                },
                lanes: 2,
                bounds: program.output_bounds(),
            },
        ];
        assert_eq!(
            (actual.digest, actual.retire, actual.work),
            legacy(&nodes, &selected, &prefix())
        );
    }
}

#[test]
fn wrong_width_tail_rejects_but_tail_only_loop_keeps_exact_work_and_identity() {
    let full = rescale(2);
    let tail = rescale(1);
    let bad_views = [
        BorrowedNode {
            lanes: 3,
            bounds: bounds(),
            recipe: BorrowedRecipe::Input("input"),
        },
        BorrowedNode {
            lanes: 3,
            bounds: full.output_bounds(),
            recipe: BorrowedRecipe::TensorLoop {
                input: LoopInput::Slice(0),
                program: &full,
                tail: Some(&full),
            },
        },
    ];
    reset_factories();
    assert!(matches!(
        quote(&Recipes(&bad_views), &[], 0),
        Err(Error::Shape)
    ));
    assert!(matches!(
        finalize(&Recipes(&bad_views), &[], &prefix(), 0, MAX_PROGRAM_BYTES),
        Err(Error::Shape)
    ));
    assert_no_factories();

    // Zero full tiles: only the one-lane tail contributes work, although both
    // original native template digests remain part of the canonical recipe.
    let views = [
        BorrowedNode {
            lanes: 1,
            bounds: bounds(),
            recipe: BorrowedRecipe::Input("input"),
        },
        BorrowedNode {
            lanes: 1,
            bounds: tail.output_bounds(),
            recipe: BorrowedRecipe::TensorLoop {
                input: LoopInput::Slice(0),
                program: &full,
                tail: Some(&tail),
            },
        },
    ];
    let selected = [(1, 0)];
    let recipes = Recipes(&views);
    let q = quote(&recipes, &selected, 0).unwrap();
    assert_eq!(q.work, tail.work_budget());
    assert_no_factories();
    assert!(matches!(
        finalize(&recipes, &selected, &prefix(), 0, q.peak_bytes - 1),
        Err(Error::Bounds)
    ));
    assert_no_factories();
    let actual = finalize(&recipes, &selected, &prefix(), 0, q.peak_bytes).unwrap();
    let nodes = [
        node(Code::Input("input".into()), 1),
        Node {
            code: Code::TensorLoop {
                input: LoopInput::Slice(0),
                program: full.clone(),
                tail: Some(tail.clone()),
            },
            lanes: 1,
            bounds: tail.output_bounds(),
        },
    ];
    assert_eq!(
        (actual.digest, actual.retire, actual.work),
        legacy(&nodes, &selected, &prefix())
    );
}
