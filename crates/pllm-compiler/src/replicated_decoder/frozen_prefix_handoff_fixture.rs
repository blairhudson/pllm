//! Genuine creation and literal original Builder fixture, scoped to its owners.
//! This creates NO runtime/coordinator capability.
use super::*;
use crate::replicated_decoder::program::original_import;

pub(in crate::replicated_decoder) fn with_fixture<R>(
    lanes: usize,
    run: impl for<'a> FnOnce(
        PendingPrefix<'a>,
        PendingPrefix<'a>,
        Arc<Program>,
        Arc<Program>,
        usize,
    ) -> R,
) -> R {
    let config = serde_json::to_vec(&serde_json::json!({"model_type":"llama","hidden_size":2,
        "intermediate_size":lanes,"num_hidden_layers":1,"num_attention_heads":1,
        "num_key_value_heads":1,"head_dim":2,"vocab_size":3,"tie_word_embeddings":false,
        "hidden_act":"silu","attention_bias":false,"max_position_embeddings":64,
        "rope_theta":10000,"rms_norm_eps":0.000001,"attention_dropout":0.0,"use_cache":true}))
    .unwrap();
    let mut plan = lower_with_weights(&config, 1, 1, WeightMode::Public).unwrap();
    let mut first = plan
        .prefill
        .operations
        .iter()
        .find(|op| op.operator == Op::Silu)
        .unwrap()
        .clone();
    first.id = "stage.first".into();
    first.inputs = vec!["state.source".into()];
    first.output_shape = vec![1, 1, u64::try_from(lanes).unwrap()];
    first.state_kind = Some(StateKind::Recurrent);
    let mut second = first.clone();
    second.id = "stage.second".into();
    second.inputs = vec![first.id.clone()];
    let mut next_stage = second.clone();
    next_stage.inputs = vec!["state.destination".into()];
    plan.decode.output = next_stage.id.clone();
    plan.decode.operations = vec![next_stage];
    let state = pllm_models::StateTensor {
        id: "state.source".into(),
        layer: first.layer,
        kind: StateKind::Recurrent,
        shape: first.output_shape.clone(),
        maximum_sequence: 1,
    };
    plan.prefill.state_inputs = vec![state.clone()];
    plan.prefill.state_outputs = vec![pllm_models::StateTensor {
        id: first.id.clone(),
        ..state.clone()
    }];
    plan.prefill.output = first.id.clone();
    // Both stages belong to the ORIGINAL global Program/PRSS. The selected
    // export is held across stage two, exactly as its original alias semantics.
    plan.prefill.operations = vec![first, second];
    plan.decode.state_inputs = vec![pllm_models::StateTensor {
        id: "state.destination".into(),
        ..state
    }];
    let sources = BTreeMap::new();
    let norms = SourceNorms::new();
    let validated = || {
        validate_source_records(
            &plan.prefill,
            &BTreeMap::new(),
            &sources,
            WeightMode::Public,
            ExecutionPolicy::default(),
            &norms,
        )
        .unwrap()
    };
    let registry = SourceRegistry::from_validated(validated()).unwrap();
    let values = Vec::new();
    let owners = original_import::tests::owners(&values);
    let prior = RequestWork::default();
    let bounds = Interval::new(-ONE / 4, ONE / 4).unwrap();
    let builder = || Builder {
        public_weights: true,
        ..Builder::default()
    };
    let mut original = builder();
    let bindings = vec![
        original
            .declare_original_input(
                &registry,
                &plan,
                &plan.prefill,
                "state.source",
                bounds,
                &prior,
                &owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap(),
    ];
    let frozen = || {
        let mut graph = CompactGraph::with_bound_inputs(
            &original,
            validated(),
            &registry,
            &plan,
            &plan.prefill,
            0,
            &bindings,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
        let first = graph.lower_next().unwrap();
        graph.lower_next().unwrap();
        graph.freeze(vec![first], MAX_PROGRAM_BYTES).unwrap()
    };
    let region = frozen();
    let literal = |region: &FrozenRegion<'_>| {
        let mut b = builder();
        let input = b
            .push(Code::Input("state.source".into()), lanes, bounds)
            .unwrap();
        let mut view = b.view(input, vec![1, 1, lanes]);
        let mut exported = None;
        let mut boundary = 0;
        for index in 0..2 {
            let prepared = region.prepared(index).unwrap();
            let full = prepared.program_at(0).unwrap().clone();
            let remainder = lanes % full.lanes();
            let tail = if remainder == 0 {
                None
            } else {
                Some(prepared.program_at(lanes - remainder).unwrap().clone())
            };
            let gathered = b.gather(&view.words).unwrap();
            let node = b
                .push(
                    Code::TensorLoop {
                        input: LoopInput::Slice(gathered),
                        program: full,
                        tail,
                    },
                    lanes,
                    prepared.output_bounds().unwrap(),
                )
                .unwrap();
            view = b.view(node, vec![1, 1, lanes]);
            if index == 0 {
                exported = Some(view.clone());
                boundary = b.nodes.len();
            }
        }
        let program = b
            .finish(
                1,
                exported.unwrap().words.iter().collect(),
                &mut AdmissionSite::default(),
                |h| {
                    h.update(b"genuine-two-prefix-alias-oracle");
                    Ok(())
                },
                || source_binding(&sources),
            )
            .unwrap();
        (Arc::new(program), boundary)
    };
    let (program, boundary) = literal(&region);
    // Same native Arcs/hash/IDs, independent REAL original Builder + Program Arc.
    let (foreign_program, _) = literal(&region);
    assert_eq!(program.digest(), foreign_program.digest());
    assert!(!Arc::ptr_eq(&program, &foreign_program));
    let mut destination = builder();
    let destination_inputs = vec![
        destination
            .declare_original_input(
                &registry,
                &plan,
                &plan.decode,
                "state.destination",
                bounds,
                &prior,
                &owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap(),
    ];
    let pending = region
        .into_prefix(
            &destination,
            destination_inputs,
            &registry,
            &plan,
            &plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    let mut foreign_destination = builder();
    let foreign_inputs = vec![
        foreign_destination
            .declare_original_input(
                &registry,
                &plan,
                &plan.decode,
                "state.destination",
                bounds,
                &prior,
                &owners,
                MAX_PROGRAM_BYTES,
            )
            .unwrap(),
    ];
    let foreign = frozen()
        .into_prefix(
            &foreign_destination,
            foreign_inputs,
            &registry,
            &plan,
            &plan.decode,
            &prior,
            &owners,
            MAX_PROGRAM_BYTES,
        )
        .unwrap();
    run(pending, foreign, program, foreign_program, boundary)
}
