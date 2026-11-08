//! Source-only regression definitions. No runtime/authority claims.
use super::checkpoint_requirements::*;
use super::{lower_qwen_decoder, DecoderGraph, DecoderMode, DecoderPlan,
    DecoderWorkload, ModelOperation, ModelOperator, QwenConfig, DECODER_PLAN_SCHEMA_VERSION};
use pllm_types::canonical_digest;
use pllm_types::bounded_canonical::borrowed::NativeFrame;
use serde_json::{json, Value};

// Per-thread counting keeps concurrent ordinary tests outside the measured scope.
// This allocator exists only in the library's test binary, never production.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
fn count_allocation() {
    let _ = COUNTING.try_with(|enabled| {
        if enabled.get() { let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1)); }
    });
}
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn limits(plan: &DecoderPlan) -> Limits {
    let mut limits = Limits {
        max_operations: plan.prefill.operations.len() + plan.decode.operations.len(),
        max_inputs: 4096, max_dimensions: 4096, max_rank: 16,
        max_string_bytes: 512, max_attribute_nodes: 8192,
        max_attribute_bytes: 65536, max_attribute_depth: 17,
        max_work: usize::MAX, max_scratch_bytes: usize::MAX,
    };
    let q = quote(limits).unwrap();
    limits.max_work = q.work_ceiling;
    limits.max_scratch_bytes = q.scratch_bytes;
    limits
}
fn inspect(plan: &DecoderPlan) -> Result<(Vec<(String, Vec<u64>)>, Census, [u64; 2]), Error> {
    inspect_with_limits(plan, limits(plan))
}
fn inspect_with_limits(plan: &DecoderPlan, mut l: Limits)
    -> Result<(Vec<(String, Vec<u64>)>, Census, [u64; 2]), Error> {
    let q = quote(l)?;
    l.max_work = q.work_ceiling;
    l.max_scratch_bytes = q.scratch_bytes;
    let mut operations = vec![OperationSlot::empty(); q.operation_slots];
    let mut requirements = vec![Requirement::empty(); q.requirement_slots];
    let mut attributes: Vec<_> = (0..q.attribute_frame_slots).map(|_| NativeFrame::empty()).collect();
    let view = check(plan, l, Scratch { operations: &mut operations,
        requirements: &mut requirements, attributes: &mut attributes })?;
    Ok((view.records().iter().map(|r| (r.name.to_owned(), r.shape.dimensions().to_vec())).collect(),
        view.census, [view.prefill_vocabulary, view.decode_vocabulary]))
}
fn op(id: &str, operator: ModelOperator, inputs: &[&str], shape: &[u64], attributes: Value) -> ModelOperation {
    ModelOperation { id: id.into(), operator, layer: None, state_kind: None,
        inputs: inputs.iter().map(|s| (*s).into()).collect(), output_shape: shape.to_vec(), attributes }
}
fn graph(mode: DecoderMode) -> DecoderGraph {
    DecoderGraph { mode, batch: 1, query_sequence: 1, maximum_key_sequence: 2,
        operations: vec![
            op("tokens", ModelOperator::TokenLookup, &["input.tokens"], &[1, 1, 4], json!({"weight":"table"})),
            op("head", ModelOperator::OutputHead, &["tokens"], &[1, 8], json!({"weight":"table", "bias":null})),
        ], state_inputs: vec![], state_outputs: vec![], output: "head".into() }
}
fn plan() -> DecoderPlan {
    DecoderPlan { schema_version: DECODER_PLAN_SCHEMA_VERSION.into(), model_family: "neutral".into(),
        adapter: "test".into(), config_digest: canonical_digest("checkpoint-requirements-test", &json!({})),
        transformations: vec![], prefill: graph(DecoderMode::Prefill), decode: graph(DecoderMode::Decode), token_feedback: true }
}
fn append_both(p: &mut DecoderPlan, operation: ModelOperation) {
    p.prefill.operations.push(operation.clone());
    p.decode.operations.push(operation);
}

fn absolute_plan() -> DecoderPlan {
    let source = json!({"model_type":"gpt2", "n_embd":4, "n_head":2, "n_layer":1,
        "n_positions":7, "vocab_size":11, "n_inner":9});
    crate::source_mappings::lower_json(&serde_json::to_vec(&source).unwrap(),
        DecoderWorkload { batch:1, max_input_tokens:2, max_new_tokens:2 }, None).unwrap()
}

#[test]
fn actual_absolute_position_source_requires_every_checkpoint_tensor() {
    let p = absolute_plan();
    let before = pllm_types::canonical_bytes(&p);
    let digest = p.digest();
    let mut expected: Vec<(String, Vec<u64>)> = vec![
        ("transformer.wte.weight".into(), vec![11, 4]),
        ("transformer.wpe.weight".into(), vec![7, 4]),
        ("transformer.h.0.ln_1.weight".into(), vec![4]),
        ("transformer.h.0.ln_1.bias".into(), vec![4]),
        ("transformer.h.0.ln_2.weight".into(), vec![4]),
        ("transformer.h.0.ln_2.bias".into(), vec![4]),
        ("transformer.ln_f.weight".into(), vec![4]),
        ("transformer.ln_f.bias".into(), vec![4]),
        ("transformer.h.0.attn.c_attn.weight".into(), vec![4, 12]),
        ("transformer.h.0.attn.c_attn.bias".into(), vec![12]),
        ("transformer.h.0.attn.c_proj.weight".into(), vec![4, 4]),
        ("transformer.h.0.attn.c_proj.bias".into(), vec![4]),
        ("transformer.h.0.mlp.c_fc.weight".into(), vec![4, 9]),
        ("transformer.h.0.mlp.c_fc.bias".into(), vec![9]),
        ("transformer.h.0.mlp.c_proj.weight".into(), vec![9, 4]),
        ("transformer.h.0.mlp.c_proj.bias".into(), vec![4]),
    ];
    expected.sort();
    let (rows, census, vocab) = inspect(&p).unwrap();
    assert_eq!(rows, expected);
    assert_eq!(vocab, [11, 11]);
    assert_eq!(census.materialized_records, 34);
    assert_eq!(census.unique_records, 16);
    assert!(census.work <= quote(limits(&p)).unwrap().work_ceiling);
    assert_eq!(pllm_types::canonical_bytes(&p), before);
    assert_eq!(p.digest(), digest);
    p.validate().unwrap();
}

#[test]
fn embedding_domains_use_declared_cardinality_not_head_vocabulary() {
    let mut p = plan();
    for graph in [&mut p.prefill, &mut p.decode] {
        graph.operations[0].attributes = json!({"weight":"token.table", "index_domain":"tokens", "weight_shape":[13,4]});
        graph.operations[1].attributes["weight"] = json!("head.table");
    }
    append_both(&mut p, op("positions", ModelOperator::TokenLookup, &["input.positions"], &[1,1,4],
        json!({"weight":"position.table", "index_domain":"absolute_positions", "weight_shape":[7,4]})));
    append_both(&mut p, op("segments", ModelOperator::TokenLookup, &["input.segments"], &[1,1,4],
        json!({"weight":"segment.table", "index_domain":"segments", "weight_shape":[2,4]})));
    let (rows, _, vocab) = inspect(&p).unwrap();
    assert_eq!(vocab, [8,8]);
    assert_eq!(rows, vec![("head.table".into(), vec![8,4]), ("position.table".into(), vec![7,4]),
        ("segment.table".into(), vec![2,4]), ("token.table".into(), vec![13,4])]);
}

#[test]
fn ambiguous_malformed_and_cross_domain_embedding_declarations_reject() {
    for (input, attrs, error) in [
        ("input.positions", json!({"weight":"p", "weight_shape":[7,4]}), Error::TokenSource),
        ("input.positions", json!({"weight":"p", "index_domain":"tokens", "weight_shape":[7,4]}), Error::TokenSource),
        ("input.tokens", json!({"index_domain":null}), Error::TokenSource),
        ("input.tokens", json!({"index_domain":"tokens"}), Error::Contract),
        ("input.other", json!({"index_domain":"segments", "weight_shape":[2,4]}), Error::TokenSource),
        ("input.positions", json!({"index_domain":"absolute_positions"}), Error::Contract),
        ("input.segments", json!({"index_domain":"segments", "weight_shape":[0,4]}), Error::Contract),
        ("input.positions", json!({"index_domain":"absolute_positions", "weight_shape":[7,5]}), Error::Contract),
        ("input.positions", json!({"index_domain":"absolute_positions", "weight_shape":[7,4,1]}), Error::Contract),
        ("input.positions", json!({"index_domain":"absolute_positions", "weight_shape":[18446744073709551616.0,4]}), Error::Contract),
        ("input.tokens", json!({"weight_shape":null}), Error::Contract),
    ] {
        let mut p = plan();
        append_both(&mut p, op("bad.embedding", ModelOperator::TokenLookup, &[input], &[1,1,4], attrs));
        assert_eq!(inspect(&p).unwrap_err(), error);
    }
}

#[test]
fn projection_layout_and_declared_shapes_are_checked_not_transposed_by_assumption() {
    let mut p = plan();
    append_both(&mut p, op("projection", ModelOperator::Linear, &["tokens"], &[1,6],
        json!({"weight":"projection", "bias":"bias", "weight_layout":"input_output", "weight_shape":[4,6], "bias_shape":[6]})));
    assert!(inspect(&p).unwrap().0.contains(&("projection".into(), vec![4,6])));
    for (key, value) in [("weight_shape", json!([6,4])), ("bias_shape", json!([4])),
        ("weight_layout", json!("unknown"))] {
        let mut bad = p.clone();
        bad.decode.operations.last_mut().unwrap().attributes[key] = value;
        assert_eq!(inspect(&bad).unwrap_err(), Error::Contract);
    }
}

#[test]
fn shared_layer_norm_contract_rejects_malformed_numeric_and_affine_declarations() {
    let p = absolute_plan();
    for (key, value) in [("epsilon", json!("NaN")), ("epsilon", json!("0")),
        ("epsilon", json!(0.00001)), ("variance", json!("sample_variance")),
        ("epsilon_placement", json!("outside_sqrt")), ("affine_order", json!("add_multiply")),
        ("compute_dtype", json!("bfloat16")), ("axis", json!(0)),
        ("weight_shape", json!([1,4])), ("bias_shape", json!([3])),
        ("weight", Value::Null), ("bias", json!(""))] {
        let mut bad = p.clone();
        bad.prefill.operations.iter_mut().find(|op| op.operator == ModelOperator::LayerNorm)
            .unwrap().attributes[key] = value;
        assert_eq!(inspect(&bad).unwrap_err(), Error::Contract);
        assert!(bad.validate().is_err());
    }
}

#[test]
fn native_positive_f32_parser_preserves_direct_decimal_rounding_and_grammar() {
    let text = "1.0000000596046447753906250000000001";
    let native = super::parse_positive_f32(text).unwrap();
    assert_eq!(native.to_bits(), 0x3f80_0001);
    assert_eq!((text.parse::<f64>().unwrap() as f32).to_bits(), 0x3f80_0000);
    for text in ["+1e-5", "1.", ".5", "1e-45", "3.4028235e38"] {
        assert_eq!(super::parse_positive_f32(text).unwrap().to_bits(), text.parse::<f32>().unwrap().to_bits());
    }
    assert_eq!(super::parse_positive_f32("1e-45").unwrap().to_bits(), 1);
    for text in ["", "bad", " 1", "1 ", "1e-46", "1e-1000", "3.4028236e38", "1e1000",
        "NaN", "+NaN", "inf", "-inf", "Infinity", "0", "-0", "-1"] {
        assert!(super::parse_positive_f32(text).is_err(), "{text}");
    }
}

#[test]
fn shared_layer_norm_requires_gamma_beta_and_optional_output_dtype_is_strict_fp32() {
    let p = absolute_plan();
    let original = p.prefill.operations.iter().find(|op| op.operator == ModelOperator::LayerNorm).unwrap();
    assert!(original.attributes.get("output_dtype").is_none());
    assert!(super::is_centered_affine_layer_norm(original));
    for (key, value) in [("output_dtype", json!("float32")),
        ("epsilon", json!("1.0000000596046447753906250000000001"))] {
        let mut changed = p.clone();
        let norm = changed.prefill.operations.iter_mut().find(|op| op.id == original.id).unwrap();
        norm.attributes[key] = value;
        assert!(super::is_centered_affine_layer_norm(norm));
        assert!(inspect(&changed).is_ok());
    }
    for dtype in [Value::Null, json!(false), json!(32), json!([]), json!({}),
        json!("bfloat16"), json!("float64"), json!("model_native")] {
        let mut bad = p.clone();
        let norm = bad.prefill.operations.iter_mut().find(|op| op.id == original.id).unwrap();
        norm.attributes["output_dtype"] = dtype;
        assert!(!super::is_centered_affine_layer_norm(norm));
        assert_eq!(inspect(&bad).unwrap_err(), Error::Contract);
    }
    for key in ["weight", "bias"] {
        for missing in [false, true] {
            let mut bad = p.clone();
            let norm = bad.prefill.operations.iter_mut().find(|op| op.id == original.id).unwrap();
            if missing { norm.attributes.as_object_mut().unwrap().remove(key); }
            else { norm.attributes[key] = Value::Null; }
            assert!(!super::is_centered_affine_layer_norm(norm));
            assert_eq!(inspect(&bad).unwrap_err(), Error::Contract);
        }
    }
    // Descriptive checks do not rewrite original source attrs or alter its
    // exact graph/digest replay contract.
    p.validate().unwrap();
    assert!(original.attributes.get("output_dtype").is_none());
}

#[test]
fn actual_absolute_source_exact_quote_and_one_short_refuse_before_storage_writes() {
    let p = absolute_plan();
    let l = limits(&p);
    let q = quote(l).unwrap();
    assert_eq!(q.requirement_slots, 2 * q.operation_slots);
    for case in 0..6 {
        let mut l = l;
        let mut operations = vec![OperationSlot::empty(); q.operation_slots - usize::from(case == 2)];
        let before = format!("{operations:?}");
        let mut requirements = vec![Requirement::empty(); q.requirement_slots - usize::from(case == 3)];
        for record in &mut requirements { record.name = "untouched"; }
        let mut attributes: Vec<_> = (0..q.attribute_frame_slots - usize::from(case == 4))
            .map(|_| NativeFrame::empty()).collect();
        if case == 0 { l.max_work -= 1; }
        if case == 1 { l.max_scratch_bytes -= 1; }
        let result = check(&p, l, Scratch { operations:&mut operations,
            requirements:&mut requirements, attributes:&mut attributes });
        if case == 5 { assert!(result.is_ok()); }
        else {
            assert_eq!(result.unwrap_err(), if case == 0 { Error::Work } else { Error::Scratch });
            assert_eq!(format!("{operations:?}"), before);
            assert!(requirements.iter().all(|r| r.name == "untouched"));
        }
    }
}

#[test]
fn actual_absolute_source_global_attribute_census_exact_and_one_short() {
    let p = absolute_plan();
    let (_, census, _) = inspect(&p).unwrap();
    let mut l = limits(&p);
    l.max_inputs = census.inputs;
    l.max_dimensions = census.dimensions;
    l.max_attribute_nodes = census.attribute_nodes;
    l.max_attribute_bytes = census.attribute_bytes;
    assert_eq!(inspect_with_limits(&p, l).unwrap().1.attribute_nodes, census.attribute_nodes);
    for (field, expected) in [(0, Error::Inputs), (1, Error::Dimensions),
        (2, Error::Json(pllm_types::bounded_canonical::borrowed::Error::NodeLimit))] {
        let mut short = l;
        match field { 0 => short.max_inputs -= 1, 1 => short.max_dimensions -= 1,
            2 => short.max_attribute_nodes -= 1, _ => unreachable!() }
        assert_eq!(inspect_with_limits(&p, short).unwrap_err(), expected);
    }
    let mut short = l;
    short.max_attribute_bytes -= 1;
    assert!(matches!(inspect_with_limits(&p, short), Err(Error::Json(_))));
    let mut overflow = l;
    overflow.max_operations = usize::MAX / MAX_REQUIREMENTS_PER_OPERATION + 1;
    assert_eq!(quote(overflow).unwrap_err(), Error::Overflow);
}

#[test]
fn actual_absolute_source_check_allocates_nothing_after_storage_exists() {
    let p = absolute_plan();
    let l = limits(&p);
    let q = quote(l).unwrap();
    let mut operations = vec![OperationSlot::empty(); q.operation_slots];
    let mut requirements = vec![Requirement::empty(); q.requirement_slots];
    let mut attributes: Vec<_> = (0..q.attribute_frame_slots).map(|_| NativeFrame::empty()).collect();
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|enabled| enabled.set(true));
    let result = check(&p, l, Scratch { operations:&mut operations,
        requirements:&mut requirements, attributes:&mut attributes });
    COUNTING.with(|enabled| enabled.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    assert!(result.is_ok());
    assert_eq!(allocations, 0);
}

#[test]
fn tied_weights_are_deduplicated_with_checked_census() {
    let p = plan();
    let (rows, census, vocab) = inspect(&p).unwrap();
    assert_eq!(rows, vec![("table".to_owned(), vec![8, 4])]);
    assert_eq!(vocab, [8, 8]);
    assert_eq!(census.operations, 4);
    assert_eq!(census.inputs, 4);
    assert_eq!(census.dimensions, 10);
    assert_eq!(census.attribute_nodes, 10);
    assert_eq!(census.materialized_records, 4);
    assert_eq!(census.unique_records, 1);
    assert!(census.work <= quote(limits(&p)).unwrap().work_ceiling);
}

#[test]
fn independent_qwen_gqa_literal_shapes_and_optional_biases() {
    let config = QwenConfig { model_type: "qwen2".into(), hidden_size: 32,
        intermediate_size: 64, num_hidden_layers: 1, num_attention_heads: 4,
        num_key_value_heads: 2, vocab_size: 128, max_position_embeddings: 64,
        hidden_act: "silu".into(), rms_norm_eps: "0.000001".into(), rope_theta: 10000,
        tie_word_embeddings: true, sliding_window: None, use_sliding_window: None };
    let mut p = lower_qwen_decoder(&config, DecoderWorkload {
        batch: 1, max_input_tokens: 2, max_new_tokens: 1 }).unwrap();
    // Independent reference literals, not computed by another requirements pass.
    let mut expected: Vec<(String, Vec<u64>)> = vec![
        ("model.embed_tokens.weight".into(), vec![128, 32]),
        ("model.layers.0.input_layernorm.weight".into(), vec![32]),
        ("model.layers.0.self_attn.q_proj.weight".into(), vec![32, 32]),
        ("model.layers.0.self_attn.q_proj.bias".into(), vec![32]),
        ("model.layers.0.self_attn.k_proj.weight".into(), vec![16, 32]),
        ("model.layers.0.self_attn.k_proj.bias".into(), vec![16]),
        ("model.layers.0.self_attn.v_proj.weight".into(), vec![16, 32]),
        ("model.layers.0.self_attn.v_proj.bias".into(), vec![16]),
        ("model.layers.0.self_attn.o_proj.weight".into(), vec![32, 32]),
        ("model.layers.0.post_attention_layernorm.weight".into(), vec![32]),
        ("model.layers.0.mlp.gate_proj.weight".into(), vec![64, 32]),
        ("model.layers.0.mlp.up_proj.weight".into(), vec![64, 32]),
        ("model.layers.0.mlp.down_proj.weight".into(), vec![32, 64]),
        ("model.norm.weight".into(), vec![32]),
    ];
    expected.sort();
    assert_eq!(inspect(&p).unwrap().0, expected);
    for graph in [&mut p.prefill, &mut p.decode] {
        for op in &mut graph.operations {
            if op.operator == ModelOperator::Linear { op.attributes["bias"] = Value::Null; }
        }
    }
    expected.retain(|(name, _)| !name.ends_with(".bias"));
    assert_eq!(inspect(&p).unwrap().0, expected);
}

#[test]
fn both_vocabularies_exposed_without_equality_policy() {
    let mut p = plan();
    p.decode.operations[1].output_shape = vec![1, 9];
    p.decode.operations[0].attributes["weight"] = json!("decode.table");
    p.decode.operations[1].attributes["weight"] = json!("decode.head");
    let (rows, _, vocab) = inspect(&p).unwrap();
    assert_eq!(vocab, [8, 9]);
    assert_eq!(rows, vec![("decode.head".to_owned(), vec![9, 4]),
        ("decode.table".to_owned(), vec![9, 4]), ("table".to_owned(), vec![8, 4])]);
    p.decode.operations[1].attributes["weight"] = json!("table");
    assert_eq!(inspect(&p).unwrap_err(), Error::Conflict);
}

#[test]
fn local_norm_decay_scalar_and_convolution_literal_contracts() {
    let mut p = plan();
    append_both(&mut p, op("norm", ModelOperator::RmsNorm, &["tokens"], &[1, 4], json!({"weight":"norm", "weight_offset":1})));
    append_both(&mut p, op("unweighted", ModelOperator::RmsNorm, &["tokens"], &[1, 4], json!({"weight":null, "with_scale":false, "output_dtype":"bfloat16"})));
    append_both(&mut p, op("gated", ModelOperator::RmsNormGated, &["tokens"], &[1, 4], json!({"weight":"gated", "activation":"silu", "weight_offset":0, "norm_before_gate":true})));
    append_both(&mut p, op("decay", ModelOperator::GatedDeltaDecay, &["tokens"], &[1, 2], json!({"a_log":"a", "dt_bias":"dt", "formula":"-exp(A_log)*softplus(a+dt_bias)", "compute_dtype":"float32"})));
    append_both(&mut p, op("scalar", ModelOperator::Scale, &["tokens"], &[1, 4], json!({"factor":{"kind":"checkpoint_scalar", "weight":"scalar", "weight_shape":[1]}})));
    append_both(&mut p, op("conv", ModelOperator::CausalConvolution, &["tokens"], &[1, 4], json!({"weight":"conv", "bias":null, "groups":4, "kernel_size":3})));
    append_both(&mut p, op("update", ModelOperator::ConvolutionStateUpdate, &["tokens"], &[1, 4, 3], json!({"weight":"conv", "groups":4, "kernel_size":3})));
    assert_eq!(inspect(&p).unwrap().0, vec![
        ("a".to_owned(), vec![2]), ("conv".to_owned(), vec![4, 1, 3]), ("dt".to_owned(), vec![2]),
        ("gated".to_owned(), vec![4]), ("norm".to_owned(), vec![4]),
        ("scalar".to_owned(), vec![1]), ("table".to_owned(), vec![8, 4])]);
    // Decode-only local/remote name collision cannot hide behind phase dedup.
    let last = p.decode.operations.last_mut().unwrap();
    last.attributes["weight"] = json!("table");
    assert_eq!(inspect(&p).unwrap_err(), Error::Conflict);
}

#[test]
fn head_token_projection_and_duplicate_identity_fail_closed() {
    let mut p = plan();
    p.decode.operations.pop();
    assert_eq!(inspect(&p).unwrap_err(), Error::Head);
    let mut p = plan();
    p.decode.operations.push(op("second", ModelOperator::OutputHead, &["tokens"], &[1, 8], json!({})));
    assert_eq!(inspect(&p).unwrap_err(), Error::Head);
    let mut p = plan();
    p.decode.operations[0].inputs.push("extra".into());
    assert_eq!(inspect(&p).unwrap_err(), Error::TokenSource);
    let mut p = plan();
    p.decode.operations[1].inputs[0] = "input.tokens".into();
    assert_eq!(inspect(&p).unwrap_err(), Error::ProjectionSource);
    let mut p = plan();
    p.decode.operations[1].inputs.clear();
    assert_eq!(inspect(&p).unwrap_err(), Error::ProjectionSource);
    let mut p = plan();
    p.decode.operations[1].id = "tokens".into();
    assert_eq!(inspect(&p).unwrap_err(), Error::DuplicateId);
}

#[test]
fn malformed_attributes_ranks_and_local_contracts() {
    for bad in [json!(""), json!(false), json!(3), json!([])] {
        let mut p = plan();
        p.decode.operations[0].attributes["weight"] = bad;
        assert_eq!(inspect(&p).unwrap_err(), Error::Artifact);
    }
    let mut p = plan();
    p.decode.operations[0].attributes = json!([]);
    assert_eq!(inspect(&p).unwrap_err(), Error::Attributes);
    let mut p = plan();
    p.decode.operations[0].attributes["unknown_weight"] = json!("hidden");
    assert_eq!(inspect(&p).unwrap_err(), Error::UnknownAttribute);
    let mut p = plan();
    p.decode.operations[0].output_shape.clear();
    assert_eq!(inspect(&p).unwrap_err(), Error::Shape);
    let mut p = plan();
    p.decode.operations[0].output_shape[0] = 0;
    assert_eq!(inspect(&p).unwrap_err(), Error::Shape);
    let mut p = plan();
    append_both(&mut p, op("unsupported", ModelOperator::LayerNorm, &["tokens"], &[4], json!({})));
    assert_eq!(inspect(&p).unwrap_err(), Error::Contract);
    let bad_locals = [
        op("bad", ModelOperator::RmsNorm, &[], &[4], json!({"weight":null})),
        op("bad", ModelOperator::RmsNormGated, &[], &[4], json!({"weight":"n", "activation":"relu", "weight_offset":0, "norm_before_gate":true})),
        op("bad", ModelOperator::GatedDeltaDecay, &[], &[4], json!({"a_log":"a", "dt_bias":"d", "formula":"other", "compute_dtype":"float32"})),
        op("bad", ModelOperator::Scale, &[], &[4], json!({"factor":{"kind":"checkpoint_scalar", "weight":"s", "weight_shape":[2]}})),
        op("bad", ModelOperator::CausalConvolution, &[], &[4], json!({"weight":"c", "groups":8193, "kernel_size":3})),
        op("bad", ModelOperator::CausalConvolution, &[], &[4], json!({"weight":"c", "groups":4, "kernel_size":0})),
        op("bad", ModelOperator::CausalConvolution, &[], &[4], json!({"weight":"c", "groups":4, "kernel_size":3, "bias":"b"})),
    ];
    for bad in bad_locals {
        let mut p = plan();
        append_both(&mut p, bad);
        assert_eq!(inspect(&p).unwrap_err(), Error::Contract);
    }
}

#[test]
fn all_weight_bias_and_local_name_conflicts_reject() {
    let mut p = plan();
    p.decode.operations[1].attributes["bias"] = json!("table");
    assert_eq!(inspect(&p).unwrap_err(), Error::Conflict);
    let mut p = plan();
    append_both(&mut p, op("norm", ModelOperator::RmsNorm, &[], &[4], json!({"weight":"table"})));
    assert_eq!(inspect(&p).unwrap_err(), Error::Conflict);
    let mut p = plan();
    append_both(&mut p, op("linear", ModelOperator::Linear, &["tokens"], &[4], json!({"weight":"same", "bias":"same"})));
    assert_eq!(inspect(&p).unwrap_err(), Error::Conflict);
}

#[test]
fn attribute_depth_nodes_and_payload_are_bounded() {
    let p = plan();
    for case in 0..3 {
        let mut l = limits(&p);
        match case {
            0 => l.max_attribute_depth = 1,
            1 => l.max_attribute_nodes = 1,
            2 => l.max_attribute_bytes = 1,
            _ => unreachable!(),
        }
        let q = quote(l).unwrap();
        l.max_work = q.work_ceiling;
        l.max_scratch_bytes = q.scratch_bytes;
        let mut operations = vec![OperationSlot::empty(); q.operation_slots];
        let mut requirements = vec![Requirement::empty(); q.requirement_slots];
        let mut attributes: Vec<_> = (0..q.attribute_frame_slots).map(|_| NativeFrame::empty()).collect();
        assert!(matches!(check(&p, l, Scratch { operations: &mut operations,
            requirements: &mut requirements, attributes: &mut attributes }), Err(Error::Json(_))));
    }
}

#[test]
fn many_small_attribute_trees_share_one_global_envelope() {
    let mut p = plan();
    for i in 0..64 {
        append_both(&mut p, op(&format!("small.{i:02}"), ModelOperator::Reshape,
            &[], &[4], json!({"axis":0})));
    }
    let mut l = limits(&p);
    l.max_attribute_nodes = 10 + 2 * 64 * 2;
    let (_, census, _) = inspect_with_limits(&p, l).unwrap();
    assert_eq!(census.attribute_nodes, l.max_attribute_nodes);
    assert_eq!(census.operations, l.max_operations);
    assert!(census.work <= quote(l).unwrap().work_ceiling);
    // One extra global node funds one envelope increment, not one per op.
    let mut larger = l;
    larger.max_attribute_nodes += 1;
    assert_eq!(quote(larger).unwrap().work_ceiling - quote(l).unwrap().work_ceiling,
        512 * (l.max_string_bytes + 1) + 64);
}

#[test]
fn one_dominant_attribute_tree_shares_envelope_with_small_trees() {
    let mut p = plan();
    p.prefill.operations.push(op("dominant", ModelOperator::Scale, &[], &[4],
        json!({"factor":{"kind":"literal", "numerator":vec![0; 4096]}})));
    let mut l = limits(&p);
    // factor object, kind leaf, numerator array and its leaves, plus root.
    l.max_attribute_nodes = 10 + 4100;
    let (_, census, _) = inspect_with_limits(&p, l).unwrap();
    assert_eq!(census.attribute_nodes, l.max_attribute_nodes);
    assert!(census.work <= quote(l).unwrap().work_ceiling);
}

#[test]
fn attribute_node_limit_is_cumulative_across_operations_and_phases() {
    let p = plan();
    let mut l = limits(&p);
    l.max_attribute_nodes = 10;
    assert_eq!(inspect_with_limits(&p, l).unwrap().1.attribute_nodes, 10);
    l.max_attribute_nodes = 9; // Every individual tree has only two or three nodes.
    assert_eq!(inspect_with_limits(&p, l).unwrap_err(),
        Error::Json(pllm_types::bounded_canonical::borrowed::Error::NodeLimit));
}

#[test]
fn global_attribute_envelope_add_multiply_and_total_overflow_are_checked() {
    let p = plan();
    let mut l = limits(&p);
    l.max_attribute_nodes = usize::MAX - l.max_operations + 1;
    assert!(l.max_attribute_nodes.checked_add(l.max_operations).is_none());
    assert_eq!(quote(l).unwrap_err(), Error::Overflow);

    // With no operation storage, isolate each fixed-envelope multiplication.
    l.max_operations = 0;
    l.max_string_bytes = 0;
    l.max_attribute_nodes = usize::MAX / 512 + 1;
    assert!(512usize.checked_mul(l.max_attribute_nodes).is_none());
    assert_eq!(quote(l).unwrap_err(), Error::Overflow);
    l.max_attribute_nodes = 1;
    l.max_string_bytes = usize::MAX / 512;
    assert!(512usize.checked_mul(l.max_string_bytes + 1).is_none());
    assert_eq!(quote(l).unwrap_err(), Error::Overflow);

    // Both terms fit separately; their checked total does not.
    l.max_string_bytes = 0;
    l.max_attribute_nodes = usize::MAX / (512 + 64) + 1;
    let fixed = 512usize.checked_mul(l.max_attribute_nodes).unwrap();
    let attributes = 64usize.checked_mul(l.max_attribute_nodes + l.max_attribute_bytes).unwrap();
    assert!(fixed.checked_add(attributes).is_none());
    assert_eq!(quote(l).unwrap_err(), Error::Overflow);
}

#[test]
fn check_allocates_nothing_after_caller_storage_exists() {
    let mut p = plan();
    append_both(&mut p, op("norm", ModelOperator::RmsNormGated, &[], &[4],
        json!({"weight":"norm", "activation":"silu", "weight_offset":0, "norm_before_gate":true})));
    append_both(&mut p, op("decay", ModelOperator::GatedDeltaDecay, &[], &[2],
        json!({"a_log":"a", "dt_bias":"dt", "formula":"-exp(A_log)*softplus(a+dt_bias)", "compute_dtype":"float32"})));
    append_both(&mut p, op("scalar", ModelOperator::Scale, &[], &[4],
        json!({"factor":{"kind":"checkpoint_scalar", "weight":"scalar", "weight_shape":[1]}})));
    append_both(&mut p, op("conv", ModelOperator::CausalConvolution, &[], &[4],
        json!({"weight":"conv", "groups":4, "kernel_size":3})));
    let l = limits(&p);
    let q = quote(l).unwrap();
    let mut operations = vec![OperationSlot::empty(); q.operation_slots];
    let mut requirements = vec![Requirement::empty(); q.requirement_slots];
    let mut attributes: Vec<_> = (0..q.attribute_frame_slots).map(|_| NativeFrame::empty()).collect();
    ALLOCATIONS.with(|count| count.set(0));
    COUNTING.with(|enabled| enabled.set(true));
    let result = check(&p, l, Scratch { operations: &mut operations,
        requirements: &mut requirements, attributes: &mut attributes });
    COUNTING.with(|enabled| enabled.set(false));
    let allocations = ALLOCATIONS.with(Cell::get);
    assert!(result.is_ok());
    assert_eq!(allocations, 0);
}

#[test]
fn missing_and_null_remote_artifacts_are_optional() {
    let mut p = plan();
    for graph in [&mut p.prefill, &mut p.decode] {
        graph.operations[0].attributes = json!({});
        graph.operations[1].attributes = json!({"weight":null,"bias":null});
    }
    assert!(inspect(&p).unwrap().0.is_empty());
}

#[test]
fn exact_and_one_short_work_bytes_and_each_pool_before_writes() {
    let p = plan();
    let l = limits(&p);
    let q = quote(l).unwrap();
    for case in 0..6 {
        let mut l = l;
        let mut operations = vec![OperationSlot::empty(); q.operation_slots - usize::from(case == 2)];
        let operations_before = format!("{operations:?}");
        let mut requirements = vec![Requirement::empty(); q.requirement_slots - usize::from(case == 3)];
        for r in &mut requirements { r.name = "untouched"; }
        let mut attributes: Vec<_> = (0..q.attribute_frame_slots - usize::from(case == 4))
            .map(|_| NativeFrame::empty()).collect();
        if case == 0 { l.max_work -= 1; }
        if case == 1 { l.max_scratch_bytes -= 1; }
        let result = check(&p, l, Scratch { operations: &mut operations,
            requirements: &mut requirements, attributes: &mut attributes });
        if case == 5 { assert!(result.is_ok()); }
        else {
            assert_eq!(result.unwrap_err(), if case == 0 { Error::Work } else { Error::Scratch });
            assert_eq!(format!("{operations:?}"), operations_before);
            assert!(requirements.iter().all(|r| r.name == "untouched"));
        }
    }
}

#[test]
fn checked_overflow_and_collection_limits() {
    let p = plan();
    let mut l = limits(&p);
    l.max_operations = usize::MAX;
    assert_eq!(quote(l).unwrap_err(), Error::Overflow);
    let mut l = limits(&p);
    l.max_string_bytes = usize::MAX;
    assert_eq!(quote(l).unwrap_err(), Error::Overflow);
    for (case, expected) in [(0, Error::Operations), (1, Error::Inputs),
        (2, Error::Dimensions), (3, Error::Rank), (4, Error::String)] {
        let mut l = limits(&p);
        match case { 0 => l.max_operations -= 1, 1 => l.max_inputs = 0,
            2 => l.max_dimensions = 0, 3 => l.max_rank = 1,
            4 => l.max_string_bytes = 1, _ => unreachable!() }
        let q = quote(l).unwrap();
        l.max_work = q.work_ceiling;
        l.max_scratch_bytes = q.scratch_bytes;
        let mut operations = vec![OperationSlot::empty(); q.operation_slots];
        let mut requirements = vec![Requirement::empty(); q.requirement_slots];
        let mut attributes: Vec<_> = (0..q.attribute_frame_slots).map(|_| NativeFrame::empty()).collect();
        assert_eq!(check(&p, l, Scratch { operations: &mut operations,
            requirements: &mut requirements, attributes: &mut attributes }).unwrap_err(), expected);
    }
}

#[test]
fn adversarial_order_uses_borrowed_heapsort_not_reconstruction_or_heap_sort_library() {
    let mut p = plan();
    for i in (0..32).rev() {
        append_both(&mut p, op(&format!("projection.{i:02}"), ModelOperator::Linear,
            &["tokens"], &[1, 4], json!({"weight":format!("weight.{i:02}"), "bias":"shared.bias"})));
    }
    let (rows, census, _) = inspect(&p).unwrap();
    assert_eq!(rows.len(), 34);
    assert_eq!(census.materialized_records, 132);
    assert!(census.work <= quote(limits(&p)).unwrap().work_ceiling);
    assert!(rows.windows(2).all(|p| p[0].0 < p[1].0));
    // Source guard supplements algorithm tests; not a compiled allocator proof.
    for (filename,source) in [
        ("checkpoint_requirements.rs",include_str!("checkpoint_requirements.rs")),
        ("transformer_blocks.rs",include_str!("transformer_blocks.rs")),
    ] {
        for forbidden in ["Vec::", "Box::", "BTreeMap::", "HashMap::", ".sort(", ".sort_by(",
            "serde_json::from", ".validate(", ".clone("] {
            assert!(!source.contains(forbidden), "{filename}: {forbidden}");
        }
    }
    assert_eq!(MAX_REQUIREMENTS_PER_OPERATION, 2);
}
