use super::*;
use pllm_types::{canonical_bytes, canonical_digest, Digest};
use serde_json::{json, Value};

const OPERATORS: &[ModelOperator] = &[
    ModelOperator::TokenLookup,
    ModelOperator::Reshape,
    ModelOperator::RmsNorm,
    ModelOperator::Linear,
    ModelOperator::RotaryEmbedding,
    ModelOperator::KvCacheAppend,
    ModelOperator::AttentionScores,
    ModelOperator::AttentionScale,
    ModelOperator::CausalMask,
    ModelOperator::Softmax,
    ModelOperator::AttentionValues,
    ModelOperator::ResidualAdd,
    ModelOperator::Silu,
    ModelOperator::Multiply,
    ModelOperator::OutputHead,
    ModelOperator::LastToken,
    ModelOperator::GreedyTokenSelection,
    ModelOperator::TokenFeedback,
    ModelOperator::AttentionImportance,
    ModelOperator::SecureTopK,
    ModelOperator::SecureGather,
    ModelOperator::ClusterBounds,
    ModelOperator::ClusterSimilarity,
    ModelOperator::SharedIndices,
    ModelOperator::CacheActiveIndices,
    ModelOperator::Scale,
    ModelOperator::Slice,
    ModelOperator::GeluTanh,
    ModelOperator::Softcap,
    ModelOperator::CacheSuffix,
    ModelOperator::StateInitialize,
    ModelOperator::Permute,
    ModelOperator::Sigmoid,
    ModelOperator::GatedDeltaDecay,
    ModelOperator::CausalConvolution,
    ModelOperator::ConvolutionStateUpdate,
    ModelOperator::GatedDeltaRule,
    ModelOperator::GatedDeltaStateUpdate,
    ModelOperator::RmsNormGated,
    ModelOperator::LayerNorm,
];
const KINDS: &[StateKind] = &[
    StateKind::Key,
    StateKind::Value,
    StateKind::CacheIndices,
    StateKind::Recurrent,
    StateKind::Convolution,
];

fn limits() -> Limits {
    Limits {
        canonical: StreamLimits {
            max_nodes: 100_000,
            max_depth: 64,
            max_bytes: 1_000_000,
            max_string_bytes: 4096,
            max_key_bytes: 4096,
            max_total_string_bytes: 1_000_000,
            max_work: 10_000_000,
            max_stack_bytes: 64 * StreamFrame::storage_bytes(),
        },
        max_operations: 10_000,
        max_states: 10_000,
        max_inputs: 10_000,
        max_rank: 16,
        max_transformations: 100,
        max_domain_bytes: 100,
    }
}

fn operation() -> ModelOperation {
    ModelOperation {
        id: "op\"\\\n\r\t\0é雪".into(),
        operator: ModelOperator::SecureTopK,
        layer: Some(u64::MAX),
        state_kind: Some(StateKind::Recurrent),
        inputs: vec!["z".into(), "a\u{0008}\u{000c}\u{001f}".into()],
        output_shape: vec![0, 1, u64::MAX],
        attributes: json!({
            "z": [null, true, false, -1, i64::MIN, u64::MAX, -0.0, 1.25, 1e-100, 1e100, f64::MIN_POSITIVE, f64::MAX],
            "a\"\\\n\0": {"雪": "é\t\r\n\u{0001}", "a": []},
            "nested": [{"transformation": [1, 2, 3]}]
        }),
    }
}

fn graph(mode: DecoderMode, n: usize) -> DecoderGraph {
    DecoderGraph {
        mode,
        batch: u64::MAX,
        query_sequence: 1,
        maximum_key_sequence: 0,
        operations: (0..n)
            .map(|index| {
                let mut op = operation();
                op.id = format!("op.{index}");
                op.operator = OPERATORS[index % OPERATORS.len()];
                op.layer = (index % 2 == 0).then_some(index as u64);
                op.state_kind = if index % 6 == 0 {
                    None
                } else {
                    Some(KINDS[index % KINDS.len()])
                };
                op
            })
            .collect(),
        state_inputs: KINDS
            .iter()
            .map(|kind| StateTensor {
                id: "state\n雪".into(),
                layer: None,
                kind: *kind,
                shape: vec![u64::MAX, 0],
                maximum_sequence: u64::MAX,
            })
            .collect(),
        state_outputs: KINDS
            .iter()
            .map(|kind| StateTensor {
                id: "output".into(),
                layer: Some(u64::MAX),
                kind: *kind,
                shape: vec![],
                maximum_sequence: 0,
            })
            .collect(),
        output: "final\"".into(),
    }
}

fn plan(n: usize) -> DecoderPlan {
    DecoderPlan {
        schema_version: DECODER_PLAN_SCHEMA_VERSION.into(),
        model_family: "model\n雪".into(),
        adapter: "adapter\0".into(),
        config_digest: Digest::from_sha256([1; 32]),
        transformations: vec![AppliedMethod {
            component: "component".into(),
            implementation: "implementation".into(),
            method_id: "method\"".into(),
            input_digest: Digest::from_sha256([2; 32]),
            configuration_digest: Digest::from_sha256([3; 32]),
        }],
        prefill: graph(DecoderMode::Prefill, n),
        decode: graph(DecoderMode::Decode, n),
        token_feedback: true,
    }
}

fn operation_parity(op: &ModelOperation) -> StreamUsage {
    let mut stack = [StreamFrame::empty(); 64];
    let mut bytes = Vec::new();
    let usage = bounded_model_operation_write(op, limits(), &mut stack, &mut bytes).unwrap();
    assert_eq!(bytes, canonical_bytes(op));
    for domain in [
        "",
        "pllm.model_operation.v1",
        "existing.operation.identity\0雪",
    ] {
        let digest = bounded_model_operation_digest(op, domain, limits(), &mut stack).unwrap();
        assert_eq!(Digest::from_sha256(digest), canonical_digest(domain, op));
    }
    usage
}

fn plan_parity(plan: &DecoderPlan) -> StreamUsage {
    let mut stack = [StreamFrame::empty(); 64];
    let mut bytes = Vec::new();
    let usage = bounded_plan_write(plan, limits(), &mut stack, &mut bytes).unwrap();
    assert_eq!(bytes, canonical_bytes(plan));
    assert_eq!(usage.bytes, bytes.len());
    let digest = bounded_plan_digest(plan, limits(), &mut stack).unwrap();
    assert_eq!(Digest::from_sha256(digest), plan.digest());
    usage
}

#[test]
fn all_enum_spellings_options_and_attribute_types_match_legacy() {
    for operator in OPERATORS {
        for kind in KINDS.iter().copied().map(Some).chain([None]) {
            for layer in [None, Some(0), Some(u64::MAX)] {
                let mut op = operation();
                op.operator = *operator;
                op.state_kind = kind;
                op.layer = layer;
                operation_parity(&op);
            }
        }
    }
    // Attributes are arbitrary JSON, including a scalar or array, even when the
    // separate semantic model validator would reject them.
    for attributes in [
        Value::Null,
        json!(false),
        json!("雪\n"),
        json!(42),
        json!([]),
        json!({}),
    ] {
        let mut op = operation();
        op.attributes = attributes;
        operation_parity(&op);
    }
    let mut empty = operation();
    empty.id.clear();
    empty.inputs.clear();
    empty.output_shape.clear();
    empty.attributes = Value::Null;
    operation_parity(&empty);
}

#[test]
fn generated_graphs_and_empty_collections_match_legacy() {
    for n in [0, 1, 2, 7, 40, 63] {
        plan_parity(&plan(n));
    }
    let mut empty = plan(0);
    empty.transformations.clear();
    empty.prefill.state_inputs.clear();
    empty.prefill.state_outputs.clear();
    empty.decode.state_inputs.clear();
    empty.decode.state_outputs.clear();
    empty.token_feedback = false;
    plan_parity(&empty);
}

#[test]
fn actual_lowered_decoder_matches_legacy() {
    let config = crate::QwenConfig {
        model_type: "qwen2".into(),
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        vocab_size: 64,
        max_position_embeddings: 128,
        hidden_act: "silu".into(),
        rms_norm_eps: "0.000001".into(),
        rope_theta: 10000,
        tie_word_embeddings: false,
        sliding_window: None,
        use_sliding_window: None,
    };
    let plan = crate::lower_qwen_decoder(
        &config,
        crate::DecoderWorkload {
            batch: 2,
            max_input_tokens: 8,
            max_new_tokens: 4,
        },
    )
    .unwrap();
    plan_parity(&plan);
}

#[test]
fn every_operation_and_plan_field_mutation_keeps_parity_and_changes_identity() {
    let original = operation();
    let baseline = canonical_digest("mutation", &original);
    macro_rules! op_mutation {
        ($op:ident, $change:expr) => {{
            let mut $op = original.clone();
            $change;
            operation_parity(&$op);
            assert_ne!(canonical_digest("mutation", &$op), baseline);
        }};
    }
    op_mutation!(op, op.id.push('x'));
    op_mutation!(op, op.operator = ModelOperator::Linear);
    op_mutation!(op, op.layer = None);
    op_mutation!(op, op.state_kind = None);
    op_mutation!(op, op.inputs.reverse());
    op_mutation!(op, op.output_shape.push(2));
    op_mutation!(op, op.attributes["z"][0] = json!("changed"));

    let original = plan(2);
    let baseline = original.digest();
    macro_rules! plan_mutation {
        ($p:ident, $change:expr) => {{
            let mut $p = original.clone();
            $change;
            plan_parity(&$p);
            assert_ne!($p.digest(), baseline);
        }};
    }
    plan_mutation!(p, p.schema_version.push('x'));
    plan_mutation!(p, p.model_family.push('x'));
    plan_mutation!(p, p.adapter.push('x'));
    plan_mutation!(p, p.config_digest = Digest::from_sha256([9; 32]));
    plan_mutation!(p, p.token_feedback = false);
    plan_mutation!(p, p.transformations[0].component.push('x'));
    plan_mutation!(p, p.transformations[0].implementation.push('x'));
    plan_mutation!(p, p.transformations[0].method_id.push('x'));
    plan_mutation!(
        p,
        p.transformations[0].input_digest = Digest::from_sha256([9; 32])
    );
    plan_mutation!(
        p,
        p.transformations[0].configuration_digest = Digest::from_sha256([9; 32])
    );
    plan_mutation!(p, p.decode.batch -= 1);
    plan_mutation!(p, p.decode.maximum_key_sequence += 1);
    plan_mutation!(p, p.decode.mode = DecoderMode::Prefill);
    plan_mutation!(p, p.decode.output.push('x'));
    plan_mutation!(p, p.decode.query_sequence += 1);
    plan_mutation!(p, p.decode.operations.reverse());
    plan_mutation!(p, p.decode.state_inputs[0].id.push('x'));
    plan_mutation!(p, p.decode.state_inputs[0].kind = StateKind::Convolution);
    plan_mutation!(p, p.decode.state_inputs[0].layer = Some(0));
    plan_mutation!(p, p.decode.state_inputs[0].shape.push(1));
    plan_mutation!(p, p.decode.state_inputs[0].maximum_sequence -= 1);
    plan_mutation!(p, p.decode.state_outputs.reverse());
    plan_mutation!(p, p.prefill.output.push('x'));
}

fn count(value: &Value, depth: usize) -> (usize, usize, usize) {
    let mut result = (1, depth, 0);
    match value {
        Value::Array(values) => {
            for value in values {
                let child = count(value, depth + 1);
                result.0 += child.0;
                result.1 = result.1.max(child.1);
                result.2 += child.2;
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                let child = count(value, depth + 1);
                result.0 += child.0;
                result.1 = result.1.max(child.1);
                result.2 += key.len() + child.2;
            }
        }
        Value::String(value) => result.2 = value.len(),
        _ => {}
    }
    result
}

#[derive(Default)]
struct Untouched(usize);
impl io::Write for Untouched {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 += 1;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn exact_global_schema_plus_attrs_counts_and_boundaries() {
    let p = plan(3);
    let usage = plan_parity(&p);
    let independent = count(&serde_json::to_value(&p).unwrap(), 1);
    assert_eq!((usage.nodes, usage.depth, usage.string_bytes), independent);
    let mut exact = limits();
    exact.canonical.max_bytes = usage.bytes;
    exact.canonical.max_nodes = usage.nodes;
    exact.canonical.max_depth = usage.depth;
    exact.canonical.max_total_string_bytes = usage.string_bytes;
    exact.canonical.max_work = usage.work;
    let mut stack = [StreamFrame::empty(); 64];
    assert_eq!(
        bounded_plan_write(&p, exact, &mut stack, &mut io::sink()).unwrap(),
        usage
    );
    bounded_plan_digest(&p, exact, &mut stack).unwrap();
    for which in 0..6 {
        let mut short = exact;
        match which {
            0 => short.canonical.max_bytes -= 1,
            1 => short.canonical.max_nodes -= 1,
            2 => short.canonical.max_depth -= 1,
            3 => short.canonical.max_total_string_bytes -= 1,
            4 => short.canonical.max_work -= 1,
            _ => short.canonical.max_stack_bytes -= 1,
        }
        let mut output = Untouched::default();
        assert!(bounded_plan_write(&p, short, &mut stack, &mut output).is_err());
        assert_eq!(output.0, 0);
        assert!(bounded_plan_digest(&p, short, &mut stack).is_err());
    }
}

#[test]
fn collection_admission_limits_are_global_and_precede_attrs() {
    let mut p = plan(3);
    // Traversing these attributes would exceed even generous test depth.
    for _ in 0..80 {
        let old = std::mem::replace(&mut p.decode.operations[0].attributes, Value::Null);
        p.decode.operations[0].attributes = Value::Array(vec![old]);
    }
    let mut exact = limits();
    exact.max_operations = 6;
    exact.max_states = 20;
    exact.max_inputs = 12;
    exact.max_rank = 3;
    exact.max_transformations = 1;
    let mut stack = [StreamFrame::empty(); 64];
    for which in 0..5 {
        let mut short = exact;
        match which {
            0 => short.max_operations -= 1,
            1 => short.max_states -= 1,
            2 => short.max_inputs -= 1,
            3 => short.max_rank -= 1,
            _ => short.max_transformations -= 1,
        }
        let mut output = Untouched::default();
        let error = bounded_plan_write(&p, short, &mut stack, &mut output).unwrap_err();
        assert!(matches!(
            error,
            Error::Operations
                | Error::States
                | Error::Inputs
                | Error::Rank
                | Error::Transformations
        ));
        assert_eq!(output.0, 0);
    }
    // Exact collection limits succeed once excessive attribute depth is removed.
    p.decode.operations[0].attributes = Value::Null;
    let mut stack = [StreamFrame::empty(); 64];
    bounded_plan_write(&p, exact, &mut stack, &mut io::sink()).unwrap();
}

#[test]
fn oversized_strings_keys_depth_and_short_stack_do_not_touch_output() {
    let mut op = operation();
    let mut stack = [StreamFrame::empty(); 64];
    op.id = "x".repeat(4097);
    let mut output = Untouched::default();
    assert!(matches!(
        bounded_model_operation_write(&op, limits(), &mut stack, &mut output),
        Err(Error::Canonical(StreamError::StringLimit))
    ));
    assert_eq!(output.0, 0);
    op = operation();
    op.attributes = Value::Object(serde_json::Map::from_iter([("x".repeat(4097), json!(1))]));
    let mut stack = [StreamFrame::empty(); 64];
    assert!(bounded_model_operation_write(&op, limits(), &mut stack, &mut output).is_err());
    assert_eq!(output.0, 0);
    op = operation();
    for _ in 0..70 {
        op.attributes = Value::Array(vec![op.attributes]);
    }
    let mut stack = [StreamFrame::empty(); 64];
    assert!(matches!(
        bounded_model_operation_write(&op, limits(), &mut stack, &mut output),
        Err(Error::Canonical(StreamError::DepthLimit))
    ));
    assert_eq!(output.0, 0);
    let mut short_stack = [StreamFrame::empty(); 1];
    assert!(matches!(
        bounded_model_operation_write(&op, limits(), &mut short_stack, &mut output),
        Err(Error::Canonical(StreamError::StackTooSmall))
    ));
    assert_eq!(output.0, 0);
}

#[test]
fn operation_exact_bytes_work_nodes_and_domain_admission() {
    let op = operation();
    let usage = operation_parity(&op);
    let mut exact = limits();
    exact.max_operations = 1;
    exact.max_inputs = op.inputs.len();
    exact.max_rank = op.output_shape.len();
    exact.max_domain_bytes = 3;
    exact.canonical.max_bytes = usage.bytes;
    exact.canonical.max_nodes = usage.nodes;
    exact.canonical.max_depth = usage.depth;
    exact.canonical.max_work = usage.work;
    let mut stack = [StreamFrame::empty(); 64];
    bounded_model_operation_digest(&op, "abc", exact, &mut stack).unwrap();
    assert!(matches!(
        bounded_model_operation_digest(&op, "abcd", exact, &mut stack),
        Err(Error::Domain)
    ));
    exact.canonical.max_bytes -= 1;
    let mut output = Untouched::default();
    assert!(matches!(
        bounded_model_operation_write(&op, exact, &mut stack, &mut output),
        Err(Error::Canonical(StreamError::ByteLimit))
    ));
    assert_eq!(output.0, 0);
}

#[test]
fn quote_uses_concrete_state_and_full_supplied_stack() {
    let quote = resources::<DomainHash>(64).unwrap();
    assert_eq!(quote.stream.stack_bytes, 64 * size_of::<StreamFrame<'_>>());
    assert_eq!(quote.stream.hash_state_bytes, size_of::<DomainHash>());
    assert_eq!(quote.stream.dependency_call_stack_bytes, None);
    assert_eq!(quote.schema_cursor_bytes, size_of::<Cursors>());
    assert_eq!(quote.admission_state_bytes, size_of::<Admission<'_>>());
    assert_eq!(quote.collection_state_bytes, size_of::<Collections>());
    assert!(resources::<DomainHash>(usize::MAX).is_err());
}

#[test]
fn every_control_escape_and_exact_raw_string_key_caps() {
    let mut op = operation();
    op.id = (0u8..32).map(char::from).collect::<String>() + "\"\\/雪é\u{2028}\u{2029}";
    operation_parity(&op);
    op.attributes = json!({"payload": "x".repeat(4096)});
    let mut exact = limits();
    let mut stack = [StreamFrame::empty(); 64];
    bounded_model_operation_write(&op, exact, &mut stack, &mut io::sink()).unwrap();
    exact.canonical.max_string_bytes -= 1;
    let mut output = Untouched::default();
    assert!(matches!(
        bounded_model_operation_write(&op, exact, &mut stack, &mut output),
        Err(Error::Canonical(StreamError::StringLimit))
    ));
    assert_eq!(output.0, 0);

    op.attributes = Value::Object(serde_json::Map::from_iter([(
        "k".repeat(4096),
        json!(null),
    )]));
    let mut stack = [StreamFrame::empty(); 64];
    exact = limits();
    bounded_model_operation_write(&op, exact, &mut stack, &mut io::sink()).unwrap();
    exact.canonical.max_key_bytes -= 1;
    assert!(matches!(
        bounded_model_operation_write(&op, exact, &mut stack, &mut output),
        Err(Error::Canonical(StreamError::KeyLimit))
    ));
    assert_eq!(output.0, 0);
}

#[test]
fn quadratic_attr_key_selection_is_in_shared_work_budget() {
    let mut op = operation();
    op.attributes = Value::Object(
        (0..80)
            .map(|i| (format!("common_long_prefix_{i:03}"), json!([i, i + 1])))
            .collect(),
    );
    let usage = operation_parity(&op);
    assert!(usage.work > usage.bytes * 10);
    let mut exact = limits();
    exact.canonical.max_work = usage.work;
    let mut stack = [StreamFrame::empty(); 64];
    bounded_model_operation_digest(&op, "work", exact, &mut stack).unwrap();
    exact.canonical.max_work -= 1;
    let mut output = Untouched::default();
    assert!(matches!(
        bounded_model_operation_write(&op, exact, &mut stack, &mut output),
        Err(Error::Canonical(StreamError::WorkLimit))
    ));
    assert_eq!(output.0, 0);
}

#[test]
fn external_writer_failure_is_not_reported_as_admission_success() {
    struct Fails;
    impl io::Write for Fails {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let op = operation();
    let mut stack = [StreamFrame::empty(); 64];
    assert!(matches!(
        bounded_model_operation_write(&op, limits(), &mut stack, &mut Fails),
        Err(Error::Canonical(StreamError::Io(_)))
    ));
}
