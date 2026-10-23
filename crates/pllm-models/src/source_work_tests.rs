use super::*;
use serde_json::json;

fn document(family: &str, layers: u64) -> Value {
    json!({"model_type":family,"num_hidden_layers":layers,"hidden_size":4,
        "intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":7,"max_position_embeddings":16,
        "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,
        "tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,
        "use_sliding_window":false,"use_cache":true,"max_window_layers":layers})
}
fn workload() -> DecoderWorkload {
    DecoderWorkload {
        batch: 1,
        max_input_tokens: 4,
        max_new_tokens: 2,
    }
}
fn expansion() -> DecoderExpansionBudget {
    DecoderExpansionBudget {
        max_layers: 8,
        max_feature_width: 128,
        max_attention_heads: 8,
        max_vocabulary: 128,
        max_tensor_elements: 32768,
    }
}
fn lowerer() -> ConfigLowererLimits {
    ConfigLowererLimits {
        max_operations_per_phase: 512,
        max_string_bytes: 1024,
        max_attribute_nodes: 1 << 20,
        max_retained_strings: 1 << 20,
        max_factory_bytes: 64 << 20,
    }
}
fn limits() -> SourceInputLimits {
    SourceInputLimits {
        max_nodes: 32768,
        max_payload_bytes: 1 << 20,
        max_string_bytes: 1024,
        max_key_bytes: 1024,
        max_depth: 17,
        max_traversal_work: 1 << 24,
        max_total_work: u64::MAX,
        max_workspace_bytes: 1 << 20,
    }
}

fn quoted(
    source: &Value,
    factory: ConfigLowererLimits,
    input: SourceInputLimits,
) -> (ConfigLowererQuote<'_>, SourceWorkQuote) {
    bootstrap(input, factory).unwrap();
    let mut frames = [const { NativeFrame::empty() }; 17];
    quote(source, workload(), expansion(), factory, input, &mut frames).unwrap()
} // Actual pool retires before caller constructs the model; no drop_non_drop.

// Independent recursive test oracle; production uses only the shared native
// cursor. Number Display is the explicitly qualified scalar-Number contract.
fn json_oracle(value: &Value) -> (usize, usize, usize) {
    let (mut nodes, mut payload, mut depth) = (1, 0, 1);
    let mut child = |value: &Value| {
        let (n, b, d) = json_oracle(value);
        nodes += n;
        payload += b;
        depth = depth.max(d + 1);
    };
    match value {
        Value::Array(values) => {
            for value in values {
                child(value);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                child(value);
            }
            payload += values.keys().map(String::len).sum::<usize>();
        }
        Value::String(value) => payload += value.len(),
        Value::Number(value) => payload += value.to_string().len(),
        _ => {}
    }
    (nodes, payload, depth)
}

#[test]
fn registered_factories_match_original_byte_lowerers_and_independent_census() {
    for family in ["llama", "qwen2", "qwen3"] {
        for layers in [1, 2, 3] {
            let source = document(family, layers);
            let expected = crate::lower_model_json_with_budget(
                &serde_json::to_vec(&source).unwrap(),
                workload(),
                expansion(),
            )
            .unwrap();
            let prepaid = bootstrap(limits(), lowerer()).unwrap();
            assert_eq!(prepaid.frame_slots, 17);
            let first = {
                let mut frames = [const { NativeFrame::empty() }; 17];
                let (_captured, work) = quote(
                    &source,
                    workload(),
                    expansion(),
                    lowerer(),
                    limits(),
                    &mut frames,
                )
                .unwrap();
                assert_eq!(work.input_usage().nodes, json_oracle(&source).0);
                assert_eq!(work.input_usage().bytes, json_oracle(&source).1);
                assert_eq!(work.input_usage().depth, json_oracle(&source).2);
                work
            }; // first pool retired BEFORE the second phase
            let (captured, work) = {
                let mut frames = [const { NativeFrame::empty() }; 17];
                quote(
                    &source,
                    workload(),
                    expansion(),
                    lowerer(),
                    limits(),
                    &mut frames,
                )
                .unwrap()
            };
            assert_eq!(first.total_work(), work.total_work());
            assert_eq!(work.slices().quote, prepaid.quote_work);
            assert_eq!(work.slices().requote, prepaid.requote_work);
            assert_eq!(work.slices().total_work().unwrap(), work.total_work());
            let retained = captured.metadata().retained_metadata_bytes;
            let plan = captured.construct().unwrap();
            assert_eq!(plan, expected);
            assert_eq!(plan.digest(), expected.digest());
            let mut frames = [const { NativeFrame::empty() }; 17];
            let mut owner_census =
                census::Census::new(&mut frames, work.post_census_limits()).unwrap();
            let report = owner_census.plan(&plan).unwrap();
            assert!(report.backing_bytes() <= retained);
            assert_eq!(
                report.backing_bytes(),
                plan.checked_owned_backing_bytes(lowerer().max_factory_bytes)
                    .unwrap()
            );
            assert!(report.usage().work as u64 <= work.census_work());
            // Independent DTO/node oracle, not the quote's own phase fields.
            let nodes: usize = [&plan.prefill, &plan.decode]
                .into_iter()
                .flat_map(|graph| &graph.operations)
                .map(|op| json_oracle(&op.attributes).0)
                .sum();
            assert_eq!(report.usage().json.nodes, nodes);
            assert!(nodes <= work.post_census_limits().max_json_nodes);
        }
    }
}

#[test]
fn invalid_epsilon_proves_query_does_not_construct_to_discover_work() {
    let mut source = document("llama", 1);
    source["rms_norm_eps"] = json!("not a decimal");
    let attempts = crate::source_budget::factory_attempts();
    let (captured, work) = quoted(&source, lowerer(), limits());
    assert_eq!(crate::source_budget::factory_attempts(), attempts);
    assert!(work.opaque_helpers().decimal_parse_calls > 0);
    assert!(work.opaque_helpers().decimal_parse_input_bytes > 0);
    let expected = crate::lower_model_json_with_budget(
        &serde_json::to_vec(&source).unwrap(),
        workload(),
        expansion(),
    )
    .unwrap_err();
    assert_eq!(captured.construct().unwrap_err(), expected);
    assert_eq!(crate::source_budget::factory_attempts(), attempts + 1);
}

#[test]
fn exact_and_one_short_total_include_both_queries_without_lending_slack() {
    let source = document("qwen3", 2);
    let work = {
        bootstrap(limits(), lowerer()).unwrap();
        let mut frames = [const { NativeFrame::empty() }; 17];
        quote(
            &source,
            workload(),
            expansion(),
            lowerer(),
            limits(),
            &mut frames,
        )
        .unwrap()
        .1
    };
    let mut exact = limits();
    exact.max_total_work = work.total_work();
    let (captured, accepted) = quoted(&source, lowerer(), exact);
    assert_eq!(accepted.total_work(), exact.max_total_work);
    let mut short = exact;
    short.max_total_work -= 1;
    let attempts = crate::source_budget::factory_attempts();
    {
        bootstrap(short, lowerer()).unwrap();
        let mut frames = [const { NativeFrame::empty() }; 17];
        assert!(matches!(
            quote(
                &source,
                workload(),
                expansion(),
                lowerer(),
                short,
                &mut frames
            ),
            Err(SourceWorkError::WorkLimit)
        ));
    }
    assert_eq!(crate::source_budget::factory_attempts(), attempts);
    assert!(captured.construct().is_ok());
    assert_eq!(accepted.slices().quote, accepted.slices().requote);
    assert!(accepted.slices().factory > 0 && accepted.slices().cleanup > 0);
}

#[test]
fn bootstrap_precedes_array_and_checks_actual_pool_size() {
    let source = document("llama", 1);
    let quoted = bootstrap(limits(), lowerer()).unwrap();
    assert!(quoted.bootstrap_bytes >= 17 * size_of::<NativeFrame<'_>>());
    let mut short = limits();
    short.max_workspace_bytes = quoted.bootstrap_bytes - 1;
    assert!(matches!(
        bootstrap(short, lowerer()),
        Err(SourceWorkError::Workspace)
    ));
    // Only now construct the quoted fixed pool. No unpriced Vec/capacity proxy.
    let mut frames = [const { NativeFrame::empty() }; 17];
    assert!(quote(
        &source,
        workload(),
        expansion(),
        lowerer(),
        limits(),
        &mut frames
    )
    .is_ok());
    assert!(matches!(
        quote(
            &source,
            workload(),
            expansion(),
            lowerer(),
            limits(),
            &mut frames[..16]
        ),
        Err(SourceWorkError::Workspace)
    ));
    let mut larger = [const { NativeFrame::empty() }; 18];
    assert!(matches!(
        quote(
            &source,
            workload(),
            expansion(),
            lowerer(),
            limits(),
            &mut larger
        ),
        Err(SourceWorkError::Workspace)
    ));
}

#[test]
fn input_caps_fail_closed_then_workspace_reuses_for_different_root() {
    let first = document("llama", 1);
    let second = document("qwen2", 2);
    bootstrap(limits(), lowerer()).unwrap();
    let (captured, work) = {
        let mut frames = [const { NativeFrame::empty() }; 17];
        for case in 0..5 {
            let mut cap = limits();
            match case {
                0 => cap.max_nodes = 1,
                1 => cap.max_payload_bytes = 0,
                2 => cap.max_string_bytes = 0,
                3 => cap.max_key_bytes = 0,
                _ => cap.max_traversal_work = 0,
            }
            assert!(matches!(
                quote(&first, workload(), expansion(), lowerer(), cap, &mut frames),
                Err(SourceWorkError::Json(_))
            ));
        }
        quote(
            &second,
            workload(),
            expansion(),
            lowerer(),
            limits(),
            &mut frames,
        )
        .unwrap()
    };
    assert_eq!(work.input_usage().nodes, json_oracle(&second).0);
    assert!(captured.construct().is_ok());
    // Earlier source owners deliberately remain alive through pool reuse.
}

#[test]
fn legacy_depth_zero_sixteen_and_native_one_seventeen_agree() {
    for nesting in [14, 15, 16] {
        let mut source = document("llama", 1);
        let mut extra = Value::Null;
        for _ in 0..nesting {
            extra = Value::Array(vec![extra]);
        }
        source["extra"] = extra;
        let depth = json_oracle(&source).2;
        bootstrap(limits(), lowerer()).unwrap();
        let mut frames = [const { NativeFrame::empty() }; 17];
        let result = quote(
            &source,
            workload(),
            expansion(),
            lowerer(),
            limits(),
            &mut frames,
        );
        if depth <= 17 {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result,
                Err(SourceWorkError::Json(json::Error::DepthLimit))
            ));
        }
    }
}

#[test]
fn overflow_and_bootstrap_work_refuse_before_workspace_or_factory() {
    let mut cap = limits();
    cap.max_nodes = usize::MAX;
    assert!(matches!(
        bootstrap(cap, lowerer()),
        Err(SourceWorkError::Overflow)
    ));
    let mut cap = limits();
    cap.max_total_work = 0;
    assert!(matches!(
        bootstrap(cap, lowerer()),
        Err(SourceWorkError::WorkLimit)
    ));
    let mut cap = limits();
    cap.max_depth = 18;
    assert!(matches!(
        bootstrap(cap, lowerer()),
        Err(SourceWorkError::InputLimit)
    ));
    let slices = SourceWorkSlices {
        quote: u64::MAX,
        requote: 1,
        factory: 0,
        cleanup: 0,
        census: 0,
    };
    assert!(matches!(
        slices.total_work(),
        Err(SourceWorkError::Overflow)
    ));
}

#[test]
fn actual_spare_backing_and_utf8_are_not_zero_or_caller_baselines() {
    let mut compact = document("llama", 1);
    compact["unknown"] = Value::String(String::from("λ🦀"));
    let compact_bytes =
        crate::owned_backing::checked_json_backing_bytes(&compact, lowerer().max_factory_bytes)
            .unwrap();
    let compact_factory = {
        bootstrap(limits(), lowerer()).unwrap();
        let mut frames = [const { NativeFrame::empty() }; 17];
        quote(
            &compact,
            workload(),
            expansion(),
            lowerer(),
            limits(),
            &mut frames,
        )
        .unwrap()
        .0
        .factory_bytes()
    };
    let mut source = compact.clone();
    let mut extra = String::from("λ🦀");
    extra.reserve(4096);
    source["unknown"] = Value::String(extra);
    let (captured, work) = quoted(&source, lowerer(), limits());
    let roomy_bytes =
        crate::owned_backing::checked_json_backing_bytes(&source, lowerer().max_factory_bytes)
            .unwrap();
    assert!(roomy_bytes > compact_bytes);
    assert_eq!(
        captured.factory_bytes() - compact_factory,
        4 * (roomy_bytes - compact_bytes)
    );
    assert_eq!(work.input_usage().bytes, json_oracle(&source).1);
    assert!(work.opaque_helpers().hash_input_bytes > 0);
    assert!(work.post_census_limits().max_backing_bytes >= captured.factory_bytes());
    assert_eq!(
        work.post_census_resources().logical_workspace_bytes,
        work.post_census_limits().max_workspace_bytes
    );
    assert!(
        work.resources().bootstrap_bytes >= work.post_census_resources().logical_workspace_bytes
    );
    assert!(captured.construct().is_ok());
}

#[test]
fn cursor_event_work_does_not_dominate_string_payload_bytes() {
    let mut source = document("llama", 1);
    source["unknown"] = Value::String("x".repeat(4096));
    let mut input = limits();
    input.max_string_bytes = 8192;
    let mut factory = lowerer();
    factory.max_string_bytes = 8192;
    bootstrap(input, factory).unwrap();
    let (captured, work) = {
        let mut frames = [const { NativeFrame::empty() }; 17];
        let (captured, work) = quote(
            &source,
            workload(),
            expansion(),
            factory,
            input,
            &mut frames,
        )
        .unwrap();
        let (nodes, bytes, _) = json_oracle(&source);
        let members = source.as_object().unwrap().len();
        // Literal B native-census bound for this flat object: key payload is NOT
        // counted as cursor work. No new production formatter recipe or B API.
        let b_cursor_bound = 3 + nodes * 34 + members * 3;
        assert!(bytes > b_cursor_bound);
        input.max_payload_bytes = b_cursor_bound;
        let attempts = crate::source_budget::factory_attempts();
        assert!(matches!(
            quote(
                &source,
                workload(),
                expansion(),
                factory,
                input,
                &mut frames
            ),
            Err(SourceWorkError::Json(json::Error::ByteLimit))
        ));
        assert_eq!(crate::source_budget::factory_attempts(), attempts);
        assert!(work.input_usage().bytes > work.input_usage().work);
        (captured, work)
    };
    assert!(work.input_usage().bytes > work.input_usage().work);
    assert!(captured.construct().is_ok());
}
