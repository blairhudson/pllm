//! Opt-in original-owner census. No physical/RSS or Budget authority assertions.

use pllm_models::owned_backing::{self as legacy, bounded};
use pllm_models::{
    AppliedMethod, DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator,
    StateKind, StateTensor,
};
use pllm_types::{
    bounded_canonical::borrowed::{self as json, NativeFrame},
    Digest,
};
use serde_json::{json, Map, Value};
use std::mem::size_of;

fn limits() -> bounded::Limits {
    bounded::Limits {
        max_backing_bytes: 1_000_000,
        max_work: 1_000_000,
        max_workspace_bytes: 1_000_000,
        max_json_nodes: 10_000,
        max_json_payload_bytes: 1_000_000,
        max_json_string_bytes: 10_000,
        max_json_key_bytes: 10_000,
        max_json_depth: 16,
    }
}

fn text(value: &str, capacity: usize) -> String {
    let mut result = String::with_capacity(capacity);
    result.push_str(value);
    result
}

fn graph(mode: DecoderMode, attributes: Vec<Value>) -> DecoderGraph {
    let mut operations = Vec::with_capacity(attributes.len() + 4);
    for attributes in attributes {
        let mut inputs = Vec::with_capacity(4);
        inputs.push(text("tokens", 63));
        inputs.push(text("positions", 67));
        let mut output_shape = Vec::with_capacity(5);
        output_shape.extend_from_slice(&[1, 2]);
        operations.push(ModelOperation {
            id: text("op", 41),
            operator: ModelOperator::TokenLookup,
            layer: None,
            state_kind: None,
            inputs,
            output_shape,
            attributes,
        });
    }
    let mut shape = Vec::with_capacity(4);
    shape.push(1);
    let mut state_inputs = Vec::with_capacity(3);
    state_inputs.push(StateTensor {
        id: text("state", 73),
        layer: None,
        kind: StateKind::Key,
        shape,
        maximum_sequence: 16,
    });
    DecoderGraph {
        mode,
        batch: 1,
        query_sequence: 1,
        maximum_key_sequence: 16,
        operations,
        state_inputs,
        state_outputs: Vec::with_capacity(2),
        output: text("op", 79),
    }
}

fn plan() -> DecoderPlan {
    let mut transformations = Vec::with_capacity(3);
    transformations.push(AppliedMethod {
        component: text("component", 83),
        implementation: text("implementation", 89),
        method_id: text("method", 97),
        input_digest: Digest::from_sha256([1; 32]),
        configuration_digest: Digest::from_sha256([2; 32]),
    });
    DecoderPlan {
        schema_version: text("pllm.decoder_plan.v1", 101),
        model_family: text("test", 103),
        adapter: text("test", 107),
        config_digest: Digest::from_sha256([3; 32]),
        transformations,
        prefill: graph(
            DecoderMode::Prefill,
            vec![json!({"x": [null, 1.0, "é"]}), Value::Null],
        ),
        decode: graph(
            DecoderMode::Decode,
            vec![json!({"β": false}), json!([true])],
        ),
        token_feedback: true,
    }
}

// Independent TEST-ONLY recursive oracle. Production uses the shared iterative
// walker. Formula is explicit opaque-map policy, not tree layout/capacity.
fn oracle_json(value: &Value) -> usize {
    match value {
        Value::String(text) => text.capacity(),
        Value::Array(values) => {
            values.capacity() * size_of::<Value>() + values.iter().map(oracle_json).sum::<usize>()
        }
        Value::Object(values) => {
            values.len().max(1) * ((size_of::<(String, Value)>() + size_of::<usize>()) * 16 + 128)
                + values
                    .iter()
                    .map(|(key, value)| key.capacity() + oracle_json(value))
                    .sum::<usize>()
        }
        _ => 0,
    }
}

fn oracle_graph(graph: &DecoderGraph) -> usize {
    graph.output.capacity()
        + graph.operations.capacity() * size_of::<ModelOperation>()
        + graph
            .operations
            .iter()
            .map(|op| {
                op.id.capacity()
                    + op.inputs.capacity() * size_of::<String>()
                    + op.inputs.iter().map(String::capacity).sum::<usize>()
                    + op.output_shape.capacity() * size_of::<u64>()
                    + oracle_json(&op.attributes)
            })
            .sum::<usize>()
        + [&graph.state_inputs, &graph.state_outputs]
            .into_iter()
            .map(|states| {
                states.capacity() * size_of::<StateTensor>()
                    + states
                        .iter()
                        .map(|state| {
                            state.id.capacity() + state.shape.capacity() * size_of::<u64>()
                        })
                        .sum::<usize>()
            })
            .sum::<usize>()
}

fn oracle_plan(plan: &DecoderPlan) -> usize {
    plan.schema_version.capacity()
        + plan.model_family.capacity()
        + plan.adapter.capacity()
        + plan.config_digest.backing_capacity_bytes()
        + plan.transformations.capacity() * size_of::<AppliedMethod>()
        + plan
            .transformations
            .iter()
            .map(|method| {
                method.component.capacity()
                    + method.implementation.capacity()
                    + method.method_id.capacity()
                    + method.input_digest.backing_capacity_bytes()
                    + method.configuration_digest.backing_capacity_bytes()
            })
            .sum::<usize>()
        + oracle_graph(&plan.prefill)
        + oracle_graph(&plan.decode)
}

fn census_json(
    value: &Value,
    limits: bounded::Limits,
) -> Result<(usize, bounded::Usage), bounded::Error> {
    bounded::resources_for_slots(17)?.check(limits)?;
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = bounded::Census::new(&mut frames, limits)?;
    let report = census.json(value)?;
    assert!(
        matches!(report.input(), bounded::Input::Value(original) if std::ptr::eq(original, value))
    );
    Ok((report.backing_bytes(), report.usage()))
}

#[test]
fn bounded_original_plan_and_graph_match_independent_and_legacy_capacity_censuses() {
    let plan = plan();
    let expected = oracle_plan(&plan);
    assert_eq!(plan.checked_owned_backing_bytes(usize::MAX), Ok(expected));
    let cap = bounded::Limits {
        max_backing_bytes: expected,
        ..limits()
    };
    bounded::resources_for_slots(17)
        .unwrap()
        .check(cap)
        .unwrap();
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = bounded::Census::new(&mut frames, cap).unwrap();
    let report = census.plan(&plan).unwrap();
    assert_eq!(report.backing_bytes(), expected); // no root or extra graph inline slots
    assert!(
        matches!(report.input(), bounded::Input::Plan(original) if std::ptr::eq(original, &plan))
    );
    assert_eq!(
        report.usage().work,
        report.usage().dto_steps + report.usage().json.work
    );
    let dto_steps = 2
        + plan.transformations.len()
        + [&plan.prefill, &plan.decode]
            .into_iter()
            .map(|graph| {
                4 + graph
                    .operations
                    .iter()
                    .map(|op| 3 + op.inputs.len())
                    .sum::<usize>()
                    + graph.state_inputs.len()
                    + graph.state_outputs.len()
            })
            .sum::<usize>();
    assert_eq!(report.usage().dto_steps, dto_steps);
    let graph_report = census.graph(&plan.decode).unwrap();
    assert_eq!(graph_report.backing_bytes(), oracle_graph(&plan.decode));
    assert_eq!(
        plan.decode.checked_owned_backing_bytes(usize::MAX),
        Ok(graph_report.backing_bytes())
    );
    assert!(
        matches!(graph_report.input(), bounded::Input::Graph(original) if std::ptr::eq(original, &plan.decode))
    );
    assert_eq!(graph_report.usage().completed_passes, 2);

    let short = bounded::Limits {
        max_backing_bytes: expected - 1,
        ..limits()
    };
    let mut other_frames = [const { NativeFrame::empty() }; 17];
    let mut short_census = bounded::Census::new(&mut other_frames, short).unwrap();
    assert_eq!(
        short_census.plan(&plan).unwrap_err(),
        bounded::Error::Backing(legacy::Error::ByteLimit)
    );
    assert_eq!(short_census.usage().completed_passes, 0);
}

#[test]
fn bounded_json_charges_actual_key_string_vec_spare_and_empty_retained_map() {
    let mut values = Vec::with_capacity(9);
    values.push(Value::String(text("é", 137)));
    let mut object = Map::new();
    let key = text("β", 149);
    assert!(key.capacity() > key.len());
    object.insert(key, Value::Array(values));
    let value = Value::Object(object);
    let expected = oracle_json(&value);
    assert_eq!(
        legacy::checked_json_backing_bytes(&value, expected),
        Ok(expected)
    );
    assert_eq!(
        census_json(
            &value,
            bounded::Limits {
                max_backing_bytes: expected,
                ..limits()
            }
        )
        .unwrap()
        .0,
        expected
    );
    assert_eq!(
        census_json(
            &value,
            bounded::Limits {
                max_backing_bytes: expected - 1,
                ..limits()
            }
        )
        .unwrap_err(),
        bounded::Error::Backing(legacy::Error::ByteLimit)
    );
    let mut retained = Map::new();
    retained.insert(text("removed", 151), Value::Null);
    retained.remove("removed");
    for empty in [
        Value::Object(retained),
        Value::Object(Map::new()),
        Value::Array(Vec::with_capacity(7)),
        Value::String(String::with_capacity(157)),
    ] {
        let expected = oracle_json(&empty);
        assert!(expected > 0);
        assert_eq!(census_json(&empty, limits()).unwrap().0, expected);
        assert_eq!(
            legacy::checked_json_backing_bytes(&empty, expected),
            Ok(expected)
        );
    }
}

fn nested(depth: usize) -> Value {
    let mut value = Value::Null;
    for _ in 0..depth {
        value = Value::Array(vec![value]);
    }
    value
}

#[test]
fn bounded_depth_translation_and_backing_refusal_before_descent() {
    let boundary = nested(16);
    let rejected = nested(17);
    assert!(legacy::checked_json_backing_bytes(&boundary, usize::MAX).is_ok());
    assert_eq!(census_json(&boundary, limits()).unwrap().1.json.depth, 17);
    assert_eq!(
        legacy::checked_json_backing_bytes(&rejected, usize::MAX),
        Err(legacy::Error::JsonDepth)
    );
    assert_eq!(
        census_json(&rejected, limits()).unwrap_err(),
        bounded::Error::Backing(legacy::Error::JsonDepth)
    );
    assert_eq!(
        census_json(
            &rejected,
            bounded::Limits {
                max_json_depth: usize::MAX,
                ..limits()
            }
        )
        .unwrap_err(),
        bounded::Error::Backing(legacy::Error::JsonDepth)
    );
    let root_capacity = match &rejected {
        Value::Array(v) => v.capacity() * size_of::<Value>(),
        _ => unreachable!(),
    };
    assert_eq!(
        census_json(
            &rejected,
            bounded::Limits {
                max_backing_bytes: root_capacity - 1,
                ..limits()
            }
        )
        .unwrap_err(),
        bounded::Error::Backing(legacy::Error::ByteLimit)
    );
    assert!(census_json(
        &Value::Null,
        bounded::Limits {
            max_json_depth: 0,
            ..limits()
        }
    )
    .is_ok());
    assert_eq!(
        census_json(
            &nested(1),
            bounded::Limits {
                max_json_depth: 0,
                ..limits()
            }
        )
        .unwrap_err(),
        bounded::Error::Backing(legacy::Error::JsonDepth)
    );
}

#[test]
fn bounded_native_linear_work_exact_short_and_poisoned_error_has_no_report() {
    let mut object = Map::new();
    for i in 0..64 {
        object.insert(format!("key{i:03}"), Value::Null);
    }
    let root = Value::Object(object);
    let (bytes, usage) = census_json(&root, limits()).unwrap();
    assert_eq!(usage.json.object_candidates, 64);
    assert_eq!(usage.json.work, 5 * 64 + 2);
    assert_eq!(usage.dto_steps, 2); // pass dispatch + JSON header
    assert_eq!(usage.work, 5 * 64 + 4);
    let exact = bounded::Limits {
        max_backing_bytes: bytes,
        max_work: usage.work,
        ..limits()
    };
    assert_eq!(census_json(&root, exact).unwrap().1, usage);
    assert_eq!(
        census_json(
            &root,
            bounded::Limits {
                max_work: usage.work - 1,
                ..exact
            }
        )
        .unwrap_err(),
        bounded::Error::WorkLimit
    );
    bounded::resources_for_slots(17)
        .unwrap()
        .check(exact)
        .unwrap();
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = bounded::Census::new(
        &mut frames,
        bounded::Limits {
            max_work: usage.work - 1,
            ..exact
        },
    )
    .unwrap();
    assert_eq!(census.json(&root).unwrap_err(), bounded::Error::WorkLimit);
    let partial = census.usage();
    assert_eq!(partial.completed_passes, 0);
    for _ in 0..3 {
        assert_eq!(census.json(&root).unwrap_err(), bounded::Error::Poisoned);
        assert_eq!(census.usage(), partial);
    }
}

#[test]
fn bounded_global_allowances_span_all_attributes_graphs_methods_and_replay_passes() {
    let plan = plan();
    let quote = bounded::resources_for_slots(17).unwrap();
    assert_eq!(quote.control_bytes, size_of::<bounded::Census<'_, '_>>());
    assert_eq!(
        quote.event_slots_bytes,
        2 * size_of::<Result<Option<json::Event<'_>>, json::Error>>()
            + 3 * size_of::<json::Event<'_>>()
    );
    assert_eq!(
        quote.result_slots_bytes,
        2 * size_of::<Result<json::Traversal<'_, '_>, json::Error>>()
            + 2 * size_of::<Result<(), legacy::Error>>()
            + 6 * size_of::<Result<(), bounded::Error>>()
            + 2 * size_of::<Result<usize, bounded::Error>>()
            + 2 * size_of::<Result<bounded::Report<'_>, bounded::Error>>()
            + 2 * size_of::<Result<usize, legacy::Error>>()
    );
    quote.check(limits()).unwrap();
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut baseline = bounded::Census::new(&mut frames, limits()).unwrap();
    let once = baseline.plan(&plan).unwrap();
    let usage = once.usage();
    let bytes = once.backing_bytes();
    let exact = bounded::Limits {
        max_backing_bytes: bytes,
        max_work: 2 * usage.work,
        max_json_nodes: 2 * usage.json.nodes,
        max_json_payload_bytes: 2 * usage.json.bytes,
        ..limits()
    };
    let mut replay_frames = [const { NativeFrame::empty() }; 17];
    let mut replay = bounded::Census::new(&mut replay_frames, exact).unwrap();
    assert_eq!(replay.plan(&plan).unwrap().backing_bytes(), bytes);
    let twice = replay.plan(&plan).unwrap();
    assert_eq!(twice.backing_bytes(), bytes); // owner count per pass, not double-charged
    assert_eq!(twice.usage().work, 2 * usage.work);
    assert_eq!(twice.usage().dto_steps, 2 * usage.dto_steps);
    assert_eq!(twice.usage().json.nodes, 2 * usage.json.nodes);
    assert_eq!(twice.usage().json.bytes, 2 * usage.json.bytes);
    assert_eq!(twice.usage().completed_passes, 2);
    assert_eq!(replay.plan(&plan).unwrap_err(), bounded::Error::WorkLimit);
    assert_eq!(replay.usage().completed_passes, 2);

    let mut short_frames = [const { NativeFrame::empty() }; 17];
    let mut short = bounded::Census::new(
        &mut short_frames,
        bounded::Limits {
            max_work: exact.max_work - 1,
            ..exact
        },
    )
    .unwrap();
    short.plan(&plan).unwrap();
    assert_eq!(short.plan(&plan).unwrap_err(), bounded::Error::WorkLimit);
    assert_eq!(short.usage().completed_passes, 1);

    let mut node_frames = [const { NativeFrame::empty() }; 17];
    let mut nodes = bounded::Census::new(
        &mut node_frames,
        bounded::Limits {
            max_json_nodes: usage.json.nodes - 1,
            ..limits()
        },
    )
    .unwrap();
    assert_eq!(
        nodes.plan(&plan).unwrap_err(),
        bounded::Error::Json(json::Error::NodeLimit)
    );
    assert_eq!(nodes.usage().completed_passes, 0);
    let mut payload_frames = [const { NativeFrame::empty() }; 17];
    let mut payload = bounded::Census::new(
        &mut payload_frames,
        bounded::Limits {
            max_json_payload_bytes: usage.json.bytes - 1,
            ..limits()
        },
    )
    .unwrap();
    assert_eq!(
        payload.plan(&plan).unwrap_err(),
        bounded::Error::Json(json::Error::ByteLimit)
    );
}

#[test]
fn bounded_quote_precedes_array_factory_and_rechecks_actual_pool_and_unknown_terms() {
    fn fixed<const N: usize>(
        value: &Value,
        limits: bounded::Limits,
        constructed: &mut bool,
    ) -> Result<usize, bounded::Error> {
        bounded::resources_for_slots(N)?.check(limits)?;
        *constructed = true;
        let mut frames = [const { NativeFrame::empty() }; N];
        Ok(bounded::Census::new(&mut frames, limits)?
            .json(value)?
            .backing_bytes())
    }
    let quote = bounded::resources_for_slots(17).unwrap();
    assert_eq!(quote.json.frame_bytes, 17 * size_of::<NativeFrame<'_>>());
    assert_eq!(
        quote.dto_iterators_bytes,
        2 * (size_of::<std::slice::Iter<'_, AppliedMethod>>()
            + size_of::<std::slice::Iter<'_, ModelOperation>>()
            + size_of::<std::slice::Iter<'_, String>>()
            + size_of::<std::array::IntoIter<&Vec<StateTensor>, 2>>()
            + size_of::<std::slice::Iter<'_, StateTensor>>())
    );
    assert_eq!(quote.json_limits_bytes, 2 * size_of::<json::Limits>());
    assert_eq!(
        quote.logical_workspace_bytes,
        quote.json.logical_workspace_bytes
            + quote.control_bytes
            + quote.backing_counter_bytes
            + quote.dto_iterators_bytes
            + quote.dto_entry_slots_bytes
            + quote.argument_bytes
            + quote.json_limits_bytes
            + quote.event_slots_bytes
            + quote.usage_transition_bytes
            + quote.result_slots_bytes
            + quote.quotation_bytes
            + quote.completion_bytes
    );
    assert_eq!(quote.dependency_buffer_bytes, None);
    assert_eq!(quote.compiled_callframe_bytes, None);
    assert_eq!(quote.rss_bytes, None);
    let root = Value::Null;
    let mut constructed = false;
    let exact = bounded::Limits {
        max_workspace_bytes: quote.logical_workspace_bytes,
        ..limits()
    };
    assert_eq!(
        fixed::<17>(
            &root,
            bounded::Limits {
                max_workspace_bytes: exact.max_workspace_bytes - 1,
                ..exact
            },
            &mut constructed
        ),
        Err(bounded::Error::WorkspaceLimit)
    );
    assert!(!constructed);
    assert_eq!(fixed::<17>(&root, exact, &mut constructed), Ok(0));
    assert!(constructed);
    // An earlier small-pool quote is NOT authority for a larger supplied pool.
    let small = bounded::resources_for_slots(1).unwrap();
    let mut actual = [const { NativeFrame::empty() }; 17];
    assert!(matches!(
        bounded::Census::new(
            &mut actual,
            bounded::Limits {
                max_workspace_bytes: small.logical_workspace_bytes,
                ..limits()
            }
        ),
        Err(bounded::Error::WorkspaceLimit)
    ));
    assert_eq!(
        bounded::resources_for_slots(usize::MAX),
        Err(bounded::Error::Overflow)
    );
    let mut empty = [];
    assert!(matches!(
        bounded::Census::new(&mut empty, limits()),
        Err(bounded::Error::Json(json::Error::WorkspaceTooSmall))
    ));
}

fn assert_poisoned_usage_stays_fixed(census: &mut bounded::Census<'_, '_>) {
    let partial = census.usage();
    for _ in 0..3 {
        assert_eq!(
            census.json(&Value::Null).unwrap_err(),
            bounded::Error::Poisoned
        );
        assert_eq!(census.usage(), partial);
    }
}

#[test]
fn bounded_replay_node_and_payload_exhaustion_are_independent_of_roomy_work() {
    let root = json!({"β": [true, "é", 12], "x": null});
    // Independent literal census: object, array, true, string, number, null.
    // Payload counts both keys, UTF-8 string and the two Number Display bytes.
    let nodes = 6;
    let payload = "β".len() + "x".len() + "é".len() + 2;
    for (cap, expected) in [
        (
            bounded::Limits {
                max_json_nodes: 2 * nodes - 1,
                ..limits()
            },
            bounded::Error::Json(json::Error::NodeLimit),
        ),
        (
            bounded::Limits {
                max_json_payload_bytes: 2 * payload - 1,
                ..limits()
            },
            bounded::Error::Json(json::Error::ByteLimit),
        ),
    ] {
        bounded::resources_for_slots(17)
            .unwrap()
            .check(cap)
            .unwrap();
        let mut frames = [const { NativeFrame::empty() }; 17];
        let mut census = bounded::Census::new(&mut frames, cap).unwrap();
        let first = census.json(&root).unwrap();
        let first_usage = first.usage();
        let first_bytes = first.backing_bytes();
        assert_eq!(first_usage.json.nodes, nodes);
        assert_eq!(first_usage.json.bytes, payload);
        assert_eq!(census.json(&root).unwrap_err(), expected);
        assert!(census.usage().work < cap.max_work); // NOT work exhaustion
        assert_eq!(census.usage().completed_passes, 1);
        assert_eq!(first.usage(), first_usage); // prior success is an immutable snapshot
        assert_eq!(first.backing_bytes(), first_bytes);
        assert_poisoned_usage_stays_fixed(&mut census);
    }
    let exact = bounded::Limits {
        max_json_nodes: 2 * nodes,
        max_json_payload_bytes: 2 * payload,
        ..limits()
    };
    bounded::resources_for_slots(17)
        .unwrap()
        .check(exact)
        .unwrap();
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = bounded::Census::new(&mut frames, exact).unwrap();
    census.json(&root).unwrap();
    let replay = census.json(&root).unwrap();
    assert_eq!(replay.usage().json.nodes, 2 * nodes);
    assert_eq!(replay.usage().json.bytes, 2 * payload);
    assert_eq!(replay.usage().completed_passes, 2);
}

#[test]
fn bounded_json_string_and_key_limits_use_utf8_lengths_not_spare_capacity() {
    let value = Value::String(text("é🔥", 101));
    let len = "é🔥".len();
    assert_eq!(len, 6);
    let exact = bounded::Limits {
        max_json_string_bytes: len,
        ..limits()
    };
    assert_eq!(census_json(&value, exact).unwrap().0, oracle_json(&value));
    assert_eq!(
        census_json(
            &value,
            bounded::Limits {
                max_json_string_bytes: len - 1,
                ..exact
            }
        )
        .unwrap_err(),
        bounded::Error::Json(json::Error::StringLimit)
    );
    let mut object = Map::new();
    let key = text("β🔥", 137);
    let len = key.len();
    object.insert(key, Value::Null);
    let value = Value::Object(object);
    let exact = bounded::Limits {
        max_json_key_bytes: len,
        max_json_string_bytes: len,
        ..limits()
    };
    assert_eq!(census_json(&value, exact).unwrap().0, oracle_json(&value));
    assert_eq!(
        census_json(
            &value,
            bounded::Limits {
                max_json_key_bytes: len - 1,
                ..exact
            }
        )
        .unwrap_err(),
        bounded::Error::Json(json::Error::KeyLimit)
    );
    // Shared raw string cap also applies to keys; independent key cap passes.
    assert_eq!(
        census_json(
            &value,
            bounded::Limits {
                max_json_string_bytes: len - 1,
                ..exact
            }
        )
        .unwrap_err(),
        bounded::Error::Json(json::Error::StringLimit)
    );
    // Shared traversal promises key-limit precedence when both lengths fail.
    assert_eq!(
        census_json(
            &value,
            bounded::Limits {
                max_json_key_bytes: len - 1,
                max_json_string_bytes: len - 1,
                ..exact
            }
        )
        .unwrap_err(),
        bounded::Error::Json(json::Error::KeyLimit)
    );
}

fn mixed_nested(depth: usize, object_first: bool) -> Value {
    let mut value = Value::String(text("é", 23));
    for level in 0..depth {
        value = if (level % 2 == 0) == object_first {
            let mut object = Map::new();
            object.insert(text("δ", 31), value);
            Value::Object(object)
        } else {
            Value::Array(vec![value])
        };
    }
    value
}

#[test]
fn bounded_mixed_object_array_depth_zero_through_sixteen_and_seventeen() {
    for object_first in [false, true] {
        for depth in 0..=17 {
            let value = mixed_nested(depth, object_first);
            let old = legacy::checked_json_backing_bytes(&value, usize::MAX);
            if depth == 17 {
                assert_eq!(old, Err(legacy::Error::JsonDepth));
                assert_eq!(
                    census_json(&value, limits()).unwrap_err(),
                    bounded::Error::Backing(legacy::Error::JsonDepth)
                );
                continue;
            }
            let exact = bounded::Limits {
                max_json_depth: depth,
                ..limits()
            };
            let (bytes, usage) = census_json(&value, exact).unwrap();
            assert_eq!(bytes, oracle_json(&value));
            assert_eq!(old, Ok(bytes));
            assert_eq!(usage.json.nodes, depth + 1);
            assert_eq!(usage.json.depth, depth + 1); // limits are zero-based
            if depth != 0 {
                assert_eq!(
                    census_json(
                        &value,
                        bounded::Limits {
                            max_json_depth: depth - 1,
                            ..exact
                        }
                    )
                    .unwrap_err(),
                    bounded::Error::Backing(legacy::Error::JsonDepth)
                );
            }
        }
    }
}

#[test]
fn bounded_backing_depth_node_payload_errors_poison_without_partial_success_report() {
    for (bad, cap, expected) in [
        (
            Value::String(text("backing", 128)),
            bounded::Limits {
                max_backing_bytes: 0,
                ..limits()
            },
            bounded::Error::Backing(legacy::Error::ByteLimit),
        ),
        (
            mixed_nested(17, true),
            limits(),
            bounded::Error::Backing(legacy::Error::JsonDepth),
        ),
        (
            json!([null]),
            bounded::Limits {
                max_json_nodes: 2,
                ..limits()
            },
            bounded::Error::Json(json::Error::NodeLimit),
        ),
        (
            json!("é"),
            bounded::Limits {
                max_json_payload_bytes: 1,
                ..limits()
            },
            bounded::Error::Json(json::Error::ByteLimit),
        ),
    ] {
        bounded::resources_for_slots(17)
            .unwrap()
            .check(cap)
            .unwrap();
        let mut frames = [const { NativeFrame::empty() }; 17];
        let mut census = bounded::Census::new(&mut frames, cap).unwrap();
        let previous = census.json(&Value::Null).unwrap();
        let snapshot = previous.usage();
        assert_eq!(census.json(&bad).unwrap_err(), expected);
        let partial = census.usage();
        assert_eq!(partial.completed_passes, 1);
        assert!(partial.work > snapshot.work);
        assert!(partial.work < cap.max_work);
        assert!(partial.json.nodes > snapshot.json.nodes);
        assert_eq!(previous.usage(), snapshot);
        assert_eq!(previous.backing_bytes(), 0);
        assert!(matches!(previous.input(), bounded::Input::Value(v) if v.is_null()));
        assert!(matches!(census.input(), Some(bounded::Input::Value(v)) if std::ptr::eq(v, &bad)));
        assert_poisoned_usage_stays_fixed(&mut census);
        assert!(matches!(census.input(), Some(bounded::Input::Value(v)) if std::ptr::eq(v, &bad)));
    }
}

#[test]
fn bounded_competing_limits_preserve_only_promised_container_and_cursor_precedence() {
    let object = json!({"x": [null]});
    let cap = bounded::Limits {
        max_backing_bytes: 0,
        max_json_nodes: 1,
        max_json_depth: 0,
        ..limits()
    };
    bounded::resources_for_slots(17)
        .unwrap()
        .check(cap)
        .unwrap();
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = bounded::Census::new(&mut frames, cap).unwrap();
    assert_eq!(
        census.json(&object).unwrap_err(),
        bounded::Error::Backing(legacy::Error::ByteLimit)
    );
    assert_eq!(census.usage().json.nodes, 1);
    assert_eq!(census.usage().json.object_candidates, 0);
    // No key/child EVENT reached. Object iterator initialization precedes the
    // container event in Native mode: this is NOT pre-iterator admission.
    assert_eq!(census.usage().json.bytes, 0);
    assert_eq!(
        census_json(
            &object,
            bounded::Limits {
                max_backing_bytes: 0,
                max_json_nodes: 0,
                ..limits()
            }
        )
        .unwrap_err(),
        bounded::Error::Json(json::Error::NodeLimit)
    );
    assert_eq!(
        census_json(
            &object,
            bounded::Limits {
                max_backing_bytes: 0,
                max_work: 0,
                ..limits()
            }
        )
        .unwrap_err(),
        bounded::Error::WorkLimit
    );

    let array = json!([null]);
    let cap = bounded::Limits {
        max_json_depth: 0,
        max_json_nodes: 1,
        ..limits()
    };
    bounded::resources_for_slots(1).unwrap().check(cap).unwrap();
    let mut one = [const { NativeFrame::empty() }; 1];
    let mut census = bounded::Census::new(&mut one, cap).unwrap();
    // Push depth fails before unavailable frame slot and child node allowance.
    assert_eq!(
        census.json(&array).unwrap_err(),
        bounded::Error::Backing(legacy::Error::JsonDepth)
    );

    let mut graph = graph(DecoderMode::Decode, vec![mixed_nested(17, false)]);
    graph.output = String::new();
    let cap = bounded::Limits {
        max_backing_bytes: 0,
        max_json_nodes: 0,
        ..limits()
    };
    bounded::resources_for_slots(17)
        .unwrap()
        .check(cap)
        .unwrap();
    let mut dto_frames = [const { NativeFrame::empty() }; 17];
    let mut census = bounded::Census::new(&mut dto_frames, cap).unwrap();
    assert_eq!(
        census.graph(&graph).unwrap_err(),
        bounded::Error::Backing(legacy::Error::ByteLimit)
    );
    assert_eq!(census.usage().dto_steps, 2); // pass + graph header; no operation body
    assert_eq!(census.usage().json.nodes, 0);
}
