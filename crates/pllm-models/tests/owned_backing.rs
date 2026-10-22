//! Ordinary tests for the logical owned-backing contract, not native/RSS limits.

use pllm_models::owned_backing::{
    checked_array_bytes, checked_json_backing_bytes, checked_tree_node_bytes, Error,
};
use pllm_models::{
    AppliedMethod, DecoderGraph, DecoderMode, DecoderPlan, ModelOperation, ModelOperator,
    StateKind, StateTensor,
};
use pllm_types::Digest;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::size_of;

struct CountingAllocator;

thread_local! {
    static RECORD: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn record_allocation() {
    let _ = RECORD.try_with(|record| {
        if record.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn without_allocations<T>(f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            RECORD.with(|record| record.set(false));
        }
    }
    ALLOCATIONS.with(|count| count.set(0));
    RECORD.with(|record| assert!(!record.replace(true)));
    let reset = Reset;
    let result = f();
    drop(reset);
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    result
}

fn text(value: &str, capacity: usize) -> String {
    let mut result = String::with_capacity(capacity);
    result.push_str(value);
    result
}

fn digest(capacity: usize) -> Digest {
    let text = text(&"a".repeat(64), capacity);
    let expected_capacity = text.capacity();
    let result = Digest::deserialize(serde::de::value::StringDeserializer::<
        serde::de::value::Error,
    >::new(text))
    .unwrap();
    assert_eq!(result.backing_capacity_bytes(), expected_capacity);
    result
}

fn graph(mode: DecoderMode) -> (DecoderGraph, usize) {
    let mut values = Vec::with_capacity(7);
    values.push(Value::String(text("attribute", 107)));
    values.push(Value::Bool(true));
    values.push(Value::Number(5.into()));
    let array_backing = values.capacity() * size_of::<Value>()
        + match &values[0] {
            Value::String(text) => text.capacity(),
            _ => unreachable!(),
        };
    let mut attributes = Map::new();
    let key = text("nested", 109);
    let key_bytes = key.capacity();
    attributes.insert(key, Value::Array(values));
    // Independent explicit full-node allowance, not production census helpers.
    let node = (size_of::<(String, Value)>() + size_of::<usize>()) * 16 + 128;
    let attributes_backing = node + key_bytes + array_backing;

    let mut inputs = Vec::with_capacity(5);
    inputs.push(text("input.tokens", 113));
    inputs.push(text("input.positions", 127));
    let input_backing =
        inputs.capacity() * size_of::<String>() + inputs[0].capacity() + inputs[1].capacity();
    let mut output_shape = Vec::with_capacity(11);
    output_shape.extend_from_slice(&[1, 2, 3]);
    let shape_backing = output_shape.capacity() * size_of::<u64>();
    let mut operations = Vec::with_capacity(3);
    operations.push(ModelOperation {
        id: text("op", 131),
        operator: ModelOperator::TokenLookup,
        layer: Some(0),
        state_kind: Some(StateKind::Key),
        inputs,
        output_shape,
        attributes: Value::Object(attributes),
    });
    let operation_backing = operations.capacity() * size_of::<ModelOperation>()
        + operations[0].id.capacity()
        + input_backing
        + shape_backing
        + attributes_backing;

    let mut state_inputs = Vec::with_capacity(4);
    let mut input_shape = Vec::with_capacity(13);
    input_shape.extend_from_slice(&[1, 2]);
    let input_shape_backing = input_shape.capacity() * size_of::<u64>();
    state_inputs.push(StateTensor {
        id: text("state.in", 137),
        layer: Some(0),
        kind: StateKind::Key,
        shape: input_shape,
        maximum_sequence: 2,
    });
    let mut state_outputs = Vec::with_capacity(6);
    let mut output_shape = Vec::with_capacity(17);
    output_shape.extend_from_slice(&[3, 4]);
    let output_shape_backing = output_shape.capacity() * size_of::<u64>();
    state_outputs.push(StateTensor {
        id: text("state.out", 139),
        layer: None,
        kind: StateKind::Value,
        shape: output_shape,
        maximum_sequence: 4,
    });
    let output = text("op", 149);
    let expected = output.capacity()
        + operation_backing
        + state_inputs.capacity() * size_of::<StateTensor>()
        + state_inputs[0].id.capacity()
        + input_shape_backing
        + state_outputs.capacity() * size_of::<StateTensor>()
        + state_outputs[0].id.capacity()
        + output_shape_backing;
    (
        DecoderGraph {
            mode,
            batch: 1,
            query_sequence: 1,
            maximum_key_sequence: 4,
            operations,
            state_inputs,
            state_outputs,
            output,
        },
        expected,
    )
}

fn plan() -> (DecoderPlan, usize) {
    let (prefill, prefill_bytes) = graph(DecoderMode::Prefill);
    let (mut decode, mut decode_bytes) = graph(DecoderMode::Decode);
    // Distinct decode-only owner catches accidental prefill-only census.
    let old_output_capacity = decode.output.capacity();
    decode.output.reserve_exact(1000);
    decode_bytes += decode.output.capacity() - old_output_capacity;
    let config_digest = digest(257);
    let input_digest = digest(263);
    let configuration_digest = digest(269);
    let digest_bytes = config_digest.backing_capacity_bytes()
        + input_digest.backing_capacity_bytes()
        + configuration_digest.backing_capacity_bytes();
    let mut transformations = Vec::with_capacity(4);
    transformations.push(AppliedMethod {
        component: text("component", 151),
        implementation: text("implementation", 157),
        method_id: text("method", 163),
        input_digest,
        configuration_digest,
    });
    let schema_version = text("pllm.decoder_plan.v1", 167);
    let model_family = text("qwen2", 173);
    let adapter = text("adapter", 179);
    let expected = schema_version.capacity()
        + model_family.capacity()
        + adapter.capacity()
        + digest_bytes
        + transformations.capacity() * size_of::<AppliedMethod>()
        + transformations[0].component.capacity()
        + transformations[0].implementation.capacity()
        + transformations[0].method_id.capacity()
        + prefill_bytes
        + decode_bytes;
    (
        DecoderPlan {
            schema_version,
            model_family,
            adapter,
            config_digest,
            transformations,
            prefill,
            decode,
            token_feedback: true,
        },
        expected,
    )
}

#[test]
fn all_plan_owners_spare_capacity_exact_cap_and_one_short() {
    let (plan, expected) = plan();
    let original_bytes = pllm_types::canonical_bytes(&plan);
    let original_digest = plan.digest();
    let alias = &plan;
    assert_eq!(
        without_allocations(|| alias.checked_owned_backing_bytes(expected)),
        Ok(expected)
    );
    assert_eq!(
        without_allocations(|| plan.checked_owned_backing_bytes(expected - 1)),
        Err(Error::ByteLimit)
    );
    assert_eq!(pllm_types::canonical_bytes(&plan), original_bytes);
    assert_eq!(plan.digest(), original_digest);
}

#[test]
fn graph_backing_excludes_inline_owner_and_borrowed_views() {
    let (graph, expected) = graph(DecoderMode::Decode);
    let borrowed_operation = &graph.operations[0];
    assert_eq!(borrowed_operation.id, "op");
    assert_eq!(
        without_allocations(|| graph.checked_owned_backing_bytes(expected)),
        Ok(expected)
    );
    assert_eq!(
        graph.checked_owned_backing_bytes(expected - 1),
        Err(Error::ByteLimit)
    );
}

#[test]
fn empty_json_objects_charge_minimum_node_even_after_removal() {
    let node = (size_of::<(String, Value)>() + size_of::<usize>()) * 16 + 128;
    assert_eq!(checked_tree_node_bytes::<(String, Value)>(), Ok(node));
    let fresh = Value::Object(Map::new());
    let mut retained = Map::new();
    retained.insert("removed".into(), Value::Null);
    retained.remove("removed");
    for value in [fresh, Value::Object(retained)] {
        assert_eq!(
            without_allocations(|| checked_json_backing_bytes(&value, node)),
            Ok(node)
        );
        assert_eq!(
            without_allocations(|| checked_json_backing_bytes(&value, node - 1)),
            Err(Error::ByteLimit)
        );
    }
}

#[test]
fn every_object_entry_and_nested_empty_object_use_shared_node_policy() {
    let node = (size_of::<(String, Value)>() + size_of::<usize>()) * 16 + 128;
    let key_a = text("a", 101);
    let key_b = text("b", 103);
    let value_b = text("value", 107);
    let expected = 3 * node + key_a.capacity() + key_b.capacity() + value_b.capacity();
    let mut map = Map::new();
    map.insert(key_a, Value::Object(Map::new()));
    map.insert(key_b, Value::String(value_b));
    let value = Value::Object(map);
    assert_eq!(
        without_allocations(|| checked_json_backing_bytes(&value, expected)),
        Ok(expected)
    );
    assert_eq!(
        without_allocations(|| checked_json_backing_bytes(&value, expected - 1)),
        Err(Error::ByteLimit)
    );
}

#[test]
fn empty_reserved_collections_keep_their_backing_charge() {
    let mut values = Vec::<Value>::with_capacity(19);
    values.push(Value::String(text("removed", 256)));
    values.clear();
    let expected = values.capacity() * size_of::<Value>();
    assert_eq!(
        checked_json_backing_bytes(&Value::Array(values), expected),
        Ok(expected)
    );
    let empty_text = text("", 193);
    let expected = empty_text.capacity();
    assert_eq!(
        checked_json_backing_bytes(&Value::String(empty_text), expected),
        Ok(expected)
    );
    let (mut graph, _) = graph(DecoderMode::Prefill);
    graph.operations.clear();
    graph.state_inputs.clear();
    graph.state_outputs.clear();
    let expected = graph.output.capacity()
        + graph.operations.capacity() * size_of::<ModelOperation>()
        + graph.state_inputs.capacity() * size_of::<StateTensor>()
        + graph.state_outputs.capacity() * size_of::<StateTensor>();
    assert_eq!(graph.checked_owned_backing_bytes(expected), Ok(expected));
}

#[test]
fn json_depth_16_accepts_17_rejects_without_allocating_to_count() {
    // Alternate objects and arrays so both recursion edges share the same rule.
    let mut value = Value::Null;
    for depth in 0..16 {
        value = if depth % 2 == 0 {
            Value::Array(vec![value])
        } else {
            let mut map = Map::new();
            map.insert("nested".into(), value);
            Value::Object(map)
        };
    }
    let bytes = without_allocations(|| checked_json_backing_bytes(&value, usize::MAX)).unwrap();
    assert_eq!(checked_json_backing_bytes(&value, bytes), Ok(bytes));
    let (mut plan, _) = plan();
    plan.decode.operations[0].attributes = value;
    assert!(without_allocations(|| plan.checked_owned_backing_bytes(usize::MAX)).is_ok());
    let attributes = std::mem::take(&mut plan.decode.operations[0].attributes);
    plan.decode.operations[0].attributes = Value::Array(vec![attributes]);
    assert_eq!(
        without_allocations(|| plan.checked_owned_backing_bytes(usize::MAX)),
        Err(Error::JsonDepth)
    );
    assert_eq!(
        without_allocations(|| checked_json_backing_bytes(
            &plan.decode.operations[0].attributes,
            usize::MAX
        )),
        Err(Error::JsonDepth)
    );
}

#[test]
fn arithmetic_overflow_rejects_without_constructing_impossible_vectors() {
    assert_eq!(
        without_allocations(|| checked_array_bytes::<u64>(usize::MAX)),
        Err(Error::Overflow)
    );
    assert_eq!(checked_array_bytes::<u8>(usize::MAX), Ok(usize::MAX));
    assert_eq!(checked_array_bytes::<u64>(0), Ok(0));
    for value in [Value::Null, Value::Bool(false), Value::Number(1.into())] {
        assert_eq!(checked_json_backing_bytes(&value, 0), Ok(0));
    }
}

#[test]
fn exhausted_cap_rejects_before_walking_later_hostile_attributes() {
    let (mut plan, _) = plan();
    let mut attributes = Value::Object(Map::new());
    for _ in 0..17 {
        attributes = Value::Array(vec![attributes]);
    }
    plan.decode.operations[0].attributes = attributes;
    assert_eq!(
        without_allocations(|| plan.checked_owned_backing_bytes(0)),
        Err(Error::ByteLimit)
    );
    assert_eq!(
        without_allocations(|| plan.checked_owned_backing_bytes(usize::MAX)),
        Err(Error::JsonDepth)
    );
}
