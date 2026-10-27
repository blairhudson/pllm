use super::*;
use crate::{DecoderMode, ModelOperator, StateKind};
use pllm_types::Digest;
use serde_json::json;

fn limits() -> Limits {
    Limits {
        max_backing_bytes: 1 << 20,
        max_work: 1 << 20,
        max_workspace_bytes: 1 << 20,
        max_json_nodes: 1024,
        max_json_payload_bytes: 1024,
        max_json_string_bytes: 1024,
        max_json_key_bytes: 1024,
        max_json_depth: 16,
    }
}

fn small_plan() -> DecoderPlan {
    DecoderPlan {
        schema_version: "v".into(),
        model_family: "f".into(),
        adapter: "a".into(),
        config_digest: Digest::from_sha256([1; 32]),
        transformations: vec![AppliedMethod {
            component: "c".into(),
            implementation: "impl".into(),
            method_id: "m".into(),
            input_digest: Digest::from_sha256([2; 32]),
            configuration_digest: Digest::from_sha256([3; 32]),
        }],
        prefill: DecoderGraph {
            mode: DecoderMode::Prefill,
            batch: 1,
            query_sequence: 1,
            maximum_key_sequence: 1,
            operations: vec![ModelOperation {
                id: "op".into(),
                operator: ModelOperator::TokenFeedback,
                layer: None,
                state_kind: None,
                inputs: vec!["src".into()],
                output_shape: vec![1, 2],
                attributes: json!({"k": [null, true, 7, "λ", {"z": false}], "empty": []}),
            }],
            state_inputs: vec![StateTensor {
                id: "st".into(),
                layer: None,
                kind: StateKind::Key,
                shape: vec![3],
                maximum_sequence: 1,
            }],
            state_outputs: vec![],
            output: "out".into(),
        },
        decode: DecoderGraph {
            mode: DecoderMode::Decode,
            batch: 1,
            query_sequence: 1,
            maximum_key_sequence: 1,
            operations: vec![],
            state_inputs: vec![],
            state_outputs: vec![],
            output: "d".into(),
        },
        token_feedback: true,
    }
}

fn assert_small_header(header: &Header, plan: &DecoderPlan) {
    assert_eq!(header.graphs(), 2);
    assert_eq!(header.methods(), 1);
    assert_eq!(header.states(), 1);
    assert_eq!(header.operations(), 1);
    assert_eq!(header.input_refs(), 1);
    assert_eq!(header.dimensions(), 3);
    // Eight charged DTO collections plus four JSON containers. Shape vectors
    // have no collection step and are counted only in dimensions.
    assert_eq!(header.containers(), 12);
    assert_eq!(header.entries(), 8);
    assert_eq!(header.text_bytes(), 22 + plan.config_digest.as_str().len()
        + plan.transformations[0].input_digest.as_str().len()
        + plan.transformations[0].configuration_digest.as_str().len());
    assert_eq!(header.key_bytes(), 7);
    assert_eq!(header.json_values(), 9); // keys are not Values; null is a Value
}

#[test]
fn actual_nested_totals_move_from_original_owner_report() {
    let plan = small_plan();
    let header = {
        let mut frames = [const { NativeFrame::empty() }; 17];
        let mut census = Census::new(&mut frames, limits()).unwrap();
        let report = census.plan(&plan).unwrap();
        assert!(matches!(report.input(), Input::Plan(owner) if std::ptr::eq(owner, &plan)));
        assert_small_header(report.header(), &plan);
        assert_eq!(report.usage().json.nodes, report.header().json_values());
        report.into_header()
    };
    assert_small_header(&header, &plan);
}

#[test]
fn repeated_pass_resets_header_not_work_and_failure_never_publishes() {
    let plan = small_plan();
    let raw = {
        let mut frames = [const { NativeFrame::empty() }; 17];
        Census::new(&mut frames, limits()).unwrap().plan(&plan).unwrap().usage().work
    };
    let mut cap = limits();
    cap.max_work = 2 * raw;
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = Census::new(&mut frames, cap).unwrap();
    for pass in 1..=2 {
        let report = census.plan(&plan).unwrap();
        assert_small_header(report.header(), &plan);
        assert_eq!(report.usage().work, pass * raw);
        assert_eq!(report.usage().completed_passes, pass);
    }
    assert!(matches!(census.header, None));
    assert_eq!(census.plan(&plan).unwrap_err(), Error::WorkLimit);
    assert_eq!(census.plan(&plan).unwrap_err(), Error::Poisoned);
    assert_eq!(census.usage().completed_passes, 2);

    let mut cap = limits();
    cap.max_work = raw - 1;
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut short = Census::new(&mut frames, cap).unwrap();
    assert_eq!(short.plan(&plan).unwrap_err(), Error::WorkLimit);
    assert_eq!(short.usage().completed_passes, 0);
    assert_eq!(short.plan(&plan).unwrap_err(), Error::Poisoned);
}

#[test]
fn json_depth_failure_keeps_partial_header_private_and_poisons_session() {
    let plan = small_plan();
    let mut rejected = Value::Null;
    for _ in 0..17 {
        rejected = Value::Array(vec![rejected]);
    }
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = Census::new(&mut frames, limits()).unwrap();
    {
        let report = census.plan(&plan).unwrap();
        assert!(matches!(report.input(), Input::Plan(owner) if std::ptr::eq(owner, &plan)));
        assert_small_header(report.header(), &plan);
    }
    let before = census.usage();
    assert_eq!(before.completed_passes, 1);
    assert!(matches!(&census.header, None));

    assert_eq!(census.json(&rejected).unwrap_err(),
        Error::Backing(crate::owned_backing::Error::JsonDepth));
    let failed = census.usage();
    assert_eq!(failed.completed_passes, before.completed_passes);
    assert!(failed.work > before.work);
    assert!(failed.json.work > before.json.work);
    assert!(census.failed);
    assert!(matches!(&census.header, Some(header) if header.json_values() > 0));

    assert_eq!(census.plan(&plan).unwrap_err(), Error::Poisoned);
    assert_eq!(census.usage(), failed);
    assert!(matches!(&census.header, Some(_)));
}

#[test]
fn enlarged_containing_types_and_each_new_slot_are_in_layout_quote() {
    let quote = resources_for_slots(17).unwrap();
    assert_eq!(quote.control_bytes, size_of::<Census<'_, '_>>());
    let slots = 5 * size_of::<Header>() + 4 * size_of::<Option<Header>>()
        + size_of::<Report<'_>>() + 13 * size_of::<&mut Option<Header>>()
        + 11 * size_of::<&mut Header>();
    assert_eq!(header_slots_bytes().unwrap(), slots);
    assert_eq!(quote.completion_bytes,
        size_of::<Report<'_>>() + 2 * size_of::<Input<'_>>() + slots);
    assert!(size_of::<Census<'_, '_>>() >= size_of::<Option<Header>>());
    assert!(size_of::<Report<'_>>() >= size_of::<Header>());
    assert!(quote.quotation_bytes >= 2 * size_of::<Result<Census<'_, '_>, Error>>());
    assert!(quote.result_slots_bytes >= 2 * size_of::<Result<Report<'_>, Error>>());
}

#[test]
fn overflow_during_visit_or_completion_poison_without_report() {
    let value = Value::Null;
    let mut frames = [const { NativeFrame::empty() }; 17];
    let mut census = Census::new(&mut frames, limits()).unwrap();
    census.usage.completed_passes = usize::MAX;
    assert_eq!(census.json(&value).unwrap_err(), Error::Overflow);
    assert_eq!(census.json(&value).unwrap_err(), Error::Poisoned);
    // Test-only access exercises checked increment without a second traversal
    // or a public constructor/adoption route.
    match &mut census.header {
        Some(header) => header.json_values = usize::MAX,
        None => panic!("failed completion must not extract header"),
    }
    let mut backing = Backing::new(usize::MAX);
    assert_eq!(charge_event(&mut backing, &mut census.header,
        Event::Value { value: &value, depth: 1, number_bytes: 0 }), Err(Error::Overflow));
}
