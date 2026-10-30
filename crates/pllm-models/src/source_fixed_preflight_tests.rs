//! Source-only authored regressions. No execution or allocation-proof claim.

use super::*;
use crate::source_budget::{
    dense_scalar_summary, preflight_dense_readout_fixed, preflight_dense_readout_source,
    SourceScalarSummary, SourceScalarSummaryError,
};
use serde_json::json;

fn document() -> Value {
    json!({"model_type":"qwen2", "num_hidden_layers":2, "hidden_size":8,
        "intermediate_size":16, "num_attention_heads":2, "num_key_value_heads":1,
        "head_dim":4, "vocab_size":32})
}

fn workload() -> DecoderWorkload {
    DecoderWorkload { batch: 2, max_input_tokens: 3, max_new_tokens: 2 }
}

fn budget() -> DecoderExpansionBudget {
    DecoderExpansionBudget {
        max_layers: 2,
        max_feature_width: 16,
        max_attention_heads: 2,
        max_vocabulary: 32,
        max_tensor_elements: 96,
    }
}

fn fixed(
    source: &Value,
    work: DecoderWorkload,
    caps: DecoderExpansionBudget,
) -> Result<(), SelectionFailure> {
    preflight_dense_readout_fixed(
        source, work, caps, HeadWidth::SourceField, ReadoutContract::LastToken,
    )
}

fn summary(width: HeadWidth, readout: ReadoutContract) -> SourceScalarSummary {
    dense_scalar_summary(&document(), workload(), width, readout).unwrap()
}

fn assert_invalid(error: ModelError) {
    match error {
        ModelError::InvalidConfig(message) => {
            assert_eq!(message, "decoder source expansion budget exceeded or invalid");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

fn forbidden_lower(_: &[u8], _: Value, _: DecoderWorkload) -> Result<DecoderPlan, ModelError> {
    panic!("byte factory must not run during selection")
}

fn forbidden_admitted(_: &Value, _: DecoderWorkload) -> Result<DecoderPlan, ModelError> {
    panic!("Value factory must not run during selection")
}

fn forbidden_preflight(
    _: &Value, _: DecoderWorkload, _: DecoderExpansionBudget,
) -> Result<(), ModelError> {
    panic!("preflight must not run during eligibility selection")
}

const SYNTHETIC: SourceMapping = SourceMapping {
    model_type: "synthetic",
    semantic_family: "synthetic",
    lower: forbidden_lower,
    lower_admitted: Some(forbidden_admitted),
    preflight: Some(SourcePreflight::Other(forbidden_preflight)),
};

#[test]
fn fixed_codes_render_literal_legacy_variants_and_messages() {
    for (code, expected) in [
        (SelectionFailure::NoAdmittedFactory, "source has no admitted Value factory"),
        (SelectionFailure::NoExpansionContract, "source has no bounded expansion contract"),
    ] {
        match code.into_model_error() {
            ModelError::Unsupported(message) => assert_eq!(message, expected),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
    assert_invalid(SelectionFailure::InvalidExpansion.into_model_error());
}

#[test]
fn missing_nonstring_unknown_and_ineligible_types_have_no_factory() {
    for source in [
        json!({}), json!({"model_type":null}), json!({"model_type":7}),
        json!({"model_type":true}), json!({"model_type":[]}),
        json!({"model_type":{}}), json!({"model_type":"unknown"}),
        json!({"model_type":"gpt2"}), json!({"model_type":"gemma4"}),
        json!({"model_type":"phi3"}), json!({"model_type":"qwen3_5"}),
    ] {
        assert_eq!(eligible_mapping(&source).err(), Some(SelectionFailure::NoAdmittedFactory));
        match admitted_lowerer(&source, workload(), budget()).err().unwrap() {
            ModelError::Unsupported(message) => {
                assert_eq!(message, "source has no admitted Value factory");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}

#[test]
fn synthetic_availability_checks_factory_before_contract() {
    const NONE: &[SourceMapping] = &[SourceMapping {
        lower_admitted: None, preflight: None, ..SYNTHETIC
    }];
    const NO_CONTRACT: &[SourceMapping] = &[SourceMapping {
        preflight: None, ..SYNTHETIC
    }];
    let source = json!({"model_type":"synthetic"});
    assert_eq!(eligible_mapping_from(&source, NONE).err(), Some(SelectionFailure::NoAdmittedFactory));
    assert_eq!(eligible_mapping_from(&source, NO_CONTRACT).err(), Some(SelectionFailure::NoExpansionContract));
}

#[test]
fn first_registry_match_is_not_skipped_for_later_eligible_entry() {
    const DUPLICATES: &[SourceMapping] = &[
        SourceMapping { lower_admitted: None, ..SYNTHETIC }, SYNTHETIC,
    ];
    const CONTRACT_DUPLICATES: &[SourceMapping] = &[
        SourceMapping { preflight: None, ..SYNTHETIC }, SYNTHETIC,
    ];
    let source = json!({"model_type":"synthetic"});
    assert_eq!(eligible_mapping_from(&source, DUPLICATES).err(), Some(SelectionFailure::NoAdmittedFactory));
    assert_eq!(eligible_mapping_from(&source, CONTRACT_DUPLICATES).err(), Some(SelectionFailure::NoExpansionContract));
}

#[test]
fn selection_retains_first_mapping_identity_without_invoking_callbacks() {
    static DUPLICATES: &[SourceMapping] = &[SYNTHETIC, SYNTHETIC];
    let selected = eligible_mapping_from(&json!({"model_type":"synthetic"}), DUPLICATES).unwrap();
    assert!(std::ptr::eq(selected.0, &DUPLICATES[0]));
    assert!(!std::ptr::eq(selected.0, &DUPLICATES[1]));
}

#[test]
fn production_eligible_inventory_and_last_token_contract_are_literal() {
    let eligible: Vec<_> = SOURCE_MAPPINGS.iter()
        .filter(|mapping| mapping.lower_admitted.is_some())
        .map(|mapping| mapping.model_type).collect();
    assert_eq!(eligible, ["qwen2", "qwen3", "llama"]);
    // SOURCE_MAPPINGS is a const slice, not a static identity promise across
    // separate promotions. Check identity within the exact slice supplied.
    let registry = SOURCE_MAPPINGS;
    for (index, family) in [(1, "qwen2"), (2, "qwen3"), (3, "llama")] {
        let mut source = document();
        source["model_type"] = json!(family);
        let selected = eligible_mapping_from(&source, registry).unwrap();
        assert!(std::ptr::eq(selected.0, &registry[index]));
        assert_eq!(eligible_mapping(&source).unwrap().0.model_type, family);
        assert!(matches!(selected.2, SourcePreflight::Dense(_, ReadoutContract::LastToken)));
        assert!(admitted_lowerer(&source, workload(), budget()).is_ok());
    }
}

#[test]
fn legacy_other_adapter_still_calls_its_original_callback() {
    fn legacy(_: &Value, _: DecoderWorkload, _: DecoderExpansionBudget) -> Result<(), ModelError> {
        Err(ModelError::Unsupported("legacy Other sentinel".into()))
    }
    let error = SourcePreflight::Other(legacy).check(&document(), workload(), budget()).unwrap_err();
    match error {
        ModelError::Unsupported(message) => assert_eq!(message, "legacy Other sentinel"),
        other => panic!("expected legacy error, got {other:?}"),
    }
}

#[test]
fn literal_geometry_covers_all_seven_shapes_and_both_readouts() {
    let expected = SourceScalarSummary {
        layers: 2, hidden: 8, intermediate: 16, attention_heads: 2,
        kv_heads: 1, head_dim: 4, query_width: 8, kv_width: 4, vocabulary: 32,
        shape_elements: [48, 96, 64, 48, 36, 16, 32], operations_per_phase: 0,
    };
    assert_eq!(summary(HeadWidth::SourceField, ReadoutContract::LastToken), expected);
    assert_eq!(summary(HeadWidth::HiddenPerHead, ReadoutContract::LastToken), expected);
    assert_eq!(summary(HeadWidth::SourceField, ReadoutContract::FullPrefix),
        SourceScalarSummary { shape_elements: [48, 96, 192, 48, 36, 16, 32], ..expected });
}

#[test]
fn scalar_missing_zero_and_wrong_types_reject_with_literal_fixed_code() {
    for key in ["num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"] {
        for value in [json!(0), json!(-1), json!(1.5), json!("2"), json!(true), json!([]), json!({})] {
            let mut source = document();
            source[key] = value;
            assert_eq!(fixed(&source, workload(), budget()), Err(SelectionFailure::InvalidExpansion), "{key}");
        }
        if key != "num_key_value_heads" {
            let mut source = document();
            source.as_object_mut().unwrap().remove(key);
            assert_eq!(fixed(&source, workload(), budget()), Err(SelectionFailure::InvalidExpansion), "{key}");
            source[key] = Value::Null;
            assert_eq!(fixed(&source, workload(), budget()), Err(SelectionFailure::InvalidExpansion), "{key}");
        }
    }
}

#[test]
fn optional_kv_missing_and_null_default_to_attention_heads() {
    for missing in [false, true] {
        let mut source = document();
        if missing { source.as_object_mut().unwrap().remove("num_key_value_heads"); }
        else { source["num_key_value_heads"] = Value::Null; }
        let actual = dense_scalar_summary(&source, workload(), HeadWidth::SourceField, ReadoutContract::LastToken).unwrap();
        assert_eq!(actual.kv_heads, 2);
        assert_eq!(actual.kv_width, 8);
        assert_eq!(actual.shape_elements, [48, 96, 64, 48, 36, 16, 64]);
        assert_eq!(fixed(&source, workload(), budget()), Ok(()));
    }
}

#[test]
fn head_width_contract_keeps_division_and_source_field_behavior() {
    let mut source = document();
    source["hidden_size"] = json!(9);
    source["head_dim"] = json!(3);
    let inferred = dense_scalar_summary(&source, workload(), HeadWidth::HiddenPerHead, ReadoutContract::LastToken).unwrap();
    assert_eq!((inferred.hidden, inferred.head_dim, inferred.query_width, inferred.kv_width), (9, 4, 8, 4));
    let explicit = dense_scalar_summary(&source, workload(), HeadWidth::SourceField, ReadoutContract::LastToken).unwrap();
    assert_eq!((explicit.head_dim, explicit.query_width, explicit.kv_width), (3, 6, 3));
    source["head_dim"] = Value::Null;
    assert!(dense_scalar_summary(&source, workload(), HeadWidth::HiddenPerHead, ReadoutContract::LastToken).is_ok());
    source["hidden_size"] = json!(1);
    assert_eq!(dense_scalar_summary(&source, workload(), HeadWidth::HiddenPerHead, ReadoutContract::LastToken), Err(SourceScalarSummaryError::Invalid));
}

#[test]
fn projection_and_shape_u64_overflows_reject_even_with_max_caps() {
    let unlimited = DecoderExpansionBudget {
        max_layers: u64::MAX, max_feature_width: u64::MAX,
        max_attention_heads: u64::MAX, max_vocabulary: u64::MAX,
        max_tensor_elements: u64::MAX,
    };
    for (key, value) in [("head_dim", u64::MAX), ("num_key_value_heads", u64::MAX),
        ("hidden_size", u64::MAX), ("intermediate_size", u64::MAX), ("vocab_size", u64::MAX)] {
        let mut source = document();
        source[key] = json!(value);
        assert_eq!(dense_scalar_summary(&source, workload(), HeadWidth::SourceField, ReadoutContract::LastToken), Err(SourceScalarSummaryError::Overflow), "{key}");
        assert_eq!(fixed(&source, workload(), unlimited), Err(SelectionFailure::InvalidExpansion), "{key}");
    }
}

#[test]
fn zero_workload_and_capacity_overflow_reject() {
    for work in [
        DecoderWorkload { batch: 0, ..workload() },
        DecoderWorkload { max_input_tokens: 0, ..workload() },
        DecoderWorkload { max_new_tokens: 0, ..workload() },
        DecoderWorkload { max_input_tokens: u64::MAX, max_new_tokens: 1, ..workload() },
        DecoderWorkload { max_input_tokens: 1, max_new_tokens: u64::MAX, ..workload() },
        DecoderWorkload { batch: u64::MAX, ..workload() },
    ] {
        assert_eq!(fixed(&document(), work, budget()), Err(SelectionFailure::InvalidExpansion));
    }
}

#[test]
fn each_budget_class_accepts_exact_boundary_and_rejects_one_short() {
    assert_eq!(fixed(&document(), workload(), budget()), Ok(()));
    for caps in [
        DecoderExpansionBudget { max_layers: 1, ..budget() },
        DecoderExpansionBudget { max_feature_width: 15, ..budget() },
        DecoderExpansionBudget { max_attention_heads: 1, ..budget() },
        DecoderExpansionBudget { max_vocabulary: 31, ..budget() },
        DecoderExpansionBudget { max_tensor_elements: 95, ..budget() },
    ] {
        assert_eq!(fixed(&document(), workload(), caps), Err(SelectionFailure::InvalidExpansion));
    }
    for caps in [
        DecoderExpansionBudget { max_layers: 0, ..budget() },
        DecoderExpansionBudget { max_feature_width: 0, ..budget() },
        DecoderExpansionBudget { max_attention_heads: 0, ..budget() },
        DecoderExpansionBudget { max_vocabulary: 0, ..budget() },
        DecoderExpansionBudget { max_tensor_elements: 0, ..budget() },
    ] {
        assert_eq!(fixed(&document(), workload(), caps), Err(SelectionFailure::InvalidExpansion));
    }
}

#[test]
fn resolved_query_and_kv_widths_have_independent_cap_cases() {
    for (heads, kv_heads) in [(3, 1), (1, 3)] {
        let mut source = document();
        source["num_attention_heads"] = json!(heads);
        source["num_key_value_heads"] = json!(kv_heads);
        source["head_dim"] = json!(8);
        let caps = DecoderExpansionBudget { max_attention_heads: 3, max_tensor_elements: 1024, ..budget() };
        assert_eq!(fixed(&source, workload(), caps), Err(SelectionFailure::InvalidExpansion));
        assert_eq!(fixed(&source, workload(), DecoderExpansionBudget { max_feature_width: 24, ..caps }), Ok(()));
    }
}

#[test]
fn full_prefix_and_last_token_caps_keep_distinct_readout_geometry() {
    assert_eq!(fixed(&document(), workload(), budget()), Ok(()));
    assert_eq!(preflight_dense_readout_fixed(&document(), workload(), budget(), HeadWidth::SourceField,
        ReadoutContract::FullPrefix), Err(SelectionFailure::InvalidExpansion));
    assert_eq!(preflight_dense_readout_fixed(&document(), workload(),
        DecoderExpansionBudget { max_tensor_elements: 192, ..budget() }, HeadWidth::SourceField,
        ReadoutContract::FullPrefix), Ok(()));
    assert_eq!(preflight_dense_readout_fixed(&document(), workload(),
        DecoderExpansionBudget { max_tensor_elements: 191, ..budget() }, HeadWidth::SourceField,
        ReadoutContract::FullPrefix), Err(SelectionFailure::InvalidExpansion));
}

#[test]
fn legacy_dense_boundary_renders_literal_invalid_config() {
    let mut source = document();
    source["num_hidden_layers"] = json!(0);
    assert_invalid(preflight_dense_readout_source(&source, workload(), budget(),
        HeadWidth::SourceField, ReadoutContract::LastToken).unwrap_err());
    assert_invalid(admitted_lowerer(&source, workload(), budget()).err().unwrap());
}

#[test]
fn every_shape_has_a_literal_dominant_boundary_case() {
    // hidden, intermediate, heads, KV heads, head dim, vocabulary, rows, new,
    // independently calculated seven-shape oracle. No wrapper/core comparison.
    for (hidden, intermediate, heads, kv, dim, vocab, rows, new, expected) in [
        (8, 1, 1, 1, 1, 1, 1, 1, [8, 1, 1, 1, 1, 1, 1]),
        (1, 8, 1, 1, 1, 1, 1, 1, [1, 8, 1, 1, 1, 1, 1]),
        (1, 1, 1, 1, 1, 8, 1, 1, [1, 1, 8, 1, 1, 1, 1]),
        (1, 1, 2, 1, 4, 1, 1, 1, [1, 1, 1, 8, 2, 2, 4]),
        (1, 1, 1, 1, 1, 1, 8, 1, [8, 8, 1, 8, 64, 8, 8]),
        (1, 1, 8, 1, 1, 1, 1, 8, [1, 1, 1, 8, 8, 64, 8]),
        (1, 1, 1, 8, 8, 1, 1, 8, [1, 1, 1, 8, 1, 8, 512]),
    ] {
        let source = json!({"num_hidden_layers":1, "hidden_size":hidden,
            "intermediate_size":intermediate, "num_attention_heads":heads,
            "num_key_value_heads":kv, "head_dim":dim, "vocab_size":vocab});
        let work = DecoderWorkload { batch: 1, max_input_tokens: rows, max_new_tokens: new };
        let actual = dense_scalar_summary(&source, work, HeadWidth::SourceField, ReadoutContract::LastToken).unwrap();
        assert_eq!(actual.shape_elements, expected);
        let ceiling = *expected.iter().max().unwrap();
        // Last fixture resolves KV width to 8 heads * 8 dimensions = 64.
        // Admit that width so the 512/511 tensor boundary is not feature-masked.
        let max_feature_width = if (kv, dim) == (8, 8) { 64 } else { 8 };
        let caps = DecoderExpansionBudget {
            max_layers: 1, max_feature_width, max_attention_heads: 8,
            max_vocabulary: 8, max_tensor_elements: ceiling,
        };
        assert_eq!(fixed(&source, work, caps), Ok(()));
        assert_eq!(fixed(&source, work, DecoderExpansionBudget {
            max_tensor_elements: ceiling - 1, ..caps
        }), Err(SelectionFailure::InvalidExpansion));
    }
}
