//! Symbolic graph metadata only: no tensors, native arithmetic, or execution.
use super::*;
use crate::{DecoderGraph, ModelOperator};
use serde_json::json;

fn source(family: &str, vocabulary: u64) -> Value {
    json!({"model_type":family,"num_hidden_layers":1,"hidden_size":4,
        "intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":2,"vocab_size":vocabulary,"max_position_embeddings":128,
        "hidden_act":"silu","rms_norm_eps":0.000001,"rope_theta":10000,
        "tie_word_embeddings":true,"attention_bias":false,"attention_dropout":0.0,
        "use_sliding_window":false,"use_cache":true,"max_window_layers":1})
}

fn elements(shape: &[u64]) -> u64 {
    shape
        .iter()
        .try_fold(1u64, |n, &d| n.checked_mul(d))
        .unwrap()
}

fn graph_max(graph: &DecoderGraph) -> u64 {
    graph
        .operations
        .iter()
        .map(|op| elements(&op.output_shape))
        .chain(
            graph
                .state_inputs
                .iter()
                .map(|state| elements(&state.shape)),
        )
        .chain(
            graph
                .state_outputs
                .iter()
                .map(|state| elements(&state.shape)),
        )
        .max()
        .unwrap()
}

fn check_graph(graph: &DecoderGraph, hidden: u64, vocabulary: u64) {
    let unique = |kind| {
        let mut ops = graph.operations.iter().filter(|op| op.operator == kind);
        let op = ops.next().unwrap();
        assert!(ops.next().is_none());
        op
    };
    let last = unique(ModelOperator::LastToken);
    let head = unique(ModelOperator::OutputHead);
    assert_eq!(last.output_shape, [graph.batch, hidden]);
    assert_eq!(head.output_shape, [graph.batch, vocabulary]);
    assert_eq!(head.inputs, [last.id.clone()]);
    let final_norm = graph
        .operations
        .iter()
        .find(|op| op.id == last.inputs[0])
        .unwrap();
    assert_eq!(final_norm.operator, ModelOperator::RmsNorm);
    assert_eq!(
        final_norm.output_shape,
        [graph.batch, graph.query_sequence, hidden]
    );
}

fn budget(summary: SourceScalarSummary) -> DecoderExpansionBudget {
    DecoderExpansionBudget {
        max_layers: summary.layers,
        max_feature_width: summary.feature_width(),
        max_attention_heads: summary.attention_heads.max(summary.kv_heads),
        max_vocabulary: summary.vocabulary,
        max_tensor_elements: summary.semantic_elements(),
    }
}

#[test]
fn registered_readout_summary_matches_symbolic_prefill_decode_and_feedback() {
    // Boundary dimensions are descriptors, never coefficient/token buffers.
    for family in ["qwen2", "qwen3", "llama"] {
        for (batch, rows, outputs, vocabulary) in [
            (1, 1, 1, 1),
            (2, 1, 3, 7),
            (1, 4, 2, 7),
            (2, 4, 2, 524288),
            (1, 30, 2, 151936),
            (1, 31, 1, 151936),
            (1, 30, 2, 1048576),
            (1, 30, 2, 1048577),
            (2, 8, 3, 17),
        ] {
            let document = source(family, vocabulary);
            for step in 0..outputs {
                let workload = DecoderWorkload {
                    batch,
                    max_input_tokens: rows + step,
                    max_new_tokens: outputs - step,
                };
                let summary = source_scalar_summary(&document, workload).unwrap();
                let bytes = serde_json::to_vec(&document).unwrap();
                let plain = crate::source_mappings::lower_json(&bytes, workload, None).unwrap();
                let bounded =
                    crate::source_mappings::lower_json(&bytes, workload, Some(budget(summary)))
                        .unwrap();
                assert_eq!(plain, bounded); // No changed graph/identity/operation order.
                assert_eq!(plain.digest(), bounded.digest());
                check_graph(&plain.prefill, summary.hidden, vocabulary);
                check_graph(&plain.decode, summary.hidden, vocabulary);
                assert_eq!(summary.shape_elements[2], batch * vocabulary);
                assert_eq!(
                    summary.semantic_elements(),
                    graph_max(&plain.prefill).max(graph_max(&plain.decode)),
                    "{family} batch={batch} rows={} outputs={}",
                    rows + step,
                    outputs - step
                );
                let mut short = budget(summary);
                short.max_tensor_elements -= 1;
                assert!(
                    crate::source_mappings::admitted_lowerer(&document, workload, short).is_err()
                );
            }
        }
    }
}

#[test]
fn readout_correction_preserves_all_other_shape_products_and_full_prefix_contract() {
    let document = source("qwen2", 151936);
    let workload = DecoderWorkload {
        batch: 1,
        max_input_tokens: 30,
        max_new_tokens: 2,
    };
    let old = dense_scalar_summary(
        &document,
        workload,
        HeadWidth::HiddenPerHead,
        ReadoutContract::FullPrefix,
    )
    .unwrap();
    let actual = dense_scalar_summary(
        &document,
        workload,
        HeadWidth::HiddenPerHead,
        ReadoutContract::LastToken,
    )
    .unwrap();
    assert_eq!(old.shape_elements[2], 4558080);
    assert_eq!(actual.shape_elements[2], 151936);
    for i in [0, 1, 3, 4, 5, 6] {
        assert_eq!(old.shape_elements[i], actual.shape_elements[i]);
    }
    // Existing nonregistered/full-prefix preflight remains conservative.
    assert!(preflight_dense_source(
        &document,
        workload,
        budget(actual),
        HeadWidth::HiddenPerHead
    )
    .is_err());
    assert!(preflight_dense_readout_source(
        &document,
        workload,
        budget(actual),
        HeadWidth::HiddenPerHead,
        ReadoutContract::LastToken
    )
    .is_ok());
    for family in ["gpt2", "phi3", "gemma4", "qwen3_5", "unknown"] {
        let mut other = document.clone();
        other["model_type"] = json!(family);
        assert_eq!(
            source_scalar_summary(&other, workload),
            Err(SourceScalarSummaryError::Unsupported)
        );
    }
}

#[test]
fn full_prefix_attention_intermediate_and_cache_still_determine_admission() {
    for (rows, outputs, hidden, intermediate, expected) in [
        (30, 2, 4, 8, 1800),   // Full 2-head prefill attention, not last row only.
        (4, 2, 4, 1024, 4096), // Full intermediate activation.
        (1, 8, 128, 8, 512),   // Original retained KV capacity.
    ] {
        let mut document = source("qwen2", 7);
        document["hidden_size"] = json!(hidden);
        document["intermediate_size"] = json!(intermediate);
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: rows,
            max_new_tokens: outputs,
        };
        let summary = source_scalar_summary(&document, workload).unwrap();
        assert_eq!(summary.semantic_elements(), expected);
        let plan = crate::source_mappings::lower_json(
            &serde_json::to_vec(&document).unwrap(),
            workload,
            Some(budget(summary)),
        )
        .unwrap();
        assert_eq!(
            expected,
            graph_max(&plan.prefill).max(graph_max(&plan.decode))
        );
        let mut short = budget(summary);
        short.max_tensor_elements -= 1;
        assert!(crate::source_mappings::admitted_lowerer(&document, workload, short).is_err());
    }
}

#[test]
fn registered_source_field_head_width_remains_independent_of_hidden_width() {
    let mut document = source("qwen3", 7);
    document["head_dim"] = json!(256);
    let workload = DecoderWorkload {
        batch: 2,
        max_input_tokens: 4,
        max_new_tokens: 2,
    };
    let summary = source_scalar_summary(&document, workload).unwrap();
    assert_eq!(summary.query_width, 512);
    assert_eq!(summary.shape_elements[3], 4096);
    let plan = crate::source_mappings::lower_json(
        &serde_json::to_vec(&document).unwrap(),
        workload,
        Some(budget(summary)),
    )
    .unwrap();
    assert_eq!(
        summary.semantic_elements(),
        graph_max(&plan.prefill).max(graph_max(&plan.decode))
    );
    check_graph(&plan.prefill, 4, 7);
}

#[test]
fn last_token_contract_keeps_checked_dimension_overflow_and_zero_refusals() {
    for (batch, rows, outputs, vocabulary) in [
        (0, 1, 1, 7),
        (1, 0, 1, 7),
        (1, 1, 0, 7),
        (u64::MAX, 2, 1, 7),
        (1, u64::MAX, 2, 7),
        (2, 1, 1, u64::MAX),
    ] {
        assert!(source_scalar_summary(
            &source("qwen2", vocabulary),
            DecoderWorkload {
                batch,
                max_input_tokens: rows,
                max_new_tokens: outputs
            }
        )
        .is_err());
    }
}
