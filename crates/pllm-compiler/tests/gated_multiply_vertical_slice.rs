use pllm_compiler::{
    decoder_coverage, define_q14_to_q7_rescale_region, lower_model_gated_multiply_q7_regions,
    lower_model_gated_multiply_q7_regions_with_components,
    prepare_bound_gated_multiply_q7_material, prepare_bound_gated_multiply_q7_tensor_material,
    CapabilityLevel, GatedMultiplyQ7Evaluator, GatedMultiplyQ7TensorEvaluator,
    TensorResourcePolicy, GATED_MULTIPLY_Q7_BINARY_TABLE_COMPONENT_ID,
    GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
    GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS,
    GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
    GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES, GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID,
    GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID, GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload};
use std::io::{Cursor, Read};

fn qwen_plan(intermediate_size: u64) -> DecoderPlan {
    let config = format!(
        r#"{{
            "model_type":"qwen2",
            "hidden_size":2,
            "intermediate_size":{intermediate_size},
            "num_hidden_layers":1,
            "num_attention_heads":1,
            "num_key_value_heads":1,
            "vocab_size":8,
            "max_position_embeddings":8,
            "hidden_act":"silu",
            "rms_norm_eps":1e-6,
            "rope_theta":10000.0,
            "tie_word_embeddings":true
        }}"#
    );
    lower_model_json(
        config.as_bytes(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

#[test]
fn semantic_qwen_gated_multiply_lowers_with_exact_provenance() {
    let plan = qwen_plan(1);
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        let regions = lower_model_gated_multiply_q7_regions(&plan, mode).unwrap();
        let [region] = regions.as_slice() else {
            panic!("expected one gated MLP region")
        };
        assert_eq!(region.layer, 0);
        assert_eq!(region.gate_linear_operation_id, "layer.0.gate_proj");
        assert_eq!(region.silu_operation_id, "layer.0.silu");
        assert_eq!(region.up_linear_operation_id, "layer.0.up_proj");
        assert_eq!(region.multiply_operation_id, "layer.0.gated_multiply");
        assert_eq!(region.gate_rescale.target_input_index, 0);
        assert_eq!(region.up_rescale.target_input_index, 1);
        assert_eq!(region.input.shape, vec![1, 1, 1]);
        assert_eq!(region.numeric_graph_id, GATED_MULTIPLY_Q7_NUMERIC_GRAPH_ID);
        assert_eq!(
            region.protected_graph_id,
            GATED_MULTIPLY_Q7_PROTECTED_GRAPH_ID
        );
    }
    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let silu = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == pllm_models::ModelOperator::Silu)
        .unwrap();
    let multiply = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == pllm_models::ModelOperator::Multiply)
        .unwrap();
    assert_eq!(silu.level, CapabilityLevel::ExecutableRegion);
    assert_eq!(
        silu.component.as_deref(),
        Some("pllm/agc-silu-q7@0.1.0-alpha.1-experimental")
    );
    assert!(silu.blocker.contains("chunked"));
    assert!(silu.blocker.contains("whole-decoder scheduling"));
    assert_eq!(multiply.level, CapabilityLevel::ExecutableRegion);
    assert_eq!(
        multiply.component.as_deref(),
        Some("pllm/agc-gated-multiply-q7@0.1.0-alpha.1-experimental")
    );
    assert!(multiply.blocker.contains("chunked"));
    assert!(multiply.blocker.contains("whole-decoder scheduling"));
    assert!(!coverage.complete);
}

#[test]
fn composed_gate_executes_without_exposing_the_silu_value() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    assert_eq!(payload.len(), 245_397);
    assert!(payload.len() <= GATED_MULTIPLY_Q7_MAX_EVALUATOR_PAYLOAD_BYTES);
    let mut evaluator = GatedMultiplyQ7Evaluator::new(&region, &payload).unwrap();
    let output = evaluator
        .evaluate(
            &material.encode_gate(-65).unwrap(),
            &material.encode_up(127).unwrap(),
        )
        .unwrap();
    assert_eq!(
        material.decode(&output).unwrap(),
        pllm_core::gated_multiply_q7(-65, 127).unwrap()
    );
    assert_eq!(
        evaluator
            .evaluate(
                &material.encode_gate(-65).unwrap(),
                &material.encode_up(127).unwrap(),
            )
            .unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn forged_payload_does_not_burn_authentic_material() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    let mut forged = payload.clone();
    *forged.last_mut().unwrap() ^= 1;
    assert!(GatedMultiplyQ7Evaluator::new(&region, &forged).is_err());

    let mut evaluator = GatedMultiplyQ7Evaluator::new(&region, &payload).unwrap();
    let output = evaluator
        .evaluate(
            &material.encode_gate(64).unwrap(),
            &material.encode_up(-96).unwrap(),
        )
        .unwrap();
    assert_eq!(
        material.decode(&output).unwrap(),
        pllm_core::gated_multiply_q7(64, -96).unwrap()
    );
}

#[test]
fn abandoned_material_releases_its_issuance_registration() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    assert!(material.cancel().unwrap());
    assert!(!material.cancel().unwrap());
    assert_eq!(
        GatedMultiplyQ7Evaluator::new(&region, &payload)
            .err()
            .unwrap(),
        "gated Q7 multiply evaluator material was not issued or was already bound"
    );

    let payload = {
        let abandoned = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
        abandoned.evaluator_payload()
    };
    assert_eq!(
        GatedMultiplyQ7Evaluator::new(&region, &payload)
            .err()
            .unwrap(),
        "gated Q7 multiply evaluator material was not issued or was already bound"
    );
}

#[test]
fn authentic_wrong_region_binding_burns_material() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let payload = material.evaluator_payload();
    let mut other_region = region.clone();
    other_region.mode = DecoderMode::Decode;
    assert_eq!(
        GatedMultiplyQ7Evaluator::new(&other_region, &payload)
            .err()
            .unwrap(),
        "gated Q7 multiply commitment does not match region"
    );
    assert_eq!(
        GatedMultiplyQ7Evaluator::new(&region, &payload)
            .err()
            .unwrap(),
        "gated Q7 multiply evaluator material was not issued or was already bound"
    );
}

#[test]
fn malformed_input_and_cross_lane_labels_burn_the_program() {
    let plan = qwen_plan(1);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let mut evaluator =
        GatedMultiplyQ7Evaluator::new(&region, &material.evaluator_payload()).unwrap();
    assert!(evaluator
        .evaluate(
            &material.encode_up(3).unwrap(),
            &material.encode_gate(5).unwrap(),
        )
        .is_err());
    assert_eq!(
        evaluator
            .evaluate(
                &material.encode_gate(5).unwrap(),
                &material.encode_up(3).unwrap(),
            )
            .unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn executor_rejects_non_scalar_and_mutated_regions() {
    let plan = qwen_plan(2);
    let region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        "gated Q7 multiply scheduler permits at most 1 elements, received 2"
    );

    let plan = qwen_plan(1);
    let mut region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    region.method.id = "forged-method".into();
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        "gated Q7 multiply region does not match the installed contract"
    );

    let mut region = lower_model_gated_multiply_q7_regions(&plan, DecoderMode::Prefill)
        .unwrap()
        .remove(0);
    region.multiply_operation_id = "layer.0.attention_scores".into();
    region.up_rescale = define_q14_to_q7_rescale_region(
        &region.up_linear_operation_id,
        &region.multiply_operation_id,
        1,
        region.input.shape.clone(),
    )
    .unwrap();
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        "gated Q7 multiply region was not lowered from the decoder plan"
    );
}

#[test]
fn method_components_are_distinct_and_numerically_equivalent() {
    let plan = qwen_plan(1);
    let mut payload_lengths = Vec::new();
    for component_id in [
        GATED_MULTIPLY_Q7_BINARY_TABLE_COMPONENT_ID,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
    ] {
        let region = lower_model_gated_multiply_q7_regions_with_components(
            &plan,
            DecoderMode::Prefill,
            component_id,
            pllm_compiler::GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID,
            1,
        )
        .unwrap()
        .remove(0);
        assert_eq!(region.method_component_id, component_id);
        let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
        let payload = material.evaluator_payload();
        let mut evaluator = GatedMultiplyQ7Evaluator::new(&region, &payload).unwrap();
        let output = evaluator
            .evaluate(
                &material.encode_gate(-65).unwrap(),
                &material.encode_up(127).unwrap(),
            )
            .unwrap();
        assert_eq!(
            material.decode(&output).unwrap(),
            pllm_core::gated_multiply_q7(-65, 127).unwrap()
        );
        payload_lengths.push(payload.len());
    }
    assert!(payload_lengths[0] > payload_lengths[1]);
}

#[test]
fn independent_lane_schedule_executes_atomically() {
    let plan = qwen_plan(4);
    let region = lower_model_gated_multiply_q7_regions_with_components(
        &plan,
        DecoderMode::Prefill,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
        GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
        4,
    )
    .unwrap()
    .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    assert_eq!(material.element_count(), 4);
    assert_eq!(material.evaluator_payload().len(), 980_573);

    let gates = [-128, -65, 64, 128];
    let ups = [127, -96, -96, 128];
    let gate_labels = material.encode_gates(&gates).unwrap();
    let up_labels = material.encode_ups(&ups).unwrap();
    let gate_refs = gate_labels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = up_labels.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut evaluator =
        GatedMultiplyQ7Evaluator::new(&region, &material.evaluator_payload()).unwrap();
    let outputs = evaluator.evaluate_tensor(&gate_refs, &up_refs).unwrap();
    assert_eq!(
        material.decode_tensor(&outputs).unwrap(),
        gates
            .iter()
            .zip(ups)
            .map(|(gate, up)| pllm_core::gated_multiply_q7(*gate, up).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        evaluator.evaluate_tensor(&gate_refs, &up_refs).unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn one_bad_lane_burns_bundle_without_partial_output() {
    let plan = qwen_plan(2);
    let region = lower_model_gated_multiply_q7_regions_with_components(
        &plan,
        DecoderMode::Prefill,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
        GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
        2,
    )
    .unwrap()
    .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let mut gates = material.encode_gates(&[5, 6]).unwrap();
    let ups = material.encode_ups(&[7, 8]).unwrap();
    gates[1].push(0);
    let gate_refs = gates.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = ups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut evaluator =
        GatedMultiplyQ7Evaluator::new(&region, &material.evaluator_payload()).unwrap();
    assert!(evaluator.evaluate_tensor(&gate_refs, &up_refs).is_err());
    assert_eq!(
        evaluator.evaluate_tensor(&gate_refs, &up_refs).unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn lane_order_is_authenticated_and_one_swap_burns_bundle() {
    let plan = qwen_plan(2);
    let region = lower_model_gated_multiply_q7_regions_with_components(
        &plan,
        DecoderMode::Prefill,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
        GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
        2,
    )
    .unwrap()
    .remove(0);
    let material = prepare_bound_gated_multiply_q7_material(&plan, &region).unwrap();
    let mut gates = material.encode_gates(&[5, 6]).unwrap();
    let ups = material.encode_ups(&[7, 8]).unwrap();
    gates.swap(0, 1);
    let gate_refs = gates.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let up_refs = ups.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut evaluator =
        GatedMultiplyQ7Evaluator::new(&region, &material.evaluator_payload()).unwrap();
    assert!(evaluator.evaluate_tensor(&gate_refs, &up_refs).is_err());
    assert_eq!(
        evaluator.evaluate_tensor(&gate_refs, &up_refs).unwrap_err(),
        "gated Q7 multiply evaluator material was already consumed"
    );
}

#[test]
fn component_lowering_rejects_unknown_or_invalid_combinations() {
    let plan = qwen_plan(1);
    for (method, schedule, elements, expected) in [
        (
            "paper-name",
            pllm_compiler::GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID,
            1,
            "unsupported gated Q7 multiply method implementation",
        ),
        (
            GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
            "paper-name",
            1,
            "unsupported gated Q7 multiply schedule implementation",
        ),
        (
            GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
            pllm_compiler::GATED_MULTIPLY_Q7_SCALAR_SCHEDULE_COMPONENT_ID,
            2,
            "scalar gated Q7 multiply scheduling requires one element",
        ),
        (
            GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
            GATED_MULTIPLY_Q7_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
            5,
            "independent-lane gated Q7 multiply scheduling supports 2..=4 elements",
        ),
    ] {
        assert_eq!(
            lower_model_gated_multiply_q7_regions_with_components(
                &plan,
                DecoderMode::Prefill,
                method,
                schedule,
                elements,
            )
            .unwrap_err(),
            expected
        );
    }
}

fn tensor_policy(elements: u64) -> TensorResourcePolicy {
    TensorResourcePolicy {
        max_elements: elements,
        max_chunk_elements: 2,
        max_chunk_bytes: 600_000,
        max_body_bytes: 2_000_000,
        max_output_bytes: elements * 1_024,
        max_client_material_bytes: 2_000_000,
        max_working_bytes: 40_000_000,
    }
}

fn chunked_region(
    plan: &DecoderPlan,
    mode: DecoderMode,
    elements: usize,
) -> pllm_compiler::ModelGatedMultiplyQ7Region {
    lower_model_gated_multiply_q7_regions_with_components(
        plan,
        mode,
        GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
        GATED_MULTIPLY_Q7_CHUNKED_INDEPENDENT_LANES_SCHEDULE_COMPONENT_ID,
        elements,
    )
    .unwrap()
    .remove(0)
}

#[test]
fn chunked_tensor_schedule_streams_more_than_four_lanes_atomically() {
    let plan = qwen_plan(5);
    let region = chunked_region(&plan, DecoderMode::Prefill, 5);
    let policy = tensor_policy(5);
    let mut body = Vec::new();
    let material =
        prepare_bound_gated_multiply_q7_tensor_material(&plan, &region, policy.clone(), &mut body)
            .unwrap();
    assert_eq!(material.element_count(), 5);
    assert_eq!(material.ticket().chunk_count(), 3);
    assert_eq!(material.ticket().body_bytes() as usize, body.len());
    assert!(body.len() > 1_000_000);

    let gates = [-128, -65, 0, 64, 128];
    let ups = [128, 127, -96, -65, 64];
    let gate_labels = material.encode_gates(&gates).unwrap();
    let up_labels = material.encode_ups(&ups).unwrap();
    let mut evaluator =
        GatedMultiplyQ7TensorEvaluator::claim(&region, material.ticket(), &policy).unwrap();
    let output = evaluator
        .evaluate(&mut Cursor::new(body), &gate_labels, &up_labels)
        .unwrap();
    assert_eq!(
        material.decode_tensor(&output).unwrap(),
        gates
            .iter()
            .zip(ups)
            .map(|(gate, up)| pllm_core::gated_multiply_q7(*gate, up).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        evaluator
            .evaluate(&mut Cursor::new(Vec::new()), &gate_labels, &up_labels)
            .unwrap_err(),
        "gated Q7 tensor evaluator material was already consumed"
    );
}

#[test]
fn authentic_wrong_region_claim_burns_whole_tensor() {
    let plan = qwen_plan(1);
    let region = chunked_region(&plan, DecoderMode::Prefill, 1);
    let other = chunked_region(&plan, DecoderMode::Decode, 1);
    let policy = tensor_policy(1);
    let mut body = Vec::new();
    let material =
        prepare_bound_gated_multiply_q7_tensor_material(&plan, &region, policy.clone(), &mut body)
            .unwrap();
    assert!(GatedMultiplyQ7TensorEvaluator::claim(&other, material.ticket(), &policy).is_err());
    assert_eq!(
        GatedMultiplyQ7TensorEvaluator::claim(&region, material.ticket(), &policy)
            .err()
            .unwrap(),
        "gated Q7 tensor material was not issued or was already claimed"
    );
}

struct ObservedReader<'a> {
    source: Cursor<Vec<u8>>,
    observed: &'a std::cell::Cell<bool>,
}

struct InterruptOnceReader<R> {
    inner: R,
    interrupted: bool,
}

impl<R: Read> Read for InterruptOnceReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if !self.interrupted {
            self.interrupted = true;
            return Err(std::io::Error::from(std::io::ErrorKind::Interrupted));
        }
        self.inner.read(buffer)
    }
}

impl Read for ObservedReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.observed.set(true);
        self.source.read(buffer)
    }
}

#[test]
fn tensor_claim_burns_before_reader_callbacks_and_corruption_releases_nothing() {
    let plan = qwen_plan(1);
    let region = chunked_region(&plan, DecoderMode::Prefill, 1);
    let policy = tensor_policy(1);
    let mut body = Vec::new();
    let material =
        prepare_bound_gated_multiply_q7_tensor_material(&plan, &region, policy.clone(), &mut body)
            .unwrap();
    let gates = material.encode_gates(&[5]).unwrap();
    let ups = material.encode_ups(&[7]).unwrap();
    let observed = std::cell::Cell::new(false);
    let mut evaluator =
        GatedMultiplyQ7TensorEvaluator::claim(&region, material.ticket(), &policy).unwrap();
    assert!(!observed.get());
    *body.last_mut().unwrap() ^= 1;
    let mut reader = ObservedReader {
        source: Cursor::new(body),
        observed: &observed,
    };
    let error = evaluator.evaluate(&mut reader, &gates, &ups).unwrap_err();
    assert!(error.contains("body commitment"));
    assert!(observed.get());
    assert_eq!(
        evaluator.evaluate(&mut reader, &gates, &ups).unwrap_err(),
        "gated Q7 tensor evaluator material was already consumed"
    );
}

#[test]
fn chunked_tensor_preflights_body_capacity_before_writing() {
    let plan = qwen_plan(1);
    let region = chunked_region(&plan, DecoderMode::Prefill, 1);
    let mut policy = tensor_policy(1);
    policy.max_body_bytes = 1;
    let mut body = Vec::new();
    let error =
        match prepare_bound_gated_multiply_q7_tensor_material(&plan, &region, policy, &mut body) {
            Ok(_) => panic!("undersized body policy was accepted"),
            Err(error) => error,
        };
    assert!(error.contains("resource policy"));
    assert!(body.is_empty());
}

#[test]
fn chunked_tensor_preflights_client_capacity_before_allocating_or_writing() {
    let plan = qwen_plan(5);
    let region = chunked_region(&plan, DecoderMode::Prefill, 5);
    let mut policy = tensor_policy(5);
    policy.max_client_material_bytes = 1;
    let mut body = Vec::new();
    let error =
        match prepare_bound_gated_multiply_q7_tensor_material(&plan, &region, policy, &mut body) {
            Ok(_) => panic!("undersized client policy was accepted"),
            Err(error) => error,
        };
    assert!(error.contains("resource policy"));
    assert!(body.is_empty());
}

#[test]
fn chunked_tensor_retries_interrupted_authenticated_body_reads() {
    let plan = qwen_plan(5);
    let region = chunked_region(&plan, DecoderMode::Prefill, 5);
    let mut body = Vec::new();
    let material = prepare_bound_gated_multiply_q7_tensor_material(
        &plan,
        &region,
        tensor_policy(5),
        &mut body,
    )
    .unwrap();
    let gates = material.encode_gates(&[1, 2, 3, 4, 5]).unwrap();
    let ups = material.encode_ups(&[5, 4, 3, 2, 1]).unwrap();
    let policy = tensor_policy(5);
    let mut evaluator =
        GatedMultiplyQ7TensorEvaluator::claim(&region, material.ticket(), &policy).unwrap();
    let mut reader = InterruptOnceReader {
        inner: Cursor::new(body),
        interrupted: false,
    };
    let labels = evaluator.evaluate(&mut reader, &gates, &ups).unwrap();
    assert_eq!(material.decode_tensor(&labels).unwrap().len(), 5);
}

#[test]
fn oversized_intermediate_keeps_gated_regions_primitive() {
    let plan = qwen_plan(GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS as u64 + 1);
    let region = chunked_region(
        &plan,
        DecoderMode::Prefill,
        GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS,
    );
    assert_eq!(
        prepare_bound_gated_multiply_q7_material(&plan, &region)
            .err()
            .unwrap(),
        format!(
            "gated Q7 multiply scheduler permits at most {} elements, received {}",
            GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS,
            GATED_MULTIPLY_Q7_CHUNKED_MAX_TENSOR_ELEMENTS + 1
        )
    );
    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let silu = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == pllm_models::ModelOperator::Silu)
        .unwrap();
    let multiply = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == pllm_models::ModelOperator::Multiply)
        .unwrap();
    assert_eq!(silu.level, CapabilityLevel::Primitive);
    assert_eq!(
        silu.component.as_deref(),
        Some("pllm/agc-silu-q7@0.1.0-alpha.1-experimental")
    );
    assert_eq!(multiply.level, CapabilityLevel::Primitive);
    assert_eq!(
        multiply.component.as_deref(),
        Some("pllm/agc-gated-multiply-q7@0.1.0-alpha.1-experimental")
    );
    assert!(!coverage.complete);
}
