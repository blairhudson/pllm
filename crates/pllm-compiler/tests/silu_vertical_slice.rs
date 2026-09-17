use pllm_compiler::{
    compile, lower_model_silu_operation, region_graph_digests, region_program_digest,
    CompileRequest, KernelDescriptor, KernelImplementation, LogicalOperation, MethodDescriptor,
    NumericType, Operator, Representation, SecurityProperties, TensorType,
    SILU_Q7_EXPERIMENT_PROFILE, SILU_Q7_KERNEL_DESCRIPTOR_ID, SILU_Q7_METHOD_ID,
    SILU_Q7_NUMERIC_GRAPH_ID, SILU_Q7_PROTECTED_GRAPH_ID,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderWorkload, ModelOperator};
use pllm_types::{
    canonical_bytes, configuration_digest_bytes, execution_plan_digest, logical_plan_digest,
    privacy_contract_digest, Digest, LockedContext, NamedDigest, PrivacyContract,
    RolePlanReference, VersionedArtifact, LOCKED_CONTEXT_SCHEMA_VERSION,
    PRIVACY_CONTRACT_SCHEMA_VERSION,
};

fn digest(character: char) -> Digest {
    serde_json::from_str(&format!("\"{}\"", character.to_string().repeat(64))).unwrap()
}

fn named(id: &str, character: char) -> NamedDigest {
    NamedDigest {
        id: id.into(),
        digest: digest(character),
    }
}

fn request(shape: Vec<u64>, allow_experimental: bool) -> CompileRequest {
    let privacy = PrivacyContract {
        schema_version: PRIVACY_CONTRACT_SCHEMA_VERSION.into(),
        id: "privacy.experimental-single-evaluator".into(),
        online_parties: 1,
        allow_online_preparation: false,
        allow_client_weights: false,
        allow_he: false,
        allow_experimental,
        required_claims: vec![],
    };
    let tensor = TensorType {
        numeric: NumericType::SignedFixedQ7,
        shape,
    };
    let operations = vec![LogicalOperation {
        id: "layer.0.mlp.silu".into(),
        operator: Operator::Silu,
        output: tensor.clone(),
        input_representation: Representation::ArithmeticLabel,
        output_representation: Representation::ArithmeticLabel,
    }];
    let graphs = region_graph_digests(
        &tensor,
        Representation::ArithmeticLabel,
        &tensor,
        Representation::ArithmeticLabel,
        &operations,
    );
    let configuration_json = canonical_bytes(&serde_json::json!({
        "profile": SILU_Q7_EXPERIMENT_PROFILE,
        "schema": "pllm.experiment.v1"
    }));
    CompileRequest {
        configuration_digest: configuration_digest_bytes(&configuration_json),
        configuration_json,
        context: LockedContext {
            schema_version: LOCKED_CONTEXT_SCHEMA_VERSION.into(),
            profile: SILU_Q7_EXPERIMENT_PROFILE.into(),
            model: named("model.qwen.fixture", '1'),
            tokenizer: named("tokenizer.fixture", '2'),
            semantic_graph: NamedDigest {
                id: "semantic.silu".into(),
                digest: graphs.semantic_graph,
            },
            numeric_graph: NamedDigest {
                id: SILU_Q7_NUMERIC_GRAPH_ID.into(),
                digest: graphs.numeric_graph,
            },
            protected_graph: NamedDigest {
                id: SILU_Q7_PROTECTED_GRAPH_ID.into(),
                digest: graphs.protected_graph,
            },
            roles: vec!["client".into(), "inference".into()],
            privacy_contract: NamedDigest {
                id: privacy.id.clone(),
                digest: privacy_contract_digest(&privacy),
            },
            workload: named("workload.scalar-fixture", '6'),
            target: named("target.cpu", '7'),
            material_requests: vec![named("material.garbled-silu-q7", '8')],
            resource_forecast: named("resources.fixture", '9'),
            compiler: VersionedArtifact {
                id: pllm_compiler::SILU_Q7_COMPILER_ID.into(),
                version: env!("CARGO_PKG_VERSION").into(),
                digest: pllm_compiler::silu_q7_compiler_artifact_digest(),
            },
            execution_role: "inference".into(),
            static_role_plans: vec![RolePlanReference {
                role: "client".into(),
                digest: digest('d'),
            }],
        },
        privacy_contract: privacy,
        input: tensor.clone(),
        input_representation: Representation::ArithmeticLabel,
        output: tensor,
        output_representation: Representation::ArithmeticLabel,
        operations,
        methods: vec![MethodDescriptor {
            id: SILU_Q7_METHOD_ID.into(),
            version: "1".into(),
            operator: Operator::Silu,
            input_representation: Representation::ArithmeticLabel,
            output_representation: Representation::ArithmeticLabel,
            properties: SecurityProperties {
                online_parties: 1,
                needs_online_preparation: false,
                needs_client_weights: false,
                uses_he: false,
                experimental: true,
            },
            artifact_digest: pllm_compiler::silu_q7_method_artifact_digest(),
        }],
        kernels: vec![KernelDescriptor {
            id: SILU_Q7_KERNEL_DESCRIPTOR_ID.into(),
            version: "1".into(),
            method_id: SILU_Q7_METHOD_ID.into(),
            method_version: "1".into(),
            input_numeric: NumericType::SignedFixedQ7,
            output_numeric: NumericType::SignedFixedQ7,
            implementation: KernelImplementation::PllmGarbleSiluQuadraticQ7,
            artifact_digest: pllm_compiler::silu_q7_kernel_artifact_digest(),
        }],
        assurance_results: vec![],
        candidate_evidence: vec![],
    }
}

#[test]
fn compiled_silu_executes_transported_garbled_material() {
    let compiled = compile(&request(vec![3], true)).unwrap();
    let values = [-128, 0, 128];
    let materials = values
        .iter()
        .map(|_| pllm_compiler::prepare_bound_silu_q7_material(&compiled).unwrap())
        .collect::<Vec<_>>();
    let gates = materials
        .iter()
        .map(|material| material.evaluator_payload())
        .collect::<Vec<_>>();
    let labels = materials
        .iter()
        .zip(values)
        .map(|(material, value)| material.encode(value).unwrap())
        .collect::<Vec<_>>();

    let mut evaluator = pllm_compiler::SiluQ7Evaluator::new(&compiled, &gates).unwrap();
    let outputs = evaluator.evaluate(&labels).unwrap();
    let decoded = materials
        .iter()
        .zip(outputs)
        .map(|(material, output)| material.decode(&output).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(decoded, vec![-32, 0, 96]);
    assert_eq!(
        evaluator.evaluate(&labels).unwrap_err(),
        "Q7 SiLU evaluator material was already consumed"
    );
}

#[test]
fn compiler_rejects_experimental_silu_when_contract_forbids_it() {
    let diagnostics = compile(&request(vec![1], false)).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message == "experimental method is not permitted"));
}

#[test]
fn compiler_rejects_plaintext_and_wrong_kernel_fallbacks() {
    let mut plaintext = request(vec![1], true);
    plaintext.input_representation = Representation::ClientPlaintext;
    plaintext.operations[0].input_representation = Representation::ClientPlaintext;
    let diagnostics = compile(&plaintext).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == pllm_compiler::DiagnosticCode::InvalidTensor
            && diagnostic.message.contains("arithmetic_label")
    }));

    let mut wrong_kernel = request(vec![1], true);
    wrong_kernel.kernels[0].implementation = KernelImplementation::PllmCoreMatrixWrap32;
    let diagnostics = compile(&wrong_kernel).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == pllm_compiler::DiagnosticCode::NotImplemented
            && diagnostic.message == "method has no compatible installed kernel"
    }));
}

#[test]
fn compiler_rejects_silu_tensor_above_bounded_slice() {
    let diagnostics = compile(&request(
        vec![pllm_compiler::SILU_Q7_MAX_TENSOR_ELEMENTS as u64 + 1],
        true,
    ))
    .unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == pllm_compiler::DiagnosticCode::InvalidTensor
            && diagnostic.message.contains("supports at most 128 elements")
    }));
}

#[test]
fn compiler_rejects_silu_regions_it_cannot_execute() {
    let mut request = request(vec![1], true);
    let mut second = request.operations[0].clone();
    second.id = "layer.1.mlp.silu".into();
    request.operations.push(second);
    let digests = region_graph_digests(
        &request.input,
        request.input_representation,
        &request.output,
        request.output_representation,
        &request.operations,
    );
    request.context.semantic_graph.digest = digests.semantic_graph;
    request.context.numeric_graph.digest = digests.numeric_graph;
    request.context.protected_graph.digest = digests.protected_graph;
    let diagnostics = compile(&request).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.message == "the Q7 SiLU executor supports only a singleton region"
    }));
}

#[test]
fn executor_revalidates_mutated_installed_descriptors() {
    let mut compiled = compile(&request(vec![1], true)).unwrap();
    compiled.region_program.steps[0].method.artifact_digest = digest('0');
    assert_eq!(
        pllm_compiler::prepare_bound_silu_q7_material(&compiled)
            .err()
            .unwrap(),
        "Q7 SiLU method or kernel descriptor does not match the installed implementation"
    );

    let mut compiled = compile(&request(vec![1], true)).unwrap();
    compiled.lock.compiler.digest = digest('0');
    compiled.locked_context.compiler.digest = digest('0');
    assert_eq!(
        pllm_compiler::prepare_bound_silu_q7_material(&compiled)
            .err()
            .unwrap(),
        "compiled Q7 SiLU compiler artifact is invalid"
    );
}

#[test]
fn compiler_rejects_online_preparation_for_silu() {
    let mut request = request(vec![1], true);
    request.methods[0].properties.needs_online_preparation = true;
    let diagnostics = compile(&request).unwrap_err();
    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message == "method requires forbidden online preparation"));
}

#[test]
fn executor_rejects_material_count_mismatch() {
    let compiled = compile(&request(vec![2], true)).unwrap();
    let error = pllm_compiler::SiluQ7Evaluator::new(&compiled, &[])
        .err()
        .unwrap();
    assert_eq!(error, "Q7 SiLU expected 2 gates, got 0");
}

#[test]
fn malformed_evaluation_burns_bound_gates() {
    let compiled = compile(&request(vec![1], true)).unwrap();
    let material = pllm_compiler::prepare_bound_silu_q7_material(&compiled).unwrap();
    let mut evaluator =
        pllm_compiler::SiluQ7Evaluator::new(&compiled, &[material.evaluator_payload()]).unwrap();
    assert_eq!(
        evaluator.evaluate(&[]).unwrap_err(),
        "Q7 SiLU expected 1 labels, got 0"
    );
    assert_eq!(
        evaluator
            .evaluate(&[material.encode(0).unwrap()])
            .unwrap_err(),
        "Q7 SiLU evaluator material was already consumed"
    );
}

#[test]
fn evaluator_payload_is_plan_bound_and_duplicate_binding_is_burned() {
    let compiled = compile(&request(vec![1], true)).unwrap();
    let material = pllm_compiler::prepare_bound_silu_q7_material(&compiled).unwrap();
    let payload = material.evaluator_payload();
    assert!(
        (12_000..=16_384).contains(&payload.len()),
        "unexpected payload size: {}",
        payload.len()
    );

    let mut other_request = request(vec![1], true);
    other_request.operations[0].id = "layer.1.mlp.silu".into();
    let digests = region_graph_digests(
        &other_request.input,
        other_request.input_representation,
        &other_request.output,
        other_request.output_representation,
        &other_request.operations,
    );
    other_request.context.semantic_graph.digest = digests.semantic_graph;
    other_request.context.numeric_graph.digest = digests.numeric_graph;
    other_request.context.protected_graph.digest = digests.protected_graph;
    let other = compile(&other_request).unwrap();
    let error = pllm_compiler::SiluQ7Evaluator::new(&other, std::slice::from_ref(&payload))
        .err()
        .unwrap();
    assert_eq!(
        error,
        "Q7 SiLU gate commitment does not match compiled plan"
    );

    let error = pllm_compiler::SiluQ7Evaluator::new(&compiled, &[payload])
        .err()
        .unwrap();
    assert_eq!(
        error,
        "Q7 SiLU evaluator material was not issued or was already bound"
    );

    let fresh = pllm_compiler::prepare_bound_silu_q7_material(&compiled).unwrap();
    let _evaluator =
        pllm_compiler::SiluQ7Evaluator::new(&compiled, &[fresh.evaluator_payload()]).unwrap();
}

#[test]
fn editable_envelope_metadata_cannot_rebind_authenticated_gate_rows() {
    let compiled = compile(&request(vec![1], true)).unwrap();
    let material = pllm_compiler::prepare_bound_silu_q7_material(&compiled).unwrap();
    let payload = material.evaluator_payload();

    let mut other_request = request(vec![1], true);
    other_request.context.target.id = "cpu-test-other".into();
    let other = compile(&other_request).unwrap();
    let other_material = pllm_compiler::prepare_bound_silu_q7_material(&other).unwrap();
    let other_payload = other_material.evaluator_payload();
    let other_header_len =
        usize::try_from(u32::from_le_bytes(other_payload[..4].try_into().unwrap())).unwrap();
    let other_header: serde_json::Value =
        serde_json::from_slice(&other_payload[4..4 + other_header_len]).unwrap();
    let header_len = usize::try_from(u32::from_le_bytes(payload[..4].try_into().unwrap())).unwrap();
    let mut header: serde_json::Value =
        serde_json::from_slice(&payload[4..4 + header_len]).unwrap();
    header["compiled_plan_digest"] = other_header["compiled_plan_digest"].clone();
    let encoded_header = pllm_types::canonical_bytes(&header);
    let mut forged = Vec::new();
    forged.extend_from_slice(&u32::try_from(encoded_header.len()).unwrap().to_le_bytes());
    forged.extend_from_slice(&encoded_header);
    forged.extend_from_slice(&payload[4 + header_len..]);

    let error = pllm_compiler::SiluQ7Evaluator::new(&other, &[forged])
        .err()
        .unwrap();
    assert_eq!(
        error,
        "Q7 SiLU evaluator material was not issued or was already bound"
    );

    let mut evaluator = pllm_compiler::SiluQ7Evaluator::new(&compiled, &[payload]).unwrap();
    let encoded = material.encode(-128).unwrap();
    assert_eq!(
        material
            .decode(&evaluator.evaluate(&[encoded]).unwrap()[0])
            .unwrap(),
        -32
    );
}

#[test]
fn compiler_rejects_silu_under_baseline_profile() {
    let mut request = request(vec![1], true);
    request.context.profile = pllm_compiler::BASELINE_EXPERIMENT_PROFILE.into();
    request.configuration_json = serde_json::to_vec(&serde_json::json!({
        "profile": pllm_compiler::BASELINE_EXPERIMENT_PROFILE,
        "schema": "pllm.experiment.v1"
    }))
    .unwrap();
    request.configuration_digest = configuration_digest_bytes(&request.configuration_json);
    let errors = compile(&request).unwrap_err();
    assert!(errors.iter().any(|error| {
        error.subject_id == "configuration.profile"
            && error.message.contains(SILU_Q7_EXPERIMENT_PROFILE)
    }));
}

#[test]
fn verification_rechecks_candidate_policy() {
    let mut compiled = compile(&request(vec![1], true)).unwrap();
    compiled.privacy_contract.allow_experimental = false;
    let privacy_digest = privacy_contract_digest(&compiled.privacy_contract);
    compiled.logical.privacy_contract.digest = privacy_digest.clone();
    compiled.locked_context.privacy_contract.digest = privacy_digest.clone();
    compiled.lock.privacy_contract_digest = privacy_digest;

    let logical_digest = logical_plan_digest(&compiled.logical);
    compiled.execution.logical_plan_digest = logical_digest.clone();
    compiled.lock.logical_plan_digest = logical_digest.clone();
    compiled.region_program.logical_plan_digest = logical_digest;
    let region_digest = region_program_digest(&compiled.region_program);
    compiled
        .execution
        .role_plans
        .iter_mut()
        .find(|plan| plan.role == compiled.region_program.execution_role)
        .unwrap()
        .digest = region_digest;
    compiled.lock.execution_plan_digest = execution_plan_digest(&compiled.execution);

    let error = compiled.verify().unwrap_err();
    assert!(
        error.contains("selected candidate no longer satisfies privacy contract"),
        "{error}"
    );
}

#[test]
fn semantic_qwen_silu_lowers_without_name_parsing() {
    let config = br#"{
        "model_type":"qwen2",
        "hidden_size":896,
        "intermediate_size":4864,
        "num_hidden_layers":1,
        "num_attention_heads":14,
        "num_key_value_heads":2,
        "vocab_size":151936,
        "max_position_embeddings":32768,
        "hidden_act":"silu",
        "rms_norm_eps":1e-6,
        "rope_theta":1000000.0,
        "tie_word_embeddings":true
    }"#;
    let plan = lower_model_json(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    let silu = plan
        .prefill
        .operations
        .iter()
        .find(|operation| operation.operator == ModelOperator::Silu)
        .unwrap();

    let (input, operation) =
        lower_model_silu_operation(&plan, DecoderMode::Prefill, &silu.id).unwrap();
    assert_eq!(operation.operator, Operator::Silu);
    assert_eq!(operation.output, input);
    assert_eq!(input.numeric, NumericType::SignedFixedQ7);
    assert_eq!(
        operation.input_representation,
        Representation::ArithmeticLabel
    );
}

#[test]
fn semantic_silu_lowering_rejects_missing_input_contract() {
    let config = br#"{
        "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
        "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
        "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
        "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
    }"#;
    let mut plan = lower_model_json(
        config,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    let silu_index = plan
        .prefill
        .operations
        .iter()
        .position(|operation| operation.operator == ModelOperator::Silu)
        .unwrap();
    let silu_id = plan.prefill.operations[silu_index].id.clone();
    plan.prefill.operations[silu_index].inputs.clear();
    assert_eq!(
        lower_model_silu_operation(&plan, DecoderMode::Prefill, &silu_id).unwrap_err(),
        format!("incomplete decoder graph: operation {silu_id} has invalid input arity")
    );
}
