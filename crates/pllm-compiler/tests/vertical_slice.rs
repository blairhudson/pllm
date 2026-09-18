use pllm_compiler::{
    compile, compile_document, decoder_coverage, define_q14_to_q7_rescale_region, diagnostics_json,
    execute_model_last_token, execute_model_output_head, execute_model_reshape,
    execute_model_residual, execute_q14_to_q7_rescale, execute_wrap32,
    lower_model_greedy_token_selection_regions, lower_model_last_token_regions,
    lower_model_linear_operation, lower_model_linear_regions, lower_model_output_head_regions,
    lower_model_reshape_regions, lower_model_residual_regions, q14_to_q7_rescale_region_digest,
    CandidateEvidence, CapabilityLevel, CompileRequest, Diagnostic, DiagnosticCode,
    KernelDescriptor, KernelImplementation, LogicalOperation, MethodDescriptor,
    ModelLastTokenSelection, ModelReshapeLayout, NumericType, Operator, Representation,
    SecurityProperties, TensorType, COMPILE_REQUEST_SCHEMA_VERSION,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderWorkload, ModelOperator};
use pllm_types::{
    assurance_result_digest, configuration_digest_bytes, privacy_contract_digest, AssuranceOrigin,
    AssuranceOutcome, AssuranceResult, ClaimRequirement, Digest, EvidenceReference,
    ImplementationRefinement, LockedContext, NamedDigest, PrivacyContract, RolePlanReference,
    VersionedArtifact, ASSURANCE_RESULT_SCHEMA_VERSION, LOCKED_CONTEXT_SCHEMA_VERSION,
    PRIVACY_CONTRACT_SCHEMA_VERSION,
};
use std::collections::BTreeSet;

fn digest(character: char) -> Digest {
    serde_json::from_str(&format!("\"{}\"", character.to_string().repeat(64))).unwrap()
}

fn named(id: &str, character: char) -> NamedDigest {
    NamedDigest {
        id: id.into(),
        digest: digest(character),
    }
}

fn tensor(batch: u64, width: u64) -> TensorType {
    TensorType {
        numeric: NumericType::Wrap32,
        shape: vec![batch, width],
    }
}

fn wrap32(value: i32) -> u32 {
    value as u32
}

fn privacy() -> PrivacyContract {
    PrivacyContract {
        schema_version: PRIVACY_CONTRACT_SCHEMA_VERSION.into(),
        id: "privacy.single-evaluator".into(),
        online_parties: 1,
        allow_online_preparation: false,
        allow_client_weights: false,
        allow_he: false,
        allow_experimental: false,
        required_claims: vec![ClaimRequirement {
            claim_id: "composition-privacy".into(),
            accepted_outcomes: BTreeSet::from([AssuranceOutcome::ProvedInModel]),
            accepted_implementation_refinements: BTreeSet::from([
                ImplementationRefinement::TestedDifferential,
                ImplementationRefinement::ProvedForLockedCode,
            ]),
        }],
    }
}

fn context(privacy: &PrivacyContract) -> LockedContext {
    LockedContext {
        schema_version: LOCKED_CONTEXT_SCHEMA_VERSION.into(),
        profile: "baseline.masked_linear_cpu".into(),
        model: named("model.fixture", '1'),
        tokenizer: named("tokenizer.fixture", '2'),
        semantic_graph: named("semantic.linear", '3'),
        numeric_graph: named("numeric.wrap32", '4'),
        protected_graph: named("protected.masked-ring", '5'),
        roles: vec!["client".into(), "inference".into(), "preparation".into()],
        privacy_contract: NamedDigest {
            id: privacy.id.clone(),
            digest: privacy_contract_digest(privacy),
        },
        workload: named("workload.fixture", '6'),
        target: named("target.cpu", '7'),
        material_requests: vec![named("material.mask", '8')],
        resource_forecast: named("resources.fixture", '9'),
        compiler: VersionedArtifact {
            id: "pllm/compiler".into(),
            version: "0.17.0-alpha.1".into(),
            digest: digest('a'),
        },
        execution_role: "inference".into(),
        static_role_plans: vec![
            RolePlanReference {
                role: "client".into(),
                digest: digest('d'),
            },
            RolePlanReference {
                role: "preparation".into(),
                digest: digest('e'),
            },
        ],
    }
}

fn method(id: &str) -> MethodDescriptor {
    MethodDescriptor {
        id: id.into(),
        version: "1".into(),
        operator: Operator::Linear,
        input_representation: Representation::MaskedRing,
        output_representation: Representation::MaskedRing,
        properties: SecurityProperties {
            online_parties: 1,
            needs_online_preparation: false,
            needs_client_weights: false,
            uses_he: false,
            experimental: false,
        },
        artifact_digest: digest('b'),
    }
}

fn kernel(id: &str, method_id: &str) -> KernelDescriptor {
    KernelDescriptor {
        id: id.into(),
        version: "1".into(),
        method_id: method_id.into(),
        method_version: "1".into(),
        input_numeric: NumericType::Wrap32,
        output_numeric: NumericType::Wrap32,
        implementation: KernelImplementation::PllmCoreMatrixWrap32,
        artifact_digest: digest('c'),
    }
}

fn assurance(outcome: AssuranceOutcome, refinement: ImplementationRefinement) -> AssuranceResult {
    AssuranceResult {
        schema_version: ASSURANCE_RESULT_SCHEMA_VERSION.into(),
        id: "assurance.composition".into(),
        claim_id: "composition-privacy".into(),
        outcome,
        scope: "masked-linear@1/core-wrap32@1".into(),
        origin: AssuranceOrigin::SolverExecuted,
        implementation_refinement: refinement,
        assumptions: vec!["declared single-evaluator model".into()],
        evidence_paths: vec!["research/assurance/composition.json".into()],
        tool: Some("fixture-checker".into()),
        tool_version: Some("1".into()),
        code_digest: None,
    }
}

fn evidence_reference(result: &AssuranceResult) -> EvidenceReference {
    EvidenceReference {
        id: result.id.clone(),
        digest: assurance_result_digest(result),
    }
}

fn configuration_json() -> Vec<u8> {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/fixtures/experiment.valid.json"
    ))
    .unwrap();
    pllm_types::canonical_bytes(&fixture)
}

fn request(batch: u64, input_width: u64, output_width: u64) -> CompileRequest {
    let privacy_contract = privacy();
    let input = tensor(batch, input_width);
    let output = tensor(batch, output_width);
    let operations = vec![LogicalOperation {
        id: "linear-0".into(),
        operator: Operator::Linear,
        output: output.clone(),
        input_representation: Representation::MaskedRing,
        output_representation: Representation::MaskedRing,
    }];
    let mut context = context(&privacy_contract);
    let graphs = pllm_compiler::region_graph_digests(
        &input,
        Representation::MaskedRing,
        &output,
        Representation::MaskedRing,
        &operations,
    );
    context.semantic_graph.digest = graphs.semantic_graph;
    context.numeric_graph.digest = graphs.numeric_graph;
    context.protected_graph.digest = graphs.protected_graph;
    let result = assurance(
        AssuranceOutcome::ProvedInModel,
        ImplementationRefinement::TestedDifferential,
    );
    let configuration_json = configuration_json();
    CompileRequest {
        configuration_digest: configuration_digest_bytes(&configuration_json),
        configuration_json,
        context,
        privacy_contract,
        input,
        input_representation: Representation::MaskedRing,
        output,
        output_representation: Representation::MaskedRing,
        operations,
        methods: vec![method("masked-linear")],
        kernels: vec![kernel("core-wrap32", "masked-linear")],
        assurance_results: vec![result.clone()],
        candidate_evidence: vec![CandidateEvidence {
            method_id: "masked-linear".into(),
            method_version: "1".into(),
            method_artifact_digest: digest('b'),
            kernel_id: "core-wrap32".into(),
            kernel_version: "1".into(),
            kernel_artifact_digest: digest('c'),
            assurance_results: vec![evidence_reference(&result)],
        }],
    }
}

fn document_bytes(request: &CompileRequest) -> Vec<u8> {
    let configuration: serde_json::Value =
        serde_json::from_slice(&request.configuration_json).unwrap();
    pllm_types::canonical_bytes(&serde_json::json!({
        "schema_version": COMPILE_REQUEST_SCHEMA_VERSION,
        "configuration": configuration,
        "context": request.context,
        "privacy_contract": request.privacy_contract,
        "input": request.input,
        "input_representation": request.input_representation,
        "output": request.output,
        "output_representation": request.output_representation,
        "operations": request.operations,
        "methods": request.methods,
        "kernels": request.kernels,
        "assurance_results": request.assurance_results,
        "candidate_evidence": request.candidate_evidence,
    }))
}

fn refresh_context(request: &mut CompileRequest) {
    request.context.privacy_contract = NamedDigest {
        id: request.privacy_contract.id.clone(),
        digest: privacy_contract_digest(&request.privacy_contract),
    };
}

fn refresh_region_graphs(request: &mut CompileRequest) {
    let graphs = pllm_compiler::region_graph_digests(
        &request.input,
        request.input_representation,
        &request.output,
        request.output_representation,
        &request.operations,
    );
    request.context.semantic_graph.digest = graphs.semantic_graph;
    request.context.numeric_graph.digest = graphs.numeric_graph;
    request.context.protected_graph.digest = graphs.protected_graph;
    refresh_context(request);
}

fn only_code(request: &CompileRequest) -> DiagnosticCode {
    let diagnostics = compile(request).unwrap_err();
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    diagnostics[0].code
}

#[test]
fn semantic_qwen_linear_compiles_and_executes_without_name_parsing() {
    let config = br#"{
        "model_type":"qwen2","hidden_size":8,"intermediate_size":16,
        "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
        "vocab_size":32,"max_position_embeddings":32,"hidden_act":"silu",
        "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
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
    let linear = plan
        .prefill
        .operations
        .iter()
        .find(|operation| operation.operator == ModelOperator::Linear)
        .unwrap();
    let prefill_regions = lower_model_linear_regions(&plan, DecoderMode::Prefill).unwrap();
    assert_eq!(prefill_regions.len(), 7);
    assert!(prefill_regions.iter().all(|region| {
        region.mode == DecoderMode::Prefill
            && region.layer == Some(0)
            && region.weight_id.starts_with("model.layers.0.")
            && region.weight_id.ends_with(".weight")
    }));
    let decode_regions = lower_model_linear_regions(&plan, DecoderMode::Decode).unwrap();
    assert_eq!(decode_regions.len(), 7);
    assert!(decode_regions
        .iter()
        .all(|region| region.input.shape[0] == 1));
    let reshape_regions = lower_model_reshape_regions(&plan, DecoderMode::Prefill).unwrap();
    assert_eq!(reshape_regions.len(), 4);
    let heads = reshape_regions
        .iter()
        .find(|region| region.layout == ModelReshapeLayout::BatchHeadsSequenceFeature)
        .unwrap();
    let values = (0_u32..16).collect::<Vec<_>>();
    let permuted = execute_model_reshape(heads, &values).unwrap();
    assert_eq!(
        permuted,
        vec![0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
    );
    let hidden = reshape_regions
        .iter()
        .find(|region| region.layout == ModelReshapeLayout::BatchSequenceHidden)
        .unwrap();
    assert_eq!(execute_model_reshape(hidden, &permuted).unwrap(), values);
    assert!(execute_model_reshape(heads, &[0; 15]).is_err());
    assert_eq!(
        lower_model_reshape_regions(&plan, DecoderMode::Decode)
            .unwrap()
            .len(),
        4
    );
    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let linear_coverage = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::Linear)
        .unwrap();
    assert_eq!(linear_coverage.occurrences, 14);
    assert_eq!(linear_coverage.level, CapabilityLevel::ExecutableRegion);
    let reshape_coverage = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::Reshape)
        .unwrap();
    assert_eq!(reshape_coverage.occurrences, 8);
    assert_eq!(reshape_coverage.level, CapabilityLevel::ExecutableRegion);
    let residual_regions = lower_model_residual_regions(&plan, DecoderMode::Prefill).unwrap();
    assert_eq!(residual_regions.len(), 2);
    assert_eq!(residual_regions[0].layer, Some(0));
    assert_eq!(residual_regions[0].input.shape, vec![1, 2, 8]);
    assert_eq!(residual_regions[0].input_ids.len(), 2);
    let left = vec![u32::MAX; 16];
    let right = vec![2; 16];
    assert_eq!(
        execute_model_residual(&residual_regions[0], &left, &right).unwrap(),
        vec![1; 16]
    );
    assert!(execute_model_residual(&residual_regions[0], &left[..15], &right).is_err());
    let mut malformed_residual = residual_regions[0].clone();
    malformed_residual.output.shape = vec![1, 1, 8];
    assert!(execute_model_residual(&malformed_residual, &left, &right).is_err());
    assert_eq!(
        lower_model_residual_regions(&plan, DecoderMode::Decode)
            .unwrap()
            .len(),
        2
    );
    let residual_coverage = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::ResidualAdd)
        .unwrap();
    assert_eq!(residual_coverage.occurrences, 4);
    assert_eq!(residual_coverage.level, CapabilityLevel::ExecutableRegion);
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        assert_eq!(
            lower_model_last_token_regions(&plan, mode).unwrap().len(),
            1
        );
    }
    let last_token_coverage = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::LastToken)
        .unwrap();
    assert_eq!(last_token_coverage.occurrences, 2);
    assert_eq!(last_token_coverage.level, CapabilityLevel::ExecutableRegion);
    assert!(last_token_coverage
        .blocker
        .contains("whole-decoder scheduling is unavailable"));
    let output_heads = lower_model_output_head_regions(&plan, DecoderMode::Prefill).unwrap();
    assert_eq!(output_heads.len(), 1);
    let output_head = &output_heads[0];
    assert_eq!(output_head.mode, DecoderMode::Prefill);
    assert_eq!(output_head.layer, None);
    assert_eq!(output_head.input_id, "last_hidden");
    assert_eq!(output_head.weight_id, "model.embed_tokens.weight");
    assert_eq!(output_head.input.shape, vec![1, 8]);
    assert_eq!(output_head.output.shape, vec![1, 32]);
    let untied_config = String::from_utf8(config.to_vec()).unwrap().replace(
        "\"tie_word_embeddings\":true",
        "\"tie_word_embeddings\":false",
    );
    let untied_plan = lower_model_json(
        untied_config.as_bytes(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    assert_eq!(
        lower_model_output_head_regions(&untied_plan, DecoderMode::Prefill).unwrap()[0].weight_id,
        "lm_head.weight"
    );
    assert_eq!(
        execute_model_output_head(
            output_head,
            &[1; 256],
            &(1_u32..=8).collect::<Vec<_>>(),
            1,
            false,
        )
        .unwrap(),
        vec![36; 32]
    );
    assert!(execute_model_output_head(output_head, &[1; 255], &[1; 8], 1, false).is_err());
    assert!(execute_model_output_head(output_head, &[1; 256], &[1; 7], 1, false).is_err());
    let mut rank_three = output_head.clone();
    rank_three.input.shape = vec![1, 1, 8];
    rank_three.output.shape = vec![1, 1, 32];
    assert_eq!(
        execute_model_output_head(&rank_three, &[1; 256], &[1; 8], 1, false).unwrap(),
        vec![8; 32]
    );
    assert_eq!(
        lower_model_output_head_regions(&plan, DecoderMode::Decode)
            .unwrap()
            .len(),
        1
    );
    let output_head_coverage = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::OutputHead)
        .unwrap();
    assert_eq!(output_head_coverage.occurrences, 2);
    assert_eq!(
        output_head_coverage.level,
        CapabilityLevel::ExecutableRegion
    );
    assert!(!coverage.complete);
    let (input, operation) =
        lower_model_linear_operation(&plan, DecoderMode::Prefill, &linear.id).unwrap();
    assert_eq!(input.shape, vec![2, 8]);
    assert_eq!(operation.output.shape, vec![2, 8]);
    assert_eq!(operation.input_representation, Representation::MaskedRing);

    let mut compile_request = request(2, 8, 8);
    compile_request.input = input;
    compile_request.output = operation.output.clone();
    compile_request.operations = vec![operation];
    refresh_region_graphs(&mut compile_request);
    let compiled = compile(&compile_request).unwrap();
    let output = execute_wrap32(
        &compiled,
        &[1; 64],
        &(1_u32..=16).collect::<Vec<_>>(),
        1,
        false,
    )
    .unwrap();
    assert_eq!(output, [vec![36; 8], vec![100; 8]].concat());
}

#[test]
fn gemma_last_token_executes_length_aware_selection() {
    let plan = lower_model_json(
        include_bytes!("../../pllm-models/tests/fixtures/gemma-4-E2B-it-3e22461f-config.json"),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap();
    for mode in [DecoderMode::Prefill, DecoderMode::Decode] {
        assert_eq!(
            lower_model_last_token_regions(&plan, mode).unwrap().len(),
            1
        );
    }
    let prefill = &lower_model_last_token_regions(&plan, DecoderMode::Prefill).unwrap()[0];
    assert_eq!(prefill.selection, ModelLastTokenSelection::LastValid);
    let elements: usize = prefill.input.shape.iter().product::<u64>() as usize;
    let inner = prefill.output.shape.iter().product::<u64>() as usize;
    let input: Vec<u32> = (0..elements as u32).collect();
    let output = execute_model_last_token(prefill, &input, Some(&[1])).unwrap();
    assert_eq!(output, input[..inner]);
    assert!(lower_model_greedy_token_selection_regions(&plan, DecoderMode::Prefill).is_err());
    let coverage = decoder_coverage(&plan, "research.single_evaluator");
    let last_token = coverage
        .operators
        .iter()
        .find(|coverage| coverage.operator == ModelOperator::LastToken)
        .unwrap();
    assert_eq!(last_token.level, CapabilityLevel::ExecutableRegion);
}

fn assert_json_fixture(actual: &[u8], fixture: &str) {
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(actual).unwrap(),
        serde_json::from_str::<serde_json::Value>(fixture).unwrap()
    );
}

#[test]
fn canonical_compile_document_matches_direct_request() {
    let request = request(2, 3, 2);
    let bytes = document_bytes(&request);
    let direct = compile(&request).unwrap();
    let document = compile_document(&bytes).unwrap();
    assert_eq!(document, direct);
    assert_eq!(
        bytes,
        include_bytes!("../../../schemas/fixtures/compile-request.valid.json")
    );
}

fn document_error_code(bytes: &[u8]) -> DiagnosticCode {
    let errors = compile_document(bytes).unwrap_err();
    assert_eq!(errors.len(), 1, "{errors:?}");
    errors[0].code
}

#[test]
fn compile_document_rejects_noncanonical_duplicate_unknown_and_wrong_schema() {
    let canonical = document_bytes(&request(2, 3, 2));
    let value: serde_json::Value = serde_json::from_slice(&canonical).unwrap();
    assert_eq!(
        document_error_code(&serde_json::to_vec_pretty(&value).unwrap()),
        DiagnosticCode::InvalidDocument
    );

    let source = String::from_utf8(canonical.clone()).unwrap();
    let duplicate = source.replacen('{', "{\"schema_version\":\"pllm.compile_request.v1\",", 1);
    assert_eq!(
        document_error_code(duplicate.as_bytes()),
        DiagnosticCode::InvalidDocument
    );

    let mut unknown = value.clone();
    unknown["methods"][0]
        .as_object_mut()
        .unwrap()
        .insert("reviewed".into(), serde_json::Value::Bool(true));
    assert_eq!(
        document_error_code(&pllm_types::canonical_bytes(&unknown)),
        DiagnosticCode::InvalidDocument
    );

    let mut wrong_document = value.clone();
    wrong_document["schema_version"] = serde_json::json!("pllm.compile_request.v2");
    assert_eq!(
        document_error_code(&pllm_types::canonical_bytes(&wrong_document)),
        DiagnosticCode::InvalidDocument
    );

    let mut wrong_configuration = value;
    wrong_configuration["configuration"]["schema"] = serde_json::json!("pllm.experiment.v2");
    assert_eq!(
        document_error_code(&pllm_types::canonical_bytes(&wrong_configuration)),
        DiagnosticCode::InvalidDocument
    );

    let mut unused_component: serde_json::Value =
        serde_json::from_slice(&document_bytes(&request(2, 3, 2))).unwrap();
    unused_component["configuration"]["pipeline"]["components"]["nonlinear"] = serde_json::json!({
        "component": pllm_compiler::GATED_MULTIPLY_Q7_R03_CRT_COMPONENT_ID,
        "params": {}
    });
    assert_eq!(
        document_error_code(&pllm_types::canonical_bytes(&unused_component)),
        DiagnosticCode::InvalidDocument
    );
}

#[test]
fn diagnostic_json_is_sorted_deduplicated_and_stable() {
    let missing = Diagnostic {
        code: DiagnosticCode::MissingClaim,
        subject_id: "z-candidate".into(),
        message: "missing claim composition-privacy".into(),
    };
    let invalid = Diagnostic {
        code: DiagnosticCode::InvalidId,
        subject_id: "a-method".into(),
        message: "method identity is malformed".into(),
    };
    assert_eq!(
        diagnostics_json(&[missing.clone(), invalid, missing]),
        br#"[{"code":"E_INVALID_ID","message":"method identity is malformed","subject_id":"a-method"},{"code":"E_MISSING_CLAIM","message":"missing claim composition-privacy","subject_id":"z-candidate"}]"#
    );
}

#[test]
fn canonical_manifests_have_exact_boundaries_and_stable_digests() {
    let fixture_request = request(2, 3, 2);
    let compiled = compile(&fixture_request).unwrap();
    assert_json_fixture(
        &compiled.logical_json(),
        include_str!("../../../schemas/fixtures/logical-plan.valid.json"),
    );
    assert_json_fixture(
        &compiled.execution_json(),
        include_str!("../../../schemas/fixtures/execution-plan.valid.json"),
    );
    assert_json_fixture(
        &compiled.lock_json(),
        include_str!("../../../schemas/fixtures/plan-lock.valid.json"),
    );
    assert_json_fixture(
        &compiled.region_program_json(),
        include_str!("../../../schemas/fixtures/region-program.valid.json"),
    );
    assert_json_fixture(
        &pllm_types::assurance_result_bytes(&fixture_request.assurance_results[0]),
        include_str!("../../../schemas/fixtures/assurance-result.valid.json"),
    );
    assert_json_fixture(
        &pllm_types::canonical_bytes(&fixture_request.privacy_contract),
        include_str!("../../../schemas/fixtures/privacy-contract.valid.json"),
    );
    assert_json_fixture(
        &pllm_types::canonical_bytes(&fixture_request.context),
        include_str!("../../../schemas/fixtures/locked-context.valid.json"),
    );
    assert_eq!(
        compiled.lock.logical_plan_digest.as_str(),
        "a2dd71d2dd5e3917b4a4578dbde9d1287a285df6fd722d3acb97f7dbcaf26961"
    );
    assert_eq!(
        compiled.lock.execution_plan_digest.as_str(),
        "bdba1776ec7496d2a41f5508c6efb0d56bc66c0ccfd49267f90ce6c6811f31b7"
    );
    assert_eq!(
        pllm_compiler::region_program_digest(&compiled.region_program).as_str(),
        "f0e1fb6b830785dd74da3cfeb633c69355e05a214160245c722030a003fed215"
    );
    assert_eq!(
        pllm_types::plan_lock_digest(&compiled.lock).as_str(),
        "32e02bfcc5a235ab148b77508ee9fe7acba64a4b51dae222eb96538882ba13ce"
    );
    assert_eq!(
        compiled.logical.configuration_digest.as_str(),
        "863af238d286ed9970ee710a9c4694a14fb43fea2ffde883b9e43ca59f90197e"
    );

    let logical = String::from_utf8(compiled.logical_json()).unwrap();
    assert!(!logical.contains("method"));
    assert!(!logical.contains("kernel"));
    assert!(!logical.contains("evidence_references"));
    let manifests = String::from_utf8(
        [
            compiled.logical_json(),
            compiled.execution_json(),
            compiled.lock_json(),
        ]
        .concat(),
    )
    .unwrap();
    for conclusion in [
        "\"outcome\"",
        "\"origin\"",
        "\"scope\"",
        "\"implementation_refinement\"",
        "\"assumptions\"",
        "\"evidence_paths\"",
        "declared single-evaluator model",
    ] {
        assert!(!manifests.contains(conclusion), "found {conclusion}");
    }
    assert_eq!(
        compiled
            .execution
            .role_plans
            .iter()
            .map(|plan| plan.role.as_str())
            .collect::<Vec<_>>(),
        ["client", "inference", "preparation"]
    );
    assert_eq!(
        compiled.logical.roles,
        ["client", "inference", "preparation"]
    );
    assert_eq!(
        compiled.execution.role_plans[1].digest,
        pllm_compiler::region_program_digest(&compiled.region_program)
    );
    assert!(compiled.verify().is_ok());
}

#[test]
fn candidate_order_is_irrelevant() {
    let mut first = request(2, 3, 2);
    let mut alternate_method = method("z-method");
    alternate_method.artifact_digest = digest('d');
    let mut alternate_kernel = kernel("z-kernel", "z-method");
    alternate_kernel.artifact_digest = digest('e');
    first.methods.push(alternate_method);
    first.kernels.push(alternate_kernel);
    first.candidate_evidence.push(CandidateEvidence {
        method_id: "z-method".into(),
        method_version: "1".into(),
        method_artifact_digest: digest('d'),
        kernel_id: "z-kernel".into(),
        kernel_version: "1".into(),
        kernel_artifact_digest: digest('e'),
        assurance_results: vec![evidence_reference(&first.assurance_results[0])],
    });
    let mut second = first.clone();
    second.methods.reverse();
    second.kernels.reverse();
    second.assurance_results.reverse();
    second.candidate_evidence.reverse();
    assert_eq!(compile(&first).unwrap(), compile(&second).unwrap());
}

#[test]
fn configuration_identity_is_canonical_plain_sha256() {
    let canonical = configuration_json();
    assert_eq!(
        configuration_digest_bytes(&canonical).as_str(),
        "863af238d286ed9970ee710a9c4694a14fb43fea2ffde883b9e43ca59f90197e"
    );

    let mut noncanonical = request(1, 3, 2);
    noncanonical.configuration_json =
        include_bytes!("../../../schemas/fixtures/experiment.valid.json").to_vec();
    noncanonical.configuration_digest =
        configuration_digest_bytes(&noncanonical.configuration_json);
    assert_eq!(only_code(&noncanonical), DiagnosticCode::InvalidContext);

    let mut wrong_digest = request(1, 3, 2);
    wrong_digest.configuration_digest = digest('f');
    assert_eq!(only_code(&wrong_digest), DiagnosticCode::InvalidContext);

    let mut wrong_schema = request(1, 3, 2);
    let mut value: serde_json::Value =
        serde_json::from_slice(&wrong_schema.configuration_json).unwrap();
    value["schema"] = "pllm.experiment.v2".into();
    wrong_schema.configuration_json = pllm_types::canonical_bytes(&value);
    wrong_schema.configuration_digest =
        configuration_digest_bytes(&wrong_schema.configuration_json);
    assert_eq!(only_code(&wrong_schema), DiagnosticCode::InvalidContext);
}

#[test]
fn configuration_profile_must_match_locked_context() {
    let mut mismatched = request(1, 2, 2);
    let mut configuration: serde_json::Value =
        serde_json::from_slice(&mismatched.configuration_json).unwrap();
    configuration["pipeline"]["profile"] = "research.single_evaluator".into();
    mismatched.configuration_json = pllm_types::canonical_bytes(&configuration);
    mismatched.configuration_digest = configuration_digest_bytes(&mismatched.configuration_json);

    let diagnostics = compile(&mismatched).unwrap_err();
    assert!(diagnostics.iter().any(|diagnostic| {
        diagnostic.code == DiagnosticCode::InvalidContext
            && diagnostic.subject_id == "configuration.profile"
            && diagnostic.message == "configuration profile does not match locked context"
    }));
}

#[test]
fn graph_digests_preserve_layer_boundaries() {
    let base = request(1, 3, 2);
    let base_graphs = pllm_compiler::region_graph_digests(
        &base.input,
        base.input_representation,
        &base.output,
        base.output_representation,
        &base.operations,
    );

    let mut numeric_input = base.input.clone();
    numeric_input.numeric = NumericType::SignedFixed16;
    let mut numeric_output = base.output.clone();
    numeric_output.numeric = NumericType::SignedFixed16;
    let mut numeric_operations = base.operations.clone();
    numeric_operations[0].output.numeric = NumericType::SignedFixed16;
    let numeric_graphs = pllm_compiler::region_graph_digests(
        &numeric_input,
        base.input_representation,
        &numeric_output,
        base.output_representation,
        &numeric_operations,
    );
    assert_eq!(base_graphs.semantic_graph, numeric_graphs.semantic_graph);
    assert_ne!(base_graphs.numeric_graph, numeric_graphs.numeric_graph);
    assert_ne!(base_graphs.protected_graph, numeric_graphs.protected_graph);

    let mut protected_operations = base.operations.clone();
    protected_operations[0].input_representation = Representation::ClientPlaintext;
    let protected_graphs = pllm_compiler::region_graph_digests(
        &base.input,
        Representation::ClientPlaintext,
        &base.output,
        base.output_representation,
        &protected_operations,
    );
    assert_eq!(base_graphs.semantic_graph, protected_graphs.semantic_graph);
    assert_eq!(base_graphs.numeric_graph, protected_graphs.numeric_graph);
    assert_ne!(
        base_graphs.protected_graph,
        protected_graphs.protected_graph
    );

    let mut unsupported_numeric = request(1, 3, 2);
    unsupported_numeric.input = numeric_input;
    unsupported_numeric.output = numeric_output;
    unsupported_numeric.operations = numeric_operations;
    refresh_region_graphs(&mut unsupported_numeric);
    assert!(compile(&unsupported_numeric)
        .unwrap_err()
        .iter()
        .any(|diagnostic| diagnostic.code == DiagnosticCode::InvalidTensor));
}

#[test]
fn logical_digest_excludes_method_kernel_and_evidence() {
    let base = compile(&request(1, 3, 2)).unwrap();
    let base_lock_digest = pllm_types::plan_lock_digest(&base.lock);

    let mut method_changed = request(1, 3, 2);
    method_changed.methods[0].id = "other-method".into();
    method_changed.methods[0].artifact_digest = digest('d');
    method_changed.candidate_evidence[0].method_artifact_digest = digest('d');
    method_changed.kernels[0].method_id = "other-method".into();
    method_changed.candidate_evidence[0].method_id = "other-method".into();
    let changed = compile(&method_changed).unwrap();
    assert_eq!(
        base.lock.logical_plan_digest,
        changed.lock.logical_plan_digest
    );
    assert_ne!(
        base.lock.execution_plan_digest,
        changed.lock.execution_plan_digest
    );
    assert_ne!(
        base_lock_digest,
        pllm_types::plan_lock_digest(&changed.lock)
    );

    let mut kernel_changed = request(1, 3, 2);
    kernel_changed.kernels[0].id = "other-kernel".into();
    kernel_changed.kernels[0].artifact_digest = digest('e');
    kernel_changed.candidate_evidence[0].kernel_artifact_digest = digest('e');
    kernel_changed.candidate_evidence[0].kernel_id = "other-kernel".into();
    let changed = compile(&kernel_changed).unwrap();
    assert_eq!(
        base.lock.logical_plan_digest,
        changed.lock.logical_plan_digest
    );
    assert_ne!(
        base.lock.execution_plan_digest,
        changed.lock.execution_plan_digest
    );

    let mut evidence_changed = request(1, 3, 2);
    evidence_changed.assurance_results[0]
        .assumptions
        .push("additional assumption".into());
    evidence_changed.candidate_evidence[0].assurance_results[0] =
        evidence_reference(&evidence_changed.assurance_results[0]);
    let changed = compile(&evidence_changed).unwrap();
    assert_eq!(
        base.lock.logical_plan_digest,
        changed.lock.logical_plan_digest
    );
    assert_ne!(
        base.lock.execution_plan_digest,
        changed.lock.execution_plan_digest
    );
    assert_ne!(
        base_lock_digest,
        pllm_types::plan_lock_digest(&changed.lock)
    );
}

#[test]
fn rejects_inconsistent_context_and_incomplete_steps() {
    let mut context = request(1, 3, 2);
    context.context.semantic_graph.digest = digest('f');
    assert_eq!(only_code(&context), DiagnosticCode::InvalidContext);

    let mut incomplete_roles = request(1, 3, 2);
    incomplete_roles.context.static_role_plans.pop();
    assert_eq!(only_code(&incomplete_roles), DiagnosticCode::InvalidContext);

    let mut privacy_digest = request(1, 3, 2);
    privacy_digest.privacy_contract.online_parties = 2;
    assert_eq!(only_code(&privacy_digest), DiagnosticCode::InvalidContract);

    let mut incomplete = request(1, 3, 2);
    incomplete.output.shape[1] = 3;
    refresh_region_graphs(&mut incomplete);
    assert_eq!(only_code(&incomplete), DiagnosticCode::IncompleteStep);

    let mut empty = request(1, 3, 2);
    empty.operations.clear();
    refresh_region_graphs(&mut empty);
    assert_eq!(only_code(&empty), DiagnosticCode::IncompleteStep);

    let mut unsupported = request(1, 3, 2);
    unsupported.operations[0].operator = Operator::Unsupported;
    refresh_region_graphs(&mut unsupported);
    assert_eq!(only_code(&unsupported), DiagnosticCode::UnsupportedOperator);

    let mut malformed = request(1, 3, 2);
    malformed.operations[0].id.clear();
    refresh_region_graphs(&mut malformed);
    assert_eq!(only_code(&malformed), DiagnosticCode::InvalidId);

    let mut representation = request(1, 3, 2);
    representation.methods[0].input_representation = Representation::ClientPlaintext;
    assert_eq!(only_code(&representation), DiagnosticCode::Representation);
}

#[test]
fn conversion_is_resolved_into_manifest_and_region_program() {
    let mut value = request(1, 3, 2);
    value.input_representation = Representation::ClientPlaintext;
    value.operations.insert(
        0,
        LogicalOperation {
            id: "convert-0".into(),
            operator: Operator::Conversion,
            output: tensor(1, 3),
            input_representation: Representation::ClientPlaintext,
            output_representation: Representation::MaskedRing,
        },
    );
    let mut conversion_method = method("explicit-mask");
    conversion_method.operator = Operator::Conversion;
    conversion_method.input_representation = Representation::ClientPlaintext;
    conversion_method.artifact_digest = digest('d');
    value.methods.push(conversion_method);
    value.kernels.push(KernelDescriptor {
        id: "mask-kernel".into(),
        version: "1".into(),
        method_id: "explicit-mask".into(),
        method_version: "1".into(),
        input_numeric: NumericType::Wrap32,
        output_numeric: NumericType::Wrap32,
        implementation: KernelImplementation::ExplicitRepresentationConversion,
        artifact_digest: digest('e'),
    });
    value.candidate_evidence.push(CandidateEvidence {
        method_id: "explicit-mask".into(),
        method_version: "1".into(),
        method_artifact_digest: digest('d'),
        kernel_id: "mask-kernel".into(),
        kernel_version: "1".into(),
        kernel_artifact_digest: digest('e'),
        assurance_results: vec![evidence_reference(&value.assurance_results[0])],
    });
    refresh_region_graphs(&mut value);

    let compiled = compile(&value).unwrap();
    assert_eq!(compiled.region_program.steps.len(), 2);
    assert_eq!(compiled.execution.components.len(), 4);
    assert_eq!(compiled.execution.conversions.len(), 1);
    assert_eq!(compiled.execution.conversions[0].id, "convert-0");
    assert!(compiled.verify().is_ok());
}

#[test]
fn assurance_policy_fails_closed_and_binds_locked_code() {
    let mut missing = request(1, 3, 2);
    missing.candidate_evidence[0].assurance_results.clear();
    assert_eq!(only_code(&missing), DiagnosticCode::MissingClaim);

    let mut refuted = request(1, 3, 2);
    refuted.assurance_results[0].outcome = AssuranceOutcome::RefutedInScope;
    refuted.candidate_evidence[0].assurance_results[0] =
        evidence_reference(&refuted.assurance_results[0]);
    assert_eq!(only_code(&refuted), DiagnosticCode::UnacceptedOutcome);
    refuted.privacy_contract.required_claims[0]
        .accepted_outcomes
        .insert(AssuranceOutcome::RefutedInScope);
    refresh_context(&mut refuted);
    assert!(compile(&refuted).is_ok());

    for artifact in [digest('b'), digest('c')] {
        let mut proved = request(1, 3, 2);
        proved.assurance_results[0].implementation_refinement =
            ImplementationRefinement::ProvedForLockedCode;
        proved.assurance_results[0].code_digest = Some(artifact);
        proved.candidate_evidence[0].assurance_results[0] =
            evidence_reference(&proved.assurance_results[0]);
        assert!(compile(&proved).is_ok());
    }
    let mut wrong_artifact = request(1, 3, 2);
    wrong_artifact.assurance_results[0].implementation_refinement =
        ImplementationRefinement::ProvedForLockedCode;
    wrong_artifact.assurance_results[0].code_digest = Some(digest('f'));
    wrong_artifact.candidate_evidence[0].assurance_results[0] =
        evidence_reference(&wrong_artifact.assurance_results[0]);
    assert_eq!(
        only_code(&wrong_artifact),
        DiagnosticCode::UnacceptedRefinement
    );

    let mut tampered = request(1, 3, 2);
    tampered.assurance_results[0].scope = "tampered".into();
    assert_eq!(only_code(&tampered), DiagnosticCode::InvalidAssurance);
}

#[test]
fn full_lock_and_region_tampering_is_rejected() {
    let compiled = compile(&request(1, 3, 2)).unwrap();

    let mut logical = compiled.clone();
    logical.logical.model.id = "tampered-model".into();
    assert_eq!(
        logical.verify().unwrap_err(),
        "canonical plans differ from locked context sidecar"
    );

    let mut execution = compiled.clone();
    execution.execution.target.id = "tampered-target".into();
    assert_eq!(
        execution.verify().unwrap_err(),
        "canonical plans differ from locked context sidecar"
    );

    let mut region = compiled.clone();
    region.region_program.steps[0].kernel.id = "tampered-kernel".into();
    assert_eq!(
        region.verify().unwrap_err(),
        "role plans do not cover roles or bind region program digest"
    );

    let mut lock = compiled;
    lock.lock.components.clear();
    assert_eq!(
        lock.verify().unwrap_err(),
        "plan lock component artifacts differ from execution plan"
    );
}

#[test]
fn rewritten_plan_hashes_cannot_hide_sidecar_or_region_tampering() {
    let compiled = compile(&request(1, 3, 2)).unwrap();

    let mut evidence = compiled.clone();
    evidence.assurance_results[0].outcome = AssuranceOutcome::RefutedInScope;
    let reference = evidence_reference(&evidence.assurance_results[0]);
    evidence.candidate_evidence[0].assurance_results[0] = reference.clone();
    evidence.execution.evidence_references[0] = reference.clone();
    evidence.lock.evidence_references[0] = reference;
    evidence.lock.execution_plan_digest = pllm_types::execution_plan_digest(&evidence.execution);
    assert_eq!(
        evidence.verify().unwrap_err(),
        "selected candidate no longer satisfies privacy contract"
    );

    let mut missing = compiled.clone();
    missing.assurance_results.clear();
    assert_eq!(
        missing.verify().unwrap_err(),
        "assurance result records do not match execution references"
    );

    let mut artifact = compiled.clone();
    artifact.region_program.steps[0].method.artifact_digest = digest('d');
    artifact.execution.components[1].artifact_digest = digest('d');
    artifact.lock.components[1].digest = digest('d');
    artifact.execution.role_plans[1].digest =
        pllm_compiler::region_program_digest(&artifact.region_program);
    artifact.lock.execution_plan_digest = pllm_types::execution_plan_digest(&artifact.execution);
    assert_eq!(
        artifact.verify().unwrap_err(),
        "candidate evidence sidecar does not match selected region candidates"
    );

    let mut contract = compiled.clone();
    contract.privacy_contract.online_parties = 2;
    let contract_digest = privacy_contract_digest(&contract.privacy_contract);
    contract.locked_context.privacy_contract.digest = contract_digest.clone();
    contract.logical.privacy_contract.digest = contract_digest.clone();
    contract.lock.privacy_contract_digest = contract_digest;
    let logical_digest = pllm_types::logical_plan_digest(&contract.logical);
    contract.execution.logical_plan_digest = logical_digest.clone();
    contract.region_program.logical_plan_digest = logical_digest.clone();
    contract.lock.logical_plan_digest = logical_digest;
    let region_digest = pllm_compiler::region_program_digest(&contract.region_program);
    contract.execution.role_plans[1].digest = region_digest;
    contract.lock.execution_plan_digest = pllm_types::execution_plan_digest(&contract.execution);
    assert_eq!(
        contract.verify().unwrap_err(),
        "selected candidate no longer satisfies privacy contract"
    );

    let mut region = compiled;
    region.region_program.output.shape[1] = 4;
    region.region_program.steps[0].output.shape[1] = 4;
    region.execution.role_plans[1].digest =
        pllm_compiler::region_program_digest(&region.region_program);
    region.lock.execution_plan_digest = pllm_types::execution_plan_digest(&region.execution);
    assert_eq!(
        region.verify().unwrap_err(),
        "region intent does not match locked graph digests"
    );
}

fn oracle(weights: &[u8], input: &[u32], rows: usize, cols: usize, batch: usize) -> Vec<u32> {
    (0..batch)
        .flat_map(|batch_index| {
            (0..rows).map(move |row| {
                (0..cols).fold(0u32, |sum, column| {
                    sum.wrapping_add(
                        (weights[row * cols + column] as i8 as i32 as u32)
                            .wrapping_mul(input[batch_index * cols + column]),
                    )
                })
            })
        })
        .collect()
}

#[test]
fn verified_region_executes_real_pllm_core_matrix() {
    let (batch, columns, rows) = (3, 17, 5);
    let weights: Vec<u8> = (0..rows * columns)
        .map(|index| ((index * 29 + 131) % 256) as u8)
        .collect();
    let input: Vec<u32> = (0..batch * columns)
        .map(|index| (index as u32).wrapping_mul(0x9e37_79b9).wrapping_sub(7))
        .collect();
    let compiled = compile(&request(batch as u64, columns as u64, rows as u64)).unwrap();
    let expected = oracle(&weights, &input, rows, columns, batch);
    for (threads, simd) in [(1, false), (1, true), (2, true)] {
        assert_eq!(
            execute_wrap32(&compiled, &weights, &input, threads, simd).unwrap(),
            expected
        );
    }

    let mut tampered = compiled;
    tampered.region_program.output.shape[1] += 1;
    assert!(execute_wrap32(&tampered, &weights, &input, 1, false).is_err());
}

#[test]
fn executes_locked_centered_q14_to_q7_rescaling() {
    let region =
        define_q14_to_q7_rescale_region("layer.0.gate_proj", "layer.0.silu", 0, vec![1, 7])
            .unwrap();
    assert_eq!(region.input.numeric, NumericType::Wrap32);
    assert_eq!(region.output.numeric, NumericType::SignedFixedQ7);
    assert_eq!(region.input_fractional_bits, 14);
    assert_eq!(region.output_fractional_bits, 7);
    assert_eq!(region.divisor, 128);

    let input = [-16_384, -192, -64, 0, 64, 192, 16_384].map(wrap32);
    assert_eq!(
        execute_q14_to_q7_rescale(&region, &input).unwrap(),
        vec![-128, -2, 0, 0, 0, 2, 128]
    );
}

#[test]
fn q14_to_q7_rescaling_rejects_range_shape_and_contract_mutation() {
    let region =
        define_q14_to_q7_rescale_region("layer.0.gate_proj", "layer.0.silu", 0, vec![1, 1])
            .unwrap();
    assert!(execute_q14_to_q7_rescale(&region, &[wrap32(16_385)]).is_err());
    assert!(execute_q14_to_q7_rescale(&region, &[]).is_err());

    let original_digest = q14_to_q7_rescale_region_digest(&region);
    let mut mutated = region;
    mutated.divisor = 64;
    assert_ne!(q14_to_q7_rescale_region_digest(&mutated), original_digest);
    assert!(execute_q14_to_q7_rescale(&mutated, &[0]).is_err());

    mutated.divisor = 128;
    mutated.source_operation_id = "layer.0.up_proj".into();
    assert_ne!(q14_to_q7_rescale_region_digest(&mutated), original_digest);
    assert!(execute_q14_to_q7_rescale(&mutated, &[0]).is_err());
}

#[test]
fn q14_to_q7_rescaling_rejects_invalid_identity_and_shape() {
    assert!(define_q14_to_q7_rescale_region("bad id", "valid.id", 0, vec![1]).is_err());
    assert!(define_q14_to_q7_rescale_region("valid.id", "target.id", 0, vec![]).is_err());
    assert!(define_q14_to_q7_rescale_region("valid.id", "target.id", 0, vec![1, 0]).is_err());
}
