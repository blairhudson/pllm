use pllm_compiler::{
    compile_dense_qwen_mlp_block, execute_bound_dense_qwen_mlp_protected_nonlinear,
    execute_dense_qwen_mlp_block, prepare_bound_dense_qwen_mlp_protected_nonlinear,
    BoundDenseQwenMlpProtectedMaterial, CompiledDenseQwenMlpBlock,
    DenseQwenMlpProtectedNonlinearExecution, DenseQwenMlpProtectedResourcePolicy,
    DenseQwenMlpResourcePolicy, DenseQwenMlpWeightBytes, DenseQwenMlpWeightManifest,
    DenseQwenMlpWeights, ExperimentalDenseQwenMlpProtectedNonlinearApproval,
    ExperimentalRmsNormQ10StreamPolicy, TensorResourcePolicy,
};
use pllm_models::{lower_model_json, DecoderMode, DecoderPlan, DecoderWorkload};
use pllm_types::digest_bytes;
use std::io::{Cursor, Read, Write};

struct TestWeights {
    norm: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
}

impl TestWeights {
    fn new(intermediate: usize) -> Self {
        let gate = [4_i8, -3, 2]
            .into_iter()
            .cycle()
            .take(intermediate * 2)
            .map(|value| value.to_ne_bytes()[0])
            .collect();
        let up = [3_i8, 2, -2]
            .into_iter()
            .cycle()
            .take(intermediate * 2)
            .map(|value| value.to_ne_bytes()[0])
            .collect();
        let down = [2_i8, -1, 3]
            .into_iter()
            .cycle()
            .take(intermediate * 2)
            .map(|value| value.to_ne_bytes()[0])
            .collect();
        Self {
            norm: [1024_i16, 1024]
                .into_iter()
                .flat_map(i16::to_le_bytes)
                .collect(),
            gate,
            up,
            down,
        }
    }

    fn input(&self) -> DenseQwenMlpWeights<'_> {
        DenseQwenMlpWeights {
            norm_q10: weight("model.layers.0.post_attention_layernorm.weight", &self.norm),
            gate_q4: weight("model.layers.0.mlp.gate_proj.weight", &self.gate),
            up_q4: weight("model.layers.0.mlp.up_proj.weight", &self.up),
            down_q3: weight("model.layers.0.mlp.down_proj.weight", &self.down),
        }
    }
}

fn weight<'a>(weight_id: &'a str, bytes: &'a [u8]) -> DenseQwenMlpWeightBytes<'a> {
    DenseQwenMlpWeightBytes { weight_id, bytes }
}

fn plan(intermediate: usize, tokens: u64) -> DecoderPlan {
    lower_model_json(
        format!(
            r#"{{
                "model_type":"qwen2","hidden_size":2,"intermediate_size":{intermediate},
                "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
                "vocab_size":8,"max_position_embeddings":128,"hidden_act":"silu",
                "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
            }}"#
        )
        .as_bytes(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: tokens,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

fn compile(
    plan: &DecoderPlan,
    mode: DecoderMode,
    weights: &TestWeights,
) -> CompiledDenseQwenMlpBlock {
    let input = weights.input();
    compile_dense_qwen_mlp_block(
        plan,
        mode,
        0,
        DenseQwenMlpWeightManifest::from_weights(input).unwrap(),
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: 1024,
            max_activation_elements: 256,
        },
        input,
    )
    .unwrap()
}

fn rms_policy() -> ExperimentalRmsNormQ10StreamPolicy {
    ExperimentalRmsNormQ10StreamPolicy::acknowledge_unreviewed_public_weights(64 * 1024 * 1024)
        .unwrap()
}

fn tensor_policy(elements: u64) -> TensorResourcePolicy {
    TensorResourcePolicy {
        max_elements: elements,
        max_chunk_elements: 2,
        max_chunk_bytes: 600_000,
        max_body_bytes: 4_000_000,
        max_output_bytes: elements * 1024,
        max_client_material_bytes: 2_000_000,
        max_working_bytes: 40_000_000,
    }
}

fn aggregate_policy() -> DenseQwenMlpProtectedResourcePolicy {
    DenseQwenMlpProtectedResourcePolicy {
        max_rows: 64,
        max_total_body_bytes: 512 * 1024 * 1024,
    }
}

fn approval() -> ExperimentalDenseQwenMlpProtectedNonlinearApproval {
    ExperimentalDenseQwenMlpProtectedNonlinearApproval::acknowledge_unreviewed_experimental_components()
}

fn wrap(values: &[i16]) -> Vec<u32> {
    values
        .iter()
        .map(|value| i32::from(*value) as u32)
        .collect()
}

fn prepare(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    elements: u64,
) -> (Vec<u8>, BoundDenseQwenMlpProtectedMaterial) {
    let mut body = Vec::new();
    let material = prepare_bound_dense_qwen_mlp_protected_nonlinear(
        plan,
        block,
        &approval(),
        &rms_policy(),
        &tensor_policy(elements),
        &aggregate_policy(),
        &mut body,
    )
    .unwrap();
    assert_eq!(material.binding().total_body_bytes(), body.len() as u64);
    (body, material)
}

fn execute(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    material: &mut BoundDenseQwenMlpProtectedMaterial,
    body: Vec<u8>,
    input: &[i16],
    elements: u64,
) -> Result<Vec<i16>, String> {
    let body_bytes = material.binding().total_body_bytes();
    execute_bound_dense_qwen_mlp_protected_nonlinear(
        plan,
        block,
        material,
        &mut Cursor::new(body),
        DenseQwenMlpProtectedNonlinearExecution {
            rms_policy: &rms_policy(),
            tensor_policy: &tensor_policy(elements),
            aggregate_policy: &aggregate_policy(),
            body_bytes,
            input_q10: &wrap(input),
            threads: 1,
            simd: false,
        },
    )
}

#[test]
fn decode_row_matches_clear_exactly() {
    let plan = plan(1, 1);
    let weights = TestWeights::new(1);
    let block = compile(&plan, DecoderMode::Decode, &weights);
    let input = [300_i16, -100];
    let expected = execute_dense_qwen_mlp_block(&plan, &block, &wrap(&input), 1, false).unwrap();
    let (body, mut material) = prepare(&plan, &block, 1);
    assert_eq!(material.binding().row_count(), 1);
    assert_eq!(material.binding().row_order(), "batch_sequence_row_major");
    assert_eq!(
        execute(&plan, &block, &mut material, body, &input, 1).unwrap(),
        expected
    );
}

#[test]
fn prefill_rows_and_more_than_four_intermediates_match_clear() {
    let plan = plan(3, 2);
    let weights = TestWeights::new(3);
    let block = compile(&plan, DecoderMode::Prefill, &weights);
    let input = [300_i16, -500, 200, -400];
    let expected = execute_dense_qwen_mlp_block(&plan, &block, &wrap(&input), 1, false).unwrap();
    let (body, mut material) = prepare(&plan, &block, 6);
    assert_eq!(material.binding().row_count(), 2);
    assert_eq!(
        execute(&plan, &block, &mut material, body, &input, 6).unwrap(),
        expected
    );
}

#[test]
fn corruption_in_each_rms_frame_and_gated_frame_returns_no_output() {
    let plan = plan(1, 2);
    let weights = TestWeights::new(1);
    let block = compile(&plan, DecoderMode::Prefill, &weights);
    let input = [100_i16, -200, 300, -400];
    for frame in 0..3 {
        let (mut body, mut material) = prepare(&plan, &block, 2);
        let first = usize::try_from(material.binding().rms_body_bytes(0).unwrap()).unwrap();
        let second = usize::try_from(material.binding().rms_body_bytes(1).unwrap()).unwrap();
        let offset = match frame {
            0 => 0,
            1 => first,
            _ => first + second,
        };
        body[offset] ^= 1;
        assert!(execute(&plan, &block, &mut material, body, &input, 2).is_err());
    }
}

#[test]
fn truncation_trailing_bytes_and_replay_fail() {
    let plan = plan(1, 1);
    let weights = TestWeights::new(1);
    let block = compile(&plan, DecoderMode::Decode, &weights);
    let input = [100_i16, -100];

    let (mut body, mut material) = prepare(&plan, &block, 1);
    body.pop();
    assert!(execute(&plan, &block, &mut material, body, &input, 1).is_err());

    let (mut body, mut material) = prepare(&plan, &block, 1);
    body.push(0);
    assert!(execute(&plan, &block, &mut material, body, &input, 1).is_err());

    let (body, mut material) = prepare(&plan, &block, 1);
    execute(&plan, &block, &mut material, body, &input, 1).unwrap();
    assert!(execute(&plan, &block, &mut material, Vec::new(), &input, 1).is_err());
}

#[test]
fn wrong_plan_block_and_policy_fail_before_frame_read() {
    let plan = plan(1, 1);
    let weights = TestWeights::new(1);
    let block = compile(&plan, DecoderMode::Decode, &weights);
    let (body, mut material) = prepare(&plan, &block, 1);
    let input = [100_i16, -100];

    let mut altered_plan = plan.clone();
    altered_plan.config_digest = digest_bytes("test.altered.plan.v1", b"altered");
    let body_bytes = material.binding().total_body_bytes();
    assert!(execute_bound_dense_qwen_mlp_protected_nonlinear(
        &altered_plan,
        &block,
        &mut material,
        &mut PanicReader,
        DenseQwenMlpProtectedNonlinearExecution {
            rms_policy: &rms_policy(),
            tensor_policy: &tensor_policy(1),
            aggregate_policy: &aggregate_policy(),
            body_bytes,
            input_q10: &wrap(&input),
            threads: 1,
            simd: false,
        },
    )
    .is_err());

    let other_weights = TestWeights::new(1);
    let mut other_input = other_weights.input();
    let changed_gate = [5_u8, 5];
    other_input.gate_q4 = weight("model.layers.0.mlp.gate_proj.weight", &changed_gate);
    let other_block = compile_dense_qwen_mlp_block(
        &plan,
        DecoderMode::Decode,
        0,
        DenseQwenMlpWeightManifest::from_weights(other_input).unwrap(),
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: 1024,
            max_activation_elements: 32,
        },
        other_input,
    )
    .unwrap();
    assert!(execute(&plan, &other_block, &mut material, body.clone(), &input, 1).is_err());

    let mut wrong_policy = tensor_policy(1);
    wrong_policy.max_output_bytes += 1;
    let body_bytes = material.binding().total_body_bytes();
    assert!(execute_bound_dense_qwen_mlp_protected_nonlinear(
        &plan,
        &block,
        &mut material,
        &mut Cursor::new(body.clone()),
        DenseQwenMlpProtectedNonlinearExecution {
            rms_policy: &rms_policy(),
            tensor_policy: &wrong_policy,
            aggregate_policy: &aggregate_policy(),
            body_bytes,
            input_q10: &wrap(&input),
            threads: 1,
            simd: false,
        },
    )
    .is_err());
    let wrong_rms_policy =
        ExperimentalRmsNormQ10StreamPolicy::acknowledge_unreviewed_public_weights(63 * 1024 * 1024)
            .unwrap();
    let body_bytes = material.binding().total_body_bytes();
    assert!(execute_bound_dense_qwen_mlp_protected_nonlinear(
        &plan,
        &block,
        &mut material,
        &mut Cursor::new(body.clone()),
        DenseQwenMlpProtectedNonlinearExecution {
            rms_policy: &wrong_rms_policy,
            tensor_policy: &tensor_policy(1),
            aggregate_policy: &aggregate_policy(),
            body_bytes,
            input_q10: &wrap(&input),
            threads: 1,
            simd: false,
        },
    )
    .is_err());
    let wrong_aggregate_policy = DenseQwenMlpProtectedResourcePolicy {
        max_rows: 63,
        ..aggregate_policy()
    };
    let body_bytes = material.binding().total_body_bytes();
    assert!(execute_bound_dense_qwen_mlp_protected_nonlinear(
        &plan,
        &block,
        &mut material,
        &mut Cursor::new(body.clone()),
        DenseQwenMlpProtectedNonlinearExecution {
            rms_policy: &rms_policy(),
            tensor_policy: &tensor_policy(1),
            aggregate_policy: &wrong_aggregate_policy,
            body_bytes,
            input_q10: &wrap(&input),
            threads: 1,
            simd: false,
        },
    )
    .is_err());
    assert!(execute(&plan, &block, &mut material, body, &input, 1).is_ok());
}

struct PanicReader;

impl Read for PanicReader {
    fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
        panic!("frame reader must not run before aggregate validation")
    }
}

struct FailOnFlush {
    flushes: usize,
    bytes: usize,
}

#[derive(Default)]
struct CountingWriter {
    callbacks: usize,
    bytes: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.callbacks += 1;
        self.bytes += buffer.len();
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.callbacks += 1;
        Ok(())
    }
}

fn assert_preparation_fails_without_writer_callback(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    elements: u64,
    tensor: &TensorResourcePolicy,
    aggregate: &DenseQwenMlpProtectedResourcePolicy,
) {
    let mut writer = CountingWriter::default();
    assert!(prepare_bound_dense_qwen_mlp_protected_nonlinear(
        plan,
        block,
        &approval(),
        &rms_policy(),
        tensor,
        aggregate,
        &mut writer,
    )
    .is_err());
    assert_eq!(writer.callbacks, 0);
    assert_eq!(writer.bytes, 0);
    assert_eq!(tensor.max_elements, elements);
}

#[test]
fn deterministic_aggregate_and_tensor_rejections_precede_writer_access() {
    let plan_65 = plan(1, 65);
    let weights_65 = TestWeights::new(1);
    let block_65 = compile(&plan_65, DecoderMode::Prefill, &weights_65);
    assert_preparation_fails_without_writer_callback(
        &plan_65,
        &block_65,
        65,
        &tensor_policy(65),
        &aggregate_policy(),
    );

    let plan = plan(1, 1);
    let weights = TestWeights::new(1);
    let block = compile(&plan, DecoderMode::Decode, &weights);
    let tiny_aggregate = DenseQwenMlpProtectedResourcePolicy {
        max_rows: 1,
        max_total_body_bytes: 1,
    };
    assert_preparation_fails_without_writer_callback(
        &plan,
        &block,
        1,
        &tensor_policy(1),
        &tiny_aggregate,
    );

    let mut invalid_tensor = tensor_policy(1);
    invalid_tensor.max_chunk_bytes = 1;
    assert_preparation_fails_without_writer_callback(
        &plan,
        &block,
        1,
        &invalid_tensor,
        &aggregate_policy(),
    );

    let (_probe_body, probe_material) = prepare(&plan, &block, 1);
    let exact_gated_body_bytes = probe_material.binding().gated_body_bytes();
    drop(probe_material);
    let mut one_byte_below_exact = tensor_policy(1);
    one_byte_below_exact.max_body_bytes = exact_gated_body_bytes - 1;
    assert_preparation_fails_without_writer_callback(
        &plan,
        &block,
        1,
        &one_byte_below_exact,
        &aggregate_policy(),
    );
}

impl Write for FailOnFlush {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.bytes += buffer.len();
        Ok(buffer.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flushes += 1;
        if self.flushes == 2 {
            Err(std::io::Error::other(
                "intentional aggregate publication failure",
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn writer_failure_after_first_row_releases_aggregate_issuances() {
    let plan = plan(1, 2);
    let weights = TestWeights::new(1);
    let block = compile(&plan, DecoderMode::Prefill, &weights);
    let mut writer = FailOnFlush {
        flushes: 0,
        bytes: 0,
    };
    let error = match prepare_bound_dense_qwen_mlp_protected_nonlinear(
        &plan,
        &block,
        &approval(),
        &rms_policy(),
        &tensor_policy(2),
        &aggregate_policy(),
        &mut writer,
    ) {
        Ok(_) => panic!("writer failure was accepted"),
        Err(error) => error,
    };
    assert!(error.contains("discard sink contents"));
    assert!(writer.bytes > 0);

    // Exact retry plus replay rejection is practical external evidence of release; registry
    // internals intentionally remain private.
    let (body, mut material) = prepare(&plan, &block, 2);
    assert!(execute(&plan, &block, &mut material, body, &[10, 11, 20, 21], 2).is_ok());
    assert!(execute(
        &plan,
        &block,
        &mut material,
        Vec::new(),
        &[10, 11, 20, 21],
        2,
    )
    .is_err());
}
