use super::*;
use crate::dense_qwen_mlp::{
    compile_dense_qwen_mlp_block, DenseQwenMlpResourcePolicy, DenseQwenMlpWeightBytes,
    DenseQwenMlpWeightManifest, DenseQwenMlpWeights,
};
use crate::gated_tensor::{
    gated_tensor_registry_snapshot, saturate_gated_tensor_registry_for_test,
};
use crate::rms_norm_stream_protected::rms_registry_snapshot;
use pllm_models::{lower_model_json, DecoderMode, DecoderWorkload};
use std::io::Write;
use std::sync::Mutex;

static REGISTRY_TEST_LOCK: Mutex<()> = Mutex::new(());

struct FailSecondFlush {
    flushes: usize,
}

impl Write for FailSecondFlush {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flushes += 1;
        if self.flushes == 2 {
            Err(std::io::Error::other("injected publication failure"))
        } else {
            Ok(())
        }
    }
}

#[derive(Default)]
struct CountingWriter {
    callbacks: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.callbacks += 1;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.callbacks += 1;
        Ok(())
    }
}

fn plan() -> DecoderPlan {
    lower_model_json(
        br#"{
            "model_type":"qwen2","hidden_size":2,"intermediate_size":1,
            "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":8,"max_position_embeddings":8,"hidden_act":"silu",
            "rms_norm_eps":1e-6,"rope_theta":10000.0,"tie_word_embeddings":true
        }"#,
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 2,
            max_new_tokens: 1,
        },
    )
    .unwrap()
}

fn weight<'a>(weight_id: &'a str, bytes: &'a [u8]) -> DenseQwenMlpWeightBytes<'a> {
    DenseQwenMlpWeightBytes { weight_id, bytes }
}

fn block(plan: &DecoderPlan) -> CompiledDenseQwenMlpBlock {
    let norm = [1024_i16, 1024_i16]
        .into_iter()
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    let gate = [4_i8, 4_i8].map(|value| value as u8);
    let up = [3_i8, 3_i8].map(|value| value as u8);
    let down = [2_i8, 2_i8].map(|value| value as u8);
    let weights = DenseQwenMlpWeights {
        norm_q10: weight("model.layers.0.post_attention_layernorm.weight", &norm),
        gate_q4: weight("model.layers.0.mlp.gate_proj.weight", &gate),
        up_q4: weight("model.layers.0.mlp.up_proj.weight", &up),
        down_q3: weight("model.layers.0.mlp.down_proj.weight", &down),
    };
    compile_dense_qwen_mlp_block(
        plan,
        DecoderMode::Prefill,
        0,
        DenseQwenMlpWeightManifest::from_weights(weights).unwrap(),
        DenseQwenMlpResourcePolicy {
            max_total_weight_bytes: 64,
            max_activation_elements: 8,
        },
        weights,
    )
    .unwrap()
}

fn rms_policy() -> RmsNormQ10StreamResourcePolicy {
    RmsNormQ10StreamResourcePolicy::acknowledge_unreviewed_public_weights(64 * 1024 * 1024).unwrap()
}

fn tensor_policy() -> TensorResourcePolicy {
    TensorResourcePolicy {
        max_elements: 2,
        max_chunk_elements: 2,
        max_chunk_bytes: 600_000,
        max_body_bytes: 4_000_000,
        max_output_bytes: 2_048,
        max_client_material_bytes: 2_000_000,
        max_working_bytes: 40_000_000,
    }
}

fn aggregate_policy() -> DenseQwenMlpProtectedResourcePolicy {
    DenseQwenMlpProtectedResourcePolicy {
        max_rows: 2,
        max_total_body_bytes: 512 * 1024 * 1024,
    }
}

fn prepare_with_writer<W: Write>(
    plan: &DecoderPlan,
    block: &CompiledDenseQwenMlpBlock,
    writer: &mut W,
) -> Result<BoundDenseQwenMlpProtectedMaterial, String> {
    prepare_bound_dense_qwen_mlp_protected_nonlinear(
        plan,
        block,
        &ExperimentalDenseQwenMlpProtectedNonlinearApproval::acknowledge_unreviewed_experimental_components(),
        &rms_policy(),
        &tensor_policy(),
        &aggregate_policy(),
        writer,
    )
}

#[test]
fn aggregate_writer_failure_restores_exact_registry_baseline() {
    let _serial = REGISTRY_TEST_LOCK.lock().unwrap();
    let _rms_serial = crate::rms_norm_stream_protected::RMS_REGISTRY_TEST_LOCK
        .lock()
        .unwrap();
    let plan = plan();
    let block = block(&plan);
    let rms_before = rms_registry_snapshot().unwrap();
    let gated_before = gated_tensor_registry_snapshot().unwrap();
    let mut writer = FailSecondFlush { flushes: 0 };

    assert!(prepare_with_writer(&plan, &block, &mut writer).is_err());
    assert!(writer.flushes >= 2);
    assert_eq!(rms_registry_snapshot().unwrap(), rms_before);
    assert_eq!(gated_tensor_registry_snapshot().unwrap(), gated_before);
}

#[test]
fn aggregate_gated_registry_admission_failure_precedes_writer() {
    let _serial = REGISTRY_TEST_LOCK.lock().unwrap();
    let _rms_serial = crate::rms_norm_stream_protected::RMS_REGISTRY_TEST_LOCK
        .lock()
        .unwrap();
    let plan = plan();
    let block = block(&plan);
    let rms_before = rms_registry_snapshot().unwrap();
    let gated_before = gated_tensor_registry_snapshot().unwrap();
    let saturation = saturate_gated_tensor_registry_for_test().unwrap();
    let mut writer = CountingWriter::default();

    assert!(prepare_with_writer(&plan, &block, &mut writer).is_err());
    assert_eq!(writer.callbacks, 0);
    assert_eq!(rms_registry_snapshot().unwrap(), rms_before);
    drop(saturation);
    assert_eq!(gated_tensor_registry_snapshot().unwrap(), gated_before);
}
