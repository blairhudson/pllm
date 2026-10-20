use super::*;
use pllm_types::canonical_bytes;
use serde_json::json;

fn builder(rotary: bool) -> SourceBuilder {
    let config = if rotary {
        json!({"model_type":"qwen2","hidden_size":2,"intermediate_size":2,
            "num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,
            "vocab_size":3,"max_position_embeddings":4,"hidden_act":"silu",
            "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true})
    } else {
        json!({"model_type":"gpt2","n_embd":2,"n_head":1,"n_layer":1,
            "n_positions":4,"vocab_size":3,"n_inner":2})
    };
    SourceBuilder::new(
        &canonical_bytes(&config),
        1,
        2,
        if rotary {
            sdk::DENSE_ROPE_NUMERIC
        } else {
            sdk::GPT2_NUMERIC
        },
    )
    .unwrap()
}

fn fill(builder: &mut SourceBuilder, dtype: WeightDtype, chunk: usize) {
    let shapes = builder.admitted().unwrap().source_shapes.clone();
    for (name, shape) in shapes {
        let n = elements(&shape).unwrap();
        let gamma = name.ends_with("ln_1.weight")
            || name.ends_with("ln_2.weight")
            || name.ends_with("ln_f.weight")
            || name.ends_with("layernorm.weight")
            || name.ends_with("model.norm.weight");
        let bytes = match dtype {
            WeightDtype::F32 => (if gamma { 1f32 } else { 0f32 }).to_le_bytes().to_vec(),
            WeightDtype::F16 => (if gamma { 0x3c00u16 } else { 0 }).to_le_bytes().to_vec(),
            WeightDtype::Bf16 => (if gamma { 0x3f80u16 } else { 0 }).to_le_bytes().to_vec(),
        };
        for offset in (0..n).step_by(chunk) {
            builder
                .append(&name, dtype, offset, &bytes.repeat(chunk.min(n - offset)))
                .unwrap();
        }
    }
}

fn rotary(builder: &mut SourceBuilder, chunk: usize) {
    let shapes = builder.admitted().unwrap().rotary_shapes.clone();
    for (name, shape) in shapes {
        let n = elements(&shape).unwrap();
        for offset in (0..n).step_by(chunk) {
            let count = chunk.min(n - offset);
            builder
                .append_rotary(
                    &name,
                    offset,
                    &1f32.to_le_bytes().repeat(count),
                    &0f32.to_le_bytes().repeat(count),
                )
                .unwrap();
        }
    }
}

#[test]
fn source_stream_chunks_preserve_original_bytes_and_program_identity() {
    for is_rotary in [false, true] {
        let mut first = builder(is_rotary);
        let mut second = builder(is_rotary);
        fill(&mut first, WeightDtype::F32, 32768);
        fill(&mut second, WeightDtype::F32, 1);
        rotary(&mut first, 32768);
        rotary(&mut second, 1);
        let a = first.seal().unwrap();
        let b = second.seal().unwrap();
        assert_eq!(a.input_digest, b.input_digest);
        assert_eq!(a.program.digest(), b.program.digest());
        assert_eq!(a.program.source_digest(), b.program.source_digest());
        let cap = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
        assert!(a.ownership.phases.serialization_peak_bound <= cap);
        assert!(a.ownership.phases.ingestion_peak_bound <= cap);
        assert!(a.ownership.phases.compile_prepare_peak_bound <= cap);
        assert!(a.ownership.schedule_metadata_credit_floor <= a.program.resources().metadata_bytes);
        assert!(
            a.ownership.schedule_retained_owner_bound
                <= 3 * a.ownership.schedule_metadata_credit_floor
        );
        assert!(a.program.resources().combined_reservation_bytes <= cap);
    }
}

#[test]
fn external_dtype_identity_remains_separate_from_equal_quantized_source() {
    let mut outputs = Vec::new();
    for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::Bf16] {
        let mut pending = builder(false);
        fill(&mut pending, dtype, 2);
        outputs.push(pending.seal().unwrap());
    }
    for pair in outputs.windows(2) {
        assert_eq!(pair[0].program.digest(), pair[1].program.digest());
        assert_ne!(pair[0].input_digest, pair[1].input_digest);
    }
}

#[test]
fn malformed_offsets_dtype_names_frames_and_values_burn_pending_source() {
    for fault in 0..7 {
        let mut pending = builder(false);
        let (name, shape) = pending
            .admitted()
            .unwrap()
            .source_shapes
            .first_key_value()
            .map(|(n, s)| (n.clone(), s.clone()))
            .unwrap();
        let n = elements(&shape).unwrap();
        let result = match fault {
            0 => pending.append("wrong", WeightDtype::F32, 0, &[0; 4]),
            1 => pending.append(&name, WeightDtype::F32, 1, &[0; 4]),
            2 => pending.append(&name, WeightDtype::F32, 0, &[0; 3]),
            3 => pending.append(&name, WeightDtype::F32, 0, &[]),
            4 => pending.append(&name, WeightDtype::F32, 0, &vec![0; MAX_FRAME_BYTES + 4]),
            5 => pending.append(
                &name,
                WeightDtype::F32,
                0,
                &f32::NAN.to_le_bytes().repeat(n),
            ),
            _ => {
                assert!(n > 1);
                pending.append(&name, WeightDtype::F32, 0, &[0; 4]).unwrap();
                pending.append(&name, WeightDtype::F16, 1, &[0; 2])
            }
        };
        assert!(result.is_err(), "fault {fault}");
        assert!(matches!(pending.admitted(), Err(Error::Consumed)));
        assert_eq!(
            pending.append(&name, WeightDtype::F32, 0, &[0; 4]),
            Err(Error::Consumed)
        );
        assert!(matches!(pending.seal(), Err(Error::Consumed)));
    }
}

#[test]
fn partial_cancelled_and_malformed_rotary_owners_cannot_seal() {
    assert!(matches!(builder(false).seal(), Err(Error::Context)));
    let mut cancelled = builder(false);
    cancelled.cancel();
    cancelled.cancel();
    assert!(matches!(cancelled.seal(), Err(Error::Consumed)));
    let mut pending = builder(true);
    fill(&mut pending, WeightDtype::F32, 32768);
    let (name, shape) = pending
        .admitted()
        .unwrap()
        .rotary_shapes
        .first_key_value()
        .map(|(n, s)| (n.clone(), s.clone()))
        .unwrap();
    let n = elements(&shape).unwrap();
    assert_eq!(
        pending.append_rotary(
            &name,
            0,
            &0f32.to_le_bytes().repeat(n),
            &0f32.to_le_bytes().repeat(n)
        ),
        Err(Error::Domain)
    );
    assert!(matches!(pending.admitted(), Err(Error::Consumed)));
}

#[test]
fn source_geometry_rejection_precedes_payload_acceptance() {
    let config = json!({"model_type":"gpt2","n_embd":768,"n_head":12,"n_layer":12,
        "n_positions":1024,"vocab_size":50257});
    assert!(SourceBuilder::new(&canonical_bytes(&config), 1, 1, sdk::GPT2_NUMERIC).is_err());
    assert!(SourceBuilder::new(b"{}", 1, 1, sdk::GPT2_NUMERIC).is_err());
}

#[test]
fn invalid_first_partial_ieee_word_burns_before_tensor_completion() {
    for (dtype, payload) in [
        (WeightDtype::F32, f32::NAN.to_le_bytes().to_vec()),
        (WeightDtype::F32, (2f32.powi(37)).to_le_bytes().to_vec()),
        (WeightDtype::F32, f32::INFINITY.to_le_bytes().to_vec()),
        (WeightDtype::F16, 0x7e00u16.to_le_bytes().to_vec()),
        (WeightDtype::Bf16, 0x7fc0u16.to_le_bytes().to_vec()),
    ] {
        let mut pending = builder(false);
        let (name, shape) = pending
            .admitted()
            .unwrap()
            .source_shapes
            .first_key_value()
            .map(|(n, s)| (n.clone(), s.clone()))
            .unwrap();
        assert!(elements(&shape).unwrap() > 1);
        assert_eq!(
            pending.append(&name, dtype, 0, &payload),
            Err(Error::Domain)
        );
        assert!(matches!(pending.admitted(), Err(Error::Consumed)));
    }
}

#[test]
fn early_word_domain_check_agrees_with_authoritative_converter_boundaries() {
    for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::Bf16] {
        let values: &[u32] = match dtype {
            WeightDtype::F32 => &[
                0, 0x80000000, 1, 0x3f800000, 0x51ffffff, 0x52000000, 0xd2000000, 0xd2000001,
                0x7f800000, 0x7fc00000,
            ],
            WeightDtype::F16 => &[0, 0x8000, 1, 0x3c00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00],
            WeightDtype::Bf16 => &[
                0, 0x8000, 1, 0x3f80, 0x51ff, 0x5200, 0xd200, 0xd201, 0x7f80, 0x7fc0,
            ],
        };
        for &bits in values {
            assert_eq!(
                validate_word(dtype, bits).is_ok(),
                Tensor::from_ieee("word", &[1], dtype, &[bits]).is_ok(),
                "{dtype:?} {bits:x}"
            );
        }
    }
}

#[test]
fn aggregate_compile_owner_admission_rejects_before_numeric_schedule() {
    let pending = builder(false);
    let admitted = pending.admitted().unwrap();
    let source = Source::new(vec![Tensor::from_ieee(
        "fixture",
        &[1],
        WeightDtype::F32,
        &[0],
    )
    .unwrap()])
    .unwrap();
    let mut phase = admitted.phase_admission;
    phase.compile_external_bytes = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES;
    assert!(matches!(
        compile_ownership(&admitted.plan, &source, phase),
        Err(Error::Resource)
    ));
    phase = admitted.phase_admission;
    phase.compile_prepare_peak_bound = pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES + 1;
    assert!(matches!(
        compile_ownership(&admitted.plan, &source, phase),
        Err(Error::Resource)
    ));
}

#[test]
fn invalid_partial_rotary_values_and_first_row_burn_immediately() {
    for value in [f32::NAN, f32::INFINITY, 1.5, 0.0] {
        let mut pending = builder(true);
        fill(&mut pending, WeightDtype::F32, 32768);
        let name = pending
            .admitted()
            .unwrap()
            .rotary_shapes
            .first_key_value()
            .unwrap()
            .0
            .clone();
        assert_eq!(
            pending.append_rotary(&name, 0, &value.to_le_bytes(), &0f32.to_le_bytes()),
            Err(Error::Domain)
        );
        assert!(matches!(pending.admitted(), Err(Error::Consumed)));
    }
}
