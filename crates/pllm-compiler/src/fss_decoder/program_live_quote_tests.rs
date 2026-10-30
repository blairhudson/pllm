//! Arithmetic and compiler-only regressions; no private execution or RNG.
use super::*;
use crate::fss_decoder::resource_quote::Quote;

#[test]
fn geometry_live_quote_preserves_existing_tariff() {
    let backing = std::mem::size_of::<Option<zeroize::Zeroizing<Vec<u64>>>>();
    // Enumerated prefix owners: padded 4 + outputs 3 = 56 bytes. Gather floor
    // is 2,097,152; wavefront is 64 + 524,288 for vocabulary 7.
    for capacity in [Capacity::Capsule, Capacity::Decoder] {
        assert_eq!(Quote::live_array_reservation(448, 4, 3, 7, 4, 33,
            capacity, false).unwrap(), 448 + 56 + 2_097_152 + 33 * backing);
        assert_eq!(Quote::live_array_reservation(448, 4, 3, 7, 4, 33,
            capacity, true).unwrap(), 448 + 56 + 2_097_152 + 33 * backing + 524_352);
    }
    // Decoder batch dominates: 4 * 262144 * 8 + 131072 = 8,519,680.
    assert_eq!(Quote::live_array_reservation(448, 4, 3, 262_144, 4, 33,
        Capacity::Decoder, false).unwrap(), 448 + 56 + 8_519_680 + 33 * backing);
    // Larger shared geometry is arithmetic only, never Program permission.
    for (input, outputs, padded, prefix_bytes) in [
        (31, 3, 33, 288), (33, 1, 33, 272), (71, 1, 71, 576),
    ] {
        let geometry = DecoderPlan::validated_phase_geometry(input, outputs).unwrap();
        assert_eq!(geometry.padded_rows(), padded);
        assert_eq!(Quote::live_array_reservation(448, geometry.padded_rows(), outputs,
            7, 4, 33, Capacity::Capsule, false).unwrap(),
            448 + prefix_bytes + 2_097_152 + 33 * backing);
    }
}

#[test]
fn geometry_live_quote_overflow_refuses() {
    let max = usize::MAX;
    // Peak sum; prefix sum/product; batch product/bytes/reader addition;
    // node backing; tournament increment/product. No allocations involved.
    for (peak, sequence, selections, vocabulary, batch, nodes, wave) in [
        (max, 4, 3, 7, 4, 33, false),
        (0, max, 1, 7, 4, 33, false),
        (0, max / 8, 1, 7, 4, 33, false),
        (0, 4, 3, max, 2, 33, false),
        (0, 4, 3, max / 8 + 1, 1, 33, false),
        (0, 4, 3, max / 8, 1, 33, false),
        (0, 4, 3, 7, 4, max, false),
    ] {
        assert!(matches!(Quote::live_array_reservation(peak, sequence, selections,
            vocabulary, batch, nodes, Capacity::Decoder, wave), Err(Error::Resource)));
    }
    for vocabulary in [max, max / 8] {
        assert!(matches!(Quote::live_array_reservation(0, 4, 3, vocabulary, 1, 0,
            Capacity::Capsule, true), Err(Error::Resource)));
    }
    assert!(matches!(Quote::live_array_reservation(max - 2_097_208, 4, 3, 7, 1, 0,
        Capacity::Capsule, true), Err(Error::Resource)));
    assert!(matches!(Quote::compiler_reservation(max / 3 + 1, 0), Err(Error::Resource)));
    assert!(matches!(Quote::compiler_reservation(0, max / 2 + 1), Err(Error::Resource)));
    assert!(matches!(Quote::compiler_reservation(max / 3, 1), Err(Error::Resource)));
}

fn padded_fixture() -> (DecoderPlan, Source) {
    let config = serde_json::json!({"model_type":"gpt2","n_embd":2,"n_head":1,
        "n_layer":1,"n_positions":4,"vocab_size":3,"n_inner":2});
    let plan = pllm_models::lower_model_json(&serde_json::to_vec(&config).unwrap(),
        pllm_models::DecoderWorkload { batch: 1, max_input_tokens: 2, max_new_tokens: 3 }).unwrap();
    let mut tensors: Vec<Tensor> = Vec::new();
    for op in &plan.prefill.operations {
        let Some(id) = op.attributes["weight"].as_str() else { continue; };
        if tensors.iter().any(|t| t.id() == id) { continue; }
        match op.operator {
            Op::TokenLookup => {
                let rows = if op.inputs[0] == "input.tokens" { 3 } else { 4 };
                tensors.push(Tensor::from_q12(id, &[rows, 2], &vec![0; rows * 2]).unwrap());
            }
            Op::LayerNorm => {
                tensors.push(Tensor::from_q12(id, &[2], &[4096; 2]).unwrap());
                tensors.push(Tensor::from_q12(op.attributes["bias"].as_str().unwrap(),
                    &[2], &[0; 2]).unwrap());
            }
            Op::Linear => {
                let columns = op.output_shape[2] as usize;
                tensors.push(Tensor::from_q12(id, &[2, columns], &vec![0; 2 * columns]).unwrap());
                tensors.push(Tensor::from_q12(op.attributes["bias"].as_str().unwrap(),
                    &[columns], &vec![0; columns]).unwrap());
            }
            _ => panic!("unexpected untied source"),
        }
    }
    (plan, Source::new(tensors).unwrap())
}

#[test]
fn padded_walk_live_quote_matches_program_resources() {
    let (plan, source) = padded_fixture();
    let program = Program::compile(&plan, source, 3, Limits::default()).unwrap();
    assert_eq!((program.initial, program.selections, program.sequence), (2, 3, 4));
    assert_eq!(program.nodes.len(), 33);
    // Independent last-use census at peak (k_heads production): embedding_sum,
    // qkv_linear, q_heads, k_slice, k_heads = 8+24+8+8+8 words = 448 bytes.
    // This is the fixed padded walk, not growing active prefixes 2,3,4, and not
    // sum of all node arrays. Public owners add 56 + gather + 33 array handles.
    let expected = 448 + 56 + 2_097_152
        + 33 * std::mem::size_of::<Option<zeroize::Zeroizing<Vec<u64>>>>();
    assert_eq!(program.resources.live_array_bytes_per_party, expected);
    assert_eq!(program.resources.combined_reservation_bytes,
        3 * program.resources.metadata_bytes + 2 * expected + program.resources.peak_native_bytes);
    for initial in [31, 33, 71] {
        let (mut plan, source) = padded_fixture();
        plan.prefill.query_sequence = initial;
        assert!(matches!(Program::compile(&plan, source, 1, Limits::default()),
            Err(Error::Resource)));
    }
    let (plan, source) = padded_fixture();
    assert!(matches!(Program::compile(&plan, source, 5, Limits::default()),
        Err(Error::Unsupported)));
}
