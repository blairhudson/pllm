//! Numerical regressions for the file-backed decoder execution capacity.
//! All reconstruction is test-only. Generated controls make no model-quality claim.
use super::*;
use crate::fss_decoder::{
    import::{ImportedProgram, SourceBuilder},
    sdk, transport, Capacity,
};
use serde_json::Value;
use std::collections::BTreeMap;

type OpenTrace = Vec<(usize, usize, Vec<i64>)>;

fn execute_open(program: &Program, prefix: &[u64]) -> (Vec<i64>, OpenTrace) {
    let (a, b) = execute_prefix(program, prefix);
    assert_eq!(a.1.len(), program.nodes.len() * program.selections());
    assert_eq!(a.1.len(), b.1.len());
    let trace =
        a.1.iter()
            .zip(&b.1)
            .enumerate()
            .map(|(ordinal, (a, b))| {
                assert_eq!((a.0, a.1), (b.0, b.1));
                assert_eq!(
                    (a.0, a.1),
                    (ordinal / program.nodes.len(), ordinal % program.nodes.len())
                );
                assert_eq!(
                    a.2.len(),
                    program.nodes[a.1].shape.iter().product::<usize>()
                );
                assert_eq!(a.2.len(), b.2.len());
                (a.0, a.1, reconstructed(&a.2, &b.2))
            })
            .collect();
    (reconstructed(a.0.values(), b.0.values()), trace)
}

fn tiny(capacity: Capacity, tied_maxima: bool) -> (DecoderPlan, Source, ImportedProgram) {
    let (config, semantic, original) = dense_fixture();
    let mut builder = SourceBuilder::new_with_capacity(
        &serde_json::to_vec(&config).unwrap(),
        1,
        2,
        sdk::DENSE_ROPE_NUMERIC,
        capacity,
    )
    .unwrap();
    let shapes = builder.admitted().unwrap().source_shapes.clone();
    let mut tensors = Vec::new();
    for (name, shape) in shapes {
        let tensor = original.tensors().iter().find(|t| t.id() == name).unwrap();
        assert_eq!(tensor.shape(), shape);
        let mut codes = tensor.values().unwrap().to_vec();
        if tied_maxima && name == "model.embed_tokens.weight" {
            // Tied head rows 1 and 2 are identical; row 0 is distinct. This tests
            // a nonzero winning tie, not an all-zero head or a constant selector.
            codes[4] = codes[2];
            codes[5] = codes[3];
        }
        for (offset, chunk) in codes.chunks(3).enumerate() {
            let bytes: Vec<_> = chunk
                .iter()
                .flat_map(|&v| (v as f32 / 4096.0).to_le_bytes())
                .collect();
            builder
                .append(&name, WeightDtype::F32, offset * 3, &bytes)
                .unwrap();
        }
        tensors.push(Tensor::from_q12(&name, &shape, &codes).unwrap());
    }
    // Original tiny fixture's exact F32 rotary inputs, not Q12-to-F32 rewrites.
    let cos = [
        1f32,
        1.,
        f32::from_bits(0x3f0a5140),
        f32::from_bits(0x3f0a5140),
    ];
    let sin = [
        0f32,
        0.,
        f32::from_bits(0x3f576aa4),
        f32::from_bits(0x3f576aa4),
    ];
    for (name, shape) in builder.admitted().unwrap().rotary_shapes.clone() {
        assert_eq!(shape, [2, 2]);
        builder
            .append_rotary(
                &name,
                0,
                &cos.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                &sin.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )
            .unwrap();
    }
    let imported = builder.seal().unwrap();
    assert_eq!(imported.program.capacity(), capacity);
    for tensor in imported.program.source.tensors() {
        assert_eq!(
            tensor.is_file_backed(),
            capacity == Capacity::Decoder && tensor.shape().len() == 2
        );
    }
    (semantic, Source::new(tensors).unwrap(), imported)
}

fn node<'a>(semantic: &DecoderPlan, trace: &'a OpenTrace, step: usize, name: &str) -> &'a [i64] {
    let index = semantic
        .prefill
        .operations
        .iter()
        .position(|op| op.id == name)
        .unwrap();
    let entry = &trace[step * semantic.prefill.operations.len() + index];
    assert_eq!((entry.0, entry.1), (step, index));
    &entry.2
}

fn check_dots(semantic: &DecoderPlan, source: &Source, trace: &OpenTrace) {
    let mut negative_floor = false;
    let mut complete_sum_floor = false;
    let mut positive_weight = false;
    let mut negative_weight = false;
    for &(step, index, ref actual) in trace {
        let op = &semantic.prefill.operations[index];
        if !matches!(op.operator, Op::Linear | Op::OutputHead) {
            continue;
        }
        let weight = source
            .tensors()
            .iter()
            .find(|t| Some(t.id()) == op.attributes["weight"].as_str())
            .unwrap();
        let weights = weight.values().unwrap();
        positive_weight |= weights.iter().any(|&v| v > 0);
        negative_weight |= weights.iter().any(|&v| v < 0);
        let bias = source
            .tensors()
            .iter()
            .find(|t| Some(t.id()) == op.attributes["bias"].as_str());
        let input = node(semantic, trace, step, &op.inputs[0]);
        let (columns, inner) = (weight.shape()[0], weight.shape()[1]);
        assert_eq!(actual.len(), input.len() / inner * columns);
        for (row, x) in input.chunks_exact(inner).enumerate() {
            for col in 0..columns {
                let products: Vec<_> = x
                    .iter()
                    .zip(&weights[col * inner..(col + 1) * inner])
                    .map(|(&a, &b)| i128::from(a) * i128::from(b))
                    .collect();
                let raw: i128 = products.iter().sum();
                let floor = raw.div_euclid(4096);
                let bias = bias.map_or(0, |b| b.values().unwrap()[col]);
                assert_eq!(
                    i128::from(actual[row * columns + col]),
                    floor + i128::from(bias),
                    "complete signed dot step={step} op={} row={row} col={col}",
                    op.id
                );
                negative_floor |= raw < 0 && raw.rem_euclid(4096) != 0 && floor != raw / 4096;
                complete_sum_floor |=
                    floor != products.iter().map(|p| p.div_euclid(4096)).sum::<i128>();
            }
        }
    }
    assert!(positive_weight && negative_weight);
    assert!(
        negative_floor,
        "fixture must distinguish negative floor from truncation"
    );
    assert!(
        complete_sum_floor,
        "fixture must distinguish full-dot from per-product floor"
    );
}

#[test]
fn tiny_qwen2_file_decoder_matches_every_capsule_node_and_private_feedback() {
    let (semantic, source, resident) = tiny(Capacity::Capsule, false);
    let (_, _, paged) = tiny(Capacity::Decoder, false);
    assert_eq!(resident.input_digest, paged.input_digest);
    assert_eq!(
        resident.program.source_digest(),
        paged.program.source_digest()
    );
    assert_eq!(resident.program.profile(), paged.program.profile());
    assert_ne!(resident.program.digest(), paged.program.digest());
    let embedding = source
        .tensors()
        .iter()
        .find(|t| t.id() == "model.embed_tokens.weight")
        .unwrap()
        .values()
        .unwrap();
    let mut logits = Vec::new();
    for token in [0, 1] {
        let reference = execute_open(&resident.program, &[token]);
        let actual = execute_open(&paged.program, &[token]);
        assert_eq!(
            actual, reference,
            "all physical rows, nodes and selections; prefix={token}"
        );
        check_dots(&semantic, &source, &actual.1);
        assert_eq!(
            &node(&semantic, &actual.1, 0, "token_lookup")[..2],
            &embedding[token as usize * 2..token as usize * 2 + 2]
        );
        let selected = actual.0[0] as usize;
        assert_eq!(
            &node(&semantic, &actual.1, 1, "token_lookup")[2..],
            &embedding[selected * 2..selected * 2 + 2]
        );
        for step in 0..2 {
            assert_eq!(
                node(&semantic, &actual.1, step, "token_selection"),
                &[actual.0[step]]
            );
            assert_eq!(
                node(&semantic, &actual.1, step, "token_feedback"),
                &[actual.0[step]]
            );
        }
        logits.push(node(&semantic, &actual.1, 0, "output_head").to_vec());
        if token == 1 {
            // Existing independent tiny source-table oracle from tests.rs.
            assert_eq!(actual.0, [1, 1]);
            assert_eq!(
                node(&semantic, &actual.1, 1, "layer.0.rope_q"),
                [-133, 369, -383, 87]
            );
            assert_eq!(
                node(&semantic, &actual.1, 1, "layer.0.rope_k"),
                [133, -346, 362, -76]
            );
            assert_eq!(
                node(&semantic, &actual.1, 1, "output_head"),
                [-303, 1472, -2622]
            );
        }
    }
    assert_ne!(
        logits[0], logits[1],
        "private input must affect complete logits"
    );
}

#[test]
fn tiny_qwen2_tied_head_keeps_lowest_nonzero_maximum_index() {
    let (semantic, source, resident) = tiny(Capacity::Capsule, true);
    let (_, _, paged) = tiny(Capacity::Decoder, true);
    let actual = execute_open(&paged.program, &[1]);
    assert_eq!(actual, execute_open(&resident.program, &[1]));
    assert_eq!(actual.0, [1, 1]);
    assert!(!source.tensors().iter().any(|t| t.id() == "lm_head.weight"));
    for step in 0..2 {
        let head = node(&semantic, &actual.1, step, "output_head");
        assert_eq!(head[1], head[2]);
        assert!(head[1] > head[0]);
        assert_eq!(node(&semantic, &actual.1, step, "token_feedback"), [1]);
    }
}

#[test]
fn tiny_qwen2_decoder_wire_transport_uses_request_long_file_guard() {
    let (_, _, paged) = tiny(Capacity::Decoder, false);
    let context = Context {
        plan: paged.program.digest(),
        session: [7; 32],
        operation: [8; 32],
        first_tensor_index: 0,
    };
    let admission = transport::admit(&paged.program, context).unwrap();
    assert!(admission.guard_disk_bytes() > 0);
    assert_eq!(
        admission.guard_capacity(),
        admission.primitive_requests_per_party()
    );
    // Existing role-bound native wire harness compares every reconstructed node
    // against direct native execution and verifies all queues retire. It is not
    // a network authentication or SDK benchmark claim.
    assert_eq!(
        transport::tests::serialized_matches_direct(&paged.program),
        [1, 1]
    );
}

fn frozen_key(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => "embedding".into(),
        "model.norm.weight" => "final.gamma".into(),
        _ => {
            let (layer, suffix) = name
                .strip_prefix("model.layers.")
                .unwrap()
                .split_once('.')
                .unwrap();
            let suffix = match suffix {
                "input_layernorm.weight" => "norm1.gamma",
                "post_attention_layernorm.weight" => "norm2.gamma",
                "self_attn.q_proj.weight" => "q.weight",
                "self_attn.q_proj.bias" => "q.bias",
                "self_attn.k_proj.weight" => "k.weight",
                "self_attn.k_proj.bias" => "k.bias",
                "self_attn.v_proj.weight" => "v.weight",
                "self_attn.v_proj.bias" => "v.bias",
                "self_attn.o_proj.weight" => "o.weight",
                "mlp.gate_proj.weight" => "gate.weight",
                "mlp.up_proj.weight" => "up.weight",
                "mlp.down_proj.weight" => "down.weight",
                _ => panic!("unexpected Qwen2 source {name}"),
            };
            format!("layer.{layer}.{suffix}")
        }
    }
}

fn ieee_bytes(tensor: &Value) -> (WeightDtype, Vec<u8>) {
    let (dtype, width) = match tensor["dtype"].as_str().unwrap() {
        "f32" => (WeightDtype::F32, 4),
        "f16" => (WeightDtype::F16, 2),
        "bf16" => (WeightDtype::Bf16, 2),
        _ => panic!("unexpected frozen dtype"),
    };
    let bytes = tensor["words"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|word| {
            let word = u32::from_str_radix(word.as_str().unwrap(), 16).unwrap();
            assert!(width == 4 || word <= u16::MAX as u32);
            word.to_le_bytes()[..width].to_vec()
        })
        .collect();
    (dtype, bytes)
}

fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}

#[test]
fn frozen_qwen2_decoder_matches_every_independent_semantic_oracle_node() {
    let input: Value = serde_json::from_str(include_str!(
        "../../../../paper/reproduction-contracts/fusefss/decoder/inputs.json"
    ))
    .unwrap();
    let expected: Value = serde_json::from_str(include_str!(
        "../../../../paper/reproduction-contracts/fusefss/decoder/semantic-expected.json"
    ))
    .unwrap();
    let case = input["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "qwen2-tied-generated")
        .unwrap();
    let oracle = expected["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["fixture_id"] == case["id"])
        .unwrap();
    let d = &case["dimensions"];
    let config = json!({"model_type":"qwen2", "hidden_size":d["hidden"], "intermediate_size":d["intermediate"],
        "num_hidden_layers":d["layers"], "num_attention_heads":d["heads"], "num_key_value_heads":d["kv_heads"],
        "head_dim":d["head_dim"], "vocab_size":d["vocab"], "max_position_embeddings":d["max_positions"],
        "hidden_act":"silu", "rms_norm_eps":"0.000001", "rope_theta":10000, "attention_bias":true,
        "attention_dropout":0, "use_sliding_window":false, "sliding_window":null, "rope_scaling":null,
        "use_cache":true, "max_window_layers":d["layers"], "tie_word_embeddings":case["tied_head"]});
    let prefix: Vec<u64> = integers(&case["prefix"])
        .into_iter()
        .map(|v| v as u64)
        .collect();
    let selections = case["generation_count"].as_u64().unwrap() as usize;
    assert_eq!(prefix, [2, 4]);
    assert_eq!(selections, 3);
    let mut builder = SourceBuilder::new_with_capacity(
        &serde_json::to_vec(&config).unwrap(),
        prefix.len(),
        selections,
        case["profile"].as_str().unwrap(),
        Capacity::Decoder,
    )
    .unwrap();
    let semantic = builder.admitted().unwrap().plan.clone();
    let mut consumed = BTreeMap::new();
    for (name, shape) in builder.admitted().unwrap().source_shapes.clone() {
        let key = frozen_key(&name);
        assert!(consumed.insert(key.clone(), name.clone()).is_none());
        let tensor = &case["tensors"][&key];
        assert_eq!(json!(shape), tensor["shape"]);
        let (dtype, bytes) = ieee_bytes(tensor);
        builder.append(&name, dtype, 0, &bytes).unwrap();
    }
    assert_eq!(
        consumed.len() + 2,
        case["tensors"].as_object().unwrap().len()
    );
    let (_, cos) = ieee_bytes(&case["tensors"]["rope.cos"]);
    let (_, sin) = ieee_bytes(&case["tensors"]["rope.sin"]);
    for (name, shape) in builder.admitted().unwrap().rotary_shapes.clone() {
        let bytes = shape.iter().product::<usize>() * 4;
        builder
            .append_rotary(&name, 0, &cos[..bytes], &sin[..bytes])
            .unwrap();
    }
    let imported = builder.seal().unwrap();
    let program = &imported.program;
    let (tokens, trace) = execute_open(program, &prefix);
    assert_eq!(program.profile(), oracle["profile"].as_str().unwrap());
    assert_eq!(tokens, integers(&oracle["final"]["generated_ids"]));
    let mut ids: Vec<_> = prefix.iter().map(|&v| v as i64).collect();
    ids.extend_from_slice(&tokens);
    assert_eq!(ids, integers(&oracle["final"]["token_ids"]));
    let nodes = oracle["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 186);
    assert_eq!(trace.len(), nodes.len());
    for ((selection, index, values), want) in trace.iter().zip(nodes) {
        assert_eq!(json!(selection), want["selection"]);
        assert_eq!(json!(index), want["semantic_index"]);
        assert_eq!(semantic.prefill.operations[*index].id, want["id"]);
        assert_eq!(json!(program.nodes[*index].shape), want["shape"]);
        assert_eq!(
            *values,
            integers(&want["values"]),
            "selection={selection} node={}",
            want["id"]
        );
    }
    for (step, state) in oracle["states"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            integers(&state["active_prefix_ids_private"]),
            ids[..prefix.len() + step]
        );
        assert_eq!(state["selected_id_private"].as_i64().unwrap(), tokens[step]);
        assert_eq!(
            node(&semantic, &trace, step, "token_feedback"),
            [tokens[step]]
        );
    }
}
