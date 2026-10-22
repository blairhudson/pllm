//! Pre-extraction graph baselines. No graph lowering implementation lives here.
//!
//! PLLM_CAPTURE_PROGRAM_GRAPH=1 prints complete public metadata for review.
//! Capture output is NOT a passing frozen-snapshot regression.
//! EXPECTED freezes SHA-256 of that exact pre-extraction canonical JSON; failures
//! print the complete actual snapshot. Six independently frozen execution digests
//! below remain separate constraints, including during explicit capture.
use super::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

// Captured before graph traversal extraction, with all six FROZEN identities
// checked. Each digest covers every field of the complete canonical JSON from
// snapshot(), not merely the execution digest or selected summary counters.
// Expectations are literals: never derive them from a freshly compiled graph.
const EXPECTED: &[(&str, &str)] = &[
    (
        "frozen_qwen2_active_false",
        "ed9114342828ce061361a9e0cc18ac6c07e063ff0bf89976adc8e6994c652fe1",
    ),
    (
        "frozen_qwen2_active_true",
        "8d8b0086d880d6dd5fd159011bd0bb87f0404d2f5c530b23b5db61245b7dc77c",
    ),
    (
        "frozen_qwen3_active_false",
        "e1687c965956604820ac7a74e3b54f2f0bab3dfe550bb9889c0aed177a1f9aa2",
    ),
    (
        "frozen_qwen3_active_true",
        "efcffd9f3e52f68bec0fed18fed192db1de41b1a6ce9e64f2920528ff0a833fa",
    ),
    (
        "frozen_llama_active_false",
        "7a49e2fac21bc632eac0a6d65f27daa2dae5db826b6b8dbc0f85e6d3447a97ec",
    ),
    (
        "frozen_llama_active_true",
        "264b850805cf289feaf5f20f0d229a703aedc0e9ae5451fe1c3638921fe8a82d",
    ),
    (
        "public_matrix_nonzero_bias_two_selection",
        "72c56b2e0ce3b2b3ad8128c755fbd46d1bf4a91010b07498b2c717c87d24e341",
    ),
    (
        "binary_mlp_128_plus_1",
        "c67a3f7e9daa6089d87dbf1a3715f487d7e2bc322b87061b63ca988e6e3ea539",
    ),
    (
        "checked_acceptance_masked_two_selection",
        "e6e90a716c0f93d57c5360ef67ca49370df7597ea0cd50daa5afac9fee2a52f8",
    ),
    (
        "public_lookup_loop_128_plus_1",
        "2906e1a114bb6e35447513606d480230a9a670d7b15293f3188365a388d2cc5a",
    ),
];

// Same independently frozen execution identities as tests.rs. These remain
// mandatory even in capture-only mode; do not replace them with captured values.
const FROZEN: &[(&str, bool, &str)] = &[
    (
        "qwen2",
        false,
        "333aa62a92ec54bdfb694e6bc356d3b4d873629d0cbee5bbbbaa3fb447d28c6d",
    ),
    (
        "qwen2",
        true,
        "4529a304429056c986676ea4b8da1adf4cf864e23e41f40a0778a32268b671a1",
    ),
    (
        "qwen3",
        false,
        "83886ad38965f9244ba95b403855fb56c6298f44392c44e022b400b19118f7d2",
    ),
    (
        "qwen3",
        true,
        "98044e61b32ee4cde96ee63a0c0accc70a776b51b689ef763930353aea60d45e",
    ),
    (
        "llama",
        false,
        "f9b5f1dcc3c06ac87013adca91751f7acae25e5993fae0137b0b90ca98c08eca",
    ),
    (
        "llama",
        true,
        "9e9593392a6058d6dac04ecb442ec83a31d6e017291e5007144ac91731e4ce8c",
    ),
];

fn hex(digest: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").unwrap();
    }
    encoded
}

/// Metadata fixture mirrored from tests.rs::fixture_with_weights, whose helper
/// is private to that sibling module. Default parameters must replay all six
/// frozen identities above, independently constraining this mirror.
#[allow(clippy::too_many_arguments)] // Keep independent frozen fixture axes explicit.
fn fixture(
    family: &str,
    active: bool,
    weights: WeightMode,
    rows: usize,
    outputs: usize,
    intermediate: usize,
    execution: ExecutionPolicy,
    nonzero_bias: bool,
    vocabulary: usize,
) -> Program {
    let config = serde_json::to_vec(&json!({
        "model_type": family, "hidden_size": 2, "intermediate_size": intermediate,
        "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": 2, "vocab_size": vocabulary, "tie_word_embeddings": false,
        "hidden_act": "silu", "attention_bias": family == "qwen2",
        "max_position_embeddings": 64, "rope_theta": 10000,
        "rms_norm_eps": 0.000001, "attention_dropout": 0.0,
        "use_sliding_window": false, "use_cache": true, "max_window_layers": 1
    }))
    .unwrap();
    let shapes = Program::source_shapes(&config, rows, outputs).unwrap();
    let mut sources = BTreeMap::new();
    for (key, shape) in shapes {
        let values: Vec<i64> = if active && key == "model.embed_tokens.weight" {
            [ONE / 4, ONE / 4, -ONE / 4, -ONE / 4, 0, 0]
                .into_iter()
                .cycle()
                .take(shape.iter().product())
                .collect()
        } else if key == "model.embed_tokens.weight" {
            [ONE, ONE, ONE, 0, -ONE, 0]
                .into_iter()
                .cycle()
                .take(shape.iter().product())
                .collect()
        } else if key == "lm_head.weight" {
            [ONE, 0, -ONE, 0, 0, 0]
                .into_iter()
                .cycle()
                .take(shape.iter().product())
                .collect()
        } else if key.contains("norm") {
            vec![ONE; shape.iter().product()]
        } else if active && key.ends_with("v_proj.weight") {
            vec![ONE, 0, 0, ONE]
        } else if active && key.ends_with("o_proj.weight") {
            vec![-ONE / 2, 0, 0, -ONE / 2]
        } else if active && key.contains("mlp.") {
            vec![ONE / 64; shape.iter().product()]
        } else if nonzero_bias && key.ends_with(".bias") {
            vec![ONE / 128; shape.iter().product()]
        } else {
            vec![0; shape.iter().product()]
        };
        let digest = Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        sources.insert(
            key.clone(),
            SourceTensor {
                shape,
                lower: *values.iter().min().unwrap(),
                upper: *values.iter().max().unwrap(),
                public_q18: key.contains("norm").then(|| values.clone()),
                digest,
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
        );
    }
    let mut program =
        Program::compile_with_policy(&config, rows, outputs, &sources, weights, execution).unwrap();
    program.bind([7; 32], [9; 32]);
    program
}

fn snapshot(program: &Program) -> Value {
    let mut templates = BTreeMap::<[u8; 32], (usize, usize, usize)>::new();
    let mut stored = BTreeSet::new();
    let mut loops = Vec::new();
    let mut tensor_sites = 0;
    let mut checked_sites = 0;
    let mut retain_template = |template: &Arc<tensor::Program>| {
        stored.insert(Arc::as_ptr(template) as usize);
        let entry = templates.entry(template.digest()).or_insert((
            template.lanes(),
            template.output_lanes(),
            0,
        ));
        entry.2 += 1;
    };
    for (owner, node) in program.nodes.iter().enumerate() {
        match &node.code {
            Code::Tensor { program, .. } => {
                tensor_sites += 1;
                retain_template(program);
            }
            Code::CheckedRange { program, .. } => {
                checked_sites += 1;
                retain_template(program);
            }
            Code::TensorLoop {
                input,
                program,
                tail,
            } => {
                retain_template(program);
                if let Some(tail) = tail {
                    retain_template(tail);
                }
                let input = match input {
                    LoopInput::Slice(owner) => json!(["slice", owner]),
                    LoopInput::TokenOffsets { owner, index } => {
                        json!(["token_offsets", owner, index])
                    }
                };
                loops.push(json!({
                    "owner": owner, "input": input, "lanes": node.lanes,
                    "full_lanes": program.lanes(), "full_digest": hex(program.digest()),
                    "tail": tail.as_ref().map(|tail| json!({"lanes": tail.lanes(), "digest": hex(tail.digest())})),
                }));
            }
            _ => (),
        }
    }
    let inputs: Vec<_> = program.nodes[..program.inputs]
        .iter()
        .enumerate()
        .map(|(owner, node)| {
            let source = match &node.code {
                Code::Input(key) => json!({"kind": "private", "key": key}),
                Code::PublicInput {
                    key,
                    digest,
                    row_l1,
                    row_squared_norm,
                    row_l1_squared_sum,
                } => json!({
                    "kind": "public", "key": key, "digest": hex(*digest),
                    "row_l1": row_l1, "row_squared_norm": row_squared_norm,
                    "row_l1_squared_sum": row_l1_squared_sum,
                }),
                _ => panic!("non-input in input prefix"),
            };
            json!({"owner": owner, "lanes": node.lanes,
            "bounds": [node.bounds.lower(), node.bounds.upper()], "source": source})
        })
        .collect();
    let cursors = program.work.cursors();
    json!({
        "digest": hex(program.digest()), "source_binding": hex(program.source_binding),
        "nodes": program.nodes.len(), "input_prefix": program.inputs,
        "inputs": inputs, "outputs": program.outputs,
        "node_input_refs": program.nodes.iter().map(Node::inputs).collect::<Vec<_>>(),
        "node_lanes": program.nodes.iter().map(|node| node.lanes).collect::<Vec<_>>(),
        "retire": program.retire,
        "work": {"operations": program.work.operations(), "public_blocks": cursors.public_blocks,
            "pair_blocks": cursors.pair_blocks, "private_blocks": cursors.private_blocks,
            "aes_work": cursors.aes_work},
        "tensor_sites": tensor_sites, "checked_sites": checked_sites,
        "stored_templates": stored.len(), "unique_template_digests": templates.len(),
        "templates": templates.into_iter().map(|(digest, (lanes, output_lanes, references))|
            json!({"digest": hex(digest), "lanes": lanes, "output_lanes": output_lanes, "references": references})
        ).collect::<Vec<_>>(),
        "loops": loops,
        // Native payload accounting contains size_of metadata, not portable RSS.
        // Freeze for this target; a new pointer width needs separately reviewed values.
        "pointer_width": usize::BITS, "payload_bytes": program.payload_bytes(),
    })
}

#[test]
#[allow(clippy::const_is_empty)] // Guard against accidentally clearing frozen baselines.
fn expanded_graph_regression_snapshots() {
    let capture = std::env::var("PLLM_CAPTURE_PROGRAM_GRAPH").as_deref() == Ok("1");
    assert!(capture || !EXPECTED.is_empty(),
        "D26 BASELINE NOT FROZEN: run explicit capture-only command, review output, then populate EXPECTED before extraction");
    let mut seen = BTreeSet::new();
    let mut check = |name: String, program: Program| {
        assert!(seen.insert(name.clone()));
        let actual = snapshot(&program);
        if capture {
            println!(
                "D26_CAPTURE_ONLY_NOT_FROZEN ({name:?}, r#\"{}\"#),",
                serde_json::to_string(&actual).unwrap()
            );
        } else {
            let expected = EXPECTED
                .iter()
                .find(|(key, _)| *key == name)
                .unwrap_or_else(|| panic!("missing independent frozen graph snapshot: {name}"));
            let serialized = serde_json::to_string(&actual).unwrap();
            let fingerprint = hex(Sha256::digest(serialized.as_bytes()).into());
            assert_eq!(
                fingerprint, expected.1,
                "expanded graph changed: {name}\nactual snapshot: {serialized}"
            );
        }
    };
    for &(family, active, expected) in FROZEN {
        let program = fixture(
            family,
            active,
            WeightMode::Private,
            2,
            2,
            2,
            ExecutionPolicy::default(),
            false,
            3,
        );
        assert_eq!(
            hex(program.digest()),
            expected,
            "independent frozen identity: {family}, active={active}"
        );
        assert_eq!(program.outputs.len(), 2); // Real compiler feedback, not synthetic repeated sites.
        check(format!("frozen_{family}_active_{active}"), program);
    }
    // Full public qwen2 graph has original Q/K/V bias paths, RMSNorm, rotary,
    // causal attention and two protected selections with full-prefix feedback.
    let public = fixture(
        "qwen2",
        true,
        WeightMode::Public,
        2,
        2,
        2,
        ExecutionPolicy::default(),
        true,
        3,
    );
    assert!(public
        .nodes
        .iter()
        .any(|node| matches!(node.code, Code::PublicProduct { .. })));
    assert!(public.nodes[..public.inputs]
        .iter()
        .any(|node| matches!(&node.code, Code::PublicInput { key, .. } if key.ends_with(".bias"))));
    check("public_matrix_nonzero_bias_two_selection".into(), public);

    // A real 129-wide MLP gate/up product emits 128+1 through Builder::binary;
    // this is deliberately not a fabricated Program with a synthetic digest.
    let tail = fixture(
        "llama",
        true,
        WeightMode::Private,
        1,
        1,
        129,
        ExecutionPolicy::default(),
        false,
        3,
    );
    let binary_lanes: Vec<_> = tail
        .nodes
        .iter()
        .filter_map(|node| match node.code {
            Code::Product {
                shape: None,
                rescale: true,
                ..
            } => Some(node.lanes),
            _ => None,
        })
        .collect();
    assert!(
        binary_lanes.windows(2).any(|pair| pair == [128, 1]),
        "fixture must exercise actual 128+1 binary emission"
    );
    check("binary_mlp_128_plus_1".into(), tail);

    let checked = fixture(
        "qwen2",
        true,
        WeightMode::Public,
        2,
        2,
        2,
        ExecutionPolicy {
            range_policy: RangePolicy::Checked,
            max_abs_activation: 1,
            ..ExecutionPolicy::default()
        },
        false,
        3,
    );
    assert!(checked
        .nodes
        .iter()
        .any(|node| matches!(node.code, Code::CheckedRange { .. })));
    assert_eq!(checked.outputs.len(), 3); // Two masked IDs and one aggregate acceptance bit.
    let masked = checked.outputs[0].0;
    assert_eq!(checked.outputs[1], (masked, 1));
    assert!(matches!(
        checked.nodes[masked].code,
        Code::Product {
            shape: None,
            rescale: false,
            ..
        }
    ));
    check("checked_acceptance_masked_two_selection".into(), checked);

    let lookup_tail = fixture(
        "llama",
        true,
        WeightMode::Public,
        1,
        1,
        2,
        ExecutionPolicy::default(),
        false,
        129,
    );
    assert!(
        lookup_tail.nodes.iter().any(|node| matches!(&node.code,
            Code::TensorLoop { input: LoopInput::TokenOffsets { .. }, program, tail: Some(tail) }
                if node.lanes == 129 && program.lanes() == 128 && tail.lanes() == 1
        )),
        "fixture must retain distinct full/tail selector templates"
    );
    check("public_lookup_loop_128_plus_1".into(), lookup_tail);
    if capture {
        eprintln!("D26_CAPTURE_ONLY_NOT_FROZEN: {} cases printed; expectations MUST be frozen before graph extraction", seen.len());
    } else {
        assert_eq!(
            seen.len(),
            EXPECTED.len(),
            "stale/missing frozen graph cases"
        );
    }
}
