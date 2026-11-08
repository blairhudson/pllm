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
// 64-bit payload accounting: owned executor headers changed from two Vecs to two
// Boxes (16 B saved per stored template); guard descriptor allowance changed from
// 128 to 64 B per checked site. Inverting only payload recovers all old JSON
// preimages; independent FROZEN numeric identities remain unchanged.
const EXPECTED: &[(&str, &str)] = &[
    (
        "frozen_qwen2_active_false",
        "2145a0218233e71f97884a0f2dbb0d09908e3c269aa23706d0b60dd90feaf963",
    ),
    (
        "frozen_qwen2_active_true",
        "ecd8091427dece79c4829a3253a69e490ce4643041d24e92e9517640673425d1",
    ),
    (
        "frozen_qwen3_active_false",
        "c89f4780ac2b8e26df953ad29fe6b7ccb7f83f21855ed6ff6f8ded2ed22636c5",
    ),
    (
        "frozen_qwen3_active_true",
        "03bc195ebc47c7cdd026be1ec01a5588aff1b289f6dcbcd8875dd7c6c06fa4cc",
    ),
    (
        "frozen_llama_active_false",
        "ff6d57f6f9dd4eedb1673d8a2878cb2eb1deebbac185e03554fa9686802133d3",
    ),
    (
        "frozen_llama_active_true",
        "2473dff44542994dcfc992e5774bba013f32a78d295d0d4b150e593a623b7878",
    ),
    (
        "public_matrix_nonzero_bias_two_selection",
        "bf6151280ea456ac9db8f31833147a7962a885973d828d269c44cefc34051e99",
    ),
    (
        "binary_mlp_128_plus_1",
        "0e7dc5bcc00667df0cb2948f1ec866c6bf5f8f14a78997f03eb97a03325673b7",
    ),
    (
        "checked_acceptance_masked_two_selection",
        "d77bf3071258c8596c9447cb0d057064ec9ffcc43e1bb5d1cdbf1949b3699fab",
    ),
    (
        "public_lookup_loop_128_plus_1",
        "a61f4c0b1505439b0e741a263e0a5da0485cc538ca88eef6520b138034db326f",
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
    let mut mismatches = Vec::new();
    let mut numeric_passed = 0usize;
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
            if fingerprint != expected.1 {
                println!("expanded graph snapshot: {name}\n{serialized}");
                mismatches.push((name, fingerprint, expected.1));
            }
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
        numeric_passed += 1;
        if !capture {
            println!("independent frozen identity PASS: {family}, active={active}, digest={expected}");
        }
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
        assert_eq!(numeric_passed, FROZEN.len());
        println!(
            "expanded graph summary: numeric_passed={numeric_passed}/{} snapshots={}/{} json_mismatch_count={}",
            FROZEN.len(),
            seen.len(),
            EXPECTED.len(),
            mismatches.len()
        );
        assert!(
            mismatches.is_empty(),
            "expanded graph JSON mismatches (name, actual, expected): {mismatches:?}"
        );
    }
}
