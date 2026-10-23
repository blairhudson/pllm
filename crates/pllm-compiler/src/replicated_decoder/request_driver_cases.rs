//! Compiler-only cases from the settled complete-decoder reference.
//! No Worker, PRSS/source construction, trace snapshot or execution qualification.
use super::*;
use crate::replicated_decoder::tests::fixture_sources;

pub(super) struct DecoderCase {
    pub program: Arc<Program>,
    pub clear: BTreeMap<String, Vec<i64>>,
    pub feedback: (usize, usize),
    pub first_validity: Option<usize>,
    pub expected: Vec<u64>,
}

pub(super) fn nonzero_case(tokens: [i64; 2]) -> DecoderCase {
    let (config, sources, mut clear) = fixture_sources("qwen2", true);
    let program = Arc::new(
        Program::compile_with_weights(&config, 2, 2, &sources, WeightMode::Public).unwrap(),
    );
    clear.insert("input.tokens".into(), tokens.to_vec());
    // Existing sign oracle: Q/K=0, V=I, O=-I/2, all MLP weights nonzero.
    let expected = match tokens {
        [0, 0] => vec![1, 1],
        [1, 0] => vec![0, 0],
        _ => panic!("whole-decoder reference requires a documented sign oracle"),
    };
    decoder_case(program, clear, expected)
}

fn decoder_case(
    program: Arc<Program>,
    clear: BTreeMap<String, Vec<i64>>,
    expected: Vec<u64>,
) -> DecoderCase {
    assert_eq!(program.inputs().collect::<Vec<_>>(), [("input.tokens", 2)]);
    let input = program
        .nodes
        .iter()
        .position(|n| matches!(&n.code, Code::Input(key) if key == "input.tokens"))
        .unwrap();
    let feedback_uses: Vec<_> = program
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(step, node)| match node.code {
            Code::TensorLoop {
                input: LoopInput::TokenOffsets { owner, index },
                ..
            } if owner != input => Some((step, (owner, index))),
            _ => None,
        })
        .collect();
    assert_eq!(feedback_uses.len(), 1);
    let (use_step, feedback) = feedback_uses[0];
    let mut boundaries = vec![
        ("first-selection/private-feedback", feedback.0 + 1),
        ("second-prefix/feedback-lookup", use_step),
    ];
    let mut first_validity = None;
    if expected.len() == 3 {
        let masked = program.outputs[0].0;
        assert!(matches!(
            program.nodes[masked].code,
            Code::Product { rescale: false, .. }
        ));
        boundaries.push(("final-acceptance/before-mask", masked));
        // Follow original sticky conjunction custody across the second prefix.
        let mut valid = program.outputs[2].0;
        while valid >= boundaries[0].1 {
            let Code::Product {
                left,
                rescale: false,
                ..
            } = program.nodes[valid].code
            else {
                panic!("expected original sticky-validity conjunction");
            };
            valid = left;
        }
        first_validity = Some(valid);
    } else {
        assert_eq!(program.outputs[0], feedback);
    }
    assert!(boundaries.windows(2).all(|w| w[0].1 < w[1].1));
    DecoderCase {
        program,
        clear,
        feedback,
        first_validity,
        expected,
    }
}

pub(super) fn checked_case(reject_feedback: bool) -> DecoderCase {
    let (config, mut sources, mut clear) = fixture_sources("qwen2", false);
    let feedback = if reject_feedback {
        3 * ONE / 2
    } else {
        3 * ONE / 4
    };
    for (key, values) in [
        (
            "model.embed_tokens.weight",
            vec![
                ONE / 4,
                -ONE / 4,
                feedback,
                -feedback,
                3 * ONE / 4,
                -3 * ONE / 4,
            ],
        ),
        (
            "lm_head.weight",
            vec![-ONE / 4, ONE / 4, ONE / 4, -ONE / 4, 0, 0],
        ),
    ] {
        let source = sources.get_mut(key).unwrap();
        source.lower = *values.iter().min().unwrap();
        source.upper = *values.iter().max().unwrap();
        if key == "model.embed_tokens.weight" {
            source.lower = -3 * ONE / 2;
            source.upper = 3 * ONE / 2;
        }
        source.digest = Sha256::digest(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .into();
        clear.insert(key.into(), values);
    }
    clear.insert("input.tokens".into(), vec![0, 0]);
    let execution = ExecutionPolicy {
        range_policy: RangePolicy::Checked,
        max_abs_activation: 1,
        ..ExecutionPolicy::default()
    };
    let program = Arc::new(
        Program::compile_with_policy(&config, 2, 2, &sources, WeightMode::Public, execution)
            .unwrap(),
    );
    assert!(
        program
            .nodes
            .iter()
            .any(|n| matches!(n.code, Code::CheckedRange { .. }))
    );
    decoder_case(
        program,
        clear,
        if reject_feedback {
            vec![0, 0, 0]
        } else {
            vec![1, 1, 1]
        },
    )
}

fn replace_source_values(
    sources: &mut BTreeMap<String, SourceTensor>,
    clear: &mut BTreeMap<String, Vec<i64>>,
    key: &str,
    values: Vec<i64>,
) {
    let source = sources.get_mut(key).unwrap();
    assert!(source.public_q18.is_none());
    assert_eq!(values.len(), source.shape.iter().product::<usize>());
    source.lower = *values.iter().min().unwrap();
    source.upper = *values.iter().max().unwrap();
    let mut hash = Sha256::new();
    for value in &values {
        hash.update(value.to_le_bytes());
    }
    source.digest = hash.finalize().into();
    clear.insert(key.into(), values);
}

pub(super) fn three_selection_case() -> DecoderCase {
    let (config, mut sources, mut clear) = fixture_sources("qwen2", false);
    replace_source_values(
        &mut sources,
        &mut clear,
        "lm_head.weight",
        vec![-ONE, 0, ONE, 0, 0, 0],
    );
    clear.insert("input.tokens".into(), vec![0, 0]);
    let program = Arc::new(
        Program::compile_with_weights(&config, 2, 3, &sources, WeightMode::Public).unwrap(),
    );
    assert_eq!(program.outputs.len(), 3);
    assert_eq!(program.inputs().collect::<Vec<_>>(), [("input.tokens", 2)]);
    let input = program
        .nodes
        .iter()
        .position(|node| matches!(&node.code, Code::Input(key) if key == "input.tokens"))
        .unwrap();
    let uses: Vec<_> = program
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(step, node)| match node.code {
            Code::TensorLoop {
                input: LoopInput::TokenOffsets { owner, index },
                ..
            } if owner != input => Some((step, (owner, index))),
            _ => None,
        })
        .collect();
    assert_eq!(uses.len(), 3);
    assert_eq!(
        uses.iter()
            .map(|(_, coordinate)| *coordinate)
            .collect::<Vec<_>>(),
        [program.outputs[0], program.outputs[0], program.outputs[1]]
    );
    let boundaries = [
        program.outputs[0].0 + 1,
        uses[0].0,
        program.outputs[1].0 + 1,
        uses[1].0,
        uses[2].0,
    ];
    assert!(boundaries.windows(2).all(|pair| pair[0] < pair[1]));
    DecoderCase {
        feedback: program.outputs[0],
        program,
        clear,
        first_validity: None,
        expected: vec![1, 1, 1],
    }
}

pub(super) fn early_rejection_case(reject_prompt: bool) -> DecoderCase {
    let (config, mut sources, mut clear) = fixture_sources("qwen2", false);
    let prompt = if reject_prompt {
        5 * ONE / 4
    } else {
        3 * ONE / 4
    };
    replace_source_values(
        &mut sources,
        &mut clear,
        "model.embed_tokens.weight",
        vec![ONE / 4, -ONE / 4, ONE / 4, -ONE / 4, prompt, -prompt],
    );
    let embedding = sources.get_mut("model.embed_tokens.weight").unwrap();
    embedding.lower = -3 * ONE / 2;
    embedding.upper = 3 * ONE / 2;
    replace_source_values(
        &mut sources,
        &mut clear,
        "lm_head.weight",
        vec![ONE / 4, -ONE / 4, -ONE / 4, ONE / 4, 0, 0],
    );
    replace_source_values(
        &mut sources,
        &mut clear,
        "model.layers.0.self_attn.v_proj.bias",
        vec![-ONE / 2, ONE / 2],
    );
    replace_source_values(
        &mut sources,
        &mut clear,
        "model.layers.0.self_attn.o_proj.weight",
        vec![ONE, 0, 0, ONE],
    );
    clear.insert("input.tokens".into(), vec![2, 0]);
    let program = Arc::new(
        Program::compile_with_policy(
            &config,
            2,
            2,
            &sources,
            WeightMode::Public,
            ExecutionPolicy {
                range_policy: RangePolicy::Checked,
                max_abs_activation: 1,
                ..ExecutionPolicy::default()
            },
        )
        .unwrap(),
    );
    decoder_case(
        program,
        clear,
        if reject_prompt {
            vec![0, 0, 0]
        } else {
            vec![1, 1, 1]
        },
    )
}
