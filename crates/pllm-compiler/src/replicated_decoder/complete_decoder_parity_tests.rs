//! Whole original decoder reference, not compact execution qualification.
//!
//! Future adapter contract: construct `DecoderCase`, consume independent
//! `fresh_original_workers(&case)` through the REAL compact adapter, and return
//! `DecoderTrace` to `assert_compact_parity(&case, trace)`. Record complete frames
//! in delivery order, use `snapshot` at every named, globally drained original
//! boundary, and preserve original owner coordinates (or explicitly map them).
//! Supply the candidate's OWN complete admitted WorkBudget as `trace_finish`'s
//! `work` argument (never substitute the reference budget). Add original compact
//! entry/exit coordinates to `case.boundaries` before running either path.
//! `trace_finish` can drive an ordinary suffix. No replacement Program, changed
//! outputs/retirement, synthetic arithmetic, or default-policy switch is needed.
use super::*;
use crate::replicated_decoder::runtime::BoundaryProgress;

pub(in crate::replicated_decoder) type DecoderFrame = (Party, Party, Vec<u8>);
type Components = (Zeroizing<Vec<u64>>, Zeroizing<Vec<u64>>);

pub(in crate::replicated_decoder) struct DecoderCase {
    pub program: Arc<Program>,
    pub clear: BTreeMap<String, Vec<i64>>,
    pub boundaries: Vec<(&'static str, usize)>,
    pub feedback: (usize, usize),
    pub first_validity: Option<usize>,
    pub expected: Vec<u64>,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct DecoderSnapshot {
    pub party: Party,
    pub step: usize,
    pub cursors: puma::prss::Cursors,
    pub shares: Vec<Option<Components>>,
}

#[derive(Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder) struct DecoderTrace {
    pub wire: Vec<DecoderFrame>,
    pub boundaries: Vec<(&'static str, [DecoderSnapshot; 3])>,
    pub final_state: [DecoderSnapshot; 3],
    pub work: WorkBudget,
    pub final_frames: [Zeroizing<Vec<u8>>; 3],
    pub outputs: Vec<u64>,
}

pub(in crate::replicated_decoder) fn nonzero_case(tokens: [i64; 2]) -> DecoderCase {
    let (program, mut clear) = fixture_with_weights("qwen2", true, WeightMode::Public);
    clear.insert("input.tokens".into(), tokens.to_vec());
    // Independent sign margins already documented by the existing active-body
    // test. Q/K=0, V=I, O=-I/2, and all three MLP weights are nonzero.
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
    // Original public PerToken lowering names every private lookup coordinate.
    // Exactly one lookup consumes the previous selection in this two-pass case.
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
        // The compiler's final unscaled product masks BOTH selected IDs using
        // one sticky validity bit. Stop before that real final product.
        let masked = program.outputs[0].0;
        assert!(matches!(
            program.nodes[masked].code,
            Code::Product { rescale: false, .. }
        ));
        boundaries.push(("final-acceptance/before-mask", masked));
        // Walk the compiler's actual sticky conjunction chain back across the
        // second-prefix frontier. No validity/output or lifetime is invented.
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
        boundaries,
        feedback,
        first_validity,
        expected,
    }
}

/// Rejection values match checked_acceptance_tests: both prefixes select 1,
/// since their embeddings have the same opposite signs and body weights are
/// zero. Only feedback's +/-1.5 embedding fails the frozen +/-1 check on the
/// second pass; the +/-0.75 control preserves both nonzero IDs.
/// Reuse the shared tiny source fixture, extending the original prompt to two
/// token-0 entries; no graph/numeric implementation is replaced or patched.
pub(in crate::replicated_decoder) fn checked_case(reject_feedback: bool) -> DecoderCase {
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
            // Same conservative public interval for success/rejection. The
            // success control still executes the identical protected guards.
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
    assert!(program
        .nodes
        .iter()
        .any(|n| matches!(n.code, Code::CheckedRange { .. })));
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

/// Genuine three-selection compile. Zero body weights preserve embedding signs;
/// token 0 = (+1,+1), token 1 = (+1,0), and swapped head rows select ID 1 for
/// BOTH. Thus prompt [0,0] independently implies [1,1,1], not captured output.
pub(in crate::replicated_decoder) fn three_selection_case() -> DecoderCase {
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
    // Prefix two uses first ID once; prefix three reuses BOTH original IDs.
    assert_eq!(uses.len(), 3);
    assert_eq!(
        uses.iter()
            .map(|(_, coordinate)| *coordinate)
            .collect::<Vec<_>>(),
        [program.outputs[0], program.outputs[0], program.outputs[1]]
    );
    let boundaries = vec![
        ("first-selection/private-feedback", program.outputs[0].0 + 1),
        ("second-prefix/feedback-lookup", uses[0].0),
        (
            "second-selection/private-feedback",
            program.outputs[1].0 + 1,
        ),
        ("third-prefix/first-feedback-lookup", uses[1].0),
        ("third-prefix/second-feedback-lookup", uses[2].0),
    ];
    assert!(boundaries.windows(2).all(|pair| pair[0].1 < pair[1].1));
    DecoderCase {
        feedback: program.outputs[0],
        program,
        clear,
        boundaries,
        first_validity: None,
        expected: vec![1, 1, 1],
    }
}

/// Early prompt row may reject; original attention then moves every residual
/// back inside the frozen range, so subsequent guards genuinely succeed.
/// Growing-prefix recompute intentionally revisits the rejected prompt row too;
/// "later valid" means genuine subsequent guards, NOT erased prompt history.
/// Both cases keep identical public envelopes and complete fixed work.
pub(in crate::replicated_decoder) fn early_rejection_case(reject_prompt: bool) -> DecoderCase {
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
    // Original Qwen2 V bias makes every value (-0.5,+0.5), independently
    // of guarded inputs. Original O=I shifts residuals by that vector. With
    // Q=K=0, each causal softmax row has unit mass (source rounding has ample
    // 0.25 margin): the bad (+1.25,-1.25) row becomes (+0.75,-0.75), while
    // last/feedback (+0.25,-0.25) rows become (-0.25,+0.25).
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
    // Last row has opposite nonzero signs; the swapped head selects ID 1.
    // Its +/-0.25 embedding remains valid and selects ID 1 again. Rejection
    // must mask these NONZERO selections, not pass vacuously on zero IDs.
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

/// PUBLIC deterministic test keys only. Every call owns fresh consuming PRSS
/// runtimes and independently authenticated ordinary public-source files. Equal
/// bindings are deliberate; never connect the separate runs' routers.
pub(in crate::replicated_decoder) fn fresh_original_workers(case: &DecoderCase) -> [Worker; 3] {
    let context = puma::Context::new([42; 32], case.program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        case.program.work_budget().limits().unwrap(),
    )
    .unwrap();
    let mut roles = runtimes.map(|runtime| {
        Worker::new_with_source(
            case.program.clone(),
            runtime,
            [42; 32],
            PublicSource::test_values(&case.program, &case.clear),
        )
        .unwrap()
    });
    // Identical ordinary input splitting to tests::workers/body_integration_tests.
    for (key, n) in case.program.inputs() {
        assert_eq!(case.clear[key].len(), n);
        let first: Vec<_> = (0..n)
            .map(|i| 0xc751f00d12345678u64.wrapping_add(i as u64 * 17))
            .collect();
        let second: Vec<_> = (0..n)
            .map(|i| 0x254277811928f93du64.wrapping_add(i as u64 * 311))
            .collect();
        let third: Vec<_> = case.clear[key]
            .iter()
            .zip(first.iter().zip(&second))
            .map(|(x, (a, b))| (*x as u64).wrapping_sub(*a).wrapping_sub(*b))
            .collect();
        let shares = [first, second, third];
        for (i, role) in roles.iter_mut().enumerate() {
            role.import(key, &shares[i], &shares[(i + 1) % 3]).unwrap();
        }
    }
    roles
}

pub(in crate::replicated_decoder) fn snapshot(worker: &Worker) -> DecoderSnapshot {
    let (step, cursors, shares) = worker.test_boundary_state().unwrap();
    DecoderSnapshot {
        party: worker.party(),
        step,
        cursors,
        shares: shares
            .map(|value| {
                value.map(|(a, b)| (Zeroizing::new(a.to_vec()), Zeroizing::new(b.to_vec())))
            })
            .collect(),
    }
}

fn route_round(
    roles: &mut [Worker; 3],
    queue: &mut Vec<DecoderFrame>,
    wire: &mut Vec<DecoderFrame>,
) {
    for (from, to, raw) in std::mem::take(queue) {
        // Preserve ALL framing/bindings/payload, sender, recipient and order.
        let replies = roles[to as usize].receive(&raw, from).unwrap();
        wire.push((from, to, raw));
        queue.extend(
            replies
                .into_iter()
                .map(|(next, raw)| (to, next, raw.as_bytes().to_vec())),
        );
    }
}

pub(in crate::replicated_decoder) fn trace_to_boundary(
    roles: &mut [Worker; 3],
    target: usize,
    wire: &mut Vec<DecoderFrame>,
) -> [DecoderSnapshot; 3] {
    let mut queue = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() {
            let mut reached = 0;
            for role in roles.iter_mut() {
                let from = role.party();
                match role.advance_before(target).unwrap() {
                    BoundaryProgress::Reached => reached += 1,
                    BoundaryProgress::Frames(frames) => queue.extend(
                        frames
                            .into_iter()
                            .map(|(to, raw)| (from, to, raw.as_bytes().to_vec())),
                    ),
                }
            }
            if reached == 3 {
                assert!(queue.is_empty());
                let state = roles.each_ref().map(snapshot);
                assert!(state.iter().all(|state| state.step == target));
                return state;
            }
        } else {
            route_round(roles, &mut queue, wire);
        }
    }
    panic!("complete decoder failed to reach drained boundary {target}");
}

/// Ordinary suffix/default baseline: ONLY Worker::advance. Snapshots borrow both
/// real components; consuming finish_frame still reveals final IDs only.
pub(in crate::replicated_decoder) fn trace_finish(
    case: &DecoderCase,
    mut roles: [Worker; 3],
    mut wire: Vec<DecoderFrame>,
    boundaries: Vec<(&'static str, [DecoderSnapshot; 3])>,
    work: WorkBudget,
) -> DecoderTrace {
    let mut queue = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() {
            if roles.iter().all(Worker::complete) {
                let final_state = roles.each_ref().map(snapshot);
                let final_frames = roles.map(|role| role.finish_frame().unwrap());
                let outputs = (0..case.program.outputs.len())
                    .map(|i| {
                        let offset = 73 + 8 * i;
                        final_frames.iter().fold(0u64, |sum, frame| {
                            sum.wrapping_add(u64::from_le_bytes(
                                frame[offset..offset + 8].try_into().unwrap(),
                            ))
                        })
                    })
                    .collect();
                return DecoderTrace {
                    wire,
                    boundaries,
                    final_state,
                    work,
                    final_frames,
                    outputs,
                };
            }
            for role in &mut roles {
                if !role.complete() {
                    let from = role.party();
                    queue.extend(
                        role.advance()
                            .unwrap()
                            .into_iter()
                            .map(|(to, raw)| (from, to, raw.as_bytes().to_vec())),
                    );
                }
            }
        } else {
            route_round(&mut roles, &mut queue, &mut wire);
        }
    }
    panic!("complete ordinary decoder did not finish");
}

fn assert_work(program: &Program, states: &[DecoderSnapshot; 3]) {
    for state in states {
        let budget = schedule_work(&program.nodes[..state.step])
            .unwrap()
            .cursors();
        assert_eq!(state.cursors.public_blocks, budget.public_blocks);
        assert_eq!(state.cursors.pair_blocks, budget.pair_blocks);
        // Helper/evaluator pivots affect actual private/AES work per party.
        // Compare exact actual counters across runs, but charge the original
        // conservative per-party schedule rather than replacing it with a sum.
        assert!(state.cursors.private_blocks <= budget.private_blocks);
        assert!(state.cursors.aes_work <= budget.aes_work);
    }
}

#[derive(Debug, PartialEq, Eq)]
enum TraceViolation {
    Work,
    Feedback,
    Acceptance,
    Frames,
}

fn component_pairs(
    states: &[DecoderSnapshot; 3],
    owner: usize,
    lane: usize,
) -> Option<[(u64, u64); 3]> {
    let pair = |i: usize| {
        let (a, b) = states[i].shares.get(owner)?.as_ref()?;
        Some((*a.get(lane)?, *b.get(lane)?))
    };
    let pairs = [pair(0)?, pair(1)?, pair(2)?];
    (0..3)
        .all(|i| pairs[i].1 == pairs[(i + 1) % 3].0)
        .then_some(pairs)
}

fn open_pairs(pairs: [(u64, u64); 3]) -> u64 {
    pairs
        .iter()
        .fold(0u64, |sum, pair| sum.wrapping_add(pair.0))
}

/// Independent reference-contract checks, not another execution implementation.
/// Catch a self-consistent but incomplete trace before ordinary/compact equality:
/// no per-selection budget reset, feedback reshare, sticky reset or early reveal.
fn trace_contract(case: &DecoderCase, trace: &DecoderTrace) -> Result<(), TraceViolation> {
    use TraceViolation::*;
    if trace.work != case.program.work_budget() || trace.boundaries.len() != case.boundaries.len() {
        return Err(Work);
    }
    let mut previous = [puma::prss::Cursors::default(); 3];
    for (states, target) in trace
        .boundaries
        .iter()
        .zip(&case.boundaries)
        .map(|((name, states), &(expected_name, target))| {
            (states, (name == &expected_name).then_some(target))
        })
        .chain(std::iter::once((
            &trace.final_state,
            Some(case.program.nodes.len()),
        )))
    {
        let target = target.ok_or(Work)?;
        let bound = schedule_work(&case.program.nodes[..target])
            .map_err(|_| Work)?
            .cursors();
        for i in 0..3 {
            let actual = states[i].cursors;
            let prior = previous[i];
            if states[i].party as usize != i
                || states[i].step != target
                || actual.public_blocks != bound.public_blocks
                || actual.pair_blocks != bound.pair_blocks
                || actual.private_blocks > bound.private_blocks
                || actual.aes_work > bound.aes_work
                || actual.public_blocks < prior.public_blocks
                || actual.pair_blocks < prior.pair_blocks
                || actual.private_blocks < prior.private_blocks
                || actual.aes_work < prior.aes_work
            {
                return Err(Work);
            }
            previous[i] = actual;
        }
    }
    let boundary = |name| {
        trace
            .boundaries
            .iter()
            .find(|(id, _)| *id == name)
            .map(|(_, states)| states)
    };
    let initial = boundary("first-selection/private-feedback").ok_or(Feedback)?;
    let lookup = boundary("second-prefix/feedback-lookup").ok_or(Feedback)?;
    let (owner, lane) = case.feedback;
    let selected = component_pairs(initial, owner, lane).ok_or(Feedback)?;
    if component_pairs(lookup, owner, lane) != Some(selected) {
        return Err(Feedback);
    }
    if let Some(valid) = case.first_validity {
        if open_pairs(component_pairs(initial, valid, 0).ok_or(Acceptance)?) != 1 {
            return Err(Acceptance);
        }
        let before_mask = boundary("final-acceptance/before-mask").ok_or(Acceptance)?;
        let Code::Product {
            left,
            right,
            rescale: false,
            ..
        } = case.program.nodes[case.program.outputs[0].0].code
        else {
            return Err(Acceptance);
        };
        if component_pairs(before_mask, left, 0) != Some(selected) {
            return Err(Feedback);
        }
        let flags = component_pairs(before_mask, right, 0).ok_or(Acceptance)?;
        let accepted = open_pairs(flags);
        if accepted > 1 || trace.outputs.last() != Some(&accepted) {
            return Err(Acceptance);
        }
        for (i, output) in trace.outputs.iter().take(2).enumerate() {
            if component_pairs(before_mask, right, i) != Some(flags)
                || *output
                    != open_pairs(component_pairs(before_mask, left, i).ok_or(Acceptance)?)
                        .wrapping_mul(accepted)
            {
                return Err(Acceptance);
            }
        }
    } else if component_pairs(&trace.final_state, owner, lane) != Some(selected) {
        return Err(Feedback);
    }
    if trace.outputs != case.expected {
        return Err(Acceptance);
    }
    for (party, frame) in trace.final_frames.iter().enumerate() {
        if frame.len() != 73 + 8 * case.program.outputs.len()
            || &frame[..4] != b"PUR1"
            || frame[4..36] != [42; 32]
            || frame[36..68] != case.program.digest()
            || frame[68] as usize != party
            || frame[69..73] != (case.program.outputs.len() as u32).to_le_bytes()
        {
            return Err(Frames);
        }
        for (i, &(owner, lane)) in case.program.outputs.iter().enumerate() {
            let pairs = component_pairs(&trace.final_state, owner, lane).ok_or(Frames)?;
            if frame[73 + 8 * i..81 + 8 * i] != pairs[party].0.to_le_bytes()
                || trace.outputs[i] != open_pairs(pairs)
            {
                return Err(Frames);
            }
        }
    }
    Ok(())
}

/// Produce the reusable expanded reference and prove diagnostic stops do not
/// change its complete ordinary wire, actual shares/cursors, or final frames.
pub(in crate::replicated_decoder) fn run_expanded(case: &DecoderCase) -> DecoderTrace {
    assert_eq!(
        case.program.work_budget(),
        schedule_work(&case.program.nodes).unwrap()
    );
    assert_eq!(
        case.program.operation_limit(),
        case.program.work_budget().operations()
    );
    let ordinary = trace_finish(
        case,
        fresh_original_workers(case),
        Vec::new(),
        Vec::new(),
        case.program.work_budget(),
    );
    let mut roles = fresh_original_workers(case);
    let mut wire = Vec::new();
    let boundaries = case
        .boundaries
        .iter()
        .map(|&(name, target)| {
            let state = trace_to_boundary(&mut roles, target, &mut wire);
            assert_work(&case.program, &state);
            (name, state)
        })
        .collect();
    let observed = trace_finish(case, roles, wire, boundaries, case.program.work_budget());
    assert_eq!(observed.wire, ordinary.wire);
    assert_eq!(observed.final_state, ordinary.final_state);
    assert_eq!(observed.final_frames, ordinary.final_frames);
    assert_eq!(observed.work, ordinary.work);
    assert_eq!(observed.outputs, ordinary.outputs);
    assert_eq!(observed.outputs, case.expected);
    assert_work(&case.program, &observed.final_state);
    assert!(!observed.wire.is_empty());
    assert_eq!(trace_contract(case, &observed), Ok(()));
    observed
}

/// Call ONLY with a trace emitted by the real compact adapter. No such adapter
/// is implemented or selected here; ordinary-vs-ordinary tests are baseline
/// determinism/observability controls, not evidence of compact execution.
#[allow(dead_code)]
pub(in crate::replicated_decoder) fn assert_compact_parity(
    case: &DecoderCase,
    actual: DecoderTrace,
) {
    assert_eq!(trace_contract(case, &actual), Ok(()));
    assert_eq!(actual, run_expanded(case));
}

fn assert_private_feedback(case: &DecoderCase, trace: &DecoderTrace, first_id: u64) {
    let (owner, lane) = case.feedback;
    let mut initial = None;
    for (_, states) in &trace.boundaries[..2] {
        let pairs: [_; 3] = std::array::from_fn(|i| {
            let (a, b) = states[i].shares[owner].as_ref().unwrap();
            (a[lane], b[lane])
        });
        for i in 0..3 {
            assert_eq!(pairs[i].1, pairs[(i + 1) % 3].0);
        }
        if let Some(initial) = initial {
            assert_eq!(pairs, initial, "feedback components changed before lookup");
        } else {
            initial = Some(pairs);
        }
    }
    // Original checked compiler gathers the selected IDs before masking; gather
    // must preserve BOTH components of the private feedback, not reconstruct it.
    if let Some(valid) = case.first_validity {
        let first_pass = &trace.boundaries[0].1;
        let accepted = first_pass.iter().fold(0u64, |sum, state| {
            sum.wrapping_add(state.shares[valid].as_ref().unwrap().0[0])
        });
        assert_eq!(
            accepted, 1,
            "first prefix must pass before private feedback"
        );
        let masked = case.program.outputs[0].0;
        let Code::Product { left, .. } = case.program.nodes[masked].code else {
            unreachable!()
        };
        for (i, state) in trace.boundaries[2].1.iter().enumerate() {
            let (a, b) = state.shares[left].as_ref().unwrap();
            assert_eq!((a[0], b[0]), initial.unwrap()[i]);
        }
    }
    if case.expected.len() == 2 {
        assert_eq!(trace.outputs[0], first_id);
    }
}

fn large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(test)
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn whole_nonzero_decoder_reference_preserves_ordered_wire_both_components_and_work() {
    large_stack(|| {
        for tokens in [[0, 0], [1, 0]] {
            let case = nonzero_case(tokens);
            let first = run_expanded(&case);
            let second = run_expanded(&case);
            assert_private_feedback(&case, &first, case.expected[0]);
            assert_eq!(first, second); // Fresh consuming owners and separate routers.
            let a = &first.boundaries[0].1;
            let b = &first.final_state;
            for i in 0..3 {
                assert!(b[i].cursors.public_blocks > a[i].cursors.public_blocks);
                assert!(b[i].cursors.pair_blocks > a[i].cursors.pair_blocks);
                assert!(b[i].cursors.aes_work > a[i].cursors.aes_work);
            }
        }
    });
}

#[test]
fn whole_checked_decoder_second_prefix_failure_masks_earlier_private_selection() {
    large_stack(|| {
        let mut accepted_work = None;
        for reject_feedback in [false, true] {
            let case = checked_case(reject_feedback);
            let first = run_expanded(&case);
            assert_private_feedback(&case, &first, 1);
            let before_mask = &first.boundaries[2].1;
            let Code::Product { left, right, .. } =
                case.program.nodes[case.program.outputs[0].0].code
            else {
                unreachable!()
            };
            // Inspect only this generated test's pre-mask selected-ID components:
            // no activations/logits reconstructed; production export unchanged.
            let selected = before_mask.iter().fold(0u64, |sum, state| {
                sum.wrapping_add(state.shares[left].as_ref().unwrap().0[0])
            });
            let valid = before_mask.iter().fold(0u64, |sum, state| {
                sum.wrapping_add(state.shares[right].as_ref().unwrap().0[0])
            });
            assert_eq!(
                selected, 1,
                "nonzero earlier selection must exist before final masking"
            );
            assert_eq!(valid, u64::from(!reject_feedback));
            if let Some(work) = accepted_work {
                assert_eq!(first.work, work, "rejection must not short-circuit work");
            } else {
                accepted_work = Some(first.work);
            }
            assert_eq!(first, run_expanded(&case));
        }
    });
}

#[test]
fn third_original_selection_reuses_both_private_ids_and_charges_all_three_prefixes() {
    large_stack(|| {
        let case = three_selection_case();
        let trace = run_expanded(&case);
        assert_eq!(trace.outputs, [1, 1, 1]); // Independent embedding/head sign oracle.
        assert_eq!(trace.boundaries.len(), 5);
        for (output, origin, uses) in [(0, 0, vec![1, 3]), (1, 2, vec![4])] {
            let (owner, lane) = case.program.outputs[output];
            let original = component_pairs(&trace.boundaries[origin].1, owner, lane).unwrap();
            assert_eq!(open_pairs(original), 1);
            for boundary in uses {
                assert_eq!(
                    component_pairs(&trace.boundaries[boundary].1, owner, lane),
                    Some(original),
                    "both original private components must survive feedback"
                );
            }
            assert_eq!(
                component_pairs(&trace.final_state, owner, lane),
                Some(original)
            );
        }
        // Independently partition the original schedule at the two protected
        // selections. No per-prefix budget substitution/reset and no missing
        // third prefix can satisfy these charged-work and actual-cursor checks.
        let first_end = case.program.outputs[0].0 + 1;
        let second_end = case.program.outputs[1].0 + 1;
        let prefix_work = [
            schedule_work(&case.program.nodes[..first_end]).unwrap(),
            schedule_work(&case.program.nodes[first_end..second_end]).unwrap(),
            schedule_work(&case.program.nodes[second_end..]).unwrap(),
        ];
        assert!(prefix_work.iter().all(|work| work.operations() > 0));
        assert_eq!(
            trace.work,
            prefix_work[0]
                .checked_add(prefix_work[1])
                .unwrap()
                .checked_add(prefix_work[2])
                .unwrap()
        );
        assert!(prefix_work[0].operations() < prefix_work[1].operations());
        assert!(prefix_work[1].operations() < prefix_work[2].operations());
        for party in 0..3 {
            let a = trace.boundaries[0].1[party].cursors;
            let b = trace.boundaries[2].1[party].cursors;
            let c = trace.final_state[party].cursors;
            assert!(a.pair_blocks < b.pair_blocks && b.pair_blocks < c.pair_blocks);
            assert!(a.aes_work < b.aes_work && b.aes_work < c.aes_work);
        }
    });
}

#[test]
fn early_rejection_remains_sticky_after_later_valid_guards_and_full_feedback_work() {
    large_stack(|| {
        let mut complete_work = None;
        for reject_prompt in [false, true] {
            let mut case = early_rejection_case(reject_prompt);
            let final_valid = case.program.outputs[2].0;
            let Code::Product {
                left: prior_valid,
                right: later_valid,
                rescale: false,
                ..
            } = case.program.nodes[final_valid].code
            else {
                panic!("original request-wide sticky conjunction required")
            };
            // Observe its REAL operands before retirement, not a replacement
            // validity owner/output. Existing two-pass acceptance remains intact.
            assert!(case.boundaries[1].1 < final_valid);
            assert!(final_valid < case.boundaries[2].1);
            case.boundaries
                .insert(2, ("late-valid-guard/before-sticky", final_valid));
            let mut roles = fresh_original_workers(&case);
            let mut wire = Vec::new();
            let expected_valid = u64::from(!reject_prompt);
            // Follow ORIGINAL sticky dependencies back to the first guard's
            // all-valid result. Observe rejection before even the first private
            // selection, rather than inferring an early failure from final zero.
            let mut first_guard = case.first_validity.unwrap();
            while let Code::Product {
                left,
                rescale: false,
                ..
            } = case.program.nodes[first_guard].code
            {
                first_guard = left;
            }
            assert!(matches!(
                case.program.nodes[first_guard].code,
                Code::Tensor { .. }
            ));
            assert!(first_guard + 1 < case.boundaries[0].1);
            let early = trace_to_boundary(&mut roles, first_guard + 1, &mut wire);
            assert_work(&case.program, &early);
            assert_eq!(
                open_pairs(component_pairs(&early, first_guard, 0).unwrap()),
                expected_valid
            );
            let boundaries: Vec<_> = case
                .boundaries
                .iter()
                .map(|&(name, target)| {
                    let states = trace_to_boundary(&mut roles, target, &mut wire);
                    assert_work(&case.program, &states);
                    (name, states)
                })
                .collect();
            let trace = trace_finish(&case, roles, wire, boundaries, case.program.work_budget());
            let first_valid = case.first_validity.unwrap();
            for boundary in [0, 1] {
                assert_eq!(
                    open_pairs(
                        component_pairs(&trace.boundaries[boundary].1, first_valid, 0).unwrap()
                    ),
                    expected_valid
                );
            }
            let late = &trace.boundaries[2].1;
            assert_eq!(
                open_pairs(component_pairs(late, prior_valid, 0).unwrap()),
                expected_valid
            );
            assert_eq!(
                open_pairs(component_pairs(late, later_valid, 0).unwrap()),
                1,
                "a genuinely later guard succeeds even after early rejection"
            );
            let before_mask = &trace.boundaries[3].1;
            let Code::Product {
                left: ids,
                right: flags,
                rescale: false,
                ..
            } = case.program.nodes[case.program.outputs[0].0].code
            else {
                panic!("original final selected-ID mask required")
            };
            let first_id =
                component_pairs(&trace.boundaries[0].1, case.feedback.0, case.feedback.1).unwrap();
            assert_eq!(open_pairs(first_id), 1);
            assert_eq!(
                component_pairs(&trace.boundaries[1].1, case.feedback.0, case.feedback.1),
                Some(first_id)
            );
            assert_eq!(component_pairs(before_mask, ids, 0), Some(first_id));
            for lane in 0..2 {
                assert_eq!(
                    open_pairs(component_pairs(before_mask, ids, lane).unwrap()),
                    1,
                    "both nonzero original IDs exist before aggregate masking"
                );
                assert_eq!(
                    open_pairs(component_pairs(before_mask, flags, lane).unwrap()),
                    expected_valid
                );
                let (owner, index) = case.program.outputs[lane];
                assert_eq!(
                    open_pairs(component_pairs(&trace.final_state, owner, index).unwrap()),
                    expected_valid
                );
            }
            assert_eq!(trace.outputs, vec![expected_valid; 3]);
            assert_eq!(trace.outputs, case.expected);
            assert_eq!(
                open_pairs(component_pairs(&trace.final_state, final_valid, 0).unwrap()),
                expected_valid
            );
            assert_eq!(trace.work, schedule_work(&case.program.nodes).unwrap());
            if let Some(work) = complete_work {
                assert_eq!(trace.work, work, "early rejection cannot omit fixed work");
            } else {
                complete_work = Some(trace.work);
            }
            assert_work(&case.program, &trace.final_state);
            assert!(!trace.wire.is_empty());
            for party in 0..3 {
                let first = trace.boundaries[0].1[party].cursors;
                let final_state = &trace.final_state[party];
                assert!(first.pair_blocks > early[party].cursors.pair_blocks);
                assert_eq!(final_state.step, case.program.nodes.len());
                assert!(final_state.cursors.pair_blocks > first.pair_blocks);
                assert!(final_state.cursors.aes_work > first.aes_work);
                let frame = &trace.final_frames[party];
                assert_eq!(frame.len(), 73 + 3 * 8);
                assert_eq!(&frame[..4], b"PUR1");
                assert_eq!(frame[4..36], [42; 32]);
                assert_eq!(frame[36..68], case.program.digest());
                assert_eq!(frame[68] as usize, party);
                assert_eq!(frame[69..73], 3u32.to_le_bytes());
                for (lane, &(owner, index)) in case.program.outputs.iter().enumerate() {
                    let pairs = component_pairs(&trace.final_state, owner, index).unwrap();
                    assert_eq!(
                        frame[73 + 8 * lane..81 + 8 * lane],
                        pairs[party].0.to_le_bytes()
                    );
                }
            }
        }
    });
}

#[test]
fn reference_trace_contract_rejects_reset_work_feedback_sticky_and_final_disclosure() {
    large_stack(|| {
        // Mutate copied diagnostics only. These are negative trace controls,
        // NOT compact execution, forged protocol owners or a substitute Program.
        let case = checked_case(true);
        let mut trace = run_expanded(&case);
        let complete = trace.work;
        trace.work = schedule_work(&case.program.nodes[..case.boundaries[0].1]).unwrap();
        assert_eq!(trace_contract(&case, &trace), Err(TraceViolation::Work));
        trace.work = complete;

        let counters = trace.final_state[0].cursors;
        trace.final_state[0].cursors = trace.boundaries[0].1[0].cursors;
        assert_eq!(trace_contract(&case, &trace), Err(TraceViolation::Work));
        trace.final_state[0].cursors = counters;

        let (owner, lane) = case.feedback;
        trace.boundaries[1].1[0].shares[owner].as_mut().unwrap().1[lane] ^= 1;
        assert_eq!(trace_contract(&case, &trace), Err(TraceViolation::Feedback));
        trace.boundaries[1].1[0].shares[owner].as_mut().unwrap().1[lane] ^= 1;

        let Code::Product { right, .. } = case.program.nodes[case.program.outputs[0].0].code else {
            unreachable!()
        };
        trace.boundaries[2].1[0].shares[right].as_mut().unwrap().0[0] ^= 1;
        assert_eq!(
            trace_contract(&case, &trace),
            Err(TraceViolation::Acceptance)
        );
        trace.boundaries[2].1[0].shares[right].as_mut().unwrap().0[0] ^= 1;

        trace.outputs[0] = 1; // Earlier nonzero selection must not leak on rejection.
        assert_eq!(
            trace_contract(&case, &trace),
            Err(TraceViolation::Acceptance)
        );
        trace.outputs[0] = 0;
        trace.outputs[2] = 1; // A prior successful prefix cannot reset sticky failure.
        assert_eq!(
            trace_contract(&case, &trace),
            Err(TraceViolation::Acceptance)
        );
        trace.outputs[2] = 0;

        trace.final_frames[0][73] ^= 1;
        assert_eq!(trace_contract(&case, &trace), Err(TraceViolation::Frames));
        trace.final_frames[0][73] ^= 1;
        assert_eq!(trace_contract(&case, &trace), Ok(()));
    });
}
