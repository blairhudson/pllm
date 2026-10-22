//! Include as a child of program.rs; see body_integration_handoff.md.
//! Generated graph, not a checkpoint or full-model qualification.
use super::*;
use crate::replicated_decoder::{
    runtime::body::integration::{constructor_entries, restored_state, Compiled, Invocation},
    tests::{execute, workers},
};

fn fixture(checked: bool) -> (Arc<Program>, usize) {
    let mut b = Builder {
        public_weights: true,
        selection: SelectionPolicy::Tiled,
        ..Builder::default()
    };
    let x = b
        .push(
            Code::Input("x".into()),
            3,
            Interval::new(-ONE, ONE).unwrap(),
        )
        .unwrap();
    let y = b
        .push(Code::Input("y".into()), 3, Interval::new(ONE, ONE).unwrap())
        .unwrap();
    let x = b.view(x, vec![3]);
    let y = b.view(y, vec![3]);
    let scores = b.binary(&x.words, &y.words, true).unwrap();
    let scores = b.binary(&scores, &x.words, false).unwrap();
    let end = b.nodes.len();
    let scores = if checked {
        b.checked_bound = Some(ONE);
        let safe = b.checked_view(&scores).unwrap();
        // Success on safe fallback must not clear first check's sticky failure.
        b.checked_bound = Some(2 * ONE);
        b.checked_view(&safe).unwrap()
    } else {
        scores
    };
    let first = b
        .argmax_tree(&View {
            words: scores,
            shape: vec![3],
        })
        .unwrap();
    // Private first selection changes second logits. No reconstruction between
    // selections; original compiler owns both stable tournaments and feedback.
    let token = b.gather(&first.words).unwrap();
    let negative = b.affine(token, None, affine(-1, 0, 0)).unwrap();
    let constant = b
        .push(Code::PublicValues(vec![1]), 1, Interval::new(1, 1).unwrap())
        .unwrap();
    let second_scores = [(token, 0), (negative, 0), (constant, 0)]
        .into_iter()
        .collect();
    let second = b
        .argmax_tree(&View {
            words: second_scores,
            shape: vec![3],
        })
        .unwrap();
    let mut outputs: Vec<_> = first.words.iter().chain(second.words.iter()).collect();
    if checked {
        let valid = b.validity.unwrap();
        let tokens = b.gather(&outputs.iter().copied().collect()).unwrap();
        let flags = b
            .gather(&[(valid, 0), (valid, 0)].into_iter().collect())
            .unwrap();
        let masked = b.product(tokens, flags, None, false).unwrap();
        outputs = vec![(masked, 0), (masked, 1), (valid, 0)];
    }
    (finish_fixture(b, 2, outputs), end)
}

fn finish_fixture(b: Builder, inputs: usize, outputs: Vec<(usize, usize)>) -> Arc<Program> {
    let work = schedule_work(&b.nodes).unwrap();
    let mut last: Vec<_> = (0..b.nodes.len()).collect();
    for (step, node) in b.nodes.iter().enumerate() {
        for id in node.inputs() {
            last[id] = step;
        }
    }
    for &(id, _) in &outputs {
        last[id] = b.nodes.len();
    }
    let mut retire = vec![vec![]; b.nodes.len()];
    for (id, step) in last.into_iter().enumerate() {
        if step < retire.len() {
            retire[step].push(id);
        }
    }
    Arc::new(Program {
        nodes: b.nodes,
        inputs,
        outputs,
        retire,
        digest: [219; 32],
        source_binding: [0; 32],
        payload_bytes: b.allocated,
        work,
    })
}

fn finish_body(mut bodies: [Invocation; 3]) -> [Worker; 3] {
    let mut queue = Vec::new();
    for _ in 0..1000 {
        if queue.is_empty() && bodies.iter().all(Invocation::complete) {
            break;
        }
        if queue.is_empty() {
            for body in &mut bodies {
                let from = body.party();
                queue.extend(
                    body.advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, raw)| (from, to, raw)),
                );
            }
        } else {
            for (from, to, raw) in std::mem::take(&mut queue) {
                queue.extend(
                    bodies[to as usize]
                        .receive(raw.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, raw)| (to, next, raw)),
                );
            }
        }
    }
    assert!(queue.is_empty());
    assert!(bodies.iter().all(Invocation::complete));
    bodies.map(|body| body.finish().unwrap())
}

#[test]
fn compiled_binary_handoff_restores_complete_state_and_two_private_selections() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for (x, expected) in [
                (vec![ONE, ONE, -ONE], vec![0, 2]),
                (vec![-ONE, ONE, ONE], vec![1, 0]),
                (vec![-ONE, 0, ONE], vec![2, 0]),
            ] {
                let (program, end) = fixture(false);
                let compiled = Compiled::new(program.clone(), WeightMode::Public, 2, end).unwrap();
                assert!(compiled.admitted_bytes() <= MAX_PROGRAM_BYTES);
                let clear = BTreeMap::from([("x".into(), x), ("y".into(), vec![ONE; 3])]);
                assert_eq!(execute(workers(program.clone(), &clear)), expected);
                let bodies = workers(program.clone(), &clear)
                    .map(|worker| compiled.clone().begin(worker).unwrap());
                let resumed = finish_body(bodies);
                for worker in &resumed {
                    let (step, started, offset, empty, idle, values) = restored_state(worker);
                    assert_eq!(step, end);
                    assert!(started && empty && idle);
                    assert_eq!(offset, 0);
                    for id in 0..end {
                        let retained = program.nodes[end..]
                            .iter()
                            .any(|node| node.inputs().contains(&id))
                            || program.outputs.iter().any(|&(owner, _)| owner == id);
                        assert_eq!(values[id], retained, "owner {id}");
                    }
                }
                assert_eq!(execute(resumed), expected);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn compiled_binary_handoff_rejects_wrong_graph_and_retires_protocol_failure() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let (program, end) = fixture(false);
            assert!(Compiled::new(program.clone(), WeightMode::Private, 2, end).is_err());
            assert!(Compiled::new(program.clone(), WeightMode::Public, 2, end + 1).is_err());
            let compiled = Compiled::new(program.clone(), WeightMode::Public, 2, end).unwrap();
            let clear = BTreeMap::from([("x".into(), vec![0; 3]), ("y".into(), vec![ONE; 3])]);
            let (other, _) = fixture(false);
            let [worker, _, _] = workers(other, &clear);
            assert!(compiled.clone().begin(worker).is_err());
            let [worker, _, _] = workers(program.clone(), &clear);
            let mut body = compiled.clone().begin(worker).unwrap();
            body.advance().unwrap();
            assert!(body.receive(&[], puma::Party::P1).is_err());
            assert!(matches!(body.advance(), Err(Error::Retired)));
            assert!(body.finish().is_err());
            let [worker, _, _] = workers(program, &clear);
            let mut body = compiled.begin(worker).unwrap();
            body.cancel();
            assert!(matches!(body.advance(), Err(Error::Retired)));
            assert!(body.finish().is_err());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn compiled_binary_handoff_preserves_sticky_rejection_and_final_masking() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for (x, expected) in [
                (vec![0, ONE / 2, -ONE / 2], vec![1, 0, 1]),
                (vec![ONE, 0, -ONE], vec![0, 0, 0]),
            ] {
                let (program, end) = fixture(true);
                let compiled = Compiled::new(program.clone(), WeightMode::Public, 2, end).unwrap();
                let clear = BTreeMap::from([("x".into(), x), ("y".into(), vec![ONE; 3])]);
                assert_eq!(execute(workers(program.clone(), &clear)), expected);
                let bodies =
                    workers(program, &clear).map(|worker| compiled.clone().begin(worker).unwrap());
                assert_eq!(execute(finish_body(bodies)), expected);
            }
            let (mut program, end) = fixture(false);
            Arc::get_mut(&mut program).unwrap().payload_bytes = MAX_PROGRAM_BYTES;
            let before = constructor_entries();
            assert!(matches!(
                Compiled::new(program, WeightMode::Public, 2, end),
                Err(Error::Bounds)
            ));
            // Fails if admission moves behind the first allocating constructor
            // region, even when a later exact validation still returns Bounds.
            assert_eq!(constructor_entries(), before);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn compiled_binary_preflight_prices_live_payload_before_any_constructor_entry() {
    let (mut program, end) = fixture(false);
    // Header still fits, but even fixed primitive workspace cannot coexist.
    Arc::get_mut(&mut program).unwrap().payload_bytes =
        MAX_PROGRAM_BYTES - puma::MAX_ADMITTED_BYTES;
    let before = constructor_entries();
    assert!(matches!(
        Compiled::new(program, WeightMode::Public, 2, end),
        Err(Error::Bounds)
    ));
    assert_eq!(constructor_entries(), before);
    let (program, end) = fixture(false);
    Compiled::new(program, WeightMode::Public, 2, end).unwrap();
    assert_eq!(constructor_entries(), before + 1); // Positive seam control.
}

// Layout/ownership regression, NOT fixed-seed PRSS parity: y is exactly one so
// integer expectations do not depend on stochastic rounding. Fixed entropy is
// private in pllm-core; exposing a test-only factory needs coordinator approval.
#[test]
fn compiled_binary_strides_broadcast_exports_and_ordinary_product_resume() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for n in [2, 3, 17, 128] {
                let mut b = Builder {
                    public_weights: true,
                    ..Builder::default()
                };
                let x = b
                    .push(
                        Code::Input("x".into()),
                        2 * n + 1,
                        Interval::new(-ONE, ONE).unwrap(),
                    )
                    .unwrap();
                let y = b
                    .push(Code::Input("y".into()), 1, Interval::new(ONE, ONE).unwrap())
                    .unwrap();
                let left: Words = (0..n).map(|i| (x, 1 + 2 * i)).collect();
                let right: Words = std::iter::repeat_n((y, 0), n).collect();
                let first = b.binary(&left, &right, true).unwrap();
                let exported_gather = 2;
                let shorter: Words = first.iter().take(n - 1).collect();
                let broadcast: Words = std::iter::repeat_n((y, 0), n - 1).collect();
                let second = b.binary(&shorter, &broadcast, true).unwrap();
                let end = b.nodes.len();
                // Subsequent protected product must consume the restored runtime,
                // not a fresh PRSS setup. Also uses first gather exported as owner.
                let retained: Words = (0..n - 1).map(|i| (exported_gather, i)).collect();
                let ordinary = b.binary(&retained, &broadcast, true).unwrap();
                let outputs: Vec<_> = second
                    .iter()
                    .chain(ordinary.iter())
                    .chain((0..n).map(|i| (exported_gather, i)))
                    .chain([(x, 0), (y, 0)])
                    .collect();
                let program = finish_fixture(b, 2, outputs);
                let compiled = Compiled::new(program.clone(), WeightMode::Public, 2, end).unwrap();
                let x_values: Vec<_> = (0..2 * n + 1)
                    .map(|i| (i as i64 % 5 - 2) * (ONE / 4))
                    .collect();
                let strided: Vec<_> = (0..n).map(|i| x_values[1 + 2 * i] as u64).collect();
                let expected: Vec<_> = strided[..n - 1]
                    .iter()
                    .copied()
                    .chain(strided[..n - 1].iter().copied())
                    .chain(strided.iter().copied())
                    .chain([x_values[0] as u64, ONE as u64])
                    .collect();
                let clear = BTreeMap::from([("x".into(), x_values), ("y".into(), vec![ONE])]);
                assert_eq!(execute(workers(program.clone(), &clear)), expected);
                let bodies =
                    workers(program, &clear).map(|worker| compiled.clone().begin(worker).unwrap());
                assert_eq!(execute(finish_body(bodies)), expected);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn compiled_binary_premature_finish_and_pending_drop_release_worker_owners() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let (program, end) = fixture(false);
            let compiled = Compiled::new(program.clone(), WeightMode::Public, 2, end).unwrap();
            let clear = BTreeMap::from([("x".into(), vec![0; 3]), ("y".into(), vec![ONE; 3])]);
            let [worker, _, _] = workers(program.clone(), &clear);
            let before = Arc::strong_count(&program);
            let body = compiled.clone().begin(worker).unwrap();
            assert!(matches!(body.finish(), Err(Error::Phase)));
            assert_eq!(Arc::strong_count(&program), before - 1);
            let [worker, _, _] = workers(program.clone(), &clear);
            let before = Arc::strong_count(&program);
            let mut body = compiled.clone().begin(worker).unwrap();
            assert!(!body.advance().unwrap().is_empty());
            drop(body);
            assert_eq!(Arc::strong_count(&program), before - 1);
            assert_eq!(Arc::strong_count(&compiled), 1);
            // Ownership/no-leak assertion, not a memory-erasure observation.
        })
        .unwrap()
        .join()
        .unwrap();
}

// Real Builder ledger throughout: no payload_bytes override. Repeated gathers
// deliberately reference the same external owner; preflight must price every
// occurrence, not silently substitute the deduplicated Plan input count.
fn pressure_fixture(owner_lanes: usize, width: usize, sites: usize) -> Arc<Program> {
    let mut b = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let x = b
        .push(
            Code::Input("x".into()),
            owner_lanes,
            Interval::new(0, ONE).unwrap(),
        )
        .unwrap();
    let words: Words = (0..width).map(|i| (x, i)).collect();
    let mut outputs = Vec::new();
    for _ in 0..sites {
        let left = b.gather(&words).unwrap();
        let right = b.gather(&words).unwrap();
        let sum = b.affine(left, Some(right), affine(1, 1, 0)).unwrap();
        // One output lane keeps each complete gather/result owner live. This
        // avoids constructing an unrelated giant output-index vector.
        outputs.extend([(left, 0), (right, 0), (sum, 0)]);
    }
    finish_fixture(b, 1, outputs)
}

#[test]
fn compiled_binary_real_program_pressure_rejects_before_lowering_and_accepts_controls() {
    for (owner_lanes, width, accepted_sites, rejected_sites) in [
        (MAX_VALUES, 1, 1, 2),
        (1, 1, 64, 2600),
        (65536, 65536, 2, 8),
    ] {
        let accepted = pressure_fixture(owner_lanes, width, accepted_sites);
        let before = constructor_entries();
        let end = accepted.nodes.len();
        let compiled = Compiled::new(accepted, WeightMode::Public, 1, end).unwrap();
        assert!(compiled.admitted_bytes() < MAX_PROGRAM_BYTES);
        assert_eq!(constructor_entries(), before + 1);
        drop(compiled);

        let rejected = pressure_fixture(owner_lanes, width, rejected_sites);
        assert!(rejected.payload_bytes() < MAX_PROGRAM_BYTES);
        let inputs = 2 * rejected_sites * owner_lanes * 16;
        let results = rejected_sites * width * 16;
        let exports = 3 * rejected_sites * width * 16;
        if owner_lanes == MAX_VALUES {
            assert!(rejected.payload_bytes() + inputs > MAX_PROGRAM_BYTES);
        } else if width == 1 {
            // 2 repeated inputs + 1 register + 3 nodes + 1 site per triple.
            assert!(4096 * (7 * rejected_sites + 16) > MAX_PROGRAM_BYTES);
        } else {
            let without_exports = rejected.payload_bytes() + inputs + results;
            assert!(without_exports < MAX_PROGRAM_BYTES);
            assert!(without_exports + exports > MAX_PROGRAM_BYTES);
        }
        let end = rejected.nodes.len();
        let before = constructor_entries();
        assert!(matches!(
            Compiled::new(rejected, WeightMode::Public, 1, end),
            Err(Error::Bounds)
        ));
        assert_eq!(constructor_entries(), before);
    }
}

#[test]
fn compiled_binary_rejects_constructible_nonbridge_operands_and_forms() {
    for case in 0..9 {
        let mut b = Builder {
            public_weights: true,
            ..Builder::default()
        };
        let bounds = Interval::new(0, ONE).unwrap();
        let x = b.push(Code::Input("x".into()), 129, bounds).unwrap();
        let y = b
            .push(
                if case == 3 {
                    Code::PublicInput {
                        key: "public-file".into(),
                        digest: [17; 32],
                        row_l1: None,
                        row_squared_norm: None,
                        row_l1_squared_sum: None,
                    }
                } else {
                    Code::Input("y".into())
                },
                129,
                bounds,
            )
            .unwrap();
        let indices: Words = match case {
            0 => [(x, 0), (y, 1), (x, 2)].into_iter().collect(), // Mixed owners.
            1 => [(x, 2), (x, 1), (x, 0)].into_iter().collect(), // Descending.
            2 => [(x, 0), (x, 1), (x, 3)].into_iter().collect(), // Nonaffine.
            4 => (0..129).map(|i| (x, i)).collect(),
            _ => [(x, 0), (x, 1), (x, 2)].into_iter().collect(),
        };
        let right: Words = (0..indices.len()).map(|i| (y, i)).collect();
        let left = b.gather(&indices).unwrap();
        let right = b.gather(&right).unwrap();
        let output = match case {
            4 => b.product(left, right, None, true).unwrap(), // Legal 129-wide primitive.
            5 => b.product(left, right, None, false).unwrap(), // Unscaled product.
            6 => b.affine(left, Some(right), affine(1, -1, 0)).unwrap(),
            7 => b.affine(left, None, affine(1, 0, 0)).unwrap(),
            8 => b
                .product(left, right, Some(MatrixShape::new(1, 3, 1).unwrap()), true)
                .unwrap(),
            _ => b.affine(left, Some(right), affine(1, 1, 0)).unwrap(),
        };
        let program = finish_fixture(b, 2, vec![(output, 0)]);
        let end = program.nodes.len();
        assert!(
            matches!(
                Compiled::new(program, WeightMode::Public, 2, end),
                Err(Error::Binding)
            ),
            "unsupported case {case}"
        );
    }
}

fn two_product_fixture(digest: u8) -> Arc<Program> {
    let mut b = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let x = b
        .push(Code::Input("x".into()), 3, Interval::new(0, ONE).unwrap())
        .unwrap();
    let words: Words = (0..3).map(|i| (x, i)).collect();
    let first = b.binary(&words, &words, true).unwrap();
    let second = b.binary(&first, &words, true).unwrap();
    let mut program = finish_fixture(b, 1, second.iter().collect());
    Arc::get_mut(&mut program).unwrap().digest = [digest; 32];
    program
}

// Copy transport bytes only, never clone/reset a live protocol/runtime owner.
type BodyFrame = (puma::Party, puma::Party, Vec<u8>);

fn advance_product(bodies: &mut [Invocation; 3]) -> Vec<BodyFrame> {
    let mut frames = Vec::new();
    for body in bodies {
        let from = body.party();
        frames.extend(
            body.advance()
                .unwrap()
                .into_iter()
                .map(|(to, raw)| (from, to, raw.as_bytes().to_vec())),
        );
    }
    use puma::{MessagePhase::Multiply, Party::*};
    assert_body_transcript(
        &frames,
        &[(P0, P2, Multiply), (P1, P0, Multiply), (P2, P1, Multiply)],
    );
    frames
}

fn assert_body_transcript(
    frames: &[BodyFrame],
    expected: &[(puma::Party, puma::Party, puma::MessagePhase)],
) {
    let actual: Vec<_> = frames
        .iter()
        .map(|(from, to, raw)| {
            let message = puma::Message::decode(raw, *from, *to).unwrap();
            let phase = message.header().phase;
            let words = if phase == puma::MessagePhase::Correlation {
                6
            } else {
                3
            };
            assert_eq!(message.words().len(), words);
            (*from, *to, phase)
        })
        .collect();
    assert_eq!(actual, expected);
}

// multiply::rotate starts protocol::begin only AFTER receiving NextRank's
// rotation. Evaluators emit one opening each; helper emits two corrections and
// finishes locally. The public pivot is OS-seeded, so derive roles from the
// correction sender, then check the entire ordered directed transcript.
fn assert_product_open_transcript(frames: &[BodyFrame]) -> (puma::Party, puma::Party) {
    use puma::{MessagePhase::*, Party::*};
    let helper = frames
        .iter()
        .find_map(|(from, to, raw)| {
            let message = puma::Message::decode(raw, *from, *to).unwrap();
            (message.header().phase == Correlation).then_some(*from)
        })
        .unwrap();
    let a = helper.next();
    let b = a.next();
    let mut expected = Vec::new();
    // advance_product emits P0->P2, P1->P0, P2->P1, in that order.
    for from in [P2, P0, P1] {
        if from == helper {
            expected.extend([(helper, a, Correlation), (helper, b, Correlation)]);
        } else {
            expected.push((from, if from == a { b } else { a }, MaskedInput));
        }
    }
    assert_body_transcript(frames, &expected);
    let binding = &frames[0].2[4..36];
    assert!(frames.iter().all(|(_, _, raw)| &raw[4..36] == binding));
    (a, b)
}

fn take_body_frame(
    frames: &mut Vec<BodyFrame>,
    to: puma::Party,
    phase: puma::MessagePhase,
) -> BodyFrame {
    let index = frames
        .iter()
        .position(|(from, recipient, raw)| {
            *recipient == to
                && puma::Message::decode(raw, *from, *recipient)
                    .unwrap()
                    .header()
                    .phase
                    == phase
        })
        .unwrap();
    frames.remove(index)
}

fn finish_product_open(bodies: &mut [Invocation; 3], mut frames: Vec<BodyFrame>) {
    use puma::MessagePhase::*;
    let (a, b) = assert_product_open_transcript(&frames);
    let binding = frames[0].2[4..36].to_vec();
    let mut reshares = Vec::new();
    for to in [a, b] {
        let opening = take_body_frame(&mut frames, to, MaskedInput);
        assert!(deliver_body_round(bodies, vec![opening]).is_empty());
        let correction = take_body_frame(&mut frames, to, Correlation);
        reshares.extend(deliver_body_round(bodies, vec![correction]));
    }
    assert!(frames.is_empty());
    assert_body_transcript(&reshares, &[(a, b, Reshare), (b, a, Reshare)]);
    assert!(reshares.iter().all(|(_, _, raw)| raw[4..36] == binding));
    assert!(deliver_body_round(bodies, reshares).is_empty());
    // receive never extracts the product or advances the body site cursor.
    assert!(bodies.iter().all(|body| !body.complete()));
}

fn deliver_body_round(bodies: &mut [Invocation; 3], frames: Vec<BodyFrame>) -> Vec<BodyFrame> {
    let mut next = Vec::new();
    for (from, to, raw) in frames {
        next.extend(
            bodies[to as usize]
                .receive(&raw, from)
                .unwrap()
                .into_iter()
                .map(|(recipient, raw)| (to, recipient, raw.as_bytes().to_vec())),
        );
    }
    next
}

#[test]
fn compiled_binary_late_wrong_context_duplicate_and_previous_product_replay_retire() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for attack in 0..3 {
                let program = two_product_fixture(219);
                let compiled =
                    Compiled::new(program.clone(), WeightMode::Public, 1, program.nodes.len())
                        .unwrap();
                let clear = BTreeMap::from([("x".into(), vec![ONE; 3])]);
                let mut bodies =
                    workers(program, &clear).map(|worker| compiled.clone().begin(worker).unwrap());
                let first = advance_product(&mut bodies);
                let replay = first[0].clone();
                let mut pending = deliver_body_round(&mut bodies, first);
                let (a, _) = assert_product_open_transcript(&pending);
                let (from, to, bad) = if attack == 1 {
                    // One genuine masked opening is buffered, but no correction
                    // has arrived. Repeating that same inbox slot must retire.
                    let frame = take_body_frame(&mut pending, a, puma::MessagePhase::MaskedInput);
                    assert!(deliver_body_round(&mut bodies, vec![frame.clone()]).is_empty());
                    frame
                } else {
                    // Finish first product, then enter second with the same live
                    // runtime/cursor. No generic round-count/drain assumption.
                    finish_product_open(&mut bodies, pending);
                    let mut second = advance_product(&mut bodies);
                    assert_eq!((second[0].0, second[0].1), (replay.0, replay.1));
                    assert_ne!(&second[0].2[4..36], &replay.2[4..36]);
                    if attack == 2 {
                        // Previously accepted first-product rotation is invalid
                        // even at the same directed phase in the second product.
                        replay
                    } else {
                        // Real well-formed frame from a different public graph
                        // context at the SAME phase, after a complete product.
                        // Another party also accepts its second rotation first.
                        // Rotation routes do not depend on the random pivot;
                        // foreign open/correction routes need not match ours.
                        // Independent fresh shares are NOT a parity comparison.
                        let progress = second.remove(1);
                        let _progress_frames = deliver_body_round(&mut bodies, vec![progress]);
                        let foreign = two_product_fixture(218);
                        let plan = Compiled::new(
                            foreign.clone(),
                            WeightMode::Public,
                            1,
                            foreign.nodes.len(),
                        )
                        .unwrap();
                        let mut foreign = workers(foreign, &clear)
                            .map(|worker| plan.clone().begin(worker).unwrap());
                        let foreign_first = advance_product(&mut foreign);
                        let foreign_open = deliver_body_round(&mut foreign, foreign_first);
                        finish_product_open(&mut foreign, foreign_open);
                        let foreign_second = advance_product(&mut foreign);
                        let (sender, recipient, expected) = &second[0];
                        let frame = foreign_second
                            .into_iter()
                            .find(|(a, b, _)| a == sender && b == recipient)
                            .unwrap();
                        assert_eq!(frame.2[38], expected[38]);
                        assert_ne!(&frame.2[4..36], &expected[4..36]);
                        frame
                    }
                };
                let body = &mut bodies[to as usize];
                assert!(
                    matches!(body.receive(&bad, from), Err(Error::Binding)),
                    "attack {attack}"
                );
                assert!(!body.complete());
                assert!(matches!(body.advance(), Err(Error::Retired)));
                assert!(matches!(body.receive(&bad, from), Err(Error::Retired)));
                assert!(matches!(body.advance(), Err(Error::Retired)));
                let body = bodies.into_iter().nth(to as usize).unwrap();
                assert!(matches!(body.finish(), Err(Error::Retired)));
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

// Still BLOCKED: fixed-seed ordered transcript/local-share parity needs an
// approved core test-support entropy seam; fresh workers cannot establish it.
// Expanded oracle capture also remains BLOCKED on compact-boundary mapping.
