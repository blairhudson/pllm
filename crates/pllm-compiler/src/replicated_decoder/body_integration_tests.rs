//! Include as a child of program.rs; see body_integration_handoff.md.
//! Generated graph, not a checkpoint or full-model qualification.
use super::*;
use crate::replicated_decoder::{
    runtime::body::integration::{
        constructor_entries, restored_state, BoundaryProgress, Compiled, Invocation,
    },
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
// supplied by the explicit insecure-test-support factory used below.
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

// Globally drain ordinary frames before asking any original runtime to advance.
// Reached is local: never enter a body merely because one role reported it.
fn reach_boundary(
    workers: &mut [Worker; 3],
    target: usize,
    compiled: Option<&Compiled>,
) -> Vec<BodyFrame> {
    let mut queue = Vec::<BodyFrame>::new();
    let mut transcript = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() {
            let mut reached = 0;
            for worker in workers.iter_mut() {
                let from = worker.party();
                let progress = match compiled {
                    Some(compiled) => compiled.advance_to_start(worker),
                    None => worker.advance_before(target),
                }
                .unwrap();
                match progress {
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
                for worker in workers.iter_mut() {
                    let state = restored_state(worker);
                    assert_eq!(state.0, target);
                    assert_eq!(state.2, 0);
                    assert!(state.3 && state.4);
                    // Same boundary is idempotent while idle, without PRSS work.
                    let before = boundary_snapshot(worker);
                    assert!(before.shares[target..].iter().all(Option::is_none));
                    assert!(matches!(
                        worker.advance_before(target).unwrap(),
                        BoundaryProgress::Reached
                    ));
                    assert_eq!(boundary_snapshot(worker), before);
                }
                return transcript;
            }
        } else {
            for (from, to, raw) in std::mem::take(&mut queue) {
                transcript.push((from, to, raw.clone()));
                queue.extend(
                    workers[to as usize]
                        .receive(&raw, from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, raw)| (to, next, raw.as_bytes().to_vec())),
                );
            }
        }
    }
    panic!("prefix failed to reach globally drained boundary");
}

// Prefix classes exercise local retirement, ordinary product, Tensor, and a
// 128+1 TensorLoop. All body/suffix products multiply by exact Q18 one.
fn midgraph_fixture(kind: usize, separated: bool) -> (Arc<Program>, [(usize, usize); 2]) {
    let mut b = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let lanes = if kind == 3 { 129 } else { 3 };
    let bounds = Interval::new(-ONE, ONE).unwrap();
    let x = b.push(Code::Input("x".into()), lanes, bounds).unwrap();
    let y = b
        .push(Code::Input("y".into()), 3, Interval::new(ONE, ONE).unwrap())
        .unwrap();
    let prefix = match kind {
        0 => b.affine(x, None, affine(1, 0, 0)).unwrap(),
        1 => b.product(x, y, None, true).unwrap(),
        2 => b
            .tensor(x, tensor::Program::negative(lanes, bounds).unwrap())
            .unwrap(),
        3 => b
            .tensor_loop(LoopInput::Slice(x), lanes, |n| {
                tensor::Program::negative(n, bounds)
            })
            .unwrap(),
        _ => unreachable!(),
    };
    let left: Words = (0..3).map(|i| (prefix, i)).collect();
    let right: Words = (0..3).map(|i| (y, i)).collect();
    let start = b.nodes.len();
    let first = b.binary(&left, &right, true).unwrap();
    let end = b.nodes.len();
    let next = if separated {
        let owner = b.gather(&first).unwrap();
        let owner = b.affine(owner, None, affine(1, 0, 0)).unwrap();
        (0..3).map(|i| (owner, i)).collect()
    } else {
        first.clone()
    };
    let second_start = b.nodes.len();
    let second = b.binary(&next, &right, true).unwrap();
    let second_end = b.nodes.len();
    let suffix = b.binary(&second, &right, true).unwrap();
    // Preserve complete prefix owner (including TensorLoop tail), a body gather,
    // external inputs, and first result across both seams.
    let outputs = suffix
        .iter()
        .chain(first.iter())
        .chain((0..lanes).map(|i| (prefix, i)))
        .chain((0..3).map(|i| (start, i)))
        .chain([(x, lanes - 1), (y, 0)])
        .collect();
    (
        finish_fixture(b, 2, outputs),
        [(start, end), (second_start, second_end)],
    )
}

fn midgraph_clear(kind: usize) -> BTreeMap<String, Vec<i64>> {
    let lanes = if kind == 3 { 129 } else { 3 };
    BTreeMap::from([
        (
            "x".into(),
            (0..lanes).map(|i| (i as i64 % 3 - 1) * ONE).collect(),
        ),
        ("y".into(), vec![ONE; 3]),
    ])
}

#[test]
fn compiled_midgraph_prefix_body_suffix_and_adjacent_or_separate_intervals() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for kind in 0..4 {
                for separated in [false, true] {
                    let (program, spans) = midgraph_fixture(kind, separated);
                    let clear = midgraph_clear(kind);
                    let prefix: Vec<u64> = clear["x"]
                        .iter()
                        .map(|&x| {
                            if kind >= 2 {
                                u64::from(x < 0)
                            } else {
                                x as u64
                            }
                        })
                        .collect();
                    let expected: Vec<_> = prefix[..3]
                        .iter()
                        .copied()
                        .chain(prefix[..3].iter().copied())
                        .chain(prefix.iter().copied())
                        .chain(prefix[..3].iter().copied())
                        .chain([*clear["x"].last().unwrap() as u64, ONE as u64])
                        .collect();
                    assert_eq!(execute(workers(program.clone(), &clear)), expected);
                    let mut ordinary = workers(program.clone(), &clear);
                    let mut resumed = workers(program.clone(), &clear);
                    for (start, end) in spans {
                        let compiled =
                            Compiled::new(program.clone(), WeightMode::Public, start, end).unwrap();
                        assert!(compiled.admitted_bytes() <= MAX_PROGRAM_BYTES);
                        reach_boundary(&mut resumed, start, Some(&compiled));
                        // Neither target gather nor any later body owner exists.
                        for worker in &resumed {
                            assert!(restored_state(worker).5[start..].iter().all(|&v| !v));
                        }
                        resumed = finish_body(
                            resumed.map(|worker| compiled.clone().begin(worker).unwrap()),
                        );
                        reach_boundary(&mut ordinary, end, None);
                        for (body, original) in resumed.iter().zip(&ordinary) {
                            assert_eq!(restored_state(body), restored_state(original));
                        }
                    }
                    assert_eq!(execute(resumed), expected);
                    assert_eq!(execute(ordinary), expected);
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn bounded_boundary_stops_before_unavailable_source_without_executing_it() {
    let mut b = Builder::default();
    let input = b
        .push(Code::Input("x".into()), 1, Interval::new(0, ONE).unwrap())
        .unwrap();
    let local = b.affine(input, None, affine(1, 0, 0)).unwrap();
    let source = b
        .push(
            Code::PublicInput {
                key: "unavailable".into(),
                digest: [17; 32],
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            1,
            Interval::new(0, ONE).unwrap(),
        )
        .unwrap();
    let target = b.nodes.len();
    let read = b.gather(&[(source, 0)].into_iter().collect()).unwrap();
    let program = finish_fixture(b, 1, vec![(local, 0), (read, 0)]);
    // Negative execution sentinel, not a valid source-admission fixture: late
    // public declaration has no installed file. Executing target must fail.
    let clear = BTreeMap::from([("x".into(), vec![ONE])]);
    let mut roles = workers(program, &clear);
    reach_boundary(&mut roles, target, None);
    for role in &roles {
        assert!(!restored_state(role).5[target]);
    }
    assert!(matches!(roles[0].advance(), Err(Error::Binding)));
}

#[test]
fn bounded_boundary_invalid_requests_cancel_and_release_original_owners() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(invalid_boundary_requests)
        .unwrap()
        .join()
        .unwrap();
}

fn invalid_boundary_requests() {
    for case in 0..4 {
        let (program, spans) = midgraph_fixture(1, false);
        let compiled =
            Compiled::new(program.clone(), WeightMode::Public, spans[0].0, spans[0].1).unwrap();
        let clear = midgraph_clear(1);
        let mut roles = workers(program.clone(), &clear);
        match case {
            0 => {
                reach_boundary(&mut roles, spans[0].0, Some(&compiled));
                assert!(matches!(roles[0].advance_before(0), Err(Error::Bounds)));
            }
            1 => assert!(matches!(
                roles[0].advance_before(program.nodes.len()),
                Err(Error::Bounds)
            )),
            2 => {
                // Input declarations retire, then ordinary product becomes active.
                assert!(!roles[0].advance().unwrap().is_empty());
                assert!(matches!(roles[0].test_boundary_state(), Err(Error::Phase)));
                assert!(!roles[0].retired()); // Read-only diagnostic rejection.
                assert!(matches!(roles[0].advance_before(2), Err(Error::Phase)));
            }
            3 => {
                let (foreign, _) = midgraph_fixture(1, false);
                let [mut worker, _, _] = workers(foreign, &clear);
                assert!(matches!(
                    compiled.advance_to_start(&mut worker),
                    Err(Error::Binding)
                ));
                assert!(matches!(worker.advance(), Err(Error::Retired)));
                roles[0].cancel();
            }
            _ => unreachable!(),
        }
        assert!(matches!(roles[0].advance(), Err(Error::Retired)));
        assert!(matches!(
            roles[0].test_boundary_state(),
            Err(Error::Retired)
        ));
        assert!(matches!(
            roles[0].advance_before(spans[0].0),
            Err(Error::Retired)
        ));
        drop(roles);
        assert_eq!(Arc::strong_count(&program), 2); // Caller + compiled, no worker leak.
    }
}

#[test]
fn compiled_midgraph_sticky_prefix_rejection_survives_body_and_later_safe_check() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for rejected in [false, true] {
                let mut b = Builder {
                    public_weights: true,
                    checked_bound: Some(ONE / 2),
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
                let words = (0..3).map(|i| (x, i)).collect();
                let safe = b.checked_view(&words).unwrap();
                let valid_before = b.validity.unwrap();
                let start = b.nodes.len();
                let right = (0..3).map(|i| (y, i)).collect();
                let body = b.binary(&safe, &right, true).unwrap();
                let end = b.nodes.len();
                // Force a second check with a wider PUBLIC incoming envelope;
                // runtime safe values pass, but sticky prefix failure must remain.
                let gathered = b.gather(&body).unwrap();
                let widened = b
                    .push(
                        Code::Gather((0..3).map(|i| (gathered, i)).collect()),
                        3,
                        Interval::new(-ONE, ONE).unwrap(),
                    )
                    .unwrap();
                let checked = b
                    .checked_view(&(0..3).map(|i| (widened, i)).collect())
                    .unwrap();
                let valid = b.validity.unwrap();
                let values = b.gather(&checked).unwrap();
                let flags = b
                    .gather(&std::iter::repeat_n((valid, 0), 3).collect())
                    .unwrap();
                let masked = b.product(values, flags, None, false).unwrap();
                let program = finish_fixture(
                    b,
                    2,
                    vec![
                        (masked, 0),
                        (masked, 1),
                        (masked, 2),
                        (valid, 0),
                        (valid_before, 0),
                    ],
                );
                let input = if rejected { ONE } else { ONE / 2 };
                let clear = BTreeMap::from([
                    ("x".into(), vec![input, 0, -ONE / 2]),
                    ("y".into(), vec![ONE; 3]),
                ]);
                let expected = if rejected {
                    vec![0; 5]
                } else {
                    vec![(ONE / 2) as u64, 0, (-ONE / 2) as u64, 1, 1]
                };
                assert_eq!(execute(workers(program.clone(), &clear)), expected);
                let compiled =
                    Compiled::new(program.clone(), WeightMode::Public, start, end).unwrap();
                let mut roles = workers(program.clone(), &clear);
                reach_boundary(&mut roles, start, Some(&compiled));
                for role in &roles {
                    assert!(restored_state(role).5[valid_before]);
                }
                let resumed = finish_body(roles.map(|role| compiled.clone().begin(role).unwrap()));
                for role in &resumed {
                    assert!(restored_state(role).5[valid_before]);
                }
                assert_eq!(execute(resumed), expected);
                // Also compare sticky success/failure under identical PUBLIC
                // deterministic setup, with independent ordinary/body routers.
                let (expected_wire, expected_shares, expected_state) =
                    trace_ordinary_finish(deterministic_workers(program.clone(), &clear));
                let mut reference = deterministic_workers(program.clone(), &clear);
                let mut roles = deterministic_workers(program, &clear);
                let mut wire = reach_boundary(&mut roles, start, Some(&compiled));
                reach_boundary(&mut reference, start, None);
                assert_boundary_parity(&roles, &reference);
                let (body_wire, resumed) =
                    trace_body_finish(roles.map(|role| compiled.clone().begin(role).unwrap()));
                wire.extend(body_wire);
                reach_boundary(&mut reference, end, None);
                assert_boundary_parity(&resumed, &reference);
                let (suffix_wire, shares, state) = trace_ordinary_finish(resumed);
                wire.extend(suffix_wire);
                assert_eq!(wire, expected_wire, "sticky rejected={rejected}");
                assert_eq!(shares, expected_shares, "sticky rejected={rejected}");
                assert_eq!(state, expected_state, "sticky rejected={rejected}");
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn bounded_boundary_pending_prefix_inbox_and_midgraph_drop_cancel() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for handoff in [false, true] {
                let (program, spans) = midgraph_fixture(1, false);
                let compiled =
                    Compiled::new(program.clone(), WeightMode::Public, spans[0].0, spans[0].1)
                        .unwrap();
                let mut roles = workers(program.clone(), &midgraph_clear(1));
                let mut rotations = Vec::new();
                for role in &mut roles {
                    let from = role.party();
                    let BoundaryProgress::Frames(frames) = compiled.advance_to_start(role).unwrap()
                    else {
                        panic!("product prefix must emit rotations before boundary");
                    };
                    rotations.extend(
                        frames
                            .into_iter()
                            .map(|(to, raw)| (from, to, raw.as_bytes().to_vec())),
                    );
                }
                let mut openings = Vec::new();
                for (from, to, raw) in rotations {
                    openings.extend(
                        roles[to as usize]
                            .receive(&raw, from)
                            .unwrap()
                            .into_iter()
                            .map(|(next, raw)| (to, next, raw.as_bytes().to_vec())),
                    );
                }
                let (from, to, raw) = openings
                    .into_iter()
                    .find(|(from, to, raw)| {
                        puma::Message::decode(raw, *from, *to)
                            .unwrap()
                            .header()
                            .phase
                            == puma::MessagePhase::MaskedInput
                    })
                    .unwrap();
                assert!(roles[to as usize].receive(&raw, from).unwrap().is_empty());
                assert!(!restored_state(&roles[to as usize]).3);
                assert!(matches!(
                    roles[to as usize].test_boundary_state(),
                    Err(Error::Phase)
                ));
                assert!(!roles[to as usize].retired());
                if handoff {
                    let role = roles.into_iter().nth(to as usize).unwrap();
                    assert!(matches!(compiled.clone().begin(role), Err(Error::Binding)));
                } else {
                    assert!(matches!(
                        compiled.advance_to_start(&mut roles[to as usize]),
                        Err(Error::Phase)
                    ));
                    assert!(matches!(roles[to as usize].advance(), Err(Error::Retired)));
                    drop(roles);
                }
                assert_eq!(Arc::strong_count(&program), 2);
            }
            for cancel in [false, true] {
                let (program, spans) = midgraph_fixture(1, false);
                let compiled =
                    Compiled::new(program.clone(), WeightMode::Public, spans[0].0, spans[0].1)
                        .unwrap();
                let mut roles = workers(program.clone(), &midgraph_clear(1));
                reach_boundary(&mut roles, spans[0].0, Some(&compiled));
                let [role, peer_one, peer_two] = roles;
                // Ignored array fields otherwise stay in the partially moved
                // array until scope exit, retaining their original Program Arcs.
                drop((peer_one, peer_two));
                let mut body = compiled.clone().begin(role).unwrap();
                assert!(!body.advance().unwrap().is_empty());
                if cancel {
                    body.cancel();
                    assert!(matches!(body.advance(), Err(Error::Retired)));
                    assert!(matches!(body.finish(), Err(Error::Retired)));
                } else {
                    drop(body);
                }
                assert_eq!(Arc::strong_count(&program), 2);
                assert_eq!(Arc::strong_count(&compiled), 1);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn compiled_midgraph_preflight_keeps_live_prefix_payload_in_original_cap() {
    for (lanes, accepted) in [(MAX_VALUES / 4, true), (MAX_VALUES, false)] {
        let mut b = Builder {
            public_weights: true,
            ..Builder::default()
        };
        let x = b
            .push(
                Code::Input("x".into()),
                lanes,
                Interval::new(0, ONE).unwrap(),
            )
            .unwrap();
        let prefix = b.affine(x, None, affine(1, 0, 0)).unwrap();
        let start = b.nodes.len();
        let words = [(prefix, 0)].into_iter().collect();
        let result = b.binary(&words, &words, false).unwrap();
        let end = b.nodes.len();
        let program = finish_fixture(
            b,
            1,
            vec![(x, 0), (prefix, lanes - 1), result.iter().next().unwrap()],
        );
        assert!(program.payload_bytes() < MAX_PROGRAM_BYTES);
        if !accepted {
            // Two repeated body inputs alone still fit. The original input
            // and live prefix owner must coexist with both copies: their real
            // Builder ledger, not an invented payload override, crosses the cap.
            let repeated_input_bytes = 2 * lanes * 16;
            assert!(repeated_input_bytes < MAX_PROGRAM_BYTES);
            assert!(program.payload_bytes() + repeated_input_bytes >= MAX_PROGRAM_BYTES);
        }
        let before = constructor_entries();
        let result = Compiled::new(program, WeightMode::Public, start, end);
        if accepted {
            assert!(result.unwrap().admitted_bytes() <= MAX_PROGRAM_BYTES);
            assert_eq!(constructor_entries(), before + 1);
        } else {
            assert!(matches!(result, Err(Error::Bounds)));
            assert_eq!(constructor_entries(), before);
        }
    }
}

#[test]
fn compiled_midgraph_body_rotation_replay_retires_ordinary_suffix() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let (program, spans) = midgraph_fixture(1, false);
            let compiled =
                Compiled::new(program.clone(), WeightMode::Public, spans[0].0, spans[1].1).unwrap();
            let mut roles = workers(program, &midgraph_clear(1));
            reach_boundary(&mut roles, spans[0].0, Some(&compiled));
            let mut bodies = roles.map(|role| compiled.clone().begin(role).unwrap());
            let first = advance_product(&mut bodies);
            let (from, to, replay) = first[0].clone();
            let openings = deliver_body_round(&mut bodies, first);
            finish_product_open(&mut bodies, openings);
            let mut resumed = finish_body(bodies);
            for role in &mut resumed {
                assert!(!role.advance().unwrap().is_empty());
            }
            let role = &mut resumed[to as usize];
            assert!(matches!(role.receive(&replay, from), Err(Error::Binding)));
            assert!(matches!(role.advance(), Err(Error::Retired)));
            assert!(matches!(role.receive(&replay, from), Err(Error::Retired)));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn compiled_midgraph_prefix_rotation_replay_retires_body() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            let (program, spans) = midgraph_fixture(1, false);
            let compiled =
                Compiled::new(program.clone(), WeightMode::Public, spans[0].0, spans[0].1).unwrap();
            let mut roles = workers(program, &midgraph_clear(1));
            let prefix = reach_boundary(&mut roles, spans[0].0, Some(&compiled));
            let (from, to, replay) = prefix
                .into_iter()
                .find(|(from, to, raw)| {
                    puma::Message::decode(raw, *from, *to)
                        .unwrap()
                        .header()
                        .phase
                        == puma::MessagePhase::Multiply
                })
                .unwrap();
            let mut bodies = roles.map(|worker| compiled.clone().begin(worker).unwrap());
            advance_product(&mut bodies);
            let body = &mut bodies[to as usize];
            assert!(matches!(body.receive(&replay, from), Err(Error::Binding)));
            assert!(matches!(body.advance(), Err(Error::Retired)));
            assert!(matches!(body.receive(&replay, from), Err(Error::Retired)));
            let body = bodies.into_iter().nth(to as usize).unwrap();
            assert!(matches!(body.finish(), Err(Error::Retired)));
        })
        .unwrap()
        .join()
        .unwrap();
}

// PUBLIC test keys only. Each call creates fresh consuming runtime owners; equal
// context bindings are intentional. Never mix these graphs' separate routers.
fn deterministic_workers(program: Arc<Program>, clear: &BTreeMap<String, Vec<i64>>) -> [Worker; 3] {
    let context = puma::Context::new([42; 32], program.digest(), 0).unwrap();
    let runtimes = puma::prss::insecure_test_support::fresh_compound_parties(
        context,
        puma::prss::PrssLimits::default(),
    )
    .unwrap();
    let mut roles =
        runtimes.map(|runtime| Worker::new(program.clone(), runtime, [42; 32]).unwrap());
    // Same ordinary input splitting as tests::workers; no masked table oracle.
    for (key, n) in program.inputs() {
        let first: Vec<_> = (0..n)
            .map(|i| 0xc751f00d12345678u64.wrapping_add(i as u64 * 17))
            .collect();
        let second: Vec<_> = (0..n)
            .map(|i| 0x254277811928f93du64.wrapping_add(i as u64 * 311))
            .collect();
        let third: Vec<_> = clear[key]
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

// Only borrowed local components are copied, into zeroizing test owners. Never
// copy or reset the consuming runtime or its source schedule.
#[derive(Debug, PartialEq, Eq)]
struct BoundarySnapshot {
    step: usize,
    cursors: puma::prss::Cursors,
    shares: Vec<Option<(zeroize::Zeroizing<Vec<u64>>, zeroize::Zeroizing<Vec<u64>>)>>,
}

fn boundary_snapshot(worker: &Worker) -> BoundarySnapshot {
    let (step, cursors, shares) = worker.test_boundary_state().unwrap();
    BoundarySnapshot {
        step,
        cursors,
        shares: shares
            .map(|value| {
                value.map(|(first, second)| {
                    (
                        zeroize::Zeroizing::new(first.to_vec()),
                        zeroize::Zeroizing::new(second.to_vec()),
                    )
                })
            })
            .collect(),
    }
}

fn assert_boundary_parity(actual: &[Worker; 3], expected: &[Worker; 3]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.party(), expected.party());
        assert_eq!(boundary_snapshot(actual), boundary_snapshot(expected));
    }
}

// Ordinary baseline uses ONLY default advance, including across every proposed
// seam. Candidate suffix uses this same driver. Record all bytes in delivery
// order, preserving complete outer tensor frames and every direction/phase.
fn trace_ordinary_finish(
    mut roles: [Worker; 3],
) -> (Vec<BodyFrame>, [Vec<u64>; 3], [BoundarySnapshot; 3]) {
    let mut queue = Vec::<BodyFrame>::new();
    let mut transcript = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() {
            if roles.iter().all(Worker::complete) {
                let state = roles.each_ref().map(boundary_snapshot);
                let shares = roles.map(|role| role.finish().unwrap().to_vec());
                return (transcript, shares, state);
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
            for (from, to, raw) in std::mem::take(&mut queue) {
                transcript.push((from, to, raw.clone()));
                queue.extend(
                    roles[to as usize]
                        .receive(&raw, from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, raw)| (to, next, raw.as_bytes().to_vec())),
                );
            }
        }
    }
    panic!("ordinary transcript did not finish");
}

fn trace_body_finish(mut bodies: [Invocation; 3]) -> (Vec<BodyFrame>, [Worker; 3]) {
    let mut queue = Vec::<BodyFrame>::new();
    let mut transcript = Vec::new();
    for _ in 0..100_000 {
        if queue.is_empty() {
            if bodies.iter().all(Invocation::complete) {
                return (transcript, bodies.map(|body| body.finish().unwrap()));
            }
            for body in &mut bodies {
                let from = body.party();
                queue.extend(
                    body.advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, raw)| (from, to, raw.as_bytes().to_vec())),
                );
            }
        } else {
            for (from, to, raw) in std::mem::take(&mut queue) {
                transcript.push((from, to, raw.clone()));
                queue.extend(
                    bodies[to as usize]
                        .receive(&raw, from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, raw)| (to, next, raw.as_bytes().to_vec())),
                );
            }
        }
    }
    panic!("body transcript did not finish");
}

#[test]
fn compiled_midgraph_exact_deterministic_full_wire_and_returned_owner_share_parity() {
    std::thread::Builder::new()
        .stack_size(16 << 20)
        .spawn(|| {
            for kind in 0..4 {
                for separated in [false, true] {
                    let (mut program, spans) = midgraph_fixture(kind, separated);
                    // Ordinary final outputs retain every lane of existing live
                    // owners and both returned body results for exact comparison.
                    // Not expanded oracle capture or intermediate reconstruction.
                    let graph = Arc::get_mut(&mut program).unwrap();
                    let mut owners: Vec<_> = graph.outputs.iter().map(|&(id, _)| id).collect();
                    owners.push(spans[1].1 - 1);
                    owners.sort_unstable();
                    owners.dedup();
                    graph.outputs = owners
                        .iter()
                        .flat_map(|&id| (0..graph.nodes[id].lanes).map(move |lane| (id, lane)))
                        .collect();
                    for retired in &mut graph.retire {
                        retired.retain(|id| !owners.contains(id));
                    }
                    if kind == 3 {
                        let Code::TensorLoop {
                            program: tile,
                            tail,
                            ..
                        } = &graph.nodes[2].code
                        else {
                            panic!("expected original tiled prefix");
                        };
                        assert_eq!(tile.lanes(), 128);
                        assert_eq!(tail.as_ref().unwrap().lanes(), 1);
                    }
                    let clear = midgraph_clear(kind);
                    // Completely separate owners/routers, not cloned live state.
                    let baseline = deterministic_workers(program.clone(), &clear);
                    let mut candidate = deterministic_workers(program.clone(), &clear);
                    let mut reference = deterministic_workers(program.clone(), &clear);
                    let (expected_wire, expected_shares, expected_state) =
                        trace_ordinary_finish(baseline);
                    let mut actual_wire = Vec::new();
                    for (start, end) in spans {
                        let compiled =
                            Compiled::new(program.clone(), WeightMode::Public, start, end).unwrap();
                        let prefix = reach_boundary(&mut candidate, start, Some(&compiled));
                        reach_boundary(&mut reference, start, None);
                        assert_boundary_parity(&candidate, &reference);
                        if kind != 0 && actual_wire.is_empty() {
                            assert!(!prefix.is_empty());
                        }
                        actual_wire.extend(prefix);
                        let bodies = candidate.map(|role| compiled.clone().begin(role).unwrap());
                        let (wire, resumed) = trace_body_finish(bodies);
                        assert!(!wire.is_empty());
                        actual_wire.extend(wire);
                        candidate = resumed;
                        reach_boundary(&mut reference, end, None);
                        assert_boundary_parity(&candidate, &reference);
                        // Compare exact prefix through this body seam, without
                        // stripping framing, bindings, headers, payload or order.
                        assert_eq!(
                            actual_wire.as_slice(),
                            &expected_wire[..actual_wire.len()],
                            "kind={kind}, separated={separated}, end={end}"
                        );
                    }
                    let before_suffix = actual_wire.len();
                    let (suffix, actual_shares, actual_state) = trace_ordinary_finish(candidate);
                    assert!(!suffix.is_empty()); // Subsequent ordinary protected product.
                    actual_wire.extend(suffix);
                    assert!(actual_wire.len() > before_suffix);
                    assert_eq!(
                        actual_wire, expected_wire,
                        "kind={kind}, separated={separated}"
                    );
                    assert_eq!(
                        actual_shares, expected_shares,
                        "kind={kind}, separated={separated}"
                    );
                    assert_eq!(
                        actual_state, expected_state,
                        "final live owners/cursors: kind={kind}, separated={separated}"
                    );
                    // State equality includes both actual stored components of
                    // EVERY live owner and all four cursor counters/AES work.
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

// Expanded oracle capture also remains BLOCKED on compact-boundary mapping.
