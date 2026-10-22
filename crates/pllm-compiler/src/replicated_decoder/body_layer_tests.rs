use super::super::*;
use crate::replicated_decoder::{
    runtime::body::{self, layer},
    tests::workers,
};
use sha2::Digest;

fn fixture(plan: &Arc<layer::Plan>, inputs: &[body::Register]) -> Arc<Program> {
    let outputs = [body::View::contiguous(0, inputs[0].lanes)];
    let mut program = super::expanded(inputs, &[], &outputs);
    let program_mut = Arc::get_mut(&mut program).unwrap();
    program_mut.work = plan.work_budget();
    program_mut.digest = plan.digest();
    program
}

type Signature = Vec<Vec<(Vec<u8>, usize)>>;
fn signature(queue: &[(puma::Party, puma::Party, PeerBytes)]) -> Vec<(Vec<u8>, usize)> {
    let mut out: Vec<_> = queue
        .iter()
        .map(|(_, _, frame)| {
            let raw = frame.as_bytes();
            let len = if raw.starts_with(b"PUT") { 48 } else { 4 };
            (raw[..len].to_vec(), raw.len())
        })
        .collect();
    out.sort();
    out
}
fn run(mut owners: [layer::Executor; 3], plan: &Arc<layer::Plan>) -> (Vec<Vec<u64>>, Signature) {
    let mut queue = Vec::new();
    let mut trace = Vec::new();
    let mut advances = 0u64;
    for _ in 0..500_000 {
        if owners.iter().all(layer::Executor::complete) {
            break;
        }
        if queue.is_empty() {
            advances += 1;
            for (i, owner) in owners.iter_mut().enumerate() {
                queue.extend(owner.advance().unwrap().into_iter().map(|(to, frame)| {
                    (
                        [puma::Party::P0, puma::Party::P1, puma::Party::P2][i],
                        to,
                        frame,
                    )
                }));
            }
        } else {
            trace.push(signature(&queue));
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, frame)| (to, next, frame)),
                );
            }
        }
        for owner in &owners {
            assert!(owner.live_bytes() <= plan.resources().peak_live_secret_bytes);
        }
    }
    assert!(queue.is_empty());
    assert!(owners.iter().all(layer::Executor::complete));
    assert_eq!(advances, plan.coordinator_advances());
    for owner in &owners {
        assert_eq!(
            owner.observed_register_peak(),
            plan.resources().peak_live_secret_bytes
        );
    }
    let finished = owners.map(|owner| owner.finish().unwrap());
    let budget = plan.work_budget().cursors();
    for (runtime, _) in &finished {
        let actual = runtime.cursors();
        assert_eq!(actual.public_blocks, budget.public_blocks);
        assert_eq!(actual.pair_blocks, budget.pair_blocks);
        assert!(actual.private_blocks <= budget.private_blocks);
        assert!(actual.aes_work <= budget.aes_work);
    }
    let outputs = (0..plan.output_count())
        .map(|output| {
            (0..finished[0].1.output_len(output).unwrap())
                .map(|index| {
                    finished.iter().fold(0u64, |sum, (_, values)| {
                        sum.wrapping_add(values.word(output, index).unwrap().0)
                    })
                })
                .collect()
        })
        .collect();
    (outputs, trace)
}

#[test]
fn body_layer_complete_guards_sticky_tail_failure_and_elision() {
    for failing in [false, true] {
        let inputs = [body::Register {
            lanes: 129,
            bounds: Interval::new(-4 * ONE, 4 * ONE).unwrap(),
        }];
        let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
        let x = b.input(0).unwrap();
        let (safe, first) = b.checked(&x, ONE, None).unwrap();
        let (safe2, sticky) = b.checked(&x, ONE, first.as_ref()).unwrap();
        let (alias, sticky2) = b.checked(&safe2, ONE, sticky.as_ref()).unwrap();
        assert_eq!(alias.len(), 129);
        let plan = b.finish(&[safe, alias, sticky2.unwrap()]).unwrap();
        let mut x = vec![0i64; 129];
        if failing {
            x[128] = 2 * ONE;
        }
        let clear = BTreeMap::from([("x0".into(), x)]);
        let program = fixture(&plan, &inputs);
        let owners =
            workers(program, &clear).map(|worker| layer::from_worker(plan.clone(), worker, None));
        let (output, _) = run(owners, &plan);
        assert_eq!(output[0], vec![0u64; 129]);
        assert_eq!(output[1], vec![0u64; 129]);
        assert_eq!(output[2], vec![u64::from(!failing)]);
    }
}

#[test]
fn body_layer_rank_four_permutation_and_concat_are_owner_retaining() {
    let inputs = [body::Register {
        lanes: 24,
        bounds: Interval::new(-ONE, ONE).unwrap(),
    }];
    let a = layer::Layout::affine(0, 0, &[2, 3, 4], &[4, 8, 1]).unwrap();
    let joined = layer::Layout::concat(&[a.slice(0, 5).unwrap(), a.slice(5, 19).unwrap()]).unwrap();
    assert_eq!(
        (0..24).map(|i| a.word(i).unwrap()).collect::<Vec<_>>(),
        (0..24).map(|i| joined.word(i).unwrap()).collect::<Vec<_>>()
    );
    assert!(a.slice(0, 4).unwrap().row_key(4).is_some());
    assert!(a.slice(0, 8).unwrap().row_key(8).is_none());
    assert!(layer::Layout::affine(0, usize::MAX, &[2], &[1]).is_err());
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let output = b.binary(&a, &joined, false).unwrap();
    let plan = b.finish(&[output]).unwrap();
    assert!(plan.resources().admitted_bytes < MAX_PROGRAM_BYTES);
}

#[test]
fn body_layer_input_layout_quote_preserves_identity_work_and_complete_row_bindings() {
    let inputs = [body::Register {
        lanes: 3,
        bounds: Interval::new(-12, 12).unwrap(),
    }];
    let make = |builder_input: bool| {
        let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
        let row = if builder_input {
            b.input(0).unwrap()
        } else {
            layer::Layout::affine(0, 0, &[3], &[1]).unwrap()
        };
        let quote = layer::Layout::quote_contiguous(0, 3).unwrap();
        assert_eq!(
            quote.destination_backing_bytes,
            row.backing_bytes().unwrap()
        );
        let output = b.residual(&row, &row, 3).unwrap();
        let plan = b.finish(&[output.clone()]).unwrap();
        let proof = plan.complete_row(&output).unwrap();
        let ordered =
            layer::Layout::concat(&[output.slice(0, 1).unwrap(), output.slice(1, 2).unwrap()])
                .unwrap();
        let alias = plan.complete_row(&ordered).unwrap();
        assert_eq!(alias.key(), proof.key());
        assert_eq!(alias.energy(), proof.energy());
        assert_eq!(alias.mass(), proof.mass());
        assert_eq!(alias.producer(), plan.digest());
        let reversed = layer::Layout::concat(&[
            output.slice(2, 1).unwrap(),
            output.slice(1, 1).unwrap(),
            output.slice(0, 1).unwrap(),
        ])
        .unwrap();
        let repeated = layer::Layout::concat(&[
            output.slice(0, 1).unwrap(),
            output.slice(0, 1).unwrap(),
            output.slice(2, 1).unwrap(),
        ])
        .unwrap();
        assert!(plan.complete_row(&reversed).is_none());
        assert!(plan.complete_row(&repeated).is_none());
        assert!(plan.complete_row(&output.slice(0, 2).unwrap()).is_none());
        assert!(plan.complete_row(&row).is_none());
        plan
    };
    let quoted = make(true);
    let direct = make(false);
    assert_eq!(quoted.digest(), direct.digest());
    assert_eq!(quoted.work_budget(), direct.work_budget());
    assert_eq!(quoted.coordinator_advances(), direct.coordinator_advances());
    assert_eq!(
        quoted.resources().admitted_bytes,
        direct.resources().admitted_bytes
    );
}

#[test]
fn body_layer_checked_row_energy_rejected_and_endpoint_schedule_controls() {
    let inputs = [
        body::Register {
            lanes: 1,
            bounds: Interval::new(-12, 12).unwrap(),
        },
        body::Register {
            lanes: 2,
            bounds: Interval::new(0, 0).unwrap(),
        },
        body::Register {
            lanes: 3,
            bounds: Interval::new(0, 0).unwrap(),
        },
    ];
    let make = |transfer: bool| {
        let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
        let input = layer::Layout::concat(&[b.input(0).unwrap(), b.input(1).unwrap()]).unwrap();
        let zero = b.input(2).unwrap();
        // Genuine native producer: only one input coordinate can be nonzero.
        // Scalar hull is [-12,12], complete-row energy is 144 < 3*9^2.
        let produced = b.residual(&input, &zero, 3).unwrap();
        let alias =
            layer::Layout::concat(&[produced.slice(0, 1).unwrap(), produced.slice(1, 2).unwrap()])
                .unwrap();
        let (safe, valid) = if transfer {
            b.checked_rows(&alias, 3, 9, None)
        } else {
            b.checked(&alias, 9, None)
        }
        .unwrap();
        let (again, sticky) = if transfer {
            b.checked_rows(&alias, 3, 9, valid.as_ref())
        } else {
            b.checked(&alias, 9, valid.as_ref())
        }
        .unwrap();
        let plan = b.finish(&[safe.clone(), again, sticky.unwrap()]).unwrap();
        let producer = plan.complete_row(&produced).unwrap();
        assert_eq!(producer.producer(), plan.digest());
        assert_eq!(producer.energy(), Some(144));
        if transfer {
            let proof = plan.complete_row(&safe).unwrap();
            assert_eq!(proof.energy(), Some(144));
            assert_eq!(proof.mass(), None);
            assert_eq!(proof.producer(), plan.digest());
            assert!(plan.complete_row(&safe.slice(0, 2).unwrap()).is_none());
        } else {
            assert!(plan.complete_row(&safe).is_none());
        }
        plan
    };
    let transferred = make(true);
    let baseline = make(false);
    assert_eq!(transferred.digest(), baseline.digest());
    assert_eq!(transferred.work_budget(), baseline.work_budget());
    assert_eq!(
        transferred.coordinator_advances(),
        baseline.coordinator_advances()
    );
    for value in [-10_i64, -9, 0, 9, 10] {
        let clear = BTreeMap::from([
            ("x0".into(), vec![value]),
            ("x1".into(), vec![0; 2]),
            ("x2".into(), vec![0; 3]),
        ]);
        let drive = |plan: &Arc<layer::Plan>| {
            let program = fixture(plan, &inputs);
            run(
                workers(program, &clear).map(|w| layer::from_worker(plan.clone(), w, None)),
                plan,
            )
        };
        let (actual, trace) = drive(&transferred);
        let (expected, baseline_trace) = drive(&baseline);
        assert_eq!(actual, expected);
        assert_eq!(trace, baseline_trace);
        let accepted = (-9..=9).contains(&value);
        let safe = vec![if accepted { value as u64 } else { 0 }, 0, 0];
        assert_eq!(actual, vec![safe.clone(), safe, vec![u64::from(accepted)]]);
    }
}

fn finish_expanded(b: Builder, inputs: usize, outputs: Vec<(usize, usize)>) -> Arc<Program> {
    let work = schedule_work(&b.nodes).unwrap();
    let mut last: Vec<_> = (0..b.nodes.len()).collect();
    for (step, node) in b.nodes.iter().enumerate() {
        for owner in node.inputs() {
            last[owner] = step;
        }
    }
    for &(owner, _) in &outputs {
        last[owner] = b.nodes.len();
    }
    let mut retire = vec![vec![]; b.nodes.len()];
    for (owner, step) in last.into_iter().enumerate() {
        if step < retire.len() {
            retire[step].push(owner);
        }
    }
    Arc::new(Program {
        nodes: b.nodes,
        inputs,
        outputs,
        retire,
        digest: [83; 32],
        source_binding: [0; 32],
        payload_bytes: b.allocated,
        work,
    })
}

fn native_rms(b: &mut Builder, a: &View, width: usize) -> View {
    let safe = b.checked_view(&a.words).unwrap();
    let mut words = Words::new();
    for row in safe.chunks_exact(width) {
        let input = b.gather(&row).unwrap();
        let gamma = vec![ONE; width];
        let (program, squared) = match b.row_norm(&row, width) {
            Some(norm) => tensor::Program::rms_norm_with_input_norm(
                1,
                width,
                b.nodes[input].bounds,
                &gamma,
                norm,
            )
            .unwrap(),
            None => tensor::Program::rms_norm_certified(1, width, b.nodes[input].bounds, &gamma)
                .unwrap(),
        };
        let output = b.tensor(input, program).unwrap();
        b.row_norms.push((output, squared));
        words.extend((0..width).map(|i| (output, i)));
    }
    View {
        words,
        shape: a.shape.clone(),
    }
}
fn native_rope(b: &mut Builder, a: &View, width: usize, sequence: usize) -> View {
    let mut words = Words::new();
    for (i, row) in a.words.chunks_exact(width).enumerate() {
        let (cos, sin) =
            tensor::Program::rotary_coefficients(10000.0, (i % sequence) as u32, width).unwrap();
        let input = b.gather(&row).unwrap();
        let p = tensor::Program::rotary_half(1, width, b.nodes[input].bounds, &cos, &sin).unwrap();
        let output = b.tensor(input, p).unwrap();
        words.extend((0..width).map(|i| (output, i)));
    }
    View {
        words,
        shape: a.shape.clone(),
    }
}
fn native_public(
    b: &mut Builder,
    a: &View,
    weight: usize,
    columns: usize,
    bias: Option<usize>,
) -> View {
    let mut output = b.public_matrix(a, weight, columns, false, true).unwrap();
    if let Some(bias) = bias {
        let repeated = (0..output.words.len())
            .map(|i| (bias, i % columns))
            .collect();
        output.words = b.binary(&output.words, &repeated, false).unwrap();
    }
    output
}
fn native_drive(mut owners: [Worker; 3]) -> (Vec<u64>, Signature) {
    let mut queue = Vec::new();
    let mut trace = Vec::new();
    for _ in 0..500_000 {
        if owners.iter().all(Worker::complete) {
            break;
        }
        if queue.is_empty() {
            for owner in &mut owners {
                let from = owner.party();
                queue.extend(
                    owner
                        .advance()
                        .unwrap()
                        .into_iter()
                        .map(|(to, frame)| (from, to, frame)),
                );
            }
        } else {
            trace.push(signature(&queue));
            for (from, to, frame) in std::mem::take(&mut queue) {
                queue.extend(
                    owners[to as usize]
                        .receive(frame.as_bytes(), from)
                        .unwrap()
                        .into_iter()
                        .map(|(next, frame)| (to, next, frame)),
                );
            }
        }
    }
    assert!(queue.is_empty());
    assert!(owners.iter().all(Worker::complete));
    let shares = owners.map(|w| w.finish().unwrap());
    (
        (0..shares[0].len())
            .map(|i| shares.iter().fold(0u64, |sum, s| sum.wrapping_add(s[i])))
            .collect(),
        trace,
    )
}

#[test]
fn body_layer_generated_complete_gqa_hidden_logits_kv_feedback_matches_expanded() {
    // Deliberately exact-zero public projections: full noninteger RMS/RoPE/SiLU/
    // softmax circuits still execute, while exported results are independent of
    // stochastic masks. This is NOT the stronger fixed-PRSS promotion proof.
    let (seq, hidden, heads, kv, width, intermediate, vocabulary) =
        (3usize, 4usize, 2usize, 1usize, 2usize, 6usize, 5usize);
    let shapes = BTreeMap::from([
        ("q", 16usize),
        ("k", 8),
        ("v", 8),
        ("o", 16),
        ("gate", 24),
        ("up", 24),
        ("down", 24),
        ("head", 20),
        ("qb", 4),
        ("kb", 2),
        ("vb", 2),
    ]);
    let mut native = Builder {
        public_weights: true,
        checked_bound: Some(ONE),
        ..Builder::default()
    };
    let mut ids = BTreeMap::new();
    let mut sources = BTreeMap::new();
    let mut clear = BTreeMap::new();
    for (&key, &lanes) in &shapes {
        let digest: [u8; 32] = Sha256::digest(vec![0u8; 8 * lanes]).into();
        let id = native
            .push(
                Code::PublicInput {
                    key: key.into(),
                    digest,
                    row_l1: None,
                    row_squared_norm: None,
                    row_l1_squared_sum: None,
                },
                lanes,
                Interval::new(0, 0).unwrap(),
            )
            .unwrap();
        ids.insert(key, id);
        sources.insert(
            key,
            layer::Source::new(id, key.into(), digest, lanes, Interval::new(0, 0).unwrap())
                .unwrap(),
        );
        clear.insert(key.into(), vec![0; lanes]);
    }
    let input = native
        .push(
            Code::Input("x0".into()),
            seq * hidden,
            Interval::new(-4 * ONE, 4 * ONE).unwrap(),
        )
        .unwrap();
    clear.insert("x0".into(), vec![0; seq * hidden]);
    let x = native.view(input, vec![1, seq, hidden]);
    let norm = native_rms(&mut native, &x, hidden);
    let q = native_public(&mut native, &norm, ids["q"], hidden, Some(ids["qb"]));
    let k = native_public(&mut native, &norm, ids["k"], kv * width, Some(ids["kb"]));
    let v = native_public(&mut native, &norm, ids["v"], kv * width, Some(ids["vb"]));
    let q = View {
        words: (0..heads)
            .flat_map(|h| {
                (0..seq).flat_map({
                    let q = &q;
                    move |r| (0..width).map(move |d| q.words.at((r * heads + h) * width + d))
                })
            })
            .collect(),
        shape: vec![1, heads, seq, width],
    };
    let k = View {
        words: k.words.clone(),
        shape: vec![1, kv, seq, width],
    };
    let q = native_rope(&mut native, &q, width, seq);
    let k = native_rope(&mut native, &k, width, seq);
    let anchor = native.gather(&x.words.slice(..1)).unwrap();
    let zero = native.constant(anchor, 0).unwrap();
    let mut scores = Words::new();
    for h in 0..heads {
        for r in 0..seq {
            let query = View {
                words: q
                    .words
                    .slice((h * seq + r) * width..(h * seq + r + 1) * width),
                shape: vec![1, width],
            };
            let keys = View {
                words: k.words.slice(..(r + 1) * width),
                shape: vec![r + 1, width],
            };
            scores.extend(native.linear(&query, &keys, true).unwrap().words);
            scores.extend(std::iter::repeat_n((zero, 0), seq - r - 1));
        }
    }
    let mut scaled = Words::new();
    let coefficient = ((1.0_f32 / (width as f32).sqrt()) * ONE as f32) as i64;
    for chunk in scores.chunks(128) {
        let input = native.gather(&chunk).unwrap();
        let mut b = tensor::Builder::new(chunk.len(), native.nodes[input].bounds).unwrap();
        let raw = b
            .affine(b.input(), None, affine(coefficient, 0, 0))
            .unwrap();
        let output = b.rescale(raw, 18).unwrap();
        let id = native.tensor(input, b.finish(output).unwrap()).unwrap();
        scaled.extend((0..chunk.len()).map(|i| (id, i)));
    }
    let mut probabilities = Words::new();
    for (i, row) in scaled.chunks_exact(seq).enumerate() {
        let n = i % seq + 1;
        let input = native.gather(&row.slice(..n)).unwrap();
        let (program, mass) =
            tensor::Program::softmax_certified(1, n, native.nodes[input].bounds).unwrap();
        let id = native.tensor(input, program).unwrap();
        native.row_masses.push((id, mass));
        probabilities.extend((0..n).map(|i| (id, i)));
        probabilities.extend(std::iter::repeat_n((zero, 0), seq - n));
    }
    let mut attention = Words::new();
    for h in 0..heads {
        for r in 0..seq {
            let valid = r + 1;
            let p = View {
                words: probabilities.slice((h * seq + r) * seq..(h * seq + r) * seq + valid),
                shape: vec![1, valid],
            };
            let selected = (0..width)
                .flat_map(|d| {
                    (0..valid).map({
                        let v = &v;
                        move |key| v.words.at(key * width + d)
                    })
                })
                .collect();
            attention.extend(
                native
                    .linear(
                        &p,
                        &View {
                            words: selected,
                            shape: vec![width, valid],
                        },
                        true,
                    )
                    .unwrap()
                    .words,
            );
        }
    }
    let merged = View {
        words: (0..seq)
            .flat_map(|r| {
                (0..heads).flat_map({
                    let attention = &attention;
                    move |h| (0..width).map(move |d| attention.at((h * seq + r) * width + d))
                })
            })
            .collect(),
        shape: vec![1, seq, hidden],
    };
    let output = native_public(&mut native, &merged, ids["o"], hidden, None);
    let residual = View {
        words: native.residual(&x, &output).unwrap(),
        shape: x.shape.clone(),
    };
    let norm2 = native_rms(&mut native, &residual, hidden);
    let gate = native_public(&mut native, &norm2, ids["gate"], intermediate, None);
    let up = native_public(&mut native, &norm2, ids["up"], intermediate, None);
    let input = native.gather(&gate.words).unwrap();
    let bounds = native.nodes[input].bounds;
    let act = native
        .tensor_loop(LoopInput::Slice(input), seq * intermediate, |n| {
            tensor::Program::activation_guarded(tensor::Activation::Silu, n, bounds)
        })
        .unwrap();
    let activation = native.view(act, vec![1, seq, intermediate]);
    let product = View {
        words: native.binary(&activation.words, &up.words, true).unwrap(),
        shape: activation.shape,
    };
    let down = native_public(&mut native, &product, ids["down"], hidden, None);
    let hidden_out = View {
        words: native.residual(&residual, &down).unwrap(),
        shape: x.shape.clone(),
    };
    let final_norm = native_rms(&mut native, &hidden_out, hidden);
    let last = View {
        words: final_norm.words.slice((seq - 1) * hidden..seq * hidden),
        shape: vec![1, 1, hidden],
    };
    let logits = native_public(&mut native, &last, ids["head"], vocabulary, None);
    let outputs: Vec<_> = hidden_out
        .words
        .iter()
        .chain(logits.words.iter())
        .chain(k.words.iter())
        .chain(v.words.iter())
        .chain(std::iter::once((native.validity.unwrap(), 0)))
        .collect();
    let expanded = finish_expanded(native, shapes.len() + 1, outputs);

    let inputs = [body::Register {
        lanes: seq * hidden,
        bounds: Interval::new(-4 * ONE, 4 * ONE).unwrap(),
    }];
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let x = b.input(0).unwrap();
    let gamma = vec![ONE; hidden];
    let (norm, valid) = b.rms(&x, hidden, &gamma, Some(ONE), None).unwrap();
    let q = b
        .public_matrix(
            &norm,
            sources["q"].clone(),
            seq,
            hidden,
            hidden,
            false,
            true,
        )
        .unwrap();
    let q = b.bias(&q, sources["qb"].clone(), hidden).unwrap();
    let k = b
        .public_matrix(
            &norm,
            sources["k"].clone(),
            seq,
            hidden,
            kv * width,
            false,
            true,
        )
        .unwrap();
    let k = b.bias(&k, sources["kb"].clone(), kv * width).unwrap();
    let v = b
        .public_matrix(
            &norm,
            sources["v"].clone(),
            seq,
            hidden,
            kv * width,
            false,
            true,
        )
        .unwrap();
    let v = b.bias(&v, sources["vb"].clone(), kv * width).unwrap();
    let q_owner = q.word(0).unwrap().0;
    let q = layer::Layout::affine(q_owner, 0, &[heads, seq, width], &[width, heads * width, 1])
        .unwrap();
    let q = b.rope(&q, width, seq, 0, 10000.0).unwrap();
    let k = b.rope(&k, width, seq, 0, 10000.0).unwrap();
    let scores = b.scores(&q, &k, heads, kv, seq, 0, width).unwrap();
    let scaled = b.attention_scale(&scores, width).unwrap();
    let p = b.softmax(&scaled, seq, 0).unwrap();
    let attention = b.values(&p, &v, heads, kv, seq, 0, width).unwrap();
    let owner = attention.word(0).unwrap().0;
    let merged =
        layer::Layout::affine(owner, 0, &[seq, heads, width], &[width, seq * width, 1]).unwrap();
    let output = b
        .public_matrix(
            &merged,
            sources["o"].clone(),
            seq,
            hidden,
            hidden,
            false,
            true,
        )
        .unwrap();
    let residual = b.residual(&x, &output, hidden).unwrap();
    let (norm2, valid) = b
        .rms(&residual, hidden, &gamma, Some(ONE), valid.as_ref())
        .unwrap();
    let gate = b
        .public_matrix(
            &norm2,
            sources["gate"].clone(),
            seq,
            hidden,
            intermediate,
            false,
            true,
        )
        .unwrap();
    let up = b
        .public_matrix(
            &norm2,
            sources["up"].clone(),
            seq,
            hidden,
            intermediate,
            false,
            true,
        )
        .unwrap();
    let activation = b.silu(&gate).unwrap();
    let product = b.binary(&activation, &up, true).unwrap();
    let down = b
        .public_matrix(
            &product,
            sources["down"].clone(),
            seq,
            intermediate,
            hidden,
            false,
            true,
        )
        .unwrap();
    let hidden_out = b.residual(&residual, &down, hidden).unwrap();
    let (final_norm, valid) = b
        .rms(&hidden_out, hidden, &gamma, Some(ONE), valid.as_ref())
        .unwrap();
    let logits = b
        .public_matrix(
            &final_norm.slice((seq - 1) * hidden, hidden).unwrap(),
            sources["head"].clone(),
            1,
            hidden,
            vocabulary,
            false,
            true,
        )
        .unwrap();
    let plan = b
        .finish(&[hidden_out, logits, k, v, valid.unwrap()])
        .unwrap();
    assert_eq!(plan.work_budget(), expanded.work_budget());
    for input in [0, 2 * ONE] {
        clear.insert("x0".into(), vec![input; seq * hidden]);
        let (expected, expanded_trace) = native_drive(workers(expanded.clone(), &clear));
        let owners = workers(expanded.clone(), &clear)
            .map(|worker| layer::from_worker(plan.clone(), worker, None));
        let (actual, body_trace) = run(owners, &plan);
        assert_eq!(actual[0], vec![input as u64; seq * hidden]);
        assert_eq!(actual[4], vec![u64::from(input == 0)]);
        assert_eq!(actual.into_iter().flatten().collect::<Vec<_>>(), expected);
        assert_eq!(body_trace, expanded_trace);
    }
    eprintln!(
        "semantic layer {}",
        serde_json::to_string(&crate::replicated_decoder::cost::semantic_body(&plan)).unwrap()
    );
}

#[test]
fn body_layer_nonzero_matrix_original_column_tiles_and_odd_tail() {
    let (rows, inner, columns) = (2, 3, 129);
    let bounds = Interval::new(ONE, ONE).unwrap();
    let inputs = [
        body::Register {
            lanes: rows * inner,
            bounds,
        },
        body::Register {
            lanes: inner * columns,
            bounds,
        },
    ];
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let left = b.input(0).unwrap();
    let right = b.input(1).unwrap();
    let output = b
        .dense_private(&left, &right, rows, inner, columns, true)
        .unwrap();
    let plan = b.finish(&[output]).unwrap();
    let mut n = Builder::default();
    let a = n
        .push(Code::Input("x0".into()), rows * inner, bounds)
        .unwrap();
    let b = n
        .push(Code::Input("x1".into()), inner * columns, bounds)
        .unwrap();
    let left = n.view(a, vec![1, rows, inner]);
    let right = n.view(b, vec![columns, inner]);
    let output = n.linear(&left, &right, true).unwrap();
    let native = finish_expanded(n, 2, output.words.iter().collect());
    assert_eq!(native.work_budget(), plan.work_budget());
    let clear = BTreeMap::from([
        ("x0".into(), vec![ONE; rows * inner]),
        ("x1".into(), vec![ONE; inner * columns]),
    ]);
    let (expected, trace) = native_drive(workers(native.clone(), &clear));
    let (actual, body_trace) = run(
        workers(native, &clear).map(|w| layer::from_worker(plan.clone(), w, None)),
        &plan,
    );
    assert_eq!(expected, vec![(3 * ONE) as u64; rows * columns]);
    assert_eq!(actual[0], expected);
    assert_eq!(trace, body_trace);
}

#[test]
fn body_layer_retires_complete_registers_not_cumulative_expansion() {
    let inputs = [
        body::Register {
            lanes: 1024,
            bounds: Interval::new(-ONE, ONE).unwrap(),
        },
        body::Register {
            lanes: 1024,
            bounds: Interval::new(0, 0).unwrap(),
        },
    ];
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let mut value = b.input(0).unwrap();
    let zero = b.input(1).unwrap();
    for _ in 0..100 {
        value = b.binary(&value, &zero, false).unwrap();
    }
    let plan = b.finish(&[value]).unwrap();
    assert_eq!(plan.resources().peak_live_secret_bytes, 3 * 1024 * 16);
    let clear = BTreeMap::from([("x0".into(), vec![ONE; 1024]), ("x1".into(), vec![0; 1024])]);
    let program = fixture(&plan, &inputs);
    let (actual, _) = run(
        workers(program, &clear).map(|w| layer::from_worker(plan.clone(), w, None)),
        &plan,
    );
    assert_eq!(actual[0], vec![ONE as u64; 1024]);
}

#[test]
fn body_layer_checked_rms_failure_and_tensor_cancellation_burn_owner() {
    let inputs = [body::Register {
        lanes: 8,
        bounds: Interval::new(-4 * ONE, 4 * ONE).unwrap(),
    }];
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let x = b.input(0).unwrap();
    let (output, valid) = b.rms(&x, 4, &[ONE; 4], Some(ONE), None).unwrap();
    let first_row = output.slice(0, 4).unwrap();
    let plan = b.finish(&[output, valid.unwrap()]).unwrap();
    assert!(plan.complete_row(&first_row).unwrap().energy().is_some());
    assert!(plan.complete_row(&first_row.slice(0, 3).unwrap()).is_none());
    let clear = BTreeMap::from([("x0".into(), vec![2 * ONE; 8])]);
    let program = fixture(&plan, &inputs);
    let (actual, _) = run(
        workers(program.clone(), &clear).map(|w| layer::from_worker(plan.clone(), w, None)),
        &plan,
    );
    assert_eq!(actual, vec![vec![0u64; 8], vec![0]]);
    let mut owners = workers(program, &clear).map(|w| layer::from_worker(plan.clone(), w, None));
    for owner in &mut owners {
        owner.advance().unwrap();
        owner.cancel();
        assert_eq!(owner.live_bytes(), 0);
        assert_eq!(owner.advance().err(), Some(Error::Retired));
    }
    assert!(layer::Builder::new(WeightMode::Private, &inputs).is_err());
}

#[test]
fn body_layer_public_transposed_nonzero_bias_and_exact_source_binding() {
    let weight = vec![ONE, 0, ONE, 0, ONE, ONE];
    let bias = vec![ONE, -ONE, 0];
    let clear = BTreeMap::from([
        ("w".into(), weight.clone()),
        ("bias".into(), bias.clone()),
        ("x0".into(), vec![ONE, 2 * ONE]),
    ]);
    let digest = |v: &[i64]| -> [u8; 32] {
        let mut h = Sha256::new();
        for x in v {
            h.update(x.to_le_bytes());
        }
        h.finalize().into()
    };
    let mut n = Builder {
        public_weights: true,
        ..Builder::default()
    };
    let wb = Interval::new(0, ONE).unwrap();
    let bb = Interval::new(-ONE, ONE).unwrap();
    let xb = Interval::new(ONE, 2 * ONE).unwrap();
    let w = n
        .push(
            Code::PublicInput {
                key: "w".into(),
                digest: digest(&weight),
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            6,
            wb,
        )
        .unwrap();
    let bias_owner = n
        .push(
            Code::PublicInput {
                key: "bias".into(),
                digest: digest(&bias),
                row_l1: None,
                row_squared_norm: None,
                row_l1_squared_sum: None,
            },
            3,
            bb,
        )
        .unwrap();
    let x = n.push(Code::Input("x0".into()), 2, xb).unwrap();
    let x = n.view(x, vec![1, 1, 2]);
    let product = n.public_matrix(&x, w, 3, true, true).unwrap();
    let repeated = (0..3).map(|i| (bias_owner, i)).collect();
    let output = n.binary(&product.words, &repeated, false).unwrap();
    let native = finish_expanded(n, 3, output.iter().collect());
    let make = |bad: bool| {
        let mut b = layer::Builder::new(
            WeightMode::Public,
            &[body::Register {
                lanes: 2,
                bounds: xb,
            }],
        )
        .unwrap();
        let input = b.input(0).unwrap();
        let source = layer::Source::new(
            w,
            "w".into(),
            if bad { [99; 32] } else { digest(&weight) },
            6,
            wb,
        )
        .unwrap();
        let out = b
            .public_matrix(&input, source, 1, 2, 3, true, true)
            .unwrap();
        let source = layer::Source::new(bias_owner, "bias".into(), digest(&bias), 3, bb).unwrap();
        let out = b.bias(&out, source, 3).unwrap();
        b.finish(&[out]).unwrap()
    };
    let plan = make(false);
    assert_eq!(plan.statistics().logical_public_file_read_bytes, 72);
    assert_eq!(plan.statistics().public_bias_vector_reads, 1);
    assert_eq!(plan.statistics().public_ring64_macs, 12);
    let (expected, trace) = native_drive(workers(native.clone(), &clear));
    let (actual, body_trace) = run(
        workers(native.clone(), &clear).map(|w| layer::from_worker(plan.clone(), w, None)),
        &plan,
    );
    assert_eq!(
        expected,
        vec![(2 * ONE) as u64, ONE as u64, (3 * ONE) as u64]
    );
    assert_eq!(actual[0], expected);
    assert_eq!(trace, body_trace);
    let bad = make(true);
    for worker in workers(native, &clear) {
        assert!(matches!(
            layer::try_from_worker(bad.clone(), worker, None),
            Err(Error::Binding)
        ));
    }
}

#[test]
fn body_layer_malformed_tensor_frame_and_original_memory_ceiling_fail_closed() {
    let inputs = [body::Register {
        lanes: 2,
        bounds: Interval::new(-ONE, ONE).unwrap(),
    }];
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let x = b.input(0).unwrap();
    let output = b.attention_scale(&x, 2).unwrap();
    let plan = b.finish(&[output]).unwrap();
    let clear = BTreeMap::from([("x0".into(), vec![0; 2])]);
    let program = fixture(&plan, &inputs);
    for mut owner in workers(program, &clear).map(|w| layer::from_worker(plan.clone(), w, None)) {
        owner.advance().unwrap();
        assert!(owner.receive(b"invalid", puma::Party::P1).is_err());
        assert_eq!(owner.live_bytes(), 0);
        assert_eq!(owner.advance().err(), Some(Error::Retired));
    }
    let inputs = [body::Register {
        lanes: MAX_VALUES,
        bounds: Interval::new(-1, 1).unwrap(),
    }; 4];
    let mut b = layer::Builder::new(WeightMode::Public, &inputs).unwrap();
    let x = b.input(0).unwrap();
    let y = b.input(1).unwrap();
    let output = b.binary(&x, &y, false).unwrap();
    assert!(matches!(b.finish(&[output]), Err(Error::Bounds)));
}

#[test]
fn body_layer_nonzero_causal_gqa_with_concatenated_previous_kv() {
    let (heads, kv, seq, position, width) = (4usize, 2usize, 3usize, 2usize, 3usize);
    let keys = position + seq;
    for values in [false, true] {
        let left_len = if values {
            heads * seq * keys
        } else {
            heads * seq * width
        };
        let left: Vec<i64> = (0..left_len)
            .map(|i| {
                if values {
                    ((i / keys + i % keys) % 5 + 1) as i64 * ONE
                } else {
                    ((i / width + i % width) % 5 + 1) as i64 * ONE
                }
            })
            .collect();
        let previous: Vec<i64> = (0..kv * position * width)
            .map(|i| ((i / width + i % width) % 4 + 1) as i64 * ONE)
            .collect();
        let current: Vec<i64> = (0..kv * seq * width)
            .map(|i| ((i / width + i % width) % 6 + 2) as i64 * ONE)
            .collect();
        let bound = Interval::new(0, 8 * ONE).unwrap();
        let specs = [
            body::Register {
                lanes: left.len(),
                bounds: bound,
            },
            body::Register {
                lanes: previous.len(),
                bounds: bound,
            },
            body::Register {
                lanes: current.len(),
                bounds: bound,
            },
        ];
        let mut b = layer::Builder::new(WeightMode::Public, &specs).unwrap();
        let a = b.input(0).unwrap();
        let old = b.input(1).unwrap();
        let new = b.input(2).unwrap();
        let mut parts = Vec::new();
        for head in 0..kv {
            parts.push(
                old.slice(head * position * width, position * width)
                    .unwrap(),
            );
            parts.push(new.slice(head * seq * width, seq * width).unwrap());
        }
        let cache = layer::Layout::concat(&parts).unwrap();
        let output = if values {
            b.values(&a, &cache, heads, kv, seq, position, width)
        } else {
            b.scores(&a, &cache, heads, kv, seq, position, width)
        }
        .unwrap();
        let plan = b.finish(&[output, cache]).unwrap();
        let mut n = Builder::default();
        let mut ids = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            ids.push(
                n.push(Code::Input(format!("x{i}")), spec.lanes, bound)
                    .unwrap(),
            );
        }
        let mut cache = Words::new();
        for head in 0..kv {
            cache.extend(
                (head * position * width..(head + 1) * position * width).map(|i| (ids[1], i)),
            );
            cache.extend((head * seq * width..(head + 1) * seq * width).map(|i| (ids[2], i)));
        }
        let mut output = Words::new();
        let zero = n.constant(ids[0], 0).unwrap();
        for head in 0..heads {
            for row in 0..seq {
                let valid = position + row + 1;
                let key_head = head / (heads / kv);
                let (left_words, right_words, columns, inner) = if values {
                    (
                        (0..valid)
                            .map(|k| (ids[0], (head * seq + row) * keys + k))
                            .collect(),
                        (0..width)
                            .flat_map(|d| {
                                (0..valid).map({
                                    let cache = &cache;
                                    move |k| cache.at((key_head * keys + k) * width + d)
                                })
                            })
                            .collect(),
                        width,
                        valid,
                    )
                } else {
                    (
                        (0..width)
                            .map(|d| (ids[0], (head * seq + row) * width + d))
                            .collect(),
                        cache.slice(key_head * keys * width..(key_head * keys + valid) * width),
                        valid,
                        width,
                    )
                };
                let a = View {
                    words: left_words,
                    shape: vec![1, inner],
                };
                let w = View {
                    words: right_words,
                    shape: vec![columns, inner],
                };
                output.extend(n.linear(&a, &w, true).unwrap().words);
                if !values {
                    output.extend(std::iter::repeat_n((zero, 0), keys - valid));
                }
            }
        }
        let native = finish_expanded(n, 3, output.iter().chain(cache.iter()).collect());
        assert_eq!(native.work_budget(), plan.work_budget());
        let clear = BTreeMap::from([
            ("x0".into(), left),
            ("x1".into(), previous),
            ("x2".into(), current),
        ]);
        let (expected, trace) = native_drive(workers(native.clone(), &clear));
        let (actual, body_trace) = run(
            workers(native, &clear).map(|w| layer::from_worker(plan.clone(), w, None)),
            &plan,
        );
        assert_eq!(actual.into_iter().flatten().collect::<Vec<_>>(), expected);
        assert_eq!(trace, body_trace);
    }
}

/// Exercise the real BODY interpreter, not a second shared-native constructor.
/// Plan templates/active native owners are private to other modules: these tests
/// observe their exact runtime reservation, not Arc pointer identity. Reverting
/// any BODY constructor to the owned API removes the independently priced
/// template reservation and must fail the corresponding activity assertion.
mod shared_template_execution {
    use super::*;
    use puma::{prss::PrssLimits, Party, ReplicatedShare};

    const PARTIES: [Party; 3] = [Party::P0, Party::P1, Party::P2];
    const LANES: usize = 3;
    type Wire = Vec<(Party, Party, Vec<u8>)>;
    type Outgoing = Vec<(Party, PeerBytes)>;
    type Components = Vec<(u64, u64)>;

    #[derive(Clone, Copy)]
    enum Family {
        Tensor,
        Check,
    }

    fn fixture(family: Family) -> (Arc<layer::Plan>, Vec<tensor::Program>) {
        let bounds = Interval::new(-2 * ONE, 2 * ONE).unwrap();
        let mut builder = layer::Builder::new(
            WeightMode::Public,
            &[body::Register {
                lanes: LANES,
                bounds,
            }],
        )
        .unwrap();
        let input = builder.input(0).unwrap();
        let (outputs, original) = match family {
            Family::Tensor => {
                let output = builder.attention_scale(&input, 4).unwrap();
                // Original native composition for this public Builder family.
                let mut native = tensor::Builder::new(LANES, bounds).unwrap();
                let raw = native
                    .affine(
                        native.input(),
                        None,
                        Affine {
                            left: ONE / 2,
                            right: 0,
                            constant: 0,
                        },
                    )
                    .unwrap();
                let output_native = native.rescale(raw, 18).unwrap();
                (vec![output], vec![native.finish(output_native).unwrap()])
            }
            Family::Check => {
                // Genuine checked producer supplies both safe words and the
                // protected validity consumed by AllValid. No ledger injection.
                let (safe, valid) = builder.checked(&input, ONE, None).unwrap();
                (
                    vec![safe, valid.unwrap()],
                    vec![
                        tensor::Program::checked_range(
                            LANES,
                            bounds,
                            Interval::new(-ONE, ONE).unwrap(),
                            0,
                        )
                        .unwrap(),
                        tensor::Program::all_valid(LANES).unwrap(),
                    ],
                )
            }
        };
        (builder.finish(&outputs).unwrap(), original)
    }

    fn components(party: Party) -> Components {
        [-ONE, 7, 2 * ONE]
            .into_iter()
            .enumerate()
            .map(|(i, clear)| {
                let a = 0xf123_4567_89ab_cdef_u64.wrapping_add(i as u64 * 17);
                let b = 0x8123_4567_89ab_cdef_u64.wrapping_sub(i as u64 * 23);
                let shares = [a, b, (clear as u64).wrapping_sub(a).wrapping_sub(b)];
                (shares[party as usize], shares[party.next() as usize])
            })
            .collect()
    }

    fn runtimes(plan: &layer::Plan, bytes: usize) -> [multiply::PartyRuntime; 3] {
        let work = plan.work_budget();
        let limits = PrssLimits::new(
            // Limits admits its declared maximum before any executor exists.
            // Checked Sign/Bits and wide Boolean conversions expand to 2*LANES.
            puma::Limits::new(2 * LANES, work.operations(), bytes).unwrap(),
            work.cursors().aes_work,
        )
        .unwrap();
        let context = puma::Context::new([111; 32], plan.digest(), 0).unwrap();
        // Fresh known-key fixtures, never a fork/reset of a live owner.
        puma::prss::insecure_test_support::fresh_compound_parties(context, limits).unwrap()
    }

    fn bodies(
        plan: &Arc<layer::Plan>,
        runtimes: [multiply::PartyRuntime; 3],
    ) -> [layer::Executor; 3] {
        runtimes.map(|runtime| {
            let mut input = Secret::new(LANES).unwrap();
            for (i, (first, second)) in components(runtime.party()).into_iter().enumerate() {
                input.first[i] = first;
                input.second[i] = second;
            }
            layer::Executor::new(plan.clone(), runtime, vec![input], None, None).unwrap()
        })
    }

    fn original_input(runtime: &multiply::PartyRuntime, words: &[(u64, u64)]) -> ReplicatedShare {
        // Preserve BODY's original ordinal-zero input rebinding, including the
        // checked-validity -> AllValid boundary. No charged export is retained.
        let operation = puma::Operation::new(
            runtime.context(),
            0,
            Party::P0,
            words.len(),
            1,
            puma::Limits::default(),
        )
        .unwrap();
        let first: Vec<_> = words.iter().map(|word| word.0).collect();
        let second: Vec<_> = words.iter().map(|word| word.1).collect();
        ReplicatedShare::from_local(
            operation,
            runtime.party(),
            &first,
            &second,
            puma::Limits::default(),
        )
        .unwrap()
    }

    // Test-only router adapter: both paths use the original typed frame codec.
    // Every full wave drains before advancing either interpreter's next step.
    trait RoundOwner {
        fn begin(&mut self) -> Result<Outgoing, Error>;
        fn incoming(&mut self, raw: &[u8], from: Party) -> Result<Outgoing, Error>;
    }
    fn native_frames(frames: [Option<tensor::Frame>; 3]) -> Result<Outgoing, Error> {
        frames
            .into_iter()
            .flatten()
            .map(|frame| Ok((frame.recipient(), PeerBytes::Tensor(frame.encode()?))))
            .collect()
    }
    impl RoundOwner for layer::Executor {
        fn begin(&mut self) -> Result<Outgoing, Error> {
            self.advance()
        }
        fn incoming(&mut self, raw: &[u8], from: Party) -> Result<Outgoing, Error> {
            self.receive(raw, from)
        }
    }
    impl RoundOwner for tensor::Executor {
        fn begin(&mut self) -> Result<Outgoing, Error> {
            native_frames(self.start_step()?)
        }
        fn incoming(&mut self, raw: &[u8], from: Party) -> Result<Outgoing, Error> {
            native_frames(self.receive_encoded(raw, from)?)
        }
    }
    fn round(owners: &mut [impl RoundOwner; 3]) -> Result<Wire, Error> {
        let mut queue = Vec::new();
        for (i, owner) in owners.iter_mut().enumerate() {
            queue.extend(
                owner
                    .begin()?
                    .into_iter()
                    .map(|(to, raw)| (PARTIES[i], to, raw)),
            );
        }
        let mut wire = Vec::new();
        for _ in 0..1024 {
            if queue.is_empty() {
                return Ok(wire);
            }
            let mut next = Vec::new();
            for (from, to, frame) in queue {
                wire.push((from, to, frame.as_bytes().to_vec()));
                next.extend(
                    owners[to as usize]
                        .incoming(frame.as_bytes(), from)?
                        .into_iter()
                        .map(|(recipient, raw)| (to, recipient, raw)),
                );
            }
            queue = next;
        }
        panic!("native instruction wave failed to drain");
    }

    fn words(value: &ReplicatedShare) -> Components {
        value
            .first()
            .iter()
            .copied()
            .zip(value.second().iter().copied())
            .collect()
    }

    fn compare_original(family: Family, bytes: usize) {
        let (plan, original) = fixture(family);
        // BODY's conservative floor is NOT the native runtime's payload limit.
        assert_eq!(
            plan.resources().primitive_ledger_allowance_bytes,
            2 * puma::MAX_ADMITTED_BYTES
        );
        let body_runtimes = runtimes(&plan, bytes);
        let body_usage = body_runtimes
            .each_ref()
            .map(multiply::PartyRuntime::payload_usage);
        let mut owners = bodies(&plan, body_runtimes);
        let mut native_runtimes = runtimes(&plan, bytes);
        let native_usage = native_runtimes
            .each_ref()
            .map(multiply::PartyRuntime::payload_usage);
        let retained_plan = Arc::downgrade(&plan);
        drop(plan);
        // Only the genuine BODY owners retain the original Plan and its template
        // Arcs now. This observes Plan lifetime, not a native Arc pointer.
        assert_eq!(retained_plan.strong_count(), 3);
        let mut incoming = PARTIES.map(components);
        let mut expected: [Vec<Components>; 3] = std::array::from_fn(|_| Vec::new());
        for program in &original {
            let resources = program.resources().unwrap();
            let template_delta = resources.immutable_template_bytes - resources.public_table_bytes;
            assert_eq!(
                template_delta,
                std::mem::size_of::<tensor::Program>() + 2 * std::mem::size_of::<usize>()
            );
            let mut reference = native_runtimes.map(|runtime| {
                let input = original_input(&runtime, &incoming[runtime.party() as usize]);
                // Frozen original BODY choice, deliberately NOT new_shared.
                tensor::Executor::new(runtime, program.clone(), input).unwrap()
            });
            for _ in 0..program.instructions() {
                assert_eq!(round(&mut owners).unwrap(), round(&mut reference).unwrap());
                assert_eq!(retained_plan.strong_count(), 3);
                for (body, native) in body_usage.iter().zip(&native_usage) {
                    // Same inputs, registers, active primitive and drained wire.
                    // Only shared immutable-owner reservation differs. This exact
                    // equality catches reverting Tensor, Checked OR AllValid.
                    assert_eq!(body.live_bytes(), native.live_bytes() + template_delta);
                }
                for owner in &mut reference {
                    owner.finish_step().unwrap();
                }
            }
            let finished = reference.map(|owner| {
                if program.validity_lanes().is_some() {
                    let (runtime, safe, valid) = owner.finish_checked().unwrap();
                    // Match BODY's store/copy boundary: owned output components
                    // are ordinary outer buffers; release native payload leases.
                    (runtime, words(&safe), words(&valid))
                } else {
                    let (runtime, output) = owner.finish().unwrap();
                    (runtime, words(&output), Vec::new())
                }
            });
            incoming = finished.each_ref().map(|(_, _, next)| next.clone());
            native_runtimes = finished.map(|(runtime, output, _)| {
                expected[runtime.party() as usize].push(output);
                runtime
            });
            assert!(native_usage.iter().all(|usage| usage.live_bytes() == 0));
        }
        assert!(round(&mut owners).unwrap().is_empty());
        assert!(owners.iter().all(layer::Executor::complete));
        assert!(body_usage.iter().all(|usage| usage.live_bytes() == 0));
        for (i, owner) in owners.into_iter().enumerate() {
            let (runtime, values) = owner.finish().unwrap();
            // Successful exports intentionally retain the original Plan; native
            // invocation reservations have already retired independently.
            assert!(retained_plan.upgrade().is_some());
            assert_eq!(runtime.cursors(), native_runtimes[i].cursors());
            for (output, expected) in expected[i].iter().enumerate() {
                assert_eq!(values.output_len(output).unwrap(), expected.len());
                for (lane, expected) in expected.iter().enumerate() {
                    assert_eq!(values.word(output, lane).unwrap(), *expected);
                }
            }
        }
        assert!(retained_plan.upgrade().is_none());
    }

    #[test]
    fn tensor_shared_reservation_full_wire_both_components_cursors_and_success_release() {
        compare_original(Family::Tensor, puma::MAX_ADMITTED_BYTES);
    }

    #[test]
    fn checked_and_all_valid_shared_reservations_preserve_original_execution_and_release() {
        compare_original(Family::Check, puma::MAX_ADMITTED_BYTES);
    }

    #[test]
    fn native_activity_cancellation_malformed_receive_and_premature_finish_release_reservations() {
        // Target each of the three ordinary native constructor branches. No
        // private Plan/Executor fields, fabricated certificates or runtime hooks.
        for stage in 0..3 {
            for exit in 0..3 {
                let family = if stage == 0 {
                    Family::Tensor
                } else {
                    Family::Check
                };
                let (plan, original) = fixture(family);
                let runtimes = runtimes(&plan, puma::MAX_ADMITTED_BYTES);
                let usage = runtimes
                    .each_ref()
                    .map(multiply::PartyRuntime::payload_usage);
                let mut owners = bodies(&plan, runtimes);
                let retained_plan = Arc::downgrade(&plan);
                drop(plan);
                assert_eq!(retained_plan.strong_count(), 3);
                if stage == 2 {
                    // Finish every real guard instruction, then next advance
                    // enters its genuine protected-validity AllValid consumer.
                    for _ in 0..original[0].instructions() {
                        round(&mut owners).unwrap();
                    }
                }
                let retained = owners.each_mut().map(|owner| owner.advance().unwrap());
                let program = &original[usize::from(stage == 2)];
                let resources = program.resources().unwrap();
                for usage in &usage {
                    assert!(
                        usage.live_bytes()
                            >= resources.invocation_container_bytes
                                + resources.immutable_template_bytes
                    );
                }
                for mut owner in owners {
                    match exit {
                        0 => {
                            owner.cancel();
                            assert_eq!(owner.advance().err(), Some(Error::Retired));
                            assert_eq!(owner.live_bytes(), 0);
                        }
                        1 => {
                            assert!(owner.receive(b"invalid", Party::P1).is_err());
                            assert_eq!(owner.advance().err(), Some(Error::Retired));
                            assert_eq!(owner.live_bytes(), 0);
                        }
                        2 => assert!(matches!(owner.finish(), Err(Error::Phase))),
                        _ => unreachable!(),
                    }
                }
                assert!(retained_plan.upgrade().is_none());
                if retained.iter().any(|frames| !frames.is_empty()) {
                    // Actual outgoing buffers retain issuer leases after native
                    // cancellation/drop; never refund somebody else's owner.
                    assert!(usage.iter().any(|usage| usage.live_bytes() > 0));
                }
                drop(retained);
                assert!(usage.iter().all(|usage| usage.live_bytes() == 0));
            }
        }
    }

    #[test]
    fn native_shared_envelope_exact_accepts_one_short_rejects_before_first_instruction() {
        for family in [Family::Tensor, Family::Check] {
            let (plan, original) = fixture(family);
            let exact = original
                .iter()
                .map(|program| program.resources().unwrap().shared_admitted_bytes)
                .max()
                .unwrap();
            // Checked precedes AllValid and has the larger envelope. Public
            // runtime limits cannot be lowered at that protected boundary, so
            // this is NOT an AllValid-specific exact/one-short assertion.
            assert_eq!(
                exact,
                original[0].resources().unwrap().shared_admitted_bytes
            );
            assert!(exact <= puma::MAX_ADMITTED_BYTES);
            compare_original(family, exact);
            let runtimes = runtimes(&plan, exact - 1);
            let usage = runtimes
                .each_ref()
                .map(multiply::PartyRuntime::payload_usage);
            let owners = bodies(&plan, runtimes);
            for mut owner in owners {
                assert!(matches!(owner.advance(), Err(Error::Bounds)));
                assert_eq!(owner.advance().err(), Some(Error::Retired));
                assert_eq!(owner.live_bytes(), 0);
            }
            assert!(usage.iter().all(|usage| usage.live_bytes() == 0));
        }
    }
}
