//! Native-only test orchestrator. Two worker threads rendezvous by Plan/context;
//! a separate scoped dealer issues fresh paired material. Only this module ever
//! holds both roles or reconstructs diagnostic outputs.
use super::*;
use pllm_garble::fusefss_backend::composition::Event;
use pllm_models::{DecoderPlan, DecoderWorkload, ModelOperator as Op};
use serde_json::json;
use std::{sync::mpsc, thread};

// Reuse the frozen integer constants without decoding JSON, copying a table, or
// introducing a production table/cache owner. This module is test-only.
#[allow(dead_code)]
#[path = "../../../pllm-garble/src/fixed_activation_table_values.rs"]
mod frozen_activation_values;

fn frozen_silu(x: i64) -> i64 {
    let reduced = x.div_euclid(64);
    let index = reduced.unsigned_abs().min(1023) as usize;
    (if reduced >= 0 { x } else { 0 }) - frozen_activation_values::SILU[index] as i64
}

fn fixture() -> (DecoderPlan, Source) {
    let config = json!({"model_type":"gpt2","n_embd":2,"n_head":1,"n_layer":1,"n_positions":4,"vocab_size":3,"n_inner":2});
    let plan = pllm_models::lower_model_json(
        &serde_json::to_vec(&config).unwrap(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut linear = 0;
    let mut norm = 0;
    for op in &plan.prefill.operations {
        let id = op.attributes["weight"].as_str();
        if id.is_none() || tensors.iter().any(|t| Some(t.id()) == id) {
            continue;
        }
        let id = id.unwrap();
        match op.operator {
            Op::TokenLookup => {
                let position = op.inputs[0] != "input.tokens";
                let rows = if position { 4 } else { 3 };
                let codes: Vec<_> = (0..rows)
                    .flat_map(|r| {
                        (0..2).map(move |c| {
                            if position {
                                ((r * 3 + c * 11) % 13 - 6) * 128
                            } else {
                                ((r * 7 + c * 5) % 17 - 8) * 256
                            }
                        })
                    })
                    .collect();
                tensors.push(Tensor::from_q12(id, &[rows as usize, 2], &codes).unwrap());
            }
            Op::LayerNorm => {
                let (gamma, beta) = match norm {
                    0 => ([4096, 4160], [32, -32]),
                    1 => ([4032, 4096], [-32, 64]),
                    _ => ([4160, 4032], [0, 32]),
                };
                norm += 1;
                tensors.push(Tensor::from_q12(id, &[2], &gamma).unwrap());
                tensors.push(
                    Tensor::from_q12(op.attributes["bias"].as_str().unwrap(), &[2], &beta).unwrap(),
                );
            }
            Op::Linear => {
                let columns = if linear == 0 { 6 } else { 2 };
                let mut values = Vec::new();
                for c in 0..2 {
                    for r in 0..columns {
                        values.push(((r * 7 + c * 3 + linear * 5) % 11 - 5) * 64);
                    }
                }
                let bias: Vec<_> = (0..columns).map(|r| ((r + linear) % 5 - 2) * 32).collect();
                tensors.push(Tensor::from_q12(id, &[2, columns as usize], &values).unwrap());
                tensors.push(
                    Tensor::from_q12(
                        op.attributes["bias"].as_str().unwrap(),
                        &[columns as usize],
                        &bias,
                    )
                    .unwrap(),
                );
                linear += 1;
            }
            _ => panic!("unexpected source artifact"),
        }
    }
    (plan, Source::new(tensors).unwrap())
}

fn native_pair(
    plan: &Plan<'_>,
    context: Context,
    a: &[u64],
    b: &[u64],
) -> Result<[NativeOutput; 2], Error> {
    thread::scope(|scope| {
        let (requests, request_rx) = mpsc::sync_channel::<usize>(0);
        let (materials, material_rx) = mpsc::sync_channel(0);
        let dealer = scope.spawn(move || {
            while let Ok(pc) = request_rx.recv() {
                if materials.send(plan.issue_step(pc, context)).is_err() {
                    break;
                }
            }
        });
        let mut left = plan.runtime(0, context, a)?;
        let mut right = plan.runtime(1, context, b)?;
        let mut le = left.poll()?;
        let mut re = right.poll()?;
        let result = loop {
            match (le, re) {
                (Event::NeedMaterial { pc: a }, Event::NeedMaterial { pc: b }) => {
                    if a != b {
                        return Err(Error::Context);
                    }
                    requests.send(a).map_err(|_| Error::Context)?;
                    let [a, b] = material_rx.recv().map_err(|_| Error::Context)??;
                    le = left.feed(a)?;
                    re = right.feed(b)?;
                }
                (Event::Frame(a), Event::Frame(b)) => {
                    le = left.advance(b.as_bytes())?;
                    re = right.advance(a.as_bytes())?;
                }
                (Event::Output(a), Event::Output(b)) => {
                    break Ok([NativeOutput::from_native(a)?, NativeOutput::from_native(b)?])
                }
                _ => break Err(Error::Context),
            }
        };
        drop(requests);
        drop(material_rx);
        dealer.join().map_err(|_| Error::Context)?;
        result
    })
}

struct Request {
    context: Context,
    digest: [u8; 32],
    input: Zeroizing<Vec<u64>>,
    reply: mpsc::SyncSender<Result<NativeOutput, Error>>,
}
type Trace = Vec<(usize, usize, Zeroizing<Vec<u64>>)>;
type KernelRecord = ([u8; 32], [u8; 32], [u8; 32], u64, [u8; 32]);
fn record(context: Context, plan: &Plan<'_>) -> KernelRecord {
    (
        context.plan,
        context.session,
        context.operation,
        context.first_tensor_index,
        plan.digest(),
    )
}
#[derive(Default)]
struct Visits {
    records: Vec<KernelRecord>,
    requests: usize,
    cancelled: bool,
}
impl MaterialVisitor for Visits {
    fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error> {
        self.records.push(record(context, plan));
        self.requests += plan.resources().secret_steps;
        Ok(())
    }
    fn cancel(&mut self) {
        self.cancelled = true;
        self.records.clear();
    }
}
struct HarnessDriver<'v> {
    role: u8,
    send: Option<mpsc::SyncSender<Request>>,
    receive: Option<mpsc::Receiver<Request>>,
    trace: Trace,
    cancelled: bool,
    expected: &'v [KernelRecord],
    next: usize,
}
impl Driver for HarnessDriver<'_> {
    fn run(
        &mut self,
        context: Context,
        plan: &Plan<'_>,
        input: &[u64],
    ) -> Result<NativeOutput, Error> {
        if self.expected.get(self.next) != Some(&record(context, plan)) {
            return Err(Error::Context);
        }
        self.next += 1;
        if self.role == 1 {
            let (reply, rx) = mpsc::sync_channel(0);
            self.send
                .as_ref()
                .unwrap()
                .send(Request {
                    context,
                    digest: plan.digest(),
                    input: Zeroizing::new(input.to_vec()),
                    reply,
                })
                .map_err(|_| Error::Context)?;
            rx.recv().map_err(|_| Error::Context)?
        } else {
            let request = self
                .receive
                .as_ref()
                .unwrap()
                .recv()
                .map_err(|_| Error::Context)?;
            let c = request.context;
            if request.digest != plan.digest()
                || (c.plan, c.session, c.operation, c.first_tensor_index)
                    != (
                        context.plan,
                        context.session,
                        context.operation,
                        context.first_tensor_index,
                    )
            {
                return Err(Error::Context);
            }
            match native_pair(plan, context, input, &request.input) {
                Ok([a, b]) => {
                    request.reply.send(Ok(b)).map_err(|_| Error::Context)?;
                    Ok(a)
                }
                Err(e) => {
                    let _ = request.reply.send(Err(e));
                    Err(e)
                }
            }
        }
    }
    fn cancel(&mut self) {
        self.cancelled = true;
    }
    fn observe(&mut self, selection: usize, node: usize, _shape: &[usize], shares: &[u64]) {
        self.trace
            .push((selection, node, Zeroizing::new(shares.to_vec())));
    }
}

fn signed(v: u64) -> i64 {
    if v & (1u64 << 49) == 0 {
        v as i64
    } else {
        v as i64 - (1i64 << 50)
    }
}

#[test]
fn bound_silu_admitted_registers_match_legacy_native_and_frozen_integer() {
    for online in [OnlineSchedule::Serial, OnlineSchedule::WavefrontV1, OnlineSchedule::TensorV1] {
        let (_, semantic, source) = dense_fixture();
        let program = Program::compile_with_schedule(&semantic, source, 2, Limits::default(), online).unwrap();
        let node = program.nodes.iter().position(|n| matches!(
            n.code, super::super::program::Code::Activation { silu: true, .. }
        )).unwrap();
        let super::super::program::Code::Activation { input, .. } = program.nodes[node].code else { unreachable!() };
        let range = program.nodes[input].range;
        assert!(range.lo <= -65 && range.hi >= 65);
        let count = super::super::program::graph_elements(program.capacity(), &program.nodes[input].shape).unwrap();
        let cases = [range.lo, -65, -64, -1, 0, 1, 63, 64, 65, range.hi];
        let obligations = program.obligations;
        let resources = format!("{:?}", program.resources());
        for batch in 0..cases.len().div_ceil(count) {
            let clear: Vec<_> = (0..count).map(|i| cases[(batch * count + i) % cases.len()]).collect();
            let left: Vec<_> = (0..count).map(|i| (123456789 + i as u64 * 71893) & MASK).collect();
            let right: Vec<_> = clear.iter().zip(&left).map(|(&x, &a)| word(x).wrapping_sub(a) & MASK).collect();
            let context = Context {
                plan: program.digest(), session: [7; 32], operation: [8; 32],
                first_tensor_index: batch as u64,
            };
            let mut legacy_visits = Visits::default();
            let legacy = super::silu_register_control(
                &program, node, context, 0, &left, None, Some(&mut legacy_visits), true,
            ).unwrap();
            let mut visits = Visits::default();
            let diagnostic = super::silu_register_control(
                &program, node, context, 0, &left, None, Some(&mut visits), false,
            ).unwrap();
            assert_eq!(visits.records, legacy_visits.records);
            assert_eq!(visits.requests, legacy_visits.requests);
            assert!(!visits.cancelled && !legacy_visits.cancelled);
            assert!(diagnostic.0.iter().all(|&v| v == 0));
            assert_eq!(diagnostic.1, legacy.1);
            assert_eq!(format!("{:?}", diagnostic.2), format!("{:?}", legacy.2));
            assert_eq!(diagnostic.3, legacy.3);
            let expected = visits.records.as_slice();
            for legacy_route in [true, false] {
                let (sender, receiver) = mpsc::sync_channel(0);
                let run = |role, send, receive, shares: &[u64]| {
                    let mut driver = HarnessDriver {
                        role, send, receive, trace: Vec::new(), cancelled: false, expected, next: 0,
                    };
                    let result = super::silu_register_control(
                        &program, node, context, role, shares, Some(&mut driver), None, legacy_route,
                    ).unwrap();
                    assert!(!driver.cancelled);
                    assert_eq!(driver.next, expected.len());
                    assert_eq!(result.1, diagnostic.1);
                    assert_eq!(format!("{:?}", result.2), format!("{:?}", diagnostic.2));
                    assert_eq!(result.3, diagnostic.3);
                    result.0
                };
                let (a, b) = thread::scope(|scope| {
                    let run = &run;
                    let left = &left;
                    let a = scope.spawn(move || run(0, None, Some(receiver), left));
                    let b = run(1, Some(sender), None, &right);
                    (a.join().unwrap(), b)
                });
                let actual = reconstructed(&a, &b);
                assert_eq!(actual, clear.iter().copied().map(frozen_silu).collect::<Vec<_>>());
                assert!(actual.iter().zip(&clear).all(|(&y, &x)| y >= -65536 && y <= range.hi.max(0) && x >= range.lo && x <= range.hi));
            }
        }
        assert_eq!(program.obligations, obligations);
        assert_eq!(format!("{:?}", program.resources()), resources);
    }
}

#[test]
fn bound_silu_driver_and_material_errors_cancel_without_publishing_registers() {
    struct Fail { calls: usize, cancelled: bool }
    impl Driver for Fail {
        fn run(&mut self, _: Context, _: &Plan<'_>, _: &[u64]) -> Result<NativeOutput, Error> {
            self.calls += 1;
            Err(Error::Context)
        }
        fn cancel(&mut self) { self.cancelled = true; }
    }
    impl MaterialVisitor for Fail {
        fn visit(&mut self, _: Context, _: &Plan<'_>) -> Result<(), Error> {
            self.calls += 1;
            Err(Error::Context)
        }
        fn cancel(&mut self) { self.cancelled = true; }
    }
    for online in [OnlineSchedule::Serial, OnlineSchedule::WavefrontV1, OnlineSchedule::TensorV1] {
        let (_, semantic, source) = dense_fixture();
        let program = Program::compile_with_schedule(&semantic, source, 2, Limits::default(), online).unwrap();
        let node = program.nodes.iter().position(|n| matches!(
            n.code, super::super::program::Code::Activation { silu: true, .. }
        )).unwrap();
        let count = super::super::program::graph_elements(program.capacity(), &program.nodes[node].shape).unwrap();
        let input = vec![0; count];
        let context = Context {
            plan: program.digest(), session: [7; 32], operation: [8; 32], first_tensor_index: 0,
        };
        for legacy in [true, false] {
            let mut driver = Fail { calls: 0, cancelled: false };
            assert!(matches!(super::silu_register_control(
                &program, node, context, 0, &input, Some(&mut driver), None, legacy,
            ), Err(Error::Context)));
            assert_eq!(driver.calls, 1);
            assert!(driver.cancelled);
            let mut visitor = Fail { calls: 0, cancelled: false };
            assert!(matches!(super::silu_register_control(
                &program, node, context, 0, &input, None, Some(&mut visitor), legacy,
            ), Err(Error::Context)));
            assert_eq!(visitor.calls, 1);
            assert!(visitor.cancelled);
        }
    }
}

// These callbacks never manufacture NativeOutput. Positive execution remains
// the real HarnessDriver/native_pair regression above; admission tests only
// count whether the boundary reached a Driver or material visitor at all.
#[derive(Default)]
struct AdmissionCallbacks { calls: usize, cancellations: usize }
impl Driver for AdmissionCallbacks {
    fn run(&mut self, _: Context, _: &Plan<'_>, _: &[u64]) -> Result<NativeOutput, Error> {
        self.calls += 1;
        Err(Error::Context)
    }
    fn cancel(&mut self) { self.cancellations += 1; }
}
impl MaterialVisitor for AdmissionCallbacks {
    fn visit(&mut self, _: Context, _: &Plan<'_>) -> Result<(), Error> {
        self.calls += 1;
        Ok(())
    }
    fn cancel(&mut self) { self.cancellations += 1; }
}

#[test]
fn bound_silu_window_rejections_precede_driver_and_material_callbacks() {
    // Private cfg(test) token corruption exercises validation, not a claim that
    // arbitrary hostile code inside this module cannot forge its private scope.
    let cases = [
        "foreign-program", "foreign-node", "other-frame-same-shape", "short-frame",
        "wrong-selection", "invalid-selection", "wrong-prefix", "invalid-prefix",
        "wrong-node", "invalid-node", "missing-input", "wrong-input-count",
        "wrong-output-count", "wrong-output-shape", "wrong-input-edge", "gelu",
        "wrong-implementation", "invalid-rank", "already-published",
    ];
    for case in cases {
        let (_, semantic, source) = dense_fixture();
        let mut program = Program::compile(&semantic, source, 2, Limits::default()).unwrap();
        let (_, foreign_semantic, foreign_source) = dense_fixture();
        let foreign = Program::compile(&foreign_semantic, foreign_source, 2, Limits::default()).unwrap();
        assert_eq!(program.digest(), foreign.digest());
        assert_eq!(program.source.digest(), foreign.source.digest());
        assert!(!std::ptr::eq(&program, &foreign));
        assert!(!std::ptr::eq(program.nodes.as_slice(), foreign.nodes.as_slice()));
        assert!(!std::ptr::eq(program.source.tensors.as_slice(), foreign.source.tensors.as_slice()));
        let node = program.nodes.iter().position(|n| matches!(n.code, Code::Activation { silu: true, .. })).unwrap();
        let Code::Activation { input, .. } = program.nodes[node].code else { unreachable!() };
        let count = graph_elements(program.capacity(), &program.nodes[node].shape).unwrap();
        match case {
            "wrong-output-shape" => program.nodes[node].shape = vec![count],
            "wrong-input-edge" => program.nodes[node].code = Code::Activation { input: node, silu: true },
            "gelu" => program.nodes[node].code = Code::Activation { input, silu: false },
            "wrong-implementation" => program.nodes[node].code = Code::Identity(input),
            "invalid-rank" => program.nodes[node].shape = vec![1; 5],
            _ => {}
        }
        let mut frame = vector(program.nodes.len()).unwrap();
        frame.resize_with(program.nodes.len(), || None);
        frame[input] = Some(Zeroizing::new(zeros(count).unwrap()));
        match case {
            "missing-input" => { frame[input] = None; }
            "wrong-input-count" => { frame[input] = Some(Zeroizing::new(zeros(count - 1).unwrap())); }
            "already-published" => { frame[node] = Some(Zeroizing::new(zeros(count).unwrap())); }
            _ => {}
        }
        let mut other_frame = vector(program.nodes.len()).unwrap();
        other_frame.resize_with(program.nodes.len(), || None);
        other_frame[input] = Some(Zeroizing::new(zeros(count).unwrap()));
        let context = Context {
            plan: program.digest(), session: [7; 32], operation: [8; 32], first_tensor_index: 0,
        };
        for driven in [false, true] {
            let budget = request_budget(&program).unwrap();
            let tables = request_tables(&program, &budget).unwrap();
            let mut driver = AdmissionCallbacks::default();
            let mut visitor = AdmissionCallbacks::default();
            let mut schedule = Schedule::new(&program, context, 0, None, &tables).unwrap();
            schedule.node = if case == "wrong-node" { node + 1 } else { node };
            if case == "invalid-node" { schedule.node = program.nodes.len(); }
            schedule.selection = if case == "invalid-selection" { program.selections } else { 0 };
            if driven { schedule.driver = Some(&mut driver); }
            schedule.visitor = Some(&mut visitor);
            let active = ActiveRegisterFrame {
                program: &program, frame: &frame, selection: schedule.selection,
                prefix: if case == "invalid-prefix" { program.sequence + 1 } else { program.initial },
            };
            let window = DispatchWindow {
                program: if case == "foreign-program" { &foreign } else { &program },
                node: if case == "foreign-node" || case == "foreign-program" { &foreign.nodes[node] } else { &program.nodes[node] },
                index: if case == "invalid-node" { program.nodes.len() } else { node },
                selection: if case == "wrong-selection" { 1 } else { active.selection },
                prefix: if case == "wrong-prefix" { program.initial + 1 } else { active.prefix },
                frame: match case {
                    "other-frame-same-shape" => &other_frame,
                    "short-frame" => &frame[..frame.len() - 1],
                    _ => &frame,
                },
                active: &active, schedule: &mut schedule,
            };
            // Output already exists, as in dispatch_context; rejection is not a
            // claim to happen before model/output allocation or table bootstrap.
            let mut output = Zeroizing::new(zeros(if case == "wrong-output-count" { count - 1 } else { count }).unwrap());
            let result = Schedule::admit_silu(window, &mut output);
            assert!(result.is_err(), "{case}, driven={driven}");
            drop(result);
            drop(schedule);
            assert_eq!((driver.calls, driver.cancellations, visitor.calls, visitor.cancellations), (0, 0, 0, 0), "{case}, driven={driven}");
        }
    }
}

#[test]
fn bound_silu_factory_requires_driver_for_native_and_keeps_named_diagnostic_walk() {
    for online in [OnlineSchedule::Serial, OnlineSchedule::WavefrontV1, OnlineSchedule::TensorV1] {
        let (_, semantic, source) = dense_fixture();
        let program = Program::compile_with_schedule(&semantic, source, 2, Limits::default(), online).unwrap();
        let node = program.nodes.iter().position(|n| matches!(n.code, Code::Activation { silu: true, .. })).unwrap();
        let Code::Activation { input, .. } = program.nodes[node].code else { unreachable!() };
        let count = graph_elements(program.capacity(), &program.nodes[node].shape).unwrap();
        let mut frame = vector(program.nodes.len()).unwrap();
        frame.resize_with(program.nodes.len(), || None);
        frame[input] = Some(Zeroizing::new(zeros(count).unwrap()));
        let context = Context {
            plan: program.digest(), session: [7; 32], operation: [8; 32], first_tensor_index: 0,
        };
        for driven in [false, true] {
            let budget = request_budget(&program).unwrap();
            let tables = request_tables(&program, &budget).unwrap();
            let mut driver = AdmissionCallbacks::default();
            let mut visitor = AdmissionCallbacks::default();
            let mut schedule = Schedule::new(&program, context, 0, None, &tables).unwrap();
            schedule.node = node;
            if driven { schedule.driver = Some(&mut driver); }
            schedule.visitor = Some(&mut visitor);
            let active = ActiveRegisterFrame { program: &program, frame: &frame, selection: 0, prefix: program.initial };
            let window = DispatchWindow {
                schedule: &mut schedule, active: &active, program: &program, node: &program.nodes[node],
                index: node, selection: 0, prefix: program.initial, frame: &frame,
            };
            let mut output = Zeroizing::new(zeros(count).unwrap());
            match Schedule::admit_silu(window, &mut output).unwrap() {
                AdmittedSiluInvocation::Native(invocation) => {
                    assert!(driven);
                    // Admission only. No fake NativeOutput or execution credit.
                    drop(invocation);
                }
                AdmittedSiluInvocation::Diagnostic(invocation) => {
                    assert!(!driven);
                    bound_tensor_kernel::diagnose_silu(invocation).unwrap();
                }
            }
            drop(schedule);
            assert_eq!(driver.calls, 0);
            assert_eq!(visitor.calls > 0, !driven);
            assert_eq!((driver.cancellations, visitor.cancellations), (0, 0));
            assert!(output.iter().all(|&value| value == 0));
        }
    }
}
fn reconstructed(a: &[u64], b: &[u64]) -> Vec<i64> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| signed(a.wrapping_add(b) & MASK))
        .collect()
}

fn execute_fixture(program: &Program, token: u64) -> ((TokenShares, Trace), (TokenShares, Trace)) {
    execute_prefix(program, &[token])
}

fn execute_prefix(
    program: &Program,
    prefix: &[u64],
) -> ((TokenShares, Trace), (TokenShares, Trace)) {
    let context = Context {
        plan: program.digest(),
        session: [7; 32],
        operation: [8; 32],
        first_tensor_index: 0,
    };
    let mut visits = Visits::default();
    program.visit_material(context, &mut visits).unwrap();
    assert!(!visits.cancelled);
    assert_eq!(visits.records.len(), program.resources().kernels);
    assert!(visits.requests > 0);
    assert_eq!(
        visits.requests,
        program.resources().primitive_requests_per_party
    );
    let expected = visits.records.as_slice();
    let (sender, receiver) = mpsc::sync_channel(0);
    let left: Vec<_> = (0..prefix.len())
        .map(|i| 123456789u64 + i as u64 * 71893)
        .collect();
    let right: Vec<_> = prefix
        .iter()
        .zip(&left)
        .map(|(&id, &share)| id.wrapping_sub(share) & MASK)
        .collect();
    thread::scope(|scope| {
        let a = scope.spawn(move || {
            let mut driver = HarnessDriver {
                role: 0,
                send: None,
                receive: Some(receiver),
                trace: Vec::new(),
                cancelled: false,
                expected,
                next: 0,
            };
            let result = Worker::new(program, 0, context, Zeroizing::new(left))
                .unwrap()
                .execute(&mut driver);
            assert!(!driver.cancelled, "worker0: {:?}", result.as_ref().err());
            assert_eq!(driver.next, expected.len());
            (result.unwrap(), driver.trace)
        });
        let b = scope.spawn(move || {
            let mut driver = HarnessDriver {
                role: 1,
                send: Some(sender),
                receive: None,
                trace: Vec::new(),
                cancelled: false,
                expected,
                next: 0,
            };
            let result = Worker::new(program, 1, context, Zeroizing::new(right))
                .unwrap()
                .execute(&mut driver);
            assert!(!driver.cancelled, "worker1: {:?}", result.as_ref().err());
            assert_eq!(driver.next, expected.len());
            (result.unwrap(), driver.trace)
        });
        (a.join().unwrap(), b.join().unwrap())
    })
}

#[test]
fn complete_generated_gpt2_two_selections_native_dealer_workers() {
    let (semantic, source) = fixture();
    let program = Program::compile(&semantic, source, 2, Limits::default()).unwrap();
    assert_eq!(program.profile(), "fusefss.gpt2-complete-ring50-q12.v1");
    assert!(program.resources().kernels > 50);
    assert!(program.resources().material_bytes > 0);
    let (a, b) = execute_fixture(&program, 1);
    // Independent Python source-table oracle, generated from identical nonzero
    // public tensors. EOS=0 deliberately does not stop fixed-T recomputation.
    assert_eq!(reconstructed(a.0.values(), b.0.values()), [0, 0]);
    assert_eq!(a.1.len(), 2 * program.nodes.len());
    assert_eq!(a.1.len(), b.1.len());
    let index = |name: &str| {
        semantic
            .prefill
            .operations
            .iter()
            .position(|op| op.id == name)
            .unwrap()
    };
    let check = |step: usize, name: &str, expected: &[i64]| {
        let n = index(name);
        let i = step * program.nodes.len() + n;
        assert_eq!((a.1[i].0, a.1[i].1), (step, n));
        let output = reconstructed(&a.1[i].2, &b.1[i].2);
        assert_eq!(&output[..expected.len()], expected, "{step}:{name}");
    };
    check(0, "embedding_sum", &[-1024, 1664]);
    check(1, "embedding_sum", &[-1024, 1664, -2432, -1408]);
    check(0, "layer.0.input_norm", &[-4069, 4132]);
    check(1, "layer.0.input_norm", &[-4069, 4132, -4064, 4128]);
    check(0, "layer.0.post_norm", &[-4072, 4167]);
    check(1, "layer.0.post_norm", &[-4072, 4167, -4065, 4160]);
    check(0, "layer.0.gelu", &[-255, 136]);
    check(1, "layer.0.gelu", &[-255, 136, -255, 136]);
    check(0, "final_norm", &[-4162, 4066]);
    check(1, "final_norm", &[-4162, 4066, -4157, 4067]);
    check(0, "output_head", &[1318, 1276, -3086]);
    check(1, "output_head", &[1315, 1276, -3084]);
}

fn dense_fixture() -> (serde_json::Value, DecoderPlan, Source) {
    let config = json!({"model_type":"qwen2","hidden_size":2,"intermediate_size":2,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"vocab_size":3,"max_position_embeddings":4,"hidden_act":"silu","rms_norm_eps":"0.000001","rope_theta":10000,"tie_word_embeddings":true});
    let semantic = pllm_models::lower_model_json(
        &serde_json::to_vec(&config).unwrap(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: 1,
            max_new_tokens: 2,
        },
    )
    .unwrap();
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut rotary = Vec::new();
    let mut linear = 0;
    let mut norm = 0;
    for op in &semantic.prefill.operations {
        if op.operator == Op::RotaryEmbedding {
            rotary.push(
                RotaryCoefficients::from_f32(
                    &op.id,
                    10000,
                    [2, 2],
                    &[
                        1.,
                        1.,
                        f32::from_bits(0x3f0a5140),
                        f32::from_bits(0x3f0a5140),
                    ],
                    &[
                        0.,
                        0.,
                        f32::from_bits(0x3f576aa4),
                        f32::from_bits(0x3f576aa4),
                    ],
                )
                .unwrap(),
            );
        }
        let Some(id) = op.attributes["weight"].as_str() else {
            continue;
        };
        if tensors.iter().any(|t| t.id() == id) {
            continue;
        }
        match op.operator {
            Op::TokenLookup => {
                let values: Vec<_> = (0..3)
                    .flat_map(|r| (0..2).map(move |c| ((r * 7 + c * 5) % 17 - 8) * 256))
                    .collect();
                tensors.push(Tensor::from_q12(id, &[3, 2], &values).unwrap());
            }
            Op::RmsNorm => {
                let gamma = match norm {
                    0 => [4096, 4160],
                    1 => [4032, 4096],
                    _ => [4160, 4032],
                };
                norm += 1;
                tensors.push(Tensor::from_q12(id, &[2], &gamma).unwrap());
            }
            Op::Linear => {
                let (kind, offset) =
                    [(0, 0), (0, 2), (0, 4), (1, 0), (4, 0), (2, 0), (3, 0)][linear];
                let values: Vec<_> = (offset..offset + 2)
                    .flat_map(|r| (0..2).map(move |c| ((r * 7 + c * 3 + kind * 5) % 11 - 5) * 64))
                    .collect();
                tensors.push(Tensor::from_q12(id, &[2, 2], &values).unwrap());
                if let Some(bias) = op.attributes["bias"].as_str() {
                    let values: Vec<_> = (offset..offset + 2)
                        .map(|r| ((r + kind) % 5 - 2) * 32)
                        .collect();
                    tensors.push(Tensor::from_q12(bias, &[2], &values).unwrap());
                }
                linear += 1;
            }
            _ => panic!("unexpected source artifact"),
        }
    }
    let source = Source::new(tensors).unwrap().with_rotary(rotary).unwrap();
    (config, semantic, source)
}

#[test]
fn complete_dense_source_rms_rope_silu_native_two_selections() {
    let (_, semantic, source) = dense_fixture();
    let program = Program::compile(&semantic, source, 2, Limits::default()).unwrap();
    assert_eq!(program.profile(), "fusefss.dense-rope-ring50-q12.v1");
    let (a, b) = execute_fixture(&program, 1);
    assert_eq!(reconstructed(a.0.values(), b.0.values()), [1, 1]);
    let check = |step: usize, name: &str, expected: &[i64]| {
        let n = semantic
            .prefill
            .operations
            .iter()
            .position(|op| op.id == name)
            .unwrap();
        let i = step * program.nodes.len() + n;
        assert_eq!(
            &reconstructed(&a.1[i].2, &b.1[i].2)[..expected.len()],
            expected,
            "{step}:{name}"
        );
    };
    check(0, "layer.0.input_norm", &[-1405, 5706]);
    check(1, "layer.0.rope_q", &[-133, 369, -383, 87]);
    check(1, "layer.0.rope_k", &[133, -346, 362, -76]);
    check(0, "layer.0.silu", &[-211, 139]);
    check(1, "layer.0.silu", &[-211, 139, -211, 139]);
    check(1, "final_norm", &[-1466, 5529, -1466, 5524]);
    check(1, "output_head", &[-303, 1472, -2622]);
}

#[test]
fn rope_signed_rotated_operand_floors_before_addition() {
    let budget = Budget::new(pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [1; 32],
        contract_digest: [2; 32],
    };
    let range = gates::InputRange { min: -3, max: -1 };
    let (plan, _) = kernels::rope(binding, [range; 2], 2048, 1024, &budget).unwrap();
    let context = Context {
        plan: plan.digest(),
        session: [3; 32],
        operation: [4; 32],
        first_tensor_index: 0,
    };
    let a = [1234, 5678];
    let b = [
        word(-3).wrapping_sub(a[0]) & MASK,
        word(-1).wrapping_sub(a[1]) & MASK,
    ];
    let [left, right] = native_pair(&plan, context, &a, &b).unwrap();
    // floor(-3/2)+floor(1/4) = -2, not floor(-3/2)-floor(-1/4) = -1.
    assert_eq!(reconstructed(&left.values, &right.values), [-2, -2]);
    let r = super::super::program::rotary_output_range(Range::new(-3, -3).unwrap(), 2048, 1024)
        .unwrap();
    assert_eq!((r.lo, r.hi), (-3, -2));
}

#[test]
fn frozen_rope_signed_operand_one_unit_witness() {
    let budget = Budget::new(pllm_garble::fusefss_backend::MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [1; 32],
        contract_digest: [2; 32],
    };
    let (plan, _) = kernels::rope(
        binding,
        [
            gates::InputRange { min: 339, max: 339 },
            gates::InputRange {
                min: 1165,
                max: 1165,
            },
        ],
        2213,
        3447,
        &budget,
    )
    .unwrap();
    let context = Context {
        plan: plan.digest(),
        session: [3; 32],
        operation: [4; 32],
        first_tensor_index: 0,
    };
    let a = [1234, 5678];
    let b = [
        339u64.wrapping_sub(a[0]) & MASK,
        1165u64.wrapping_sub(a[1]) & MASK,
    ];
    let [left, right] = native_pair(&plan, context, &a, &b).unwrap();
    assert_eq!(reconstructed(&left.values, &right.values), [-798, 914]);
}

#[test]
fn shape_source_and_work_obligations_fail_before_driver() {
    let (plan, source) = fixture();
    let limits = Limits {
        max_work: 1,
        ..Limits::default()
    };
    assert!(matches!(
        Program::compile(&plan, source, 2, limits),
        Err(Error::Resource)
    ));
    let (plan, source) = fixture();
    assert!(Program::compile(&plan, source, 0, Limits::default()).is_err());
    let (plan, mut source) = fixture();
    source.tensors[0].shape[0] += 1;
    assert!(Program::compile(&plan, source, 2, Limits::default()).is_err());
}

#[test]
fn driver_error_consumes_worker_and_cancels_transport() {
    struct Fail(bool);
    impl Driver for Fail {
        fn run(&mut self, _: Context, _: &Plan<'_>, _: &[u64]) -> Result<NativeOutput, Error> {
            Err(Error::Context)
        }
        fn cancel(&mut self) {
            self.0 = true;
        }
    }
    let (plan, source) = fixture();
    let program = Program::compile(&plan, source, 2, Limits::default()).unwrap();
    let context = Context {
        plan: program.digest(),
        session: [3; 32],
        operation: [4; 32],
        first_tensor_index: 0,
    };
    let mut input = Vec::with_capacity(8);
    input.push(0);
    let worker = Worker::new(&program, 0, context, Zeroizing::new(input)).unwrap();
    let mut driver = Fail(false);
    assert!(matches!(worker.execute(&mut driver), Err(Error::Context)));
    assert!(driver.0);
}

#[test]
fn material_visit_failure_cancels_and_retires_retained_owners() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Retained(Arc<AtomicUsize>);
    impl Drop for Retained {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    struct Failing {
        live: Arc<AtomicUsize>,
        retained: Vec<Retained>,
        calls: usize,
        cancelled: bool,
    }
    impl MaterialVisitor for Failing {
        fn visit(&mut self, _: Context, _: &Plan<'_>) -> Result<(), Error> {
            self.calls += 1;
            self.live.fetch_add(1, Ordering::SeqCst);
            self.retained.push(Retained(self.live.clone()));
            if self.calls == 3 {
                Err(Error::Context)
            } else {
                Ok(())
            }
        }
        fn cancel(&mut self) {
            self.cancelled = true;
            self.retained.clear();
        }
    }
    let (plan, source) = fixture();
    let program = Program::compile(&plan, source, 2, Limits::default()).unwrap();
    let context = Context {
        plan: program.digest(),
        session: [7; 32],
        operation: [8; 32],
        first_tensor_index: 0,
    };
    let live = Arc::new(AtomicUsize::new(0));
    let mut visitor = Failing {
        live: live.clone(),
        retained: Vec::new(),
        calls: 0,
        cancelled: false,
    };
    assert!(matches!(
        program.visit_material(context, &mut visitor),
        Err(Error::Context)
    ));
    assert_eq!(visitor.calls, 3);
    assert!(visitor.cancelled);
    assert_eq!(live.load(Ordering::SeqCst), 0);
    let mut visitor = Visits::default();
    let bad = Context {
        plan: [0; 32],
        ..context
    };
    assert!(matches!(
        program.visit_material(bad, &mut visitor),
        Err(Error::Context)
    ));
    assert!(visitor.cancelled);
    assert!(visitor.records.is_empty());
}

#[path = "decoder_execution_tests.rs"]
mod decoder_execution_tests;

#[path = "checked_range_tests.rs"]
mod checked_range_tests;
