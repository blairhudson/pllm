//! Native-only test orchestrator. Two worker threads rendezvous by Plan/context;
//! a separate scoped dealer issues fresh paired material. Only this module ever
//! holds both roles or reconstructs diagnostic outputs.
use super::*;
use pllm_garble::fusefss_backend::composition::Event;
use pllm_models::{DecoderPlan, DecoderWorkload, ModelOperator as Op};
use serde_json::json;
use std::{sync::mpsc, thread};

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
fn reconstructed(a: &[u64], b: &[u64]) -> Vec<i64> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| signed(a.wrapping_add(b) & MASK))
        .collect()
}

fn execute_fixture(program: &Program, token: u64) -> ((TokenShares, Trace), (TokenShares, Trace)) {
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
    let share = 123456789u64;
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
            let result = Worker::new(program, 0, context, Zeroizing::new(vec![share]))
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
            let result = Worker::new(
                program,
                1,
                context,
                Zeroizing::new(vec![token.wrapping_sub(share) & MASK]),
            )
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

#[test]
fn complete_dense_source_rms_rope_silu_native_two_selections() {
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
