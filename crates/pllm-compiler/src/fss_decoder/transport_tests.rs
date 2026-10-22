use super::*;

#[path = "transport_batch_tests.rs"]
mod batching;
#[path = "transport_diagnostics_admission_tests.rs"]
mod diagnostics;
use crate::fss_decoder::{Limits, Source, Tensor};
use pllm_models::{DecoderPlan, DecoderWorkload, ModelOperator as Op};
use serde_json::json;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread,
};

const MASK: u64 = (1u64 << 50) - 1;
type Trace = Vec<(usize, usize, Zeroizing<Vec<u64>>)>;

/// Review witness, not a proposed lifecycle: context binding rejects old bytes,
/// but does not preserve ImportGuard's stronger context-independent nonce ledger.
/// Emulate a trusted issuer repeating an envelope nonce on genuinely FRESH
/// material; no private key/mask bytes are reused or changed. This is not an
/// authenticated-network forgery test and does not claim malicious-dealer safety.
#[test]
fn retiring_guard_changes_same_nonce_different_context_contract() {
    use pllm_garble::fusefss_backend::{
        composition::{gates, kernels},
        wire::MaterialPacket,
    };
    let budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let binding = kernels::Binding {
        source_digest: [11; 32],
        contract_digest: [12; 32],
    };
    let (plan, _) = kernels::public_scale_q12(
        binding,
        gates::InputRange { min: -4, max: 4 },
        2048,
        gates::InputRange {
            min: 2048,
            max: 2048,
        },
        &budget,
    )
    .unwrap();
    let first = Context {
        plan: [13; 32],
        session: [14; 32],
        operation: [15; 32],
        first_tensor_index: 0,
    };
    let second = Context {
        operation: [16; 32],
        ..first
    };
    let pc = plan.material_steps().next().unwrap();
    let [a, b] = plan.issue_step(pc, first).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let nonce = packet.nonce();
    let original = Zeroizing::new(packet.as_bytes().to_vec());
    drop(packet);
    let [a, b] = plan.issue_step(pc, second).unwrap();
    drop(b);
    let packet = a.export_packet().unwrap();
    let mut fresh = Zeroizing::new(packet.as_bytes().to_vec());
    drop(packet);
    fresh[48..80].copy_from_slice(&nonce);
    let checksum_at = fresh.len() - 32;
    let checksum = Sha256::digest(&fresh[..checksum_at]);
    fresh[checksum_at..].copy_from_slice(&checksum);
    let mut request_guard = ImportGuard::new(2, &budget).unwrap();
    drop(
        plan.import_step(
            pc,
            first,
            0,
            nonce,
            MaterialPacket::from_bytes(&original, &budget).unwrap(),
            &mut request_guard,
        )
        .unwrap(),
    );
    assert!(matches!(
        plan.import_step(
            pc,
            second,
            0,
            nonce,
            MaterialPacket::from_bytes(&fresh, &budget).unwrap(),
            &mut request_guard
        ),
        Err(Error::Consumed)
    ));
    // A fresh per-kernel guard silently accepts the same (role,nonce).
    let mut replacement_guard = ImportGuard::new(2, &budget).unwrap();
    drop(
        plan.import_step(
            pc,
            second,
            0,
            nonce,
            MaterialPacket::from_bytes(&fresh, &budget).unwrap(),
            &mut replacement_guard,
        )
        .unwrap(),
    );
    // Old, unretagged bytes still fail the immutable expected context. This
    // narrower fact alone is insufficient to justify dropping the global ledger.
    let mut guard = ImportGuard::new(2, &budget).unwrap();
    assert!(matches!(
        plan.import_step(
            pc,
            second,
            0,
            nonce,
            MaterialPacket::from_bytes(&original, &budget).unwrap(),
            &mut guard
        ),
        Err(Error::Context)
    ));
    let mut guard = ImportGuard::new(2, &budget).unwrap();
    assert!(matches!(
        plan.import_step(
            pc,
            first,
            1,
            nonce,
            MaterialPacket::from_bytes(&original, &budget).unwrap(),
            &mut guard
        ),
        Err(Error::Context)
    ));
}

fn fixture(layers: u32, initial: u64, selections: u64) -> Program {
    fixture_schedule(
        layers,
        initial,
        selections,
        crate::fss_decoder::OnlineSchedule::Serial,
    )
}

fn fixture_schedule(
    layers: u32,
    initial: u64,
    selections: u64,
    online: crate::fss_decoder::OnlineSchedule,
) -> Program {
    let config = json!({"model_type":"gpt2","n_embd":2,"n_head":1,"n_layer":layers,"n_positions":8,"vocab_size":3,"n_inner":2});
    let plan = pllm_models::lower_model_json(
        &serde_json::to_vec(&config).unwrap(),
        DecoderWorkload {
            batch: 1,
            max_input_tokens: initial,
            max_new_tokens: selections,
        },
    )
    .unwrap();
    let source = source(&plan);
    Program::compile_with_schedule(
        &plan,
        source,
        selections as usize,
        Limits::default(),
        online,
    )
    .unwrap()
}
fn source(plan: &DecoderPlan) -> Source {
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut linear = 0;
    let mut norm = 0;
    for op in &plan.prefill.operations {
        let Some(id) = op.attributes["weight"].as_str() else {
            continue;
        };
        if tensors.iter().any(|t| t.id() == id) {
            continue;
        }
        match op.operator {
            Op::TokenLookup => {
                let position = op.inputs[0] != "input.tokens";
                let rows = if position { 8 } else { 3 };
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
                let (gamma, beta) = match norm % 3 {
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
                let kind = linear % 4;
                let columns = if kind == 0 { 6 } else { 2 };
                let mut values = Vec::new();
                for c in 0..2 {
                    for r in 0..columns {
                        values.push(((r * 7 + c * 3 + kind * 5) % 11 - 5) * 64);
                    }
                }
                let bias: Vec<_> = (0..columns).map(|r| ((r + kind) % 5 - 2) * 32).collect();
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
            _ => panic!("unexpected source"),
        }
    }
    Source::new(tensors).unwrap()
}
fn context(program: &Program) -> Context {
    Context {
        plan: program.digest(),
        session: [7; 32],
        operation: [8; 32],
        first_tensor_index: 0,
    }
}
fn same_context(a: Context, b: Context) -> bool {
    (a.plan, a.session, a.operation, a.first_tensor_index)
        == (b.plan, b.session, b.operation, b.first_tensor_index)
}
fn same_kernel(a: Kernel, b: Kernel) -> bool {
    a.ordinal == b.ordinal && a.plan_digest == b.plan_digest && same_context(a.context, b.context)
}

struct Queue<T> {
    state: Mutex<(VecDeque<T>, bool)>,
    changed: Condvar,
}
impl<T> Queue<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new((VecDeque::new(), false)),
            changed: Condvar::new(),
        }
    }
    fn send(&self, value: T) -> Result<(), Error> {
        let mut s = self.state.lock().unwrap();
        while s.0.len() == 1 && !s.1 {
            s = self.changed.wait(s).unwrap();
        }
        if s.1 {
            return Err(Error::Consumed);
        }
        s.0.push_back(value);
        self.changed.notify_all();
        Ok(())
    }
    fn receive(&self) -> Result<T, Error> {
        let mut s = self.state.lock().unwrap();
        while s.0.is_empty() && !s.1 {
            s = self.changed.wait(s).unwrap();
        }
        let out = s.0.pop_front().ok_or(Error::Consumed);
        self.changed.notify_all();
        out
    }
    fn drained(&self) -> Result<(), Error> {
        let mut s = self.state.lock().unwrap();
        while !s.0.is_empty() && !s.1 {
            s = self.changed.wait(s).unwrap();
        }
        if s.1 {
            Err(Error::Consumed)
        } else {
            Ok(())
        }
    }
    fn close(&self) {
        let mut s = self.state.lock().unwrap();
        s.1 = true;
        s.0.clear();
        self.changed.notify_all();
    }
    fn len(&self) -> usize {
        self.state.lock().unwrap().0.len()
    }
}
struct Cancel {
    close: Vec<Box<dyn Fn() + Send + Sync>>,
}
impl Cancel {
    fn all(&self) {
        for f in &self.close {
            f();
        }
    }
}
struct Material {
    meta: MaterialRequest,
    bytes: Zeroizing<Vec<u8>>,
}
struct Frame {
    meta: PeerRequest,
    bytes: Zeroizing<Vec<u8>>,
}
#[derive(Default)]
struct Counts {
    dealer: AtomicUsize,
    worker: [AtomicUsize; 2],
    delivered: AtomicUsize,
}
struct DealerEndpoint {
    material: [Arc<Queue<Material>>; 2],
    cancel: Arc<Cancel>,
    counts: Arc<Counts>,
    settled: bool,
    fail_after: Option<usize>,
}
impl DealerTransport for DealerEndpoint {
    fn deliver(&mut self, meta: MaterialRequest, packet: &[u8]) -> Result<(), Error> {
        let n = self.counts.delivered.fetch_add(1, Ordering::SeqCst);
        if self.fail_after == Some(n) {
            return Err(Error::Context);
        }
        self.material[meta.role as usize].send(Material {
            meta,
            bytes: Zeroizing::new(packet.to_vec()),
        })
    }
    fn finish(&mut self) -> Result<(), Error> {
        for q in &self.material {
            q.drained()?;
        }
        self.settled = true;
        Ok(())
    }
    fn cancel(&mut self) {
        if !self.settled {
            self.settled = true;
            self.counts.dealer.fetch_add(1, Ordering::SeqCst);
            self.cancel.all();
        }
    }
}
impl Drop for DealerEndpoint {
    fn drop(&mut self) {
        if !self.settled {
            self.cancel();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    CorruptMaterial,
    ReplayMaterial,
    ReplayMaterialAfter4096,
    DowngradeMaterial,
    NoReply,
    DoubleWrite,
    CallbackError,
    CorruptFrame,
    ReplayFrame,
    FinishError,
    Panic,
}
struct RoleEndpoint {
    role: u8,
    material: Arc<Queue<Material>>,
    send: Arc<Queue<Frame>>,
    receive: Arc<Queue<Frame>>,
    cancel: Arc<Cancel>,
    counts: Arc<Counts>,
    settled: bool,
    fault: Fault,
    material_calls: usize,
    frame_calls: usize,
    saved: Option<Zeroizing<Vec<u8>>>,
    trace: Trace,
}
impl WorkerTransport for RoleEndpoint {
    fn material(&mut self, meta: MaterialRequest, reply: &mut Reply) -> Result<(), Error> {
        let packet = self.material.receive()?;
        if !same_kernel(meta.kernel, packet.meta.kernel)
            || meta.pc != packet.meta.pc
            || meta.role != packet.meta.role
        {
            return Err(Error::Context);
        }
        self.material_calls += 1;
        let mut bytes = packet.bytes;
        match self.fault {
            Fault::Panic => panic!("injected consumer drop"),
            Fault::CallbackError => return Err(Error::Context),
            Fault::CorruptMaterial => {
                let n = bytes.len() - 1;
                bytes[n] ^= 1;
            }
            Fault::DowngradeMaterial => bytes[7] = b'0',
            Fault::NoReply => return Ok(()),
            Fault::DoubleWrite => {
                reply.write(&bytes)?;
                assert!(reply.write(&bytes).is_err());
                return Ok(());
            }
            Fault::ReplayMaterial => {
                if let Some(saved) = &self.saved {
                    return reply.write(saved);
                }
                self.saved = Some(Zeroizing::new(bytes.to_vec()));
            }
            Fault::ReplayMaterialAfter4096 => {
                if self.material_calls > 4096 {
                    return reply.write(self.saved.as_ref().unwrap());
                }
                if self.saved.is_none() {
                    self.saved = Some(Zeroizing::new(bytes.to_vec()));
                }
            }
            _ => {}
        }
        reply.write(&bytes)
    }
    fn exchange(
        &mut self,
        meta: PeerRequest,
        frame: &[u8],
        reply: &mut Reply,
    ) -> Result<(), Error> {
        self.send.send(Frame {
            meta,
            bytes: Zeroizing::new(frame.to_vec()),
        })?;
        let peer = self.receive.receive()?;
        if !same_kernel(meta.kernel, peer.meta.kernel)
            || meta.frame != peer.meta.frame
            || meta.role == peer.meta.role
        {
            return Err(Error::Context);
        }
        self.frame_calls += 1;
        let mut bytes = peer.bytes;
        if self.fault == Fault::CorruptFrame {
            bytes[0] ^= 1;
        }
        if self.fault == Fault::ReplayFrame {
            if let Some(saved) = &self.saved {
                return reply.write(saved);
            }
            self.saved = Some(Zeroizing::new(bytes.to_vec()));
        }
        reply.write(&bytes)
    }
    fn finish(&mut self) -> Result<(), Error> {
        if self.fault == Fault::FinishError {
            return Err(Error::Context);
        }
        self.material.drained()?;
        self.send.drained()?;
        self.receive.drained()?;
        self.settled = true;
        Ok(())
    }
    fn cancel(&mut self) {
        if !self.settled {
            self.settled = true;
            self.counts.worker[self.role as usize].fetch_add(1, Ordering::SeqCst);
            self.saved = None;
            self.cancel.all();
        }
    }
    fn observe(&mut self, selection: usize, node: usize, _shape: &[usize], shares: &[u64]) {
        self.trace
            .push((selection, node, Zeroizing::new(shares.to_vec())));
    }
}
impl Drop for RoleEndpoint {
    fn drop(&mut self) {
        if !self.settled {
            self.cancel();
        }
    }
}
struct Inspection {
    materials: [Arc<Queue<Material>>; 2],
    frames: [Arc<Queue<Frame>>; 2],
    counts: Arc<Counts>,
}
impl Inspection {
    fn drained(&self) {
        for q in &self.materials {
            assert_eq!(q.len(), 0);
        }
        for q in &self.frames {
            assert_eq!(q.len(), 0);
        }
    }
}
fn endpoints(fault: Fault) -> (DealerEndpoint, [RoleEndpoint; 2], Inspection) {
    let material = [Arc::new(Queue::new()), Arc::new(Queue::new())];
    let frames = [Arc::new(Queue::new()), Arc::new(Queue::new())];
    let mut close: Vec<Box<dyn Fn() + Send + Sync>> = Vec::new();
    for q in &material {
        let q = Arc::clone(q);
        close.push(Box::new(move || q.close()));
    }
    for q in &frames {
        let q = Arc::clone(q);
        close.push(Box::new(move || q.close()));
    }
    let cancel = Arc::new(Cancel { close });
    let counts = Arc::new(Counts::default());
    let dealer = DealerEndpoint {
        material: material.clone(),
        cancel: cancel.clone(),
        counts: counts.clone(),
        settled: false,
        fail_after: None,
    };
    let roles = std::array::from_fn(|role| RoleEndpoint {
        role: role as u8,
        material: material[role].clone(),
        send: frames[role].clone(),
        receive: frames[1 - role].clone(),
        cancel: cancel.clone(),
        counts: counts.clone(),
        settled: false,
        fault: if role == 1 { fault } else { Fault::None },
        material_calls: 0,
        frame_calls: 0,
        saved: None,
        trace: Vec::new(),
    });
    (
        dealer,
        roles,
        Inspection {
            materials: material,
            frames,
            counts,
        },
    )
}

type Execution = (Result<TokenShares, Error>, Trace);

#[test]
fn wavefront_wire_preserves_complete_tensors_with_fewer_rounds_and_burns_failures() {
    let serial = fixture(1, 2, 2);
    let wave = fixture_schedule(1, 2, 2, crate::fss_decoder::OnlineSchedule::WavefrontV1);
    assert_eq!(serial.source_digest(), wave.source_digest());
    assert_ne!(serial.digest(), wave.digest());
    let a = admit(&serial, context(&serial)).unwrap();
    let b = admit(&wave, context(&wave)).unwrap();
    assert!(b.frames_per_party() < a.frames_per_party() / 2);
    assert!(b.combined_reservation_bytes() <= MAX_ALLOCATION_BYTES);
    let (dealer, sa, sb, si) = serialized(&serial, Fault::None);
    dealer.unwrap();
    si.drained();
    let (dealer, wa, wb, wi) = serialized(&wave, Fault::None);
    dealer.unwrap();
    wi.drained();
    assert_eq!(
        open(wa.0.unwrap().values(), wb.0.unwrap().values()),
        open(sa.0.unwrap().values(), sb.0.unwrap().values())
    );
    assert_eq!(wa.1.len(), sa.1.len());
    for ((a, b), (c, d)) in wa.1.iter().zip(&wb.1).zip(sa.1.iter().zip(&sb.1)) {
        assert_eq!((a.0, a.1), (c.0, c.1));
        assert_eq!(open(&a.2, &b.2), open(&c.2, &d.2));
    }
    for fault in [
        Fault::ReplayMaterial,
        Fault::CorruptMaterial,
        Fault::CorruptFrame,
        Fault::ReplayFrame,
        Fault::NoReply,
        Fault::FinishError,
    ] {
        let (_, a, b, inspection) = serialized(&wave, fault);
        assert!(a.0.is_err() || b.0.is_err(), "{fault:?}");
        inspection.drained();
    }
}
fn inputs(program: &Program) -> [Zeroizing<Vec<u64>>; 2] {
    let left: Vec<_> = (0..program.initial_tokens())
        .map(|i| 123456789 + i as u64 * 31)
        .collect();
    let right = left
        .iter()
        .enumerate()
        .map(|(i, &share)| (1 + i as u64 % 2).wrapping_sub(share) & MASK)
        .collect();
    [Zeroizing::new(left), Zeroizing::new(right)]
}
fn serialized(
    program: &Program,
    fault: Fault,
) -> (Result<(), Error>, Execution, Execution, Inspection) {
    let (mut dealer, [mut left, mut right], inspection) = endpoints(fault);
    let context = context(program);
    let [input_a, input_b] = inputs(program);
    let result = thread::scope(|scope| {
        let d = scope.spawn(move || prepare(program, context, &mut dealer));
        let a = scope.spawn(move || {
            let r = execute(program, 0, context, input_a, &mut left);
            (r, std::mem::take(&mut left.trace))
        });
        let b = scope.spawn(move || {
            let r = execute(program, 1, context, input_b, &mut right);
            (r, std::mem::take(&mut right.trace))
        });
        (d.join().unwrap(), a.join().unwrap(), b.join().unwrap())
    });
    (result.0, result.1, result.2, inspection)
}

#[test]
fn single_peer_frame_bound_does_not_grow_with_cumulative_traffic() {
    let shorter = fixture(1, 1, 2);
    let longer = fixture(2, 1, 2);
    let a = admit(&shorter, context(&shorter)).unwrap();
    let b = admit(&longer, context(&longer)).unwrap();
    assert!(a.peer_frame_bytes_bound() > 0);
    assert!(a.peer_frame_bytes_bound() < shorter.resources().frame_bytes);
    assert_eq!(a.peer_frame_bytes_bound(), b.peer_frame_bytes_bound());
    assert!(b.frames_per_party() > a.frames_per_party());
    assert!(longer.resources().frame_bytes > shorter.resources().frame_bytes);
}

#[test]
fn admission_and_reply_caps_are_pre_input_and_consuming() {
    let program = fixture(1, 1, 2);
    let a = admit(&program, context(&program)).unwrap();
    assert_eq!(
        a.primitive_requests_per_party(),
        program.resources().primitive_requests_per_party
    );
    assert_eq!(a.guard_capacity(), a.primitive_requests_per_party());
    assert!(a.combined_reservation_bytes() <= MAX_ALLOCATION_BYTES);
    assert!(a.largest_packet_bytes() > ENVELOPE_BYTES);
    // A forged public cost cannot allocate or issue before overflow rejection.
    // Real larger schedules have their full exact ledger priced below.
    let mut too_large = fixture(2, 2, 2);
    too_large.resources.primitive_requests_per_party = usize::MAX;
    let (mut dealer, [mut role, _], inspection) = endpoints(Fault::None);
    assert!(matches!(
        prepare(&too_large, context(&too_large), &mut dealer),
        Err(Error::Resource)
    ));
    assert!(matches!(
        execute(
            &too_large,
            0,
            context(&too_large),
            Zeroizing::new(vec![0, 0]),
            &mut role
        ),
        Err(Error::Resource)
    ));
    assert_eq!(inspection.counts.delivered.load(Ordering::SeqCst), 0);
    inspection.drained();
    let mut reply = Reply::new(4).unwrap();
    assert_eq!(reply.bound(), 4);
    assert!(reply.write(&[1, 2, 3, 4, 5]).is_err());
    assert!(reply.bytes().is_err());
    assert!(reply.write(&[1]).is_err());
    let mut reply = Reply::new(4).unwrap();
    reply.write(&[1]).unwrap();
    assert!(reply.write(&[2]).is_err());
    assert!(reply.bytes().is_err());
}

#[test]
fn material_and_peer_wire_failures_burn_and_drain() {
    let program = fixture(1, 1, 2);
    for fault in [
        Fault::CorruptMaterial,
        Fault::ReplayMaterial,
        Fault::DowngradeMaterial,
        Fault::NoReply,
        Fault::DoubleWrite,
        Fault::CallbackError,
        Fault::CorruptFrame,
        Fault::ReplayFrame,
    ] {
        let (_dealer, left, right, inspection) = serialized(&program, fault);
        assert!(right.0.is_err(), "{fault:?}");
        assert!(left.0.is_err(), "{fault:?}");
        assert_eq!(
            inspection.counts.worker[1].load(Ordering::SeqCst),
            1,
            "{fault:?}"
        );
        inspection.drained();
    }
}

#[test]
fn rejected_second_recipient_retires_first_delivery_and_wakes_consumer() {
    let program = fixture(1, 1, 2);
    let (mut dealer, [mut left, mut right], inspection) = endpoints(Fault::None);
    dealer.fail_after = Some(1);
    let context = context(&program);
    thread::scope(|scope| {
        let p = &program;
        let d = scope.spawn(move || prepare(p, context, &mut dealer));
        let a = scope.spawn(move || execute(p, 0, context, Zeroizing::new(vec![0]), &mut left));
        let b = scope.spawn(move || execute(p, 1, context, Zeroizing::new(vec![1]), &mut right));
        assert!(d.join().unwrap().is_err());
        assert!(a.join().unwrap().is_err());
        assert!(b.join().unwrap().is_err());
    });
    assert_eq!(inspection.counts.dealer.load(Ordering::SeqCst), 1);
    inspection.drained();
}

#[test]
fn consumer_unwind_cancels_without_peer_or_dealer_deadlock() {
    let program = fixture(1, 1, 2);
    let (mut dealer, [mut left, mut right], inspection) = endpoints(Fault::Panic);
    let context = context(&program);
    thread::scope(|scope| {
        let p = &program;
        let d = scope.spawn(move || prepare(p, context, &mut dealer));
        let a = scope.spawn(move || execute(p, 0, context, Zeroizing::new(vec![0]), &mut left));
        let b = scope.spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute(p, 1, context, Zeroizing::new(vec![1]), &mut right)
            }))
        });
        assert!(b.join().unwrap().is_err());
        assert!(a.join().unwrap().is_err());
        assert!(d.join().unwrap().is_err());
    });
    inspection.drained();
}

#[test]
fn final_consumer_rejection_burns_output_and_drains() {
    let program = fixture(1, 1, 1);
    let (_dealer, _left, right, inspection) = serialized(&program, Fault::FinishError);
    assert!(right.0.is_err());
    assert_eq!(inspection.counts.worker[1].load(Ordering::SeqCst), 1);
    inspection.drained();
}

// Test-only direct native control: separate dealer thread, no wire export/import.
// This orchestrator holds both roles; no production transport implementation does.
fn direct_pair(
    plan: &Plan<'_>,
    context: Context,
    a: &[u64],
    b: &[u64],
) -> Result<[NativeOutput; 2], Error> {
    thread::scope(|scope| {
        let (send, recv) = mpsc::sync_channel::<usize>(0);
        let (material, materials) = mpsc::sync_channel(0);
        let dealer = scope.spawn(move || {
            while let Ok(pc) = recv.recv() {
                if material.send(plan.issue_step(pc, context)).is_err() {
                    break;
                }
            }
        });
        let mut a = plan.runtime(0, context, a)?;
        let mut b = plan.runtime(1, context, b)?;
        let mut x = a.poll()?;
        let mut y = b.poll()?;
        let result = loop {
            match (x, y) {
                (Event::NeedMaterial { pc: a_pc }, Event::NeedMaterial { pc: b_pc }) => {
                    if a_pc != b_pc {
                        return Err(Error::Context);
                    }
                    send.send(a_pc).map_err(|_| Error::Context)?;
                    let [left, right] = materials.recv().map_err(|_| Error::Context)??;
                    x = a.feed(left)?;
                    y = b.feed(right)?;
                }
                (Event::Frame(left), Event::Frame(right)) => {
                    x = a.advance(right.as_bytes())?;
                    y = b.advance(left.as_bytes())?;
                }
                (Event::Output(left), Event::Output(right)) => {
                    break Ok([
                        NativeOutput::from_native(left)?,
                        NativeOutput::from_native(right)?,
                    ])
                }
                _ => break Err(Error::Context),
            }
        };
        drop(send);
        drop(materials);
        dealer.join().unwrap();
        result
    })
}
struct DirectRequest {
    context: Context,
    digest: [u8; 32],
    input: Zeroizing<Vec<u64>>,
    reply: mpsc::SyncSender<Result<NativeOutput, Error>>,
}
struct Direct {
    role: u8,
    send: Option<mpsc::SyncSender<DirectRequest>>,
    recv: Option<mpsc::Receiver<DirectRequest>>,
    trace: Trace,
}
impl Driver for Direct {
    fn run(
        &mut self,
        context: Context,
        plan: &Plan<'_>,
        input: &[u64],
    ) -> Result<NativeOutput, Error> {
        if self.role == 1 {
            let (reply, receive) = mpsc::sync_channel(0);
            self.send
                .as_ref()
                .unwrap()
                .send(DirectRequest {
                    context,
                    digest: plan.digest(),
                    input: Zeroizing::new(input.to_vec()),
                    reply,
                })
                .map_err(|_| Error::Context)?;
            receive.recv().map_err(|_| Error::Context)?
        } else {
            let peer = self
                .recv
                .as_ref()
                .unwrap()
                .recv()
                .map_err(|_| Error::Context)?;
            if peer.digest != plan.digest() || !same_context(peer.context, context) {
                return Err(Error::Context);
            }
            let [a, b] = direct_pair(plan, context, input, &peer.input)?;
            peer.reply.send(Ok(b)).map_err(|_| Error::Context)?;
            Ok(a)
        }
    }
    fn cancel(&mut self) {}
    fn observe(&mut self, selection: usize, node: usize, _shape: &[usize], shares: &[u64]) {
        self.trace
            .push((selection, node, Zeroizing::new(shares.to_vec())));
    }
}
fn direct(program: &Program) -> [Execution; 2] {
    let (send, recv) = mpsc::sync_channel(0);
    let context = context(program);
    let [input_a, input_b] = inputs(program);
    thread::scope(|scope| {
        let a = scope.spawn(move || {
            let mut driver = Direct {
                role: 0,
                send: None,
                recv: Some(recv),
                trace: Vec::new(),
            };
            let result = Worker::new(program, 0, context, input_a)
                .unwrap()
                .execute(&mut driver);
            (result, driver.trace)
        });
        let b = scope.spawn(move || {
            let mut driver = Direct {
                role: 1,
                send: Some(send),
                recv: None,
                trace: Vec::new(),
            };
            let result = Worker::new(program, 1, context, input_b)
                .unwrap()
                .execute(&mut driver);
            (result, driver.trace)
        });
        [a.join().unwrap(), b.join().unwrap()]
    })
}
fn open(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| a.wrapping_add(b) & MASK)
        .collect()
}

#[test]
fn serialized_three_actors_match_direct_native_full_tokens_and_observations() {
    let program = fixture(1, 1, 2);
    let tokens = serialized_matches_direct(&program);
    assert_eq!(tokens, [0, 0]);
}

#[test]
fn complete_larger_request_keeps_global_replay_ownership_and_exact_observations() {
    let program = fixture(2, 2, 2);
    let admission = admit(&program, context(&program)).unwrap();
    assert!(admission.primitive_requests_per_party() > 4096);
    assert_eq!(
        admission.guard_capacity(),
        admission.primitive_requests_per_party()
    );
    assert!(admission.combined_reservation_bytes() <= MAX_ALLOCATION_BYTES);
    assert_eq!(serialized_matches_direct(&program).len(), 2);

    // Replaying material after crossing the former ledger limit must still burn
    // the request and wake both other actors. No per-kernel ledger replacement.
    let (_, left, right, inspection) = serialized(&program, Fault::ReplayMaterialAfter4096);
    assert!(left.0.is_err());
    assert!(right.0.is_err());
    assert!(inspection.counts.delivered.load(Ordering::SeqCst) > 8192);
    assert_eq!(inspection.counts.worker[1].load(Ordering::SeqCst), 1);
    inspection.drained();
}

pub(in crate::fss_decoder) fn serialized_matches_direct(program: &Program) -> Vec<u64> {
    let (dealer, wire_a, wire_b, inspection) = serialized(program, Fault::None);
    dealer.unwrap();
    let a = wire_a.0.unwrap();
    let b = wire_b.0.unwrap();
    let tokens = open(a.values(), b.values());
    inspection.drained();
    assert_eq!(inspection.counts.dealer.load(Ordering::SeqCst), 0);
    assert_eq!(inspection.counts.worker[0].load(Ordering::SeqCst), 0);
    assert_eq!(inspection.counts.worker[1].load(Ordering::SeqCst), 0);
    let [control_a, control_b] = direct(program);
    let ca = control_a.0.unwrap();
    let cb = control_b.0.unwrap();
    assert_eq!(open(a.values(), b.values()), open(ca.values(), cb.values()));
    assert!(!wire_a.1.is_empty());
    assert_eq!(wire_a.1.len(), wire_b.1.len());
    assert_eq!(wire_a.1.len(), control_a.1.len());
    assert_eq!(wire_a.1.len(), control_b.1.len());
    for i in 0..wire_a.1.len() {
        let wa = &wire_a.1[i];
        let wb = &wire_b.1[i];
        let ca = &control_a.1[i];
        let cb = &control_b.1[i];
        assert_eq!((wa.0, wa.1), (ca.0, ca.1));
        assert_eq!((wb.0, wb.1), (cb.0, cb.1));
        assert_eq!(
            open(&wa.2, &wb.2),
            open(&ca.2, &cb.2),
            "selection={},node={}",
            wa.0,
            wa.1
        );
    }
    tokens
}
