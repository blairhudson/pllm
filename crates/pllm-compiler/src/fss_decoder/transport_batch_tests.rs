use super::*;
use pllm_garble::fusefss_backend::composition::kernels;

#[derive(Default)]
struct TapeDealer {
    records: [VecDeque<Material>; 2],
    calls: Vec<(bool, u8, usize)>,
    cross_plan: bool,
    failure: Option<(usize, usize)>,
    cancel_after_left: bool,
    cancelled: usize,
    finished: bool,
}
impl DealerTransport for TapeDealer {
    fn deliver(&mut self, request: MaterialRequest, packet: &[u8]) -> Result<(), Error> {
        self.calls.push((false, request.role, 1));
        self.records[request.role as usize].push_back(Material {
            meta: request,
            bytes: Zeroizing::new(packet.to_vec()),
        });
        Ok(())
    }
    fn deliver_batch(&mut self, role: u8, records: &[MaterialRecord<'_>]) -> Result<(), Error> {
        assert!(!records.is_empty() && records.len() <= 64);
        assert!(
            1024 + records
                .iter()
                .map(|r| 8 + 1024 + r.packet().len())
                .sum::<usize>()
                <= 1 << 20
        );
        let call = self.calls.len();
        self.calls.push((true, role, records.len()));
        self.cross_plan |= records.first().unwrap().request().kernel.ordinal
            != records.last().unwrap().request().kernel.ordinal;
        for (index, record) in records.iter().enumerate() {
            if self.failure == Some((call, index)) {
                return Err(Error::Context); // Prefix accepted, no batch ACK.
            }
            assert_eq!(record.request().role, role);
            self.records[role as usize].push_back(Material {
                meta: record.request(),
                bytes: Zeroizing::new(record.packet().to_vec()),
            });
        }
        if self.failure == Some((call, usize::MAX)) {
            return Err(Error::Context); // All appended, ACK lost.
        }
        Ok(())
    }
    fn check_active(&mut self) -> Result<(), Error> {
        if self.cancel_after_left && !self.calls.is_empty() {
            Err(Error::Consumed)
        } else {
            Ok(())
        }
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.finished = true;
        Ok(())
    }
    fn cancel(&mut self) {
        self.cancelled += 1;
        for records in &mut self.records {
            records.clear();
        }
    }
}

fn requests(ordinal: usize, bytes: usize) -> [MaterialRequest; 2] {
    [0, 1].map(|role| MaterialRequest {
        kernel: Kernel {
            context: Context {
                plan: [3; 32],
                session: [4; 32],
                operation: [5; 32],
                first_tensor_index: 0,
            },
            plan_digest: [6; 32],
            ordinal,
        },
        role,
        pc: ordinal,
        packet_bytes_bound: bytes,
    })
}

#[test]
fn cross_plan_copies_outlive_original_budgets_and_preserve_exact_envelopes() {
    let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
    let mut queues = batch::Queues::new(&budget).unwrap();
    let mut expected: [Vec<Material>; 2] = [Vec::new(), Vec::new()];
    for ordinal in 0..2 {
        // Each complete issuing Budget dies before the next Plan is constructed.
        let original_budget = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let plan = kernels::floor(
            kernels::Binding {
                source_digest: [7; 32],
                contract_digest: [8; 32],
            },
            kernels::InputRange {
                min: -4096,
                max: 4096,
            },
            12,
            &original_budget,
        )
        .unwrap()
        .0;
        let mut context = requests(ordinal, 1)[0].kernel.context;
        context.operation = [20 + ordinal as u8; 32];
        for pc in plan.material_steps() {
            let [a, b] = plan.issue_step(pc, context).unwrap();
            let [a, b] = [a.export_packet().unwrap(), b.export_packet().unwrap()];
            let mut req = requests(
                ordinal,
                plan.step_transport_resources(pc)
                    .unwrap()
                    .packet_bytes_bound,
            );
            for role in 0..2 {
                req[role].pc = pc;
                req[role].kernel.context = context;
                req[role].kernel.plan_digest = plan.digest();
            }
            for (role, packet) in [&a, &b].into_iter().enumerate() {
                expected[role].push(Material {
                    meta: req[role],
                    bytes: Zeroizing::new(packet.as_bytes().to_vec()),
                });
            }
            queues.push_pair(req, [a.as_bytes(), b.as_bytes()]).unwrap();
        }
        drop(plan);
        assert_eq!(original_budget.used_bytes(), 0);
    }
    assert!(budget.used_bytes() > 0);
    let mut callback = TapeDealer::default();
    queues.flush(&mut callback).unwrap();
    assert!(callback.cross_plan);
    assert_eq!(budget.used_bytes(), 0);
    for role in 0..2 {
        assert_eq!(callback.records[role].len(), expected[role].len());
        for (actual, original) in callback.records[role].iter().zip(&expected[role]) {
            assert!(same_kernel(actual.meta.kernel, original.meta.kernel));
            assert_eq!(
                (
                    actual.meta.role,
                    actual.meta.pc,
                    actual.meta.packet_bytes_bound
                ),
                (
                    original.meta.role,
                    original.meta.pc,
                    original.meta.packet_bytes_bound
                )
            );
            assert_eq!(*actual.bytes, *original.bytes);
        }
    }
}

#[test]
fn count_byte_and_final_tails_are_bounded_independently() {
    for (packet_len, total, expected) in [
        (ENVELOPE_BYTES, 65, vec![64, 1]),
        (((1 << 20) - 1024) / 2 - 1032, 3, vec![2, 1]),
    ] {
        let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
        let mut queues = batch::Queues::new(&budget).unwrap();
        let mut callback = TapeDealer::default();
        let packet = Zeroizing::new(vec![91; packet_len]);
        for ordinal in 0..total {
            if queues.needs_flush(packet_len).unwrap() {
                queues.flush(&mut callback).unwrap();
            }
            queues
                .push_pair(requests(ordinal, packet_len), [&packet, &packet])
                .unwrap();
            if queues.full() {
                queues.flush(&mut callback).unwrap();
            }
        }
        queues.flush(&mut callback).unwrap();
        assert!(queues.complete(total));
        assert_eq!(budget.used_bytes(), 0);
        for role in 0..2 {
            let sizes: Vec<_> = callback
                .calls
                .iter()
                .filter(|c| c.1 == role)
                .map(|c| c.2)
                .collect();
            assert_eq!(sizes, expected);
        }
    }
}

#[test]
fn second_copy_budget_failure_never_enqueues_half_a_pair() {
    let packet = [13; ENVELOPE_BYTES];
    for prior in [false, true] {
        let limit = if prior { 4 } else { 2 } * packet.len() - 1;
        let budget = Budget::new(limit).unwrap();
        let mut queues = batch::Queues::new(&budget).unwrap();
        if prior {
            queues
                .push_pair(requests(0, packet.len()), [&packet, &packet])
                .unwrap();
        }
        let before = budget.used_bytes();
        assert_eq!(
            queues.push_pair(requests(1, packet.len()), [&packet, &packet]),
            Err(Error::Resource)
        );
        assert_eq!(budget.used_bytes(), before);
        queues.clear();
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn oversized_record_uses_original_singleton_after_both_batch_acks() {
    let program = fixture(1, 1, 1);
    let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
    let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
    let mut callback = TapeDealer::default();
    let large = (1 << 20) - 1024 - 1032 + 1;
    let mut admission =
        admit_with_delivery(&program, context(&program), MaterialDelivery::BatchV1).unwrap();
    // Isolate carrier routing with bounded opaque packets, not fake execution.
    admission.packet_bytes = large;
    let mut dealer = Dealer {
        transport: &mut callback,
        batches: Some(batch::Queues::new(&budget).unwrap()),
        admission,
        kernels: 0,
        requests: 0,
        hash: hash_start(&program),
        cancelled: false,
        complete: false,
    };
    for (ordinal, bytes) in [ENVELOPE_BYTES, large, ENVELOPE_BYTES]
        .into_iter()
        .enumerate()
    {
        let singleton = dealer.before_pair(bytes).unwrap();
        assert_eq!(singleton, ordinal == 1);
        if singleton {
            assert_eq!(budget.used_bytes(), 0);
        }
        let packet = Zeroizing::new(vec![17; bytes]);
        dealer
            .deliver_pair(
                requests(ordinal, bytes),
                [
                    MaterialPacket::from_bytes(&packet, &original).unwrap(),
                    MaterialPacket::from_bytes(&packet, &original).unwrap(),
                ],
                singleton,
            )
            .unwrap();
    }
    dealer
        .batches
        .as_mut()
        .unwrap()
        .flush(dealer.transport)
        .unwrap();
    assert!(dealer.batches.as_ref().unwrap().complete(3));
    assert_eq!(budget.used_bytes(), 0);
    dealer.complete = true;
    drop(dealer);
    assert_eq!(
        callback.calls,
        [
            (true, 0, 1),
            (true, 1, 1),
            (false, 0, 1),
            (false, 1, 1),
            (true, 0, 1),
            (true, 1, 1)
        ]
    );
    assert_eq!(original.used_bytes(), 0);
}

#[test]
fn batch_admission_prices_bounded_overlap_and_preserves_single_schedule() {
    let program = fixture(1, 1, 1);
    let single = admit(&program, context(&program)).unwrap();
    let explicit =
        admit_with_delivery(&program, context(&program), MaterialDelivery::Single).unwrap();
    let batched =
        admit_with_delivery(&program, context(&program), MaterialDelivery::BatchV1).unwrap();
    assert_eq!(
        single.combined_reservation_bytes(),
        explicit.combined_reservation_bytes()
    );
    assert!(single.delivery_batch().is_none());
    let b = batched.delivery_batch().unwrap();
    assert_eq!(
        batched.combined_reservation_bytes(),
        single.combined_reservation_bytes()
            + b.native_reservation_bytes()
            + b.python_overlap_bytes()
    );
    assert!(batched.combined_reservation_bytes() <= MAX_ALLOCATION_BYTES);
    assert_eq!(b.records_bound(), 64);
    assert_eq!(b.body_bytes_bound(), 1 << 20);
    assert_eq!(single.transcript_digest(), batched.transcript_digest());
    assert_eq!(
        single.primitive_requests_per_party(),
        batched.primitive_requests_per_party()
    );
    assert_eq!(
        single.cumulative_packet_bytes_per_party_bound(),
        batched.cumulative_packet_bytes_per_party_bound()
    );
    let mut callback = TapeDealer::default();
    prepare(&program, context(&program), &mut callback).unwrap();
    assert!(callback.finished);
    for pair in callback.calls.chunks_exact(2) {
        assert_eq!(pair, [(false, 0, 1), (false, 1, 1)]);
    }
}

#[test]
fn partial_lost_ack_and_cancel_burn_both_queues_without_finish() {
    let program = fixture(1, 1, 1);
    for (failure, cancel_after_left) in [
        (Some((1, 1)), false),
        (Some((1, usize::MAX)), false),
        (None, true),
    ] {
        let mut callback = TapeDealer {
            failure,
            cancel_after_left,
            ..Default::default()
        };
        assert!(prepare_with_delivery(
            &program,
            context(&program),
            MaterialDelivery::BatchV1,
            &mut callback
        )
        .is_err());
        assert_eq!(callback.cancelled, 1);
        assert!(!callback.finished);
        assert!(callback.records.iter().all(VecDeque::is_empty));
        assert!(callback.calls.len() <= 2);
    }
}

struct OfflineRole {
    endpoint: RoleEndpoint,
    material: VecDeque<Material>,
}
impl WorkerTransport for OfflineRole {
    fn material(&mut self, request: MaterialRequest, reply: &mut Reply) -> Result<(), Error> {
        let record = self.material.pop_front().ok_or(Error::Context)?;
        if !same_kernel(record.meta.kernel, request.kernel)
            || (record.meta.role, record.meta.pc) != (request.role, request.pc)
        {
            return Err(Error::Context);
        }
        reply.write(&record.bytes)
    }
    fn exchange(
        &mut self,
        request: PeerRequest,
        frame: &[u8],
        reply: &mut Reply,
    ) -> Result<(), Error> {
        self.endpoint.exchange(request, frame, reply)
    }
    fn finish(&mut self) -> Result<(), Error> {
        if !self.material.is_empty() {
            return Err(Error::Context);
        }
        self.endpoint.finish()
    }
    fn cancel(&mut self) {
        self.material.clear();
        self.endpoint.cancel();
    }
    fn observe(&mut self, selection: usize, node: usize, shape: &[usize], shares: &[u64]) {
        self.endpoint.observe(selection, node, shape, shares)
    }
}

#[test]
fn sealed_cross_plan_batches_match_all_direct_nodes_and_private_feedback() {
    let program = fixture(1, 1, 2);
    let context = context(&program);
    let mut dealer = TapeDealer::default();
    prepare_with_delivery(&program, context, MaterialDelivery::BatchV1, &mut dealer).unwrap();
    assert!(dealer.finished && dealer.cross_plan);
    for records in &dealer.records {
        assert_eq!(
            records.len(),
            program.resources().primitive_requests_per_party
        );
    }
    let (mut unused, [left, right], inspection) = endpoints(Fault::None);
    unused.settled = true;
    drop(unused);
    let mut left = OfflineRole {
        endpoint: left,
        material: std::mem::take(&mut dealer.records[0]),
    };
    let mut right = OfflineRole {
        endpoint: right,
        material: std::mem::take(&mut dealer.records[1]),
    };
    let [a, b] = inputs(&program);
    let (a, b) = thread::scope(|s| {
        let a = s.spawn(|| {
            execute_with_delivery(
                &program,
                0,
                context,
                a,
                MaterialDelivery::BatchV1,
                &mut left,
            )
        });
        let b = s.spawn(|| {
            execute_with_delivery(
                &program,
                1,
                context,
                b,
                MaterialDelivery::BatchV1,
                &mut right,
            )
        });
        (a.join().unwrap().unwrap(), b.join().unwrap().unwrap())
    });
    let [(direct_a, trace_a), (direct_b, trace_b)] = direct(&program);
    assert_eq!(
        open(a.values(), b.values()),
        open(direct_a.unwrap().values(), direct_b.unwrap().values())
    );
    assert_eq!(left.endpoint.trace.len(), trace_a.len());
    assert_eq!(right.endpoint.trace.len(), trace_b.len());
    for (((a, b), c), d) in left
        .endpoint
        .trace
        .iter()
        .zip(&right.endpoint.trace)
        .zip(&trace_a)
        .zip(&trace_b)
    {
        assert_eq!((a.0, a.1), (c.0, c.1));
        assert_eq!(open(&a.2, &b.2), open(&c.2, &d.2));
    }
    inspection.drained();
}
