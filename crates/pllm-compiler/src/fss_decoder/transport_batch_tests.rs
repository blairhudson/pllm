use super::*;
use pllm_garble::fusefss_backend::composition::kernels;

#[path = "transport_receive_tests.rs"]
mod receiving;

#[derive(Default)]
struct TapeDealer {
    records: [VecDeque<Material>; 2],
    packet_pointers: [Vec<usize>; 2],
    calls: Vec<(bool, u8, usize)>,
    cross_plan: bool,
    failure: Option<(usize, usize)>,
    panic: Option<(u8, Box<dyn std::any::Any + Send>)>,
    cancel_after_left: bool,
    cancelled: usize,
    finished: bool,
}
impl DealerTransport for TapeDealer {
    fn deliver(&mut self, request: MaterialRequest, packet: &[u8]) -> Result<(), Error> {
        self.calls.push((false, request.role, 1));
        self.packet_pointers[request.role as usize].push(packet.as_ptr() as usize);
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
            self.packet_pointers[role as usize].push(record.packet().as_ptr() as usize);
            self.records[role as usize].push_back(Material {
                meta: record.request(),
                bytes: Zeroizing::new(record.packet().to_vec()),
            });
            if self
                .panic
                .as_ref()
                .is_some_and(|(recipient, _)| *recipient == role)
            {
                // Panic inside delivery, after a retained prefix but before ACK.
                let (_, payload) = self.panic.take().unwrap();
                std::panic::resume_unwind(payload);
            }
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
        for pointers in &mut self.packet_pointers {
            pointers.clear();
        }
    }
}

fn opaque_pair<'a>(bytes: &[u8], budget: &'a Budget) -> [MaterialPacket<'a>; 2] {
    [
        MaterialPacket::from_bytes(bytes, budget).unwrap(),
        MaterialPacket::from_bytes(bytes, budget).unwrap(),
    ]
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
fn cross_plan_handoffs_outlive_original_budgets_and_preserve_pointers_and_envelopes() {
    let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
    let mut queues = batch::Queues::new(&budget).unwrap();
    let mut expected: [Vec<Material>; 2] = [Vec::new(), Vec::new()];
    let mut pointers: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
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
            for request in &mut req {
                request.pc = pc;
                request.kernel.context = context;
                request.kernel.plan_digest = plan.digest();
            }
            for (role, packet) in [&a, &b].into_iter().enumerate() {
                pointers[role].push(packet.as_bytes().as_ptr() as usize);
                expected[role].push(Material {
                    meta: req[role],
                    bytes: Zeroizing::new(packet.as_bytes().to_vec()),
                });
            }
            queues.push_pair(req, [a, b]).unwrap();
        }
        drop(plan);
        assert_eq!(original_budget.used_bytes(), 0);
    }
    assert!(budget.used_bytes() > 0);
    let mut callback = TapeDealer::default();
    queues.flush(&mut callback).unwrap();
    assert!(callback.cross_plan);
    assert_eq!(callback.packet_pointers, pointers);
    assert_eq!(budget.used_bytes(), 0);
    for (role, expected) in expected.iter().enumerate() {
        assert_eq!(callback.records[role].len(), expected.len());
        for (actual, original) in callback.records[role].iter().zip(expected) {
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
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut queues = batch::Queues::new(&budget).unwrap();
        let mut callback = TapeDealer::default();
        let packet = Zeroizing::new(vec![91; packet_len]);
        for ordinal in 0..total {
            if queues.needs_flush(packet_len).unwrap() {
                queues.flush(&mut callback).unwrap();
            }
            queues
                .push_pair(
                    requests(ordinal, packet_len),
                    opaque_pair(&packet, &original),
                )
                .unwrap();
            assert_eq!(original.used_bytes(), 0);
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
fn second_transfer_budget_failure_never_enqueues_half_a_pair() {
    let packet = [13; ENVELOPE_BYTES];
    for prior in [false, true] {
        let limit = if prior { 4 } else { 2 } * packet.len() - 1;
        let budget = Budget::new(limit).unwrap();
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut queues = batch::Queues::new(&budget).unwrap();
        if prior {
            queues
                .push_pair(requests(0, packet.len()), opaque_pair(&packet, &original))
                .unwrap();
        }
        let before = budget.used_bytes();
        assert_eq!(
            queues.push_pair(requests(1, packet.len()), opaque_pair(&packet, &original)),
            Err(Error::Resource)
        );
        assert_eq!(budget.used_bytes(), before);
        assert_eq!(original.used_bytes(), 0);
        let mut callback = TapeDealer::default();
        queues.flush(&mut callback).unwrap();
        assert!(queues.complete(usize::from(prior)));
        for records in &callback.records {
            assert_eq!(records.len(), usize::from(prior));
            assert!(records.iter().all(|record| record.meta.kernel.ordinal == 0));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn pair_handoff_destination_exact_fit_and_same_budget_overlap() {
    let bytes = [19; ENVELOPE_BYTES];
    for same_budget in [false, true] {
        for short in [false, true] {
            // With the same Budget, both originals plus the current incoming
            // full backing charge must fit. Each source refunds only afterward.
            let limit = (if same_budget { 3 } else { 2 }) * bytes.len() - usize::from(short);
            let budget = Budget::new(limit).unwrap();
            let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let source = if same_budget { &budget } else { &original };
            let mut queues = batch::Queues::new(&budget).unwrap();
            let packets = opaque_pair(&bytes, source);
            let pointers = packets
                .each_ref()
                .map(|packet| packet.as_bytes().as_ptr() as usize);
            let result = queues.push_pair(requests(0, bytes.len()), packets);
            if short {
                assert_eq!(result, Err(Error::Resource));
                assert_eq!(budget.used_bytes(), 0);
            } else {
                result.unwrap();
                assert_eq!(budget.used_bytes(), 2 * bytes.len());
            }
            assert_eq!(original.used_bytes(), 0);
            let mut callback = TapeDealer::default();
            queues.flush(&mut callback).unwrap();
            assert!(queues.complete(usize::from(!short)));
            for (role, pointer) in pointers.iter().enumerate() {
                assert_eq!(callback.records[role].len(), usize::from(!short));
                if !short {
                    assert_eq!(callback.packet_pointers[role], [*pointer]);
                }
            }
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn rejected_pair_preserves_existing_queues_and_consumes_both_originals() {
    let bytes = [23; ENVELOPE_BYTES];
    for role in 0..2 {
        for bad_recipient in [false, true] {
            let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
            let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
            let mut queues = batch::Queues::new(&budget).unwrap();
            queues
                .push_pair(requests(0, bytes.len()), opaque_pair(&bytes, &original))
                .unwrap();
            let before = budget.used_bytes();
            let mut invalid = requests(1, bytes.len());
            if bad_recipient {
                invalid[role].role = (1 - role) as u8;
            } else {
                invalid[role].packet_bytes_bound -= 1;
            }
            assert_eq!(
                queues.push_pair(invalid, opaque_pair(&bytes, &original)),
                Err(Error::Context)
            );
            assert_eq!(original.used_bytes(), 0);
            assert_eq!(budget.used_bytes(), before);
            let mut callback = TapeDealer::default();
            queues.flush(&mut callback).unwrap();
            assert!(queues.complete(1));
            assert_eq!(callback.calls, [(true, 0, 1), (true, 1, 1)]);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn handoff_keeps_record_and_body_caps_before_queue_publication() {
    for count_limit in [false, true] {
        let bytes = Zeroizing::new(vec![
            29;
            if count_limit {
                ENVELOPE_BYTES
            } else {
                (1 << 20) - 1024 - 1032
            }
        ]);
        let count = if count_limit { 64 } else { 1 };
        let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut queues = batch::Queues::new(&budget).unwrap();
        for ordinal in 0..count {
            queues
                .push_pair(
                    requests(ordinal, bytes.len()),
                    opaque_pair(&bytes, &original),
                )
                .unwrap();
        }
        assert!(queues.full());
        let before = budget.used_bytes();
        let small = [31; ENVELOPE_BYTES];
        assert_eq!(
            queues.push_pair(requests(count, small.len()), opaque_pair(&small, &original)),
            Err(if count_limit {
                Error::Context
            } else {
                Error::Resource
            })
        );
        assert_eq!(original.used_bytes(), 0);
        assert_eq!(budget.used_bytes(), before);
        let mut callback = TapeDealer::default();
        queues.flush(&mut callback).unwrap();
        assert!(queues.complete(count));
        assert_eq!(callback.calls, [(true, 0, count), (true, 1, count)]);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn transferred_queues_refund_on_clear_drop_and_unwind() {
    for mode in 0..3 {
        let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut queues = batch::Queues::new(&budget).unwrap();
            let bytes = [37; ENVELOPE_BYTES];
            queues
                .push_pair(requests(0, bytes.len()), opaque_pair(&bytes, &original))
                .unwrap();
            assert_eq!(original.used_bytes(), 0);
            assert_eq!(budget.used_bytes(), 2 * bytes.len());
            match mode {
                0 => queues.clear(),
                1 => drop(queues),
                _ => panic!("unwind transferred queue"),
            }
        }));
        assert_eq!(outcome.is_err(), mode == 2);
        assert_eq!(original.used_bytes(), 0);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn transferred_backings_stay_charged_until_both_batch_acks() {
    struct Ack<'a> {
        budget: &'a Budget,
        bytes: usize,
        fail: Option<u8>,
        roles: Vec<u8>,
    }
    impl DealerTransport for Ack<'_> {
        fn deliver(&mut self, _: MaterialRequest, _: &[u8]) -> Result<(), Error> {
            Err(Error::Unsupported)
        }
        fn deliver_batch(&mut self, role: u8, records: &[MaterialRecord<'_>]) -> Result<(), Error> {
            assert_eq!(self.budget.used_bytes(), self.bytes);
            assert_eq!(records.len(), 1);
            self.roles.push(role);
            if self.fail == Some(role) {
                Err(Error::Context)
            } else {
                Ok(())
            }
        }
        fn finish(&mut self) -> Result<(), Error> {
            Ok(())
        }
        fn cancel(&mut self) {}
    }
    for fail in [None, Some(0), Some(1)] {
        let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let bytes = [43; ENVELOPE_BYTES];
        let mut queues = batch::Queues::new(&budget).unwrap();
        queues
            .push_pair(requests(0, bytes.len()), opaque_pair(&bytes, &original))
            .unwrap();
        assert_eq!(original.used_bytes(), 0);
        let mut callback = Ack {
            budget: &budget,
            bytes: 2 * bytes.len(),
            fail,
            roles: Vec::new(),
        };
        let result = queues.flush(&mut callback);
        assert_eq!(
            callback.roles,
            if fail == Some(0) { vec![0] } else { vec![0, 1] }
        );
        if fail.is_some() {
            assert_eq!(result, Err(Error::Context));
            assert!(!queues.complete(1));
            assert_eq!(budget.used_bytes(), 2 * bytes.len());
            queues.clear(); // Dealer::abort uses this same burn path.
        } else {
            result.unwrap();
            assert!(queues.complete(1));
        }
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn transferred_dealer_queues_burn_on_abort_drop_and_unwind() {
    let program = fixture(1, 1, 1);
    for mode in 0..3 {
        let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let mut callback = TapeDealer::default();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let admission =
                admit_with_delivery(&program, context(&program), MaterialDelivery::BatchV1)
                    .unwrap();
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
            let bytes = [41; ENVELOPE_BYTES];
            let singleton = dealer.before_pair(bytes.len()).unwrap();
            assert!(!singleton);
            dealer
                .deliver_pair(
                    requests(0, bytes.len()),
                    opaque_pair(&bytes, &original),
                    singleton,
                )
                .unwrap();
            assert_eq!(original.used_bytes(), 0);
            assert_eq!(budget.used_bytes(), 2 * bytes.len());
            match mode {
                0 => {
                    dealer.abort();
                    dealer.abort(); // Cancellation remains idempotent.
                    assert_eq!(budget.used_bytes(), 0);
                }
                1 => drop(dealer),
                _ => panic!("unwind transferred dealer queue"),
            }
        }));
        assert_eq!(outcome.is_err(), mode == 2);
        assert_eq!(callback.cancelled, 1);
        assert!(!callback.finished && callback.calls.is_empty());
        assert_eq!(original.used_bytes(), 0);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn delivery_callback_panic_burns_both_queues_and_preserves_original_payload() {
    struct PanicMarker(u8);
    let program = fixture(1, 1, 1);
    for role in 0..2 {
        let budget = Budget::new(batch::BatchResources::packet_budget_bytes()).unwrap();
        let original = Budget::new(MAX_ALLOCATION_BYTES).unwrap();
        let payload = Box::new(PanicMarker(role));
        let identity = (&*payload) as *const PanicMarker;
        let mut callback = TapeDealer {
            panic: Some((role, payload)),
            ..Default::default()
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let admission =
                admit_with_delivery(&program, context(&program), MaterialDelivery::BatchV1)
                    .unwrap();
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
            let bytes = [47; ENVELOPE_BYTES];
            let singleton = dealer.before_pair(bytes.len()).unwrap();
            assert!(!singleton);
            dealer
                .deliver_pair(
                    requests(0, bytes.len()),
                    opaque_pair(&bytes, &original),
                    singleton,
                )
                .unwrap();
            assert_eq!(original.used_bytes(), 0);
            assert_eq!(budget.used_bytes(), 2 * bytes.len());
            dealer
                .batches
                .as_mut()
                .unwrap()
                .flush(dealer.transport)
                .unwrap();
        }));
        let caught = outcome
            .expect_err("delivery callback did not panic")
            .downcast::<PanicMarker>()
            .unwrap_or_else(|_| panic!("panic payload replaced"));
        assert_eq!(caught.0, role);
        assert_eq!((&*caught) as *const PanicMarker, identity);
        assert_eq!(
            callback.calls,
            if role == 0 {
                vec![(true, 0, 1)]
            } else {
                vec![(true, 0, 1), (true, 1, 1)]
            }
        );
        assert!(callback.panic.is_none());
        assert_eq!(callback.cancelled, 1);
        assert!(!callback.finished);
        assert!(callback.records.iter().all(VecDeque::is_empty));
        assert!(callback.packet_pointers.iter().all(Vec::is_empty));
        assert_eq!(original.used_bytes(), 0);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn prepare_delivery_callback_panic_preserves_payload_and_cancels_once() {
    struct PanicMarker(u8);
    let program = fixture(1, 1, 1);
    for role in 0..2 {
        let payload = Box::new(PanicMarker(role));
        let identity = (&*payload) as *const PanicMarker;
        let mut callback = TapeDealer {
            panic: Some((role, payload)),
            ..Default::default()
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prepare_with_delivery(
                &program,
                context(&program),
                MaterialDelivery::BatchV1,
                &mut callback,
            )
        }));
        let caught = match outcome {
            Err(payload) => payload
                .downcast::<PanicMarker>()
                .unwrap_or_else(|_| panic!("panic payload replaced")),
            Ok(_) => panic!("prepare delivery callback did not panic"),
        };
        assert_eq!(caught.0, role);
        assert_eq!((&*caught) as *const PanicMarker, identity);
        assert_eq!(
            callback.calls.iter().map(|call| call.1).collect::<Vec<_>>(),
            if role == 0 { vec![0] } else { vec![0, 1] }
        );
        assert!(callback.panic.is_none());
        assert_eq!(callback.cancelled, 1);
        assert!(!callback.finished);
        assert!(callback.records.iter().all(VecDeque::is_empty));
        assert!(callback.packet_pointers.iter().all(Vec::is_empty));
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
