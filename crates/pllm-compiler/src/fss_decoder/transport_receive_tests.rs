//! Fixed sealed fixture is replayed ONLY inside this equivalence oracle, using
//! isolated test guards. Not a supported serving lifecycle or material reuse API.
use super::*;

struct WindowRole {
    inner: OfflineRole,
    enabled: bool,
    scalar_calls: usize,
    windows: Vec<usize>,
    pcs: Vec<(usize, usize)>,
    wire: Vec<Vec<u8>>,
    fault: Option<&'static str>,
    cancellations: usize,
    finishes: usize,
}
impl WindowRole {
    fn new(endpoint: RoleEndpoint, material: VecDeque<Material>, enabled: bool) -> Self {
        Self {
            inner: OfflineRole { endpoint, material },
            enabled,
            scalar_calls: 0,
            windows: Vec::new(),
            pcs: Vec::new(),
            wire: Vec::new(),
            fault: None,
            cancellations: 0,
            finishes: 0,
        }
    }
}
impl WorkerTransport for WindowRole {
    fn material(&mut self, request: MaterialRequest, reply: &mut Reply) -> Result<(), Error> {
        self.scalar_calls += 1;
        self.pcs.push((request.kernel.ordinal, request.pc));
        self.inner.material(request, reply)
    }
    fn material_window(
        &mut self,
        requests: &[MaterialRequest],
        replies: &mut [ReplyView<'_>],
    ) -> Result<(), Error> {
        assert!(
            self.enabled,
            "scalar oracle must not call optional window provider"
        );
        self.windows.push(requests.len());
        if self.fault == Some("cancel") {
            return Err(Error::Consumed);
        }
        for (index, (request, reply)) in requests.iter().zip(replies).enumerate() {
            self.pcs.push((request.kernel.ordinal, request.pc));
            let record = self.inner.material.pop_front().ok_or(Error::Context)?;
            if !same_kernel(record.meta.kernel, request.kernel)
                || (record.meta.role, record.meta.pc) != (request.role, request.pc)
            {
                return Err(Error::Context);
            }
            if self.fault == Some("panic") {
                std::panic::panic_any(7721u64);
            }
            if self.fault == Some("partial") && index == 1 {
                return Err(Error::Context);
            }
            if self.fault == Some("capacity") && index == 1 {
                return Err(Error::Resource);
            }
            if self.fault == Some("missing") {
                return Ok(());
            }
            if self.fault == Some("duplicate") && index == 1 {
                reply.write(self.inner.endpoint.saved.as_ref().unwrap())?;
            } else {
                if index == 0 {
                    self.inner.endpoint.saved = Some(Zeroizing::new(record.bytes.to_vec()));
                }
                reply.write(&record.bytes)?;
            }
        }
        Ok(())
    }
    fn exchange(
        &mut self,
        request: PeerRequest,
        frame: &[u8],
        reply: &mut Reply,
    ) -> Result<(), Error> {
        self.wire.push(frame.to_vec());
        self.inner.exchange(request, frame, reply)
    }
    fn finish(&mut self) -> Result<(), Error> {
        self.finishes += 1;
        self.inner.finish()
    }
    fn cancel(&mut self) {
        self.cancellations += 1;
        self.inner.cancel();
    }
    fn observe(&mut self, selection: usize, node: usize, shape: &[usize], shares: &[u64]) {
        self.inner.observe(selection, node, shape, shares);
    }
}

fn clone_records(records: &VecDeque<Material>) -> VecDeque<Material> {
    records
        .iter()
        .map(|record| Material {
            meta: record.meta,
            bytes: Zeroizing::new(record.bytes.to_vec()),
        })
        .collect()
}
fn run_sealed(
    program: &Program,
    records: [VecDeque<Material>; 2],
    enabled: bool,
    fault: Option<&'static str>,
) -> ([Result<TokenShares, Error>; 2], [WindowRole; 2], Inspection) {
    let (mut unused, [left, right], inspection) = endpoints(Fault::None);
    unused.settled = true;
    drop(unused);
    let [ra, rb] = records;
    let mut left = WindowRole::new(left, ra, enabled);
    let mut right = WindowRole::new(right, rb, enabled);
    right.fault = fault;
    let [a, b] = inputs(program);
    let (a, b) = thread::scope(|scope| {
        let a = scope.spawn(|| {
            execute_with_receive(
                program,
                0,
                context(program),
                a,
                MaterialDelivery::BatchV1,
                if enabled {
                    MaterialReceive::FrozenWindow
                } else {
                    MaterialReceive::Scalar
                },
                &mut left,
            )
        });
        let b = scope.spawn(|| {
            execute_with_receive(
                program,
                1,
                context(program),
                b,
                MaterialDelivery::BatchV1,
                if enabled {
                    MaterialReceive::FrozenWindow
                } else {
                    MaterialReceive::Scalar
                },
                &mut right,
            )
        });
        (a.join(), b.join())
    });
    let a = a.expect("non-panicking peer must retire without losing its original result");
    let b = match b {
        Ok(result) => result,
        Err(payload) => {
            assert_eq!(*payload.downcast::<u64>().unwrap(), 7721);
            Err(Error::Consumed)
        }
    };
    ([a, b], [left, right], inspection)
}

#[test]
fn sealed_frozen_group_reduces_callbacks_preserves_each_party_wire_cursors_and_feedback() {
    let program = fixture_schedule(1, 1, 2, crate::fss_decoder::OnlineSchedule::TensorV1);
    let scalar =
        admit_with_delivery(&program, context(&program), MaterialDelivery::BatchV1).unwrap();
    let window = admit_with_receive(
        &program,
        context(&program),
        MaterialDelivery::BatchV1,
        MaterialReceive::FrozenWindow,
    )
    .unwrap();
    assert_eq!(scalar.transcript_digest(), window.transcript_digest());
    assert_eq!(scalar.frames_per_party(), window.frames_per_party());
    assert_eq!(
        scalar.primitive_requests_per_party(),
        window.primitive_requests_per_party()
    );
    assert!(window.combined_reservation_bytes() >= scalar.combined_reservation_bytes());
    // Python copies are a separate role owner, never charged to the native cap.
    assert!(
        window.combined_reservation_bytes()
            <= scalar.combined_reservation_bytes()
                + 2 * window
                    .material_receive()
                    .unwrap()
                    .native_reservation_bytes()
    );
    assert!(window.combined_reservation_bytes() <= MAX_ALLOCATION_BYTES);
    let mut dealer = TapeDealer::default();
    prepare_with_delivery(
        &program,
        context(&program),
        MaterialDelivery::BatchV1,
        &mut dealer,
    )
    .unwrap();
    assert!(dealer.finished && dealer.cross_plan);
    let original = dealer.records.each_ref().map(clone_records);
    let (sa, sr, si) = run_sealed(&program, original, false, None);
    let (wa, wr, wi) = run_sealed(&program, std::mem::take(&mut dealer.records), true, None);
    for ((scalar_result, window_result), (scalar_role, window_role)) in
        sa.iter().zip(&wa).zip(sr.iter().zip(&wr))
    {
        assert_eq!(
            scalar_result.as_ref().unwrap().values(),
            window_result.as_ref().unwrap().values()
        );
        assert_eq!(scalar_role.pcs, window_role.pcs);
        assert_eq!(scalar_role.wire, window_role.wire);
        assert_eq!(window_role.wire.len(), window.frames_per_party());
        assert!(window_role.scalar_calls + window_role.windows.len() < scalar_role.scalar_calls);
        assert!(window_role.windows.iter().any(|&n| n > 1));
        assert!(window_role
            .windows
            .iter()
            .all(|&n| (1..=MATERIAL_WINDOW_RECORDS).contains(&n)));
        assert!(window_role.inner.material.is_empty());
        assert_eq!(window_role.finishes, 1);
        assert_eq!(window_role.cancellations, 0);
    }
    let [(direct_a, trace_a), (direct_b, trace_b)] = direct(&program);
    assert_eq!(
        open(
            wa[0].as_ref().unwrap().values(),
            wa[1].as_ref().unwrap().values()
        ),
        open(direct_a.unwrap().values(), direct_b.unwrap().values())
    );
    for (((a, b), c), d) in wr[0]
        .inner
        .endpoint
        .trace
        .iter()
        .zip(&wr[1].inner.endpoint.trace)
        .zip(&trace_a)
        .zip(&trace_b)
    {
        assert_eq!((a.0, a.1), (c.0, c.1));
        assert_eq!(open(&a.2, &b.2), open(&c.2, &d.2));
    }
    si.drained();
    wi.drained();
}

#[test]
fn partial_missing_replay_cancel_and_provider_panic_burn_no_finish_no_suffix_fallback() {
    let program = fixture_schedule(1, 1, 2, crate::fss_decoder::OnlineSchedule::TensorV1);
    let mut dealer = TapeDealer::default();
    prepare_with_delivery(
        &program,
        context(&program),
        MaterialDelivery::BatchV1,
        &mut dealer,
    )
    .unwrap();
    for fault in [
        "partial",
        "capacity",
        "missing",
        "duplicate",
        "cancel",
        "panic",
    ] {
        let records = dealer.records.each_ref().map(clone_records);
        let (results, roles, inspection) = run_sealed(&program, records, true, Some(fault));
        assert!(results.iter().all(Result::is_err), "{fault}");
        for role in &roles {
            assert_eq!(role.cancellations, 1, "{fault}");
            assert_eq!(role.finishes, 0, "{fault}");
            assert!(role.inner.material.is_empty());
        }
        assert!(!roles[1].windows.is_empty());
        inspection.drained();
    }
}
