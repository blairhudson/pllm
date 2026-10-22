//! Wire as a child module of transport_tests.rs to reuse its existing fixture.
use super::*;

// Derive transport terms from individual public material steps, not Admission or
// Plan::transport_resources*. No material is issued and no private Plan is edited.
struct PublicWalk {
    hash: Sha256,
    kernels: usize,
    requests: usize,
    frames: usize,
    packet: usize,
    cumulative: usize,
    frame: usize,
    peak: usize,
    guard: usize,
    wavefront: bool,
}

impl PublicWalk {
    fn new(program: &Program) -> Self {
        Self {
            hash: Sha256::new()
                .chain_update(b"pllm.fss-decoder.transport-schedule.v1\0")
                .chain_update(program.digest()),
            kernels: 0,
            requests: 0,
            frames: 0,
            packet: 0,
            cumulative: 0,
            frame: 0,
            peak: 0,
            guard: ImportGuard::indexed_reservation_bytes(
                program.resources().primitive_requests_per_party.max(1),
            )
            .unwrap(),
            wavefront: program.online_schedule().uses_wavefront(),
        }
    }
}

impl MaterialVisitor for PublicWalk {
    fn visit(&mut self, context: Context, plan: &Plan<'_>) -> Result<(), Error> {
        // Spell out the identity encoding independently of hash_start/hash_kernel.
        self.hash.update((self.kernels as u64).to_le_bytes());
        self.hash.update(plan.digest());
        self.hash.update(context.plan);
        self.hash.update(context.session);
        self.hash.update(context.operation);
        self.hash.update(context.first_tensor_index.to_le_bytes());
        let r = plan.resources();
        let mut packet = 0;
        let mut import = 0;
        let mut requests = 0;
        for pc in plan.material_steps() {
            let step = plan.step_transport_resources(pc)?;
            requests += 1;
            self.cumulative += step.packet_bytes_bound;
            packet = packet.max(step.packet_bytes_bound);
            import = import.max(step.import_overlap_bytes);
        }
        assert_eq!(requests, r.secret_steps);
        self.requests += requests;
        self.packet = self.packet.max(packet);
        let mut frame = r.retained_frame_overlap_bytes / 2;
        let mut rounds = r.frames_per_party;
        let mut extra = 0;
        if self.wavefront && !plan.is_atomic_vector() {
            let schedule = wavefront::Schedule::new(plan)?;
            self.hash.update(schedule.digest());
            frame = schedule.resources().largest_frame_bytes;
            rounds = schedule.resources().rounds;
            extra = schedule.resources().additional_reservation_bytes;
        }
        self.frames += rounds;
        self.frame = self.frame.max(frame);
        // Six retained packet slots; two workers, one dealer; eight frame copies
        // per role. These are ownership terms, not an opaque debug snapshot.
        let worker = r.plan_reservation_bytes
            + r.register_owner_bytes_per_party
            + import
            + 6 * packet
            + r.retained_frame_overlap_bytes / 2
            + self.guard;
        let dealer = r.plan_reservation_bytes + r.largest_primitive_reservation_bytes + 8 * packet;
        self.peak = self.peak.max(2 * worker + dealer + 16 * frame + extra);
        self.kernels += 1;
        Ok(())
    }

    fn cancel(&mut self) {}
}

fn assert_failure(
    program: &Program,
    binding: Context,
    delivery: MaterialDelivery,
    expected: AdmissionFailure,
) {
    let failure = admit_with_delivery_diagnostic(program, binding, delivery).unwrap_err();
    assert_eq!(failure, expected);
    assert_eq!(Error::from(failure), expected.error);
    assert_eq!(
        admit_with_delivery(program, binding, delivery).unwrap_err(),
        expected.error
    );
}

#[test]
fn tiny_public_walk_pins_identity_and_transport_accounting() {
    for online in [
        crate::fss_decoder::OnlineSchedule::Serial,
        crate::fss_decoder::OnlineSchedule::WavefrontV1,
        crate::fss_decoder::OnlineSchedule::TensorV1,
    ] {
        let program = fixture_schedule(1, 1, 1, online);
        let binding = context(&program);
        let mut walk = PublicWalk::new(&program);
        program.visit_material(binding, &mut walk).unwrap();
        if online == crate::fss_decoder::OnlineSchedule::Serial {
            // One token selector, three norms, 15 linear outputs (6+2+2+2+3),
            // one score, one scale, one softmax, two values, two GELUs, one argmax.
            assert_eq!(walk.kernels, 1 + 3 + 15 + 1 + 1 + 1 + 2 + 2 + 1);
        }
        let r = program.resources();
        let base = 3 * r.metadata_bytes
            + 2 * r.live_array_bytes_per_party
            + 3 * r.source_table_cache_bytes_per_role
            + walk.peak
            + 64 * 1024;
        let transcript: [u8; 32] = walk.hash.finalize().into();
        for delivery in [MaterialDelivery::Single, MaterialDelivery::BatchV1] {
            let admitted = admit_with_delivery_diagnostic(&program, binding, delivery).unwrap();
            assert_eq!(admitted.material_delivery(), delivery);
            assert_eq!(admitted.kernels(), walk.kernels);
            assert_eq!(admitted.primitive_requests_per_party(), walk.requests);
            assert_eq!(admitted.frames_per_party(), walk.frames);
            assert_eq!(admitted.largest_packet_bytes(), walk.packet);
            assert_eq!(admitted.peer_frame_bytes_bound(), walk.frame);
            assert_eq!(
                admitted.cumulative_packet_bytes_per_party_bound(),
                walk.cumulative
            );
            assert_eq!(admitted.guard_capacity(), walk.requests);
            assert_eq!(admitted.guard_reservation_bytes(), walk.guard);
            assert_eq!(admitted.guard_disk_bytes(), 0);
            assert_eq!(admitted.transcript_digest(), transcript);
            let batch = admitted.delivery_batch();
            assert_eq!(batch.is_some(), delivery == MaterialDelivery::BatchV1);
            assert_eq!(
                admitted.combined_reservation_bytes(),
                base + batch.map_or(0, |b| {
                    b.native_reservation_bytes() + b.python_overlap_bytes()
                })
            );
        }
    }
}

#[test]
fn completed_visitors_do_not_leak_last_kernel_stage_into_schedule_failure() {
    for online in [
        crate::fss_decoder::OnlineSchedule::Serial,
        crate::fss_decoder::OnlineSchedule::WavefrontV1,
        crate::fss_decoder::OnlineSchedule::TensorV1,
    ] {
        let mut program = fixture_schedule(1, 1, 1, online);
        let binding = context(&program);
        // Only the final obligation comparison changes: every actual visitor has
        // passed transport, wavefront, counts, cumulative packets and peak first.
        program.obligations[0] ^= 1;
        let mut walk = PublicWalk::new(&program);
        assert_eq!(
            program.visit_material(binding, &mut walk),
            Err(Error::Context)
        );
        assert_eq!(walk.kernels, program.resources().kernels);
        assert!(walk.kernels > 1);
        for delivery in [MaterialDelivery::Single, MaterialDelivery::BatchV1] {
            assert_failure(
                &program,
                binding,
                delivery,
                AdmissionFailure::new(AdmissionStage::ScheduleWalk, Error::Context),
            );
        }
    }
}

#[test]
fn completed_walk_checks_both_kernel_and_request_totals() {
    for requests in [false, true] {
        let mut program = fixture(1, 1, 1);
        let binding = context(&program);
        if requests {
            program.resources.primitive_requests_per_party += 1;
        } else {
            program.resources.kernels += 1;
        }
        for delivery in [MaterialDelivery::Single, MaterialDelivery::BatchV1] {
            assert_failure(
                &program,
                binding,
                delivery,
                AdmissionFailure::new(AdmissionStage::ScheduleConsistency, Error::Context),
            );
        }
    }
}

#[test]
fn completed_walk_reports_exact_aggregate_limit_not_stale_packet_bound() {
    let mut program = fixture(1, 1, 1);
    let binding = context(&program);
    let mut walk = PublicWalk::new(&program);
    program.visit_material(binding, &mut walk).unwrap();
    // This resource summary is not used to build Plans or fund the native walk.
    // Inflate only its charged cache owner; leave schedules and all caps intact.
    program.resources.source_table_cache_bytes_per_role = MAX_ALLOCATION_BYTES / 3;
    let r = program.resources();
    for delivery in [MaterialDelivery::Single, MaterialDelivery::BatchV1] {
        let batch = (delivery == MaterialDelivery::BatchV1).then(|| BatchResources::new().unwrap());
        let owners = AdmissionOwners {
            metadata_bytes: 3 * r.metadata_bytes,
            live_array_bytes: 2 * r.live_array_bytes_per_party,
            source_table_cache_bytes: 3 * r.source_table_cache_bytes_per_role,
            transport_peak_bytes: walk.peak,
            fixed_metadata_bytes: 64 * 1024,
            batch_native_bytes: batch.as_ref().map_or(0, |b| b.native_reservation_bytes()),
            batch_python_bytes: batch.as_ref().map_or(0, |b| b.python_overlap_bytes()),
        };
        let observed = owners.metadata_bytes
            + owners.live_array_bytes
            + owners.source_table_cache_bytes
            + owners.transport_peak_bytes
            + owners.fixed_metadata_bytes
            + owners.batch_native_bytes
            + owners.batch_python_bytes;
        assert!(observed > MAX_ALLOCATION_BYTES);
        assert_failure(
            &program,
            binding,
            delivery,
            AdmissionFailure {
                stage: AdmissionStage::AggregateReservation,
                error: Error::Resource,
                kernel_ordinal: None,
                bound: Some(AdmissionBound {
                    observed,
                    limit: MAX_ALLOCATION_BYTES,
                }),
                owners: Some(owners),
            },
        );
    }

    // Same real walk, but multiplication overflow has no representable bound or
    // complete owner record. Do not reuse the preceding failure's geometry.
    program.resources.source_table_cache_bytes_per_role = usize::MAX;
    for delivery in [MaterialDelivery::Single, MaterialDelivery::BatchV1] {
        assert_failure(
            &program,
            binding,
            delivery,
            AdmissionFailure::new(AdmissionStage::AggregateOwners, Error::Resource),
        );
    }
}

#[test]
fn diagnostic_and_legacy_admission_have_identical_success_and_identity() {
    for delivery in [MaterialDelivery::Single, MaterialDelivery::BatchV1] {
        let program = fixture(1, 1, 1);
        let binding = context(&program);
        let old = admit_with_delivery(&program, binding, delivery).unwrap();
        let diagnostic = admit_with_delivery_diagnostic(&program, binding, delivery).unwrap();
        // Every Admission field, including schedule identity, remains identical.
        assert_eq!(format!("{old:?}"), format!("{diagnostic:?}"));
    }
}

#[test]
fn native_failures_keep_legacy_errors_and_truthful_enclosing_stages() {
    let mut program = fixture(1, 1, 1);
    let mut binding = context(&program);
    binding.session = [0; 32];
    let failure =
        admit_with_delivery_diagnostic(&program, binding, MaterialDelivery::Single).unwrap_err();
    assert_eq!(failure.stage, AdmissionStage::ScheduleWalk);
    assert_eq!(failure.error, Error::Context);
    assert_eq!(failure.kernel_ordinal, None);
    assert_eq!(failure.bound, None);
    assert_eq!(
        admit_with_delivery(&program, binding, MaterialDelivery::Single).unwrap_err(),
        failure.error
    );

    let binding = context(&program);
    program.resources.primitive_requests_per_party = usize::MAX;
    let failure =
        admit_with_delivery_diagnostic(&program, binding, MaterialDelivery::BatchV1).unwrap_err();
    assert_eq!(failure.stage, AdmissionStage::ReplayGuard);
    assert_eq!(failure.error, Error::Resource);
    assert_eq!(failure.bound, None); // Nested guard API exposes no failing limit.
    assert_eq!(failure.owners, None);
    assert_eq!(
        admit_with_delivery(&program, binding, MaterialDelivery::BatchV1).unwrap_err(),
        failure.error
    );

    let mut program = fixture(1, 1, 1);
    let binding = context(&program);
    program.resources.kernels += 1;
    let failure =
        admit_with_delivery_diagnostic(&program, binding, MaterialDelivery::Single).unwrap_err();
    assert_eq!(failure.stage, AdmissionStage::ScheduleConsistency);
    assert_eq!(failure.error, Error::Context);
    assert_eq!(admit(&program, binding).unwrap_err(), failure.error);
}
