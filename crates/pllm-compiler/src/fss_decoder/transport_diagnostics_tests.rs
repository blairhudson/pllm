use super::*;
use crate::fss_decoder::transport::MAX_ALLOCATION_BYTES;

const STAGES: [(AdmissionStage, &str); 13] = [
    (AdmissionStage::Manifest, "manifest"),
    (AdmissionStage::ReplayGuard, "replay_guard"),
    (AdmissionStage::ScheduleWalk, "schedule_walk"),
    (AdmissionStage::PlanTransport, "plan_transport"),
    (AdmissionStage::Wavefront, "wavefront"),
    (AdmissionStage::KernelCounts, "kernel_counts"),
    (AdmissionStage::CumulativePackets, "cumulative_packets"),
    (AdmissionStage::TransportPeak, "transport_peak"),
    (AdmissionStage::ScheduleConsistency, "schedule_consistency"),
    (AdmissionStage::AggregateOwners, "aggregate_owners"),
    (AdmissionStage::AggregateArithmetic, "aggregate_arithmetic"),
    (AdmissionStage::BatchResources, "batch_resources"),
    (
        AdmissionStage::AggregateReservation,
        "aggregate_reservation",
    ),
];

#[test]
fn aggregate_equation_exact_limit_and_one_short() {
    let r = Resources {
        metadata_bytes: 17,
        live_array_bytes_per_party: 23,
        source_table_cache_bytes_per_role: 29,
        ..Resources::default()
    };
    let batch = BatchResources::new().unwrap();
    let batch_bytes = batch.native_reservation_bytes() + batch.python_overlap_bytes();
    let peak = MAX_ALLOCATION_BYTES - (3 * 17 + 2 * 23 + 3 * 29 + 64 * 1024 + batch_bytes);
    let (owners, base) = aggregate_base(r, peak).unwrap();
    assert_eq!(base, 3 * 17 + 2 * 23 + 3 * 29 + peak + 64 * 1024);
    assert_eq!(owners.metadata_bytes, 3 * 17);
    assert_eq!(owners.live_array_bytes, 2 * 23);
    assert_eq!(owners.source_table_cache_bytes, 3 * 29);
    assert_eq!(owners.transport_peak_bytes, peak);
    assert_eq!(owners.fixed_metadata_bytes, 64 * 1024);
    assert_eq!(owners.batch_native_bytes, 0);
    assert_eq!(owners.batch_python_bytes, 0);
    owners.check_limit(base, MAX_ALLOCATION_BYTES).unwrap();

    let combined = aggregate_batch(owners, base, &batch).unwrap();
    let owners = owners.with_batch(Some(&batch));
    assert_eq!(combined, MAX_ALLOCATION_BYTES);
    assert_eq!(owners.batch_native_bytes, batch.native_reservation_bytes());
    assert_eq!(owners.batch_python_bytes, batch.python_overlap_bytes());
    owners.check_limit(combined, MAX_ALLOCATION_BYTES).unwrap();
    let failure = owners
        .check_limit(combined, MAX_ALLOCATION_BYTES - 1)
        .unwrap_err();
    assert_eq!(failure.stage, AdmissionStage::AggregateReservation);
    assert_eq!(failure.error, Error::Resource);
    assert_eq!(failure.kernel_ordinal, None);
    assert_eq!(
        failure.bound,
        Some(AdmissionBound {
            observed: combined,
            limit: MAX_ALLOCATION_BYTES - 1,
        })
    );
    assert_eq!(failure.owners, Some(owners));
    assert_eq!(Error::from(failure), Error::Resource);

    // Same production limit, exactly one byte over, without changing any cap.
    let (over_owners, over_base) = aggregate_base(r, peak + 1).unwrap();
    let over = aggregate_batch(over_owners, over_base, &batch).unwrap();
    let failure = over_owners
        .with_batch(Some(&batch))
        .check_limit(over, MAX_ALLOCATION_BYTES)
        .unwrap_err();
    assert_eq!(failure.bound.unwrap().observed, MAX_ALLOCATION_BYTES + 1);
    assert_eq!(failure.bound.unwrap().limit, MAX_ALLOCATION_BYTES);
}

#[test]
fn aggregate_overflow_preserves_resource_without_inventing_observation() {
    for r in [
        Resources {
            metadata_bytes: usize::MAX,
            ..Resources::default()
        },
        Resources {
            live_array_bytes_per_party: usize::MAX,
            ..Resources::default()
        },
        Resources {
            source_table_cache_bytes_per_role: usize::MAX,
            ..Resources::default()
        },
        Resources {
            metadata_bytes: usize::MAX / 3,
            live_array_bytes_per_party: 1,
            ..Resources::default()
        },
        Resources {
            metadata_bytes: usize::MAX / 3,
            source_table_cache_bytes_per_role: 1,
            ..Resources::default()
        },
    ] {
        let failure = aggregate_base(r, 0).unwrap_err();
        assert_eq!(failure.stage, AdmissionStage::AggregateOwners);
        assert_eq!(failure.bound, None);
        assert_eq!(failure.owners, None); // Not all scaled terms are representable.
        assert_eq!(Error::from(failure), Error::Resource);
    }
    for metadata_bytes in [0, 1] {
        let failure = aggregate_base(
            Resources {
                metadata_bytes,
                ..Resources::default()
            },
            usize::MAX,
        )
        .unwrap_err();
        assert_eq!(failure.stage, AdmissionStage::AggregateArithmetic);
        assert_eq!(failure.bound, None);
        let owners = failure.owners.unwrap();
        assert_eq!(owners.metadata_bytes, 3 * metadata_bytes);
        assert_eq!(owners.transport_peak_bytes, usize::MAX);
        assert_eq!(Error::from(failure), Error::Resource);
    }
    let batch = BatchResources::new().unwrap();
    let (owners, _) = aggregate_base(Resources::default(), 0).unwrap();
    for combined in [usize::MAX, usize::MAX - batch.native_reservation_bytes()] {
        let failure = aggregate_batch(owners, combined, &batch).unwrap_err();
        assert_eq!(failure.stage, AdmissionStage::AggregateArithmetic);
        assert_eq!(failure.bound, None);
        assert_eq!(failure.owners, Some(owners.with_batch(Some(&batch))));
        assert_eq!(Error::from(failure), Error::Resource);
    }
}

#[test]
fn all_stages_map_every_legacy_error_without_payload_or_new_error_kinds() {
    for (stage, name) in STAGES {
        assert_eq!(stage.name(), name);
        for error in [
            Error::Domain,
            Error::Unsupported,
            Error::Resource,
            Error::Allocation,
            Error::Randomness,
            Error::Context,
            Error::Consumed,
        ] {
            let failure = AdmissionFailure::new(stage, error);
            assert_eq!(failure.stage, stage);
            assert_eq!(failure.error, error);
            assert_eq!(failure.kernel_ordinal, None);
            assert_eq!(failure.bound, None);
            assert_eq!(failure.owners, None);
            assert_eq!(Error::from(failure), error);
        }
    }
    assert!(!std::mem::needs_drop::<AdmissionFailure>());
    assert!(!std::mem::needs_drop::<Progress>());
    assert!(std::mem::size_of::<AdmissionFailure>() < 128);
}

#[test]
fn visitor_progress_distinguishes_enclosing_schedule_from_failed_kernel_stage() {
    for stage in [
        AdmissionStage::PlanTransport,
        AdmissionStage::Wavefront,
        AdmissionStage::KernelCounts,
        AdmissionStage::CumulativePackets,
        AdmissionStage::TransportPeak,
    ] {
        let mut progress = Progress::kernel(stage, 7);
        if stage == AdmissionStage::CumulativePackets {
            progress.bound = Some(AdmissionBound {
                observed: 101,
                limit: 100,
            });
        }
        let failure = progress.failure(Error::Resource);
        assert_eq!(failure.stage, stage);
        assert_eq!(failure.kernel_ordinal, Some(7));
        assert_eq!(failure.bound, progress.bound);
        assert_eq!(failure.owners, None);
    }
    let failure = Progress::schedule().failure(Error::Context);
    assert_eq!(failure.stage, AdmissionStage::ScheduleWalk);
    assert_eq!(failure.kernel_ordinal, None);
    assert_eq!(failure.bound, None);
}
