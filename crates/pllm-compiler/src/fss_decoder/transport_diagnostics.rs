//! Fixed-size public admission diagnostics; no retained Plan, payload or strings.
use super::{add, mul, BatchResources, Error};
use crate::fss_decoder::Resources;

// Existing fixed adapter/ledger/callback metadata reservation, unchanged.
pub(super) const ADMISSION_METADATA_SLACK_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionStage {
    Manifest,
    ReplayGuard,
    ScheduleWalk,
    PlanTransport,
    Wavefront,
    KernelCounts,
    CumulativePackets,
    TransportPeak,
    ScheduleConsistency,
    AggregateOwners,
    AggregateArithmetic,
    BatchResources,
    AggregateReservation,
}

impl AdmissionStage {
    /// Closed allowlist, never a formatted nested error or caller-supplied text.
    pub fn name(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::ReplayGuard => "replay_guard",
            Self::ScheduleWalk => "schedule_walk",
            Self::PlanTransport => "plan_transport",
            Self::Wavefront => "wavefront",
            Self::KernelCounts => "kernel_counts",
            Self::CumulativePackets => "cumulative_packets",
            Self::TransportPeak => "transport_peak",
            Self::ScheduleConsistency => "schedule_consistency",
            Self::AggregateOwners => "aggregate_owners",
            Self::AggregateArithmetic => "aggregate_arithmetic",
            Self::BatchResources => "batch_resources",
            Self::AggregateReservation => "aggregate_reservation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionBound {
    pub observed: usize,
    pub limit: usize,
}

/// Exact charged terms, including role multiplicities. Not raw source metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdmissionOwners {
    pub metadata_bytes: usize,
    pub live_array_bytes: usize,
    pub source_table_cache_bytes: usize,
    pub transport_peak_bytes: usize,
    pub fixed_metadata_bytes: usize,
    pub batch_native_bytes: usize,
    pub batch_python_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionFailure {
    pub stage: AdmissionStage,
    pub error: Error,
    pub kernel_ordinal: Option<usize>,
    /// Absent when a nested API does not expose the failing count/limit, or when
    /// arithmetic overflow prevents representing an exact observed value.
    pub bound: Option<AdmissionBound>,
    pub owners: Option<AdmissionOwners>,
}

impl AdmissionFailure {
    pub fn new(stage: AdmissionStage, error: Error) -> Self {
        Self {
            stage,
            error,
            kernel_ordinal: None,
            bound: None,
            owners: None,
        }
    }
}

impl From<AdmissionFailure> for Error {
    fn from(failure: AdmissionFailure) -> Self {
        failure.error
    }
}

#[derive(Clone, Copy)]
pub(super) struct Progress {
    pub stage: AdmissionStage,
    kernel_ordinal: Option<usize>,
    pub bound: Option<AdmissionBound>,
}

impl Progress {
    pub fn schedule() -> Self {
        Self {
            stage: AdmissionStage::ScheduleWalk,
            kernel_ordinal: None,
            bound: None,
        }
    }

    pub fn kernel(stage: AdmissionStage, ordinal: usize) -> Self {
        Self {
            stage,
            kernel_ordinal: Some(ordinal),
            bound: None,
        }
    }

    pub fn failure(self, error: Error) -> AdmissionFailure {
        AdmissionFailure {
            stage: self.stage,
            error,
            kernel_ordinal: self.kernel_ordinal,
            bound: self.bound,
            owners: None,
        }
    }
}

// Diagnostic owners are inline, allocation-free, never retained by Admission.
// Their fixed stack metadata fits the existing metadata reservation; no new
// allowance or discount to any transport/Plan owner is introduced.
const _: () = assert!(
    std::mem::size_of::<AdmissionFailure>() + std::mem::size_of::<Progress>()
        <= ADMISSION_METADATA_SLACK_BYTES
);

pub(super) fn aggregate_base(
    r: Resources,
    peak: usize,
) -> Result<(AdmissionOwners, usize), AdmissionFailure> {
    // Preserve the original checked-operation order and underlying Error.
    let owner_error = |error| AdmissionFailure::new(AdmissionStage::AggregateOwners, error);
    let metadata = mul(3, r.metadata_bytes).map_err(owner_error)?;
    let arrays = mul(2, r.live_array_bytes_per_party).map_err(owner_error)?;
    let partial = add(metadata, arrays).map_err(owner_error)?;
    let cache = mul(3, r.source_table_cache_bytes_per_role).map_err(owner_error)?;
    let base = add(partial, cache).map_err(owner_error)?;
    let owners = AdmissionOwners {
        metadata_bytes: metadata,
        live_array_bytes: arrays,
        source_table_cache_bytes: cache,
        transport_peak_bytes: peak,
        fixed_metadata_bytes: ADMISSION_METADATA_SLACK_BYTES,
        batch_native_bytes: 0,
        batch_python_bytes: 0,
    };
    let combined = add(base, peak)
        .and_then(|n| add(n, ADMISSION_METADATA_SLACK_BYTES))
        .map_err(|error| owners.arithmetic_failure(error))?;
    Ok((owners, combined))
}

impl AdmissionOwners {
    pub(super) fn with_batch(mut self, batch: Option<&BatchResources>) -> Self {
        if let Some(batch) = batch {
            self.batch_native_bytes = batch.native_reservation_bytes();
            self.batch_python_bytes = batch.python_overlap_bytes();
        }
        self
    }

    fn arithmetic_failure(self, error: Error) -> AdmissionFailure {
        AdmissionFailure {
            owners: Some(self),
            ..AdmissionFailure::new(AdmissionStage::AggregateArithmetic, error)
        }
    }

    pub(super) fn check_limit(self, combined: usize, limit: usize) -> Result<(), AdmissionFailure> {
        if combined > limit {
            return Err(AdmissionFailure {
                bound: Some(AdmissionBound {
                    observed: combined,
                    limit,
                }),
                owners: Some(self),
                ..AdmissionFailure::new(AdmissionStage::AggregateReservation, Error::Resource)
            });
        }
        Ok(())
    }
}

pub(super) fn aggregate_batch(
    owners: AdmissionOwners,
    combined: usize,
    batch: &BatchResources,
) -> Result<usize, AdmissionFailure> {
    let owners = owners.with_batch(Some(batch));
    add(combined, batch.native_reservation_bytes())
        .and_then(|n| add(n, batch.python_overlap_bytes()))
        .map_err(|error| owners.arithmetic_failure(error))
}

#[cfg(test)]
#[path = "transport_diagnostics_tests.rs"]
mod tests;
