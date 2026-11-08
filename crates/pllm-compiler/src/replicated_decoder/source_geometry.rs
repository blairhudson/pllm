//! Compiler-private geometry custody, not numeric/executable admission.
//! No manifest, collection, directory, snapshot, issuer or source activation.
use super::*;
use pllm_models::DecoderPlan;

// Inactive candidate only. No ingress/request caller or owner-layout change.
mod metadata;

#[cfg(test)]
#[path = "source_geometry_metadata_tests.rs"]
mod metadata_tests;

/// OriginalSourcePlan owns the configured Arc<OwnedCharged<Context>> and its
/// configuration. Plan views borrow that owner; no clone or self-reference.
/// Payloads retire before their controller, which retires before the root.
pub(in crate::replicated_decoder) struct GeometryOwner {
    original: OriginalSourcePlan,
    experiment: OwnedCharged<json::ImportedValue>,
    limits: original_source::Limits,
    controller: OwnedExternalCharge,
    root: OwnedBudget,
}

type IngressParts = (
    OriginalSourcePlan,
    OwnedCharged<json::ImportedValue>,
    original_source::Limits,
    OwnedExternalCharge,
    OwnedBudget,
);

// Additional logical SOURCE controls introduced by this extraction: owner
// assembly/return/destination, Result return/propagation, decomposition moves,
// input and view references, and aggregate helper arguments/result. These are
// conservative typed slots, not a physical-stack or cleanup-funding proof.
fn additional_controller_bytes() -> Result<usize, Error> {
    [
        size_of::<GeometryOwner>().checked_mul(3),
        size_of::<Result<GeometryOwner, Error>>().checked_mul(2),
        size_of::<IngressParts>().checked_mul(3),
        size_of::<&[u8]>().checked_mul(2),
        Some(size_of::<&GeometryOwner>()),
        Some(size_of::<&OriginalSourcePlan>()),
        Some(size_of::<&DecoderPlan>()),
        size_of::<usize>().checked_mul(5),
        Some(size_of::<Result<usize, Error>>()),
    ]
    .into_iter()
    .try_fold(0, |sum, term| plus(sum, term.ok_or(Error::Bounds)?))
}

// No refund for anything except the actual replaced bootstrap payload.
fn aggregate_phase(used: usize, bootstrap: usize, working: usize) -> Result<usize, Error> {
    let phase = plus(used.checked_sub(bootstrap).ok_or(Error::Bounds)?, working)?;
    if phase > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    Ok(phase)
}

impl GeometryOwner {
    pub(in crate::replicated_decoder) fn prepare(
        experiment: &[u8], config: &[u8], rows: usize, outputs: usize,
    ) -> Result<Self, Error> {
        if experiment.is_empty()
            || experiment.len() > 32 << 10
            || config.is_empty()
            || config.len() > source_policy::CONFIG_BYTES
        {
            return Err(Error::Binding);
        }
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).map_err(|_| Error::Bounds)?;
        let controller = root
            .charge_external(plus(geometry_controller_floor()?, additional_controller_bytes()?)?)
            .map_err(|_| Error::Bounds)?;
        let experiment = import_json(&root, experiment)?;
        let config = import_json(&root, config)?;
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: u64::try_from(rows).map_err(|_| Error::Bounds)?,
            max_new_tokens: u64::try_from(outputs).map_err(|_| Error::Bounds)?,
        };
        let (limits, storage) =
            source_policy::admitted_choice(&experiment, &config, None, workload)?;
        if storage != SourceStorage::Compact {
            return Err(Error::Binding);
        }
        let bootstrap =
            original_source::bootstrap_bytes(&config, limits).map_err(|_| Error::Bounds)?;
        let quote = original_source::quote(
            config,
            root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,
            workload,
            limits,
        )
        .map_err(|_| Error::Binding)?;
        let phase = aggregate_phase(
            root.budget().used_bytes(), bootstrap, quote.resources().working_charge_bytes,
        )?;
        root.budget().admit_peak(phase).map_err(|_| Error::Bounds)?;
        let original = quote.construct().map_err(|_| Error::Binding)?;
        Ok(Self { original, experiment, limits, controller, root })
    }

    pub(in crate::replicated_decoder) fn original(&self) -> &OriginalSourcePlan {
        &self.original
    }

    pub(in crate::replicated_decoder) fn plan(&self) -> &DecoderPlan {
        self.original.plan()
    }

    // Only ingress (and its descendants) may consume this ownership capsule.
    pub(super) fn into_ingress(self) -> IngressParts {
        let Self { original, experiment, limits, controller, root } = self;
        (original, experiment, limits, controller, root)
    }
}

#[cfg(test)]
#[path = "source_geometry_tests.rs"]
mod tests;
