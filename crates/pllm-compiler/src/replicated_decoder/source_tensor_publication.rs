//! Source-bound preparation between original numeric emission and finalization.
//!
//! This first contract covers the original public-matrix raw product of ONE
//! genuine original input declaration, followed by its original Q18 rescale.
//! It does not certify an arbitrary derived View, graph operation, checked row,
//! sticky bit, or complete request. Those obligations cannot be inferred from
//! the input interval, source digest, or the closed factory classification.
use super::*;
use super::original_import::{InputRecord, OriginalImportedView};
use crate::replicated_decoder::{
    prepared_tensor_map::PreparedTensorMap,
    source_registry::{decoder_plan_metadata_bytes, source_metadata_bytes},
    PublicSource,
};
use std::cell::Cell;

/// Intentionally unresolved. No method turns these declarations into proofs.
pub(in crate::replicated_decoder::program) struct RequiredChecks {
    pub input_owner: usize,
    pub input_bounds: Interval,
    pub rows: usize,
    pub input_width: usize,
    pub output_width: usize,
}

/// Original source and input owners outlive the native templates. Fields are
/// private; neither a digest nor an arbitrary native Program constructs this.
pub(in crate::replicated_decoder::program) struct SourceTensorPublication<'a> {
    prepared: PreparedTensorMap,
    source: Arc<PublicSource>,
    input: OriginalImportedView<'a>,
    raw: usize,
    weight: usize,
    required: RequiredChecks,
    preparation_peak: usize,
    native_work: WorkBudget,
}

fn plus(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}

fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}

/// Exact rescale invocation schedule, not one aggregate truncation estimate.
fn rescale_work(lanes: usize) -> Result<WorkBudget, Error> {
    if lanes == 0 || lanes > MAX_VALUES {
        return Err(Error::Bounds);
    }
    let width = lanes.min(tensor::MAX_TILE_LANES);
    let work = WorkBudget::rescale(width)?.checked_mul((lanes / width) as u64)?;
    if lanes % width == 0 {
        Ok(work)
    } else {
        work.checked_add(WorkBudget::rescale(lanes % width)?)
    }
}

/// Deliberately bounded raw-prefix census. Do not silently accept more numeric
/// ancestry: those emitters need a genuine derived-value seal and their actual
/// owner census, not a coordinate/digest assertion retrofitted onto this input.
fn prefix_bytes(builder: &Builder, input: usize) -> Result<usize, Error> {
    if !builder.tensors.is_empty()
        || !builder.row_norms.is_empty()
        || !builder.row_energies.is_empty()
        || !builder.row_masses.is_empty()
        || !builder.attention_energies.is_empty()
        || !builder.public_norms.is_empty()
        || !builder.token_domains.is_empty()
        || builder.validity.is_some()
    {
        return Err(Error::Binding);
    }
    // allocated includes the original dynamic Words, Input keys and payload
    // allowances. Charge inline spare Node capacity as well (no alias credit).
    let mut bytes = plus(builder.allocated, size_of::<Builder>())?;
    bytes = plus(bytes, array::<Node>(builder.nodes.capacity())?)?;
    bytes = plus(
        bytes,
        array::<Arc<InputRecord>>(builder.original_inputs.capacity())?,
    )?;
    for reserved in [
        array::<(usize, u128)>(builder.row_norms.capacity())?,
        array::<(Words, u128)>(builder.row_energies.capacity())?,
        array::<(usize, u64)>(builder.row_masses.capacity())?,
        array::<(Words, u128)>(builder.attention_energies.capacity())?,
    ] {
        bytes = plus(bytes, reserved)?;
    }
    for node in &builder.nodes {
        match &node.code {
            Code::Input(_) if std::ptr::eq(node, &builder.nodes[input]) => {}
            Code::PublicInput { key, .. } => bytes = plus(bytes, key.capacity())?,
            Code::Gather(_) | Code::PublicProduct { .. } => {}
            _ => return Err(Error::Binding),
        }
    }
    Ok(bytes)
}

impl<'a> SourceTensorPublication<'a> {
    /// Call immediately after the ORIGINAL public_matrix(..., rescale=false)
    /// emitted its raw PublicProduct. No Program/Compiled/finalization is needed.
    /// `owners` remains the enclosing caller's real outside-owner obligation;
    /// the actual prefix, complete original plan, sources, file metadata and new
    /// wrapper/move slots are added here, not replaced by caller byte estimates.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn prepare(
        builder: &mut Builder,
        validated: ValidatedSourceDescriptors<'a>,
        input: &OriginalImportedView<'a>,
        source: &Arc<PublicSource>,
        raw: usize,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<Self, Error> {
        input.check(builder, input.registry(), input.plan(), input.graph())?;
        let (records, norms) = validated.public_records()?;
        if !std::ptr::eq(records, input.registry().source_records())
            || !std::ptr::eq(norms, input.registry().source_norms())
            || !builder.public_weights
        {
            return Err(Error::Binding);
        }
        source.check_registry(input.registry())?;
        let node = builder.nodes.get(raw).ok_or(Error::Binding)?;
        let Code::PublicProduct {
            input: gathered,
            weight,
            rows,
            inner,
            columns,
            transposed: false,
        } = node.code
        else {
            return Err(Error::Binding);
        };
        let Code::Gather(words) = &builder.nodes.get(gathered).ok_or(Error::Binding)?.code else {
            return Err(Error::Binding);
        };
        let logical = input.logical();
        let input_lanes = logical
            .dimensions()
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(Error::Bounds))?;
        if logical.dimensions().last() != Some(&inner)
            || words.len() != input_lanes
            || builder.nodes[gathered].bounds != input.bounds()
            || !words
                .iter()
                .enumerate()
                .all(|(i, word)| word == (input.owner(), i))
            || rows.checked_mul(inner) != Some(words.len())
            || rows.checked_mul(columns) != Some(node.lanes)
        {
            return Err(Error::Shape);
        }
        let source_node = builder.nodes.get(weight).ok_or(Error::Binding)?;
        let Code::PublicInput {
            key,
            digest,
            row_l1,
            row_squared_norm,
            row_l1_squared_sum,
        } = &source_node.code
        else {
            return Err(Error::Binding);
        };
        let record = records.get(key).ok_or(Error::Binding)?;
        if record.shape != [columns, inner]
            || record.public_q18.is_some()
            || record.digest != *digest
            || record.row_l1 != *row_l1
            || record.row_squared_norm != *row_squared_norm
            || record.row_l1_squared_sum != *row_l1_squared_sum
            || source_node.bounds != Interval::new(record.lower, record.upper)?
            || source_node.lanes != inner.checked_mul(columns).ok_or(Error::Bounds)?
        {
            return Err(Error::Binding);
        }
        // Retain the actual authenticated PublicFile owner, not just its digest.
        source.file(key)?;
        let native_work = rescale_work(node.lanes)?;
        let (outside, work, descriptors) =
            compact_admission::input_construction_baseline(prior, owners)?;
        work.checked_add(schedule_work(&builder.nodes)?)?
            .checked_add(native_work)?
            .limits()?;
        if plus(plus(descriptors, builder.nodes.len())?, 1)? > MAX_NODES {
            return Err(Error::Bounds);
        }
        let mut baseline = plus(outside, prefix_bytes(builder, input.owner())?)?;
        baseline = plus(baseline, decoder_plan_metadata_bytes(input.plan(), cap)?)?;
        baseline = plus(baseline, source_metadata_bytes(records, norms)?)?;
        baseline = plus(baseline, source.resident_metadata_bytes()?)?;
        baseline = plus(baseline, array::<usize>(2)?)?; // original PublicSource Arc counters
        for bytes in [
            size_of::<SourceRegistry<'_>>(),
            size_of::<Self>(),                     // assembled wrapper
            size_of::<Result<Self, Error>>(),      // return/move
            size_of::<Result<Self, Error>>(),      // caller destination
            size_of::<PreparedTensorMap>(),        // prepared local -> wrapper move
            size_of::<OriginalImportedView<'_>>(), // genuine seal clone/move
            size_of::<Result<OriginalImportedView<'_>, Error>>(), // checked retain result
            size_of::<Arc<PublicSource>>(),        // source clone/move
            size_of::<Cell<usize>>(),              // observed actual admitted maximum
            size_of::<RequiredChecks>(),
            size_of::<WorkBudget>(),           // native-work destination
            size_of::<WorkBudget>(),           // combined enclosing-work destination
            size_of::<Result<usize, Error>>(), // publication result
        ] {
            baseline = plus(baseline, bytes)?;
        }
        let peak = Cell::new(0usize);
        let limit = cap.min(MAX_PROGRAM_BYTES);
        if owners.prior_request_peak_bytes > limit {
            return Err(Error::Bounds);
        }
        let prepared =
            PreparedTensorMap::rescale(node.lanes, node.bounds, 18, baseline, |bytes| {
                if bytes > limit {
                    return Err(Error::Bounds);
                }
                peak.set(peak.get().max(bytes));
                Ok(())
            })?;
        if prepared.work_budget()? != native_work {
            return Err(Error::Binding);
        }
        Ok(Self {
            prepared,
            source: source.clone(),
            input: input.retain_checked(builder)?,
            raw,
            weight,
            required: RequiredChecks {
                input_owner: input.owner(),
                input_bounds: input.bounds(),
                rows,
                input_width: inner,
                output_width: columns,
            },
            preparation_peak: peak.get().max(owners.prior_request_peak_bytes),
            native_work,
        })
    }

    pub(in crate::replicated_decoder::program) fn prepared(&self) -> &PreparedTensorMap {
        &self.prepared
    }

    pub(in crate::replicated_decoder::program) fn required_checks(&self) -> &RequiredChecks {
        &self.required
    }

    pub(in crate::replicated_decoder::program) fn preparation_peak(&self) -> usize {
        self.preparation_peak
    }

    pub(in crate::replicated_decoder::program) fn native_work(&self) -> WorkBudget {
        self.native_work
    }

    pub(in crate::replicated_decoder::program) fn check(
        &self,
        builder: &Builder,
        input: &OriginalImportedView<'_>,
        source: &Arc<PublicSource>,
        candidate: &PreparedTensorMap,
    ) -> Result<(), Error> {
        self.input
            .check(builder, input.registry(), input.plan(), input.graph())?;
        if !self.input.same_declaration(input)
            || !Arc::ptr_eq(&self.source, source)
            || !std::ptr::eq(self.input.registry(), input.registry())
            || candidate.lanes() != self.prepared.lanes()
        {
            return Err(Error::Binding);
        }
        for (original, candidate) in self.prepared.programs().zip(candidate.programs()) {
            if !Arc::ptr_eq(original, candidate) {
                return Err(Error::Binding);
            }
        }
        if self.prepared.programs().count() != candidate.programs().count() {
            return Err(Error::Binding);
        }
        let node = builder.nodes.get(self.raw).ok_or(Error::Binding)?;
        if !matches!(node.code, Code::PublicProduct { weight, rows, inner, columns, transposed: false, .. }
            if weight == self.weight && rows == self.required.rows
            && inner == self.required.input_width && columns == self.required.output_width)
            || !self.prepared.accepts(node.lanes, node.bounds)
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    /// Ordinary original publication; no runtime, default dispatch, proof or
    /// drain gate is enabled. Keep this owner alive: it carries the UNRESOLVED
    /// checks alongside the emitted logical ordinal. No Arc is rebuilt.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn publish_original(
        &self,
        builder: &mut Builder,
        input: &OriginalImportedView<'_>,
        source: &Arc<PublicSource>,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<usize, Error> {
        self.check(builder, input, source, &self.prepared)?;
        let (outside, work, descriptors) =
            compact_admission::input_construction_baseline(prior, owners)?;
        work.checked_add(schedule_work(&builder.nodes)?)?
            .checked_add(self.native_work)?
            .limits()?;
        if plus(plus(descriptors, builder.nodes.len())?, 1)? > MAX_NODES {
            return Err(Error::Bounds);
        }
        let mut bytes = plus(outside, prefix_bytes(builder, input.owner())?)?;
        for owned in [
            decoder_plan_metadata_bytes(input.plan(), cap)?,
            source_metadata_bytes(
                input.registry().source_records(),
                input.registry().source_norms(),
            )?,
            source.resident_metadata_bytes()?,
            array::<usize>(2)?,
            size_of::<Self>(),
            self.prepared.baseline_bytes()?,
            size_of::<Node>(),
            array::<u64>(plus(self.prepared.lanes(), self.prepared.lanes())?)?,
            4096, // existing Builder TensorLoop charge; not a new allowance
            size_of::<Result<usize, Error>>(),
        ] {
            bytes = plus(bytes, owned)?;
        }
        // Original interner charges remain in addition to this retained owner.
        for program in self.prepared.programs() {
            bytes = plus(bytes, plus(program.admitted_bytes(), 1024)?)?;
        }
        let next = plus(builder.nodes.len(), 1)?;
        let grows = next > builder.nodes.capacity();
        if grows {
            bytes = plus(bytes, array::<Node>(next)?)?;
        }
        if bytes.max(owners.prior_request_peak_bytes) > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        if grows {
            builder
                .nodes
                .try_reserve_exact(1)
                .map_err(|_| Error::Allocation)?;
            if builder.nodes.capacity() != next {
                return Err(Error::Allocation);
            }
        }
        builder.prepared_tensor_map(self.raw, &self.prepared)
    }
}

#[path = "original_produced_view.rs"]
pub(in crate::replicated_decoder::program) mod produced;

#[path = "projection_custody.rs"]
mod custody;

impl<'s> SourceTensorPublication<'s> {
    /// Storage-neutral source lowering over genuine imports OR produced owners.
    /// Unlike prepare(raw), this consumes no expanded raw owner/Words at all.
    /// Complete-row/sticky obligations remain separate and fail closed above.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder::program) fn prepare_binding(
        builder: &Builder,
        input: &impl produced::OriginalInputBinding<'s>,
        source: &Arc<PublicSource>,
        operation: &'s pllm_models::ModelOperation,
        prior: &compact_admission::RequestWork,
        owners: &compact_admission::RequestOwners<'_>,
        cap: usize,
    ) -> Result<produced::OriginalProducedView<'s>, Error> {
        produced::OriginalProducedView::projection(
            builder, input, source, operation, prior, owners, cap,
        )
    }
}

#[cfg(test)]
#[path = "source_tensor_publication_tests.rs"]
mod tests;
