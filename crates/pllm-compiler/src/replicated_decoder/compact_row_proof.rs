//! Producer-qualified complete-row obligations and original bridge copy custody.
//!
//! Mounted as a child of body_tensor_integration. The
//! consuming export constructor must inspect that owner's PRIVATE Invocation,
//! Compiled and CoordinatorDrained fields. Do not mount at the decoder root or
//! replace this access with caller-provided Values/Plan/prepared-owner labels.
//!
//! Native producer qualification stays read-only; closed runtime custody may
//! additionally copy the exact row into a private pending input payload. The
//! destination proof gate stays closed. The original Plan Arc is retained;
//! digest equality is never owner identity.
//!
//! IMPORTANT: ConsumerObligation is NOT an imported proof. Existing APIs expose
//! no authenticated producer-export/consumer-input edge. Caller-selected input
//! IDs, matching intervals, source statistics, attention energy and compact
//! ProjectionRow coordinates cannot establish that edge. Consequently this
//! module neither creates CompleteRow values nor writes any certificate ledger.
//!
//! Minimal integration seam: the compiler's closed import/export mapping must
//! authenticate (original producer Arc, export index, export start, RowKey,
//! original consuming import map, exact destination Builder/input RowKey).
//! Only that owner may discharge a
//! ConsumerObligation and publish a destination proof via existing admission.
//! Do not replace this seam with a public label, digest, bool or numeric ticket.
//! Body Builder now supplies an admitted consuming import path pinned by an
//! exclusive borrow of the actual destination Builder. Its closed edge issuer
//! remains fail-closed: the main emitter has no authenticated compact copy map.
//! The missing seam is that executable emitter's immutable original
//! export-to-input binding, NOT permission to turn ConsumerObligation into proof.
//! Closed Plan endpoints below are inspection obligations, not that missing
//! mutable-Builder import authority or proof of an executable handoff.

use super::{CoordinatorDrained, Invocation};
use crate::replicated_decoder::runtime::body::layer::{
    Builder, CompleteRow, Layout, Plan, RowKey, Values,
};
use crate::replicated_decoder::{
    Code, Error, MAX_PROGRAM_BYTES, Secret, Worker, multiply::PartyRuntime, tensor,
};
use std::{mem::size_of, sync::Arc};

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}

fn admit(bytes: usize, limit: usize) -> Result<(), Error> {
    if bytes > limit.min(MAX_PROGRAM_BYTES) {
        Err(Error::Bounds)
    } else {
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    // Before retaining any new Arc owner, not after successful publication.
    static OWNER_RETAIN_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn retain(plan: &Arc<Plan>) -> Arc<Plan> {
    #[cfg(test)]
    OWNER_RETAIN_ATTEMPTS.with(|n| n.set(n.get() + 1));
    Arc::clone(plan)
}

/// Read-only native proof bound to an actual original producer export. This is
/// not authority for a consumer row, even if both rows have identical numbers.
pub(in crate::replicated_decoder) struct ExportedRow {
    producer: Arc<Plan>,
    export: usize,
    start: usize,
    proof: CompleteRow,
}

/// Allocation-free lookup: no temporary Layout::slice, Words or lane vector.
fn exported_proof(
    producer: &Plan,
    export: usize,
    start: usize,
    row: &Layout,
) -> Result<CompleteRow, Error> {
    let output = producer.output_layout(export)?;
    let end = start.checked_add(row.len()).ok_or(Error::Bounds)?;
    if row.len() == 0 || end > output.len() {
        return Err(Error::Shape);
    }
    // Lookup in the existing native ledger, not coordinates or numeric labels.
    let proof = producer.complete_row(row).ok_or(Error::Binding)?;
    if proof.producer() != producer.digest() {
        return Err(Error::Binding);
    }
    for index in 0..row.len() {
        if output.word(start + index)? != row.word(index)? {
            return Err(Error::Binding);
        }
    }
    Ok(proof)
}

impl ExportedRow {
    /// Owner-local quote before retaining an Arc. Native admitted bytes are a
    /// deliberately conservative bound, including workspace, not a sharing
    /// discount. ADD external original Program/Worker, authenticated file/source
    /// owners, caller containers and other live owners in retained_baseline.
    /// The borrowed query layout remains live during construction and is charged
    /// separately. No new layout backing or heap collection is constructed here.
    pub(in crate::replicated_decoder) fn quote(
        producer: &Arc<Plan>,
        row: &Layout,
        retained_baseline: usize,
    ) -> Result<usize, Error> {
        [
            producer.resources().admitted_bytes,
            2 * size_of::<usize>(), // Retained original Arc control block.
            size_of::<Layout>(),
            row.backing_bytes()?,
            2 * size_of::<Self>(), // Candidate and return overlap.
            size_of::<Result<Self, Error>>(),
        ]
        .into_iter()
        .try_fold(retained_baseline, add)
    }

    pub(in crate::replicated_decoder) fn from_export(
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        row: &Layout,
        retained_baseline: usize,
        limit: usize,
    ) -> Result<Self, Error> {
        let proof = exported_proof(producer, export, start, row)?;
        admit(Self::quote(producer, row, retained_baseline)?, limit)?;
        Ok(Self {
            producer: retain(producer),
            export,
            start,
            proof,
        })
    }

    /// Check this retained original owner/export, not an equal-digest replacement.
    /// Exact ordered aliases are accepted; layout Arc identity is not provenance.
    pub(in crate::replicated_decoder) fn check_export(
        &self,
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        row: &Layout,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(producer, &self.producer) || export != self.export || start != self.start {
            return Err(Error::Binding);
        }
        let proof = exported_proof(producer, export, start, row)?;
        if proof.key() != self.proof.key()
            || proof.energy() != self.proof.energy()
            || proof.mass() != self.proof.mass()
        {
            return Err(Error::Binding);
        }
        Ok(())
    }

    /// Original producer proof only; never rebound to a consumer's key/digest.
    pub(in crate::replicated_decoder) fn producer_proof(&self) -> CompleteRow {
        self.proof
    }

    /// Necessary binding/owner checks for a future destination import. The
    /// expected producer is the Arc retained here, not another caller label.
    /// This only quotes; it cannot issue Builder's authenticated consumer edge.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn destination_import_quote(
        &self,
        destination: &Builder,
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        source_row: &Layout,
        input_row: &Layout,
        retained_baseline: usize,
        temporary: usize,
    ) -> Result<usize, Error> {
        self.check_export(producer, export, start, source_row)?;
        // Include this still-live retained handle as well as all caller owners.
        let baseline = add(retained_baseline, size_of::<Self>())?;
        destination.complete_row_import_quote(
            &self.producer,
            producer,
            export,
            start,
            source_row,
            input_row,
            baseline,
            temporary,
        )
    }
}

fn input_key(consumer: &Plan, row: &Layout) -> Result<RowKey, Error> {
    let key = row.row_key(row.len()).ok_or(Error::Binding)?;
    let input = consumer
        .input_specs()
        .get(key.owner)
        .ok_or(Error::Binding)?;
    if key.offset.checked_add(key.width).ok_or(Error::Bounds)? > input.lanes {
        return Err(Error::Shape);
    }
    Ok(key)
}

/// Pinned endpoints needing authenticated edge discharge by the enclosing
/// compiler. Construction and check_endpoints do NOT authenticate value flow.
/// No conversion to CompleteRow or general input energy exists here.
pub(in crate::replicated_decoder) struct ConsumerObligation {
    exported: ExportedRow,
    consumer: Arc<Plan>,
    input: RowKey,
}

impl ConsumerObligation {
    /// No sharing credit, including producer == consumer. This quote adds both
    /// original owners and all construction/result slots before either retention.
    /// External source files and enclosing owners still require caller baseline.
    pub(in crate::replicated_decoder) fn quote(
        exported: &ExportedRow,
        consumer: &Arc<Plan>,
        row: &Layout,
        retained_baseline: usize,
    ) -> Result<usize, Error> {
        [
            exported.producer.resources().admitted_bytes,
            consumer.resources().admitted_bytes,
            4 * size_of::<usize>(),
            size_of::<ExportedRow>(), // Still-live incoming handle.
            size_of::<Layout>(),
            row.backing_bytes()?,
            2 * size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
        ]
        .into_iter()
        .try_fold(retained_baseline, add)
    }

    pub(in crate::replicated_decoder) fn new(
        exported: &ExportedRow,
        consumer: &Arc<Plan>,
        row: &Layout,
        retained_baseline: usize,
        limit: usize,
    ) -> Result<Self, Error> {
        let input = input_key(consumer, row)?;
        if input.width != exported.proof.key().width {
            return Err(Error::Binding);
        }
        admit(
            Self::quote(exported, consumer, row, retained_baseline)?,
            limit,
        )?;
        Ok(Self {
            exported: ExportedRow {
                producer: retain(&exported.producer),
                export: exported.export,
                start: exported.start,
                proof: exported.proof,
            },
            consumer: retain(consumer),
            input,
        })
    }

    /// Necessary endpoint checks only. A successful result still cannot authorize
    /// certificate import: the private original dataflow edge is unverified.
    pub(in crate::replicated_decoder) fn check_endpoints(
        &self,
        producer: &Arc<Plan>,
        export: usize,
        start: usize,
        exported_row: &Layout,
        consumer: &Arc<Plan>,
        input_row: &Layout,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(consumer, &self.consumer) || input_key(consumer, input_row)? != self.input {
            return Err(Error::Binding);
        }
        self.exported
            .check_export(producer, export, start, exported_row)
    }
}

/// Additive owner/control quote for extracting, retaining and restoring an
/// original export. The original phase ceiling is retained in FULL; none of its
/// Invocation/Executor/Values slots, fixed reserves or now-unused storage funds
/// this wrapper. Moves duplicate control slots here, never owning payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OriginalExportQuote {
    original_bytes: usize,
    finish_bytes: usize,
    retained_bytes: usize,
    restoration_bytes: usize,
    admitted_bytes: usize,
}

/// Solely a restoration output owner, not authority to construct an export. The
/// original Compiled Arc MOVES here without cloning. Keep even the exhausted
/// executor slot until Values erase; no moved-out source/runtime gap on failure.
struct OriginalRestoreOutputs {
    values: Values,
    _executor: Option<super::Executor>,
    _compiled: Arc<super::Compiled>,
}

impl OriginalExportQuote {
    fn quote_controls() -> Result<usize, Error> {
        [
            size_of::<Self>(), // Read-only census local.
            size_of::<Self>(), // Returned/moved census control.
            size_of::<Result<Self, Error>>(),
            size_of::<Result<(), Error>>(), // Preflight result.
            size_of::<[usize; 3]>(),        // Baseline, phase maximum and limit.
        ]
        .into_iter()
        .try_fold(0, add)
    }

    fn finish_controls() -> Result<usize, Error> {
        [
            Self::quote_controls()?,
            size_of::<CoordinatorDrained<Invocation>>(), // Consumed argument.
            size_of::<Invocation>(),                     // Extracted original owner control.
            size_of::<PartyRuntime>(),                   // Executor::finish tuple destructuring.
            size_of::<Values>(),
            size_of::<Result<(PartyRuntime, Values), Error>>(),
            size_of::<OriginalBridgeExport>(), // Wrapper construction candidate.
            size_of::<OriginalBridgeExport>(), // Moved wrapper destination.
            size_of::<Result<OriginalBridgeExport, Error>>(),
            size_of::<Result<usize, Error>>(), // Export length check.
        ]
        .into_iter()
        .try_fold(0, add)
    }

    fn retained_controls() -> Result<usize, Error> {
        add(Self::quote_controls()?, size_of::<OriginalBridgeExport>())
    }

    fn owned_restoration_controls() -> Result<usize, Error> {
        [
            // Caller-side output-owner construction, separate from helper's
            // consumed argument/destruction and full guard/move/result controls.
            size_of::<OriginalRestoreOutputs>(),
            crate::replicated_decoder::runtime::body::handoff::owned_restore_inline_bytes::<
                OriginalRestoreOutputs,
            >()?,
            size_of::<Result<usize, Error>>(), // Checked helper census result.
        ]
        .into_iter()
        .try_fold(0, add)
    }

    fn restoration_controls() -> Result<usize, Error> {
        [
            Self::quote_controls()?,
            Self::owned_restoration_controls()?,
            size_of::<OriginalBridgeExport>(), // Consuming restore argument.
            size_of::<OriginalBridgeExport>(), // Move/field-extraction overlap.
            size_of::<Worker>(),               // Restored original return owner control.
            size_of::<Result<Worker, Error>>(),
            size_of::<super::Binding>(),
            size_of::<[usize; 1]>(),              // Original output IDs.
            size_of::<&OriginalRestoreOutputs>(), // Nonallocating owned-export reader.
            size_of::<Secret>(),                  // Exported-copy construction control.
            size_of::<Result<Secret, Error>>(),
            size_of::<Result<(u64, u64), Error>>(),
        ]
        .into_iter()
        .try_fold(0, add)
    }

    /// TOTALS, not increments. Keep every original phase, including copies,
    /// under the unchanged cap. ADD caller-owned outside owners; no sharing or
    /// existing-slack credit. Original restoration already quotes its exported
    /// payload allocation; the additions here are new explicit control owners.
    fn new(phases: &super::Phases, retained_baseline: usize) -> Result<Self, Error> {
        let phase = phases
            .construction
            .max(phases.copies)
            .max(phases.execution)
            .max(phases.restoration);
        let original_bytes = add(retained_baseline, add(phases.baseline, phase)?)?;
        let finish_bytes = add(original_bytes, Self::finish_controls()?)?;
        let retained_bytes = add(original_bytes, Self::retained_controls()?)?;
        let restoration_bytes = add(original_bytes, Self::restoration_controls()?)?;
        Ok(Self {
            original_bytes,
            finish_bytes,
            retained_bytes,
            restoration_bytes,
            admitted_bytes: finish_bytes.max(retained_bytes).max(restoration_bytes),
        })
    }

    /// The same complete lifetime quote guards BOTH transitions. Refuse before
    /// executor extraction or restoration allocation, not after owning new state.
    fn preflight(&self, limit: usize) -> Result<(), Error> {
        admit(self.admitted_bytes, limit)
    }
}

/// Genuine original completion custody. The sole constructor consumes the
/// coordinator's closed ORIGINAL invocation and its original executor. Neither
/// arbitrary Values nor matching Plan/Program digests can construct this type.
/// Abandonment retains the original suspended Worker cancellation behavior.
pub(in crate::replicated_decoder) struct OriginalBridgeExport {
    // Declaration order is destruction order: erase Values while BOTH original
    // runtime and Compiled/source/Worker owners remain alive. No custom Drop;
    // restore may still move its original worker/runtime fields unchanged.
    values: Values,
    runtime: PartyRuntime,
    invocation: Invocation,
    quote: OriginalExportQuote,
}

impl OriginalBridgeExport {
    pub(in crate::replicated_decoder) fn finish(
        drained: CoordinatorDrained<Invocation>,
    ) -> Result<Self, Error> {
        let quote = OriginalExportQuote::new(&drained.owner.compiled.phases, 0)?;
        quote.preflight(MAX_PROGRAM_BYTES)?;
        let mut invocation = drained.owner;
        let (runtime, values) = invocation.executor.take().ok_or(Error::Retired)?.finish()?;
        if values.output_len(0)? != invocation.compiled.binding.lanes {
            return Err(Error::Shape);
        }
        Ok(Self {
            values,
            runtime,
            invocation,
            quote,
        })
    }

    fn producer(&self) -> &Arc<Plan> {
        &self.invocation.compiled.plan
    }

    fn prepared(&self, start: usize) -> Result<&Arc<tensor::Program>, Error> {
        let compiled = &self.invocation.compiled;
        if start >= compiled.binding.lanes {
            return Err(Error::Shape);
        }
        let Code::TensorLoop { program, tail, .. } =
            &compiled.program.nodes[compiled.binding.output].code
        else {
            return Err(Error::Binding);
        };
        let remainder = compiled.binding.lanes % program.lanes();
        if remainder != 0 && start >= compiled.binding.lanes - remainder {
            tail.as_ref().ok_or(Error::Binding)
        } else {
            Ok(program)
        }
    }

    /// Restore the SAME original worker/runtime through the original export
    /// mapping. This does not release copied destination inputs or admit a new
    /// consumer. A failed restoration consumes and cancels the original owner.
    pub(in crate::replicated_decoder) fn restore(self) -> Result<Worker, Error> {
        self.quote.preflight(MAX_PROGRAM_BYTES)?;
        let binding = self.invocation.compiled.binding;
        // Only infallible field moves before entering the ordered guard. Do not
        // lend &self.values into a callee that already owns worker/runtime: its
        // allocation error/unwind would destroy those owners before this Values.
        let Self {
            values,
            runtime,
            invocation,
            quote: _,
        } = self;
        let Invocation {
            compiled,
            worker,
            executor,
        } = invocation;
        let outputs = OriginalRestoreOutputs {
            values,
            _executor: executor,
            _compiled: compiled,
        };
        worker.restore_owned(
            runtime,
            binding.end,
            &[binding.output],
            outputs,
            |outputs, output, lane| outputs.values.word(output, lane),
        )
    }
}

/// Immutable source/destination recipe plus single-use progress. Private shape
/// object, NOT proof authority. Production copies read only through Values.word;
/// tests exercise the identical private loop without inventing bridge custody.
struct RowCopyRecipe {
    export: usize,
    start: usize,
    source: RowKey,
    input: RowKey,
    attempted: bool,
}

impl RowCopyRecipe {
    fn copy_with(
        &mut self,
        output: &mut Secret,
        mut read: impl FnMut(usize, usize) -> Result<(u64, u64), Error>,
    ) -> Result<(), Error> {
        if self.attempted {
            return Err(Error::Retired);
        }
        // Every attempt burns this mapping, including shape/partial-copy errors.
        self.attempted = true;
        if output.first.len() != self.input.width
            || output.second.len() != self.input.width
            || self.input.width != self.source.width
        {
            return Err(Error::Shape);
        }
        for index in 0..self.source.width {
            let lane = self.start.checked_add(index).ok_or(Error::Bounds)?;
            let (first, second) = read(self.export, lane)?;
            output.first[index] = first;
            output.second[index] = second;
        }
        Ok(())
    }
}

/// Checked recipe, still not consumer authority. The exclusive borrow captures
/// the ACTUAL destination Builder, not an address stored past a move or an Arc
/// minted from labels. No mutable Builder access or detached proof is exposed.
struct CheckedRowCopy<'a> {
    destination: &'a mut Builder,
    producer: Arc<Plan>,
    prepared: Arc<tensor::Program>,
    source_row: Layout,
    input_row: Layout,
    recipe: RowCopyRecipe,
    baseline: usize,
}

impl<'a> CheckedRowCopy<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        destination: &'a mut Builder,
        producer: &Arc<Plan>,
        original_prepared: &Arc<tensor::Program>,
        candidate_prepared: &Arc<tensor::Program>,
        export: usize,
        start: usize,
        source_row: &Layout,
        input_row: &Layout,
        retained_baseline: usize,
        temporary: usize,
    ) -> Result<Self, Error> {
        let native = exported_proof(producer, export, start, source_row)?;
        producer.check_original_prepared_row_owner(
            source_row,
            original_prepared,
            candidate_prepared,
        )?;
        let input = input_row
            .row_key(native.key().width)
            .ok_or(Error::Binding)?;
        let payload = native.key().width.checked_mul(16).ok_or(Error::Bounds)?;
        let baseline = [
            2 * std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<Secret>(),
            payload,
            128, // Both Secret component allocator/header allowances.
            original_prepared.resources()?.immutable_template_bytes,
            2 * size_of::<usize>(),
        ]
        .into_iter()
        .try_fold(retained_baseline, add)?;
        destination.complete_row_import_preflight(
            producer, producer, export, start, source_row, input_row, baseline, temporary,
        )?;
        Ok(Self {
            destination,
            producer: retain(producer),
            prepared: Arc::clone(original_prepared),
            source_row: source_row.clone(),
            input_row: input_row.clone(),
            recipe: RowCopyRecipe {
                export,
                start,
                source: native.key(),
                input,
                attempted: false,
            },
            baseline,
        })
    }

    fn check_destination(&self, candidate: &Builder) -> Result<(), Error> {
        if std::ptr::eq::<Builder>(&*self.destination, candidate) {
            Ok(())
        } else {
            Err(Error::Binding)
        }
    }

    fn check_owner(
        &self,
        producer: &Arc<Plan>,
        prepared: &Arc<tensor::Program>,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.producer, producer) || !Arc::ptr_eq(&self.prepared, prepared) {
            return Err(Error::Binding);
        }
        Ok(())
    }

    fn copy_values(&mut self, original: &OriginalBridgeExport) -> Result<Secret, Error> {
        if self.recipe.attempted {
            return Err(Error::Retired);
        }
        self.check_owner(original.producer(), original.prepared(self.recipe.start)?)?;
        let mut output = Secret::new(self.recipe.input.width)?;
        self.recipe.copy_with(&mut output, |export, lane| {
            original.values.word(export, lane)
        })?;
        Ok(output)
    }
}

/// Closed executable export-to-input COPY custody. Construction consumes actual
/// original bridge completion; the retained immutable recipe performs the value
/// copy itself. Caller ConsumerObligation/projection/source/attention metadata
/// cannot construct it. No Clone or copied-input extraction exists.
///
/// This is not yet a closed EXECUTABLE DESTINATION. Until the main compact
/// emitter retains this exact Builder + copied input through finish/Executor
/// construction, bind_import must remain closed rather than release a proof.
pub(in crate::replicated_decoder) struct ExecutableRowCopy<'a> {
    // Pending copied shares erase before the producer/export custody is dropped.
    copied: Secret,
    checked: CheckedRowCopy<'a>,
    original: OriginalBridgeExport,
}

impl<'a> ExecutableRowCopy<'a> {
    /// Preserve the EXACT export lifetime quote, including its new increments.
    /// Do not reconstruct only original Phases and silently lose wrapper costs.
    fn retained_baseline(
        export: &OriginalExportQuote,
        retained_baseline: usize,
    ) -> Result<usize, Error> {
        [
            export.admitted_bytes,
            size_of::<Self>(),
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<OriginalBridgeExport>(), // This transition's incoming owner.
        ]
        .into_iter()
        .try_fold(retained_baseline, add)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::replicated_decoder) fn from_original_bridge(
        original: OriginalBridgeExport,
        destination: &'a mut Builder,
        start: usize,
        source_row: &Layout,
        input_row: &Layout,
        retained_baseline: usize,
        temporary: usize,
    ) -> Result<Self, Error> {
        original.quote.preflight(MAX_PROGRAM_BYTES)?;
        let baseline = Self::retained_baseline(&original.quote, retained_baseline)?;
        let prepared = original.prepared(start)?;
        let mut checked = CheckedRowCopy::new(
            destination,
            original.producer(),
            prepared,
            prepared,
            0,
            start,
            source_row,
            input_row,
            baseline,
            temporary,
        )?;
        let copied = checked.copy_values(&original)?;
        Ok(Self {
            copied,
            checked,
            original,
        })
    }

    /// Required consuming main-emitter seam. Do not open this by converting
    /// coordinates into AuthenticatedRowEdge. The final compact consumer must
    /// be constructed from checked.destination AND consume self.copied through
    /// this recipe's exact input slot, retaining original bridge custody and
    /// unchanged validity/source/lifecycle routing. No such destination exists.
    pub(in crate::replicated_decoder) fn bind_import(self, temporary: usize) -> Result<(), Error> {
        self.checked.check_destination(&*self.checked.destination)?;
        self.checked.check_owner(
            self.original.producer(),
            self.original.prepared(self.checked.recipe.start)?,
        )?;
        if !self.checked.recipe.attempted
            || self.copied.first.len() != self.checked.recipe.input.width
            || self.copied.second.len() != self.checked.recipe.input.width
        {
            return Err(Error::Binding);
        }
        self.checked.destination.complete_row_import_preflight(
            &self.checked.producer,
            &self.checked.producer,
            self.checked.recipe.export,
            self.checked.recipe.start,
            &self.checked.source_row,
            &self.checked.input_row,
            self.checked.baseline,
            temporary,
        )?;
        Err(Error::Binding)
    }
}

#[cfg(test)]
#[path = "compact_row_proof_tests.rs"]
mod tests;
