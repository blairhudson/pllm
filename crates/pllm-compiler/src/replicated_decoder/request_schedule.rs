//! Production finalization of an actually emitted original request schedule.
//! No ExpandedCompiled/oracle/semantic-label constructor exists here. This first
//! join uses original expanded storage, NOT compact-memory qualification.
use super::super::schedule_finalization;
use super::*;
use std::io::Write;

/// Owned, non-borrowed creation custody follows the SAME Program Arc through
/// ordinary Worker, cancellation and consuming BODY handoff. Native nodes retire
/// before this final field releases the original source/parameter guards.
pub(in crate::replicated_decoder) struct Custody {
    pub(super) projections: Vec<Option<source_bias::Creation>>,
    pub(super) epochs: Vec<Arc<OriginalSourcePlan>>,
    pub(super) fragments: Vec<Fragment>,
    _tokens: View,
    _scratch: Words,
    pub(super) origin: HistoryOrigin,
    _history_charge: OwnedExternalCharge,
    named_retained_bytes: usize,
    // Release finalization's real same-counter guard LAST.
    schedule_charge: OwnedExternalCharge,
}

impl Custody {
    pub(super) fn prepay_bias_metadata(&self, visits: usize) -> Result<(), Error> {
        self.origin
            .payload()
            .source
            .prepay_compiler_metadata(
                &self.schedule_charge,
                u64::try_from(visits).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)
    }
    pub(in crate::replicated_decoder) fn public(&self) -> &Arc<PublicSource> {
        &self.origin.payload().public
    }
    pub(in crate::replicated_decoder) fn retained_bytes(&self) -> usize {
        self.named_retained_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Resources {
    pub phase_payload_bytes: usize,
    pub outside_counter_bytes: usize,
    pub peak_bytes: usize,
    pub metadata_visits: u64,
    pub schedule: schedule_finalization::FinalizationQuote,
}

fn native_bytes(program: &Arc<tensor::Program>) -> Result<usize, Error> {
    add(
        program.resources()?.immutable_template_bytes,
        bytes::<usize>(2)?,
    )
}

/// Actual retained capacity, counting repeated Arc backing conservatively rather
/// than discounting equal digests/pointers. Upstream emission keeps its original
/// admission; this observation does NOT retroactively admit its construction.
fn schedule_bytes(builder: &Builder) -> Result<usize, Error> {
    let mut total = add(
        size_of::<Builder>(),
        bytes::<Node>(builder.nodes.capacity())?,
    )?;
    // Keep original cumulative nonvalue/value/certificate ledger independently.
    total = add(total, builder.allocated)?;
    for node in &builder.nodes {
        let backing = match &node.code {
            Code::Input(key) | Code::PublicInput { key, .. } => key.capacity(),
            Code::PublicValues(values) => bytes::<i64>(values.capacity())?,
            Code::Gather(words) => words.metadata_bytes(),
            Code::SourceGather(_) | Code::Affine { .. } | Code::PublicProduct { .. } => 0,
            Code::Product { norm, .. } => {
                if norm.is_some() {
                    add(
                        size_of::<puma::multiply::norm::RowEnergyProof>(),
                        bytes::<usize>(2)?,
                    )?
                } else {
                    0
                }
            }
            Code::Tensor { program, .. } | Code::CheckedRange { program, .. } => {
                native_bytes(program)?
            }
            Code::TensorLoop { program, tail, .. } => add(
                native_bytes(program)?,
                tail.as_ref().map_or(Ok(0), native_bytes)?,
            )?,
            Code::PublicLookup { plan, .. } => add(plan.metadata_bytes()?, bytes::<usize>(2)?)?,
            Code::TrieLookup { plan, .. } => add(plan.metadata_bytes(), bytes::<usize>(2)?)?,
            Code::Selection { plan, .. } => add(plan.metadata_bytes()?, bytes::<usize>(2)?)?,
        };
        total = add(total, backing)?;
    }
    Ok(total)
}

fn check(builder: &Builder, complete: &LogicalComplete<'_>) -> Result<(), Error> {
    let history = &complete.pending.history;
    check_origin_state(history, builder, COMPLETE)?;
    history.check_tokens(&complete.pending.tokens)?;
    if builder.nodes.len() != complete.nodes
        || builder.request_emission.is_some()
        || complete.outputs.len() != history.resources.outputs + 1
        || builder.validity != history.sticky
    {
        return Err(Error::Binding);
    }
    for (owner, record) in &complete.accepted {
        if !record.matches(builder.nodes.get(*owner).ok_or(Error::Binding)?) {
            return Err(Error::Binding);
        }
    }
    let masked = complete.accepted[0].0;
    let valid = complete.accepted[1].0;
    if complete.outputs[..history.resources.outputs]
        .iter()
        .copied()
        .ne((0..history.resources.outputs).map(|lane| (masked, lane)))
        || complete.outputs.last() != Some(&(valid, 0))
    {
        return Err(Error::Binding);
    }
    Ok(())
}

/// Supply a zero-payload same-counter guard before quoting. Growing that guard
/// does not change the recomputed outside baseline: only its OWN checked payload
/// is excluded, not any ancestor/source/sibling reservation.
pub(super) fn quote(
    builder: &Builder,
    complete: &LogicalComplete<'_>,
    guard: &OwnedExternalCharge,
) -> Result<Resources, Error> {
    check(builder, complete)?;
    let history = &complete.pending.history;
    let budget = history.origin.payload().source.budget();
    if !guard.is_for(budget) {
        return Err(Error::Binding);
    }
    let outside_counter_bytes = budget
        .used_bytes()
        .checked_sub(guard.payload_bytes())
        .ok_or(Error::Bounds)?;
    let sources = history.sources.registry;
    let borrowed = add(
        source_registry::source_metadata_bytes(sources.source_records(), sources.source_norms())?,
        sources.owned_backing_bytes(),
    )?;
    let mut baseline = add(outside_counter_bytes, schedule_bytes(builder)?)?;
    baseline = add(baseline, borrowed)?;
    baseline = add(baseline, history.sources.public.resident_metadata_bytes()?)?;
    baseline = add(
        baseline,
        bytes::<(usize, usize)>(complete.outputs.capacity())?,
    )?;
    for inline in [
        size_of::<LogicalComplete<'_>>(),
        size_of::<Custody>(),
        size_of::<Program>(),
        bytes::<usize>(2)?,
        size_of::<Sha256>(),
        size_of::<HashWriter<'_>>(),
        size_of::<Attempt>(),
        size_of::<OwnedExternalCharge>(),
        size_of::<Resources>(),
        size_of::<schedule_finalization::FinalizedSchedule>(),
        size_of::<Result<Arc<Program>, Error>>(),
        size_of::<Box<Custody>>(),
        original_source::OriginalSourcePlan::compiler_metadata_control_bytes()
            .map_err(|_| Error::Bounds)?,
    ] {
        baseline = add(baseline, inline)?;
    }
    let schedule =
        schedule_finalization::quote(builder.nodes.as_slice(), &complete.outputs, baseline)?;
    let phase_payload_bytes = schedule
        .peak_bytes
        .checked_sub(outside_counter_bytes)
        .ok_or(Error::Bounds)?;
    let mut visits = add(
        builder.nodes.len(),
        complete
            .outputs
            .len()
            .checked_mul(builder.nodes.len())
            .ok_or(Error::Bounds)?,
    )?;
    for node in &builder.nodes {
        visits = add(
            visits,
            match &node.code {
                Code::Gather(words) => words.len(),
                Code::PublicValues(values) => values.len(),
                Code::Input(key) | Code::PublicInput { key, .. } => key.len(),
                _ => 1,
            },
        )?;
    }
    Ok(Resources {
        phase_payload_bytes,
        outside_counter_bytes,
        peak_bytes: schedule.peak_bytes,
        metadata_visits: u64::try_from(visits).map_err(|_| Error::Bounds)?,
        schedule,
    })
}

struct HashWriter<'a>(&'a mut Sha256);
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Same original canonical prefix; streaming JSON avoids a hidden serialized
/// configuration/source copy. No manifest.program/digest is creation authority.
fn prefix(complete: &LogicalComplete<'_>) -> Result<Sha256, Error> {
    let history = &complete.pending.history;
    let manifest = history.sources.manifest;
    let mut hash = Sha256::new();
    hash.update(b"pllm.replicated-decoder.source-q18.full-prefix-final-ids.v1");
    hash.update(b"public-source-paged-ring64.v1");
    if !manifest.execution.is_default() {
        hash.update(b"explicit-public-execution-policy.v1");
        serde_json::to_writer(HashWriter(&mut hash), &manifest.execution)
            .map_err(|_| Error::Binding)?;
    }
    hash.update(history.origin.payload().source.raw_configuration());
    hash.update((history.resources.rows as u64).to_le_bytes());
    hash.update((history.resources.outputs as u64).to_le_bytes());
    serde_json::to_writer(HashWriter(&mut hash), &manifest.tensors).map_err(|_| Error::Binding)?;
    if !manifest.norms.is_empty() {
        hash.update(b"public-head-row-norms.v1");
        hash.update(source_norms::digest(&manifest.norms));
    }
    Ok(hash)
}

struct Attempt {
    origin: Arc<OwnedCharged<Origin>>,
    committed: bool,
}
impl Drop for Attempt {
    fn drop(&mut self) {
        if !self.committed {
            self.origin.payload().state.store(BURNED, Ordering::Release);
        }
    }
}

/// Opt-in production entry after actual source imports + token declaration.
/// Uses the ONE original request driver and epilogue; no previously compiled
/// Program, Manifest identity, or oracle qualification is accepted as input.
#[allow(clippy::too_many_arguments)]
pub(in crate::replicated_decoder::program) fn compile(
    builder: Builder,
    permit: RequestInputPermit<'_>,
    inputs: &BTreeMap<String, SourceView>,
    controller: OwnedExternalCharge,
    limits: original_source::Limits,
    mut schedule_guard: OwnedExternalCharge,
    cap: usize,
) -> Result<Arc<Program>, Error> {
    let mut attempt = Attempt {
        origin: Arc::clone(&permit.history.origin),
        committed: false,
    };
    // Move the parameter into a later local: on error/unwind the native owners
    // in Builder drop BEFORE Attempt releases its original source/parameter guard.
    let mut builder = builder;
    if !schedule_guard.is_for(permit.history.origin.payload().source.budget()) {
        return Err(Error::Binding);
    }
    let registry = permit.history.sources.registry;
    let sources = registry.source_records();
    let pending = request_driver::drive_owned(
        permit,
        &mut builder,
        inputs,
        sources,
        controller,
        limits,
        &mut AdmissionSite::default(),
    )?;
    let complete = builder.final_acceptance_owned(pending)?;
    let resources = quote(&builder, &complete, &schedule_guard)?;
    resources.schedule.admit(cap)?;
    // Growth atomically reserves positive delta before finalization creates its
    // retirement/hash/Program backing. Original outside guards are not refunded.
    if schedule_guard.payload_bytes() < resources.phase_payload_bytes {
        schedule_guard
            .resize_payload(resources.phase_payload_bytes)
            .map_err(|_| Error::Bounds)?;
    }
    let program = finish(builder, complete, schedule_guard, cap)?;
    attempt.committed = true;
    Ok(program)
}

/// Consumes actual construction authority, not an arbitrary nodes+identity tuple.
/// Only original expanded storage is installed; shared borrowed finalization does
/// the hash/work/retirement construction before ordinary Program/Worker reuse.
pub(super) fn finish(
    builder: Builder,
    complete: LogicalComplete<'_>,
    guard: OwnedExternalCharge,
    cap: usize,
) -> Result<Arc<Program>, Error> {
    let mut attempt = Attempt {
        origin: Arc::clone(&complete.pending.history.origin),
        committed: false,
    };
    // Same ordering for direct finalization failures, including a short quote.
    let mut builder = builder;
    let resources = quote(&builder, &complete, &guard)?;
    if guard.payload_bytes() < resources.phase_payload_bytes {
        return Err(Error::Bounds);
    }
    resources.schedule.admit(cap)?;
    complete
        .pending
        .history
        .origin
        .payload()
        .source
        .prepay_compiler_metadata(&guard, resources.metadata_visits)
        .map_err(|_| Error::Bounds)?;
    let prefix = prefix(&complete)?;
    let finalized = schedule_finalization::finalize(
        builder.nodes.as_slice(),
        &complete.outputs,
        &prefix,
        resources.schedule.retained_baseline_bytes,
        cap,
    )?;
    let binding = complete.pending.history.sources.registry.binding();
    let inputs = complete
        .pending
        .history
        .origin
        .payload()
        .input
        .owner()
        .checked_add(1)
        .ok_or(Error::Bounds)?;
    // Every declaration is still the actual original leading PublicInput/Input.
    for (ordinal, node) in builder.nodes.iter().enumerate() {
        if (ordinal < inputs) != matches!(node.code, Code::Input(_) | Code::PublicInput { .. }) {
            return Err(Error::Binding);
        }
    }
    let LogicalComplete {
        outputs, pending, ..
    } = complete;
    let PendingRequest {
        history, tokens, ..
    } = pending;
    let History {
        epochs,
        fragments,
        projections,
        scratch,
        origin,
        charge,
        ..
    } = history;
    let custody = Box::new(Custody {
        projections,
        epochs,
        fragments,
        _tokens: tokens,
        _scratch: scratch,
        origin,
        _history_charge: charge,
        named_retained_bytes: resources.peak_bytes,
        schedule_charge: guard,
    });
    let program = Arc::new(Program {
        nodes: std::mem::take(&mut builder.nodes),
        inputs,
        outputs,
        retire: finalized.retire,
        digest: finalized.digest,
        payload_bytes: builder.allocated,
        work: finalized.work,
        source_binding: binding,
        request_custody: Some(custody),
    });
    program
        .request_custody
        .as_ref()
        .ok_or(Error::Binding)?
        .public()
        .check(&program)?;
    attempt.committed = true;
    Ok(program)
}

impl Program {
    /// Ordinary Worker/context/PRSS constructors remain authoritative. The source
    /// is the exact retained authenticated owner, not a re-imported file set.
    pub(in crate::replicated_decoder) fn source_request_worker(
        self: &Arc<Self>,
        runtime: multiply::PartyRuntime,
        session: [u8; 32],
    ) -> Result<Worker, Error> {
        let owner = self.request_custody.as_ref().ok_or(Error::Binding)?;
        if owner.origin.payload().state.load(Ordering::Acquire) != COMPLETE {
            return Err(Error::Retired);
        }
        Worker::new_with_shared_source(
            Arc::clone(self),
            runtime,
            session,
            Arc::clone(owner.public()),
        )
    }
}
