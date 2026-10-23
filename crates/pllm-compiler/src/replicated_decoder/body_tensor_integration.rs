//! Staged, compiler-private bridge for ONE original Slice/Rescale18 TensorLoop.
//!
//! No default routing, publisher/SDK surface, new context or native construction.
//! Preparation consumes the original closed capability, never a digest-matched
//! replacement. Execution remains GATED pending original compiler-produced
//! Program/Worker full-transcript qualification, not a missing resource quote.
//! Distributed drain has no Rust-local proof: consuming coordinator capabilities
//! below deliberately have NO production issuer until the existing authenticated
//! three-role coordinator/API owns that transition. No caller-supplied bool.
//! This remains one original loop, NOT a complete compact request plan: compact
//! publication still needs its full executable/export/retirement mapping and
//! enclosing admission before this lifecycle can be generalized to that plan.
use super::prepared_tensor_map::{OneLoopScope, PreparedTensorMap};
use super::runtime::{body, BoundaryProgress};
use super::source_registry::{source_metadata_bytes, SourceRegistry};
use super::{
    puma, Code, Error, Interval, LoopInput, Node, PeerBytes, Program, PublicSource, Secret,
    WeightMode, Worker,
};
use body::handoff::{owned_restore_inline_bytes, SuspendedWorker, TRANSITION_INLINE_BYTES};
use body::layer::{self, Builder, Executor, Layout, Plan, Values};
use std::{mem::size_of, sync::Arc};

// Closed original-invocation export custody needs this module's private owners.
// Destination proof import remains fail-closed; no production drain issuer.
#[path = "compact_row_proof.rs"]
mod compact_row_proof;

// No new coordinator issuer. This consumes only the existing typed capability;
// its caller additionally checks genuine frozen Program/layout/native owners.
#[cfg(test)]
impl CoordinatorDrained<Worker> {
    pub(in crate::replicated_decoder) fn into_frozen_prefix_worker(self) -> Worker {
        self.owner
    }
    pub(in crate::replicated_decoder) fn frozen_prefix_worker_ref(&self) -> &Worker {
        &self.owner
    }
}

#[cfg(test)]
#[path = "frozen_prefix_handoff_tests.rs"]
mod frozen_prefix_handoff_tests;

#[derive(Debug)]
pub(super) enum PreparationError {
    Contract(Error),
    UnqualifiedOriginalWorkerParity,
}
impl From<Error> for PreparationError {
    fn from(error: Error) -> Self {
        Self::Contract(error)
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn array<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}
fn admit(bytes: usize) -> Result<(), Error> {
    if bytes > super::MAX_PROGRAM_BYTES {
        Err(Error::Bounds)
    } else {
        Ok(())
    }
}

// Values are owned by the handoff guard. Keep the original Compiled Arc in the
// caller until the guard returns/unwinds: its source dispatch, Plan and original
// binding must outlive output erasure, with no replacement Arc or runtime.
struct RestoreOutputs {
    values: Values,
    #[cfg(test)]
    _erased: restoration_tests::ErasureProbe,
}

/// NEW named restoration slots, additive to the old phase quote and baseline.
/// Never spend an old Worker/control allowance on these owning transitions.
fn owned_restoration_controls() -> Result<usize, Error> {
    [
        size_of::<RestoreOutputs>(), // Output-owner construction candidate.
        owned_restore_inline_bytes::<RestoreOutputs>()?,
        size_of::<Executor>(), // Extracted executor argument to finish.
        size_of::<Result<(super::multiply::PartyRuntime, Values), Error>>(),
        size_of::<usize>(),                // Original exclusive end.
        size_of::<[usize; 1]>(),           // Original logical output ID.
        size_of::<&[usize]>(),             // Borrowed output-ID slice passed to handoff.
        size_of::<&RestoreOutputs>(),      // Nonallocating reader borrow.
        size_of::<Result<usize, Error>>(), // Census result.
        size_of::<[usize; 2]>(),           // Census sum and unchanged admission limit.
    ]
    .into_iter()
    .try_fold(0, add)
}

fn restoration_preflight(bytes: usize, limit: usize) -> Result<(), Error> {
    if bytes > limit.min(super::MAX_PROGRAM_BYTES) {
        Err(Error::Bounds)
    } else {
        Ok(())
    }
}

#[cfg(test)]
// Adapter-local hooks use different Compiled/Plan types and isolated fault state.
#[allow(clippy::duplicate_mod)]
#[path = "body_restore_test_support.rs"]
pub(in crate::replicated_decoder) mod restoration_tests;

/// Immutable original IDs, never adapter-local IDs or user-selected output views.
#[derive(Clone, Copy)]
struct Binding {
    input: usize,
    output: usize,
    end: usize,
    lanes: usize,
    input_bounds: Interval,
}
impl Binding {
    fn validate(
        nodes: &[Node],
        imports: usize,
        weights: WeightMode,
        input_id: usize,
        output_id: usize,
        prepared: &PreparedTensorMap,
    ) -> Result<Self, Error> {
        if weights != WeightMode::Public
            || output_id < imports
            || input_id >= output_id
            || prepared.one_loop_scope() != Some(OneLoopScope::Rescale18)
        {
            return Err(Error::Binding);
        }
        prepared.validate()?;
        let source = nodes.get(input_id).ok_or(Error::Binding)?;
        let node = nodes.get(output_id).ok_or(Error::Binding)?;
        let Code::TensorLoop {
            input: LoopInput::Slice(original_input),
            program,
            tail,
        } = &node.code
        else {
            return Err(Error::Binding);
        };
        if *original_input != input_id
            || matches!(source.code, Code::PublicInput { .. })
            || source.lanes != node.lanes
            || !prepared.accepts(source.lanes, source.bounds)
            || node.bounds != prepared.output_bounds()?
            || node.work_budget()? != prepared.work_budget()?
            || !Arc::ptr_eq(program, prepared.program_at(0)?)
        {
            return Err(Error::Binding);
        }
        // Exact original full/tail owners, not digest equivalence. validate()
        // establishes nonzero bounded lanes, native tile width and output shape.
        let remainder = node.lanes % program.lanes();
        match (remainder, tail) {
            (0, None) => (),
            (0, Some(_)) | (_, None) => return Err(Error::Shape),
            (_, Some(tail)) => {
                if !Arc::ptr_eq(tail, prepared.program_at(node.lanes - remainder)?) {
                    return Err(Error::Binding);
                }
            }
        }
        Ok(Self {
            input: input_id,
            output: output_id,
            end: output_id.checked_add(1).ok_or(Error::Bounds)?,
            lanes: node.lanes,
            input_bounds: source.bounds,
        })
    }
}

/// No credit for shared template backing: preserve original admission and add
/// explicit graph/control slots, construction owners and capability baseline.
/// Source-free entry retains its rejection. Only the separately checked registry
/// route below may compose this census with original authenticated source owners.
fn retained_baseline(program: &Program, prepared: &PreparedTensorMap) -> Result<usize, Error> {
    if program
        .nodes
        .iter()
        .any(|node| matches!(node.code, Code::PublicInput { .. }))
    {
        return Err(Error::Binding);
    }
    retained_program_baseline(program, prepared)
}

fn retained_program_baseline(
    program: &Program,
    prepared: &PreparedTensorMap,
) -> Result<usize, Error> {
    let mut bytes = add(program.payload_bytes(), prepared.baseline_bytes()?)?;
    for owner in [
        size_of::<Program>(),
        array::<Node>(program.nodes.capacity())?,
        array::<Vec<usize>>(program.retire.capacity())?,
        array::<(usize, usize)>(program.outputs.capacity())?,
        array::<Option<Secret>>(program.nodes.len())?,
        size_of::<Worker>(),
        size_of::<Compiled>(),
        size_of::<Result<Arc<Compiled>, PreparationError>>(),
        size_of::<Invocation>(),
        size_of::<Result<Invocation, Error>>(),
        TRANSITION_INLINE_BYTES,
        // Explicit Arc control blocks for original Program, semantic Plan and
        // compiled mapping, without relying on a new unexplained fixed reserve.
        6 * size_of::<usize>(),
    ] {
        bytes = add(bytes, owner)?;
    }
    for retire in &program.retire {
        bytes = add(bytes, array::<usize>(retire.capacity())?)?;
    }
    admit(bytes)?;
    Ok(bytes)
}

/// Construction-only source authority is borrowed, never retained. The executor
/// and this dispatch retain the SAME already authenticated PublicSource Arc.
/// Original Worker's by-value source remains separate and independently priced.
struct SourceDispatch {
    public: Arc<PublicSource>,
    worker_metadata_bytes: usize,
    registry_metadata_bytes: usize,
    baseline_bytes: usize,
}
impl SourceDispatch {
    fn new(
        program: &Arc<Program>,
        worker: &Worker,
        public: Arc<PublicSource>,
        registry: &SourceRegistry<'_>,
        outside_baseline_bytes: usize,
    ) -> Result<Self, Error> {
        // Registry validation is not Manifest::compile or execution identity.
        // Program must still be the original admitted compiler-produced owner.
        public.check(program)?;
        public.check_registry(registry)?;
        let worker_metadata_bytes = SuspendedWorker::public_metadata_bytes(worker, program)?;
        let registry_metadata_bytes =
            source_metadata_bytes(registry.source_records(), registry.source_norms())?;
        let mut baseline_bytes = outside_baseline_bytes;
        for bytes in [
            public.resident_metadata_bytes()?,
            worker_metadata_bytes,
            registry_metadata_bytes,
            size_of::<SourceRegistry<'_>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<super::source_norms::ImportEnvelope>(),
            size_of::<Result<super::source_norms::ImportEnvelope, Error>>(),
            2 * size_of::<usize>(), // Original PublicSource Arc header, no alias credit.
        ] {
            baseline_bytes = add(baseline_bytes, bytes)?;
        }
        admit(baseline_bytes)?;
        Ok(Self {
            public,
            worker_metadata_bytes,
            registry_metadata_bytes,
            baseline_bytes,
        })
    }

    fn check_begin(
        &self,
        program: &Arc<Program>,
        worker: &Worker,
        registry: &SourceRegistry<'_>,
    ) -> Result<(), Error> {
        self.public.check(program)?;
        self.public.check_registry(registry)?;
        // Borrowed metadata can have a different allocation/capacity even with
        // equal records. Never spend an earlier quote on a larger live owner.
        if SuspendedWorker::public_metadata_bytes(worker, program)? > self.worker_metadata_bytes
            || source_metadata_bytes(registry.source_records(), registry.source_norms())?
                > self.registry_metadata_bytes
        {
            return Err(Error::Bounds);
        }
        Ok(())
    }

    fn executor(
        &self,
        plan: Arc<Plan>,
        runtime: super::multiply::PartyRuntime,
        inputs: Vec<Secret>,
        registry: &SourceRegistry<'_>,
    ) -> Result<Executor, Error> {
        // In particular, authenticate files even for this one-loop Plan (whose
        // own ops do not read a public matrix). No registry lifetime escapes.
        self.public.check_registry(registry)?;
        if source_metadata_bytes(registry.source_records(), registry.source_norms())?
            > self.registry_metadata_bytes
        {
            return Err(Error::Bounds);
        }
        Executor::new_with_registry(plan, runtime, inputs, Some(self.public.clone()), registry)
    }
}

#[cfg(test)]
thread_local! {
    static CONSTRUCTOR_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static FINISH_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn stage_slots() -> Result<usize, Error> {
    [
        size_of::<Result<Builder, Error>>(),
        size_of::<Result<(Builder, Layout), Error>>(),
        size_of::<Result<(Builder, Layout, usize), PreparationError>>(),
        size_of::<Result<(Arc<Plan>, usize), Error>>(),
        size_of::<[Layout; 1]>(),
    ]
    .into_iter()
    .try_fold(0, add)
}

/// Metadata-only staging. Each allocation reached here has an additive existing
/// owner gate. In particular the initial gate precedes Builder AND input Layout
/// allocation; prepared_tensor_map then applies its existing capacity-aware
/// publication gates with this baseline, consuming the original capability.
fn stage(
    binding: Binding,
    prepared: PreparedTensorMap,
    retained_baseline_bytes: usize,
) -> Result<(Builder, Layout), Error> {
    let inputs = [body::Register {
        lanes: binding.lanes,
        bounds: binding.input_bounds,
    }];
    let initial = Builder::input_container_quote(&inputs)?;
    let layout = Layout::quote_contiguous(0, binding.lanes)?;
    let baseline = add(retained_baseline_bytes, stage_slots()?)?;
    admit(add(
        add(baseline, initial)?,
        layout.construction_peak_bytes,
    )?)?;
    #[cfg(test)]
    CONSTRUCTOR_ENTRIES.with(|n| n.set(n.get() + 1));
    let mut builder = Builder::new(WeightMode::Public, &inputs)?;
    let input = builder.input(0)?;
    let output = builder.prepared_tensor_map(&input, prepared, baseline)?;
    Ok((builder, output))
}

/// Finalization-only incremental quote. Earlier staging peaks were independently
/// preadmitted, not inferred from this later quote. Preserve all stage result
/// slots conservatively; no template-sharing credit or post-allocation quote.
fn finish_stage(
    builder: Builder,
    output: Layout,
    baseline: usize,
    limit: usize,
) -> Result<(Arc<Plan>, usize), Error> {
    let outputs = [output];
    let construction = builder.proof_finish_quote(&outputs, stage_slots()?)?;
    if add(baseline, construction)? > limit.min(super::MAX_PROGRAM_BYTES) {
        return Err(Error::Bounds);
    }
    #[cfg(test)]
    FINISH_ENTRIES.with(|n| n.set(n.get() + 1));
    Ok((builder.finish(&outputs)?, construction))
}

/// Phase maxima, never max(original owners, native owners). Construction here
/// means the separately preadmitted FINISH phase, not earlier staging peaks.
struct Phases {
    baseline: usize,
    construction: usize,
    copies: usize,
    execution: usize,
    restoration: usize,
}
impl Phases {
    fn new(baseline: usize, construction: usize, lanes: usize, plan: &Plan) -> Result<Self, Error> {
        let payload = lanes.checked_mul(16).ok_or(Error::Bounds)?;
        let copies = add(payload, add(size_of::<Vec<Secret>>(), size_of::<Secret>())?)?;
        let execution = add(plan.resources().admitted_bytes, copies)?;
        let old_restoration = add(
            add(execution, payload)?,
            add(
                size_of::<Result<(super::multiply::PartyRuntime, Values), Error>>(),
                size_of::<&Values>(),
            )?,
        )?;
        let restoration = add(old_restoration, owned_restoration_controls()?)?;
        let phases = Self {
            baseline,
            construction,
            copies,
            execution,
            restoration,
        };
        phases.admit()?;
        Ok(phases)
    }
    fn admit(&self) -> Result<(), Error> {
        self.admit_with_limit(super::MAX_PROGRAM_BYTES)
    }
    fn required_bytes(&self) -> Result<usize, Error> {
        add(
            self.baseline,
            self.construction
                .max(self.copies)
                .max(self.execution)
                .max(self.restoration),
        )
    }
    fn admit_with_limit(&self, limit: usize) -> Result<(), Error> {
        restoration_preflight(self.required_bytes()?, limit)
    }
}

pub(super) struct Compiled {
    program: Arc<Program>,
    plan: Arc<Plan>,
    binding: Binding,
    phases: Phases,
    source: Option<SourceDispatch>,
}

/// One exact local owner frozen by the authenticated coordinator AFTER all three
/// roles reach the same boundary and the router drains every frozen wave/ACK.
/// Owning, not borrowing, prevents advance/receive/cancel between certification
/// and consumption; no detached permit can be replayed against another worker.
/// No Clone, public fields, constructor, Deref, or production issuer exists.
///
/// The parent Python/API seam must issue this only inside the existing
/// replicated_worker.run / drain_peer_waves coordinator, bound to the admitted
/// attempt and exact start/end boundary. Local Reached/complete, an empty local
/// queue, and a client-provided acknowledgement are NEVER issuance authority.
/// That API must distinguish bounded body readiness from request completion,
/// freeze each original native owner under its attempt lock, await all roles and
/// delivery ACKs, then consume custody once; failure retires the whole attempt.
/// Until that distributed integration exists, production cannot begin/restore
/// even if preparation's independent qualification gate is opened later.
#[repr(transparent)]
pub(super) struct CoordinatorDrained<T> {
    owner: T,
}

const _: () = assert!(size_of::<CoordinatorDrained<Worker>>() == size_of::<Worker>());
const _: () = assert!(size_of::<CoordinatorDrained<Invocation>>() == size_of::<Invocation>());

impl Compiled {
    pub(super) fn prepare(
        program: Arc<Program>,
        weights: WeightMode,
        input: usize,
        output: usize,
        prepared: PreparedTensorMap,
    ) -> Result<Arc<Self>, PreparationError> {
        Self::prepare_checked(
            program,
            weights,
            input,
            output,
            prepared,
            None,
            worker_parity_gate,
        )
    }

    // The only ungated entry is cfg(test), private to this module's tests. All
    // binding/admission/construction checks remain identical to production.
    #[cfg(test)]
    fn prepare_candidate(
        program: Arc<Program>,
        weights: WeightMode,
        input: usize,
        output: usize,
        prepared: PreparedTensorMap,
    ) -> Result<Arc<Self>, PreparationError> {
        Self::prepare_checked(program, weights, input, output, prepared, None, || Ok(()))
    }

    /// Opt-in bounded source route. Caller supplies the original already compiled
    /// Program and authenticated Arc, plus the existing worker owner for census.
    /// outside_baseline_bytes includes Manifest/config/validation backing and any
    /// other retained owners, excluding source/norm maps inventoried here. This
    /// does NOT replace Manifest::compile or admit a complete compact request.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_with_registry(
        program: Arc<Program>,
        weights: WeightMode,
        input: usize,
        output: usize,
        prepared: PreparedTensorMap,
        worker: &Worker,
        public: Arc<PublicSource>,
        registry: &SourceRegistry<'_>,
        outside_baseline_bytes: usize,
    ) -> Result<Arc<Self>, PreparationError> {
        let source =
            SourceDispatch::new(&program, worker, public, registry, outside_baseline_bytes)?;
        Self::prepare_checked(
            program,
            weights,
            input,
            output,
            prepared,
            Some(source),
            worker_parity_gate,
        )
    }

    fn prepare_checked(
        program: Arc<Program>,
        weights: WeightMode,
        input: usize,
        output: usize,
        prepared: PreparedTensorMap,
        source: Option<SourceDispatch>,
        qualification: impl FnOnce() -> Result<(), PreparationError>,
    ) -> Result<Arc<Self>, PreparationError> {
        let binding = Binding::validate(
            &program.nodes,
            program.inputs,
            weights,
            input,
            output,
            &prepared,
        )?;
        let baseline = match &source {
            Some(source) => add(
                retained_program_baseline(&program, &prepared)?,
                source.baseline_bytes,
            )?,
            None => retained_baseline(&program, &prepared)?,
        };
        // The full retained+source baseline precedes Builder/Layout allocation;
        // stage then adds its existing result slots and native publication peaks.
        admit(baseline)?;
        let (builder, output) = stage(binding, prepared, baseline)?;
        let (plan, construction) =
            finish_stage(builder, output, baseline, super::MAX_PROGRAM_BYTES)?;
        if plan.work_budget() != program.nodes[binding.output].work_budget()?
            || plan.input_count() != 1
            || plan.output_count() != 1
            || plan.output_layout(0)?.row_key(binding.lanes)
                != Some(layer::RowKey {
                    owner: 1,
                    offset: 0,
                    width: binding.lanes,
                })
        {
            return Err(Error::Binding.into());
        }
        let phases = Phases::new(baseline, construction, binding.lanes, &plan)?;
        qualification()?;
        Ok(Arc::new(Self {
            program,
            plan,
            binding,
            phases,
            source,
        }))
    }

    /// Caller drains ALL roles/router first; local Reached is not that proof.
    pub(super) fn advance_to_start(&self, worker: &mut Worker) -> Result<BoundaryProgress, Error> {
        SuspendedWorker::advance_to_start(&self.program, self.binding.output, worker)
    }
    /// Consume coordinator custody, then independently check local handoff.
    /// SuspendedWorker itself cannot prove the distributed drain.
    pub(super) fn begin(
        self: Arc<Self>,
        drained: CoordinatorDrained<Worker>,
    ) -> Result<Invocation, Error> {
        self.begin_checked(drained, None)
    }

    pub(super) fn begin_with_registry(
        self: Arc<Self>,
        drained: CoordinatorDrained<Worker>,
        registry: &SourceRegistry<'_>,
    ) -> Result<Invocation, Error> {
        self.begin_checked(drained, Some(registry))
    }

    fn begin_checked(
        self: Arc<Self>,
        drained: CoordinatorDrained<Worker>,
        registry: Option<&SourceRegistry<'_>>,
    ) -> Result<Invocation, Error> {
        let mut worker = drained.owner;
        if let Err(error) = self.phases.admit() {
            worker.cancel();
            return Err(error);
        }
        match (&self.source, registry) {
            (Some(source), Some(registry)) => {
                source.check_begin(&self.program, &worker, registry)?;
            }
            (None, None) => (),
            _ => return Err(Error::Binding),
        }
        let (mut worker, runtime, inputs) = SuspendedWorker::begin(
            worker,
            &self.program,
            self.binding.output,
            &[self.binding.input],
        )?;
        let executor = match (&self.source, registry) {
            (Some(source), Some(registry)) => {
                source.executor(self.plan.clone(), runtime, inputs, registry)
            }
            (None, None) => Executor::new(self.plan.clone(), runtime, inputs, None, None),
            _ => Err(Error::Binding),
        };
        match executor {
            Ok(executor) => Ok(Invocation {
                compiled: self,
                worker,
                executor: Some(executor),
            }),
            Err(error) => {
                worker.cancel();
                Err(error)
            }
        }
    }
}

/// Qualification gate, NOT an accounting dependency. Genuine original fixture
/// parity is exercised only through cfg(test) prepare_candidate. Production
/// remains closed until the parent audits those tests and resource composition.
fn worker_parity_gate() -> Result<(), PreparationError> {
    Err(PreparationError::UnqualifiedOriginalWorkerParity)
}

// On abandonment/rejected preflight BODY teardown precedes the original Worker;
// Compiled remains last. Completed Values use the existing ordered restore guard.
pub(super) struct Invocation {
    executor: Option<Executor>,
    worker: SuspendedWorker,
    compiled: Arc<Compiled>,
}
impl Invocation {
    pub(super) fn party(&self) -> puma::Party {
        self.worker.party()
    }
    pub(super) fn complete(&self) -> bool {
        self.executor.as_ref().is_some_and(Executor::complete)
    }
    pub(super) fn cancel(&mut self) {
        self.executor = None;
        self.worker.cancel();
    }
    pub(super) fn advance(&mut self) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        let result = self.executor.as_mut().ok_or(Error::Retired)?.advance();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub(super) fn receive(
        &mut self,
        raw: &[u8],
        sender: puma::Party,
    ) -> Result<Vec<(puma::Party, PeerBytes)>, Error> {
        let result = self
            .executor
            .as_mut()
            .ok_or(Error::Retired)?
            .receive(raw, sender);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    /// Only consuming coordinator custody may reach this local finish. Restore
    /// consumes the original runtime and copies both components through the
    /// admitted read-only export layout. Local completion is not drain authority.
    fn restore(mut self) -> Result<Worker, Error> {
        #[cfg(not(test))]
        let limit = super::MAX_PROGRAM_BYTES;
        #[cfg(test)]
        let limit = restoration_tests::limit();
        // Includes the new owned-output/guard transition, added by Phases::new.
        // Recheck the complete ceiling BEFORE executor extraction or finish.
        self.compiled.phases.admit_with_limit(limit)?;
        #[cfg(test)]
        restoration_tests::extracting();
        let (runtime, values) = self.executor.take().ok_or(Error::Retired)?.finish()?;
        if values.output_len(0)? != self.compiled.binding.lanes {
            return Err(Error::Shape);
        }
        let outputs = RestoreOutputs {
            values,
            #[cfg(test)]
            _erased: restoration_tests::ErasureProbe::take(),
        };
        self.worker.restore_owned(
            runtime,
            self.compiled.binding.end,
            &[self.compiled.binding.output],
            outputs,
            |outputs, output, lane| {
                #[cfg(test)]
                restoration_tests::reading()?;
                outputs.values.word(output, lane)
            },
        )
    }
}

impl CoordinatorDrained<Invocation> {
    /// Read-only additive inventory of THIS already-owned invocation. Not a
    /// producer association, copy receipt or new coordinator capability issuer.
    pub(in crate::replicated_decoder) fn prefix_custody_inventory(
        &self,
    ) -> Result<(usize, puma::WorkBudget), Error> {
        Ok((
            self.owner.compiled.phases.required_bytes()?,
            self.owner.compiled.plan.work_budget(),
        ))
    }

    pub(super) fn restore(self) -> Result<Worker, Error> {
        self.owner.restore()
    }
}

#[cfg(test)]
#[path = "body_tensor_integration_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "body_tensor_accounting_tests.rs"]
mod accounting_tests;
