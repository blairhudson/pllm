//! Consuming source-factory -> original request join. No serialized plan,
//! executable oracle, caller-selected coordinates or Python-issued permit.
use super::*;
use crate::original_source::{self, OriginalSourcePlan};
use crate::plan_input::json;
use crate::replicated_decoder::source_registry::{self, SourceConstructionOwners, SourceRegistry};
use pllm_garble::shared_allocation::{
    Budget, Error as NativeError, OwnedBudgetResources, OwnedExternalCharge,
};
use pllm_garble::shared_allocation::{OwnedBudget, OwnedCharged, MAX_ALLOCATION_BYTES};

#[path = "source_entry_compact_pending.rs"]
pub(in crate::replicated_decoder::program) mod compact_pending;
use serde::Deserialize;

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn bytes<T>(n: usize) -> Result<usize, Error> {
    n.checked_mul(size_of::<T>()).ok_or(Error::Bounds)
}

fn prefix_bounds_error<T>(_: T) -> Error {
    Error::Bounds
}

// These helpers are used only to evaluate the typed constants below. Overflow
// fails constant evaluation rather than silently weakening the source union.
const fn source_union_weight<T>(count: usize) -> usize {
    match size_of::<T>().checked_mul(count) {
        Some(value) => value,
        None => panic!("source union weight overflow"),
    }
}

const fn source_union_sum<const N: usize>(weights: [usize; N]) -> usize {
    let mut total = 0usize;
    let mut index = 0;
    while index < N {
        total = match total.checked_add(weights[index]) {
            Some(value) => value,
            None => panic!("source union sum overflow"),
        };
        index += 1;
    }
    total
}

// Corrected zero-padding recipe (24): conservative typed, non-test SOURCE
// union, not a physical peak, exact minimum or resource/work funding proof.
// Q remains unchanged; this additional E neither subtracts Q nor repeats Arc
// backing. Opaque U, CAS retries, cleanup and post-JSON custody remain open.
const KCTOR: usize = source_union_sum([
    source_union_weight::<usize>(13),
    source_union_weight::<bool>(4),
    source_union_weight::<NativeError>(2),
    source_union_weight::<std::sync::atomic::AtomicUsize>(2),
    source_union_weight::<Budget>(4),
    source_union_weight::<Result<Budget, NativeError>>(3),
    source_union_weight::<Result<OwnedBudget, NativeError>>(6),
    source_union_weight::<OwnedBudgetResources>(2),
    source_union_weight::<Result<OwnedBudgetResources, NativeError>>(1),
    source_union_weight::<&Budget>(4),
    source_union_weight::<Result<(), NativeError>>(4),
    source_union_weight::<()>(3),
    source_union_weight::<Arc<Budget>>(1),
    source_union_weight::<OwnedBudget>(1),
]);
const KUSED: usize = source_union_sum([
    source_union_weight::<&Budget>(1),
    source_union_weight::<&std::sync::atomic::AtomicUsize>(1),
    source_union_weight::<std::sync::atomic::Ordering>(1),
    source_union_weight::<usize>(2),
]);
// 61 source occurrences; Layout::extend returns (Layout, usize), in that order.
const KRESOURCES: usize = source_union_sum([
    source_union_weight::<usize>(26),
    source_union_weight::<std::alloc::Layout>(3),
    source_union_weight::<&std::alloc::Layout>(3),
    source_union_weight::<(std::alloc::Layout, usize)>(1),
    source_union_weight::<Result<(std::alloc::Layout, usize), std::alloc::LayoutError>>(1),
    source_union_weight::<Result<(std::alloc::Layout, usize), NativeError>>(1),
    source_union_weight::<std::alloc::LayoutError>(1),
    source_union_weight::<[usize; 7]>(1),
    source_union_weight::<std::array::IntoIter<usize, 7>>(1),
    source_union_weight::<&mut std::array::IntoIter<usize, 7>>(1),
    source_union_weight::<Option<usize>>(3),
    source_union_weight::<Result<usize, NativeError>>(3),
    source_union_weight::<NativeError>(9),
    source_union_weight::<OwnedBudgetResources>(1),
    source_union_weight::<Result<OwnedBudgetResources, NativeError>>(6),
]);
const KRESERVE_FIXED: usize = source_union_sum([
    source_union_weight::<usize>(19),
    source_union_weight::<&Budget>(1),
    source_union_weight::<&std::sync::atomic::AtomicUsize>(2),
    source_union_weight::<std::sync::atomic::Ordering>(3),
    source_union_weight::<Option<usize>>(5),
    source_union_weight::<bool>(1),
    source_union_weight::<Result<usize, usize>>(1),
    source_union_weight::<Result<usize, NativeError>>(1),
    source_union_weight::<NativeError>(3),
    source_union_weight::<Result<(), NativeError>>(2),
    source_union_weight::<()>(1),
]);
const KCHARGE: usize = source_union_sum([
    source_union_weight::<usize>(7),
    source_union_weight::<&OwnedBudget>(1),
    source_union_weight::<&Budget>(1),
    source_union_weight::<&Arc<Budget>>(1),
    source_union_weight::<Option<usize>>(1),
    source_union_weight::<Result<usize, NativeError>>(1),
    source_union_weight::<NativeError>(3),
    source_union_weight::<Result<(), NativeError>>(1),
    source_union_weight::<()>(1),
    source_union_weight::<Arc<Budget>>(1),
    source_union_weight::<OwnedExternalCharge>(1),
    source_union_weight::<Result<OwnedExternalCharge, NativeError>>(4),
]);
const KCALLER: usize = source_union_sum([
    source_union_weight::<usize>(56),
    source_union_weight::<i32>(1),
    source_union_weight::<bool>(3),
    source_union_weight::<&[u8]>(3),
    source_union_weight::<Option<(&[u8], &[u8])>>(1),
    source_union_weight::<&OwnedBudget>(2),
    source_union_weight::<OwnedBudget>(2),
    source_union_weight::<OwnedExternalCharge>(3),
    source_union_weight::<Option<OwnedExternalCharge>>(4),
    source_union_weight::<Result<OwnedBudget, NativeError>>(1),
    source_union_weight::<Result<OwnedBudget, Error>>(1),
    source_union_weight::<Result<OwnedExternalCharge, NativeError>>(2),
    source_union_weight::<Result<OwnedExternalCharge, Error>>(2),
    source_union_weight::<Result<SourceRequestImport, Error>>(5),
    source_union_weight::<Error>(10),
    source_union_weight::<NativeError>(3),
]);
const DELTA_CALLER: usize = source_union_sum([
    source_union_weight::<usize>(3),
    source_union_weight::<&OwnedBudget>(1),
    source_union_weight::<()>(1),
    source_union_weight::<Result<usize, NativeError>>(1),
    source_union_weight::<Result<usize, Error>>(1),
    source_union_weight::<Result<(), NativeError>>(1),
    source_union_weight::<Result<(), Error>>(1),
    source_union_weight::<Result<SourceRequestImport, Error>>(2),
    source_union_weight::<Error>(6),
    source_union_weight::<NativeError>(2),
]);
const KAPI: usize = source_union_sum([
    source_union_weight::<usize>(2),
    source_union_weight::<&OwnedBudget>(1),
    source_union_weight::<&Budget>(1),
    source_union_weight::<Result<(), NativeError>>(2),
]);
const HFIXED: usize = source_union_sum([
    source_union_weight::<usize>(31),
    source_union_weight::<(usize, usize)>(6),
    source_union_weight::<Option<usize>>(8),
    source_union_weight::<Result<usize, NativeError>>(10),
    source_union_weight::<NativeError>(4),
]);
const NC: usize = 16;
const NR: usize = 2;
const FIXED: usize = source_union_sum([
    KCTOR,
    KUSED,
    KRESOURCES,
    match KRESERVE_FIXED.checked_mul(4) {
        Some(value) => value,
        None => panic!("source union reserve multiplicity overflow"),
    },
    match KCHARGE.checked_mul(2) {
        Some(value) => value,
        None => panic!("source union charge multiplicity overflow"),
    },
    KCALLER,
    DELTA_CALLER,
    KAPI,
    HFIXED,
]);

fn source_union_quote() -> Result<usize, NativeError> {
    let widths = OwnedBudgetResources::reserve_update_layout();
    let closure_bytes = NC.checked_mul(widths.0);
    let closure_bytes = match closure_bytes {
        Some(value) => value,
        None => return Err(NativeError::Resource),
    };
    let reference_bytes = NR.checked_mul(widths.1);
    let reference_bytes = match reference_bytes {
        Some(value) => value,
        None => return Err(NativeError::Resource),
    };
    let subtotal = FIXED.checked_add(closure_bytes);
    let subtotal = match subtotal {
        Some(value) => value,
        None => return Err(NativeError::Resource),
    };
    let total = subtotal.checked_add(reference_bytes);
    let total = match total {
        Some(value) => value,
        None => return Err(NativeError::Resource),
    };
    Ok(total)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImportQuote {
    pub payload_bytes: usize,
    pub metadata_work: u64,
    pub declarations: usize,
}

fn check_source(source: &OriginalSourcePlan, manifest: &Manifest) -> Result<(), Error> {
    let workload = source.workload();
    if manifest.schema != "pllm.replicated-decoder.manifest.v1"
        || manifest.weights != WeightMode::Public
        || manifest.execution.checked_bound().is_none()
        || manifest.source == [0; 32]
        || manifest.composition == [0; 32]
        || manifest.session == [0; 32]
        || source.raw_configuration() != manifest.config
        || workload.batch != 1
        || workload.max_input_tokens != u64::try_from(manifest.rows).map_err(|_| Error::Bounds)?
        || workload.max_new_tokens != u64::try_from(manifest.outputs).map_err(|_| Error::Bounds)?
    {
        return Err(Error::Binding);
    }
    Ok(())
}

/// Input factories only; graph/native emission retains the original Builder's
/// admission contract. No complete CPU/RSS or compact-capacity claim.
pub(super) fn quote_imports(
    original: &OriginalSourcePlan,
    manifest: &Manifest,
    registry: &SourceRegistry<'_>,
    public: &Arc<PublicSource>,
    guard: &OwnedExternalCharge,
) -> Result<ImportQuote, Error> {
    if !guard.is_for(original.budget()) {
        return Err(Error::Binding);
    }
    check_source(original, manifest)?;
    let mut declarations = 0usize;
    let mut payload = add(size_of::<CreatedInputs<'_>>(), size_of::<Builder>())?;
    let mut work = 1usize;
    for (key, source) in &manifest.tensors {
        work = add(work, add(key.len(), source.shape.len())?)?;
        if source.public_q18.is_some() {
            continue;
        }
        declarations = add(declarations, 1)?;
        // Separate actual strings in Code::PublicInput and the source-view map.
        payload = add(
            payload,
            bytes::<u8>(key.capacity().checked_mul(2).ok_or(Error::Bounds)?)?,
        )?;
        payload = add(payload, bytes::<usize>(source.shape.capacity())?)?;
        payload = add(
            payload,
            source_registry::tree_entry_bytes::<(String, SourceView)>()?,
        )?;
    }
    // Reserve leading public declarations and the genuine token Input together.
    // request_origin::declare owns/prices the token record/Words/shape factory.
    payload = add(payload, bytes::<Node>(add(declarations, 1)?)?)?;
    payload = add(payload, size_of::<BTreeMap<String, SourceView>>())?;
    // Same conservative norm-copy envelope as original expanded import. Do not
    // discount shared/equal records. This covers actual norm map spare backing.
    let norm_bytes = source_norms::import_envelope(&manifest.norms)?.total_bytes;
    payload = add(payload, norm_bytes)?;
    work = add(work, norm_bytes)?;
    for n in [
        size_of::<ImportQuote>(),
        size_of::<Result<ImportQuote, Error>>(),
        size_of::<request_origin::RequestSources<'_>>(),
        size_of::<Result<CreatedInputs<'_>, Error>>(),
        size_of::<SourceView>(),
        size_of::<Code>(),
        size_of::<Node>(),
        OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
    ] {
        payload = add(payload, n)?;
    }
    // Empty policy-identical Builder overquotes token-node growth: the real
    // import pool reserves its token slot already. No fake declarations/views.
    let seed = empty_builder(manifest, registry);
    let sources = request_origin::RequestSources::new(manifest, registry, Arc::clone(public))?;
    let request = request_origin::quote(&seed, original, &sources)?;
    payload = add(
        payload,
        add(request.payload_bytes, size_of::<OwnedExternalCharge>())?,
    )?;
    let metadata_work = u64::try_from(work)
        .map_err(|_| Error::Bounds)?
        .checked_add(request.metadata_work.checked_mul(2).ok_or(Error::Bounds)?)
        .ok_or(Error::Bounds)?;
    Ok(ImportQuote {
        payload_bytes: payload,
        metadata_work,
        declarations,
    })
}

fn empty_builder(manifest: &Manifest, registry: &SourceRegistry<'_>) -> Builder {
    Builder {
        public_weights: true,
        checked_bound: manifest.execution.checked_bound(),
        lookup: manifest.execution.lookup,
        selection: manifest.execution.selection,
        token_source_binding: if manifest.execution.lookup == LookupPolicy::Trie {
            registry.binding()
        } else {
            [0; 32]
        },
        ..Builder::default()
    }
}

/// This aggregate is never returned to Python. Drop order retires the emitted
/// input/map/permit owners before their original continuous factory reservation.
pub(super) struct CreatedInputs<'s> {
    pub(super) builder: Builder,
    pub(super) inputs: BTreeMap<String, SourceView>,
    pub(super) permit: request_origin::RequestInputPermit<'s>,
    pub(super) charge: OwnedExternalCharge,
    // Only the prequoted actual input-emission path can mint this aggregate.
    created: (),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReceiverState {
    Empty,
    Entering,
    CleanupOnly,
    Complete,
    Drained,
}

struct ReceiverAttempt<'a>(&'a mut ReceiverState);
impl Drop for ReceiverAttempt<'_> {
    fn drop(&mut self) {
        if *self.0 == ReceiverState::Entering {
            *self.0 = ReceiverState::CleanupOnly;
        }
    }
}

#[derive(Default)]
struct PendingMap {
    key: Option<String>,
    shape: Vec<usize>,
    owner: Option<usize>,
    norm: Option<BlockNorms>,
}

/// Inactive input/declaration/context prerequisite, NOT importer/seal custody.
/// New resident slots, move overlap and control work remain UNFUNDED/OPEN.
/// Field order destroys payloads before their actual covering reservations.
struct InputReceiver<'loan> {
    builder: Option<Builder>,
    inputs: Option<BTreeMap<String, SourceView>>,
    pending_code: Option<Code>,
    pending_map: PendingMap,
    declaration: request_origin::DeclarationReceiver,
    setup: Option<request_origin::compact_storage::ResidentSetup>,
    binding: Option<request_origin::PreparedSourceBinding>,
    original: Arc<OriginalSourcePlan>,
    public: Arc<PublicSource>,
    state: ReceiverState,
    imports: Option<OwnedExternalCharge>,
    controller: Option<std::cell::RefCell<request_origin::compact_storage::ControllerLoan<'loan>>>,
}

// Source-logical inline component; not a complete frontier/admission quote.
fn input_receiver_bytes() -> Result<usize, Error> {
    bytes::<InputReceiver<'_>>(1)
}

impl Drop for InputReceiver<'_> {
    fn drop(&mut self) {
        drop(self.builder.take());
        drop(self.inputs.take());
        drop(self.pending_code.take());
        drop(std::mem::take(&mut self.pending_map));
        self.declaration.retire_payloads();
        drop(self.setup.take());
        #[cfg(test)]
        RECEIVER_DROP.with(|record| record.set(Some((
            self.builder.is_none() && self.inputs.is_none()
                && self.pending_map.shape.capacity() == 0 && self.pending_code.is_none(),
            self.original.budget().used_bytes(),
            self.imports.as_ref().map(|c| (c.charged_bytes(), c.budget() as *const _ as usize)).unwrap_or((0, 0)),
            self.controller.as_ref().map(|c| {
                use request_origin::compact_storage::ControllerAccounting;
                let loan = c.borrow();
                (loan.charged_bytes(), if loan.is_for(self.original.budget()) {
                    self.original.budget() as *const _ as usize
                } else { 0 })
            }).unwrap_or((0, 0)),
        ))));
        // Import coverage follows dependent fields. The lexical loan cannot
        // release its original controller, retained by the enclosing owner.
    }
}

impl<'loan> InputReceiver<'loan> {
    /// No validation/fallible factory before genuine owners become resident.
    /// Arc wrapping allocates; it is not an allocation-free custody claim.
    fn new(
        original: OriginalSourcePlan,
        public: Arc<PublicSource>,
        imports: OwnedExternalCharge,
        controller: request_origin::compact_storage::ControllerLoan<'loan>,
        cap: usize,
    ) -> Self {
        Self {
            builder: Some(Builder::default()), inputs: Some(BTreeMap::new()),
            pending_code: None, pending_map: PendingMap::default(),
            declaration: request_origin::DeclarationReceiver::empty(),
            setup: Some(request_origin::compact_storage::ResidentSetup::empty(cap)), binding: None,
            original: Arc::new(original), public, state: ReceiverState::Empty,
            imports: Some(imports), controller: Some(std::cell::RefCell::new(controller)),
        }
    }

    fn prepare<'s>(
        &mut self,
        manifest: &'s Manifest,
        registry: &'s SourceRegistry<'s>,
    ) -> Result<(), Error> {
        use request_origin::compact_storage::ControllerAccounting;
        if self.state != ReceiverState::Empty {
            return Err(Error::Retired);
        }
        self.state = ReceiverState::Entering;
        let Self { builder, inputs, pending_code, pending_map, declaration, setup,
            binding, original, public, state, imports, controller } = self;
        // This guard borrows ONLY state; all payloads remain in their fields.
        let attempt = ReceiverAttempt(state);
        let builder = builder.as_mut().ok_or(Error::Binding)?;
        let inputs = inputs.as_mut().ok_or(Error::Binding)?;
        let setup = setup.as_mut().ok_or(Error::Binding)?;
        let imports = imports.as_mut().ok_or(Error::Binding)?;
        let controller = controller.as_ref().ok_or(Error::Binding)?.borrow();
        if !controller.is_for(original.budget()) {
            return Err(Error::Binding);
        }
        let q = quote_imports(original, manifest, registry, public, imports)?;
        if imports.payload_bytes() < q.payload_bytes {
            return Err(Error::Bounds);
        }
        original.prepay_compiler_metadata(imports, q.metadata_work).map_err(|_| Error::Bounds)?;
        public.check_registry(registry)?;
        let sources = request_origin::RequestSources::new(manifest, registry, Arc::clone(public))?;
        original.budget().admit_peak(original.budget().used_bytes()).map_err(|_| Error::Bounds)?;
        *builder = empty_builder(manifest, registry);
        builder.allocated = source_norms::import_envelope(&manifest.norms)?.total_bytes;
        let node_capacity = add(q.declarations, 1)?;
        builder.nodes.try_reserve_exact(node_capacity).map_err(|_| Error::Allocation)?;
        if builder.nodes.capacity() != node_capacity {
            return Err(Error::Allocation);
        }
        for (key, source) in &manifest.tensors {
            if source.public_q18.is_some() {
                continue;
            }
            let lanes = source.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::Bounds))?;
            *pending_code = Some(Code::PublicInput {
                key: key.clone(), digest: source.digest, row_l1: source.row_l1,
                row_squared_norm: source.row_squared_norm,
                row_l1_squared_sum: source.row_l1_squared_sum,
            });
            pending_map.key = Some(key.clone());
            pending_map.shape = source.shape.clone();
            pending_map.norm = manifest.norms.get(key).cloned();
            pending_map.owner = Some(builder.push_retained(pending_code, lanes, Interval::new(source.lower, source.upper)?)?);
            #[cfg(test)]
            receiver_checkpoint(1)?;
            // Concrete String/usize ordering has no user callbacks or returned
            // allocator errors. No failpoint/fallibility between take and insert.
            let owner = pending_map.owner.take().expect("resident public owner");
            inputs.insert(pending_map.key.take().expect("resident map key"), SourceView {
                owner, shape: std::mem::take(&mut pending_map.shape),
            });
            if let Some(norm) = pending_map.norm.take() {
                builder.public_norms.insert(owner, norm);
            }
        }
        request_origin::declare_into(builder, original, &sources, imports, pending_code, declaration)?;
        setup.prepare(declaration, &sources, builder, imports, &*controller)?;
        *binding = Some(sources.prepared_binding());
        *attempt.0 = ReceiverState::Complete;
        Ok(())
    }

    fn refuse_drain(&mut self) {
        if self.state != ReceiverState::Drained { self.state = ReceiverState::CleanupOnly; }
        self.declaration.refuse_drain();
    }

    fn check_transport(&self, sources: Option<&request_origin::RequestSources<'_>>, empty: bool) -> Result<(), Error> {
        use request_origin::compact_storage::ControllerAccounting;
        if self.state != ReceiverState::Complete || !empty || self.pending_code.is_some()
            || self.pending_map.key.is_some() || self.pending_map.owner.is_some()
            || self.pending_map.norm.is_some() || self.pending_map.shape.capacity() != 0
        { return Err(Error::Binding); }
        let builder = self.builder.as_ref().ok_or(Error::Binding)?;
        let inputs = self.inputs.as_ref().ok_or(Error::Binding)?;
        let sources = sources.ok_or(Error::Binding)?;
        if !sources.matches_prepared(self.binding.as_ref().ok_or(Error::Binding)?)
            || !self.declaration.prepared_for(&self.original, &self.public)
            || !self.imports.as_ref().ok_or(Error::Binding)?.is_for(self.original.budget())
            || !self.controller.as_ref().ok_or(Error::Binding)?.borrow().is_for(self.original.budget())
            || self.setup.is_none()
        { return Err(Error::Binding); }
        // Immutable successful-prepare invariants, not a new source authority.
        sources.check_prepared_maps(builder, inputs)?;
        Ok(())
    }

    fn drain_complete<'s>(&mut self, sources: &mut Option<request_origin::RequestSources<'s>>,
        dst: &mut Option<request_origin::compact_storage::ResidentEmission<'s, 'loan>>,
    ) -> Result<(), Error> {
        resident_commit::drain(self, sources, dst)
    }
}

// Separate leaf: compact_pending must not inherit field construction access.
pub(in crate::replicated_decoder::program) mod resident_commit {
    use super::*;

    pub(in crate::replicated_decoder::program) struct ResidentCommit<'s, 'loan> {
        created: CreatedInputs<'s>,
        setup: request_origin::compact_storage::ResidentSetup,
        context: request_origin::compact_storage::ResidentContextOwners,
        loan: std::cell::RefCell<request_origin::compact_storage::ControllerLoan<'loan>>,
    }

    // Source-logical inline component; not a complete frontier/admission quote.
    fn resident_commit_bytes() -> Result<usize, Error> {
        bytes::<ResidentCommit<'_, '_>>(1)
    }

    impl<'s, 'loan> ResidentCommit<'s, 'loan> {
        // Only storage can construct this destination. No raw-parts getter,
        // arbitrary callback, constructor, Clone or Default is exposed.
        pub(in crate::replicated_decoder::program) fn install(self,
            destination: request_origin::compact_storage::ResidentCommitDestination<'_, 's, 'loan>,
        ) {
            let Self { created, setup, context, loan } = self;
            destination.install(created, setup, context, loan);
        }
    }

    pub(super) fn drain<'s, 'loan>(receiver: &mut InputReceiver<'loan>,
        sources: &mut Option<request_origin::RequestSources<'s>>,
        dst: &mut Option<request_origin::compact_storage::ResidentEmission<'s, 'loan>>,
    ) -> Result<(), Error> {
        if let Err(error) = receiver.check_transport(sources.as_ref(), dst.is_none()) {
            receiver.refuse_drain();
            return Err(error);
        }
        let InputReceiver { builder, inputs, declaration, setup, imports, controller, state, binding, .. } = receiver;
        // Both receivers and the destination are checked before extraction.
        let checked = match declaration.prevalidate_drain(builder.as_ref().expect("checked builder"), sources, true) {
            Ok(checked) => checked,
            Err(error) => { *state = ReceiverState::CleanupOnly; return Err(error); }
        };
        // Clone existing context references before extracting any owner.
        let context = request_origin::compact_storage::ResidentContextOwners::for_drain(&checked);
        // Concrete infallible commit: no callback, allocation, new RefCell,
        // fallible validation or live-payload destruction until installation.
        let permit = checked.commit();
        let created = CreatedInputs { builder: builder.take().expect("checked builder"),
            inputs: inputs.take().expect("checked maps"), permit,
            charge: imports.take().expect("checked import coverage"), created: () };
        let commit = ResidentCommit {
            created, setup: setup.take().expect("checked setup"), context,
            loan: controller.take().expect("checked lexical loan"),
        };
        request_origin::compact_storage::install_resident_commit(commit, dst);
        *binding = None;
        *state = ReceiverState::Drained;
        Ok(())
    }
}

/// Whole-owner commit: rejection leaves both slots, backing and debit intact.
fn transfer_complete<'loan>(
    src: &mut Option<InputReceiver<'loan>>,
    dst: &mut Option<InputReceiver<'loan>>,
) -> Result<(), Error> {
    if src.as_ref().ok_or(Error::Retired)?.state != ReceiverState::Complete {
        return Err(Error::Binding);
    }
    if dst.is_some() {
        return Err(Error::Binding);
    }
    *dst = src.take();
    Ok(())
}

#[cfg(test)]
thread_local! {
    // (resident checkpoint, panic rather than returned refusal).
    static RECEIVER_FAULT: std::cell::Cell<Option<(u8, bool)>> = const { std::cell::Cell::new(None) };
    static RECEIVER_DROP: std::cell::Cell<Option<(bool, usize, (usize, usize), (usize, usize))>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
pub(super) fn receiver_checkpoint(point: u8) -> Result<(), Error> {
    RECEIVER_FAULT.with(|fault| {
        if let Some((armed, panic)) = fault.get() {
            if armed == point {
                fault.set(None);
                if panic {
                    panic!("resident input checkpoint {point}");
                }
                return Err(Error::Binding);
            }
        }
        Ok(())
    })
}

/// Consumes source + admission on every outcome: no retry token/borrowed plan is
/// returned on failure. Request origin's existing Attempt burns later failures.
pub(super) fn create_inputs<'s>(
    original: OriginalSourcePlan,
    manifest: &'s Manifest,
    registry: &'s SourceRegistry<'s>,
    public: Arc<PublicSource>,
    mut charge: OwnedExternalCharge,
) -> Result<CreatedInputs<'s>, Error> {
    let q = quote_imports(&original, manifest, registry, &public, &charge)?;
    if charge.payload_bytes() < q.payload_bytes {
        return Err(Error::Bounds);
    }
    original
        .prepay_compiler_metadata(&charge, q.metadata_work)
        .map_err(|_| Error::Bounds)?;
    public.check_registry(registry)?;
    let sources = request_origin::RequestSources::new(manifest, registry, public)?;
    original
        .budget()
        .admit_peak(original.budget().used_bytes())
        .map_err(|_| Error::Bounds)?;
    let mut builder = empty_builder(manifest, registry);
    builder.allocated = source_norms::import_envelope(&manifest.norms)?.total_bytes;
    let node_capacity = add(q.declarations, 1)?;
    builder
        .nodes
        .try_reserve_exact(node_capacity)
        .map_err(|_| Error::Allocation)?;
    if builder.nodes.capacity() != node_capacity {
        return Err(Error::Allocation);
    }
    let mut inputs = BTreeMap::new();
    for (key, source) in &manifest.tensors {
        if source.public_q18.is_some() {
            continue;
        }
        let lanes = source
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::Bounds))?;
        let owner = builder.push(
            Code::PublicInput {
                key: key.clone(),
                digest: source.digest,
                row_l1: source.row_l1,
                row_squared_norm: source.row_squared_norm,
                row_l1_squared_sum: source.row_l1_squared_sum,
            },
            lanes,
            Interval::new(source.lower, source.upper)?,
        )?;
        inputs.insert(
            key.clone(),
            SourceView {
                owner,
                shape: source.shape.clone(),
            },
        );
        if let Some(norm) = manifest.norms.get(key) {
            builder.public_norms.insert(owner, norm.clone());
        }
    }
    let request = request_origin::quote(&builder, &original, &sources)?;
    let declaration = charge
        .split_owned(request.payload_bytes)
        .map_err(|_| Error::Bounds)?;
    let permit = request_origin::declare(&mut builder, original, sources, declaration)?;
    Ok(CreatedInputs {
        builder,
        inputs,
        permit,
        charge,
        created: (),
    })
}

/// No temporary Manifest/registry borrow escapes: finalization moves genuine
/// source/epoch/sticky/public-file custody into the SAME Program used by Worker.
/// Registry construction is SolBIAS's shared original-plan checked helper.
pub(in crate::replicated_decoder) struct EntryGuards {
    pub registry: OwnedExternalCharge,
    pub imports: OwnedExternalCharge,
    pub controller: OwnedExternalCharge,
}

pub(in crate::replicated_decoder) fn compile(
    original: OriginalSourcePlan,
    owners: SourceConstructionOwners<'_>,
    public: Arc<PublicSource>,
    guards: EntryGuards,
    limits: original_source::Limits,
    cap: usize,
) -> Result<Arc<Program>, Error> {
    #[cfg(test)]
    test_policy_factory_trap();
    let EntryGuards {
        registry: mut registry_guard,
        imports: mut imports_guard,
        mut controller,
    } = guards;
    if !controller.is_for(original.budget()) || !imports_guard.is_for(original.budget()) {
        return Err(Error::Binding);
    }
    let validation = SourceRegistry::original_construction_quote(&original, &owners, cap)?;
    if registry_guard.payload_bytes() < validation.new_owned_peak_bytes {
        registry_guard
            .resize_payload(validation.new_owned_peak_bytes)
            .map_err(|_| Error::Bounds)?;
    }
    let registry =
        SourceRegistry::from_original_checked(&original, &owners, &public, &registry_guard, cap)?;
    let imports = quote_imports(
        &original,
        owners.manifest,
        &registry,
        &public,
        &imports_guard,
    )?;
    if imports_guard.payload_bytes() < imports.payload_bytes {
        imports_guard
            .resize_payload(imports.payload_bytes)
            .map_err(|_| Error::Bounds)?;
    }
    let CreatedInputs {
        builder,
        inputs,
        permit,
        charge,
        created: (),
    } = create_inputs(original, owners.manifest, &registry, public, imports_guard)?;
    let lowering = permit.lowering_control_bytes(limits)?;
    if controller.payload_bytes() < lowering {
        controller
            .resize_payload(lowering)
            .map_err(|_| Error::Bounds)?;
    }
    // The import reservation itself becomes finalization's retained charge.
    // Never drop/reacquire or use a second unrelated counter at this handoff.
    let mut program = request_origin::schedule::compile(
        builder, permit, &inputs, controller, limits, charge, cap,
    )?;
    // Preserve existing binding/framing. Manifest digest is a POSTCONSTRUCTION
    // check only; never a source/plan/import/feedback constructor argument.
    Arc::get_mut(&mut program)
        .ok_or(Error::Binding)?
        .bind(owners.manifest.source, owners.manifest.composition);
    if program.digest() != owners.manifest.program {
        return Err(Error::Binding);
    }
    Ok(program)
}

/// Staged native ingress. The Option is a sticky consuming entry: every append
/// error takes/drops the entire owner, so neither a failed seal nor input can be
/// retried. No reference into an owned field is stored across calls.
pub struct SourceRequestImport(Option<Pending>);
#[path = "source_entry_ingress.rs"]
pub(in crate::replicated_decoder) mod ingress;
#[path = "source_snapshot_directory.rs"]
pub(in crate::replicated_decoder) mod snapshot_directory;
pub(super) enum SourceIngressRaw {
    Wire(OwnedCharged<json::ImportedValue>),
    CollectedHeader(OwnedCharged<json::ImportedValue>),
}
impl SourceIngressRaw {
    fn budget(&self) -> &pllm_garble::shared_allocation::Budget {
        match self {
            Self::Wire(raw) | Self::CollectedHeader(raw) => raw.budget(),
        }
    }
}
struct Pending {
    directory: Option<Arc<snapshot_directory::OwnedSnapshotDirectory>>,
    importer: PublicSourceImport,
    original: OriginalSourcePlan,
    manifest: OwnedCharged<Manifest>,
    raw: SourceIngressRaw,
    limits: original_source::Limits,
    storage: crate::replicated_decoder::sdk::SourceStorage,
    registry: OwnedExternalCharge,
    imports: OwnedExternalCharge,
    controller: OwnedExternalCharge,
    root: OwnedBudget,
}

fn json_limits() -> json::Limits {
    json::Limits {
        max_source_bytes: 1 << 20,
        max_nodes: 65536,
        max_depth: 16,
        max_string_bytes: 1 << 20,
        max_number_bytes: 64,
        max_number_work: 100_000_000,
        max_work: 1_000_000_000,
        max_payload_bytes: 8 << 20,
    }
}
fn source_limits() -> original_source::Limits {
    original_source::Limits {
        expansion: DecoderExpansionBudget {
            max_layers: 8,
            max_feature_width: 128,
            max_attention_heads: 8,
            max_vocabulary: 128,
            max_tensor_elements: 32768,
        },
        factory: pllm_models::ConfigLowererLimits {
            max_operations_per_phase: 256,
            max_string_bytes: 1024,
            max_attribute_nodes: 32768,
            max_retained_strings: 1 << 20,
            max_factory_bytes: MAX_ALLOCATION_BYTES,
        },
        max_work: 1_000_000_000_000,
        max_working_payload: MAX_ALLOCATION_BYTES,
    }
}

fn server_controller_bytes() -> usize {
    size_of::<Option<(&[u8], &[u8])>>()
        + size_of::<Option<OwnedCharged<json::ImportedValue>>>()
        + size_of::<crate::replicated_decoder::source_policy::ServerSourcePolicy<'_>>()
        + size_of::<Result<original_source::Limits, Error>>()
        + size_of::<Result<original_source::SourceQuote, crate::original_source::Error>>()
        + size_of::<original_source::Resources>()
        + size_of::<pllm_models::SourceScalarSummary>()
        + size_of::<Result<pllm_models::SourceScalarSummary, pllm_models::SourceScalarSummaryError>>(
        )
        + size_of::<std::iter::Zip<std::slice::Iter<'_, serde_json::Value>, std::slice::Iter<'_, u8>>>(
        )
        + size_of::<u128>()
        + size_of::<Option<u64>>()
        + 3 * size_of::<original_source::SourceAdmission>()
        + 3 * size_of::<original_source::PrivatePreparedConfiguration>()
        + 3 * size_of::<OriginalSourcePlan>()
        + 3 * size_of::<Result<(OriginalSourcePlan, original_source::Limits, super::sdk::SourceStorage), Error>>()
        + size_of::<
            Result<
                (
                    original_source::Limits,
                    crate::replicated_decoder::sdk::SourceStorage,
                ),
                Error,
            >,
        >()
}

fn pending_controller_bytes() -> usize {
    // Logical coverage assignment: three inline phase envelopes for ingress
    // staging, pending assembly/transfer and destination Pending. Wrapper/result
    // terms below are unchanged. This is not a measured SOURCE census, a full
    // transition-capacity proof, transitive payload coverage or a physical stack
    // lifetime claim. Other storage, cleanup and opaque U remain unproved.
    3 * size_of::<Pending>()
        + size_of::<SourceRequestImport>()
        + size_of::<Result<SourceRequestImport, Error>>()
        + size_of::<Result<crate::replicated_decoder::sdk::SourceRequestProgram, Error>>()
}

#[cfg(test)]
thread_local! { static POLICY_FACTORY_TRAP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
pub(in crate::replicated_decoder) struct PolicyFactoryTrap(bool);
#[cfg(test)]
impl PolicyFactoryTrap {
    pub(in crate::replicated_decoder) fn arm() -> Self {
        Self(POLICY_FACTORY_TRAP.with(|v| v.replace(true)))
    }
}
#[cfg(test)]
impl Drop for PolicyFactoryTrap {
    fn drop(&mut self) {
        POLICY_FACTORY_TRAP.with(|v| v.set(self.0));
    }
}
#[cfg(test)]
fn test_policy_factory_trap() {
    POLICY_FACTORY_TRAP.with(|v| {
        assert!(
            !v.get(),
            "metadata-only query entered source/payload/request factory"
        )
    });
}
fn import_json(root: &OwnedBudget, raw: &[u8]) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    let original = root
        .charge_external(raw.len())
        .map_err(|_| Error::Bounds)?
        .try_build(|_| {
            // Only the named borrowed slice participates in this COPY window;
            // its caller's allocation capacity is unknown. The destination and
            // owner control are ALREADY reserved on this same counter.
            root.budget().admit_peak(
                root.budget()
                    .used_bytes()
                    .checked_add(raw.len())
                    .ok_or(pllm_garble::shared_allocation::Error::Resource)?,
            )?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(raw.len())
                .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
            if bytes.capacity() != raw.len() {
                return Err(pllm_garble::shared_allocation::Error::Resource);
            }
            bytes.extend_from_slice(raw);
            Ok(bytes)
        })
        .map_err(|_| Error::Bounds)?;
    import_json_owner(root, original)
}
fn import_json_owner(
    root: &OwnedBudget,
    original: OwnedCharged<Vec<u8>>,
) -> Result<OwnedCharged<json::ImportedValue>, Error> {
    json::quote(
        original,
        root.charge_external(json::bootstrap_bytes().map_err(|_| Error::Bounds)?)
            .map_err(|_| Error::Bounds)?,
        json_limits(),
    )
    .map_err(|_| Error::Binding)?
    .construct()
    .map_err(|_| Error::Binding)
}

/// Staged native unit, deliberately NOT wired into ingress. Caller has one
/// admission from before C9 config import, never a loaded-config adoption path.
/// Server/manifest authentication remains the existing independent obligation.
/// Not a replacement for ordinary-serde SDK/Python configuration acceptance;
/// C9 numeric/duplicate faithfulness and that full domain remain unestablished.
pub(in crate::replicated_decoder) fn configure_bootstrap(
    admission: original_source::SourceAdmission,
    prepared: original_source::PrivatePreparedConfiguration,
    server: &OwnedCharged<json::ImportedValue>,
    manifest: Option<&OwnedCharged<json::ImportedValue>>,
    mut working: OwnedExternalCharge,
) -> Result<(OriginalSourcePlan, original_source::Limits, super::sdk::SourceStorage), Error> {
    admission.check_prepared(&prepared).map_err(|_| Error::Binding)?;
    if !working.is_for(prepared.configuration().budget())
        || working.payload_bytes() < server_controller_bytes()
    { return Err(Error::Bounds); }
    let (limits, storage) = super::source_policy::bootstrap_choice(server, &prepared, manifest, prepared.workload())?;
    let bytes = prepared.bootstrap_bytes(limits).map_err(|_| Error::Bounds)?;
    working.resize_payload(working.payload_bytes().max(bytes)).map_err(|_| Error::Bounds)?;
    let original = admission.configure(prepared, working, limits).map_err(|_| Error::Bounds)?;
    Ok((original, limits, storage))
}
#[cfg(test)]
thread_local! {
    static PUBLIC_OWNER_DROP:std::cell::Cell<Option<(bool,usize,usize)>>=const{std::cell::Cell::new(None)};
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_public_owner_drop(
) -> Option<(bool, usize, usize)> {
    PUBLIC_OWNER_DROP.with(|state| state.take())
}

// Field drop order: last outer PUBLIC payload, weak observer/control, THEN
// its independent same-counter reservation. No credit from request/import
// guards, which may already have retired inside emit.
// Staged private compact ingress; production dispatch waits for owning Storage.
#[cfg_attr(not(test), allow(dead_code))]
struct PublicSnapshotOwner {
    public: Arc<PublicSource>,
    charge: PublicSnapshotCharge,
}
// Same staged ingress owner; tests exercise payload-before-reservation retirement.
#[cfg_attr(not(test), allow(dead_code))]
struct PublicSnapshotCharge {
    #[cfg(test)]
    weak: std::sync::Weak<PublicSource>,
    guard: OwnedExternalCharge,
}
#[cfg(test)]
impl Drop for PublicSnapshotCharge {
    fn drop(&mut self) {
        PUBLIC_OWNER_DROP.with(|state| {
            state.set(Some((
                self.weak.upgrade().is_none(),
                self.guard.budget().used_bytes(),
                self.guard.charged_bytes(),
            )))
        });
    }
}
impl PublicSnapshotOwner {
    // Not a default-route factory: only the staged consuming ingress calls this.
    #[cfg_attr(not(test), allow(dead_code))]
    fn new(public: PublicSource, root: &OwnedBudget) -> Result<Self, Error> {
        let payload = add(
            add(public.resident_metadata_bytes()?, 2 * size_of::<usize>())?,
            size_of::<Self>(),
        )?;
        let phase = add(
            root.budget().used_bytes(),
            add(payload, size_of::<OwnedExternalCharge>())?,
        )?;
        if phase > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let guard = root.charge_external(payload).map_err(|_| Error::Bounds)?;
        // Reservation precedes Arc allocation. Its peak is recorded by SAME
        // source counter; descriptor/file backing comes from actual finish.
        let public = Arc::new(public);
        #[cfg(test)]
        let weak = Arc::downgrade(&public);
        Ok(Self {
            public,
            charge: PublicSnapshotCharge {
                #[cfg(test)]
                weak,
                guard,
            },
        })
    }
}

impl SourceRequestImport {
    /// Lexical test adapter over actual production source/import/input factories.
    /// Not a storage publication or an executable entry. The complete compact
    /// entry must take over this custody before production activation.
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn with_created<R>(
        self,
        emit: impl FnOnce(CreatedInputs<'_>, OwnedExternalCharge) -> Result<R, Error>,
    ) -> Result<R, Error> {
        self.consume_compact(|created, guard, _, _, _| emit(created, guard))
            .map(compact_pending::SourceCustody::into_test_payload)
    }

    /// Fixed bounded opt-in admission, not a new numeric/protocol choice. Large
    /// or unknown shapes fail closed; old constructors never enter this route.
    pub fn new(raw: &[u8]) -> Result<Self, Error> {
        Self::new_inner(raw, None)
    }

    /// Private native server opt-in. These bytes come from the server's trusted
    /// Experiment/config path, NEVER a client Manifest policy selector. This is
    /// resource permission only; original config factory and public snapshot
    /// verification remain independent and unchanged.
    #[doc(hidden)]
    pub fn new_checked_public(
        raw: &[u8],
        server_experiment: &[u8],
        server_config: &[u8],
    ) -> Result<Self, Error> {
        if server_experiment.is_empty()
            || server_experiment.len() > 32 << 10
            || server_config.is_empty()
            || server_config.len() > crate::replicated_decoder::source_policy::CONFIG_BYTES
        {
            return Err(Error::Bounds);
        }
        Self::new_inner(raw, Some((server_experiment, server_config)))
    }

    /// Nonconstructing original source quote. No checkpoint/tensor reads, native
    /// arithmetic factory, request, worker or child process is part of this API.
    /// Finite memory/phase/work refusal remains a normal typed result.
    /// Returns ORIGINAL source-owner resources only, not aggregate admission:
    /// server C9/policy work is separately removed from its ledger ceiling and
    /// same-counter server owners remain live during the entire query phase.
    #[doc(hidden)]
    pub fn quote_checked_public(
        server_experiment: &[u8],
        server_config: &[u8],
        rows: usize,
        outputs: usize,
    ) -> Result<original_source::Resources, Error> {
        use crate::replicated_decoder::source_policy;
        if server_experiment.is_empty()
            || server_experiment.len() > 32 << 10
            || server_config.is_empty()
            || server_config.len() > source_policy::CONFIG_BYTES
        {
            return Err(Error::Bounds);
        }
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).map_err(|_| Error::Bounds)?;
        let _controls = root
            .charge_external(server_controller_bytes())
            .map_err(|_| Error::Bounds)?;
        let server = import_json(&root, server_experiment)?;
        let config = import_json(&root, server_config)?;
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: u64::try_from(rows).map_err(|_| Error::Bounds)?,
            max_new_tokens: u64::try_from(outputs).map_err(|_| Error::Bounds)?,
        };
        let limits = source_policy::admitted_limits(&server, &config, None, workload)?;
        let bootstrap =
            original_source::bootstrap_bytes(&config, limits).map_err(|_| Error::Bounds)?;
        let quote = original_source::quote(
            config,
            root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,
            workload,
            limits,
        )
        .map_err(|_| Error::Bounds)?;
        Ok(quote.resources())
    }

    fn new_inner(raw: &[u8], server: Option<(&[u8], &[u8])>) -> Result<Self, Error> {
        use crate::replicated_decoder::source_policy;
        if raw.is_empty() || raw.len() > 1 << 20 {
            return Err(Error::Bounds);
        }
        let root = OwnedBudget::new(MAX_ALLOCATION_BYTES).map_err(prefix_bounds_error)?;
        let bootstrap = source_union_quote().map_err(prefix_bounds_error)?;
        root.reserve_bootstrap(bootstrap).map_err(prefix_bounds_error)?;
        // Full actual Pending/control layouts, before all source factories.
        // SAME guard subsequently transfers into emission/finalization.
        let controller = root
            .charge_external(pending_controller_bytes())
            .map_err(prefix_bounds_error)?;
        // Additional named server controller/return slots BEFORE C9 factories.
        // Caller CPython backing/allocator/RSS remains outside this logical
        // counter; import_json preserves E11's explicit borrowed-copy window.
        let _server_controls = match server {
            Some(_) => Some(
                root.charge_external(server_controller_bytes())
                    .map_err(prefix_bounds_error)?,
            ),
            None => None,
        };
        let server_document = server
            .map(|(experiment, _)| import_json(&root, experiment))
            .transpose()?;
        let imported = import_json(&root, raw)?;
        let counts = imported.payload().counts();
        let document = imported.payload().value();
        if document.get("storage").is_some() {
            return Err(Error::Binding);
        }
        if server.is_some()
            && (document.get("source_policy").is_some() || document.get("source_limits").is_some())
        {
            return Err(Error::Binding);
        }
        let config_values = document
            .get("config")
            .and_then(|v| v.as_array())
            .ok_or(Error::Binding)?;
        if config_values.is_empty() || config_values.len() > 1 << 20 {
            return Err(Error::Bounds);
        }
        let workload = DecoderWorkload {
            batch: 1,
            max_input_tokens: document
                .get("rows")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)?,
            max_new_tokens: document
                .get("outputs")
                .and_then(|v| v.as_u64())
                .ok_or(Error::Binding)?,
        };
        let config = if let Some((_, actual)) = server {
            if config_values.len() != actual.len()
                || !config_values
                    .iter()
                    .zip(actual)
                    .all(|(value, byte)| value.as_u64() == Some(u64::from(*byte)))
            {
                return Err(Error::Binding);
            }
            // The actual SERVER configuration is the original factory input.
            import_json(&root, actual)?
        } else {
            let config = root
                .charge_external(config_values.len())
                .map_err(|_| Error::Bounds)?
                .try_build(|_| {
                    // Config destination, source JSON owner, raw owner and result
                    // overlap are already represented by these live reservations.
                    root.budget().admit_peak(root.budget().used_bytes())?;
                    let mut bytes = Vec::new();
                    bytes
                        .try_reserve_exact(config_values.len())
                        .map_err(|_| pllm_garble::shared_allocation::Error::Resource)?;
                    if bytes.capacity() != config_values.len() {
                        return Err(pllm_garble::shared_allocation::Error::Resource);
                    }
                    for value in config_values {
                        bytes.push(
                            value
                                .as_u64()
                                .and_then(|n| u8::try_from(n).ok())
                                .ok_or(pllm_garble::shared_allocation::Error::Resource)?,
                        );
                    }
                    Ok(bytes)
                })
                .map_err(|_| Error::Binding)?;
            import_json_owner(&root, config)?
        };
        let (limits, storage) = if let Some(server_document) = &server_document {
            source_policy::admitted_choice(server_document, &config, Some(&imported), workload)?
        } else {
            (
                source_limits(),
                crate::replicated_decoder::sdk::SourceStorage::Expanded,
            )
        };
        let bootstrap =
            original_source::bootstrap_bytes(&config, limits).map_err(|_| Error::Bounds)?;
        let quote = original_source::quote(
            config,
            root.charge_external(bootstrap).map_err(|_| Error::Bounds)?,
            workload,
            limits,
        )
        .map_err(|_| Error::Binding)?;
        #[cfg(test)]
        test_policy_factory_trap();
        let original = quote.construct().map_err(|_| Error::Binding)?;
        // Manifest DTO bound from the SAME admitted immutable C9 tree. Price a
        // complete typed tree node per JSON span (including staging/overwrite),
        // plus all decoded text. This is deliberately conservative, no MAX-cap
        // credit or unbounded serde parse. Borrowed Value deserialization only.
        let node = source_registry::tree_entry_bytes::<(String, SourceTensor)>()?
            .max(source_registry::tree_entry_bytes::<(String, BlockNorms)>()?);
        let dto_bytes = counts
            .spans
            .checked_mul(node)
            .and_then(|n| n.checked_add(counts.decoded_string_bytes))
            .and_then(|n| n.checked_add(size_of::<Manifest>()))
            .ok_or(Error::Bounds)?;
        let dto_guard = root.charge_external(dto_bytes).map_err(|_| Error::Bounds)?;
        // Prepay the borrowed typed-DTO pass under the genuine source's same
        // cumulative ledger before serde allocates. No JSON reparse or Value
        // clone. Opaque serde/private allocator CPU remains unqualified.
        let dto_work = counts
            .spans
            .checked_mul(
                counts
                    .decoded_string_bytes
                    .checked_add(counts.spans)
                    .ok_or(Error::Bounds)?,
            )
            .ok_or(Error::Bounds)?;
        original
            .prepay_compiler_metadata(
                &dto_guard,
                u64::try_from(dto_work).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)?;
        let manifest = dto_guard
            .try_build(|_| {
                root.budget().admit_peak(root.budget().used_bytes())?;
                Manifest::deserialize(imported.payload().value())
                    .map_err(|_| pllm_garble::shared_allocation::Error::Resource)
            })
            .map_err(|_| Error::Binding)?;
        ingress::OwnedSourceIngress {
            directory: None,
            original,
            raw: SourceIngressRaw::Wire(imported),
            limits,
            storage,
            controller,
            root,
        }
        .into_pending(manifest)
    }

    pub fn session(&self) -> Result<[u8; 32], Error> {
        Ok(self
            .0
            .as_ref()
            .ok_or(Error::Retired)?
            .manifest
            .payload()
            .session)
    }
    pub fn append(&mut self, key: &str, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        #[cfg(test)]
        test_policy_factory_trap();
        let mut owner = self.0.take().ok_or(Error::Retired)?;
        owner
            .importer
            .append(key, offset, bytes)
            .map_err(|_| Error::Binding)?;
        self.0 = Some(owner);
        Ok(())
    }
    pub fn finish(mut self) -> Result<crate::replicated_decoder::sdk::SourceRequestProgram, Error> {
        if self.0.as_ref().ok_or(Error::Retired)?.storage
            == crate::replicated_decoder::sdk::SourceStorage::Compact
        {
            return crate::replicated_decoder::sdk::SourceRequestProgram::finish_compact(self);
        }
        let Pending {
            directory: _,
            importer,
            original,
            manifest,
            raw,
            limits,
            storage: _,
            registry,
            imports,
            controller,
            root,
        } = self.0.take().ok_or(Error::Retired)?;
        root.budget()
            .admit_peak(root.budget().used_bytes())
            .map_err(|_| Error::Bounds)?;
        let public = Arc::new(importer.finish().map_err(|_| Error::Binding)?);
        let result = crate::replicated_decoder::sdk::SourceRequestProgram::construct(
            original,
            manifest,
            public,
            registry,
            imports,
            controller,
            limits,
            MAX_PROGRAM_BYTES,
        );
        drop(raw);
        drop(root);
        result
    }
}

#[cfg(not(test))]
type HandoffIngress = ingress::OwnedSourceIngress;
#[cfg(not(test))]
pub(in crate::replicated_decoder::program) type HandoffSlot<T> = T;
#[cfg(test)]
pub(in crate::replicated_decoder::program) type HandoffSlot<T> = Observed<T>;

// Test-only mirror preserves the actual ingress field/drop order. No observer
// enters real ingress, Pending or downstream owners.
#[cfg(test)]
struct HandoffIngress {
    directory: Observed<Option<Arc<snapshot_directory::OwnedSnapshotDirectory>>>,
    original: Observed<OriginalSourcePlan>,
    raw: Observed<SourceIngressRaw>,
    limits: original_source::Limits,
    storage: crate::replicated_decoder::sdk::SourceStorage,
    controller: Observed<OwnedExternalCharge>,
    root: OwnedBudget,
}

#[cfg(test)]
pub(in crate::replicated_decoder::program) struct Observed<T> {
    value: T,
    after: AfterDrop,
}
#[cfg(test)]
struct AfterDrop {
    trace: Option<Arc<Trace>>,
    event: Event,
    armed: bool,
}
#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::replicated_decoder::program) enum Event {
    DirectorySlot,
    OriginalSlot,
    RawSlot,
    ManifestSlot,
    ControllerCharge,
    CreatedInputs,
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) struct Trace {
    next: std::sync::atomic::AtomicUsize,
    completed: [std::sync::atomic::AtomicUsize; 5],
    // Inert identity only; never dereferenced.
    budget_id: usize,
}
#[cfg(test)]
impl Drop for AfterDrop {
    fn drop(&mut self) {
        // Normal nonpanicking destruction only, not unwind-completion evidence.
        // The test retains its Arc until all markers have retired.
        if self.armed {
            if let Some(trace) = &self.trace {
                let ordinal = trace.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Separate per-instance traces reuse the original-slot cell for
                // the constructor's complete CreatedInputs aggregate.
                let index = match self.event {
                    Event::CreatedInputs => Event::OriginalSlot as usize,
                    event => event as usize,
                };
                trace.completed[index]
                    .store(ordinal, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

#[cfg(not(test))]
pub(in crate::replicated_decoder::program) fn slot_ref<T>(slot: &HandoffSlot<T>) -> &T {
    slot
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn slot_ref<T>(slot: &HandoffSlot<T>) -> &T {
    &slot.value
}
#[cfg(not(test))]
fn slot_mut<T>(slot: &mut HandoffSlot<T>) -> &mut T {
    slot
}
#[cfg(test)]
fn slot_mut<T>(slot: &mut HandoffSlot<T>) -> &mut T {
    &mut slot.value
}
#[cfg(not(test))]
pub(in crate::replicated_decoder::program) fn slot_take<T>(slot: HandoffSlot<T>) -> T {
    slot
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn slot_take<T>(slot: HandoffSlot<T>) -> T {
    let Observed { value, mut after } = slot;
    after.armed = false;
    value
}

#[cfg(test)]
pub(in crate::replicated_decoder::program) fn observe<T>(value: T, trace: Option<&Arc<Trace>>, event: Event) -> Observed<T> {
    Observed {
        value,
        after: AfterDrop { trace: trace.map(Arc::clone), event, armed: true },
    }
}

#[cfg(test)]
impl Trace {
    pub(in crate::replicated_decoder::program) fn new(budget: &Budget) -> Arc<Self> {
        Arc::new(Self {
            next: std::sync::atomic::AtomicUsize::new(1),
            completed: std::array::from_fn(|_| std::sync::atomic::AtomicUsize::new(0)),
            budget_id: budget as *const _ as usize,
        })
    }
    pub(in crate::replicated_decoder::program) fn completion(&self, event: Event) -> usize {
        let index = match event {
            Event::CreatedInputs => Event::OriginalSlot as usize,
            event => event as usize,
        };
        self.completed[index].load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(in crate::replicated_decoder::program) fn budget_id(&self) -> usize {
        self.budget_id
    }
    pub(in crate::replicated_decoder::program) fn next_ordinal(&self) -> usize {
        self.next.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
fn wrap_handoff(
    ingress: ingress::OwnedSourceIngress,
    manifest: OwnedCharged<Manifest>,
    trace: Option<&Arc<Trace>>,
) -> (HandoffIngress, HandoffSlot<OwnedCharged<Manifest>>) {
    let ingress::OwnedSourceIngress {
        directory, original, raw, limits, storage, controller, root,
    } = ingress;
    (
        HandoffIngress {
            directory: observe(directory, trace, Event::DirectorySlot),
            original: observe(original, trace, Event::OriginalSlot),
            raw: observe(raw, trace, Event::RawSlot),
            limits,
            storage,
            controller: observe(controller, trace, Event::ControllerCharge),
            root,
        },
        observe(manifest, trace, Event::ManifestSlot),
    )
}

impl ingress::OwnedSourceIngress {
    // Shared worker/publisher transition; never constructs a Budget or reparses
    // raw configuration. All inputs are owned fields of the genuine ingress.
    fn into_pending(self, manifest: OwnedCharged<Manifest>) -> Result<SourceRequestImport, Error> {
        #[cfg(not(test))]
        {
            pending_transition(self, manifest)
        }
        #[cfg(test)]
        {
            let (ingress, manifest) = wrap_handoff(self, manifest, None);
            pending_transition(ingress, manifest)
        }
    }

    #[cfg(test)]
    fn into_pending_observed(
        self,
        manifest: OwnedCharged<Manifest>,
        trace: &Arc<Trace>,
    ) -> Result<SourceRequestImport, Error> {
        let (ingress, manifest) = wrap_handoff(self, manifest, Some(trace));
        pending_transition(ingress, manifest)
    }
}

fn pending_transition(
    ingress: HandoffIngress,
    manifest: HandoffSlot<OwnedCharged<Manifest>>,
) -> Result<SourceRequestImport, Error> {
    // Bind in this order so refusal/unwind retires manifest before intact
    // ingress. This makes no claim that moved-from parameter storage retires.
    let ingress = ingress;
    let manifest = manifest;
    if !std::ptr::eq(slot_ref(&manifest).budget(), ingress.root.budget())
        || !std::ptr::eq(slot_ref(&ingress.original).budget(), ingress.root.budget())
        || !std::ptr::eq(slot_ref(&ingress.raw).budget(), ingress.root.budget())
        || !slot_ref(&ingress.controller).is_for(ingress.root.budget())
    {
        return Err(Error::Binding);
    }
    check_source(slot_ref(&ingress.original), slot_ref(&manifest).payload())?;
    let new_directory = match (ingress.storage, slot_ref(&ingress.directory).as_ref()) {
        (crate::replicated_decoder::sdk::SourceStorage::Compact, Some(owner)) => {
            owner.check_budget(ingress.root.budget())?;
            None
        }
        (crate::replicated_decoder::sdk::SourceStorage::Compact, None) => Some(
            snapshot_directory::OwnedSnapshotDirectory::create(
                slot_ref(&ingress.original),
                slot_ref(&ingress.controller),
                &ingress.root,
            )?,
        ),
        (crate::replicated_decoder::sdk::SourceStorage::Expanded, None) => None,
        _ => return Err(Error::Binding),
    };
    // Preparation borrows intact ingress; later preparation locals retire
    // before new_directory, manifest and ingress on refusal/unwind. Partial
    // directory creation likewise borrows ingress. Escaped Arcs, allocator
    // cleanup, abort and full-resource funding are outside this ordering claim.
    let (importer, registry, imports) = {
        let directory = slot_ref(&ingress.directory).as_ref().or(new_directory.as_ref());
        let mut registry = ingress.root.charge_external(0).map_err(|_| Error::Bounds)?;
        let mut imports = ingress.root.charge_external(0).map_err(|_| Error::Bounds)?;
        let owners = SourceConstructionOwners {
            manifest: slot_ref(&manifest).payload(),
            incoming: None,
            public: None,
            retained_baseline_bytes: ingress.root
                .budget()
                .used_bytes()
                .checked_sub(registry.payload_bytes())
                .ok_or(Error::Bounds)?,
        };
        let q = SourceRegistry::original_construction_quote(slot_ref(&ingress.original), &owners, MAX_PROGRAM_BYTES)?;
        registry
            .resize_payload(q.new_owned_peak_bytes)
            .map_err(|_| Error::Bounds)?;
        let descriptors = SourceRegistry::from_original_records_checked(
            slot_ref(&ingress.original),
            &owners,
            &registry,
            MAX_PROGRAM_BYTES,
        )?;
        let q = PublicSourceImport::construction_quote(&descriptors, ingress.root.budget().used_bytes())?;
        q.admit(MAX_PROGRAM_BYTES)?;
        imports
            .resize_payload(q.new_owned_peak_bytes)
            .map_err(|_| Error::Bounds)?;
        ingress.root.budget()
            .admit_peak(ingress.root.budget().used_bytes())
            .map_err(|_| Error::Bounds)?;
        let importer = PublicSourceImport::new_checked_location(
            &descriptors,
            ingress.root.budget()
                .used_bytes()
                .checked_sub(imports.payload_bytes())
                .ok_or(Error::Bounds)?,
            MAX_PROGRAM_BYTES,
            directory.map(|owner| {
                Arc::clone(owner) as Arc<dyn pllm_core::public_file::SnapshotDirectory>
            }),
        )?;
        (importer, registry, imports)
    };
    // Commit only after the last fallible call and the preparation borrows.
    let HandoffIngress {
        directory,
        original,
        raw,
        limits,
        storage,
        controller,
        root,
    } = ingress;
    let directory = slot_take(directory).or(new_directory);
    let original = slot_take(original);
    let raw = slot_take(raw);
    let controller = slot_take(controller);
    let manifest = slot_take(manifest);
    Ok(SourceRequestImport(Some(Pending {
        directory,
        importer,
        original,
        manifest,
        raw,
        limits,
        storage,
        registry,
        imports,
        controller,
        root,
    })))
}

#[cfg(test)]
#[path = "source_entry_tests.rs"]
mod tests;
