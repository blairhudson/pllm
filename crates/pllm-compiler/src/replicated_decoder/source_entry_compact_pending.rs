//! Consuming lexical conversion boundary. This owns authentic ingress backing,
//! not an executable schedule. Only the complete storage finalizer may publish.
use super::*;

#[path = "source_entry_compact_handles.rs"]
pub(in crate::replicated_decoder::program) mod handles;
#[path = "compact_request_sealed.rs"]
pub(in crate::replicated_decoder::program) mod sealed;
use handles::SourceCatalog;
use request_origin::compact_storage as compact;

#[cfg(not(test))]
type CompactPending = Pending;
// Nonfallible test adapter; owns the actual fields, with no payload retainers.
#[cfg(test)]
struct CompactPending {
    directory: HandoffSlot<Option<Arc<snapshot_directory::OwnedSnapshotDirectory>>>,
    importer: PublicSourceImport,
    original: HandoffSlot<OriginalSourcePlan>,
    manifest: HandoffSlot<OwnedCharged<Manifest>>,
    raw: HandoffSlot<SourceIngressRaw>,
    limits: original_source::Limits,
    storage: crate::replicated_decoder::sdk::SourceStorage,
    registry: OwnedExternalCharge,
    imports: OwnedExternalCharge,
    controller: HandoffSlot<OwnedExternalCharge>,
    root: OwnedBudget,
}
#[cfg(test)]
fn wrap_compact(pending: Pending, trace: Option<&Arc<Trace>>) -> CompactPending {
    let Pending { directory, importer, original, manifest, raw, limits, storage, registry, imports,
        controller, root } = pending;
    CompactPending {
        directory: observe(directory, trace, Event::DirectorySlot),
        importer,
        original: observe(original, trace, Event::OriginalSlot),
        manifest: observe(manifest, trace, Event::ManifestSlot),
        raw: observe(raw, trace, Event::RawSlot),
        limits, storage,
        registry, imports,
        controller: observe(controller, trace, Event::ControllerCharge),
        root,
    }
}

/// Inactive prerequisite entry. Caller already owns genuine original/charges in
/// receiver. No consuming importer, graph emission, seal or runtime activation.
fn prepare_resident_inputs<'s>(
    receiver: &mut InputReceiver<'_>,
    manifest: &'s Manifest,
    registry: &'s SourceRegistry<'s>,
) -> Result<(), Error> {
    receiver.prepare(manifest, registry)
}

#[cfg(test)]
#[path = "resident_input_tests.rs"]
mod resident_input_tests;

// Nonfallible assembly immediately after Pending extraction. Both ordinary
// failure and unwind retire importer payload BEFORE its ORIGINAL reservation.
// Private staging only until the complete owning request publisher is wired.
#[cfg_attr(not(test), allow(dead_code))]
struct ImportStage {
    importer: Option<PublicSourceImport>,
    charge: Option<OwnedExternalCharge>,
}
impl Drop for ImportStage {
    fn drop(&mut self) {
        #[cfg(test)]
        let had_importer = self.importer.is_some();
        drop(self.importer.take());
        #[cfg(test)]
        if had_importer {
            let charge = self
                .charge
                .as_ref()
                .expect("importer retains original charge");
            IMPORT_DROP.with(|record| {
                record.set(Some((
                    self.importer.is_none(),
                    charge.budget().used_bytes(),
                    charge.charged_bytes(),
                )))
            });
        }
        // charge field drops ONLY after actual importer destruction returned.
    }
}
#[cfg(test)]
thread_local! {
    static IMPORT_DROP:std::cell::Cell<Option<(bool,usize,usize)>>=const{std::cell::Cell::new(None)};
    static PANIC_BEFORE_BOX:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_drop_record() -> Option<(bool, usize, usize)> {
    IMPORT_DROP.with(|record| record.take())
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_panic_before_box() {
    PANIC_BEFORE_BOX.with(|flag| flag.set(true));
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_hold_remaining(
    entry: &SourceRequestImport,
) -> OwnedExternalCharge {
    let root = &entry.0.as_ref().unwrap().root;
    let mut guard = root.charge_external(0).unwrap();
    let remaining = MAX_PROGRAM_BYTES - root.budget().used_bytes();
    guard.resize_payload(remaining).unwrap();
    guard
}
#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_root(
    entry: &SourceRequestImport,
) -> OwnedBudget {
    entry.0.as_ref().unwrap().root.try_clone().unwrap()
}

#[cfg(test)]
pub(in crate::replicated_decoder::program) fn test_controller_identity(
    entry: &SourceRequestImport,
) -> (usize, usize) {
    let pending = entry.0.as_ref().unwrap();
    assert!(pending.controller.is_for(pending.root.budget()));
    assert!(std::ptr::eq(pending.original.budget(), pending.root.budget()));
    assert!(std::ptr::eq(pending.manifest.budget(), pending.root.budget()));
    assert!(std::ptr::eq(pending.raw.budget(), pending.root.budget()));
    (pending.controller.budget() as *const Budget as usize,
        pending.controller.charged_bytes())
}

// Validation only; neither implementation exposes or transfers its controller.
trait CustodyController {
    fn check_budget(&self, budget: &pllm_garble::shared_allocation::Budget) -> Result<(), Error>;
}
impl CustodyController for () {
    fn check_budget(&self, _: &pllm_garble::shared_allocation::Budget) -> Result<(), Error> { Ok(()) }
}
impl CustodyController for OwnedExternalCharge {
    fn check_budget(&self, budget: &pllm_garble::shared_allocation::Budget) -> Result<(), Error> {
        if self.is_for(budget) { Ok(()) } else { Err(Error::Binding) }
    }
}

/// Payloads precede reservations in field drop order. No borrowed registry,
/// FrozenRegion or self-referential pointer is stored beside its referents.
// Ingress custody, not executable Storage; no production route activates it yet.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::replicated_decoder) struct SourceCustody<R, C = ()> {
    produced: R,
    original: Arc<OriginalSourcePlan>,
    public: PublicSnapshotOwner,
    manifest: Box<OwnedCharged<Manifest>>,
    raw: SourceIngressRaw,
    directory: Option<Arc<snapshot_directory::OwnedSnapshotDirectory>>,
    catalog: SourceCatalog,
    instance: source_registry::Instance,
    binding: [u8; 32],
    registry_charge: OwnedExternalCharge,
    manifest_box_charge: OwnedExternalCharge,
    catalog_charge: OwnedExternalCharge,
    controller: C,
    root: OwnedBudget,
}

impl<R, C: CustodyController> SourceCustody<R, C> {
    fn canonical_prefix_for_shape(&self, rows: usize, outputs: usize) -> Result<sha2::Sha256, Error> {
        request_origin::schedule::canonical_prefix(
            &self.original, rows, outputs, self.manifest.payload(),
        )
    }
    pub(in crate::replicated_decoder::program) fn directory(
        &self,
    ) -> Result<&Arc<snapshot_directory::OwnedSnapshotDirectory>, Error> {
        let owner = self.directory.as_ref().ok_or(Error::Binding)?;
        owner.check_budget(self.root.budget())?;
        Ok(owner)
    }
    pub(in crate::replicated_decoder::program) fn check_emission(
        &self,
        emission: &request_origin::compact_storage::owning::OwnedEmission<impl Sized>,
    ) -> Result<(), Error> {
        self.check_ingress(self.instance, self.binding)?;
        emission.check_source_custody(self.manifest.payload(), &self.public.public, self.instance)
    }
    /// Reader ingress checks the actual epoch's retained instance/source root;
    /// neither a binding digest nor a caller-provided Manifest can replace it.
    pub(in crate::replicated_decoder::program) fn check_source_epoch(
        &self,
        instance: source_registry::Instance,
        plan: &OriginalSourcePlan,
    ) -> Result<(), Error> {
        self.check_ingress(instance, self.binding)?;
        if !self.original.same_origin(plan) || !std::ptr::eq(plan.budget(), self.root.budget()) {
            return Err(Error::Binding);
        }
        Ok(())
    }
    /// Borrow only; no payload extraction or reservation detachment.
    pub(in crate::replicated_decoder::program) fn produced(&self) -> &R {
        &self.produced
    }
    // Called by staged ingress, with genuine source lifecycle coverage in tests.
    #[cfg_attr(not(test), allow(dead_code))]
    fn check_ingress(
        &self,
        instance: source_registry::Instance,
        binding: [u8; 32],
    ) -> Result<(), Error> {
        let budget = self.root.budget();
        if self.instance != instance
            || self.binding != binding
            || !std::ptr::eq(self.original.budget(), budget)
            || !std::ptr::eq(self.manifest.budget(), budget)
            || !std::ptr::eq(self.raw.budget(), budget)
            || !self.public.charge.guard.is_for(budget)
            || !self.registry_charge.is_for(budget)
            || !self.manifest_box_charge.is_for(budget)
            || !self.catalog_charge.is_for(budget)
        {
            return Err(Error::Binding);
        }
        self.controller.check_budget(budget)?;
        self.catalog.check_owned(
            self.manifest.payload(),
            &self.original,
            &self.catalog_charge,
        )
    }
}

impl<R> SourceCustody<R> {
    #[cfg(test)]
    pub(super) fn into_test_payload(self) -> R {
        self.produced
    }
}

impl
    SourceCustody<
        request_origin::compact_storage::owning::OwnedEmission<
            request_origin::compact_storage::compact_history::CompletionLoan,
        >,
    >
{
    pub(in crate::replicated_decoder::program) fn compact_canonical_prefix(
        &self,
    ) -> Result<sha2::Sha256, Error> {
        self.check_emission(&self.produced)?;
        let (rows, outputs) = self.produced.original_request_shape();
        self.canonical_prefix_for_shape(rows, outputs)
    }
}

// Resident operations always use the private, genuinely paired payload and
// original controller. No getter can independently select or extract a guard.
impl<R> SourceCustody<compact::owning::DetachedEmission<R>, OwnedExternalCharge> {
    pub(in crate::replicated_decoder::program) fn check_resident_emission(&self) -> Result<(), Error> {
        self.check_ingress(self.instance, self.binding)?;
        self.produced.check_source_custody_with(
            &self.controller, self.manifest.payload(), &self.public.public, self.instance,
        )
    }
    pub(in crate::replicated_decoder::program) fn check_open(&self) -> Result<(), Error> {
        self.produced.check_open_with(&self.controller)
    }
    pub(in crate::replicated_decoder::program) fn checked_imports(
        &self, association: &compact::owning::EpochAssociation,
    ) -> Result<&Builder, Error> {
        self.produced.checked_imports_with(&self.controller, association)
    }
    pub(in crate::replicated_decoder::program) fn checked_epoch_imports(
        &self, association: &compact::owning::EpochAssociation, epoch: usize,
    ) -> Result<&Builder, Error> {
        self.produced.checked_epoch_imports_with(&self.controller, association, epoch)
    }
    pub(in crate::replicated_decoder::program) fn prepay_import_metadata(&self, visits: usize) -> Result<(), Error> {
        self.produced.prepay_import_metadata_with(&self.controller, visits)
    }
    pub(in crate::replicated_decoder::program) fn checked_import_binding(&self, owner: usize) -> Result<(&str, &[usize]), Error> {
        self.produced.checked_import_binding_with(&self.controller, owner)
    }
    pub(in crate::replicated_decoder::program) fn inspection_attempt(&self) -> Result<compact::owning::InspectionAttempt<'_>, Error> {
        self.produced.inspection_attempt_with(&self.controller)
    }
    pub(in crate::replicated_decoder::program) fn charge_inspection(
        &self, bytes: usize, visits: u64,
    ) -> Result<pllm_garble::shared_allocation::ExternalCharge<'_>, Error> {
        self.produced.charge_inspection_with(&self.controller, bytes, visits)
    }
    pub(in crate::replicated_decoder::program) fn original_request_shape(&self) -> (usize, usize) {
        self.produced.original_request_shape()
    }
    pub(in crate::replicated_decoder::program) fn source_plan(&self) -> &OriginalSourcePlan {
        self.produced.source_plan()
    }
    pub(in crate::replicated_decoder::program) fn work(&self) -> pllm_core::puma::WorkBudget {
        self.produced.work()
    }
    pub(in crate::replicated_decoder::program) fn program_cap(&self) -> usize {
        self.produced.program_cap()
    }
}
impl SourceCustody<compact::owning::DetachedEmission<compact::compact_history::CompletionLoan>, OwnedExternalCharge> {
    pub(in crate::replicated_decoder::program) fn compact_canonical_prefix(&self) -> Result<sha2::Sha256, Error> {
        self.check_resident_emission()?;
        let (rows, outputs) = self.produced.original_request_shape();
        self.canonical_prefix_for_shape(rows, outputs)
    }
    pub(in crate::replicated_decoder::program) fn completed_imports(&self) -> Result<&Builder, Error> {
        self.produced.completed_imports_with(&self.controller)
    }
    pub(in crate::replicated_decoder::program) fn check_completed_attempt(&self, attempt: &request_origin::schedule::Attempt) -> Result<(), Error> {
        self.produced.check_completed_attempt_with(&self.controller, attempt)
    }
    pub(in crate::replicated_decoder::program) fn prepay_publication(
        &self, attempt: &request_origin::schedule::Attempt, visits: u64,
    ) -> Result<(), Error> {
        self.produced.prepay_publication_with(&self.controller, attempt, visits)
    }
    pub(in crate::replicated_decoder::program) fn finish_lease(&mut self) -> Result<OwnedExternalCharge, Error> {
        self.produced.finish_lease_with(&mut self.controller)
    }
    pub(in crate::replicated_decoder::program) fn prepare_compact_history(&mut self, bytes: usize) -> Result<request_origin::schedule::Attempt, Error> {
        self.produced.prepare_compact_history_with(&mut self.controller, bytes)
    }
    pub(in crate::replicated_decoder::program) fn complete_compact_history(&self, attempt: &request_origin::schedule::Attempt) -> Result<(), Error> {
        self.produced.complete_compact_history_with(&self.controller, attempt)
    }
    pub(in crate::replicated_decoder::program) fn checked_epoch_imports_for_finish(&self) -> Result<usize, Error> {
        self.produced.checked_epoch_imports_for_finish_with(&self.controller)
    }
    pub(in crate::replicated_decoder::program) fn epochs(&self) -> &[compact_lowering::frozen_region::owning::OwnedEpoch] {
        self.produced.epochs()
    }
    pub(in crate::replicated_decoder::program) fn outputs(&self) -> [(usize, usize); 3] {
        self.produced.outputs()
    }
    pub(in crate::replicated_decoder::program) fn logical_len(&self) -> usize {
        self.produced.logical_len()
    }
    pub(in crate::replicated_decoder::program) fn acceptance_node(&self, ordinal: usize) -> Result<schedule_finalization::BorrowedNode<'_>, Error> {
        self.produced.acceptance_node(ordinal)
    }
}

impl SourceRequestImport {
    /// Compiler-private opt-in. The default SDK route remains unchanged. This
    /// consumes the actual authenticated import, not a detached manifest seal.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::replicated_decoder::program) fn compile_compact(
        self,
    ) -> Result<Arc<Program>, Error> {
        self.compile_compact_storage()?.into_program()
    }
    pub(in crate::replicated_decoder) fn compile_compact_storage(
        self,
    ) -> Result<sealed::SealedOwnedStorage, Error> {
        let limits = self.0.as_ref().ok_or(Error::Retired)?.limits;
        let custody=self.consume_compact(|created,guard,_,_,source|{
            let scope=compact::EmissionScope::new(created,guard,MAX_PROGRAM_BYTES)?;
            let (history, second)=emit_compact_epochs(scope,source,limits)?;
            history.finish(second)
        })?;
        custody.seal_compact()
    }

    // Private and inactive: no seal/publication consumer selects this route yet.
    fn consume_resident<R>(
        mut self,
        emit: impl for<'s> FnOnce(
            compact::ResidentEmission<'s, 's>,
            &'s SourceRegistry<'s>,
            &'s SourceCatalog,
            &'s handles::SourceScope<'s>,
        ) -> Result<R, Error>,
    ) -> Result<SourceCustody<R, OwnedExternalCharge>, Error> {
        consume_resident_transition(self.0.take().ok_or(Error::Retired)?, emit)
    }
    /// One consuming entry. Conversion sees the SAME genuine registry and all
    /// actual source/norm associations. HRTB prevents any lexical borrow escaping
    /// in R; conversion must produce owning records/handles. Then those borrows
    /// end and the SAME whole OwnedCharged wrappers move, without refund/rebuy.
    // Deliberately inactive until borrow-free recipes/history can publish Storage.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(in crate::replicated_decoder::program) fn consume_compact<R>(
        mut self,
        emit: impl for<'s> FnOnce(
            CreatedInputs<'s>,
            OwnedExternalCharge,
            &'s SourceRegistry<'s>,
            &'s SourceCatalog,
            &'s handles::SourceScope<'s>,
        ) -> Result<R, Error>,
    ) -> Result<SourceCustody<R>, Error> {
        let pending = self.0.take().ok_or(Error::Retired)?;
        #[cfg(test)]
        let pending = wrap_compact(pending, None);
        consume_transition(pending, emit)
    }

    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn consume_compact_observed<R>(
        mut self,
        trace: &Arc<Trace>,
        emit: impl for<'s> FnOnce(CreatedInputs<'s>, OwnedExternalCharge,
            &'s SourceRegistry<'s>, &'s SourceCatalog, &'s handles::SourceScope<'s>) -> Result<R, Error>,
    ) -> Result<SourceCustody<R>, Error> {
        let pending = self.0.take().ok_or(Error::Retired)?;
        consume_transition(wrap_compact(pending, Some(trace)), emit)
    }
}

fn consume_transition<R>(
    pending: CompactPending,
    emit: impl for<'s> FnOnce(CreatedInputs<'s>, OwnedExternalCharge,
        &'s SourceRegistry<'s>, &'s SourceCatalog, &'s handles::SourceScope<'s>) -> Result<R, Error>,
) -> Result<SourceCustody<R>, Error> {
        let mut pending = pending;
        let mut importing = ImportStage {
            importer: Some(pending.importer),
            charge: Some(pending.imports),
        };
        #[cfg(test)]
        PANIC_BEFORE_BOX.with(|flag| {
            if flag.replace(false) {
                panic!("ingress unwind before box reservation")
            }
        });
        // Stabilize the SAME whole wrapper BEFORE registry/origin borrows. Its
        // source/norm map addresses and BTree entry allocations then survive the
        // outer custody move unchanged. No field extraction or raw-Arc adoption.
        let boxed_bytes = add(
            size_of::<OwnedCharged<Manifest>>(),
            2 * size_of::<Box<OwnedCharged<Manifest>>>(),
        )?;
        if add(pending.root.budget().used_bytes(), boxed_bytes)? > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let manifest_box_charge = pending.root
            .charge_external(boxed_bytes)
            .map_err(|_| Error::Bounds)?;
        let manifest = Box::new(slot_take(pending.manifest));
        let outer = PublicSnapshotOwner::new(
            importing
                .importer
                .take()
                .ok_or(Error::Retired)?
                .finish()
                .map_err(|_| Error::Binding)?,
            &pending.root,
        )?;
        if !outer.charge.guard.is_for(pending.root.budget()) {
            return Err(Error::Binding);
        }
        let context = add(
            request_origin::compact_storage::context_bytes()?,
            OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
        )?;
        if slot_ref(&pending.controller).payload_bytes() < context {
            slot_mut(&mut pending.controller)
                .resize_payload(context)
                .map_err(|_| Error::Bounds)?;
        }
        let (produced, retained_original, catalog, instance, binding, catalog_charge) = {
            let public = &outer.public;
            let owners = SourceConstructionOwners {
                manifest: manifest.payload(),
                incoming: None,
                public: Some(public),
                retained_baseline_bytes: pending.root
                    .budget()
                    .used_bytes()
                    .checked_sub(pending.registry.payload_bytes())
                    .ok_or(Error::Bounds)?,
            };
            let q =
                SourceRegistry::original_construction_quote(slot_ref(&pending.original), &owners, MAX_PROGRAM_BYTES)?;
            if pending.registry.payload_bytes() < q.new_owned_peak_bytes {
                pending.registry
                    .resize_payload(q.new_owned_peak_bytes)
                    .map_err(|_| Error::Bounds)?;
            }
            let registry = SourceRegistry::from_original_checked(
                slot_ref(&pending.original),
                &owners,
                public,
                &pending.registry,
                MAX_PROGRAM_BYTES,
            )?;
            // Includes empty maps: identity is the actual Manifest's maps, never
            // equality of their contents or a detached seal paired by a caller.
            if !std::ptr::eq(registry.source_records(), &manifest.payload().tensors)
                || !std::ptr::eq(registry.source_norms(), &manifest.payload().norms)
            {
                return Err(Error::Binding);
            }
            public.check_registry(&registry)?;
            let quote =
                SourceCatalog::quote::<R>(&registry, manifest.payload(), slot_ref(&pending.original), slot_ref(&pending.controller))?;
            if add(pending.root.budget().used_bytes(), quote.bytes)? > MAX_PROGRAM_BYTES {
                return Err(Error::Bounds);
            }
            let catalog_charge = pending.root
                .charge_external(quote.bytes)
                .map_err(|_| Error::Bounds)?;
            let catalog = SourceCatalog::build(
                &registry,
                manifest.payload(),
                slot_ref(&pending.original),
                &catalog_charge,
                quote,
            )?;
            let import_guard = importing.charge.as_mut().ok_or(Error::Retired)?;
            let q = quote_imports(
                slot_ref(&pending.original),
                manifest.payload(),
                &registry,
                public,
                import_guard,
            )?;
            if import_guard.payload_bytes() < q.payload_bytes {
                import_guard
                    .resize_payload(q.payload_bytes)
                    .map_err(|_| Error::Bounds)?;
            }
            let created = create_inputs(
                slot_take(pending.original),
                manifest.payload(),
                &registry,
                Arc::clone(public),
                importing.charge.take().ok_or(Error::Retired)?,
            )?;
            let retained_original = Arc::clone(created.permit.retained_original()?);
            let scope = handles::SourceScope::new(
                &catalog,
                &registry,
                manifest.payload(),
                &retained_original,
                &catalog_charge,
            )?;
            // Boundary ends here: emit errors can still outlive this controller.
            let produced = emit(created, slot_take(pending.controller), &registry, &catalog, &scope)?;
            (
                produced,
                retained_original,
                catalog,
                registry.instance(),
                registry.binding(),
                catalog_charge,
            )
        }; // all Manifest/registry/catalog lenses end BEFORE their owners move.
        let custody = SourceCustody {
            directory: slot_take(pending.directory),
            produced,
            original: retained_original,
            public: outer,
            manifest,
            raw: slot_take(pending.raw),
            catalog,
            instance,
            binding,
            registry_charge: pending.registry,
            manifest_box_charge,
            catalog_charge,
            controller: (),
            root: pending.root,
        };
        custody.check_ingress(instance, binding)?;
        Ok(custody)
}

// One executor for owning and resident controllers. Only the inactive resident
// adapter selects detached completion; the legacy route still finishes owning.
fn emit_compact_epochs<'s, G: compact::ControllerAccounting>(
    mut scope: compact::EmissionScope<'s, G>,
    source: &handles::SourceScope<'s>,
    limits: original_source::Limits,
) -> Result<(
    compact::compact_history::PendingCompactHistory<'s, G>,
    compact_lowering::frozen_region::owning::OwnedEpoch,
), Error> {
    let first = {
        let (imports, context, _) = scope.parts();
        let mut graph = compact_lowering::CompactGraph::from_source_request(imports, context, MAX_PROGRAM_BYTES)?;
        for _ in &context.plan().plan().prefill.operations {
            graph.lower_next()?;
        }
        graph.freeze_owned(Vec::new(), source)?
    };
    let history = compact::compact_history::PendingCompactHistory::continue_original(scope, first, limits)?;
    let second = {
        let count = history.checked_graph_parts()?.1.plan().plan().prefill.operations.len();
        let mut graph = compact_lowering::CompactGraph::from_pending_history(&history)?;
        for _ in 0..count {
            graph.lower_next()?;
        }
        graph.freeze_owned(Vec::new(), source)?
    };
    Ok((history, second))
}

#[allow(dead_code)]
fn emit_resident_compact<'s>(
    resident: compact::ResidentEmission<'s, 's>,
    source: &'s handles::SourceScope<'s>,
    limits: original_source::Limits,
) -> Result<compact::owning::DetachedEmission<compact::compact_history::CompletionLoan>, Error> {
    let scope = compact::EmissionScope::from_resident(resident);
    let (history, second) = emit_compact_epochs(scope, source, limits)?;
    history.finish_detached(second)
}

// All dependent owners are installed before fallible preparation. Empty slots
// are transport, not admission or retry rights. Ordinary field drop retires
// payloads before covering guards, then controller, then root. Destructor
// panic/abort is outside this narrow ordering claim. New slot/work funding is
// unproved; existing quotes below do not establish that coverage.
struct ResidentStage<R> {
    produced: Option<R>,
    importing: ImportStage,
    original: Option<OriginalSourcePlan>,
    retained_original: Option<Arc<OriginalSourcePlan>>,
    public: Option<PublicSnapshotOwner>,
    manifest: Option<Box<OwnedCharged<Manifest>>>,
    unboxed_manifest: Option<OwnedCharged<Manifest>>,
    raw: Option<SourceIngressRaw>,
    directory: Option<Option<Arc<snapshot_directory::OwnedSnapshotDirectory>>>,
    catalog: Option<SourceCatalog>,
    registry_charge: Option<OwnedExternalCharge>,
    manifest_box_charge: Option<OwnedExternalCharge>,
    catalog_charge: Option<OwnedExternalCharge>,
    controller: Option<OwnedExternalCharge>,
    root: Option<OwnedBudget>,
}

// Source-logical inline components; not a complete frontier/admission quote.
fn resident_stage_bytes<R>() -> Result<usize, Error> {
    bytes::<ResidentStage<R>>(1)
}

fn callback_result_bytes<R>() -> Result<usize, Error> {
    bytes::<Result<R, Error>>(1)
}

// Logical shared-backing component for the initial Arc<OriginalSourcePlan> only;
// not ABI/RSS/allocator charge, payer/witness, F adoption or a complete frontier.
fn resident_original_plan_backing_bytes() -> Result<usize, Error> {
    add(
        bytes::<OriginalSourcePlan>(1)?,
        crate::original_source::arc_header_bytes::<OriginalSourcePlan>()
            .map_err(|_| Error::Bounds)?,
    )
}

fn consume_resident_transition<R>(
    pending: Pending,
    emit: impl for<'s> FnOnce(
        compact::ResidentEmission<'s, 's>,
        &'s SourceRegistry<'s>,
        &'s SourceCatalog,
        &'s handles::SourceScope<'s>,
    ) -> Result<R, Error>,
) -> Result<SourceCustody<R, OwnedExternalCharge>, Error> {
    let Pending {
        directory, importer, original, manifest, raw, limits: _, storage: _,
        registry, imports, controller, root,
    } = pending;
    let mut stage = ResidentStage {
        produced: None,
        importing: ImportStage { importer: Some(importer), charge: Some(imports) },
        original: Some(original), retained_original: None, public: None,
        manifest: None, unboxed_manifest: Some(manifest), raw: Some(raw),
        directory: Some(directory), catalog: None, registry_charge: Some(registry),
        manifest_box_charge: None, catalog_charge: None,
        controller: Some(controller), root: Some(root),
    };
    #[cfg(test)]
    PANIC_BEFORE_BOX.with(|flag| {
        if flag.replace(false) { panic!("ingress unwind before box reservation") }
    });
    let root = stage.root.as_ref().expect("resident root installed");
    let boxed_bytes = add(
        size_of::<OwnedCharged<Manifest>>(),
        2 * size_of::<Box<OwnedCharged<Manifest>>>(),
    )?;
    if add(root.budget().used_bytes(), boxed_bytes)? > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    stage.manifest_box_charge = Some(root.charge_external(boxed_bytes).map_err(|_| Error::Bounds)?);
    stage.manifest = Some(Box::new(stage.unboxed_manifest.take().expect("resident manifest installed")));
    stage.public = Some(PublicSnapshotOwner::new(
        stage.importing.importer.take().expect("resident importer installed")
            .finish().map_err(|_| Error::Binding)?,
        root,
    )?);
    let outer = stage.public.as_ref().expect("resident public installed");
    if !outer.charge.guard.is_for(root.budget()) { return Err(Error::Binding); }
    let context = add(
        compact::context_bytes()?,
        OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
    )?;
    let controller = stage.controller.as_mut().expect("resident controller installed");
    if controller.payload_bytes() < context {
        controller.resize_payload(context).map_err(|_| Error::Bounds)?;
    }
    let (instance, binding) = {
        let manifest = stage.manifest.as_ref().expect("resident manifest boxed");
        let public = &outer.public;
        let original = stage.original.as_ref().expect("resident original installed");
        let registry_guard = stage.registry_charge.as_mut().expect("resident registry charge installed");
        let owners = SourceConstructionOwners {
            manifest: manifest.payload(), incoming: None, public: Some(public),
            retained_baseline_bytes: root.budget().used_bytes()
                .checked_sub(registry_guard.payload_bytes()).ok_or(Error::Bounds)?,
        };
        let q = SourceRegistry::original_construction_quote(original, &owners, MAX_PROGRAM_BYTES)?;
        if registry_guard.payload_bytes() < q.new_owned_peak_bytes {
            registry_guard.resize_payload(q.new_owned_peak_bytes).map_err(|_| Error::Bounds)?;
        }
        let registry = SourceRegistry::from_original_checked(
            original, &owners, public, registry_guard, MAX_PROGRAM_BYTES,
        )?;
        if !std::ptr::eq(registry.source_records(), &manifest.payload().tensors)
            || !std::ptr::eq(registry.source_norms(), &manifest.payload().norms) {
            return Err(Error::Binding);
        }
        public.check_registry(&registry)?;
        let quote = SourceCatalog::quote_for::<R, OwnedExternalCharge>(
            &registry, manifest.payload(), original, controller,
        )?;
        let catalog_bytes = quote.bytes;
        if add(root.budget().used_bytes(), catalog_bytes)? > MAX_PROGRAM_BYTES { return Err(Error::Bounds); }
        stage.catalog_charge = Some(root.charge_external(catalog_bytes).map_err(|_| Error::Bounds)?);
        let catalog_charge = stage.catalog_charge.as_ref().expect("resident catalog charge installed");
        stage.catalog = Some(SourceCatalog::build(
            &registry, manifest.payload(), original, catalog_charge, quote,
        )?);
        let import_guard = stage.importing.charge.as_mut().expect("resident import charge installed");
        let q = quote_imports(original, manifest.payload(), &registry, public, import_guard)?;
        if import_guard.payload_bytes() < q.payload_bytes {
            import_guard.resize_payload(q.payload_bytes).map_err(|_| Error::Bounds)?;
        }
        // Preserve both declaration quotes and counter/size/debit/public order
        // inside prepare. A caller quote never substitutes for either check.
        let mut receiver = InputReceiver::new(
            stage.original.take().expect("resident original not moved"),
            Arc::clone(public),
            stage.importing.charge.take().expect("resident imports not moved"),
            compact::ControllerLoan::new(controller),
            MAX_PROGRAM_BYTES,
        );
        prepare_resident_inputs(&mut receiver, manifest.payload(), &registry)?;
        // Same immutable owners: repeated lens checks add no new source right.
        let mut sources = Some(request_origin::RequestSources::new(
            manifest.payload(), &registry, Arc::clone(public),
        )?);
        let mut destination = None;
        receiver.drain_complete(&mut sources, &mut destination)?;
        let resident = destination.take().expect("combined resident drain");
        stage.retained_original = Some(Arc::clone(resident.retained_original()?));
        // Drained does not mean destructor-free: retire remaining Arc fields and
        // empty slots before the source-scope borrow and callback.
        drop(receiver);
        drop(destination);
        drop(sources);
        let retained_original = stage.retained_original.as_ref().expect("resident original retained");
        let catalog = stage.catalog.as_ref().expect("resident catalog installed");
        let source = handles::SourceScope::new(
            catalog, &registry, manifest.payload(), retained_original, catalog_charge,
        )?;
        // The same opaque transport/RefCell reaches the callback. Its diagonal
        // lifetime is inferred at creation, never shortened through invariance.
        stage.produced = Some(emit(resident, &registry, catalog, &source)?);
        (registry.instance(), registry.binding())
    };
    // All scope/registry/controller loans end before custody extraction. Every
    // take below follows an installed-state invariant; no returned-error call,
    // allocation or live-payload destructor occurs until assembly is complete.
    let custody = SourceCustody {
        produced: stage.produced.take().expect("resident callback completed"),
        original: stage.retained_original.take().expect("resident original retained"),
        public: stage.public.take().expect("resident public retained"),
        manifest: stage.manifest.take().expect("resident manifest retained"),
        raw: stage.raw.take().expect("resident raw retained"),
        directory: stage.directory.take().expect("resident directory slot retained"),
        catalog: stage.catalog.take().expect("resident catalog retained"),
        instance, binding,
        registry_charge: stage.registry_charge.take().expect("resident registry charge retained"),
        manifest_box_charge: stage.manifest_box_charge.take().expect("resident box charge retained"),
        catalog_charge: stage.catalog_charge.take().expect("resident catalog charge retained"),
        controller: stage.controller.take().expect("resident SAME controller retained"),
        root: stage.root.take().expect("resident root retained"),
    };
    // Final fallible ingress check remains after complete custody assembly.
    custody.check_ingress(instance, binding)?;
    Ok(custody)
}
