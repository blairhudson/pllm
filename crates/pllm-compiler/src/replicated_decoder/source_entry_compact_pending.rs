//! Consuming lexical conversion boundary. This owns authentic ingress backing,
//! not an executable schedule. Only the complete storage finalizer may publish.
use super::*;

#[path = "source_entry_compact_handles.rs"]
pub(in crate::replicated_decoder::program) mod handles;
#[path = "compact_request_sealed.rs"]
pub(in crate::replicated_decoder::program) mod sealed;
use handles::SourceCatalog;

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

/// Payloads precede reservations in field drop order. No borrowed registry,
/// FrozenRegion or self-referential pointer is stored beside its referents.
// Ingress custody, not executable Storage; no production route activates it yet.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::replicated_decoder::program) struct SourceCustody<R> {
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
    root: OwnedBudget,
}

impl<R> SourceCustody<R> {
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
        self.catalog.check_owned(
            self.manifest.payload(),
            &self.original,
            &self.catalog_charge,
        )
    }
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
        request_origin::schedule::canonical_prefix(
            &self.original,
            rows,
            outputs,
            self.manifest.payload(),
        )
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
            let mut scope=request_origin::compact_storage::EmissionScope::new(created,guard,MAX_PROGRAM_BYTES)?;
            let first={
                let(imports,context,_)=scope.parts();
                let mut graph=compact_lowering::CompactGraph::from_source_request(imports,context,MAX_PROGRAM_BYTES)?;
                for _ in &context.plan().plan().prefill.operations{graph.lower_next()?;}
                graph.freeze_owned(Vec::new(),source)?
            };
            let history=request_origin::compact_storage::compact_history::PendingCompactHistory::continue_original(scope,first,limits)?;
            let second={
                let count=history.checked_graph_parts()?.1.plan().plan().prefill.operations.len();
                let mut graph=compact_lowering::CompactGraph::from_pending_history(&history)?;
                for _ in 0..count{graph.lower_next()?;}
                graph.freeze_owned(Vec::new(),source)?
            };
            history.finish(second)
        })?;
        custody.seal_compact()
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
        let Pending {
            directory,
            importer,
            original,
            manifest,
            raw,
            limits: _,
            storage: _,
            registry: mut registry_guard,
            imports: import_guard,
            mut controller,
            root,
        } = self.0.take().ok_or(Error::Retired)?;
        let mut importing = ImportStage {
            importer: Some(importer),
            charge: Some(import_guard),
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
        if add(root.budget().used_bytes(), boxed_bytes)? > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let manifest_box_charge = root
            .charge_external(boxed_bytes)
            .map_err(|_| Error::Bounds)?;
        let manifest = Box::new(manifest);
        let outer = PublicSnapshotOwner::new(
            importing
                .importer
                .take()
                .ok_or(Error::Retired)?
                .finish()
                .map_err(|_| Error::Binding)?,
            &root,
        )?;
        if !outer.charge.guard.is_for(root.budget()) {
            return Err(Error::Binding);
        }
        let context = add(
            request_origin::compact_storage::context_bytes(),
            OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
        )?;
        if controller.payload_bytes() < context {
            controller
                .resize_payload(context)
                .map_err(|_| Error::Bounds)?;
        }
        let (produced, retained_original, catalog, instance, binding, catalog_charge) = {
            let public = &outer.public;
            let owners = SourceConstructionOwners {
                manifest: manifest.payload(),
                incoming: None,
                public: Some(public),
                retained_baseline_bytes: root
                    .budget()
                    .used_bytes()
                    .checked_sub(registry_guard.payload_bytes())
                    .ok_or(Error::Bounds)?,
            };
            let q =
                SourceRegistry::original_construction_quote(&original, &owners, MAX_PROGRAM_BYTES)?;
            if registry_guard.payload_bytes() < q.new_owned_peak_bytes {
                registry_guard
                    .resize_payload(q.new_owned_peak_bytes)
                    .map_err(|_| Error::Bounds)?;
            }
            let registry = SourceRegistry::from_original_checked(
                &original,
                &owners,
                public,
                &registry_guard,
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
                SourceCatalog::quote::<R>(&registry, manifest.payload(), &original, &controller)?;
            if add(root.budget().used_bytes(), quote.bytes)? > MAX_PROGRAM_BYTES {
                return Err(Error::Bounds);
            }
            let catalog_charge = root
                .charge_external(quote.bytes)
                .map_err(|_| Error::Bounds)?;
            let catalog = SourceCatalog::build(
                &registry,
                manifest.payload(),
                &original,
                &catalog_charge,
                quote,
            )?;
            let import_guard = importing.charge.as_mut().ok_or(Error::Retired)?;
            let q = quote_imports(
                &original,
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
                original,
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
            let produced = emit(created, controller, &registry, &catalog, &scope)?;
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
            directory,
            produced,
            original: retained_original,
            public: outer,
            manifest,
            raw,
            catalog,
            instance,
            binding,
            registry_charge: registry_guard,
            manifest_box_charge,
            catalog_charge,
            root,
        };
        custody.check_ingress(instance, binding)?;
        Ok(custody)
    }
}
