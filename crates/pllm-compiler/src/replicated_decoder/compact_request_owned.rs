//! Ownership transfer of an existing consuming emission, never a second issuer.
//! This is unpublished construction custody, not finalized executable Storage.
use super::*;

/// Read-only identity retained from the consumed context. This cannot issue a
/// source/input/epoch; it only checks the owner accompanying already-built data.
pub(in crate::replicated_decoder::program) struct EpochAssociation {
    origin: Arc<OwnedCharged<Origin>>,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    prefix_allocation: usize,
}
impl EpochAssociation {
    pub(in crate::replicated_decoder::program) fn epoch(&self) -> usize {
        self.epoch
    }
    pub(in crate::replicated_decoder::program) fn check_context(
        &self,
        context: &EmissionContext<'_>,
    ) -> Result<(), Error> {
        context.check_open()?;
        if !Arc::ptr_eq(&self.origin, &context.origin)
            || !Arc::ptr_eq(&self.plan, &context.plan)
            || self.epoch != context.epoch
            || self.prefix_allocation != context.prefix_allocation
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}
impl EmissionContext<'_> {
    pub(in crate::replicated_decoder::program) fn epoch_association(
        &self,
    ) -> Result<EpochAssociation, Error> {
        self.check_open()?;
        Ok(EpochAssociation {
            origin: Arc::clone(&self.origin),
            plan: Arc::clone(&self.plan),
            epoch: self.epoch,
            prefix_allocation: self.prefix_allocation,
        })
    }
}

pub(in crate::replicated_decoder::program) struct InspectionAttempt<'a> {
    state: &'a AtomicU8,
    committed: bool,
}
impl InspectionAttempt<'_> {
    pub(in crate::replicated_decoder::program) fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for InspectionAttempt<'_> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self
                .state
                .compare_exchange(OPEN, BURNED, Ordering::AcqRel, Ordering::Acquire);
        }
    }
}

// Payloads precede all reservations. The original HistoryOrigin stays armed
// until real completion consumes it; dropping this staging owner burns history.
pub(in crate::replicated_decoder::program) struct OwnedEmission<R> {
    pub(in crate::replicated_decoder::program) produced: R,
    imports: Box<Builder>,
    inputs: BTreeMap<String, SourceView>,
    epochs: Vec<Arc<OriginalSourcePlan>>,
    fragments: Vec<Fragment>,
    projections: Vec<Option<rotary_creation::CreatedOperator>>,
    scratch: Words,
    origin: HistoryOrigin,
    public: Arc<PublicSource>,
    resources: Resources,
    sticky: Option<usize>,
    tokens: View,
    plan: Arc<OriginalSourcePlan>,
    epoch: usize,
    rows: usize,
    cap: usize,
    historical_peak: usize,
    work: WorkBudget,
    history_charge: OwnedExternalCharge,
    guard: OwnedExternalCharge,
    imports_charge: OwnedExternalCharge,
}

impl<R> OwnedEmission<R> {
    pub(in crate::replicated_decoder::program) fn original_request_shape(&self) -> (usize, usize) {
        (self.resources.rows, self.resources.outputs)
    }
    pub(in crate::replicated_decoder::program) fn checked_epoch_imports(
        &self,
        association: &EpochAssociation,
        epoch: usize,
    ) -> Result<&Builder, Error> {
        if association.epoch != epoch {
            return Err(Error::Binding);
        }
        self.checked_imports(association)
    }
    pub(in crate::replicated_decoder::program) fn prepay_import_metadata(
        &self,
        visits: usize,
    ) -> Result<(), Error> {
        self.check_open()?;
        self.plan
            .prepay_compiler_metadata(
                &self.guard,
                u64::try_from(visits).map_err(|_| Error::Bounds)?,
            )
            .map_err(|_| Error::Bounds)
    }
    /// ORIGINAL create_inputs map moved with its imports, not reconstructed from
    /// public digests. Owner lookup is bounded and rejects duplicate bindings.
    pub(in crate::replicated_decoder::program) fn checked_import_binding(
        &self,
        owner: usize,
    ) -> Result<(&str, &[usize]), Error> {
        self.check_open()?;
        self.prepay_import_metadata(self.inputs.len().checked_add(1).ok_or(Error::Bounds)?)?;
        let mut bindings = self.inputs.iter().filter(|(_, view)| view.owner == owner);
        let (key, view) = bindings.next().ok_or(Error::Binding)?;
        if bindings.next().is_some() {
            return Err(Error::Binding);
        }
        Ok((key.as_str(), view.shape.as_slice()))
    }
    pub(in crate::replicated_decoder::program) fn checked_imports(
        &self,
        association: &EpochAssociation,
    ) -> Result<&Builder, Error> {
        self.check_open()?;
        let origin = self.origin.payload();
        if !Arc::ptr_eq(&association.origin, &self.origin.0)
            || self
                .epochs
                .get(association.epoch)
                .is_none_or(|plan| !Arc::ptr_eq(&association.plan, plan))
            || association.prefix_allocation != self.imports.nodes.as_ptr() as usize
            || !self.plan.same_origin(&origin.source)
            || !self
                .imports
                .original_inputs
                .iter()
                .any(|input| Arc::ptr_eq(input, &origin.input))
        {
            return Err(Error::Binding);
        }
        let input = self
            .imports
            .nodes
            .get(origin.input.owner())
            .ok_or(Error::Binding)?;
        if !matches!(&input.code,Code::Input(key) if key=="input.tokens")
            || input.lanes != self.resources.rows
            || input.bounds != origin.bounds
            || self.imports.nodes.len()
                != origin.input.owner().checked_add(1).ok_or(Error::Bounds)?
        {
            return Err(Error::Binding);
        }
        Ok(&self.imports)
    }
    pub(in crate::replicated_decoder::program) fn check_source_custody(
        &self,
        manifest: &Manifest,
        public: &Arc<PublicSource>,
        instance: source_registry::Instance,
    ) -> Result<(), Error> {
        self.check_open()?;
        let origin = self.origin.payload();
        if origin.registry_identity != instance
            || !Arc::ptr_eq(public, &self.public)
            || origin.source_records != std::ptr::from_ref(&manifest.tensors) as usize
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn inspection_attempt(
        &self,
    ) -> Result<InspectionAttempt<'_>, Error> {
        self.check_open()?;
        Ok(InspectionAttempt {
            state: &self.origin.payload().state,
            committed: false,
        })
    }
    /// New inspection scratch only, on the SAME counter. Existing native and
    /// source reservations never move/refund. Reader stores payload before lease.
    pub(in crate::replicated_decoder::program) fn charge_inspection(
        &self,
        bytes: usize,
        visits: u64,
    ) -> Result<pllm_garble::shared_allocation::ExternalCharge<'_>, Error> {
        let mut attempt = self.inspection_attempt()?;
        let total = self
            .plan
            .budget()
            .used_bytes()
            .checked_add(bytes)
            .and_then(|n| {
                n.checked_add(size_of::<pllm_garble::shared_allocation::ExternalCharge<'_>>())
            })
            .ok_or(Error::Bounds)?;
        if total > self.cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        self.plan
            .prepay_compiler_metadata(&self.guard, visits)
            .map_err(|_| Error::Bounds)?;
        let charge = self
            .plan
            .budget()
            .charge_external(bytes)
            .map_err(|_| Error::Bounds)?;
        attempt.commit();
        Ok(charge)
    }
    pub(in crate::replicated_decoder::program) fn source_plan(&self) -> &OriginalSourcePlan {
        &self.plan
    }
    pub(in crate::replicated_decoder::program) fn work(&self) -> WorkBudget {
        self.work
    }
    pub(in crate::replicated_decoder::program) fn check_open(&self) -> Result<(), Error> {
        if self.origin.payload().state.load(Ordering::Acquire) != OPEN {
            return Err(Error::Binding);
        }
        if !Arc::ptr_eq(&self.public, &self.origin.payload().public)
            || self
                .epochs
                .get(self.epoch)
                .is_none_or(|plan| !Arc::ptr_eq(plan, &self.plan))
            || !self.guard.is_for(self.plan.budget())
            || !self.history_charge.is_for(self.plan.budget())
            || !self.imports_charge.is_for(self.plan.budget())
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
}

impl OwnedEmission<super::compact_history::CompletionLoan> {
    pub(in crate::replicated_decoder::program) fn prepay_publication(
        &self,
        attempt: &schedule::Attempt,
        visits: u64,
    ) -> Result<(), Error> {
        self.check_completed_attempt(attempt)?;
        self.plan
            .prepay_compiler_metadata(&self.guard, visits)
            .map_err(|_| Error::Bounds)
    }
    pub(in crate::replicated_decoder::program) fn check_completed_attempt(
        &self,
        attempt: &schedule::Attempt,
    ) -> Result<(), Error> {
        if !Arc::ptr_eq(&attempt.origin, &self.origin.0) {
            return Err(Error::Binding);
        }
        self.completed_imports()?;
        Ok(())
    }
    // Only the canonical-finalized storage module calls this retained-root lens.
    // It does not reopen the generic inspection/charging entry.
    pub(in crate::replicated_decoder::program) fn completed_imports(
        &self,
    ) -> Result<&Builder, Error> {
        if self.origin.payload().state.load(Ordering::Acquire) != COMPLETE
            || !Arc::ptr_eq(&self.public, &self.origin.payload().public)
            || self
                .epochs
                .get(self.epoch)
                .is_none_or(|plan| !Arc::ptr_eq(plan, &self.plan))
            || !self.guard.is_for(self.plan.budget())
            || !self.imports_charge.is_for(self.plan.budget())
            || !self.history_charge.is_for(self.plan.budget())
        {
            return Err(Error::Binding);
        }
        Ok(&self.imports)
    }
    pub(in crate::replicated_decoder::program) fn checked_epoch_imports_for_finish(
        &self,
    ) -> Result<usize, Error> {
        self.check_open()?;
        for (index, epoch) in self.produced.epochs().iter().enumerate() {
            epoch.check_owned_emission(self, index, self.origin.payload().registry_identity)?;
        }
        Ok(self.imports.nodes.len())
    }
    pub(in crate::replicated_decoder::program) fn finish_lease(
        &mut self,
    ) -> Result<OwnedExternalCharge, Error> {
        self.check_open()?;
        self.guard.split_owned(0).map_err(|_| Error::Bounds)
    }
    pub(in crate::replicated_decoder::program) fn program_cap(&self) -> usize {
        self.cap.min(MAX_PROGRAM_BYTES)
    }
    /// Validate authentic pending history and retain its original finish Attempt.
    /// OPEN-only preparation finishes before the original COMPLETE transition.
    pub(in crate::replicated_decoder::program) fn prepare_compact_history(
        &mut self,
        control_bytes: usize,
    ) -> Result<schedule::Attempt, Error> {
        self.check_open()?;
        let visits = self
            .produced
            .epochs()
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_add(self.tokens.words.len()))
            .ok_or(Error::Bounds)?;
        self.prepay_import_metadata(visits)?;
        if self.epoch != 1
            || self.rows != self.resources.rows.checked_add(1).ok_or(Error::Bounds)?
            || self.resources.outputs != 2
            || self.epochs.len() != 2
            || self.produced.epochs().len() != 2
            || self.tokens.words.len() != self.rows
            || self.tokens.shape.as_slice() != [1, self.rows]
            || self.imports.request_emission.is_some()
            || self.imports.checked_bound.is_none()
        {
            return Err(Error::Binding);
        }
        for (index, epoch) in self.produced.epochs().iter().enumerate() {
            epoch.check_owned_emission(self, index, self.origin.payload().registry_identity)?;
        }
        let selected = self.produced.selected_words();
        for (lane, word) in self.tokens.words.iter().enumerate() {
            let expected = if lane < self.resources.rows {
                (self.origin.payload().input.owner(), lane)
            } else {
                selected[0]
            };
            if word != expected {
                return Err(Error::Binding);
            }
        }
        self.produced
            .check_final_sticky(self.sticky, self.origin.payload().input.owner())?;
        if self.produced.epochs()[1]
            .work()
            .checked_add(WorkBudget::product(2, false)?)?
            != self.work
        {
            return Err(Error::Binding);
        }
        let budget = self.plan.budget();
        let outside = budget
            .used_bytes()
            .checked_sub(self.guard.payload_bytes())
            .ok_or(Error::Binding)?;
        let payload = self
            .guard
            .payload_bytes()
            .checked_add(control_bytes)
            .ok_or(Error::Bounds)?;
        if outside
            .checked_add(payload)
            .is_none_or(|total| total > self.cap.min(MAX_PROGRAM_BYTES))
        {
            return Err(Error::Bounds);
        }
        self.guard
            .resize_payload(payload)
            .map_err(|_| Error::Bounds)?;
        let attempt = schedule::Attempt {
            origin: Arc::clone(&self.origin.0),
            committed: false,
        };
        Ok(attempt)
    }
    pub(in crate::replicated_decoder::program) fn complete_compact_history(
        &self,
        attempt: &schedule::Attempt,
    ) -> Result<(), Error> {
        self.check_open()?;
        if !Arc::ptr_eq(&attempt.origin, &self.origin.0) || attempt.committed {
            return Err(Error::Binding);
        }
        self.origin
            .payload()
            .state
            .compare_exchange(OPEN, COMPLETE, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Binding)?;
        Ok(())
    }
}

impl EmissionScope<'_> {
    /// The HRTB requires a genuinely borrow-free result. Conversion happens while
    /// actual imports/context are alive; only afterwards move their SAME owners.
    /// No request completion, new permit, native ledger, or execution is issued.
    pub(in crate::replicated_decoder::program) fn into_owned<R>(
        self,
        convert: impl for<'a> FnOnce(
            &'a Builder,
            &'a EmissionContext<'a>,
            &'a BTreeMap<String, SourceView>,
        ) -> Result<R, Error>,
    ) -> Result<OwnedEmission<R>, Error> {
        self.context.check_imports(&self.builder)?;
        if !Arc::ptr_eq(
            &self.context.public,
            &self.context.request.history.sources.public,
        ) {
            return Err(Error::Binding);
        }
        let overlap = add(
            size_of::<OwnedEmission<R>>()
                .checked_mul(2)
                .ok_or(Error::Bounds)?,
            add(
                size_of::<Builder>(),
                size_of::<Box<Builder>>()
                    .checked_mul(2)
                    .ok_or(Error::Bounds)?,
            )?,
        )?;
        self.context.admit_phase(overlap, 0)?;
        // Keep old/new wrapper and boxed-import destination reserved across all
        // nested conversion factories. No scratch credit/refund during transfer.
        self.context.retain(overlap)?;
        let produced = convert(&self.builder, &self.context, &self.inputs)?;
        self.context.check_imports(&self.builder)?;
        let EmissionScope {
            builder,
            inputs,
            context,
            imports_charge,
        } = self;
        let imports = Box::new(builder);
        let EmissionContext {
            manifest: _,
            registry: _,
            public,
            plan,
            origin: _,
            epoch,
            rows,
            prefix_allocation: _,
            historical_peak,
            retained_payload: _,
            registered_builder: _,
            cap,
            history_input: _,
            work,
            request,
            guard,
        } = context;
        let RequestInputPermit { history, tokens } = request;
        let History {
            epochs,
            fragments,
            projections,
            scratch,
            origin,
            sources:
                RequestSources {
                    manifest: _,
                    registry: _,
                    public: history_public,
                },
            resources,
            sticky,
            charge: history_charge,
        } = history;
        // Both references originated from this consumed history, never a digest.
        drop(history_public);
        let owned = OwnedEmission {
            produced,
            imports,
            inputs,
            epochs,
            fragments,
            projections,
            scratch,
            origin,
            public,
            resources,
            sticky,
            tokens,
            plan,
            epoch,
            rows,
            cap,
            historical_peak: historical_peak.into_inner(),
            work: work.into_inner(),
            history_charge,
            guard: guard.into_inner(),
            imports_charge,
        };
        owned.check_open()?;
        Ok(owned)
    }
}
