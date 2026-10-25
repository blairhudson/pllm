//! Source-request storage issuance boundary. No expanded executable, recovered
//! seals, digest witness, partial region, or runtime handoff can issue Storage.
//! Construction is admitted phase by phase on the ORIGINAL counter. Historical
//! admission is not current retention, RSS, source authority, or execution.
use super::*;
#[path = "compact_request_history.rs"]
pub(in crate::replicated_decoder::program) mod compact_history;
use std::cell::{Cell, RefCell};
#[path = "compact_request_owned.rs"]
pub(in crate::replicated_decoder::program) mod owning;

/// Lexical source context shared by future compact and RMS emission. It owns
/// actual origin/plan/public-source roots and a same-counter reservation; source
/// descriptor borrows come from the genuine permit, never from Builder caches.
pub(in crate::replicated_decoder) struct EmissionContext<'s> {
    manifest: &'s Manifest,
    registry: &'s SourceRegistry<'s>,
    public: Arc<PublicSource>,
    plan: Arc<OriginalSourcePlan>,
    origin: Arc<OwnedCharged<Origin>>,
    epoch: usize,
    rows: usize,
    prefix_allocation: usize,
    historical_peak: Cell<usize>,
    retained_payload: Cell<usize>,
    registered_builder: Cell<usize>,
    cap: usize,
    history_input: Option<compact_history::HistoryInput>,
    work: RefCell<WorkBudget>,
    // The ONLY token permit stays consumed here. No &permit -> fresh ledger.
    request: RequestInputPermit<'s>,
    guard: RefCell<OwnedExternalCharge>,
}

pub(in crate::replicated_decoder) struct EmissionAttempt<'c, 's> {
    context: &'c EmissionContext<'s>,
    committed: bool,
}
impl EmissionAttempt<'_, '_> {
    pub(in crate::replicated_decoder) fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for EmissionAttempt<'_, '_> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.context.origin.payload().state.compare_exchange(
                OPEN,
                BURNED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

impl EmissionContext<'_> {
    pub(in crate::replicated_decoder) fn attempt(&self) -> EmissionAttempt<'_, '_> {
        EmissionAttempt {
            context: self,
            committed: false,
        }
    }
    #[cfg(test)]
    pub(super) fn test_origin(&self) -> Arc<OwnedCharged<Origin>> {
        Arc::clone(&self.origin)
    }
    pub(in crate::replicated_decoder) fn check_open(&self) -> Result<(), Error> {
        if self.origin.payload().state.load(Ordering::Acquire) == OPEN {
            Ok(())
        } else {
            Err(Error::Binding)
        }
    }
    pub(in crate::replicated_decoder::program) fn check_imports(
        &self,
        imports: &Builder,
    ) -> Result<(), Error> {
        self.check_open()?;
        if imports.nodes.as_ptr() as usize != self.prefix_allocation
            || imports
                .original_inputs
                .iter()
                .all(|record| !Arc::ptr_eq(record, &self.origin.payload().input))
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn check_active(
        &self,
        active: &Active,
    ) -> Result<(), Error> {
        self.check_open()?;
        if !Arc::ptr_eq(&self.origin, &active.origin)
            || !Arc::ptr_eq(&self.plan, &active.plan)
            || self.epoch != active.epoch
            || active.registry_identity != self.registry.instance()
            || active.source_records != std::ptr::from_ref(self.registry.source_records()) as usize
            || self.registry.instance() != self.origin.payload().registry_identity
            || std::ptr::from_ref(self.registry.source_records()) as usize
                != self.origin.payload().source_records
            || !Arc::ptr_eq(&self.public, &self.origin.payload().public)
            || !self.plan.same_origin(&self.origin.payload().source)
            || !std::ptr::eq(self.plan.budget(), self.origin.payload().source.budget())
            || !self.guard.borrow().is_for(self.plan.budget())
            || self
                .request
                .history
                .epochs
                .get(self.epoch)
                .is_none_or(|plan| !Arc::ptr_eq(plan, &self.plan))
        {
            return Err(Error::Binding);
        }
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn initial_input(&self) -> (usize, usize, Interval) {
        (
            self.origin.payload().input.owner(),
            self.rows,
            self.origin.payload().bounds,
        )
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn token_words_for_tests(&self) -> &Words {
        &self.request.tokens.words
    }
    pub(in crate::replicated_decoder::program) fn prelude(&self) -> Result<Prelude, Error> {
        Ok(Prelude {
            origin: Arc::clone(&self.origin),
            first: self.history_input.map_or_else(
                || add(self.origin.payload().input.owner(), 1),
                |history| Ok(history.first()),
            )?,
        })
    }
    pub(in crate::replicated_decoder::program) fn manifest(&self) -> &Manifest {
        self.manifest
    }
    pub(in crate::replicated_decoder::program) fn registry(&self) -> &SourceRegistry<'_> {
        self.registry
    }
    pub(in crate::replicated_decoder::program) fn public(&self) -> &Arc<PublicSource> {
        &self.public
    }
    pub(in crate::replicated_decoder::program) fn plan(&self) -> &Arc<OriginalSourcePlan> {
        &self.plan
    }
    pub(in crate::replicated_decoder::program) fn reservation(
        &self,
    ) -> std::cell::Ref<'_, OwnedExternalCharge> {
        self.guard.borrow()
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_retained_payload(&self) -> usize {
        self.retained_payload.get()
    }
    pub(in crate::replicated_decoder::program) fn historical_peak(&self) -> usize {
        self.historical_peak
            .get()
            .max(self.plan.budget().admitted_peak_bytes())
    }

    /// All arguments describe NEW, named simultaneously live storage above this
    /// context's retained payload. Other same-counter guards stay in `outside`;
    /// no digest credit or independently measured maxima are combined here.
    pub(in crate::replicated_decoder) fn admit_phase(
        &self,
        new_live: usize,
        uncovered: usize,
    ) -> Result<(), Error> {
        let mut attempt = self.attempt();
        self.check_open()?;
        let result = self.admit_phase_inner(new_live, uncovered);
        if result.is_ok() {
            attempt.commit();
        }
        result
    }
    fn admit_phase_inner(&self, new_live: usize, uncovered: usize) -> Result<(), Error> {
        let mut guard = self.guard.borrow_mut();
        let outside = self
            .plan
            .budget()
            .used_bytes()
            .checked_sub(guard.payload_bytes())
            .and_then(|n| n.checked_add(uncovered))
            .ok_or(Error::Bounds)?;
        // Admission never refunds an older live phase. Only retire_phase, at an
        // explicit post-publication boundary, can shrink this reservation.
        let payload = add(self.retained_payload.get(), new_live)?.max(guard.payload_bytes());
        let total = add(outside, payload)?;
        if total > self.cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        // Check the complete total BEFORE changing reservation or entering a
        // factory. Successful guard growth is itself recorded by this counter.
        guard.resize_payload(payload).map_err(|_| Error::Bounds)?;
        self.plan
            .budget()
            .admit_peak(total)
            .map_err(|_| Error::Bounds)?;
        self.historical_peak
            .set(self.historical_peak.get().max(total));
        Ok(())
    }

    /// Call only AFTER dead construction scratch/results have left scope. The
    /// actual recipe/native payload stays covered by this guard through freeze.
    pub(in crate::replicated_decoder) fn retain(&self, new_retained: usize) -> Result<(), Error> {
        let retained = add(self.retained_payload.get(), new_retained)?;
        let guard = self.guard.borrow();
        if retained > guard.payload_bytes() {
            return Err(Error::Binding);
        }
        self.retained_payload.set(retained);
        Ok(())
    }

    /// CURRENT Builder census replaces only its own prior attribution. Compact
    /// recipe ownership remains additive. No still-live guard is refunded here.
    pub(in crate::replicated_decoder::program) fn retain_builder(
        &self,
        bytes: usize,
    ) -> Result<(), Error> {
        let base = self
            .retained_payload
            .get()
            .checked_sub(self.registered_builder.get())
            .ok_or(Error::Binding)?;
        let retained = add(base, bytes)?;
        if retained > self.guard.borrow().payload_bytes() {
            return Err(Error::Binding);
        }
        self.retained_payload.set(retained);
        self.registered_builder.set(bytes);
        Ok(())
    }

    pub(in crate::replicated_decoder::program) fn retire_phase(&self) -> Result<(), Error> {
        self.guard
            .borrow_mut()
            .resize_payload(self.retained_payload.get())
            .map_err(|_| Error::Bounds)
    }

    pub(in crate::replicated_decoder) fn admit_work<'p>(
        &self,
        ticket: tensor::WorkPreflight<'p>,
    ) -> Result<tensor::AdmittedWorkSchedule<'p>, Error> {
        let mut attempt = self.attempt();
        self.check_open()?;
        let mut total = self.work.borrow_mut();
        let result = tensor::CallerWorkLedger::new(
            &mut total,
            puma::MAX_PROGRAM_OPERATIONS,
            puma::prss::MAX_PROGRAM_AES_BLOCKS,
        )
        .and_then(|mut ledger| ledger.admit(ticket))
        .map_err(|_| Error::Bounds);
        if result.is_ok() {
            attempt.commit();
        }
        result
    }
    pub(in crate::replicated_decoder::program) fn work(&self) -> WorkBudget {
        *self.work.borrow()
    }
    /// Original scalar multiplication schedule, not a native factory query.
    /// Sticky conjunctions consume this once BEFORE any checked-row factory.
    pub(in crate::replicated_decoder::program) fn admit_products(
        &self,
        count: usize,
    ) -> Result<(), Error> {
        let mut attempt = self.attempt();
        self.check_open()?;
        let work = WorkBudget::product(1, false)?
            .checked_mul(u64::try_from(count).map_err(|_| Error::Bounds)?)?;
        self.admit_protocol_work(work)?;
        attempt.commit();
        Ok(())
    }

    /// Actual ordinary primitive summary from the single original composition
    /// driver. Not native-ticket substitution or reconstructed prior debt.
    pub(in crate::replicated_decoder::program) fn admit_protocol_work(
        &self,
        work: WorkBudget,
    ) -> Result<(), Error> {
        let mut attempt = self.attempt();
        self.check_open()?;
        let mut ledger = self.work.borrow_mut();
        let next = ledger.checked_add(work)?;
        next.limits()?;
        *ledger = next;
        attempt.commit();
        Ok(())
    }
    pub(in crate::replicated_decoder::program) fn check_final_work(
        &self,
        recomputed: WorkBudget,
    ) -> Result<(), Error> {
        // Equality only: finalization must never debit a second time.
        if recomputed == self.work() {
            Ok(())
        } else {
            Err(Error::Binding)
        }
    }
}

/// Original graph Begin's gather(first token)/constant-zero recipes. Born only
/// from the sealed original token context; never from coordinates or a Program.
pub(in crate::replicated_decoder::program) struct Prelude {
    origin: Arc<OwnedCharged<Origin>>,
    first: usize,
}
impl Prelude {
    pub(in crate::replicated_decoder::program) fn first(&self) -> usize {
        self.first
    }
    pub(in crate::replicated_decoder::program) fn node(
        &self,
        ordinal: usize,
    ) -> Result<schedule_finalization::BorrowedNode<'_>, Error> {
        use schedule_finalization::{BorrowedNode, BorrowedRecipe};
        let (recipe, bounds) = if ordinal == self.first {
            (BorrowedRecipe::Gather(self), self.origin.payload().bounds)
        } else if ordinal == add(self.first, 1)? {
            (
                BorrowedRecipe::Affine {
                    left: self.first,
                    right: None,
                    coefficients: affine(0, 0, 0),
                },
                Interval::new(0, 0)?,
            )
        } else {
            return Err(Error::Binding);
        };
        Ok(BorrowedNode {
            lanes: 1,
            bounds,
            recipe,
        })
    }
}
impl schedule_finalization::BorrowedLaneRefs for Prelude {
    fn len(&self) -> usize {
        1
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index == 0 {
            Ok((self.origin.payload().input.owner(), 0))
        } else {
            Err(Error::Shape)
        }
    }
}

pub(in crate::replicated_decoder::program) const fn context_bytes() -> usize {
    // Source context local/result/move slots; Arc clones reuse actual backing.
    3 * size_of::<EmissionContext<'static>>()
        + size_of::<Result<EmissionContext<'static>, Error>>()
        + size_of::<OwnedExternalCharge>()
        + 2 * size_of::<Result<compact_lowering::PublishedView, Error>>()
        + size_of::<BTreeMap<(u64, pllm_models::StateKind), compact_lowering::PublishedView>>()
        + 2 * size_of::<graph::DispatchStep<'_, '_, compact_lowering::PublishedView>>()
}

impl EmissionContext<'_> {
    /// Only the prequoted original source/input production path supplies the
    /// live import guard. The permit is consumed and held until final custody.
    fn check_initial(
        &self,
        imports: &Builder,
        imports_charge: &OwnedExternalCharge,
    ) -> Result<(), Error> {
        let history = &self.request.history;
        let guard = self.guard.borrow();
        let plan = history.epochs.first().ok_or(Error::Binding)?;
        if history.origin.payload().state.load(Ordering::Acquire) != OPEN
            || history.epochs.len() != 1
            || !history.fragments.is_empty()
            || !Arc::ptr_eq(plan, &history.origin.payload().source)
            || !Arc::ptr_eq(&history.sources.public, &history.origin.payload().public)
            || history.sources.registry.instance() != history.origin.payload().registry_identity
            || std::ptr::from_ref(history.sources.registry.source_records()) as usize
                != history.origin.payload().source_records
            || !guard.is_for(plan.budget())
            || !imports_charge.is_for(plan.budget())
        {
            return Err(Error::Binding);
        }
        if guard.payload_bytes()
            < add(
                context_bytes(),
                OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
            )?
        {
            return Err(Error::Bounds);
        }
        history.check_tokens(&self.request.tokens)?;
        let owner = history.origin.payload().input.owner();
        if owner.checked_add(1) != Some(imports.nodes.len())
            || imports
                .nodes
                .iter()
                .any(|n| !matches!(n.code, Code::Input(_) | Code::PublicInput { .. }))
            || imports
                .original_inputs
                .iter()
                .all(|record| !Arc::ptr_eq(record, &history.origin.payload().input))
        {
            return Err(Error::Binding);
        }
        history
            .sources
            .public
            .check_registry(history.sources.registry)?;
        for (id, node) in imports.nodes.iter().enumerate() {
            match &node.code {
                Code::PublicInput { key, digest, .. } => history
                    .sources
                    .registry
                    .check_descriptor(id, key, *digest, node.lanes, node.bounds)?,
                Code::Input(key)
                    if id == owner
                        && key == "input.tokens"
                        && node.lanes == history.resources.rows
                        && node.bounds == history.origin.payload().bounds => {}
                _ => return Err(Error::Binding),
            }
        }
        let historical_peak = plan.budget().admitted_peak_bytes();
        if historical_peak > self.cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        plan.prepay_compiler_metadata(
            &guard,
            u64::try_from(imports.nodes.len()).map_err(|_| Error::Bounds)?,
        )
        .map_err(|_| Error::Bounds)?;
        Ok(())
    }
}

/// Private drop order is important: ALL emitted payloads/maps precede context
/// and import charges. Scoped borrows never expose a second token permit.
pub(in crate::replicated_decoder::program) struct EmissionScope<'s> {
    builder: Builder,
    inputs: BTreeMap<String, SourceView>,
    context: EmissionContext<'s>,
    imports_charge: OwnedExternalCharge,
}
impl<'s> EmissionScope<'s> {
    pub(in crate::replicated_decoder::program) fn new(
        created: source_entry::CreatedInputs<'s>,
        guard: OwnedExternalCharge,
        cap: usize,
    ) -> Result<Self, Error> {
        let required = add(
            context_bytes(),
            OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
        )?;
        if guard.payload_bytes() < required {
            return Err(Error::Bounds);
        }
        let source_entry::CreatedInputs {
            builder,
            inputs,
            permit,
            charge,
            ..
        } = created;
        let history = &permit.history;
        let context = EmissionContext {
            manifest: history.sources.manifest,
            registry: history.sources.registry,
            public: Arc::clone(&history.sources.public),
            plan: Arc::clone(&history.origin.payload().source),
            origin: Arc::clone(&history.origin),
            epoch: 0,
            rows: history.resources.rows,
            prefix_allocation: builder.nodes.as_ptr() as usize,
            historical_peak: Cell::new(
                history
                    .origin
                    .payload()
                    .source
                    .budget()
                    .admitted_peak_bytes(),
            ),
            retained_payload: Cell::new(guard.payload_bytes()),
            registered_builder: Cell::new(0),
            cap,
            history_input: None,
            work: RefCell::new(WorkBudget::default()),
            request: permit,
            guard: RefCell::new(guard),
        };
        let scope = Self {
            builder,
            inputs,
            context,
            imports_charge: charge,
        };
        scope
            .context
            .check_initial(&scope.builder, &scope.imports_charge)?;
        Ok(scope)
    }
    pub(in crate::replicated_decoder::program) fn parts(
        &mut self,
    ) -> (
        &mut Builder,
        &EmissionContext<'s>,
        &BTreeMap<String, SourceView>,
    ) {
        (&mut self.builder, &self.context, &self.inputs)
    }
}
