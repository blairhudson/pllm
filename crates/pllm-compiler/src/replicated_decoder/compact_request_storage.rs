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

// Only the installed context can construct these operation tickets. Borrowers
// cannot manufacture a retirement target, resize amount, or split destination.
pub(in crate::replicated_decoder) struct ContextPhase<'a> {
    plan: &'a OriginalSourcePlan,
    setup: &'a ResidentSetup,
    new_live: usize,
    uncovered: usize,
}
pub(in crate::replicated_decoder) struct ContextRetirement<'a> {
    setup: &'a ResidentSetup,
}
mod controller_accounting {
    use super::*;
    mod sealed {
        pub trait Sealed {}
    }
    /// Accounting only: no source issuance, guard extraction, general split,
    /// mutable guard callback, Deref, BorrowMut, or blanket implementation.
    pub(in crate::replicated_decoder) trait ControllerAccounting: sealed::Sealed {
        fn controller_identity(&self) -> usize;
        fn is_for(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool;
        fn payload_bytes(&self) -> usize;
        fn charged_bytes(&self) -> usize;
        fn debit_metadata(&self, plan: &OriginalSourcePlan, visits: u64) -> Result<(), Error>;
        fn debit_primitive(&self, plan: &OriginalSourcePlan, work: u64) -> Result<(), Error>;
        fn admit_context_phase(&mut self, phase: ContextPhase<'_>) -> Result<(), Error>;
        fn retire_context_phase(&mut self, phase: ContextRetirement<'_>) -> Result<(), Error>;
        fn continue_checked_history(&mut self, transfer: compact_history::HistoryBootstrap<'_, '_>) -> Result<(), Error>;
    }
    /// Exclusive lexical loan of the already retained controller. No owned
    /// guard, clone, mutable guard view, or independent release is obtainable.
    pub(in crate::replicated_decoder::program) struct ControllerLoan<'a> {
        controller: &'a mut OwnedExternalCharge,
    }
    impl<'a> ControllerLoan<'a> {
        pub(in crate::replicated_decoder::program) fn new(controller: &'a mut OwnedExternalCharge) -> Self {
            Self { controller }
        }
    }
    impl sealed::Sealed for OwnedExternalCharge {}
    impl sealed::Sealed for ControllerLoan<'_> {}
    fn admit(controller: &mut OwnedExternalCharge, phase: ContextPhase<'_>) -> Result<(), Error> {
        let outside = phase.plan.budget().used_bytes()
            .checked_sub(controller.payload_bytes())
            .and_then(|n| n.checked_add(phase.uncovered)).ok_or(Error::Bounds)?;
        // Admission never refunds an older live phase. Retirement alone shrinks.
        let payload = add(phase.setup.retained_payload.get(), phase.new_live)?.max(controller.payload_bytes());
        let total = add(outside, payload)?;
        if total > phase.setup.cap.min(MAX_PROGRAM_BYTES) { return Err(Error::Bounds); }
        controller.resize_payload(payload).map_err(|_| Error::Bounds)?;
        phase.plan.budget().admit_peak(total).map_err(|_| Error::Bounds)?;
        phase.setup.historical_peak.set(phase.setup.historical_peak.get().max(total));
        Ok(())
    }
    fn retire(controller: &mut OwnedExternalCharge, phase: ContextRetirement<'_>) -> Result<(), Error> {
        controller.resize_payload(phase.setup.retained_payload.get()).map_err(|_| Error::Bounds)
    }
    fn continue_history(controller: &mut OwnedExternalCharge, transfer: compact_history::HistoryBootstrap<'_, '_>) -> Result<(), Error> {
        // Split precedes capacity checks, lowering and prepayment, preserving
        // original error priority and child-charge cleanup on failure.
        let lowering = controller.split_owned(transfer.bytes()).map_err(|_| Error::Bounds)?;
        transfer.lower_original(lowering)
    }
    impl ControllerAccounting for OwnedExternalCharge {
        fn controller_identity(&self) -> usize { self as *const Self as usize }
        fn is_for(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool { OwnedExternalCharge::is_for(self, budget) }
        fn payload_bytes(&self) -> usize { OwnedExternalCharge::payload_bytes(self) }
        fn charged_bytes(&self) -> usize { OwnedExternalCharge::charged_bytes(self) }
        fn debit_metadata(&self, plan: &OriginalSourcePlan, visits: u64) -> Result<(), Error> {
            plan.prepay_compiler_metadata(self, visits).map_err(|_| Error::Bounds)
        }
        fn debit_primitive(&self, plan: &OriginalSourcePlan, work: u64) -> Result<(), Error> {
            plan.prepay_primitive_work(self, work).map_err(|_| Error::Bounds)
        }
        fn admit_context_phase(&mut self, phase: ContextPhase<'_>) -> Result<(), Error> { admit(self, phase) }
        fn retire_context_phase(&mut self, phase: ContextRetirement<'_>) -> Result<(), Error> { retire(self, phase) }
        fn continue_checked_history(&mut self, transfer: compact_history::HistoryBootstrap<'_, '_>) -> Result<(), Error> { continue_history(self, transfer) }
    }
    impl ControllerAccounting for ControllerLoan<'_> {
        fn controller_identity(&self) -> usize { std::ptr::from_ref(&*self.controller) as usize }
        fn is_for(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool { self.controller.is_for(budget) }
        fn payload_bytes(&self) -> usize { self.controller.payload_bytes() }
        fn charged_bytes(&self) -> usize { self.controller.charged_bytes() }
        fn debit_metadata(&self, plan: &OriginalSourcePlan, visits: u64) -> Result<(), Error> {
            plan.prepay_compiler_metadata(self.controller, visits).map_err(|_| Error::Bounds)
        }
        fn debit_primitive(&self, plan: &OriginalSourcePlan, work: u64) -> Result<(), Error> {
            plan.prepay_primitive_work(self.controller, work).map_err(|_| Error::Bounds)
        }
        fn admit_context_phase(&mut self, phase: ContextPhase<'_>) -> Result<(), Error> { admit(self.controller, phase) }
        fn retire_context_phase(&mut self, phase: ContextRetirement<'_>) -> Result<(), Error> { retire(self.controller, phase) }
        fn continue_checked_history(&mut self, transfer: compact_history::HistoryBootstrap<'_, '_>) -> Result<(), Error> { continue_history(self.controller, transfer) }
    }
}
pub(in crate::replicated_decoder) use controller_accounting::ControllerAccounting;
pub(in crate::replicated_decoder::program) use controller_accounting::ControllerLoan;

/// Immutable accumulator record; contains neither live loan nor Cell/RefCell.
pub(in crate::replicated_decoder) struct FrozenSetup {
    epoch: usize, rows: usize, prefix_allocation: usize,
    historical_peak: usize, retained_payload: usize, registered_builder: usize,
    cap: usize, history_input: Option<compact_history::HistoryInput>, work: WorkBudget,
}
impl ResidentSetup {
    fn into_owned_record(self) -> FrozenSetup {
        FrozenSetup { epoch: self.epoch, rows: self.rows, prefix_allocation: self.prefix_allocation,
            historical_peak: self.historical_peak.into_inner(),
            retained_payload: self.retained_payload.into_inner(),
            registered_builder: self.registered_builder.into_inner(), cap: self.cap,
            history_input: self.history_input, work: self.work.into_inner() }
    }
}

/// Scalar compatibility observations only, not a guard or accounting authority.
pub(in crate::replicated_decoder::program) struct ControllerRead<'a, G> {
    controller: std::cell::Ref<'a, G>,
}
impl<G: ControllerAccounting> ControllerRead<'_, G> {
    pub(in crate::replicated_decoder::program) fn payload_bytes(&self) -> usize { self.controller.payload_bytes() }
    pub(in crate::replicated_decoder::program) fn charged_bytes(&self) -> usize { self.controller.charged_bytes() }
    pub(in crate::replicated_decoder::program) fn is_for(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool { self.controller.is_for(budget) }
}

/// Shared, model-neutral resident lifecycle inventory. Empty/source(2),
/// install(4), bind(2), preflight(3), admit/ledger(3), burn(1) writes; phase,
/// transfer, ledger and output owner changes <=12. Schedule/body dispatch(2).
/// No native constructor/payload/Arc allocation is included in this row.
pub(in crate::replicated_decoder) fn resident_schedule_events() -> tensor::LogicalEvents {
    tensor::LogicalEvents {
        writes: 2 + 4 + 2 + 3 + 3 + 1, validation: 64,
        transitions: 12, cleanup_dispatches: 2,
        ..tensor::LogicalEvents::default()
    }
}

/// Lexical source context shared by future compact and RMS emission. It owns
/// actual origin/plan/public-source roots and a same-counter reservation; source
/// descriptor borrows come from the genuine permit, never from Builder caches.
pub(in crate::replicated_decoder) struct EmissionContext<'s, G = OwnedExternalCharge> {
    manifest: &'s Manifest,
    registry: &'s SourceRegistry<'s>,
    public: Arc<PublicSource>,
    plan: Arc<OriginalSourcePlan>,
    origin: Arc<OwnedCharged<Origin>>,
    setup: ResidentSetup,
    // The ONLY token permit stays consumed here. No &permit -> fresh ledger.
    request: RequestInputPermit<'s>,
    guard: RefCell<G>,
}
// Read-only delegation preserves scalar/Cell access, never mutable authority.
impl<G> std::ops::Deref for EmissionContext<'_, G> {
    type Target = ResidentSetup;
    fn deref(&self) -> &ResidentSetup { &self.setup }
}

pub(in crate::replicated_decoder) struct EmissionAttempt<'c, 's, G = OwnedExternalCharge> {
    context: &'c EmissionContext<'s, G>,
    committed: bool,
}
impl<G> EmissionAttempt<'_, '_, G> {
    pub(in crate::replicated_decoder) fn commit(&mut self) {
        self.committed = true;
    }
}
impl<G> Drop for EmissionAttempt<'_, '_, G> {
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

#[cfg(test)]
pub(in crate::replicated_decoder::program) struct TestOriginProbe {
    origin: Arc<OwnedCharged<Origin>>,
}
#[cfg(test)]
impl TestOriginProbe {
    pub(in crate::replicated_decoder::program) fn identity(&self) -> usize {
        Arc::as_ptr(&self.origin) as usize
    }
    pub(in crate::replicated_decoder::program) fn state(&self) -> (bool, bool) {
        let state = self.origin.payload().state.load(Ordering::Acquire);
        (state == OPEN, state == BURNED)
    }
}

impl<G: ControllerAccounting> EmissionContext<'_, G> {
    pub(in crate::replicated_decoder) fn attempt(&self) -> EmissionAttempt<'_, '_, G> {
        EmissionAttempt {
            context: self,
            committed: false,
        }
    }
    #[cfg(test)]
    pub(super) fn test_origin(&self) -> Arc<OwnedCharged<Origin>> {
        Arc::clone(&self.origin)
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_origin_probe(&self) -> TestOriginProbe {
        TestOriginProbe { origin: Arc::clone(&self.origin) }
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
            || !self.controller_is_for(self.plan.budget())
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
    pub(in crate::replicated_decoder::program) fn controller_identity(&self) -> usize { self.guard.borrow().controller_identity() }
    pub(in crate::replicated_decoder::program) fn reservation(&self) -> ControllerRead<'_, G> { ControllerRead { controller: self.guard.borrow() } }
    pub(in crate::replicated_decoder::program) fn controller_payload_bytes(&self) -> usize { self.guard.borrow().payload_bytes() }
    pub(in crate::replicated_decoder::program) fn controller_charged_bytes(&self) -> usize { self.guard.borrow().charged_bytes() }
    pub(in crate::replicated_decoder::program) fn controller_is_for(&self, budget: &pllm_garble::shared_allocation::Budget) -> bool { self.guard.borrow().is_for(budget) }
    pub(in crate::replicated_decoder::program) fn prepay_metadata(&self, visits: u64) -> Result<(), Error> {
        self.guard.borrow().debit_metadata(&self.plan, visits)
    }
    pub(in crate::replicated_decoder::program) fn prepay_primitive(&self, work: u64) -> Result<(), Error> {
        self.guard.borrow().debit_primitive(&self.plan, work)
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
        self.guard.borrow_mut().admit_context_phase(ContextPhase {
            plan: &self.plan, setup: &self.setup, new_live, uncovered,
        })
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
        if retained > self.controller_payload_bytes() {
            return Err(Error::Binding);
        }
        self.retained_payload.set(retained);
        self.registered_builder.set(bytes);
        Ok(())
    }

    pub(in crate::replicated_decoder::program) fn retire_phase(&self) -> Result<(), Error> {
        self.guard.borrow_mut().retire_context_phase(ContextRetirement { setup: &self.setup })
    }

    /// Validate open state and retained size before pending cleanup. This does
    /// not release bytes or confer entry authority. Cleanup precedes retirement's
    /// resize_payload, which remains fallible after pending payloads drop.
    pub(in crate::replicated_decoder::program) fn validate_retirement(&self) -> Result<(), Error> {
        self.check_open()?;
        if self.retained_payload.get() > self.controller_payload_bytes() { return Err(Error::Binding); }
        Ok(())
    }

    /// Uses the authentic reservation and the ORIGINAL cumulative native ledger.
    /// Schedule ownership must already be installed in the graph pending owner.
    pub(in crate::replicated_decoder) fn admit_resident(&self, schedule: &mut tensor::ResidentNativeSchedule<'_>) -> Result<(), Error> {
        let mut attempt = self.attempt();
        self.check_open()?;
        let mut total = self.work.borrow_mut();
        let mut ledger = tensor::CallerWorkLedger::new(&mut total,
            puma::MAX_PROGRAM_OPERATIONS, puma::prss::MAX_PROGRAM_AES_BLOCKS)
            .map_err(|_| Error::Bounds)?;
        schedule.admit(&mut ledger).map_err(|_| Error::Bounds)?;
        attempt.commit();
        Ok(())
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

pub(in crate::replicated_decoder::program) fn context_bytes() -> Result<usize, Error> {
    context_bytes_for::<OwnedExternalCharge>()
}

fn context_bytes_for<G>() -> Result<usize, Error> {
    // Source context local/result/move slots; Arc clones reuse actual backing.
    [
        bytes::<EmissionContext<'static, G>>(3)?,
        bytes::<Result<EmissionContext<'static, G>, Error>>(1)?,
        bytes::<G>(1)?,
        bytes::<Result<compact_lowering::PublishedView, Error>>(2)?,
        bytes::<BTreeMap<(u64, pllm_models::StateKind), compact_lowering::PublishedView>>(1)?,
        bytes::<graph::DispatchStep<'_, '_, compact_lowering::PublishedView>>(2)?,
    ].into_iter().try_fold(0, add)
}

// Alternative baseline profiles, not simultaneous owning instances or general
// credit. Two actual loan-context slots remain independently assigned within Q's
// three slots; its third/result slots confer no general credit. This unused max
// is not a complete frontier, admission quote or current funding proof.
fn resident_context_capacity_bytes() -> Result<usize, Error> {
    let owning = context_bytes_for::<OwnedExternalCharge>()?;
    let loan = context_bytes_for::<ControllerLoan<'_>>()?;
    Ok(owning.max(loan))
}

impl<G: ControllerAccounting> EmissionContext<'_, G> {
    /// Only the prequoted original source/input production path supplies the
    /// live import guard. The permit is consumed and held until final custody.
    fn check_initial(
        &self,
        imports: &Builder,
        imports_charge: &OwnedExternalCharge,
    ) -> Result<(), Error> {
        let history = &self.request.history;
        let guard = self.guard.borrow();
        check_initial_parts::<G>(
            DeclarationView { epochs: &history.epochs, fragments: &history.fragments,
                origin: &history.origin, resources: &history.resources },
            &self.request.tokens, &history.sources, imports, imports_charge, &*guard, self.cap,
        )
    }
}

fn check_initial_parts<G: ControllerAccounting>(
    history: DeclarationView<'_>,
    tokens: &View,
    sources: &RequestSources<'_>,
    imports: &Builder,
    imports_charge: &OwnedExternalCharge,
    guard: &G,
    cap: usize,
) -> Result<(), Error> {
        let plan = history.epochs.first().ok_or(Error::Binding)?;
        if history.origin.payload().state.load(Ordering::Acquire) != OPEN
            || history.epochs.len() != 1
            || !history.fragments.is_empty()
            || !Arc::ptr_eq(plan, &history.origin.payload().source)
            || !Arc::ptr_eq(&sources.public, &history.origin.payload().public)
            || sources.registry.instance() != history.origin.payload().registry_identity
            || std::ptr::from_ref(sources.registry.source_records()) as usize
                != history.origin.payload().source_records
            || !guard.is_for(plan.budget())
            || !imports_charge.is_for(plan.budget())
        {
            return Err(Error::Binding);
        }
        if guard.payload_bytes()
            < add(
                context_bytes_for::<G>()?,
                OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
            )?
        {
            return Err(Error::Bounds);
        }
        history.check_tokens(tokens)?;
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
        sources.public.check_registry(sources.registry)?;
        for (id, node) in imports.nodes.iter().enumerate() {
            match &node.code {
                Code::PublicInput { key, digest, .. } => sources.registry
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
        if historical_peak > cap.min(MAX_PROGRAM_BYTES) {
            return Err(Error::Bounds);
        }
        guard.debit_metadata(plan, u64::try_from(imports.nodes.len()).map_err(|_| Error::Bounds)?)?;
        Ok(())
}

/// Borrow-free setup only. No token permit or descriptor borrow escapes a lens.
/// This single accumulator is initialized once, never per lexical lens.
pub(in crate::replicated_decoder) struct ResidentSetup {
    epoch: usize,
    rows: usize,
    prefix_allocation: usize,
    historical_peak: Cell<usize>,
    retained_payload: Cell<usize>,
    registered_builder: Cell<usize>,
    cap: usize,
    history_input: Option<compact_history::HistoryInput>,
    work: RefCell<WorkBudget>,
}

impl ResidentSetup {
    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_seed_work(&self) {
        *self.work.borrow_mut() = WorkBudget::local();
    }

    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn test_work(&self) -> WorkBudget {
        *self.work.borrow()
    }

    pub(in crate::replicated_decoder::program) fn empty(cap: usize) -> Self {
        Self {
            epoch: 0, rows: 0, prefix_allocation: 0,
            historical_peak: Cell::new(0), retained_payload: Cell::new(0),
            registered_builder: Cell::new(0), cap, history_input: None,
            work: RefCell::new(WorkBudget::default()),
        }
    }

    pub(in crate::replicated_decoder::program) fn prepare<G: ControllerAccounting>(
        &mut self,
        declaration: &DeclarationReceiver,
        sources: &RequestSources<'_>,
        imports: &Builder,
        imports_charge: &OwnedExternalCharge,
        controller: &G,
    ) -> Result<(), Error> {
        let origin = declaration.origin.as_ref().ok_or(Error::Binding)?;
        let resources = declaration.resources.as_ref().ok_or(Error::Binding)?;
        self.rows = resources.rows;
        self.prefix_allocation = imports.nodes.as_ptr() as usize;
        self.historical_peak.set(origin.payload().source.budget().admitted_peak_bytes());
        self.retained_payload.set(controller.payload_bytes());
        #[cfg(test)]
        source_entry::receiver_checkpoint(4)?;
        check_initial_parts::<G>(
            DeclarationView { epochs: &declaration.epochs, fragments: &declaration.fragments,
                origin, resources },
            &declaration.tokens, sources, imports, imports_charge, controller, self.cap,
        )
    }
}

/// Private drop order is important: ALL emitted payloads/maps precede context
/// and import charges. Scoped borrows never expose a second token permit.
pub(in crate::replicated_decoder::program) struct EmissionScope<'s, G = OwnedExternalCharge> {
    builder: Builder,
    inputs: BTreeMap<String, SourceView>,
    context: EmissionContext<'s, G>,
    imports_charge: OwnedExternalCharge,
}
// No fallible operation before actual inputs and controller share this order.
// This wrapper's storage/control coverage remains unproved.
struct PreEmissionScope<'s, G = OwnedExternalCharge> {
    created: source_entry::HandoffSlot<source_entry::CreatedInputs<'s>>,
    guard: source_entry::HandoffSlot<G>,
}
impl<'s> EmissionScope<'s> {
    pub(in crate::replicated_decoder::program) fn new(
        created: source_entry::CreatedInputs<'s>,
        guard: OwnedExternalCharge,
        cap: usize,
    ) -> Result<Self, Error> {
        let pending = PreEmissionScope {
            #[cfg(not(test))]
            created,
            #[cfg(test)]
            created: source_entry::observe(created, None, source_entry::Event::CreatedInputs),
            #[cfg(not(test))]
            guard,
            #[cfg(test)]
            guard: source_entry::observe(guard, None, source_entry::Event::ControllerCharge),
        };
        Self::from_pre_emission(pending, cap)
    }

    #[cfg(test)]
    pub(in crate::replicated_decoder::program) fn new_observed(
        created: source_entry::CreatedInputs<'s>,
        guard: OwnedExternalCharge,
        cap: usize,
        trace: &Arc<source_entry::Trace>,
    ) -> Result<Self, Error> {
        let pending = PreEmissionScope {
            created: source_entry::observe(created, Some(trace), source_entry::Event::CreatedInputs),
            guard: source_entry::observe(guard, Some(trace), source_entry::Event::ControllerCharge),
        };
        Self::from_pre_emission(pending, cap)
    }

}
impl<'s, G: ControllerAccounting> EmissionScope<'s, G> {
    fn from_pre_emission(pending: PreEmissionScope<'s, G>, cap: usize) -> Result<Self, Error> {
        let required = add(
            context_bytes_for::<G>()?,
            OriginalSourcePlan::compiler_metadata_control_bytes().map_err(|_| Error::Bounds)?,
        )?;
        if source_entry::slot_ref(&pending.guard).payload_bytes() < required {
            return Err(Error::Bounds);
        }
        let PreEmissionScope { created, guard } = pending;
        let created = source_entry::slot_take(created);
        let guard = source_entry::slot_take(guard);
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
            setup: ResidentSetup {
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
            },
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
        &EmissionContext<'s, G>,
        &BTreeMap<String, SourceView>,
    ) {
        (&mut self.builder, &self.context, &self.inputs)
    }
}

// Clone only existing context roots, before the checked receiver extracts any
// owner. Commit and scope entry move these references without cloning them.
pub(in crate::replicated_decoder::program) struct ResidentContextOwners {
    public: Arc<PublicSource>,
    plan: Arc<OriginalSourcePlan>,
    origin: Arc<OwnedCharged<Origin>>,
}

/// Opaque transport of the checked receiver's original owners and accumulators.
/// Dependent inputs/context precede the lexical loan; no raw-parts access exists.
pub(in crate::replicated_decoder::program) struct ResidentEmission<'s, 'loan> {
    created: source_entry::CreatedInputs<'s>,
    setup: ResidentSetup,
    context: ResidentContextOwners,
    loan: RefCell<ControllerLoan<'loan>>,
}
impl ResidentEmission<'_, '_> {
    pub(in crate::replicated_decoder::program) fn retained_original(&self) -> Result<&Arc<OriginalSourcePlan>, Error> {
        self.created.permit.retained_original()
    }
}

// Only storage constructs this destination token. No getter, constructor,
// Clone, Default, callback, or returned destination is available to callers.
pub(in crate::replicated_decoder::program) struct ResidentCommitDestination<'d, 's, 'loan> {
    destination: &'d mut Option<ResidentEmission<'s, 'loan>>,
}

// Source-logical inline components; not a complete frontier/admission quote.
fn resident_context_owners_bytes() -> Result<usize, Error> {
    bytes::<ResidentContextOwners>(1)
}

fn resident_commit_destination_bytes() -> Result<usize, Error> {
    bytes::<ResidentCommitDestination<'_, '_, '_>>(1)
}

fn resident_emission_bytes() -> Result<usize, Error> {
    bytes::<ResidentEmission<'_, '_>>(1)
}

fn resident_emission_option_bytes() -> Result<usize, Error> {
    bytes::<Option<ResidentEmission<'_, '_>>>(1)
}

impl<'s, 'loan> ResidentCommitDestination<'_, 's, 'loan> {
    pub(in crate::replicated_decoder::program) fn install(self,
        created: source_entry::CreatedInputs<'s>, setup: ResidentSetup,
        context: ResidentContextOwners, loan: RefCell<ControllerLoan<'loan>>,
    ) {
        *self.destination = Some(ResidentEmission { created, setup, context, loan });
    }
}
pub(in crate::replicated_decoder::program) fn install_resident_commit<'s, 'loan>(
    commit: source_entry::resident_commit::ResidentCommit<'s, 'loan>,
    destination: &mut Option<ResidentEmission<'s, 'loan>>,
) {
    // Only the checked receiver constructs commit, after validating an empty
    // destination. No returned-error operation, allocation, fresh RefCell,
    // arbitrary callback, or destruction of a live destination occurs here.
    commit.install(ResidentCommitDestination { destination });
}
impl ResidentContextOwners {
    pub(in crate::replicated_decoder::program) fn for_drain(checked: &super::CheckedDeclarationDrain<'_, '_>) -> Self {
        let origin = checked.receiver.origin.as_ref().expect("prevalidated origin");
        Self {
            public: Arc::clone(&origin.payload().public),
            plan: Arc::clone(&origin.payload().source),
            origin: Arc::clone(origin),
        }
    }
}
impl<'s, 'loan> EmissionScope<'s, ControllerLoan<'loan>> {
    pub(in crate::replicated_decoder::program) fn from_resident(resident: ResidentEmission<'s, 'loan>) -> Self {
        let ResidentEmission { created, setup, context, loan } = resident;
        let source_entry::CreatedInputs { builder, inputs, permit, charge, .. } = created;
        let ResidentContextOwners { public, plan, origin } = context;
        let history = &permit.history;
        let context = EmissionContext {
            manifest: history.sources.manifest, registry: history.sources.registry,
            public, plan, origin, setup, request: permit, guard: loan,
        };
        Self { builder, inputs, context, imports_charge: charge }
    }
}
#[cfg(test)]
impl ResidentSetup {
    pub(in crate::replicated_decoder::program) fn test_transport_snapshot(&self) -> (usize, usize, usize, usize, usize, usize, usize, bool, WorkBudget) {
        (self.epoch, self.rows, self.prefix_allocation, self.historical_peak.get(),
            self.retained_payload.get(), self.registered_builder.get(), self.cap,
            self.history_input.is_some(), *self.work.borrow())
    }
}
#[cfg(test)]
impl ResidentEmission<'_, '_> {
    pub(in crate::replicated_decoder::program) fn test_builder_backing(&self) -> (usize, usize) {
        (self.created.builder.nodes.as_ptr() as usize, self.created.builder.nodes.capacity())
    }
    pub(in crate::replicated_decoder::program) fn test_transport_snapshot(&self) -> (usize, usize, usize, usize, usize, usize, usize, bool, WorkBudget) {
        self.setup.test_transport_snapshot()
    }
    pub(in crate::replicated_decoder::program) fn test_declaration_backings(&self) -> [(usize, usize); 6] {
        self.created.permit.test_resident_backings()
    }
}
#[cfg(test)]
impl<G: ControllerAccounting> EmissionContext<'_, G> {
    pub(in crate::replicated_decoder::program) fn test_transport_snapshot(&self) -> (usize, usize, usize, usize, usize, usize, usize, bool, WorkBudget) {
        self.setup.test_transport_snapshot()
    }
    pub(in crate::replicated_decoder::program) fn test_source_identities(&self) -> (usize, usize, usize) {
        (Arc::as_ptr(&self.plan) as usize, Arc::as_ptr(&self.public) as usize, Arc::as_ptr(&self.origin) as usize)
    }
}

#[cfg(test)]
mod context_quote_tests {
    use super::*;

    fn expected_context_bytes<G>() -> usize {
        3 * size_of::<EmissionContext<'static, G>>()
            + size_of::<Result<EmissionContext<'static, G>, Error>>()
            + size_of::<G>()
            + 2 * size_of::<Result<compact_lowering::PublishedView, Error>>()
            + size_of::<BTreeMap<(u64, pllm_models::StateKind), compact_lowering::PublishedView>>()
            + 2 * size_of::<graph::DispatchStep<'_, '_, compact_lowering::PublishedView>>()
    }

    #[test]
    fn owning_profile_matches_nonoverflow_formula() {
        let expected = expected_context_bytes::<OwnedExternalCharge>();
        assert_eq!(context_bytes_for::<OwnedExternalCharge>().unwrap(), expected);
        assert_eq!(context_bytes().unwrap(), expected);
    }

    #[test]
    fn loan_profile_matches_nonoverflow_formula() {
        assert_eq!(context_bytes_for::<ControllerLoan<'_>>().unwrap(),
            expected_context_bytes::<ControllerLoan<'_>>());
    }

    #[test]
    fn resident_capacity_is_exact_profile_maximum() {
        let owning = context_bytes_for::<OwnedExternalCharge>().unwrap();
        let loan = context_bytes_for::<ControllerLoan<'_>>().unwrap();
        let capacity = resident_context_capacity_bytes().unwrap();
        assert!(capacity >= owning);
        assert!(capacity >= loan);
        assert_eq!(capacity, owning.max(loan));
    }

    #[test]
    fn checked_component_multiplication_refuses_overflow() {
        assert_eq!(bytes::<u8>(usize::MAX).unwrap(), usize::MAX);
        assert!(matches!(bytes::<u16>(usize::MAX), Err(Error::Bounds)));
    }

    #[test]
    fn checked_component_addition_refuses_overflow() {
        assert_eq!(add(usize::MAX, 0).unwrap(), usize::MAX);
        assert!(matches!(add(usize::MAX, 1), Err(Error::Bounds)));
    }
}

#[cfg(test)]
mod accounting_setup_tests {
    use super::*;
    #[test]
    fn setup_ownership_move_preserves_accumulators() {
        // Structural test only, not a counter/source/funding fixture.
        let setup = ResidentSetup::empty(MAX_PROGRAM_BYTES);
        setup.test_seed_work();
        setup.retained_payload.set(17);
        setup.registered_builder.set(11);
        setup.historical_peak.set(23);
        let work = setup.test_work();
        let moved = Some(setup).take().unwrap();
        assert_eq!(moved.test_work(), work);
        assert_eq!(moved.retained_payload.get(), 17);
        assert_eq!(moved.registered_builder.get(), 11);
        assert_eq!(moved.historical_peak.get(), 23);
    }
}
