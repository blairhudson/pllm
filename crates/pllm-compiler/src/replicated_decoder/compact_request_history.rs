//! Consuming compact growing-prefix history. No completion bit or executable
//! storage is issued here: whole canonical finalization remains mandatory.
use super::*;
use crate::replicated_decoder::program::compact_lowering::frozen_region::owning::OwnedEpoch;
use pllm_garble::shared_allocation::OwnedExternalCharge;

/// Closed two-output continuation map. Only `continue_original` constructs it
/// from an actually completed source epoch and its checked feedback loan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::replicated_decoder::program) struct HistoryInput {
    original: usize,
    rows: usize,
    selected: (usize, usize),
    first: usize,
    sticky: Option<usize>,
    bounds: Interval,
}
impl HistoryInput {
    pub(in crate::replicated_decoder::program) fn physical(self) -> (usize, usize, (usize, usize)) {
        (self.original, self.rows, self.selected)
    }
    pub(in crate::replicated_decoder::program) fn first(self) -> usize {
        self.first
    }
    pub(in crate::replicated_decoder::program) fn rows(self) -> usize {
        self.rows + 1
    }
    pub(in crate::replicated_decoder::program) fn bounds(self) -> Interval {
        self.bounds
    }
    pub(in crate::replicated_decoder::program) fn sticky(self) -> Option<usize> {
        self.sticky
    }
    pub(in crate::replicated_decoder::program) fn word(
        self,
        lane: usize,
    ) -> Result<(usize, usize), Error> {
        match lane.cmp(&self.rows) {
            std::cmp::Ordering::Less => Ok((self.original, lane)),
            std::cmp::Ordering::Equal => Ok(self.selected),
            std::cmp::Ordering::Greater => Err(Error::Shape),
        }
    }
}
impl<G: ControllerAccounting> EmissionContext<'_, G> {
    pub(in crate::replicated_decoder::program) fn program_cap(&self) -> usize {
        self.cap
    }
    pub(in crate::replicated_decoder::program) fn request_epoch(&self) -> usize {
        self.epoch
    }
    pub(in crate::replicated_decoder::program) fn history_input(&self) -> Option<HistoryInput> {
        self.history_input
    }
}

// Epoch payloads precede the original scope/import/history reservations.
pub(in crate::replicated_decoder::program) struct PendingCompactHistory<'s, G = OwnedExternalCharge> {
    epochs: Vec<OwnedEpoch>,
    selected: (usize, usize),
    selected_bounds: Interval,
    scope: EmissionScope<'s, G>,
}
impl<'s, G: ControllerAccounting> PendingCompactHistory<'s, G> {
    pub(in crate::replicated_decoder::program) fn checked_graph_parts(
        &self,
    ) -> Result<(&Builder, &EmissionContext<'s, G>, &OwnedEpoch), Error> {
        let context = &self.scope.context;
        context.check_imports(&self.scope.builder)?;
        if context.epoch != 1
            || self.epochs.len() != 1
            || context.history_input.ok_or(Error::Binding)?.selected != self.selected
        {
            return Err(Error::Binding);
        }
        Ok((&self.scope.builder, context, &self.epochs[0]))
    }
    pub(in crate::replicated_decoder::program) fn continue_original(
        scope: EmissionScope<'s, G>,
        first: OwnedEpoch,
        limits: original_source::Limits,
    ) -> Result<Self, Error> {
        scope.context.check_imports(&scope.builder)?;
        if scope.context.epoch != 0
            || scope.context.request.history.resources.outputs != 2
            || scope.context.request.history.epochs.len() != 1
            || scope.context.history_input.is_some()
        {
            return Err(Error::Binding);
        }
        let visits = scope
            .context
            .plan
            .plan()
            .prefill
            .operations
            .len()
            .checked_mul(2)
            .ok_or(Error::Bounds)?;
        scope.context.prepay_metadata(u64::try_from(visits).map_err(|_| Error::Bounds)?)?;
        let (plan_layout, _) = std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
            .extend(std::alloc::Layout::new::<OriginalSourcePlan>())
            .map_err(|_| Error::Bounds)?;
        let retained = add(
            OwnedEpoch::feedback_control_bytes()?,
            add(
                size_of::<Self>().checked_mul(2).ok_or(Error::Bounds)?,
                add(
                    size_of::<OwnedEpoch>()
                        .checked_mul(2)
                        .ok_or(Error::Bounds)?,
                    add(
                        size_of::<OwnedExternalCharge>()
                            .checked_mul(2)
                            .ok_or(Error::Bounds)?,
                        plan_layout.pad_to_align().size(),
                    )?,
                )?,
            )?,
        )?;
        scope.context.admit_phase(retained, 0)?;
        scope.context.retain(retained)?;
        let feedback = first.completed_feedback(&scope.context)?;
        let selected = feedback.word();
        let selected_bounds = feedback.bounds();
        let history_input = HistoryInput {
            original: scope.context.origin.payload().input.owner(),
            rows: scope.context.rows,
            selected,
            first: feedback.next(),
            sticky: feedback.sticky(),
            bounds: Interval::new(
                scope
                    .context
                    .origin
                    .payload()
                    .bounds
                    .lower()
                    .min(selected_bounds.lower()),
                scope
                    .context
                    .origin
                    .payload()
                    .bounds
                    .upper()
                    .max(selected_bounds.upper()),
            )?,
        };
        let bootstrap = scope
            .context
            .origin
            .payload()
            .source
            .lower_bootstrap_bytes(limits)
            .map_err(|_| Error::Bounds)?;
        scope.context.admit_phase(bootstrap, 0)?;
        let mut epochs = Vec::new();
        epochs.try_reserve_exact(2).map_err(|_| Error::Allocation)?;
        if epochs.capacity() != 2 {
            return Err(Error::Allocation);
        }
        epochs.push(first);
        let mut pending = Self {
            epochs,
            selected,
            selected_bounds,
            scope,
        };
        let context = &mut pending.scope.context;
        context.guard.borrow_mut().continue_checked_history(HistoryBootstrap {
            request: &mut context.request, bytes: bootstrap, limits,
        })?;
        let history = &mut context.request.history;
        context.request.tokens.words.push(selected);
        context.request.tokens.shape[1] = history_input.rows();
        history.sticky = history_input.sticky();
        context.plan = Arc::clone(&history.epochs[1]);
        context.setup.epoch = 1;
        context.setup.rows = history_input.rows();
        context.setup.history_input = Some(history_input);
        context.historical_peak.set(
            context
                .historical_peak()
                .max(context.plan.budget().admitted_peak_bytes()),
        );
        Ok(pending)
    }
    pub(in crate::replicated_decoder::program) fn parts(
        &mut self,
    ) -> (
        &mut Builder,
        &EmissionContext<'s, G>,
        &BTreeMap<String, SourceView>,
    ) {
        self.scope.parts()
    }
    fn prepare_second(
        mut self,
        second: OwnedEpoch,
    ) -> Result<(EmissionScope<'s, G>, CompletionTransport), Error> {
        let context = &self.scope.context;
        context.prepay_metadata(u64::try_from(context.plan.plan().prefill.operations.len()).map_err(|_| Error::Bounds)?)?;
        if context.epoch != 1 || self.epochs.len() != 1 || context.request.history.epochs.len() != 2
        {
            return Err(Error::Binding);
        }
        let bytes = add(
            OwnedEpoch::feedback_control_bytes()?,
            size_of::<CompletionLoan>()
                .checked_mul(3)
                .ok_or(Error::Bounds)?,
        )?;
        context.admit_phase(bytes, 0)?;
        context.retain(bytes)?;
        let feedback = second.completed_feedback(context)?;
        let incoming_sticky = feedback.sticky();
        let fallback = if incoming_sticky.is_none() {
            Some(ValidFallback {
                first: feedback.next(),
                input: Single((context.origin.payload().input.owner(), 0)),
                bounds: context.origin.payload().bounds,
            })
        } else {
            None
        };
        let valid = incoming_sticky.unwrap_or_else(|| fallback.as_ref().unwrap().first + 1);
        // The genuine fallback is constant one; authenticated checked-row
        // sticky remains boolean, regardless of any observed successful run.
        let valid_bounds = if fallback.is_some() {
            Interval::new(1, 1)?
        } else {
            Interval::new(0, 1)?
        };
        let first = feedback
            .next()
            .checked_add(if fallback.is_some() { 2 } else { 0 })
            .ok_or(Error::Bounds)?;
        if first.checked_add(3).is_none_or(|end| end > MAX_NODES) {
            return Err(Error::Bounds);
        }
        let bounds = Interval::new(
            self.selected_bounds.lower().min(feedback.bounds().lower()),
            self.selected_bounds.upper().max(feedback.bounds().upper()),
        )?;
        let product_bounds = crate::replicated_decoder::program::compact_lowering::product_bounds(
            InputBounds {
                left: bounds,
                right: valid_bounds,
            },
            1,
            false,
            false,
            None,
        )?;
        let work = WorkBudget::product(2, false)?;
        context.admit_protocol_work(work)?;
        let acceptance = Acceptance {
            first,
            selected: Gather([self.selected, feedback.word()]),
            flags: Gather([(valid, 0); 2]),
            bounds,
            valid_bounds,
            product_bounds,
            valid,
            fallback,
        };
        self.epochs.push(second);
        self.scope.context.request.history.sticky = incoming_sticky;
        let Self { epochs, scope, .. } = self;
        Ok((scope, CompletionTransport { epochs, acceptance }))
    }
}

// Constructed only after original source/feedback checks and phase admission.
// The operation cannot expose its child lowering reservation to a borrower.
pub(in crate::replicated_decoder) struct HistoryBootstrap<'a, 's> {
    request: &'a mut RequestInputPermit<'s>,
    bytes: usize,
    limits: original_source::Limits,
}
impl HistoryBootstrap<'_, '_> {
    pub(super) fn bytes(&self) -> usize { self.bytes }
    pub(super) fn lower_original(self, lowering: OwnedExternalCharge) -> Result<(), Error> {
        let history = &mut self.request.history;
        if history.epochs.len() == history.epochs.capacity()
            || self.request.tokens.words.metadata_bytes()
                < Words::span_pool_bytes(history.resources.outputs + 1).ok_or(Error::Bounds)?
        { return Err(Error::Bounds); }
        history.lower_original_epoch(1, lowering, self.limits)?;
        history.prepay_epoch(1)
    }
}
// Value-only transport: no request/context/controller borrow or callback.
struct CompletionTransport { epochs: Vec<OwnedEpoch>, acceptance: Acceptance }
impl CompletionTransport {
    fn finish<G: ControllerAccounting>(self, context: &EmissionContext<'_, G>) -> Result<CompletionLoan, Error> {
        // Keep this inside conversion, after original phase admission/retention.
        if self.epochs.len() != 2
            || context.request.tokens.words.len() != context.request.history.resources.rows + 1
        { return Err(Error::Binding); }
        Ok(CompletionLoan { epochs: self.epochs, acceptance: self.acceptance })
    }
}
impl<'s> PendingCompactHistory<'s> {
    pub(in crate::replicated_decoder::program) fn finish(self, second: OwnedEpoch) -> Result<owning::OwnedEmission<CompletionLoan>, Error> {
        let (scope, completion) = self.prepare_second(second)?;
        scope.into_owned(move |_, context, _| completion.finish(context))
    }
}
impl<'s, 'loan> PendingCompactHistory<'s, ControllerLoan<'loan>> {
    pub(in crate::replicated_decoder::program) fn finish_detached(self, second: OwnedEpoch) -> Result<owning::DetachedEmission<CompletionLoan>, Error> {
        let (scope, completion) = self.prepare_second(second)?;
        scope.into_detached(move |_, context, _| completion.finish(context))
    }
}

struct Gather([(usize, usize); 2]);
impl schedule_finalization::BorrowedLaneRefs for Gather {
    fn len(&self) -> usize {
        2
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        self.0.get(index).copied().ok_or(Error::Shape)
    }
}
struct Acceptance {
    first: usize,
    selected: Gather,
    flags: Gather,
    bounds: Interval,
    valid_bounds: Interval,
    product_bounds: Interval,
    valid: usize,
    fallback: Option<ValidFallback>,
}
// Original request_acceptance::acceptance_nodes' no-sticky path: root token
// Gather then constant-one Affine. This is not a fabricated public validity bit.
struct ValidFallback {
    first: usize,
    input: Single,
    bounds: Interval,
}
struct Single((usize, usize));
impl schedule_finalization::BorrowedLaneRefs for Single {
    fn len(&self) -> usize {
        1
    }
    fn lane_ref(&self, index: usize) -> Result<(usize, usize), Error> {
        if index == 0 {
            Ok(self.0)
        } else {
            Err(Error::Shape)
        }
    }
}
/// Owns both authenticated epoch records and the original final masked Product.
/// Not yet LogicalComplete. Authenticated acceptance can precede finalization,
/// matching the original compiler, but canonical identity/work/retirement still
/// gate executable publication under its failure-burning finish attempt.
pub(in crate::replicated_decoder) struct CompletionLoan {
    epochs: Vec<OwnedEpoch>,
    acceptance: Acceptance,
}
impl CompletionLoan {
    pub(super) fn check_final_sticky(
        &self,
        sticky: Option<usize>,
        input: usize,
    ) -> Result<(), Error> {
        match (sticky, self.acceptance.fallback.as_ref()) {
            (Some(valid), None) if valid == self.acceptance.valid => Ok(()),
            (None, Some(f))
                if f.input.0 == (input, 0)
                    && self.acceptance.valid == f.first + 1
                    && self.acceptance.first == f.first + 2 =>
            {
                Ok(())
            }
            _ => Err(Error::Binding),
        }
    }
    pub(in crate::replicated_decoder::program) fn logical_len(&self) -> usize {
        self.acceptance.first + 3
    }
    pub(super) fn selected_words(&self) -> [(usize, usize); 2] {
        self.acceptance.selected.0
    }
    pub(super) fn valid_owner(&self) -> usize {
        self.acceptance.valid
    }
    pub(in crate::replicated_decoder::program) fn epochs(&self) -> &[OwnedEpoch] {
        &self.epochs
    }
    pub(in crate::replicated_decoder::program) fn outputs(&self) -> [(usize, usize); 3] {
        [
            (self.acceptance.first + 2, 0),
            (self.acceptance.first + 2, 1),
            (self.acceptance.valid, 0),
        ]
    }
    pub(in crate::replicated_decoder::program) fn acceptance_node(
        &self,
        ordinal: usize,
    ) -> Result<schedule_finalization::BorrowedNode<'_>, Error> {
        use schedule_finalization::{BorrowedNode, BorrowedRecipe};
        let owner = &self.acceptance;
        if let Some(f) = &owner.fallback {
            let node = if ordinal == f.first {
                Some(BorrowedNode {
                    lanes: 1,
                    bounds: f.bounds,
                    recipe: BorrowedRecipe::Gather(&f.input),
                })
            } else if ordinal == f.first + 1 {
                Some(BorrowedNode {
                    lanes: 1,
                    bounds: Interval::new(1, 1)?,
                    recipe: BorrowedRecipe::Affine {
                        left: f.first,
                        right: None,
                        coefficients: affine(0, 0, 1),
                    },
                })
            } else {
                None
            };
            if let Some(node) = node {
                return Ok(node);
            }
        }
        let (bounds, recipe) = match ordinal.checked_sub(owner.first).ok_or(Error::Binding)? {
            0 => (owner.bounds, BorrowedRecipe::Gather(&owner.selected)),
            1 => (owner.valid_bounds, BorrowedRecipe::Gather(&owner.flags)),
            2 => (
                owner.product_bounds,
                BorrowedRecipe::Product {
                    left: owner.first,
                    right: owner.first + 1,
                    shape: None,
                    rescale: false,
                    norm: None,
                },
            ),
            _ => return Err(Error::Binding),
        };
        Ok(BorrowedNode {
            lanes: 2,
            bounds,
            recipe,
        })
    }
}
