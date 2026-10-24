//! Closed D36 M-X -> unchanged D29 body. No clear reconstruction, root encode,
//! generic CommonMasked import or public provisional-output issuer.
use super::super::super::causal_softmax::{
    CausalSoftmaxInput, DifferenceMaskFor, DifferenceValueFor,
};
use super::*;
use std::sync::MutexGuard;

// Metadata only, not a permit. Actual constructors consume original owners.
pub(in crate::masked_wire_reference::typed) fn spec(
    edges: [EdgeDescriptor; 14],
    operation: [u8; 32],
    next: [u8; 32],
) -> Spec {
    Spec {
        edges,
        operation,
        next,
    }
}

fn error(error: BuildError) -> Error {
    match error {
        BuildError::Native(e) => e,
        _ => Error::Domain,
    }
}

pub(in crate::masked_wire_reference::typed) fn bootstrap_bytes<C: AllocationContext>() -> usize {
    super::bootstrap_bytes::<C>()
}

/// Extra-child quote BEFORE parent entropy. Borrows actual original scores,
/// without copying/recovering/adopting them as an nExp input. Outer admission
/// separately retains/prices its genuine Difference/source/control leases.
pub(in crate::masked_wire_reference::typed) fn resources<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    s: &Spec,
    original: &CausalSoftmaxInput,
) -> Result<Quote, Error> {
    let n = preflight_bindings(dealer, s, 13)?;
    if !std::ptr::eq(original.budget(), dealer.budget.budget()) {
        return Err(Error::Context);
    }
    let minimum = sum(&[
        size_of::<CausalSoftmaxInput>(),
        original
            .payload()
            .capacity()
            .checked_mul(size_of::<i64>())
            .ok_or(Error::Resource)?,
    ])?;
    if original.charged_bytes() < minimum || original.payload().len() < n {
        return Err(Error::Resource);
    }
    resources_common(dealer, s, n, original.charged_bytes(), 0, 0, false).map_err(error)
}

pub(super) fn mask_resources<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    s: &Spec,
    mask: &DifferenceMaskFor<C>,
) -> Result<Quote, BuildError> {
    let n = preflight_bindings(dealer, s, 13)?;
    let binding = mask.binding();
    if !std::ptr::eq(mask.budget(), dealer.budget.budget())
        || binding.descriptor != s.edges[0]
        || binding.session != dealer.session
        || binding.namespace != dealer.namespace
        || binding.consumer != s.operation
        || binding.issuance >= dealer.counter
        || binding.interval
            != Some(Interval {
                lower: 0,
                upper: 1i128 << 38,
            })
        || mask.values().len() != n
    {
        return Err(Error::Context.into());
    }
    // Actual incoming leases remain intact; outer owner admitted them BEFORE
    // max/difference creation. No historical reservation credit/refund here.
    let retained = mask.retained_bytes()?;
    if retained < mask.original_source_bytes() {
        return Err(Error::Resource.into());
    }
    resources_common(dealer, s, n, retained, 0, 0, false)
}

/// X binding was precommitted once by the actual difference creator. Never
/// reminted. No role values exist yet; each role later consumes its own owner.
pub(in crate::masked_wire_reference::typed) fn construct<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: Spec,
    mask: DifferenceMaskFor<C>,
    working: LeaseFor<C>,
) -> Result<Created<C>, Error> {
    construct_with(dealer, s, mask, working, |bytes| {
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
}

// Same original constructor's closed entropy seam. Parent construction owns
// this native fill; no callback runs under a parent/child runtime lock.
pub(in crate::masked_wire_reference::typed) fn construct_with<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    s: Spec,
    mask: DifferenceMaskFor<C>,
    working: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<Created<C>, Error> {
    construct_input_with(dealer, s, InputOrigin::Inherited(mask), working, fill).map_err(error)
}

/// Native internal poll only. External cancellation callback must already have
/// completed with BOTH outer/child locks released. No callback invoked here.
pub(in crate::masked_wire_reference::typed) fn begin<C: AllocationContext>(
    role: &mut Role<C>,
    value: DifferenceValueFor<C>,
) -> Result<Frame<C>, Error> {
    role.shared.operation(|_| {
        let mut state = role.slot.state.lock().map_err(|_| Error::Consumed)?;
        let InputOrigin::Inherited(mask) = &state.tables.input else {
            return Err(Error::Context);
        };
        if state.phase != 0
            || state.root.is_some()
            || value.role() != role.role
            || !value.is_for(mask)
            || value.binding() != state.bindings[0]
            || !std::ptr::eq(value.budget(), mask.budget())
            || value.values().len() != mask.values().len()
        {
            return Err(Error::Context);
        }
        value.retained_bytes()?;
        state.root = Some(RootValue::Inherited(value));
        let native = state.begin()?;
        Ok(Frame {
            native,
            shared: Arc::clone(&role.shared),
        })
    })
}

pub(in crate::masked_wire_reference::typed) fn advance<C: AllocationContext>(
    role: &mut Role<C>,
    peer: &[u8],
) -> Result<Step<C>, Error> {
    role.shared.operation(|control| {
        let mut state = role.slot.state.lock().map_err(|_| Error::Consumed)?;
        if !matches!(&state.tables.input, InputOrigin::Inherited(_)) {
            return Err(Error::Context);
        }
        match state.advance(peer)? {
            Some(native) => Ok(Step::Frame(Frame {
                native,
                shared: Arc::clone(&role.shared),
            })),
            None => {
                control.mark_ready(role.role);
                Ok(Step::Ready)
            }
        }
    })
}

/// Closed parent abnormal-teardown hook. All child loans must have retired
/// before this call; no callback/reentry occurs while the child control is held.
/// Successful per-role continuation retirement must NEVER call this hook.
pub(in crate::masked_wire_reference::typed) fn abort<C: AllocationContext>(
    coordinator: &Coordinator<C>,
) {
    let mut control = coordinator.shared.control();
    if !control.aborted {
        coordinator.shared.abort_locked(&mut control);
    }
}

/// Provisional receipt only. E stays inside its ORIGINAL child slot/Shared so
/// cancellation synchronously erases it. Parent fixed use-ledger + both-ready
/// barrier still govern outward results. No Clone or raw-owner extraction.
pub(in crate::masked_wire_reference::typed) struct EValueFor<C: AllocationContext> {
    slot: Arc<Slot<C>>,
    shared: Arc<Shared<C>>,
    role: u8,
}
pub(in crate::masked_wire_reference::typed) fn accept_ready<C: AllocationContext>(
    mut role: Role<C>,
) -> Result<EValueFor<C>, Error> {
    let receipt = role.shared.operation(|control| {
        if !control.ready[usize::from(role.role)] {
            return Err(Error::Consumed);
        }
        let state = role.slot.state.lock().map_err(|_| Error::Consumed)?;
        if !matches!(&state.tables.input, InputOrigin::Inherited(_)) || state.output.is_none() {
            return Err(Error::Consumed);
        }
        Ok(EValueFor {
            slot: Arc::clone(&role.slot),
            shared: Arc::clone(&role.shared),
            role: role.role,
        })
    });
    role.handed_off = receipt.is_ok();
    receipt
}
// A private provisional receipt is not a public finished output. Ordinary
// retirement after its fixed outer uses must not abort the running parent or
// invalidate child frames. Parent Keepalive retains genuine Mask/Coordinator;
// abnormal parent Retire MUST call inherited::abort(&coordinator), even after
// both child Roles have handed off. Mask/Coordinator Drop/cancel also erase E.

/// State guard retires before control guard. Panic aborts both child slots only
/// AFTER those guards release; no callback/reentry allowed while loan is held.
pub(in crate::masked_wire_reference::typed) struct EValueLoan<'a, C: AllocationContext> {
    state: Option<MutexGuard<'a, State<C>>>,
    control: Option<MutexGuard<'a, custody::SplitControl>>,
    shared: &'a Shared<C>,
    role: u8,
}
impl<C: AllocationContext> EValueFor<C> {
    pub(in crate::masked_wire_reference::typed) fn loan(&self) -> Result<EValueLoan<'_, C>, Error> {
        let result = (|| {
            let control = self.shared.control();
            if control.aborted || !control.ready[usize::from(self.role)] {
                return Err(Error::Consumed);
            }
            let state = self.slot.state.lock().map_err(|_| Error::Consumed)?;
            if state.output.is_none() {
                return Err(Error::Consumed);
            }
            Ok(EValueLoan {
                state: Some(state),
                control: Some(control),
                shared: &self.shared,
                role: self.role,
            })
        })();
        if result.is_err() {
            self.shared.abort();
        }
        result
    }
    pub(in crate::masked_wire_reference::typed) fn same_region(&self, mask: &Mask<C>) -> bool {
        Arc::ptr_eq(&self.shared, &mask.shared)
    }
}
impl<C: AllocationContext> EValueLoan<'_, C> {
    pub(in crate::masked_wire_reference::typed) fn binding(&self) -> Binding {
        self.state
            .as_ref()
            .expect("live native loan")
            .output
            .as_ref()
            .expect("ready native output")
            .binding
    }
    pub(in crate::masked_wire_reference::typed) fn values(&self) -> &[u64] {
        self.state
            .as_ref()
            .expect("live native loan")
            .output
            .as_ref()
            .expect("ready native output")
            .backing
            .values()
    }
    pub(in crate::masked_wire_reference::typed) fn role(&self) -> u8 {
        self.role
    }
}
impl<C: AllocationContext> Drop for EValueLoan<'_, C> {
    fn drop(&mut self) {
        drop(self.state.take());
        drop(self.control.take());
        if std::thread::panicking() {
            self.shared.abort();
        }
    }
}

pub(in crate::masked_wire_reference::typed) struct MaskLoan<'a, C: AllocationContext> {
    control: Option<MutexGuard<'a, custody::SplitControl>>,
    mask: &'a Mask<C>,
}
pub(in crate::masked_wire_reference::typed) fn loan_mask<C: AllocationContext>(
    mask: &Mask<C>,
) -> Result<MaskLoan<'_, C>, Error> {
    let result = (|| {
        let control = mask.shared.control();
        if control.aborted {
            return Err(Error::Consumed);
        }
        {
            let state = mask.shared.slots[0]
                .state
                .lock()
                .map_err(|_| Error::Consumed)?;
            if !matches!(&state.tables.input, InputOrigin::Inherited(_)) {
                return Err(Error::Context);
            }
        }
        Ok(MaskLoan {
            control: Some(control),
            mask,
        })
    })();
    if result.is_err() {
        mask.shared.abort();
    }
    result
}
impl<C: AllocationContext> MaskLoan<'_, C> {
    pub(in crate::masked_wire_reference::typed) fn binding(&self) -> Binding {
        self.mask.native.binding
    }
    pub(in crate::masked_wire_reference::typed) fn values(&self) -> &[u64] {
        self.mask.native.backing.values()
    }
}
impl<C: AllocationContext> Drop for MaskLoan<'_, C> {
    fn drop(&mut self) {
        drop(self.control.take());
        if std::thread::panicking() {
            self.mask.shared.abort();
        }
    }
}

// Known logical controls/old-new/result overlaps, NOT compiled stack or RSS.
#[allow(dead_code)]
pub(super) struct Controls<'a, C: AllocationContext> {
    original: &'a CausalSoftmaxInput,
    input: InputOrigin<C>,
    input_result: Result<InputOrigin<C>, Error>,
    root: RootValue<C>,
    root_result: Result<RootValue<C>, Error>,
    receipt: EValueFor<C>,
    receipt_result: Result<EValueFor<C>, Error>,
    value_loan: EValueLoan<'a, C>,
    value_result: Result<EValueLoan<'a, C>, Error>,
    mask_loan: MaskLoan<'a, C>,
    mask_result: Result<MaskLoan<'a, C>, Error>,
    source_bytes: usize,
    retained_bytes: usize,
}
pub(super) fn control_bytes<C: AllocationContext>() -> Result<usize, Error> {
    sum(&[
        size_of::<Controls<'_, C>>(),
        size_of::<Result<Quote, Error>>(),
        size_of::<Result<Created<C>, Error>>(),
        2 * size_of::<Option<RootValue<C>>>(),
    ])
}
