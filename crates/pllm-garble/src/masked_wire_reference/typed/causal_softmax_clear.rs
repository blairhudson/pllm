//! D36 clear-source native entry only. No compiler/inherited-input conversion,
//! arbitrary table, native lease escape, transport or SDK registration.
use super::*;
use crate::shared_allocation::{OwnedBudgetContext, OwnedExternalCharge};

pub use super::region::{
    Coordinator as CausalSoftmaxCoordinatorFor, Created as CausalSoftmaxCreatedFor,
    Frame as CausalSoftmaxFrameFor, Mask as CausalSoftmaxMaskFor, Output as CausalSoftmaxOutputFor,
    Role as CausalSoftmaxRoleFor, Step as CausalSoftmaxStepFor,
};

#[derive(Debug, PartialEq, Eq)]
pub struct CausalSoftmaxError(region::BuildError);
impl From<Error> for CausalSoftmaxError {
    fn from(e: Error) -> Self {
        Self(e.into())
    }
}
impl From<region::BuildError> for CausalSoftmaxError {
    fn from(e: region::BuildError) -> Self {
        Self(e)
    }
}
impl CausalSoftmaxError {
    pub fn native(&self) -> Option<Error> {
        self.0.native()
    }
    /// Unresolved public proof remains typed and closed; never numeric fallback.
    pub fn unresolved_floor(&self) -> Option<(u64, u64)> {
        self.0.unresolved_floor()
    }
}

pub fn causal_softmax_bootstrap_bytes() -> Result<usize, Error> {
    region::bootstrap_bytes::<OwnedBudgetContext>()
}
fn bootstrap(dealer: &OwnedLocalDealer, working: &OwnedExternalCharge) -> Result<(), Error> {
    if !std::ptr::eq(working.budget(), dealer.budget.budget())
        || working.payload_bytes() < causal_softmax_bootstrap_bytes()?
    {
        return Err(Error::Resource);
    }
    Ok(())
}
/// Actual original capacities and original counter. This numeric projection is
/// not a reservation; grow the SAME working charge before issuance.
pub fn causal_softmax_resources(
    dealer: &OwnedLocalDealer,
    spec: &CausalSoftmaxSpec,
    input: &CausalSoftmaxInput,
    working: &OwnedExternalCharge,
) -> Result<CausalSoftmaxResources, CausalSoftmaxError> {
    bootstrap(dealer, working)?;
    quote::resources(dealer, spec, input).map_err(Into::into)
}

/// Consumes the original clear owner and exact SAME-counter reservation. Whole
/// score validation precedes public proof/hash/private factories/operation RNG.
/// No claim of checkpoint authentication, attention qualification or privacy.
///
/// No inherited/native input adoption:
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{causal_softmax_from_reserved,
///     CausalSoftmaxSpec,CausalSoftmaxWorkLimit,CommonMaskedFor,OwnedLocalDealer,Ring};
/// use pllm_garble::shared_allocation::{OwnedBudgetContext,OwnedExternalCharge};
/// fn upgrade(dealer:&mut OwnedLocalDealer,spec:CausalSoftmaxSpec,
///     value:CommonMaskedFor<OwnedBudgetContext,Ring>,working:OwnedExternalCharge) {
///     let _=causal_softmax_from_reserved(dealer,spec,value,working,CausalSoftmaxWorkLimit(u128::MAX));
/// }
/// ```
/// Original consumed source cannot be retried:
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{causal_softmax_from_reserved,
///     CausalSoftmaxSpec,CausalSoftmaxWorkLimit,CausalSoftmaxInput,OwnedLocalDealer};
/// use pllm_garble::shared_allocation::OwnedExternalCharge;
/// fn retry(dealer:&mut OwnedLocalDealer,spec:CausalSoftmaxSpec,
///     input:CausalSoftmaxInput,a:OwnedExternalCharge,b:OwnedExternalCharge) {
///     let _=causal_softmax_from_reserved(dealer,spec,input,a,CausalSoftmaxWorkLimit(u128::MAX));
///     let _=causal_softmax_from_reserved(dealer,spec,input,b,CausalSoftmaxWorkLimit(u128::MAX));
/// }
/// ```
/// No private/raw output getter:
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::CausalSoftmaxOutputFor;
/// use pllm_garble::shared_allocation::OwnedBudgetContext;
/// fn raw(output:&CausalSoftmaxOutputFor<OwnedBudgetContext>){let _=output.values();}
/// ```
pub fn causal_softmax_from_reserved(
    dealer: &mut OwnedLocalDealer,
    spec: CausalSoftmaxSpec,
    input: CausalSoftmaxInput,
    mut working: OwnedExternalCharge,
    work: CausalSoftmaxWorkLimit,
) -> Result<CausalSoftmaxCreatedFor<OwnedBudgetContext>, CausalSoftmaxError> {
    bootstrap(dealer, &working)?;
    // Transfer the WHOLE existing guard, not a caller quote. The private owning
    // constructor re-quotes actual immutable owners and checks exact size/work
    // before any factory; no redundant third quote/reset work pass here.
    let bytes = working.payload_bytes();
    let native = working.split_native(input.budget(), bytes)?;
    region::construct(dealer, spec, input, native, work).map_err(Into::into)
}
