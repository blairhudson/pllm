//! Checked, local CLEAR-root entry to the original D29 region.
//!
//! The caller admits raw input creation separately, retains a real bootstrap
//! charge before constructing the declaration/query records, grows that SAME
//! charge to `total_reserved_bytes`, then moves both original owners here.
//! Source labels are local declarations, never authentication. No inherited
//! entry, caller tables, remasking, output extraction or SDK registration.

use super::super::{Error, OwnedLocalDealer};
use super::region;
use crate::shared_allocation::{
    AllocationContext, LeaseFor, OwnedBudgetContext, OwnedExternalCharge,
};
use pllm_core::edge_contract::EdgeDescriptor;
use std::mem::size_of;

pub use super::region::{
    Coordinator as ClearNexpCoordinatorFor, Created as ClearNexpCreatedFor,
    Frame as ClearNexpFrameFor, Input as ClearNexpInput, Mask as ClearNexpMaskFor,
    Output as ClearNexpOutputFor, Quote as ClearNexpResources, Role as ClearNexpRoleFor,
    Spec as ClearNexpSpec, Step as ClearNexpStepFor, WorkQuote as ClearNexpWork,
};

/// Fixed, nonformatting failure. Original typed native/floor failure is retained;
/// callers may inspect it without allocating diagnostic strings. Formatting and
/// dependency/native stack/allocator/RSS are not admitted by this interface.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct ClearNexpError(region::BuildError);

impl ClearNexpError {
    pub fn native(&self) -> Option<Error> {
        match &self.0 {
            region::BuildError::Native(error) => Some(*error),
            _ => None,
        }
    }

    /// An unresolved arithmetic floor stays typed and closed; never repaired or
    /// converted into a supplied table/rounding fallback.
    pub fn unresolved_floor(&self) -> Option<(u64, u64)> {
        use super::super::activation::certified_exp_lookup;
        use crate::{exp_floor, fixed_exp_tables};
        match &self.0 {
            region::BuildError::Quote(exp_floor::Error::Unresolved {
                lower_floor,
                upper_floor,
            })
            | region::BuildError::Floor(fixed_exp_tables::Error::Floor(
                exp_floor::Error::Unresolved {
                    lower_floor,
                    upper_floor,
                },
            ))
            | region::BuildError::Certified(certified_exp_lookup::Error::Floor(
                fixed_exp_tables::Error::Floor(exp_floor::Error::Unresolved {
                    lower_floor,
                    upper_floor,
                }),
            )) => Some((*lower_floor, *upper_floor)),
            _ => None,
        }
    }
}
impl From<region::BuildError> for ClearNexpError {
    fn from(error: region::BuildError) -> Self {
        Self(error)
    }
}
impl From<Error> for ClearNexpError {
    fn from(error: Error) -> Self {
        Self(region::BuildError::Native(error))
    }
}

// Additional public caller/callee/old/new/result slots, not native stack frames.
// Included in ORIGINAL ConstructionSlots/bootstrap/control before any factory.
// Reexported handles have exactly the original layouts: no owner wrapper/Arc.
#[allow(dead_code)]
pub(super) struct CallSlots<'a, C: AllocationContext> {
    dealer: &'a mut super::super::LocalDealerFor<C>,
    query_dealers: [&'a super::super::LocalDealerFor<C>; 2],
    working_argument: &'a OwnedExternalCharge,
    spec_arguments: [&'a region::Spec; 2],
    input_argument: &'a region::Input,
    edges: [EdgeDescriptor; 14],
    declarations: [region::Spec; 2],
    declaration_results: [Result<region::Spec, Error>; 2],
    operation: [u8; 32],
    next: [u8; 32],
    quotes: [region::Quote; 2],
    native_quote: Result<region::Quote, region::BuildError>,
    public_quote: Result<region::Quote, ClearNexpError>,
    input_owner: region::Input,
    original_charge: OwnedExternalCharge,
    native_lease: LeaseFor<C>,
    native_lease_result: Result<LeaseFor<C>, Error>,
    native_result: Result<region::Created<C>, region::BuildError>,
    public_result: Result<region::Created<C>, ClearNexpError>,
    native_error: region::BuildError,
    public_error: ClearNexpError,
    fixed_result: Result<usize, Error>,
    required: usize,
}

/// Real charge required BEFORE declaration/resource records, not Budget authority.
/// Existing dealer creation/input import and caller-owned originals are separate.
pub fn clear_nexp_bootstrap_bytes() -> usize {
    region::bootstrap_bytes::<OwnedBudgetContext>()
}

fn bootstrap(dealer: &OwnedLocalDealer, working: &OwnedExternalCharge) -> Result<(), Error> {
    if !working.is_for(dealer.budget.budget())
        || working.payload_bytes() < clear_nexp_bootstrap_bytes()
    {
        return Err(Error::Resource);
    }
    Ok(())
}

impl ClearNexpSpec {
    /// Check EXACT wire order X/X40/A/H/B/W/V/C/J0/J1/U0/U1/P/E, geometry,
    /// identity, numeric/representation/visibility and counter headroom using the
    /// same region preflight. Metadata alone grants no input or source authority.
    /// The live original bootstrap must predate these fixed control records.
    pub fn checked(
        dealer: &OwnedLocalDealer,
        edges: [EdgeDescriptor; 14],
        operation: [u8; 32],
        next: [u8; 32],
        working: &OwnedExternalCharge,
    ) -> Result<Self, Error> {
        bootstrap(dealer, working)?;
        let declaration = Self {
            edges,
            operation,
            next,
        };
        region::preflight(dealer, &declaration)?;
        Ok(declaration)
    }

    pub fn input_descriptor(&self) -> EdgeDescriptor {
        self.edges[0]
    }

    pub fn output_descriptor(&self) -> EdgeDescriptor {
        self.edges[13]
    }
}

impl ClearNexpResources {
    /// Exact native slice transferred from the original external reservation.
    /// Original input/dealer/external-guard charges remain separately resident.
    pub fn total_reserved_bytes(&self) -> usize {
        self.total
    }
    pub fn incoming_owner_bytes(&self) -> usize {
        self.incoming_owner_bytes
    }
    pub fn peer_bytes(&self) -> usize {
        self.peer_bytes
    }
    pub fn work(&self) -> ClearNexpWork {
        self.work
    }
}

/// Allocation/entropy-free original-owner quote. Uses actual retained capacity;
/// never trusts a caller baseline, quote token or numeric cap as a reservation.
pub fn clear_nexp_resources(
    dealer: &OwnedLocalDealer,
    declaration: &ClearNexpSpec,
    input: &ClearNexpInput,
    working: &OwnedExternalCharge,
) -> Result<ClearNexpResources, ClearNexpError> {
    bootstrap(dealer, working)?;
    region::resources(dealer, declaration, input).map_err(Into::into)
}

/// Consume ORIGINAL clear input and EXACT original-counter full reservation.
/// Requotes actual immutable owners before splitting; no stale numeric quote can
/// authorize construction. Whole-root range rejection precedes proof/hash/secret
/// factories and operation entropy in the unchanged region constructor.
///
/// The caller grows the same bootstrap with `resize_payload`, never refunds and
/// reacquires. Native leases/opaque outputs retain that original counter after
/// this residual external guard/root/dealer retire. Both Ready are required.
///
/// No raw output or inherited native input conversion:
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::ClearNexpOutputFor;
/// use pllm_garble::shared_allocation::OwnedBudgetContext;
/// fn raw(output: &ClearNexpOutputFor<OwnedBudgetContext>) {
///     let _descriptor = output.descriptor();
///     let _ = output.values();
/// }
/// ```
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{
///     clear_nexp_from_reserved, ClearNexpMaskFor, ClearNexpSpec, OwnedLocalDealer,
/// };
/// use pllm_garble::shared_allocation::{OwnedBudgetContext, OwnedExternalCharge};
/// fn inherited(dealer: &mut OwnedLocalDealer, declaration: ClearNexpSpec,
///              mask: ClearNexpMaskFor<OwnedBudgetContext>, working: OwnedExternalCharge) {
///     let _ = clear_nexp_from_reserved(dealer, declaration, mask, working);
/// }
/// ```
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{
///     clear_nexp_from_reserved, ClearNexpInput, ClearNexpSpec, OwnedLocalDealer,
/// };
/// use pllm_garble::shared_allocation::OwnedExternalCharge;
/// fn twice(dealer: &mut OwnedLocalDealer, declaration: ClearNexpSpec,
///          input: ClearNexpInput, a: OwnedExternalCharge, b: OwnedExternalCharge) {
///     let _ = clear_nexp_from_reserved(dealer, declaration, input, a);
///     let _ = clear_nexp_from_reserved(dealer, declaration, input, b);
/// }
/// ```
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::ClearNexpRoleFor;
/// use pllm_garble::shared_allocation::OwnedBudgetContext;
/// fn must_send<T: Send>() {}
/// must_send::<ClearNexpRoleFor<OwnedBudgetContext>>();
/// ```
pub fn clear_nexp_from_reserved(
    dealer: &mut OwnedLocalDealer,
    declaration: ClearNexpSpec,
    input: ClearNexpInput,
    mut working: OwnedExternalCharge,
) -> Result<ClearNexpCreatedFor<OwnedBudgetContext>, ClearNexpError> {
    bootstrap(dealer, &working)?;
    let quote = region::resources(dealer, &declaration, &input)?;
    if working.payload_bytes() != quote.total {
        return Err(Error::Resource.into());
    }
    let native = working.split_native(input.budget(), quote.total)?;
    region::construct(dealer, declaration, input, native).map_err(Into::into)
}

// Public methods above do not add a field or backing owner. Keep the public
// transparent error's old/new Result inventory explicit rather than guessing.
const _: () = assert!(size_of::<ClearNexpError>() == size_of::<region::BuildError>());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_original_unresolved_error_remains_typed_without_numeric_fallback() {
        use super::super::super::activation::certified_exp_lookup;
        use crate::{exp_floor, fixed_exp_tables};
        for original in [
            region::BuildError::Quote(exp_floor::Error::Unresolved {
                lower_floor: 7,
                upper_floor: 8,
            }),
            region::BuildError::Floor(fixed_exp_tables::Error::Floor(
                exp_floor::Error::Unresolved {
                    lower_floor: 7,
                    upper_floor: 8,
                },
            )),
            region::BuildError::Certified(certified_exp_lookup::Error::Floor(
                fixed_exp_tables::Error::Floor(exp_floor::Error::Unresolved {
                    lower_floor: 7,
                    upper_floor: 8,
                }),
            )),
        ] {
            let error = ClearNexpError::from(original);
            assert_eq!(error.unresolved_floor(), Some((7, 8)));
            assert_eq!(error.native(), None);
        }
        assert_eq!(
            ClearNexpError::from(Error::Resource).native(),
            Some(Error::Resource)
        );
    }

    #[test]
    fn public_error_moves_and_control_inventory_have_explicit_concrete_layouts() {
        use std::mem::align_of;
        assert_eq!(size_of::<ClearNexpError>(), size_of::<region::BuildError>());
        assert_eq!(
            align_of::<ClearNexpError>(),
            align_of::<region::BuildError>()
        );
        assert_eq!(
            size_of::<Result<ClearNexpCreatedFor<OwnedBudgetContext>, ClearNexpError>>(),
            size_of::<Result<region::Created<OwnedBudgetContext>, region::BuildError>>()
        );
        assert!(clear_nexp_bootstrap_bytes() >= size_of::<CallSlots<'_, OwnedBudgetContext>>());
    }
}
