//! Private consuming D19-output adapter, not a public Multiply or import API.
//! Existing CommonMasked/DealerMask layouts and D22/D27 remain unchanged.
//! Only the future reviewed nExp issuer may create these wrappers around its
//! actual original native owners and predeclared bindings, under original leases.

use super::super::{elementwise_product as native, source_lookup};
use super::*;

pub(super) struct LookupMaskFor<C: AllocationContext> {
    pub(super) native: source_lookup::OutputMasksFor<C>,
    pub(super) binding: Binding,
    pub(super) lease: LeaseFor<C>,
}
pub(super) struct LookupValueFor<C: AllocationContext> {
    pub(super) native: source_lookup::MaskedOutputFor<C>,
    pub(super) binding: Binding,
    pub(super) role: u8,
    pub(super) lease: LeaseFor<C>,
}
pub(super) struct ProductMaskFor<C: AllocationContext> {
    pub(super) native: native::MasksFor<C>,
    pub(super) binding: Binding,
    pub(super) lease: LeaseFor<C>,
}
pub(super) struct ProductValueFor<C: AllocationContext> {
    pub(super) native: native::OutputFor<C>,
    pub(super) binding: Binding,
    pub(super) role: u8,
    pub(super) lease: LeaseFor<C>,
}
pub(super) struct PartyFor<C: AllocationContext> {
    native: Option<native::PartyFor<C>>,
    inputs: [Binding; 2],
    output: Binding,
    role: u8,
    retained: Option<[LookupValueFor<C>; 2]>,
    lease: LeaseFor<C>,
}

// A noncloneable REAL reservation/declaration owner, never a lookup capability.
// Its output binding is available for acyclic precommit, before either lookup
// material exists; only issue consumes the actual matching native owners.
pub(super) struct DeclarationFor<C: AllocationContext> {
    inputs: [Binding; 2],
    output: Binding,
    native: LeaseFor<C>,
    left: LeaseFor<C>,
    right: LeaseFor<C>,
    mask: LeaseFor<C>,
}

fn control_bytes<C: AllocationContext>() -> Result<usize, Error> {
    [
        size_of::<[LookupMaskFor<C>; 2]>(),
        size_of::<[LookupValueFor<C>; 2]>(),
        size_of::<Result<(ProductMaskFor<C>, [PartyFor<C>; 2]), Error>>(),
        size_of::<Result<ProductValueFor<C>, Error>>(),
        size_of::<[Binding; 3]>(),
        size_of::<[LeaseFor<C>; 3]>(),
        size_of::<native::Context>(),
        size_of::<native::Resources>(),
        size_of::<Result<native::Resources, Error>>(),
        size_of::<[Sha256; 2]>(),
        size_of::<[&[u8]; 10]>(),
        size_of::<[[u8; 32]; 4]>(),
        size_of::<DeclarationFor<C>>(),
        size_of::<Result<DeclarationFor<C>, Error>>(),
    ]
    .into_iter()
    .try_fold(0usize, |a, b| a.checked_add(b).ok_or(Error::Resource))
}
fn party_bytes<C: AllocationContext>() -> Result<usize, Error> {
    size_of::<PartyFor<C>>()
        .checked_add(size_of::<ProductValueFor<C>>())
        .and_then(|n| n.checked_add(control_bytes::<C>().ok()?))
        .ok_or(Error::Resource)
}
pub(super) fn wrapper_bytes<C: AllocationContext>() -> Result<usize, Error> {
    party_bytes::<C>()?
        .checked_mul(2)
        .and_then(|n| n.checked_add(size_of::<ProductMaskFor<C>>()))
        .and_then(|n| n.checked_add(control_bytes::<C>().ok()?))
        .ok_or(Error::Resource)
}

/// Metadata/declaration only, never authority to issue. Full reservation precedes
/// binding/hash/entropy. The consuming issue checks actual original D19 owners.
pub(super) fn declare<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    operation: [u8; 32],
    inputs: [Binding; 2],
    output: EdgeDescriptor,
    next: [u8; 32],
) -> Result<DeclarationFor<C>, Error> {
    let native_cost = checked_declaration(dealer, operation, inputs, output)?;
    let total = declaration_bytes::<C>(native_cost)?;
    let reservation = LeaseFor::reserve(&dealer.budget, total)?;
    declare_checked(dealer, inputs, output, next, native_cost, reservation)
}

/// Exact original aggregate slice; no refund/re-reserve or counter replacement.
/// Same validation order as declare, before binding/entropy/private factories.
pub(super) fn declare_reserved<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    operation: [u8; 32],
    inputs: [Binding; 2],
    output: EdgeDescriptor,
    next: [u8; 32],
    reservation: LeaseFor<C>,
) -> Result<DeclarationFor<C>, Error> {
    let native_cost = checked_declaration(dealer, operation, inputs, output)?;
    if reservation.bytes != declaration_bytes::<C>(native_cost)?
        || !std::ptr::eq(reservation.budget.budget(), dealer.budget.budget())
    {
        return Err(Error::Resource);
    }
    declare_checked(dealer, inputs, output, next, native_cost, reservation)
}

fn checked_declaration<C: AllocationContext>(
    dealer: &LocalDealerFor<C>,
    operation: [u8; 32],
    inputs: [Binding; 2],
    output: EdgeDescriptor,
) -> Result<native::Resources, Error> {
    for input in &inputs {
        dealer.input(*input, operation)?;
        let numeric = input.descriptor.numeric();
        if numeric.ring_bits() != 50 || numeric.fractional_bits() != 12 {
            return Err(Error::Domain);
        }
        let range = input.interval.ok_or(Error::Domain)?;
        if range.lower < 0 || range.upper > 4097 || range.lower >= range.upper {
            return Err(Error::Domain);
        }
    }
    dealer.validate(output, false)?;
    let first = inputs[0].descriptor;
    let second = inputs[1].descriptor;
    if inputs[0].mask_id == inputs[1].mask_id
        || first.identity().wire() == second.identity().wire()
        || first.identity().wire() == output.identity().wire()
        || second.identity().wire() == output.identity().wire()
        || first.shape() != second.shape()
        || first.shape() != output.shape()
        || first.identity().first_tensor_index() != second.identity().first_tensor_index()
        || first.identity().first_tensor_index() != output.identity().first_tensor_index()
    {
        return Err(Error::Context);
    }
    if output.numeric().ring_bits() != 50 || output.numeric().fractional_bits() != 24 {
        return Err(Error::Domain);
    }
    first
        .identity()
        .first_tensor_index()
        .checked_add(first.shape().elements() as u64)
        .ok_or(Error::Resource)?;
    native::resources_for::<C>(first.shape().elements())
}

fn declaration_bytes<C: AllocationContext>(native_cost: native::Resources) -> Result<usize, Error> {
    native_cost
        .combined_bytes
        .checked_add(wrapper_bytes::<C>()?)
        .ok_or(Error::Resource)
}

fn declare_checked<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    inputs: [Binding; 2],
    output: EdgeDescriptor,
    next: [u8; 32],
    native_cost: native::Resources,
    mut reservation: LeaseFor<C>,
) -> Result<DeclarationFor<C>, Error> {
    let native_lease = reservation.split(native_cost.combined_bytes)?;
    let left_lease = reservation.split(party_bytes::<C>()?)?;
    let right_lease = reservation.split(party_bytes::<C>()?)?;
    let result_binding = dealer.binding(
        output,
        next,
        Some(Interval {
            lower: 0,
            upper: 16_777_217,
        }),
    )?;
    Ok(DeclarationFor {
        inputs,
        output: result_binding,
        native: native_lease,
        left: left_lease,
        right: right_lease,
        mask: reservation,
    })
}

impl<C: AllocationContext> DeclarationFor<C> {
    pub(super) fn output_binding(&self) -> Binding {
        self.output
    }

    /// Original masks remain alive through their last correlation read. Nothing
    /// clones/mints/replaces an input; failure consumes the whole declaration.
    pub(super) fn issue(
        self,
        region: [u8; 32],
        stage: [u8; 32],
        inputs: [LookupMaskFor<C>; 2],
    ) -> Result<(ProductMaskFor<C>, [PartyFor<C>; 2]), Error> {
        self.issue_with(region, stage, inputs, |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    /// Construction-only entropy injection, identical native issue/validation.
    /// Not callable under the interactive metadata/state lock.
    pub(super) fn issue_with(
        self,
        region: [u8; 32],
        stage: [u8; 32],
        inputs: [LookupMaskFor<C>; 2],
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(ProductMaskFor<C>, [PartyFor<C>; 2]), Error> {
        for (actual, expected) in inputs.iter().zip(self.inputs.iter()) {
            if actual.binding != *expected
                || !std::ptr::eq(actual.lease.budget.budget(), self.mask.budget.budget())
            {
                return Err(Error::Context);
            }
        }
        let context = native::Context {
            region,
            stage,
            inputs: [self.inputs[0].digest(), self.inputs[1].digest()],
            output: self.output.digest(),
            first: self.output.descriptor.identity().first_tensor_index(),
        };
        let (masks, [left, right]) = native::issue_reserved(
            context,
            inputs[0].native.values(),
            inputs[1].native.values(),
            self.native,
            fill,
        )?;
        Ok((
            ProductMaskFor {
                native: masks,
                binding: self.output,
                lease: self.mask,
            },
            [
                PartyFor {
                    native: Some(left),
                    inputs: self.inputs,
                    output: self.output,
                    role: 0,
                    retained: None,
                    lease: self.left,
                },
                PartyFor {
                    native: Some(right),
                    inputs: self.inputs,
                    output: self.output,
                    role: 1,
                    retained: None,
                    lease: self.right,
                },
            ],
        ))
    }
}

impl<C: AllocationContext> PartyFor<C> {
    pub(super) fn cancel(&mut self) {
        self.native = None;
        self.retained = None;
        // Keep still-existing wrapper/transition charge until actual drop.
    }
    pub(super) fn begin(
        &mut self,
        inputs: [LookupValueFor<C>; 2],
    ) -> Result<native::FrameFor<C>, Error> {
        let result = (|| {
            if self.retained.is_some() {
                return Err(Error::Consumed);
            }
            for (input, expected) in inputs.iter().zip(&self.inputs) {
                if input.binding != *expected
                    || input.role != self.role
                    || !std::ptr::eq(input.lease.budget.budget(), self.lease.budget.budget())
                {
                    return Err(Error::Context);
                }
            }
            let frame = self
                .native
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(inputs[0].native.values(), inputs[1].native.values())?;
            self.retained = Some(inputs);
            Ok(frame)
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub(super) fn finish(&mut self, peer: &[u8]) -> Result<ProductValueFor<C>, Error> {
        let result = (|| {
            let native = self.native.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
            self.native = None;
            self.retained = None;
            let lease = self.lease.split(size_of::<ProductValueFor<C>>())?;
            Ok(ProductValueFor {
                native,
                binding: self.output,
                role: self.role,
                lease,
            })
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
}

#[cfg(test)]
#[path = "nexp_product_tests.rs"]
mod tests;
