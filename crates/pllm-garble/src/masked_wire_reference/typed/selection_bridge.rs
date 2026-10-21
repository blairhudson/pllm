//! Consuming common-XOR-predicate selection. No fanout or arithmetic-bit cast.

use super::*;

/// One party's five-word Select material. Both typed input capabilities are
/// consumed by begin, including on a binding failure. No input capability escapes.
///
/// A ring value cannot stand in for the XOR-masked predicate:
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::{CommonMasked, Ring, SelectionParty};
/// fn wrong_kind<'b>(party: &mut SelectionParty<'b>, a: CommonMasked<'b, Ring>, b: CommonMasked<'b, Ring>) {
///     party.begin(a, b);
/// }
/// ```
///
/// Selection owners cannot be cloned:
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::SelectionParty;
/// fn duplicate<'b>(party: SelectionParty<'b>) -> SelectionParty<'b> {
///     party.clone()
/// }
/// ```
pub struct SelectionParty<'b> {
    native: Option<selection::Party<'b>>,
    predicate: Binding,
    input: Binding,
    output: Binding,
    role: u8,
    lease: Option<Lease<'b>>,
}

fn interval(input: Option<Interval>) -> Option<Interval> {
    input.map(|i| Interval {
        lower: i.lower.min(0),
        upper: i.upper.max(1),
    })
}

impl<'b> LocalDealer<'b> {
    /// Select `predicate * input` without converting the predicate to an
    /// arithmetic share. Both dealer capabilities are consumed; the result keeps
    /// the ring input's exact width/scaling and the conditional zero/input hull.
    /// There is no relation asserted between two independently supplied roots.
    pub fn select(
        &mut self,
        operation: [u8; 32],
        predicate: DealerMask<'b, Predicate>,
        input: DealerMask<'b, Ring>,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<(DealerMask<'b, Ring>, [SelectionParty<'b>; 2]), Error> {
        self.validate(predicate.binding.descriptor, true)?;
        if predicate.binding.session != self.session
            || predicate.binding.namespace != self.namespace
            || predicate.binding.consumer != operation
        {
            return Err(Error::Context);
        }
        self.input(input.binding, operation)?;
        self.validate(output, false)?;
        let p = predicate.binding.descriptor;
        let x = input.binding.descriptor;
        if p.shape() != x.shape()
            || x.shape() != output.shape()
            || p.identity().first_tensor_index() != x.identity().first_tensor_index()
            || x.identity().first_tensor_index() != output.identity().first_tensor_index()
            || p.identity().wire() == x.identity().wire()
            || p.identity().wire() == output.identity().wire()
            || x.identity().wire() == output.identity().wire()
        {
            return Err(Error::Context);
        }
        if output.numeric() != x.numeric() {
            return Err(Error::Domain);
        }
        let spec = selection::Spec {
            bits: x.numeric().ring_bits(),
            lanes: x.shape().elements(),
        };
        let cost = selection::resources(spec)?;
        // Preflight the simultaneous incremental peak while original mask
        // backings retain their own charges. Native owners reserve their own
        // already-checked portion below, without allocating between release and
        // reacquisition of that native portion. Wrapper leases stay continuous.
        let party_bytes = size_of::<SelectionParty<'static>>() + common_charge();
        let mut reservation = self
            .budget
            .reserve(cost.combined_reservation_bytes + 2 * party_bytes + mask_charge())?;
        let left_lease = reservation.split(party_bytes)?;
        let right_lease = reservation.split(party_bytes)?;
        let mask_lease = reservation.split(mask_charge())?;
        let binding = self.binding(output, next_consumer, interval(input.binding.interval))?;
        let op = digest(&[
            b"pllm.sigma.typed.select-five-word.v1\0",
            selection::DOMAIN.as_bytes(),
            &operation,
            &predicate.binding.digest(),
            &input.binding.digest(),
            &binding.digest(),
        ]);
        let context = selection::Context {
            plan: self.plan,
            session: self.session,
            operation: op,
            predicate_wire: p.digest(),
            input_wire: x.digest(),
            output_wire: output.digest(),
            first_tensor_index: x.identity().first_tensor_index(),
        };
        drop(reservation);
        let (masks, [left, right]) = selection::issue(
            spec,
            context,
            predicate.backing.values(),
            input.backing.values(),
            self.budget,
        )?;
        Ok((
            DealerMask {
                backing: MaskBacking::Selection(masks),
                binding,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [
                SelectionParty {
                    native: Some(left),
                    predicate: predicate.binding,
                    input: input.binding,
                    output: binding,
                    role: 0,
                    lease: Some(left_lease),
                },
                SelectionParty {
                    native: Some(right),
                    predicate: predicate.binding,
                    input: input.binding,
                    output: binding,
                    role: 1,
                    lease: Some(right_lease),
                },
            ],
        ))
    }
}

impl<'b> SelectionParty<'b> {
    pub fn cancel(&mut self) {
        self.native = None;
        self.lease = None;
    }

    pub fn begin(
        &mut self,
        predicate: CommonMasked<'b, Predicate>,
        input: CommonMasked<'b, Ring>,
    ) -> Result<selection::Frame<'b>, Error> {
        let result = (|| {
            if predicate.binding != self.predicate
                || input.binding != self.input
                || predicate.role != self.role
                || input.role != self.role
            {
                return Err(Error::Context);
            }
            self.native
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(predicate.backing.values(), input.backing.values())
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }

    pub fn finish(&mut self, peer: &[u8]) -> Result<CommonMasked<'b, Ring>, Error> {
        let result = (|| {
            let output = self.native.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
            let lease = self.lease.take().ok_or(Error::Consumed)?;
            self.native = None;
            Ok(CommonMasked {
                backing: CommonBacking::Selection(output),
                binding: self.output,
                role: self.role,
                _kind: PhantomData,
                _lease: lease,
            })
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
}
