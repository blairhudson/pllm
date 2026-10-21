//! Fixed Q12 TR6 -> DReLU -> Select owning region. Not a graph executor.
//!
//! Public declarations never authorize execution. Four private ordered slots
//! authorize borrowing original owners; no intermediate capability is minted.
//! The coordinator must cancel the sibling whenever either party fails.
//! Q12 descriptors admit n=12..64 under the unchanged core numeric contract.
//! Propagated intervals remain local source assumptions, not compiler proofs.

use super::*;
use pllm_core::edge_contract::ConsumerUse;

/// Immutable, public-only declaration of the single supported template.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixSpec {
    operation: [u8; 32],
    input: EdgeDescriptor,
    y: EdgeDescriptor,
    d: EdgeDescriptor,
    r: EdgeDescriptor,
    next_consumer: [u8; 32],
}

impl PrefixSpec {
    pub fn new(
        operation: [u8; 32],
        input: EdgeDescriptor,
        y: EdgeDescriptor,
        d: EdgeDescriptor,
        r: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<Self, Error> {
        let spec = Self {
            operation,
            input,
            y,
            d,
            r,
            next_consumer,
        };
        spec.validate()?;
        Ok(spec)
    }

    fn validate(self) -> Result<(), Error> {
        let edges = [self.input, self.y, self.d, self.r];
        let bits = self.input.numeric().ring_bits();
        let lanes = self.input.shape().elements();
        if !(12..=64).contains(&bits) || !(1..=256).contains(&lanes) {
            return Err(Error::Domain);
        }
        for (i, edge) in edges.iter().enumerate() {
            if edge.identity().source() != self.input.identity().source()
                || edge.identity().plan() != self.input.identity().plan()
                || edge.shape() != self.input.shape()
                || edge.identity().first_tensor_index()
                    != self.input.identity().first_tensor_index()
                || edges[..i]
                    .iter()
                    .any(|old| old.identity().wire() == edge.identity().wire())
            {
                return Err(Error::Context);
            }
            if edge.weight_visibility().is_some() {
                return Err(Error::Unsupported);
            }
            let predicate = i == 2;
            let numeric = if predicate {
                Numeric::new(1, 0, Interpretation::Predicate)
            } else {
                Numeric::new(
                    if i == 0 { bits } else { bits - 6 },
                    if i == 0 { 12 } else { 6 },
                    Interpretation::Signed,
                )
            }
            .map_err(|_| Error::Domain)?;
            if edge.numeric() != numeric
                || edge.representation()
                    != if predicate {
                        Representation::CommonXorMasked
                    } else {
                        Representation::CommonAdditiveMasked
                    }
            {
                return Err(Error::Domain);
            }
        }
        Ok(())
    }

    fn stages(self) -> [[u8; 32]; 3] {
        [0u8, 1, 2].map(|stage| {
            digest(&[
                b"pllm.sigma.typed.prefix-stage.v1\0",
                &self.operation,
                &[stage],
                &self.input.digest(),
                &self.y.digest(),
                &self.d.digest(),
                &self.r.digest(),
                &self.next_consumer,
            ])
        })
    }

    fn uses(self) -> [ConsumerUse; 4] {
        let [tr, drelu, select] = self.stages();
        [
            ConsumerUse::new(self.input, tr, 0, 0),
            ConsumerUse::new(self.y, drelu, 0, 1),
            ConsumerUse::new(self.d, select, 0, 2),
            ConsumerUse::new(self.y, select, 1, 3),
        ]
    }

    /// Declaration digest, not an execution permit or authentication token.
    pub fn digest(self) -> [u8; 32] {
        let uses = self.uses().map(|u| u.digest());
        digest(&[
            b"pllm.sigma.typed.q12-prefix.v1\0",
            arithmetic::DOMAIN.as_bytes(),
            selection::DOMAIN.as_bytes(),
            Profile::DOMAIN.as_bytes(),
            &self.operation,
            &self.r.digest(),
            &self.next_consumer,
            &uses[0],
            &uses[1],
            &uses[2],
            &uses[3],
        ])
    }

    fn native_specs(self) -> (arithmetic::Spec, arithmetic::Spec, selection::Spec) {
        let bits = self.input.numeric().ring_bits();
        let lanes = self.input.shape().elements();
        (
            arithmetic::Spec {
                bits,
                lanes,
                gate: arithmetic::Gate::Tr { shift: 6 },
            },
            arithmetic::Spec {
                bits: bits - 6,
                lanes,
                gate: arithmetic::Gate::Drelu,
            },
            selection::Spec {
                bits: bits - 6,
                lanes,
            },
        )
    }
}

struct Ledger {
    uses: [ConsumerUse; 4],
    bindings: [Binding; 4],
    role: u8,
    next: u8,
}
impl Ledger {
    fn digest(&self) -> [u8; 32] {
        let slots: [[u8; 32]; 4] = std::array::from_fn(|index| {
            digest(&[
                b"pllm.sigma.typed.prefix-private-use.v1\0",
                &self.uses[index].digest(),
                &self.bindings[index].digest(),
                &[self.role],
            ])
        });
        digest(&[
            b"pllm.sigma.typed.prefix-private-ledger.v1\0",
            &slots[0],
            &slots[1],
            &slots[2],
            &slots[3],
        ])
    }

    // Burn before checking: neither metadata nor a failed attempt grants retry.
    fn burn(&mut self, expected: u8, binding: Binding, role: u8) -> Result<(), Error> {
        let slot = self.next;
        self.next = self.next.saturating_add(1);
        if slot != expected || slot >= 4 {
            return Err(Error::Consumed);
        }
        if self.bindings[usize::from(slot)] != binding
            || self.role != role
            || self.uses[usize::from(slot)].edge() != binding.descriptor
        {
            return Err(Error::Context);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Tr,
    Drelu,
    Select,
    Dead,
}

/// Party-local native owners only. Internal Y/D never leave this owner.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::ActivationPrefixParty;
/// fn duplicate<'b>(party: ActivationPrefixParty<'b>) -> ActivationPrefixParty<'b> {
///     party.clone()
/// }
/// ```
///
/// Intermediate Y remains private, not an exportable evaluator capability:
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::ActivationPrefixParty;
/// fn escape_y(party: &mut ActivationPrefixParty<'_>) {
///     let _escaped_y = party.y.take().unwrap();
/// }
/// ```
///
/// Beginning the region moves the original input, even if execution fails:
///
/// ```compile_fail,E0382
/// use pllm_garble::masked_wire_reference::typed::{ActivationPrefixParty, CommonMasked, Ring};
/// fn reuse_input<'b>(party: &mut ActivationPrefixParty<'b>, x: CommonMasked<'b, Ring>) {
///     let _frame = party.begin(x);
///     let _descriptor = x.descriptor(); // The original input has moved.
/// }
/// ```
///
/// Final R exposes metadata, not its common masked payload:
///
/// ```compile_fail,E0616
/// use pllm_garble::masked_wire_reference::typed::{ActivationPrefixParty, Error, PrefixStep};
/// fn read_final(party: &mut ActivationPrefixParty<'_>, peer: &[u8]) -> Result<(), Error> {
///     if let PrefixStep::Output(output) = party.advance(peer)? {
///         let _descriptor = output.descriptor();
///         let _payload = &output.backing; // Private common masked payload.
///     }
///     Ok(())
/// }
/// ```
///
/// The returned dealer mask likewise keeps native mask values private:
///
/// ```compile_fail,E0616
/// use pllm_garble::masked_wire_reference::typed::{DealerMask, Error, LocalDealer, PrefixSpec, Ring};
/// fn read_mask<'b>(dealer: &mut LocalDealer<'b>, spec: PrefixSpec, input: DealerMask<'b, Ring>)
///     -> Result<(), Error>
/// {
///     let (mask, _parties) = dealer.activation_prefix(spec, input)?;
///     let _values = &mask.backing; // Private dealer mask values.
///     Ok(())
/// }
/// ```
pub struct ActivationPrefixParty<'b> {
    tr: Option<arithmetic::Party<'b>>,
    drelu: Option<arithmetic::Party<'b>>,
    select: Option<selection::Party<'b>>,
    x: Option<CommonMasked<'b, Ring>>,
    y: Option<arithmetic::MaskedOutput<'b>>,
    d: Option<arithmetic::MaskedOutput<'b>>,
    input: Binding,
    internal: [Binding; 2],
    output: Binding,
    spec: PrefixSpec,
    ledger: Ledger,
    role: u8,
    phase: Phase,
    // Last: zeroize/drop every native payload before refunding wrapper capacity.
    lease: Option<Lease<'b>>,
}

enum NativeFrame<'b> {
    Arithmetic(arithmetic::Frame<'b>),
    Selection(selection::Frame<'b>),
}

/// Native frame remains inline and keeps its own native lease.
pub struct PrefixFrame<'b> {
    native: NativeFrame<'b>,
    _lease: Lease<'b>,
}
impl PrefixFrame<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.native {
            NativeFrame::Arithmetic(f) => f.as_bytes(),
            NativeFrame::Selection(f) => f.as_bytes(),
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum PrefixStep<'b> {
    Frame(PrefixFrame<'b>),
    Output(CommonMasked<'b, Ring>),
}

fn frame_wrapper_charge() -> usize {
    size_of::<PrefixFrame<'static>>() + size_of::<PrefixStep<'static>>()
}
fn party_wrapper_charge() -> usize {
    size_of::<ActivationPrefixParty<'static>>()
        + common_charge()
        + 4 * frame_wrapper_charge()
        + (8 << 10)
}
fn issuance_workspace_charge() -> usize {
    // Inline specs, bindings, ledgers, contexts, return arrays, result enums and
    // hashing temporaries. Native workspace/payload is separately fully priced.
    2 * size_of::<ActivationPrefixParty<'static>>()
        + size_of::<PrefixSpec>()
        + 4 * size_of::<Binding>()
        + 2 * size_of::<arithmetic::Context>()
        + size_of::<selection::Context>()
        + size_of::<DealerMask<'static, Ring>>()
        + (16 << 10)
}

impl<'b> LocalDealer<'b> {
    pub fn activation_prefix(
        &mut self,
        spec: PrefixSpec,
        input: DealerMask<'b, Ring>,
    ) -> Result<(DealerMask<'b, Ring>, [ActivationPrefixParty<'b>; 2]), Error> {
        self.activation_prefix_with(spec, input, |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    fn activation_prefix_with(
        &mut self,
        spec: PrefixSpec,
        input: DealerMask<'b, Ring>,
        mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [ActivationPrefixParty<'b>; 2]), Error> {
        spec.validate()?;
        self.input(input.binding, spec.operation)?;
        if input.binding.descriptor != spec.input {
            return Err(Error::Context);
        }
        for (edge, predicate) in [(spec.y, false), (spec.d, true), (spec.r, false)] {
            self.validate(edge, predicate)?;
        }
        let (tr_spec, d_spec, s_spec) = spec.native_specs();
        let costs = [
            arithmetic::resources(tr_spec)?.combined_reservation_bytes,
            arithmetic::resources(d_spec)?.combined_reservation_bytes,
            selection::resources(s_spec)?.combined_reservation_bytes,
        ];
        let wrappers = 2 * party_wrapper_charge() + mask_charge();
        let native = costs
            .iter()
            .try_fold(0usize, |sum, cost| sum.checked_add(*cost))
            .ok_or(Error::Resource)?;
        let total = native
            .checked_add(wrappers)
            .and_then(|v| v.checked_add(issuance_workspace_charge()))
            .ok_or(Error::Resource)?;
        // Existing input/common roots retain original leases, already charged.
        // Reserve ALL three simultaneous native gates before entropy/allocation.
        let mut reservation = self.budget.reserve(total)?;
        let left_lease = reservation.split(party_wrapper_charge())?;
        let right_lease = reservation.split(party_wrapper_charge())?;
        let mask_lease = reservation.split(mask_charge())?;
        let interval = input.binding.interval.map(|i| i.shift(6));
        // Exact conditional zero/input hull, without upgrading source assumptions.
        let result_interval = interval.map(|i| Interval {
            lower: i.lower.min(0),
            upper: i.upper.max(1),
        });
        let y_binding = self.binding(spec.y, spec.digest(), interval)?;
        let d_binding = self.binding(spec.d, spec.digest(), None)?;
        let output = self.binding(spec.r, spec.next_consumer, result_interval)?;
        let ledger = |role| Ledger {
            uses: spec.uses(),
            bindings: [input.binding, y_binding, d_binding, y_binding],
            role,
            next: 0,
        };
        let left_ledger = ledger(0);
        let right_ledger = ledger(1);
        let context_id = digest(&[
            b"pllm.sigma.typed.prefix-instance.v1\0",
            &spec.digest(),
            &input.binding.digest(),
            &y_binding.digest(),
            &d_binding.digest(),
            &output.digest(),
            &left_ledger.digest(),
            &right_ledger.digest(),
        ]);
        let [tr_id, d_id, s_id] = spec.stages();
        let arithmetic_context =
            |stage: [u8; 32], x: EdgeDescriptor, y: EdgeDescriptor| arithmetic::Context {
                plan: self.plan,
                session: self.session,
                operation: digest(&[&context_id, &stage]),
                input_wire: x.digest(),
                output_wire: y.digest(),
                first_tensor_index: x.identity().first_tensor_index(),
            };
        let tr_context = arithmetic_context(tr_id, spec.input, spec.y);
        let d_context = arithmetic_context(d_id, spec.y, spec.d);
        let s_context = selection::Context {
            plan: self.plan,
            session: self.session,
            operation: digest(&[&context_id, &s_id]),
            predicate_wire: spec.d.digest(),
            input_wire: spec.y.digest(),
            output_wire: spec.r.digest(),
            first_tensor_index: spec.y.identity().first_tensor_index(),
        };
        // Only native preflight slots are released immediately before native
        // reserve. All wrapper and not-yet-issued native reservations stay live.
        drop(reservation.split(costs[0])?);
        let (y_masks, [t0, t1]) = arithmetic::issue_with(
            tr_spec,
            tr_context,
            input.backing.values(),
            self.budget,
            &mut fill,
        )?;
        drop(reservation.split(costs[1])?);
        let (d_masks, [d0, d1]) =
            arithmetic::issue_with(d_spec, d_context, y_masks.values(), self.budget, &mut fill)?;
        drop(reservation.split(costs[2])?);
        let (r_masks, [s0, s1]) = selection::issue_with(
            s_spec,
            s_context,
            d_masks.values(),
            y_masks.values(),
            self.budget,
            &mut fill,
        )?;
        let party = |ledger: Ledger, tr, drelu, select, lease| ActivationPrefixParty {
            tr: Some(tr),
            drelu: Some(drelu),
            select: Some(select),
            x: None,
            y: None,
            d: None,
            input: input.binding,
            internal: [y_binding, d_binding],
            output,
            spec,
            role: ledger.role,
            ledger,
            phase: Phase::Fresh,
            lease: Some(lease),
        };
        Ok((
            DealerMask {
                backing: MaskBacking::Selection(r_masks),
                binding: output,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [
                party(left_ledger, t0, d0, s0, left_lease),
                party(right_ledger, t1, d1, s1, right_lease),
            ],
        ))
    }
}

impl<'b> ActivationPrefixParty<'b> {
    pub fn spec(&self) -> PrefixSpec {
        self.spec
    }

    pub fn cancel(&mut self) {
        self.phase = Phase::Dead;
        self.ledger.next = 4;
        self.tr = None;
        self.drelu = None;
        self.select = None;
        self.x = None;
        self.y = None;
        self.d = None;
        self.lease = None;
    }

    fn frame(&mut self, native: NativeFrame<'b>) -> Result<PrefixFrame<'b>, Error> {
        Ok(PrefixFrame {
            native,
            _lease: self
                .lease
                .as_mut()
                .ok_or(Error::Consumed)?
                .split(frame_wrapper_charge())?,
        })
    }

    /// Consume the sole external evaluator capability, including on failure.
    pub fn begin(&mut self, input: CommonMasked<'b, Ring>) -> Result<PrefixFrame<'b>, Error> {
        let result = (|| {
            self.ledger.burn(0, input.binding, input.role)?;
            if self.phase != Phase::Fresh {
                return Err(Error::Consumed);
            }
            if input.binding != self.input || input.role != self.role {
                return Err(Error::Context);
            }
            self.x = Some(input);
            self.phase = Phase::Tr;
            let frame = self
                .tr
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(self.x.as_ref().ok_or(Error::Consumed)?.backing.values())?;
            self.frame(NativeFrame::Arithmetic(frame))
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }

    /// Deliver exactly one peer native frame. No intermediate reconstruction
    /// is added or collapsed; TR retains both native exchange phases.
    pub fn advance(&mut self, peer: &[u8]) -> Result<PrefixStep<'b>, Error> {
        let result = self.advance_inner(peer);
        if result.is_err() {
            self.cancel();
        }
        result
    }

    fn advance_inner(&mut self, peer: &[u8]) -> Result<PrefixStep<'b>, Error> {
        match self.phase {
            Phase::Tr => match self.tr.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
                arithmetic::Step::Frame(f) => {
                    Ok(PrefixStep::Frame(self.frame(NativeFrame::Arithmetic(f))?))
                }
                arithmetic::Step::Output(y) => {
                    self.y = Some(y);
                    self.tr = None;
                    self.ledger.burn(1, self.internal[0], self.role)?;
                    self.phase = Phase::Drelu;
                    let f = self
                        .drelu
                        .as_mut()
                        .ok_or(Error::Consumed)?
                        .begin(self.y.as_ref().ok_or(Error::Consumed)?.values())?;
                    Ok(PrefixStep::Frame(self.frame(NativeFrame::Arithmetic(f))?))
                }
            },
            Phase::Drelu => match self.drelu.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
                arithmetic::Step::Frame(_) => Err(Error::Context),
                arithmetic::Step::Output(d) => {
                    self.d = Some(d);
                    self.drelu = None;
                    self.ledger.burn(2, self.internal[1], self.role)?;
                    self.ledger.burn(3, self.internal[0], self.role)?;
                    self.phase = Phase::Select;
                    let f = self.select.as_mut().ok_or(Error::Consumed)?.begin(
                        self.d.as_ref().ok_or(Error::Consumed)?.values(),
                        self.y.as_ref().ok_or(Error::Consumed)?.values(),
                    )?;
                    Ok(PrefixStep::Frame(self.frame(NativeFrame::Selection(f))?))
                }
            },
            Phase::Select => {
                let output = self.select.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
                let lease = self.lease.take().ok_or(Error::Consumed)?;
                let binding = self.output;
                let role = self.role;
                self.cancel();
                Ok(PrefixStep::Output(CommonMasked {
                    backing: CommonBacking::Selection(output),
                    binding,
                    role,
                    _kind: PhantomData,
                    _lease: lease,
                }))
            }
            Phase::Fresh | Phase::Dead => Err(Error::Consumed),
        }
    }
}

#[cfg(test)]
mod tests;
