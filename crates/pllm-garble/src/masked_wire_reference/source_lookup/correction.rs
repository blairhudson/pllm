//! Private consuming scan-to-correction boundary. Not a raw-mask issuer.
use super::*;
use zeroize::Zeroize;
mod interval50;

#[cfg(test)]
thread_local! {
    pub(super) static PANIC_BEGIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static PANIC_ADVANCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) struct ScanResultFor<C: AllocationContext> {
    pub(super) sums: Zeroizing<Vec<u64>>,
    pub(super) counts: Zeroizing<Vec<i64>>,
    pub(super) role: u8,
    pub(super) spec: Spec,
    pub(super) binding: [u8; 32],
    pub(super) digest: [u8; 32],
    pub(super) lease: LeaseFor<C>,
}

pub(super) struct CorrectionState {
    pub(super) arithmetic: Zeroizing<Vec<u64>>,
    pub(super) correction: Zeroizing<Vec<u64>>,
    pub(super) digest: [u8; 32],
    pub(super) phase: u8,
}

pub(super) struct CorrectionFor<C: AllocationContext> {
    pub(super) masks: Zeroizing<Vec<[u64; 2]>>,
    pub(super) state: Option<CorrectionState>,
    // Private Norm profiles retain consumed scan backing until outer retirement.
    // Ordinary D19 retains its original immediate count retirement behavior.
    retained_counts: Option<Zeroizing<Vec<i64>>>,
    pub(super) select: Option<selection::PartyFor<C>>,
    pub(super) scan_lease: Option<LeaseFor<C>>,
    pub(super) role: u8,
    pub(super) spec: Spec,
    pub(super) binding: [u8; 32],
    pub(super) lease: LeaseFor<C>,
}

// Named live call/result slots, included before issuance, not a Budget token.
pub(super) struct Controls<'a, C: AllocationContext> {
    _scan: Option<ScanResultFor<C>>,
    _state: Option<CorrectionState>,
    _frame: Result<FrameFor<C>, Error>,
    _step: Result<StepFor<C>, Error>,
    _abort: Option<AbortOnDrop<'a, C>>,
}

#[derive(Clone, Copy)]
pub(super) struct CorrectionDescriptor {
    pub(super) wire_spec: Spec,
}

/// Checked numeric profile, NOT a source/key/scan receipt. Default callers can
/// obtain this only through the unchanged index8/10 validation. Private D43
/// scanners must additionally present their creation-issued source seal at the
/// material finishing boundary; this resource token cannot issue an owner.
#[derive(Clone, Copy)]
pub(super) struct ValidatedDescriptor(CorrectionDescriptor);
impl ValidatedDescriptor {
    fn lookup(descriptor: CorrectionDescriptor) -> Result<Self, Error> {
        descriptor.wire_spec.validate()?;
        Ok(Self(descriptor))
    }
    pub(super) fn norm_inverse() -> Self {
        Self(CorrectionDescriptor {
            wire_spec: Spec {
                index_bits: 13,
                output_bits: 50,
                lanes: 1,
                boundary: Boundary::Reconstructed,
            },
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct CorrectionResources {
    pub(super) party: usize,
    pub(super) dealer: usize,
    pub(super) combined: usize,
    pub(super) select: selection::Resources,
}

pub(super) struct CorrectionMasksFor<C: AllocationContext> {
    pub(super) values: Zeroizing<Vec<u64>>,
    pub(super) select_masks: selection::OutputMasksFor<C>,
    pub(super) dealer: LeaseFor<C>,
}
pub(super) type CorrectionResult<C> = Result<(CorrectionMasksFor<C>, [CorrectionFor<C>; 2]), Error>;

pub(super) struct CorrectionBuildFor<C: AllocationContext> {
    descriptor: CorrectionDescriptor,
    next_lane: usize,
    failed: bool,
    masks: [Zeroizing<Vec<[u64; 2]>>; 2],
    correction: [Zeroizing<Vec<u64>>; 2],
    rho: Zeroizing<Vec<u64>>,
    predicate_masks: Zeroizing<Vec<u64>>,
    arithmetic_masks: Zeroizing<Vec<u64>>,
    select_reservation: LeaseFor<C>,
    party_reservations: [LeaseFor<C>; 2],
    dealer_reservation: LeaseFor<C>,
}

#[allow(dead_code)] // Explicit child factory call/move/result overlaps, not backings.
pub(super) struct IssuanceControls<'a, C: AllocationContext> {
    checked_descriptor: ValidatedDescriptor,
    checked_result: Result<ValidatedDescriptor, Error>,
    quote: CorrectionResources,
    quote_result: Result<CorrectionResources, Error>,
    build: Option<CorrectionBuildFor<C>>,
    seal: LookupIssueSeal,
    lane: IssuedKeyLane,
    result: CorrectionResult<C>,
    select_result: Result<(selection::OutputMasksFor<C>, [selection::PartyFor<C>; 2]), Error>,
    select_masks: Option<selection::OutputMasksFor<C>>,
    select_parties: Option<[selection::PartyFor<C>; 2]>,
    parties: Option<[CorrectionFor<C>; 2]>,
    masks: Option<CorrectionMasksFor<C>>,
    words: Zeroizing<[u64; 4]>,
    draw_bytes: Zeroizing<[u8; 8]>,
    aggregate: Option<LeaseFor<C>>,
    select: Option<LeaseFor<C>>,
    party_leases: [Option<LeaseFor<C>>; 2],
    dealer: Option<LeaseFor<C>>,
    splits: [Result<LeaseFor<C>, Error>; 3],
    push_guard: Option<PushGuard<'a, C>>,
    allocation_link: Option<C>,
    descriptor: CorrectionDescriptor,
    lane_index: usize,
    push_result: Result<(), Error>,
    finishing: Option<FinishingFor<C>>,
    party_spec: Spec,
    party_binding: [u8; 32],
    party_role: u8,
}

// Keeps all outer payloads before their charges while Select is fallible or
// unwinds. The pending Select lease alone moves to the original native issuer.
struct FinishingFor<C: AllocationContext> {
    masks: [Zeroizing<Vec<[u64; 2]>>; 2],
    correction: [Zeroizing<Vec<u64>>; 2],
    rho: Zeroizing<Vec<u64>>,
    predicate_masks: Zeroizing<Vec<u64>>,
    arithmetic_masks: Zeroizing<Vec<u64>>,
    select: Option<LeaseFor<C>>,
    parties: [LeaseFor<C>; 2],
    dealer: LeaseFor<C>,
}

pub(super) fn resources<C: AllocationContext>(
    descriptor: CorrectionDescriptor,
) -> Result<CorrectionResources, Error> {
    resources_validated::<C>(ValidatedDescriptor::lookup(descriptor)?)
}

pub(super) fn resources_validated<C: AllocationContext>(
    descriptor: ValidatedDescriptor,
) -> Result<CorrectionResources, Error> {
    let spec = descriptor.0.wire_spec;
    let select = selection::resources_for_context::<C>(spec.select(), spec.boundary)?;
    if select.combined_reservation_bytes
        != add(&[
            select.dealer_reservation_bytes,
            mul(2, select.party_reservation_bytes)?,
        ])?
    {
        return Err(Error::Resource);
    }
    let reconstructed = spec.boundary == Boundary::Reconstructed;
    let party = add(&[
        size_of::<CorrectionFor<C>>(),
        mul(16, spec.lanes)?,
        ALLOCATOR,
        mul(8, spec.lanes)?,
        ALLOCATOR,
        control_bytes::<C>(),
        first_frame_bytes(spec),
        if reconstructed { select.frame_bytes } else { 0 },
        frame_charge_for::<C>(spec),
        output_charge_for::<C>(spec),
        WORKSPACE,
        if reconstructed {
            size_of::<FrameFor<C>>()
        } else {
            0
        },
        select.party_reservation_bytes,
    ])?;
    let dealer = add(&[
        size_of::<CorrectionMasksFor<C>>(),
        C::TRANSIENT_LINK_BYTES,
        mul(24, spec.lanes)?,
        3 * ALLOCATOR,
        WORKSPACE,
        size_of::<IssuanceControls<'_, C>>(),
        select.dealer_reservation_bytes,
    ])?;
    let combined = add(&[dealer, mul(2, party)?])?;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(CorrectionResources {
        party,
        dealer,
        combined,
        select,
    })
}

pub(super) fn prepare_reserved<C: AllocationContext>(
    descriptor: CorrectionDescriptor,
    reservation: LeaseFor<C>,
) -> Result<CorrectionBuildFor<C>, Error> {
    prepare_validated(ValidatedDescriptor::lookup(descriptor)?, reservation)
}

pub(super) fn prepare_validated<C: AllocationContext>(
    descriptor: ValidatedDescriptor,
    mut reservation: LeaseFor<C>,
) -> Result<CorrectionBuildFor<C>, Error> {
    let quote = resources_validated::<C>(descriptor)?;
    let descriptor = descriptor.0;
    if reservation.bytes != quote.combined {
        return Err(Error::Resource);
    }
    let allocation = reservation.budget.share();
    let budget = allocation.budget();
    // All split guards precede every payload allocation.
    let select_reservation = reservation.split(quote.select.combined_reservation_bytes)?;
    let party = quote
        .party
        .checked_sub(quote.select.party_reservation_bytes)
        .ok_or(Error::Resource)?;
    let a = reservation.split(party)?;
    let b = reservation.split(party)?;
    if reservation.bytes
        != quote
            .dealer
            .checked_sub(quote.select.dealer_reservation_bytes)
            .ok_or(Error::Resource)?
    {
        return Err(Error::Resource);
    }
    let lanes = descriptor.wire_spec.lanes;
    let masks = [
        Zeroizing::new(allocate(lanes, budget)?),
        Zeroizing::new(allocate(lanes, budget)?),
    ];
    let correction = [zeros(lanes, budget)?, zeros(lanes, budget)?];
    let rho = zeros(lanes, budget)?;
    let predicate_masks = zeros(lanes, budget)?;
    let arithmetic_masks = zeros(lanes, budget)?;
    Ok(CorrectionBuildFor {
        descriptor,
        next_lane: 0,
        failed: false,
        masks,
        correction,
        rho,
        predicate_masks,
        arithmetic_masks,
        select_reservation,
        party_reservations: [a, b],
        dealer_reservation: reservation,
    })
}

struct PushGuard<'a, C: AllocationContext> {
    build: &'a mut CorrectionBuildFor<C>,
}
impl<C: AllocationContext> Drop for PushGuard<'_, C> {
    fn drop(&mut self) {
        if self.build.failed {
            for masks in &mut self.build.masks {
                masks.zeroize();
            }
            for words in &mut self.build.correction {
                words.zeroize();
            }
            self.build.rho.zeroize();
            self.build.predicate_masks.zeroize();
            self.build.arithmetic_masks.zeroize();
        }
    }
}

impl<C: AllocationContext> CorrectionBuildFor<C> {
    pub(super) fn push_after_keypair(
        &mut self,
        lane: IssuedKeyLane,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.push_index(lane.index, fill)
    }
    pub(super) fn push_norm13_keypair(
        &mut self,
        lane: Norm13KeyLane,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if self.descriptor.wire_spec != ValidatedDescriptor::norm_inverse().0.wire_spec {
            self.failed = true;
            return Err(Error::Context);
        }
        self.push_index(lane.index, fill)
    }
    fn push_index(
        &mut self,
        index: usize,
        fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if self.failed {
            return Err(Error::Consumed);
        }
        self.failed = true;
        let guard = PushGuard { build: self };
        let spec = guard.build.descriptor.wire_spec;
        if index != guard.build.next_lane || index >= spec.lanes {
            return Err(Error::Context);
        }
        // EXACT original draw order. Every partial/error/unwind wipes all build
        // payloads and leaves failed set before any caller can attempt a suffix.
        let r = Zeroizing::new(draw(spec.output_bits, fill)?);
        let q = Zeroizing::new(draw(1, fill)?);
        let r0 = Zeroizing::new(draw(spec.output_bits, fill)?);
        let q0 = Zeroizing::new(draw(1, fill)?);
        guard.build.arithmetic_masks[index] = *r;
        guard.build.predicate_masks[index] = *q;
        guard.build.masks[0].push([*r0, *q0]);
        guard.build.masks[1].push([r.wrapping_sub(*r0) & mask(spec.output_bits), *q ^ *q0]);
        guard.build.next_lane += 1;
        guard.build.failed = false;
        Ok(())
    }

    pub(super) fn finish(
        self,
        seal: LookupIssueSeal,
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> CorrectionResult<C> {
        self.finish_fields(seal.spec, seal.binding, seal.select_context, fill)
    }
    pub(super) fn finish_norm13(
        self,
        seal: Norm13IssueSeal,
        fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> CorrectionResult<C> {
        if seal.spec != ValidatedDescriptor::norm_inverse().0.wire_spec {
            return Err(Error::Context);
        }
        self.finish_fields(seal.spec, seal.binding, seal.select_context, fill)
    }
    fn finish_fields(
        self,
        sealed_spec: Spec,
        binding: [u8; 32],
        select_context: selection::Context,
        mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> CorrectionResult<C> {
        if self.failed {
            return Err(Error::Consumed);
        }
        if self.next_lane != self.descriptor.wire_spec.lanes
            || sealed_spec != self.descriptor.wire_spec
        {
            return Err(Error::Context);
        }
        let spec = self.descriptor.wire_spec;
        let mut owner = FinishingFor {
            masks: self.masks,
            correction: self.correction,
            rho: self.rho,
            predicate_masks: self.predicate_masks,
            arithmetic_masks: self.arithmetic_masks,
            select: Some(self.select_reservation),
            parties: self.party_reservations,
            dealer: self.dealer_reservation,
        };
        let select_reservation = owner.select.take().ok_or(Error::Consumed)?;
        let (select_masks, [select0, select1]) = selection::issue_reserved_with(
            spec.select(),
            select_context,
            &owner.predicate_masks,
            &owner.arithmetic_masks,
            spec.boundary,
            select_reservation,
            &mut fill,
        )?;
        for i in 0..spec.lanes {
            owner.rho[i] = select_masks.values()[i]
                .wrapping_mul(2)
                .wrapping_sub(owner.arithmetic_masks[i])
                & mask(spec.output_bits);
        }
        // No fallible factory or indexed work remains after these moves.
        let FinishingFor {
            masks: [left_masks, right_masks],
            correction: [left_correction, right_correction],
            rho,
            parties: [a, b],
            dealer: dealer_reservation,
            ..
        } = owner;
        let make = |role, masks, correction, select, lease| CorrectionFor {
            masks,
            retained_counts: None,
            state: Some(CorrectionState {
                arithmetic: Zeroizing::new(Vec::new()),
                correction,
                digest: [0; 32],
                phase: 0,
            }),
            select: Some(select),
            scan_lease: None,
            role,
            spec,
            binding,
            lease,
        };
        Ok((
            CorrectionMasksFor {
                values: rho,
                select_masks,
                dealer: dealer_reservation,
            },
            [
                make(0, left_masks, left_correction, select0, a),
                make(1, right_masks, right_correction, select1, b),
            ],
        ))
    }
}

pub(super) struct AbortOnDrop<'a, C: AllocationContext> {
    pub(super) correction: &'a mut CorrectionFor<C>,
    pub(super) armed: bool,
}
impl<C: AllocationContext> Drop for AbortOnDrop<'_, C> {
    fn drop(&mut self) {
        if self.armed {
            self.correction.cancel();
        }
    }
}

pub(super) fn scan_bytes<C: AllocationContext>(spec: Spec) -> Result<usize, Error> {
    add(&[
        size_of::<ScanResultFor<C>>(),
        mul(16, spec.lanes)?,
        2 * ALLOCATOR,
    ])
}

pub(super) fn control_bytes<C: AllocationContext>() -> usize {
    size_of::<Controls<'_, C>>()
}

impl<C: AllocationContext> CorrectionFor<C> {
    pub(super) fn cancel(&mut self) {
        self.state = None;
        self.retained_counts = None;
        self.masks.zeroize();
        if let Some(select) = &mut self.select {
            select.cancel();
        }
    }

    pub(super) fn begin(&mut self, scan: ScanResultFor<C>) -> Result<FrameFor<C>, Error> {
        let result = self.begin_inner(scan);
        if result.is_err() {
            self.cancel();
        }
        result
    }

    fn begin_inner(&mut self, scan: ScanResultFor<C>) -> Result<FrameFor<C>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if state.phase != 0 {
            return Err(Error::Consumed);
        }
        if !std::ptr::eq(scan.lease.budget.budget(), self.lease.budget.budget())
            || scan.role != self.role
            || scan.spec != self.spec
            || scan.binding != self.binding
        {
            return Err(Error::Context);
        }
        if scan.sums.len() != self.spec.lanes || scan.counts.len() != self.spec.lanes {
            return Err(Error::Domain);
        }
        // Validate provenance before reading scan words. No arbitrary raw words
        // can enter: only the parent scanner constructs ScanResultFor.
        let ScanResultFor {
            mut sums,
            counts,
            digest,
            lease,
            ..
        } = scan;
        self.scan_lease = Some(lease);
        #[cfg(test)]
        PANIC_BEGIN.with(|armed| {
            if armed.replace(false) {
                panic!("injected correction begin unwind");
            }
        });
        for i in 0..self.spec.lanes {
            sums[i] = sums[i].wrapping_add(self.masks[i][0]) & mask(self.spec.output_bits);
            state.correction[i] = super::correction(self.role, counts[i]) ^ self.masks[i][1];
        }
        state.arithmetic = sums;
        self.masks.zeroize();
        state.digest = digest;
        // Scan count backing retires before its original reservation.
        if matches!(self.spec.index_bits, 13 | 50) {
            self.retained_counts = Some(counts);
        } else {
            drop(counts);
        }
        let frame = self.first_frame(&state)?;
        state.phase = 1;
        self.state = Some(state);
        Ok(frame)
    }
}
