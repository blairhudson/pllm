//! D39 private four-lane GapARS12 terminal. Not a LayerNorm/source issuer.
//! Parent MUST price its concrete slots/handles/source and cumulative work
//! before activating these native owners. No production parent is installed.
use super::*;
use crate::shared_allocation::MAX_ALLOCATION_BYTES;
use zeroize::Zeroize;

const LANES: usize = 4;
const BITS: u8 = 50;
fn spec() -> arithmetic::Spec {
    arithmetic::Spec {
        bits: BITS,
        lanes: LANES,
        gate: arithmetic::Gate::GapArs { shift: 12 },
    }
}

pub(super) struct Resources {
    pub native: arithmetic::Resources,
    pub typed_native_bytes: usize,
    pub extra_bytes: usize,
    pub reservation_bytes: usize,
    pub extra_random_bytes: usize,
    pub material_payload_bytes: usize,
    pub terminal_frames_per_party: usize,
    pub terminal_frame_bytes: usize,
    pub dealer_mask_sums: usize,
    pub dealer_sum_reductions: usize,
    pub split_draw_masks: usize,
    pub split_subtractions: usize,
    pub split_reductions: usize,
    pub beta_additions_per_party: usize,
    pub h_subtractions_per_party: usize,
    pub reductions_per_party: usize,
}

// All fields inline except original native/common backing. Leases last.
pub(in crate::masked_wire_reference) struct MaterialFor<C: AllocationContext> {
    h: Zeroizing<[u64; LANES]>,
    role: u8,
    native_binding: [u8; 32],
    beta_binding: Binding,
    lease: LeaseFor<C>,
}
pub(super) struct PartyFor<C: AllocationContext> {
    arithmetic: ArithmeticPartyFor<C, Ring>,
    material: MaterialFor<C>,
}
pub(super) struct ShareFor<C: AllocationContext> {
    local: arithmetic::TerminalLocalFor<C>,
    _beta: CommonMaskedFor<C, Ring>,
    _material: MaterialFor<C>,
    _typed: LeaseFor<C>,
}

// Concrete query/issuance/move-result controls. No second coordinator: parent
// adds its actual CoordinationFor/RoleSlotFor and retained frame handles.
struct Controls<C: AllocationContext> {
    _quote: Resources,
    _quote_result: Result<Resources, Error>,
    _output: EdgeDescriptor,
    _binding: Binding,
    _native_binding: [u8; 32],
    _reservation: Option<LeaseFor<C>>,
    _return_slot: Result<[PartyFor<C>; 2], Error>,
    _issued: Option<ArithmeticIssuedFor<C, Ring>>,
    _parties: Option<[PartyFor<C>; 2]>,
    _result: Option<Result<ShareFor<C>, Error>>,
    _material: [Option<MaterialFor<C>>; 2],
    _shares: [Option<ShareFor<C>>; 2],
    _split: Zeroizing<[u64; 2]>,
    _sum: Zeroizing<u64>,
    _draw: Zeroizing<[u8; 8]>,
}

pub(super) fn resources<C: AllocationContext>() -> Result<Resources, Error> {
    let native = arithmetic::resources_for_context::<C>(spec())?;
    let typed_native_bytes = arithmetic_reservation_bytes_for::<C, Ring>(spec())?;
    let extra_bytes = [
        2usize.checked_mul(size_of::<MaterialFor<C>>()),
        2usize.checked_mul(size_of::<ShareFor<C>>()),
        Some(size_of::<Controls<C>>()),
        Some(arithmetic::terminal_move_bytes::<C>()),
        4usize.checked_mul(C::TRANSIENT_LINK_BYTES),
    ]
    .into_iter()
    .try_fold(0usize, |a, b| {
        a.checked_add(b.ok_or(Error::Resource)?)
            .ok_or(Error::Resource)
    })?;
    let reservation_bytes = typed_native_bytes
        .checked_add(extra_bytes)
        .ok_or(Error::Resource)?;
    if reservation_bytes > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        typed_native_bytes,
        extra_bytes,
        reservation_bytes,
        extra_random_bytes: LANES * size_of::<u64>(),
        material_payload_bytes: 2 * size_of::<[u64; LANES]>(),
        terminal_frames_per_party: 1,
        terminal_frame_bytes: native.frame_bytes,
        dealer_mask_sums: LANES,
        dealer_sum_reductions: LANES,
        split_draw_masks: LANES,
        split_subtractions: LANES,
        split_reductions: LANES,
        beta_additions_per_party: LANES,
        h_subtractions_per_party: LANES,
        reductions_per_party: LANES,
        native,
    })
}

/// Original typed dealer consumes real floor-input/beta masks. No caller mask
/// words, digest adoption, output-source declaration or clear-root recovery.
#[allow(clippy::too_many_arguments)]
pub(super) fn issue_reserved_with<C: AllocationContext>(
    dealer: &mut LocalDealerFor<C>,
    operation: [u8; 32],
    input: DealerMaskFor<C, Ring>,
    beta: DealerMaskFor<C, Ring>,
    output: EdgeDescriptor,
    next: [u8; 32],
    mut reservation: LeaseFor<C>,
    fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<[PartyFor<C>; 2], Error> {
    let q = resources::<C>()?;
    if reservation.bytes != q.reservation_bytes
        || !std::ptr::eq(reservation.budget.budget(), dealer.budget.budget())
        || !std::ptr::eq(beta._lease.budget.budget(), dealer.budget.budget())
        || !std::ptr::eq(input._lease.budget.budget(), dealer.budget.budget())
    {
        return Err(Error::Resource);
    }
    dealer.input(beta.binding, operation)?;
    if beta.binding.descriptor.numeric() != output.numeric()
        || output.numeric().ring_bits() != BITS
        || output.numeric().fractional_bits() != 12
        || beta.binding.descriptor.shape() != output.shape()
        || output.shape().elements() != LANES
    {
        return Err(Error::Domain);
    }
    let native = reservation.split(q.typed_native_bytes)?;
    let left = reservation.split(q.extra_bytes / 2)?;
    let (floor, [a, b]) =
        dealer.gap_ars_reserved_with(operation, input, 12, output, next, native, fill)?;
    let (aid, abinding) = a
        .native
        .as_ref()
        .ok_or(Error::Consumed)?
        .terminal_identity()?;
    let (bid, bbinding) = b
        .native
        .as_ref()
        .ok_or(Error::Consumed)?
        .terminal_identity()?;
    if aid != 0 || bid != 1 || abinding != bbinding {
        return Err(Error::Context);
    }
    let mut h0 = Zeroizing::new([0u64; LANES]);
    let mut h1 = Zeroizing::new([0u64; LANES]);
    for i in 0..LANES {
        let sum = Zeroizing::new(
            floor.backing.values()[i].wrapping_add(beta.backing.values()[i])
                & super::super::mask(BITS),
        );
        let split = Zeroizing::new(arithmetic::split(*sum, BITS, fill)?);
        h0[i] = split[0];
        h1[i] = split[1];
    }
    let binding = beta.binding;
    Ok([
        PartyFor {
            arithmetic: a,
            material: MaterialFor {
                h: h0,
                role: 0,
                native_binding: abinding,
                beta_binding: binding,
                lease: left,
            },
        },
        PartyFor {
            arithmetic: b,
            material: MaterialFor {
                h: h1,
                role: 1,
                native_binding: bbinding,
                beta_binding: binding,
                lease: reservation,
            },
        },
    ])
}

impl<C: AllocationContext> MaterialFor<C> {
    fn check(
        &self,
        role: u8,
        native_binding: [u8; 32],
        beta: &CommonMaskedFor<C, Ring>,
    ) -> Result<(), Error> {
        if role != self.role
            || native_binding != self.native_binding
            || beta.role != role
            || beta.binding != self.beta_binding
            || beta.backing.values().len() != LANES
            || !std::ptr::eq(self.lease.budget.budget(), beta._lease.budget.budget())
        {
            return Err(Error::Context);
        }
        Ok(())
    }

    pub(in crate::masked_wire_reference) fn apply(
        &mut self,
        role: u8,
        native_binding: [u8; 32],
        beta: &CommonMaskedFor<C, Ring>,
        local: &mut [u64],
    ) -> Result<(), Error> {
        self.check(role, native_binding, beta)?;
        if local.len() != LANES {
            return Err(Error::Domain);
        }
        for (i, value) in local.iter_mut().enumerate() {
            *value = value
                .wrapping_add(if role == 1 {
                    beta.backing.values()[i]
                } else {
                    0
                })
                .wrapping_sub(self.h[i])
                & super::super::mask(BITS);
        }
        self.h.zeroize();
        Ok(())
    }
}

impl<C: AllocationContext> PartyFor<C> {
    pub(super) fn begin(
        &mut self,
        input: CommonMaskedFor<C, Ring>,
    ) -> Result<arithmetic::FrameFor<C>, Error> {
        self.arithmetic.begin(input)
    }
    /// Consuming native terminal. Parent must run inside its existing operation
    /// guard, store this secret in its role slot, then require BOTH ready.
    pub(super) fn finish(
        self,
        beta: CommonMaskedFor<C, Ring>,
        peer: &[u8],
    ) -> Result<ShareFor<C>, Error> {
        let Self {
            mut arithmetic,
            mut material,
        } = self;
        let native = arithmetic.native.take().ok_or(Error::Consumed)?;
        let (role, binding) = native.terminal_identity()?;
        material.check(role, binding, &beta)?;
        let mut local = native.finish_terminal_carry(peer)?;
        local.add_beta(&mut material, &beta)?;
        Ok(ShareFor {
            local,
            _beta: beta,
            _material: material,
            _typed: arithmetic.lease.take().ok_or(Error::Consumed)?,
        })
    }
}

#[cfg(test)]
#[path = "gapars_terminal_tests.rs"]
mod tests;
