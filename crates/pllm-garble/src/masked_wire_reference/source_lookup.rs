//! Bounded SIGMA source lookup over the ascending GPU127 complete expansion.
//!
//! This is a local, unauthenticated research primitive, not a typed decoder
//! bridge, remote transport, full activation, or security qualification.
//! Reconstructed output is `T[x] + rho`; deferred outputs are additive shares
//! of that value. The dealer samples full-ring `r,u` independently and sets
//! `rho = 2*u-r`: in particular `rho mod 2 = r mod 2`, not independent masks.

use super::{mask, selection};
use crate::compact_dpf::gpu127::{self, GpuKey, Profile};
use crate::shared_allocation::{allocate, AllocationContext, LeaseFor, MAX_ALLOCATION_BYTES};
pub use crate::shared_allocation::{Budget, Error};
pub use selection::Boundary;
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

mod correction;
use correction::{
    CorrectionBuildFor, CorrectionDescriptor, CorrectionFor, CorrectionState, ScanResultFor,
};

// Created only at the original validated parent issuance sites below. Neither
// seal is public, cloneable or deserializable; a digest is not an issuer.
struct LookupIssueSeal {
    spec: Spec,
    binding: [u8; 32],
    select_context: selection::Context,
}
struct IssuedKeyLane {
    index: usize,
}

// Distinct private family: never manufacture the ordinary D19 lane/seal from
// a caller profile or table. Only issue_parts after real GPU13 key issuance.
struct Norm13KeyLane {
    index: usize,
}
struct Norm13IssueSeal {
    spec: Spec,
    binding: [u8; 32],
    select_context: selection::Context,
}
type LookupResult<C> = Result<(OutputMasksFor<C>, [PartyFor<C>; 2]), Error>;

#[allow(dead_code)] // Concrete quote-only simultaneous call/move/result slots.
struct ParentControls<'a, C: AllocationContext> {
    quote: Resources,
    quote_result: Result<Resources, Error>,
    correction_quote: correction::CorrectionResources,
    correction_quote_result: Result<correction::CorrectionResources, Error>,
    descriptor: CorrectionDescriptor,
    context: Context,
    table: &'a [u64],
    input_masks: &'a [u64],
    nonce: [u8; 32],
    table_identity: [u8; 32],
    binding: [u8; 32],
    seal: Option<LookupIssueSeal>,
    seal_result: Result<LookupIssueSeal, Error>,
    lane_ticket: Option<IssuedKeyLane>,
    lane_ticket_result: Result<IssuedKeyLane, Error>,
    lane_index: usize,
    key_profile: Profile,
    key_profile_result: Result<Profile, gpu127::Error>,
    aggregate: Option<LeaseFor<C>>,
    aggregate_result: Result<LeaseFor<C>, Error>,
    correction_reservation: Option<LeaseFor<C>>,
    scanner_reservations: [Option<LeaseFor<C>>; 2],
    scanner_dealer: Option<LeaseFor<C>>,
    correction_dealer_move: Option<LeaseFor<C>>,
    scanner_dealer_move: Option<LeaseFor<C>>,
    split_results: [Result<LeaseFor<C>, Error>; 3],
    build: Option<CorrectionBuildFor<C>>,
    build_result: Result<CorrectionBuildFor<C>, Error>,
    scanner_executions: [Option<Execution>; 2],
    scanner_execution_results: [Result<Execution, Error>; 2],
    child_result: correction::CorrectionResult<C>,
    child_masks_move: Option<correction::CorrectionMasksFor<C>>,
    child_parties_move: Option<[CorrectionFor<C>; 2]>,
    parent_parties_move: Option<[PartyFor<C>; 2]>,
    output_move: Option<OutputMasksFor<C>>,
    final_result: LookupResult<C>,
    allocation_link: Option<C>,
}

#[allow(dead_code)] // Scanner-only controls; never enter the key/table-free child.
struct ScannerControls<'a, C: AllocationContext> {
    execution: Option<Execution>,
    abort: Option<correction::AbortOnDrop<'a, C>>,
    scan: Option<ScanResultFor<C>>,
}

pub const DOMAIN: &str = "pllm.sigma.source-lookup.gpu127.v1";
pub const MAX_LANES: usize = super::MAX_LANES;
const MAGIC: &[u8; 8] = b"PLLMLU01";
const HEADER: usize = 80;
const ALLOCATOR: usize = 64;
const WORKSPACE: usize = 8 << 10;

pub(super) mod interval50;
pub(super) mod norm13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub index_bits: u8,
    pub output_bits: u8,
    pub lanes: usize,
    pub boundary: Boundary,
}
impl Spec {
    fn validate(self) -> Result<(), Error> {
        if !matches!(self.index_bits, 8 | 10)
            || !(3..=64).contains(&self.output_bits)
            || !(1..=MAX_LANES).contains(&self.lanes)
        {
            return Err(Error::Domain);
        }
        Ok(())
    }
    fn select(self) -> selection::Spec {
        selection::Spec {
            bits: self.output_bits,
            lanes: self.lanes,
        }
    }
    fn entries(self) -> usize {
        1usize << self.index_bits
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub plan: [u8; 32],
    pub session: [u8; 32],
    pub operation: [u8; 32],
    pub input_wire: [u8; 32],
    pub output_wire: [u8; 32],
    pub arithmetic_wire: [u8; 32],
    pub correction_wire: [u8; 32],
    pub select_wire: [u8; 32],
    pub numeric_contract: [u8; 32],
    pub first_tensor_index: u64,
}
impl Context {
    fn validate(self, lanes: usize) -> Result<(), Error> {
        let wires = [
            self.input_wire,
            self.output_wire,
            self.arithmetic_wire,
            self.correction_wire,
            self.select_wire,
        ];
        for i in 0..wires.len() {
            if wires[..i].contains(&wires[i]) {
                return Err(Error::Context);
            }
        }
        self.first_tensor_index
            .checked_add(lanes as u64)
            .ok_or(Error::Resource)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub index_bits: u8,
    pub output_bits: u8,
    pub table_entries: usize,
    pub logical_key_bits_per_party: usize,
    pub random_bytes: usize,
    pub output_mask_entropy_bits: usize,
    pub paired_issue_aes_blocks: usize,
    pub evaluation_aes_blocks_per_party: usize,
    pub frames_per_party: usize,
    /// Lookup's first frame. Select's unchanged second frame is separately listed.
    pub frame_bytes: usize,
    pub select_frame_bytes: usize,
    pub total_peer_frame_bytes: usize,
    pub party_material_payload_bytes: usize,
    pub borrowed_input_mask_bytes: usize,
    pub borrowed_table_bytes: usize,
    pub borrowed_masked_input_bytes: usize,
    pub full_expansion_scratch_bytes: usize,
    pub dealer_reservation_bytes: usize,
    pub party_reservation_bytes: usize,
    pub combined_reservation_bytes: usize,
}

fn add(parts: &[usize]) -> Result<usize, Error> {
    parts
        .iter()
        .try_fold(0usize, |n, &v| n.checked_add(v).ok_or(Error::Resource))
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}
fn first_frame_bytes(spec: Spec) -> usize {
    HEADER + 16 * spec.lanes
}
#[cfg(test)]
fn frame_charge(spec: Spec) -> usize {
    frame_charge_for::<&Budget>(spec)
}
#[cfg(test)]
fn output_charge(spec: Spec) -> usize {
    output_charge_for::<&Budget>(spec)
}
fn frame_charge_for<C: AllocationContext>(spec: Spec) -> usize {
    first_frame_bytes(spec) + size_of::<FrameFor<C>>() + ALLOCATOR
}
fn output_charge_for<C: AllocationContext>(spec: Spec) -> usize {
    // Charge the larger capability even in reconstructed mode.
    8 * spec.lanes
        + size_of::<DeferredOutputFor<C>>().max(size_of::<MaskedOutputFor<C>>())
        + ALLOCATOR
}

/// Aggregate logical live capacity, not process RSS. Includes nested Select,
/// actual padded inline GPU127 lane owners, two immutable tables, caller/peer
/// allowances, retained frames and output overlaps, and lane-serial N-byte scan.
/// Eight KiB per owner covers bounded hash/AES/stack work; dealer additionally
/// prices the transient paired inline keys during issuance.
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    resources_for_context::<&Budget>(spec)
}

pub(crate) fn resources_for_context<C: AllocationContext>(spec: Spec) -> Result<Resources, Error> {
    spec.validate()?;
    let gpu = Profile::new(spec.index_bits)
        .map_err(gpu_error)?
        .resources();
    let child = correction::resources::<C>(CorrectionDescriptor { wire_spec: spec })?;
    resources_parts::<C>(spec, gpu, child)
}

fn resources_parts<C: AllocationContext>(
    spec: Spec,
    gpu: gpu127::Resources,
    child: correction::CorrectionResources,
) -> Result<Resources, Error> {
    let select = child.select;
    let table = mul(8, spec.entries())?;
    let lane = mul(size_of::<LookupLane>(), spec.lanes)?;
    let select_frame = if spec.boundary == Boundary::Reconstructed {
        select.frame_bytes
    } else {
        0
    };
    let scanner = add(&[
        size_of::<PartyFor<C>>()
            .checked_sub(size_of::<CorrectionFor<C>>())
            .ok_or(Error::Resource)?,
        lane,
        table,
        2 * ALLOCATOR,
        mul(8, spec.lanes)?, // borrowed common masked indices
        spec.entries(),      // one lane-serial complete expansion
        size_of::<Zeroizing<Vec<u8>>>() + ALLOCATOR,
        size_of::<ScannerControls<'_, C>>(),
        WORKSPACE,
    ])?;
    let scanner_dealer = add(&[
        mul(8, spec.lanes)?, // borrowed input masks
        table,               // borrowed complete public table
        gpu.pair_owner_bytes,
        WORKSPACE,
        size_of::<ParentControls<'_, C>>(),
        size_of::<OutputMasksFor<C>>()
            .checked_sub(size_of::<correction::CorrectionMasksFor<C>>())
            .ok_or(Error::Resource)?,
    ])?;
    let party = add(&[scanner, correction::scan_bytes::<C>(spec)?, child.party])?;
    let dealer = add(&[scanner_dealer, child.dealer])?;
    let combined = add(&[dealer, mul(2, party)?])?;
    if combined > MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    Ok(Resources {
        index_bits: spec.index_bits,
        output_bits: spec.output_bits,
        table_entries: spec.entries(),
        logical_key_bits_per_party: mul(
            gpu.logical_key_bits + 6 * usize::from(spec.output_bits) + 1,
            spec.lanes,
        )?,
        // Lookup nonce + GPU paired roots + r,q and independent additive/XOR
        // mask shares. Select prices its own nonce and six full-word draws.
        random_bytes: add(&[32, mul(64, spec.lanes)?, select.random_bytes])?,
        output_mask_entropy_bits: mul(usize::from(spec.output_bits), spec.lanes)?,
        paired_issue_aes_blocks: mul(gpu.paired_issue_aes_blocks, spec.lanes)?,
        evaluation_aes_blocks_per_party: mul(
            gpu.full_expansion_aes_blocks.ok_or(Error::Resource)?,
            spec.lanes,
        )?,
        frames_per_party: if select_frame == 0 { 1 } else { 2 },
        frame_bytes: first_frame_bytes(spec),
        select_frame_bytes: select_frame,
        total_peer_frame_bytes: mul(2, add(&[first_frame_bytes(spec), select_frame])?)?,
        party_material_payload_bytes: mul(spec.lanes, gpu.codec_bytes + 16 + 40)?,
        borrowed_input_mask_bytes: mul(8, spec.lanes)?,
        borrowed_table_bytes: table,
        borrowed_masked_input_bytes: mul(16, spec.lanes)?,
        full_expansion_scratch_bytes: spec.entries(),
        dealer_reservation_bytes: dealer,
        party_reservation_bytes: party,
        combined_reservation_bytes: combined,
    })
}

fn gpu_error(error: gpu127::Error) -> Error {
    match error {
        gpu127::Error::Domain => Error::Domain,
        gpu127::Error::Encoding => Error::Context,
        gpu127::Error::Resource => Error::Resource,
        gpu127::Error::Allocation => Error::Allocation,
        gpu127::Error::Randomness => Error::Randomness,
    }
}
fn allocation_error(error: Error) -> gpu127::Error {
    match error {
        Error::Allocation => gpu127::Error::Allocation,
        _ => gpu127::Error::Resource,
    }
}
fn zeros(len: usize, budget: &Budget) -> Result<Zeroizing<Vec<u64>>, Error> {
    let mut out = Zeroizing::new(allocate(len, budget)?);
    out.resize(len, 0);
    Ok(out)
}
fn draw(bits: u8, fill: &mut impl FnMut(&mut [u8]) -> Result<(), Error>) -> Result<u64, Error> {
    let mut bytes = Zeroizing::new([0; 8]);
    fill(&mut *bytes)?;
    Ok(u64::from_le_bytes(*bytes) & mask(bits))
}
fn boundary_byte(boundary: Boundary) -> u8 {
    match boundary {
        Boundary::Reconstructed => 0,
        Boundary::Deferred => 1,
    }
}

struct LookupLane {
    key: GpuKey,
}
// GpuKey erases on drop; correction owns the separate zeroizing mask backing.
// Ordinary Vec drop invokes every initialized key on partial issuance.
struct Execution {
    lanes: Vec<LookupLane>,
    table: Zeroizing<Vec<u64>>,
    sums: Zeroizing<Vec<u64>>,
    counts: Zeroizing<Vec<i64>>,
}
pub type Party<'a> = PartyFor<&'a Budget>;
pub struct PartyFor<C: AllocationContext> {
    role: u8,
    spec: Spec,
    binding: [u8; 32],
    state: Option<Execution>,
    // Cancellation burns correction state without refunding its retained Select
    // metadata/material reservation before the original Party drops.
    correction: CorrectionFor<C>,
    // Secret owners must drop before leases, including on error paths.
    lease: LeaseFor<C>,
}
pub type OutputMasks<'a> = OutputMasksFor<&'a Budget>;
pub struct OutputMasksFor<C: AllocationContext> {
    values: Zeroizing<Vec<u64>>,
    _select_masks: selection::OutputMasksFor<C>,
    _correction_dealer: LeaseFor<C>,
    _scanner_dealer: LeaseFor<C>,
}
impl<C: AllocationContext> OutputMasksFor<C> {
    /// Low-level trusted dealer view only. rho=2u-r, hence rho mod 2=r mod 2.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}
enum FrameBody<C: AllocationContext> {
    Lookup(Zeroizing<Vec<u8>>),
    Select(selection::FrameFor<C>),
}
pub type Frame<'a> = FrameFor<&'a Budget>;
pub struct FrameFor<C: AllocationContext> {
    body: FrameBody<C>,
    _lease: LeaseFor<C>,
}
impl<C: AllocationContext> FrameFor<C> {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.body {
            FrameBody::Lookup(bytes) => bytes,
            FrameBody::Select(frame) => frame.as_bytes(),
        }
    }
}
pub type MaskedOutput<'a> = MaskedOutputFor<&'a Budget>;
pub struct MaskedOutputFor<C: AllocationContext> {
    values: Zeroizing<Vec<u64>>,
    binding: [u8; 32],
    bits: u8,
    _lease: LeaseFor<C>,
}
impl<C: AllocationContext> MaskedOutputFor<C> {
    /// Common masked value T[x]+rho, not clear T[x].
    pub fn values(&self) -> &[u64] {
        &self.values
    }
    pub fn binding(&self) -> &[u8; 32] {
        &self.binding
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn lanes(&self) -> usize {
        self.values.len()
    }
}
pub type DeferredOutput<'a> = DeferredOutputFor<&'a Budget>;
pub struct DeferredOutputFor<C: AllocationContext> {
    values: Zeroizing<Vec<u64>>,
    role: u8,
    binding: [u8; 32],
    bits: u8,
    _lease: LeaseFor<C>,
}
impl<C: AllocationContext> DeferredOutputFor<C> {
    /// Own additive share, never a reconstructed/common masked value.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
    pub fn role(&self) -> u8 {
        self.role
    }
    pub fn binding(&self) -> &[u8; 32] {
        &self.binding
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn lanes(&self) -> usize {
        self.values.len()
    }
}
pub type Step<'a> = StepFor<&'a Budget>;
pub enum StepFor<C: AllocationContext> {
    Frame(FrameFor<C>),
    Reconstructed(MaskedOutputFor<C>),
    Deferred(DeferredOutputFor<C>),
}

pub fn issue<'a>(
    spec: Spec,
    context: Context,
    table: &[u64],
    input_masks: &[u64],
    budget: &'a Budget,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    issue_with(spec, context, table, input_masks, budget, |bytes| {
        getrandom::fill(bytes).map_err(|_| Error::Randomness)
    })
}

fn issue_with<'a>(
    spec: Spec,
    context: Context,
    table: &[u64],
    input_masks: &[u64],
    budget: &'a Budget,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    let cost = resources(spec)?;
    context.validate(spec.lanes)?;
    validate_inputs(spec, table, input_masks)?;
    let reservation = budget.reserve(cost.combined_reservation_bytes)?;
    issue_reserved_with(spec, context, table, input_masks, reservation, fill)
}

fn validate_inputs(spec: Spec, table: &[u64], input_masks: &[u64]) -> Result<(), Error> {
    if table.len() != spec.entries()
        || input_masks.len() != spec.lanes
        || table.iter().any(|&v| v & !mask(spec.output_bits) != 0)
        || input_masks.iter().any(|&v| v & !mask(spec.index_bits) != 0)
    {
        return Err(Error::Domain);
    }
    Ok(())
}

/// Consume original aggregate admission, including the nested Select lease.
pub(super) fn issue_reserved_with<C: AllocationContext>(
    spec: Spec,
    context: Context,
    table: &[u64],
    input_masks: &[u64],
    reservation: LeaseFor<C>,
    fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(OutputMasksFor<C>, [PartyFor<C>; 2]), Error> {
    let cost = resources_for_context::<C>(spec)?;
    let descriptor = CorrectionDescriptor { wire_spec: spec };
    let child = correction::resources::<C>(descriptor)?;
    issue_parts(
        spec,
        context,
        table,
        input_masks,
        reservation,
        fill,
        cost,
        child,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn issue_parts<C: AllocationContext>(
    spec: Spec,
    context: Context,
    table: &[u64],
    input_masks: &[u64],
    mut reservation: LeaseFor<C>,
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    cost: Resources,
    child: correction::CorrectionResources,
    private: Option<correction::ValidatedDescriptor>,
) -> Result<(OutputMasksFor<C>, [PartyFor<C>; 2]), Error> {
    context.validate(spec.lanes)?;
    validate_inputs(spec, table, input_masks)?;
    if reservation.bytes != cost.combined_reservation_bytes {
        return Err(Error::Resource);
    }
    let allocation = reservation.budget.share();
    let budget = allocation.budget();
    // Entire public and canonical input preflight precedes admission, entropy,
    // and allocation. Identity hashes full canonical bytes, not a caller label.
    let identity = table_identity(spec, context.numeric_contract, table);
    let descriptor = CorrectionDescriptor { wire_spec: spec };
    let child_reservation = reservation.split(child.combined)?;
    let scanner_bytes = cost
        .party_reservation_bytes
        .checked_sub(child.party)
        .ok_or(Error::Resource)?;
    let scanner_a = reservation.split(scanner_bytes)?;
    let scanner_b = reservation.split(scanner_bytes)?;
    if reservation.bytes
        != cost
            .dealer_reservation_bytes
            .checked_sub(child.dealer)
            .ok_or(Error::Resource)?
    {
        return Err(Error::Resource);
    }
    // Declare all reservations before secret owners: reverse local drop order
    // erases backing before refunds on every early return.
    let mut build = match private {
        Some(profile) => correction::prepare_validated(profile, child_reservation)?,
        None => correction::prepare_reserved(descriptor, child_reservation)?,
    };
    let new_execution = || -> Result<Execution, Error> {
        let mut own_table = zeros(spec.entries(), budget)?;
        own_table.copy_from_slice(table);
        Ok(Execution {
            lanes: allocate(spec.lanes, budget)?,
            table: own_table,
            sums: zeros(spec.lanes, budget)?,
            counts: {
                let mut counts = Zeroizing::new(allocate(spec.lanes, budget)?);
                counts.resize(spec.lanes, 0i64);
                counts
            },
        })
    };
    let mut left = new_execution()?;
    let mut right = new_execution()?;
    let mut nonce = [0; 32];
    fill(&mut nonce)?;
    let mut binding = invocation_binding(spec, context, identity, nonce);
    if private.is_some() {
        let mut hash = Sha256::new();
        hash.update(b"pllm.sigma.layernorm4.inverse13.invocation.v1");
        hash.update(binding);
        binding = hash.finalize().into();
    }
    let profile = Profile::new(spec.index_bits).map_err(gpu_error)?;
    for (i, &rin) in input_masks.iter().enumerate() {
        let [k0, k1] = GpuKey::issue_with(
            profile,
            0u64.wrapping_sub(rin) & mask(spec.index_bits),
            |bytes| fill(bytes).map_err(|_| gpu127::Error::Randomness),
        )
        .map_err(gpu_error)?;
        left.lanes.push(LookupLane { key: k0 });
        right.lanes.push(LookupLane { key: k1 });
        // Only this original-key creation site mints the one-shot lane ticket.
        if private.is_some() {
            build.push_norm13_keypair(Norm13KeyLane { index: i }, &mut fill)?;
        } else {
            build.push_after_keypair(IssuedKeyLane { index: i }, &mut fill)?;
        }
    }
    let (
        correction::CorrectionMasksFor {
            values,
            select_masks,
            dealer,
        },
        [correction0, correction1],
    ) = if private.is_some() {
        build.finish_norm13(
            Norm13IssueSeal {
                spec,
                binding,
                select_context: select_context(context, binding),
            },
            &mut fill,
        )?
    } else {
        build.finish(
            LookupIssueSeal {
                spec,
                binding,
                select_context: select_context(context, binding),
            },
            &mut fill,
        )?
    };
    Ok((
        OutputMasksFor {
            values,
            _select_masks: select_masks,
            _correction_dealer: dealer,
            _scanner_dealer: reservation,
        },
        [
            PartyFor {
                role: 0,
                spec,
                binding,
                state: Some(left),
                correction: correction0,
                lease: scanner_a,
            },
            PartyFor {
                role: 1,
                spec,
                binding,
                state: Some(right),
                correction: correction1,
                lease: scanner_b,
            },
        ],
    ))
}

fn table_identity(spec: Spec, numeric_contract: [u8; 32], table: &[u64]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"pllm.sigma.source-lookup.canonical-table.v1");
    hash.update([spec.index_bits, spec.output_bits]);
    hash.update(numeric_contract);
    for value in table {
        hash.update(value.to_le_bytes());
    }
    hash.finalize().into()
}
fn invocation_binding(
    spec: Spec,
    context: Context,
    table_identity: [u8; 32],
    nonce: [u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(DOMAIN.as_bytes());
    hash.update(MAGIC);
    hash.update(Profile::DOMAIN.as_bytes());
    hash.update(b"EzPC:aa0f2f860266fb79901405ef5e9965b8c9fe2e88:LUT_dfpet");
    hash.update([
        spec.index_bits,
        spec.output_bits,
        boundary_byte(spec.boundary),
    ]);
    hash.update((spec.lanes as u64).to_le_bytes());
    for part in [
        context.plan,
        context.session,
        context.operation,
        context.input_wire,
        context.output_wire,
        context.arithmetic_wire,
        context.correction_wire,
        context.select_wire,
        context.numeric_contract,
        table_identity,
        nonce,
    ] {
        hash.update(part);
    }
    hash.update(context.first_tensor_index.to_le_bytes());
    hash.finalize().into()
}
fn select_context(context: Context, binding: [u8; 32]) -> selection::Context {
    let mut hash = Sha256::new();
    hash.update(b"pllm.sigma.source-lookup.select-context.v1");
    hash.update(binding);
    selection::Context {
        plan: context.plan,
        session: context.session,
        operation: hash.finalize().into(),
        predicate_wire: context.correction_wire,
        input_wire: context.arithmetic_wire,
        output_wire: context.select_wire,
        first_tensor_index: context.first_tensor_index,
    }
}

/// Source's modulo-four correction uses each complete signed count, NOT an
/// assumption that individual counts are +/-1. n<=10 bounds signed arithmetic.
fn correction(role: u8, signed_count: i64) -> u64 {
    if role == 0 {
        ((signed_count + 1).rem_euclid(4) / 2) as u64
    } else {
        (2 - (4 - signed_count.rem_euclid(4)) / 2).rem_euclid(2) as u64
    }
}

impl<C: AllocationContext> PartyFor<C> {
    /// Private invocation identity for owning-region deferred-output checks.
    pub(super) fn internal_binding(&self) -> [u8; 32] {
        self.binding
    }

    /// Burns execution; retained owners and their outstanding leases remain
    /// charged until drop. Sibling cancellation is caller's responsibility.
    pub fn cancel(&mut self) {
        self.state = None;
        self.correction.cancel();
    }

    pub fn begin(&mut self, masked_indices: &[u64]) -> Result<FrameFor<C>, Error> {
        let result = self.begin_inner(masked_indices);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn begin_inner(&mut self, masked_indices: &[u64]) -> Result<FrameFor<C>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        let mut abort = correction::AbortOnDrop {
            correction: &mut self.correction,
            armed: true,
        };
        if state.sums.len() != self.spec.lanes || state.counts.len() != self.spec.lanes {
            return Err(Error::Consumed);
        }
        if masked_indices.len() != self.spec.lanes
            || masked_indices
                .iter()
                .any(|&v| v & !mask(self.spec.index_bits) != 0)
        {
            return Err(Error::Domain);
        }
        let mut hash = Sha256::new();
        hash.update(b"pllm.sigma.source-lookup.masked-indices.v1");
        hash.update(self.binding);
        for &index in masked_indices {
            hash.update(index.to_le_bytes());
        }
        let digest = hash.finalize().into();
        for (lane, &index) in masked_indices.iter().enumerate() {
            let material = &state.lanes[lane];
            let expanded = if self.spec.index_bits == 13 {
                material
                    .key
                    .all_layernorm13_with_allocator(self.spec.entries(), |n| {
                        allocate(n, self.lease.budget.budget()).map_err(allocation_error)
                    })
            } else {
                material.key.all_with_allocator(self.spec.entries(), |n| {
                    allocate(n, self.lease.budget.budget()).map_err(allocation_error)
                })
            }
            .map_err(gpu_error)?;
            let mut sum = Zeroizing::new(0u64);
            let mut count = Zeroizing::new(0i64);
            // GPU127 outputs already use ascending indices and LSB-first leaf
            // offsets. Neither bit reversal nor CPU terminal mapping belongs here.
            for (i, &bit) in expanded.iter().enumerate() {
                *count += i64::from(bit);
                *sum = sum.wrapping_add(
                    state.table[(i + index as usize) & (self.spec.entries() - 1)]
                        .wrapping_mul(u64::from(bit)),
                );
            }
            if self.role == 1 {
                *sum = 0u64.wrapping_sub(*sum);
                *count = -*count;
            }
            state.sums[lane] = *sum;
            state.counts[lane] = *count;
        }
        let scan = ScanResultFor {
            sums: std::mem::take(&mut state.sums),
            counts: std::mem::take(&mut state.counts),
            role: self.role,
            spec: self.spec,
            binding: self.binding,
            digest,
            lease: self.lease.split(correction::scan_bytes::<C>(self.spec)?)?,
        };
        let frame = abort.correction.begin(scan)?;
        // Only a successful correction handoff restores scanner custody. On
        // error or unwind the local execution erases its keys/table immediately.
        self.state = Some(state);
        abort.armed = false;
        Ok(frame)
    }

    pub fn advance(&mut self, peer: &[u8]) -> Result<StepFor<C>, Error> {
        self.advance_residency(peer, false)
    }
    fn advance_residency(
        &mut self,
        peer: &[u8],
        retain_completed: bool,
    ) -> Result<StepFor<C>, Error> {
        // Take locally, as before extraction: error/terminal/unwind retires the
        // actual scanner owners even when the caller catches a panic.
        let state = self.state.take().ok_or(Error::Consumed)?;
        let mut abort = correction::AbortOnDrop {
            correction: &mut self.correction,
            armed: true,
        };
        let step = abort.correction.advance(peer)?;
        if retain_completed || matches!(&step, StepFor::Frame(_)) {
            self.state = Some(state);
        }
        abort.armed = false;
        Ok(step)
    }
}

impl<C: AllocationContext> CorrectionFor<C> {
    fn advance(&mut self, peer: &[u8]) -> Result<StepFor<C>, Error> {
        let result = self.advance_inner(peer);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn advance_inner(&mut self, peer: &[u8]) -> Result<StepFor<C>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        #[cfg(test)]
        correction::PANIC_ADVANCE.with(|armed| {
            if armed.replace(false) {
                panic!("injected correction advance unwind");
            }
        });
        match state.phase {
            1 => {
                self.validate_first(&state, peer)?;
                // Complete header AND all canonical payload words validated
                // before mutating either opened array or invoking Select.
                for (i, words) in peer[HEADER..].chunks_exact(16).enumerate() {
                    let r = u64::from_le_bytes(words[..8].try_into().unwrap());
                    let c = u64::from_le_bytes(words[8..].try_into().unwrap());
                    state.arithmetic[i] =
                        state.arithmetic[i].wrapping_add(r) & mask(self.spec.output_bits);
                    state.correction[i] ^= c;
                }
                if self.spec.boundary == Boundary::Deferred {
                    let select = self.select.take().ok_or(Error::Consumed)?;
                    let shares = select.into_deferred(&state.correction, &state.arithmetic)?;
                    // Genuine consuming Select shares, not bytes or a relabeled
                    // reconstructed capability. Keep shares/lease through copy.
                    for i in 0..self.spec.lanes {
                        state.arithmetic[i] =
                            shares.values()[i]
                                .wrapping_mul(2)
                                .wrapping_sub(if self.role == 0 {
                                    state.arithmetic[i]
                                } else {
                                    0
                                })
                                & mask(self.spec.output_bits);
                    }
                    let lease = self.lease.split(output_charge_for::<C>(self.spec))?;
                    Ok(StepFor::Deferred(DeferredOutputFor {
                        values: state.arithmetic,
                        role: self.role,
                        binding: self.binding,
                        bits: self.spec.output_bits,
                        _lease: lease,
                    }))
                } else {
                    let frame = self
                        .select
                        .as_mut()
                        .ok_or(Error::Consumed)?
                        .begin(&state.correction, &state.arithmetic)?;
                    let lease = self.lease.split(size_of::<FrameFor<C>>())?;
                    state.phase = 2;
                    self.state = Some(state);
                    Ok(StepFor::Frame(FrameFor {
                        body: FrameBody::Select(frame),
                        _lease: lease,
                    }))
                }
            }
            2 if self.spec.boundary == Boundary::Reconstructed => {
                let output = self.select.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
                for i in 0..self.spec.lanes {
                    state.arithmetic[i] = output.values()[i]
                        .wrapping_mul(2)
                        .wrapping_sub(state.arithmetic[i])
                        & mask(self.spec.output_bits);
                }
                let lease = self.lease.split(output_charge_for::<C>(self.spec))?;
                Ok(StepFor::Reconstructed(MaskedOutputFor {
                    values: state.arithmetic,
                    binding: self.binding,
                    bits: self.spec.output_bits,
                    _lease: lease,
                }))
            }
            _ => Err(Error::Consumed),
        }
    }
    fn first_frame(&mut self, state: &CorrectionState) -> Result<FrameFor<C>, Error> {
        let mut bytes = Zeroizing::new(allocate(
            first_frame_bytes(self.spec),
            self.lease.budget.budget(),
        )?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&state.digest);
        bytes.extend_from_slice(&[self.role, 0]);
        bytes.extend_from_slice(&(self.spec.lanes as u16).to_le_bytes());
        bytes.extend_from_slice(&[
            self.spec.index_bits,
            self.spec.output_bits,
            boundary_byte(self.spec.boundary),
            0,
        ]);
        for (&r, &c) in state.arithmetic.iter().zip(state.correction.iter()) {
            bytes.extend_from_slice(&r.to_le_bytes());
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        let lease = self.lease.split(frame_charge_for::<C>(self.spec))?;
        Ok(FrameFor {
            body: FrameBody::Lookup(bytes),
            _lease: lease,
        })
    }
    fn validate_first(&self, state: &CorrectionState, peer: &[u8]) -> Result<(), Error> {
        if peer.len() != first_frame_bytes(self.spec)
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != state.digest
            || peer[72] != 1 - self.role
            || peer[73] != 0
            || peer[74..76] != (self.spec.lanes as u16).to_le_bytes()
            || peer[76] != self.spec.index_bits
            || peer[77] != self.spec.output_bits
            || peer[78] != boundary_byte(self.spec.boundary)
            || peer[79] != 0
        {
            return Err(Error::Context);
        }
        for words in peer[HEADER..].chunks_exact(16) {
            let r = u64::from_le_bytes(words[..8].try_into().unwrap());
            let c = u64::from_le_bytes(words[8..].try_into().unwrap());
            if r & !mask(self.spec.output_bits) != 0 || c > 1 {
                return Err(Error::Domain);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
