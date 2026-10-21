//! Bounded SIGMA source lookup over the ascending GPU127 complete expansion.
//!
//! This is a local, unauthenticated research primitive, not a typed decoder
//! bridge, remote transport, full activation, or security qualification.
//! Reconstructed output is `T[x] + rho`; deferred outputs are additive shares
//! of that value. The dealer samples full-ring `r,u` independently and sets
//! `rho = 2*u-r`: in particular `rho mod 2 = r mod 2`, not independent masks.

use super::{mask, selection};
use crate::compact_dpf::gpu127::{self, GpuKey, Profile};
use crate::shared_allocation::{allocate, Lease, MAX_ALLOCATION_BYTES};
pub use crate::shared_allocation::{Budget, Error};
pub use selection::Boundary;
use sha2::{Digest, Sha256};
use std::mem::size_of;
use zeroize::Zeroizing;

pub const DOMAIN: &str = "pllm.sigma.source-lookup.gpu127.v1";
pub const MAX_LANES: usize = super::MAX_LANES;
const MAGIC: &[u8; 8] = b"PLLMLU01";
const HEADER: usize = 80;
const ALLOCATOR: usize = 64;
const WORKSPACE: usize = 8 << 10;

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
fn frame_charge(spec: Spec) -> usize {
    first_frame_bytes(spec) + size_of::<Frame<'static>>() + ALLOCATOR
}
fn output_charge(spec: Spec) -> usize {
    // Charge the larger capability even in reconstructed mode.
    8 * spec.lanes
        + size_of::<DeferredOutput<'static>>().max(size_of::<MaskedOutput<'static>>())
        + ALLOCATOR
}

/// Aggregate logical live capacity, not process RSS. Includes nested Select,
/// actual padded inline GPU127 lane owners, two immutable tables, caller/peer
/// allowances, retained frames and output overlaps, and lane-serial N-byte scan.
/// Eight KiB per owner covers bounded hash/AES/stack work; dealer additionally
/// prices the transient paired inline keys during issuance.
pub fn resources(spec: Spec) -> Result<Resources, Error> {
    spec.validate()?;
    let gpu = Profile::new(spec.index_bits)
        .map_err(gpu_error)?
        .resources();
    let select = selection::resources_for(spec.select(), spec.boundary)?;
    let table = mul(8, spec.entries())?;
    let lane = mul(size_of::<LookupLane>(), spec.lanes)?;
    let select_frame = if spec.boundary == Boundary::Reconstructed {
        select.frame_bytes
    } else {
        0
    };
    let own_party = add(&[
        size_of::<Party<'static>>(),
        lane,
        table,
        mul(16, spec.lanes)?, // local R/C backing, later output/C
        mul(8, spec.lanes)?,  // borrowed common masked indices
        4 * ALLOCATOR,        // lane, table, R, C vectors
        spec.entries(),       // one lane-serial complete expansion
        size_of::<Zeroizing<Vec<u8>>>() + ALLOCATOR,
        first_frame_bytes(spec), // borrowed peer first frame
        select_frame,            // borrowed peer Select frame
        frame_charge(spec),      // retained first send frame
        if select_frame != 0 {
            size_of::<Frame<'static>>() // retained Select wrapper
        } else {
            0
        },
        output_charge(spec), // final output overlaps nested Select output
        WORKSPACE,
    ])?;
    let own_dealer = add(&[
        size_of::<OutputMasks<'static>>(),
        mul(8, spec.lanes)?,  // derived output masks
        mul(8, spec.lanes)?,  // borrowed input masks
        table,                // borrowed complete public table
        mul(16, spec.lanes)?, // transient Select predicate/arithmetic masks
        3 * ALLOCATOR,
        gpu.pair_owner_bytes,
        WORKSPACE,
    ])?;
    let party = add(&[own_party, select.party_reservation_bytes])?;
    let dealer = add(&[own_dealer, select.dealer_reservation_bytes])?;
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
    // One independent additive share of r, one XOR share of q.
    masks: Zeroizing<[u64; 2]>,
}
// GpuKey and masks each erase on drop. No Zeroize bound is needed on Vec:
// ordinary Vec drop invokes every initialized lane, including partial issuance.
struct Execution {
    lanes: Vec<LookupLane>,
    table: Zeroizing<Vec<u64>>,
    arithmetic: Zeroizing<Vec<u64>>,
    correction: Zeroizing<Vec<u64>>,
    digest: [u8; 32],
    phase: u8,
}
pub struct Party<'a> {
    role: u8,
    spec: Spec,
    binding: [u8; 32],
    state: Option<Execution>,
    // Keep this owner outside execution: cancellation burns its state without
    // refunding retained Select metadata/material reservation before Party drop.
    select: Option<selection::Party<'a>>,
    // Secret owners must drop before leases, including on error paths.
    lease: Lease<'a>,
}
pub struct OutputMasks<'a> {
    values: Zeroizing<Vec<u64>>,
    _select_masks: selection::OutputMasks<'a>,
    _lease: Lease<'a>,
}
impl OutputMasks<'_> {
    /// Low-level trusted dealer view only. rho=2u-r, hence rho mod 2=r mod 2.
    pub fn values(&self) -> &[u64] {
        &self.values
    }
}
enum FrameBody<'a> {
    Lookup(Zeroizing<Vec<u8>>),
    Select(selection::Frame<'a>),
}
pub struct Frame<'a> {
    body: FrameBody<'a>,
    _lease: Lease<'a>,
}
impl Frame<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.body {
            FrameBody::Lookup(bytes) => bytes,
            FrameBody::Select(frame) => frame.as_bytes(),
        }
    }
}
pub struct MaskedOutput<'a> {
    values: Zeroizing<Vec<u64>>,
    binding: [u8; 32],
    bits: u8,
    _lease: Lease<'a>,
}
impl MaskedOutput<'_> {
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
pub struct DeferredOutput<'a> {
    values: Zeroizing<Vec<u64>>,
    role: u8,
    binding: [u8; 32],
    bits: u8,
    _lease: Lease<'a>,
}
impl DeferredOutput<'_> {
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
pub enum Step<'a> {
    Frame(Frame<'a>),
    Reconstructed(MaskedOutput<'a>),
    Deferred(DeferredOutput<'a>),
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
    mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
) -> Result<(OutputMasks<'a>, [Party<'a>; 2]), Error> {
    let cost = resources(spec)?;
    context.validate(spec.lanes)?;
    if table.len() != spec.entries()
        || input_masks.len() != spec.lanes
        || table.iter().any(|&v| v & !mask(spec.output_bits) != 0)
        || input_masks.iter().any(|&v| v & !mask(spec.index_bits) != 0)
    {
        return Err(Error::Domain);
    }
    // Entire public and canonical input preflight precedes admission, entropy,
    // and allocation. Identity hashes full canonical bytes, not a caller label.
    let identity = table_identity(spec, context.numeric_contract, table);
    let select_cost = selection::resources_for(spec.select(), spec.boundary)?;
    let mut reservation = budget.reserve(cost.combined_reservation_bytes)?;
    let select_reservation = reservation.split(select_cost.combined_reservation_bytes)?;
    let a =
        reservation.split(cost.party_reservation_bytes - select_cost.party_reservation_bytes)?;
    let b =
        reservation.split(cost.party_reservation_bytes - select_cost.party_reservation_bytes)?;
    // Declare all reservations before secret owners: reverse local drop order
    // erases backing before refunds on every early return.
    let mut masks = zeros(spec.lanes, budget)?;
    let mut predicate_masks = zeros(spec.lanes, budget)?;
    let mut arithmetic_masks = zeros(spec.lanes, budget)?;
    let new_execution = || -> Result<Execution, Error> {
        let mut own_table = zeros(spec.entries(), budget)?;
        own_table.copy_from_slice(table);
        Ok(Execution {
            lanes: allocate(spec.lanes, budget)?,
            table: own_table,
            arithmetic: zeros(spec.lanes, budget)?,
            correction: zeros(spec.lanes, budget)?,
            digest: [0; 32],
            phase: 0,
        })
    };
    let mut left = new_execution()?;
    let mut right = new_execution()?;
    let mut nonce = [0; 32];
    fill(&mut nonce)?;
    let binding = invocation_binding(spec, context, identity, nonce);
    let profile = Profile::new(spec.index_bits).map_err(gpu_error)?;
    for (i, &rin) in input_masks.iter().enumerate() {
        let [k0, k1] = GpuKey::issue_with(
            profile,
            0u64.wrapping_sub(rin) & mask(spec.index_bits),
            |bytes| fill(bytes).map_err(|_| gpu127::Error::Randomness),
        )
        .map_err(gpu_error)?;
        let r = Zeroizing::new(draw(spec.output_bits, &mut fill)?);
        let q = Zeroizing::new(draw(1, &mut fill)?);
        let r0 = Zeroizing::new(draw(spec.output_bits, &mut fill)?);
        let q0 = Zeroizing::new(draw(1, &mut fill)?);
        arithmetic_masks[i] = *r;
        predicate_masks[i] = *q;
        left.lanes.push(LookupLane {
            key: k0,
            masks: Zeroizing::new([*r0, *q0]),
        });
        right.lanes.push(LookupLane {
            key: k1,
            masks: Zeroizing::new([r.wrapping_sub(*r0) & mask(spec.output_bits), *q ^ *q0]),
        });
    }
    let select_context = select_context(context, binding);
    let (select_masks, [select0, select1]) = selection::issue_reserved_with(
        spec.select(),
        select_context,
        &predicate_masks,
        &arithmetic_masks,
        spec.boundary,
        select_reservation,
        &mut fill,
    )?;
    for i in 0..spec.lanes {
        masks[i] = select_masks.values()[i]
            .wrapping_mul(2)
            .wrapping_sub(arithmetic_masks[i])
            & mask(spec.output_bits);
    }
    Ok((
        OutputMasks {
            values: masks,
            _select_masks: select_masks,
            _lease: reservation,
        },
        [
            Party {
                role: 0,
                spec,
                binding,
                state: Some(left),
                select: Some(select0),
                lease: a,
            },
            Party {
                role: 1,
                spec,
                binding,
                state: Some(right),
                select: Some(select1),
                lease: b,
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

impl<'a> Party<'a> {
    /// Burns execution; retained owners and their outstanding leases remain
    /// charged until drop. Sibling cancellation is caller's responsibility.
    pub fn cancel(&mut self) {
        self.state = None;
        if let Some(select) = &mut self.select {
            select.cancel();
        }
    }

    pub fn begin(&mut self, masked_indices: &[u64]) -> Result<Frame<'a>, Error> {
        let result = self.begin_inner(masked_indices);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn begin_inner(&mut self, masked_indices: &[u64]) -> Result<Frame<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if state.phase != 0 {
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
        state.digest = hash.finalize().into();
        for (lane, &index) in masked_indices.iter().enumerate() {
            let material = &state.lanes[lane];
            let expanded = material
                .key
                .all_with_allocator(self.spec.entries(), |n| {
                    allocate(n, self.lease.budget).map_err(allocation_error)
                })
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
            state.arithmetic[lane] =
                sum.wrapping_add(material.masks[0]) & mask(self.spec.output_bits);
            state.correction[lane] = correction(self.role, *count) ^ material.masks[1];
        }
        let frame = self.first_frame(&state)?;
        state.phase = 1;
        self.state = Some(state);
        Ok(frame)
    }

    pub fn advance(&mut self, peer: &[u8]) -> Result<Step<'a>, Error> {
        let result = self.advance_inner(peer);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn advance_inner(&mut self, peer: &[u8]) -> Result<Step<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
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
                    let lease = self.lease.split(output_charge(self.spec))?;
                    Ok(Step::Deferred(DeferredOutput {
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
                    let lease = self.lease.split(size_of::<Frame<'static>>())?;
                    state.phase = 2;
                    self.state = Some(state);
                    Ok(Step::Frame(Frame {
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
                let lease = self.lease.split(output_charge(self.spec))?;
                Ok(Step::Reconstructed(MaskedOutput {
                    values: state.arithmetic,
                    binding: self.binding,
                    bits: self.spec.output_bits,
                    _lease: lease,
                }))
            }
            _ => Err(Error::Consumed),
        }
    }
    fn first_frame(&mut self, state: &Execution) -> Result<Frame<'a>, Error> {
        let mut bytes = Zeroizing::new(allocate(first_frame_bytes(self.spec), self.lease.budget)?);
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
        let lease = self.lease.split(frame_charge(self.spec))?;
        Ok(Frame {
            body: FrameBody::Lookup(bytes),
            _lease: lease,
        })
    }
    fn validate_first(&self, state: &Execution, peer: &[u8]) -> Result<(), Error> {
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
