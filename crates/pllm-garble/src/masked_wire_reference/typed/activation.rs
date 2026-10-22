//! Fixed D20 mode1 SIGMA activation owning region.
//!
//! Local assumptions and unauthenticated frames only. Internal projections,
//! affine wires and deferred shares never become ordinary capabilities. Failure
//! burns this party; the coordinator must explicitly cancel its sibling.

use super::internal_use::Ledger as UseLedger;
use super::*;
use crate::masked_wire_reference::source_lookup;
use pllm_core::edge_contract::{ConsumerUse, EdgeIdentity};

pub const DOMAIN: &str = "pllm.sigma.activation-reconstruction.v1";
const MAGIC: &[u8; 8] = b"PLLMAC01";
const HEADER: usize = 76;
const WIRES: usize = 17;
const USES: usize = 20;
const XN: usize = 0;
const Y: usize = 1;
const D: usize = 2;
const P: usize = 3;
const A: usize = 4;
const C: usize = 5;
const H: usize = 6;
const E: usize = 7;
const Q: usize = 8;
const V: usize = 9;
const J: usize = 10;
const L: usize = 11;
const F: usize = 12;
const O: usize = 13;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    Gelu,
    Silu,
}
impl ActivationKind {
    pub fn index_bits(self) -> u8 {
        match self {
            Self::Gelu => 8,
            Self::Silu => 10,
        }
    }
    fn table(self) -> &'static [u64] {
        match self {
            Self::Gelu => crate::fixed_activation_tables::gelu_q12(),
            Self::Silu => crate::fixed_activation_tables::silu_q12(),
        }
    }
    fn table_digest(self) -> [u8; 32] {
        match self {
            Self::Gelu => [
                0xb6, 0xa4, 0x5a, 0x82, 0xdc, 0x8f, 0x66, 0x07, 0x24, 0x94, 0x3c, 0x1e, 0xb7, 0x66,
                0x54, 0x70, 0x4c, 0x2e, 0x49, 0x56, 0x3f, 0xfc, 0x0b, 0x1a, 0x0b, 0xc5, 0x38, 0x8b,
                0x34, 0x25, 0xa0, 0x1b,
            ],
            Self::Silu => [
                0xff, 0x69, 0x6c, 0xf6, 0x35, 0x79, 0xaa, 0x28, 0x57, 0x1a, 0xf2, 0xc5, 0x68, 0x95,
                0xc0, 0x6b, 0x1e, 0xd2, 0xe6, 0x4c, 0xe6, 0xfa, 0xb9, 0x42, 0xfc, 0xbb, 0xe9, 0x45,
                0x47, 0x44, 0xf6, 0x3f,
            ],
        }
    }
    fn validate_table(self) -> Result<(), Error> {
        let table = self.table();
        if table.len() != 1usize << self.index_bits()
            || table.iter().any(|&x| x > super::super::mask(50))
        {
            return Err(Error::Domain);
        }
        let mut h = Sha256::new();
        for value in table {
            h.update(value.to_le_bytes());
        }
        let actual: [u8; 32] = h.finalize().into();
        if actual != self.table_digest() {
            return Err(Error::Context);
        }
        Ok(())
    }
}

/// Public declaration, not a capability, range proof or execution permit.
/// All internal wire/use identities are fixed by this template and declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationSpec {
    kind: ActivationKind,
    operation: [u8; 32],
    input: EdgeDescriptor,
    output: EdgeDescriptor,
    next_consumer: [u8; 32],
}
impl ActivationSpec {
    pub fn new(
        kind: ActivationKind,
        operation: [u8; 32],
        input: EdgeDescriptor,
        output: EdgeDescriptor,
        next_consumer: [u8; 32],
    ) -> Result<Self, Error> {
        let spec = Self {
            kind,
            operation,
            input,
            output,
            next_consumer,
        };
        spec.validate()?;
        Ok(spec)
    }
    pub fn kind(self) -> ActivationKind {
        self.kind
    }
    pub fn input(self) -> EdgeDescriptor {
        self.input
    }
    pub fn output(self) -> EdgeDescriptor {
        self.output
    }

    fn validate(self) -> Result<(), Error> {
        if !(1..=256).contains(&self.input.shape().elements()) {
            return Err(Error::Domain);
        }
        for edge in [self.input, self.output] {
            if edge.numeric()
                != Numeric::new(50, 12, Interpretation::Signed).map_err(|_| Error::Domain)?
                || edge.representation() != Representation::CommonAdditiveMasked
                || edge.weight_visibility().is_some()
            {
                return Err(Error::Domain);
            }
        }
        if self.input.identity().source() != self.output.identity().source()
            || self.input.identity().plan() != self.output.identity().plan()
            || self.input.shape() != self.output.shape()
            || self.input.identity().first_tensor_index()
                != self.output.identity().first_tensor_index()
            || self.input.identity().wire() == self.output.identity().wire()
        {
            return Err(Error::Context);
        }
        if self.input.identity().source().implementation() != approved_implementation() {
            return Err(Error::Unsupported);
        }
        self.kind.validate_table()?;
        let wires = self.wires()?;
        for (i, edge) in wires.iter().enumerate() {
            if edge.identity().wire() == self.input.identity().wire()
                || edge.identity().wire() == self.output.identity().wire()
                || wires[..i]
                    .iter()
                    .any(|old| old.identity().wire() == edge.identity().wire())
            {
                return Err(Error::Context);
            }
        }
        Ok(())
    }

    fn numeric_contract(self) -> [u8; 32] {
        digest(&[
            b"pllm.sigma.typed.activation-numeric.v1\0",
            b"EzPC:aa0f2f860266fb79901405ef5e9965b8c9fe2e88:mode1",
            b"LocalAssumption:X=[-2^37,2^37);50/Q12;38/Q12;TR6;32/Q6;A=unsigned[0,2^31]",
            &[self.kind.index_bits()],
            &self.kind.table_digest(),
        ])
    }
    fn declaration(self) -> [u8; 32] {
        digest(&[
            DOMAIN.as_bytes(),
            &self.operation,
            &self.input.digest(),
            &self.output.digest(),
            &self.next_consumer,
            &self.numeric_contract(),
        ])
    }
    fn stage(self, index: u8) -> [u8; 32] {
        digest(&[
            b"pllm.sigma.typed.activation-stage.v1\0",
            &self.declaration(),
            &[index],
        ])
    }
    fn wires(self) -> Result<[EdgeDescriptor; WIRES], Error> {
        let mut edges = [self.input; WIRES];
        for (i, edge) in edges.iter_mut().enumerate() {
            let (bits, frac, interpretation) = match i {
                XN => (38, 12, Interpretation::Signed),
                Y | P | C => (32, 6, Interpretation::Signed),
                A => (32, 6, Interpretation::Unsigned),
                D | H | E | 15 => (1, 0, Interpretation::Predicate),
                Q | V | J => (self.kind.index_bits(), 6, Interpretation::Unsigned),
                _ => (50, 12, Interpretation::Signed),
            };
            let id = self.input.identity();
            *edge = EdgeDescriptor::new(
                EdgeIdentity::new(
                    id.source(),
                    id.plan(),
                    self.operation,
                    digest(&[
                        b"pllm.sigma.typed.activation-wire.v1\0",
                        &self.declaration(),
                        &[i as u8],
                    ]),
                    id.first_tensor_index(),
                ),
                self.input.shape(),
                Numeric::new(bits, frac, interpretation).map_err(|_| Error::Domain)?,
                if matches!(i, L | F | O | 16) {
                    Representation::AdditiveShare
                } else if interpretation == Interpretation::Predicate {
                    Representation::CommonXorMasked
                } else {
                    Representation::CommonAdditiveMasked
                },
                None,
            )
            .map_err(|_| Error::Domain)?;
        }
        Ok(edges)
    }
    fn uses(self, w: &[EdgeDescriptor; WIRES]) -> [ConsumerUse; USES] {
        let edges = [
            self.input, w[XN], w[Y], w[D], w[Y], w[P], w[Y], w[A], w[C], w[H], w[C], w[E], w[Q],
            w[V], w[J], w[D], self.input, w[F], w[L], w[O],
        ];
        let stages = [
            0, 1, 2, 3, 3, 5, 5, 7, 8, 9, 10, 11, 11, 13, 14, 15, 15, 17, 17, 19,
        ];
        let operands = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0];
        std::array::from_fn(|i| {
            ConsumerUse::new(edges[i], self.stage(stages[i]), operands[i], i as u32)
        })
    }
    pub fn digest(self) -> [u8; 32] {
        // Construction already checked descriptor geometry. No execution state.
        let mut h = Sha256::new();
        h.update(self.declaration());
        h.update(arithmetic::DOMAIN.as_bytes());
        h.update(selection::DOMAIN.as_bytes());
        h.update(source_lookup::DOMAIN.as_bytes());
        h.update(Profile::DOMAIN.as_bytes());
        if let Ok(w) = self.wires() {
            for edge in w {
                h.update(edge.digest());
            }
            for use_slot in self.uses(&w) {
                h.update(use_slot.digest());
            }
        }
        h.finalize().into()
    }
    fn arithmetic(self, bits: u8, gate: arithmetic::Gate) -> arithmetic::Spec {
        arithmetic::Spec {
            bits,
            lanes: self.input.shape().elements(),
            gate,
        }
    }
    fn select(self, bits: u8) -> selection::Spec {
        selection::Spec {
            bits,
            lanes: self.input.shape().elements(),
        }
    }
    fn lookup(self) -> source_lookup::Spec {
        source_lookup::Spec {
            index_bits: self.kind.index_bits(),
            output_bits: 50,
            lanes: self.input.shape().elements(),
            boundary: selection::Boundary::Deferred,
        }
    }
}

/// Checked logical capacity, not RSS, serialized material, or security evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationResources {
    pub lanes: usize,
    pub exchanges: usize,
    pub total_peer_frame_bytes: usize,
    pub random_bytes: usize,
    pub party_material_payload_bytes: usize,
    /// Additional reservation; original root/dealer owners remain charged.
    pub combined_reservation_bytes: usize,
    pub native_reservation_bytes: usize,
    pub wrapper_reservation_bytes: usize,
    /// Entire new clear-input operation, excluding existing LocalDealer owner.
    pub clear_input_reservation_bytes: usize,
    /// Includes original root masks, excludes existing dealer namespace entropy.
    pub clear_input_random_bytes: usize,
}
fn frame_wrapper_charge() -> usize {
    size_of::<ActivationFrame<'static>>() + size_of::<ActivationStep<'static>>()
}
fn party_charge(lanes: usize) -> usize {
    size_of::<ActivationParty<'static>>() + common_charge()
        + 8 * frame_wrapper_charge() + 6 * (8 * lanes + ALLOCATOR)
        + 3 * (HEADER + 8 * lanes + ALLOCATOR) // final send, receive, output overlap
        + (16 << 10)
}
fn dealer_charge(lanes: usize) -> usize {
    mask_charge() + 8 * lanes + ALLOCATOR
}
fn workspace_charge(lanes: usize) -> usize {
    2 * size_of::<ActivationParty<'static>>()
        + 2 * size_of::<UseLedger<USES>>()
        + WIRES * size_of::<Binding>()
        + 4 * (8 * lanes + ALLOCATOR)
        + (32 << 10)
}
fn native_costs(spec: ActivationSpec) -> Result<[usize; 7], Error> {
    Ok([
        arithmetic::resources(spec.arithmetic(38, arithmetic::Gate::Tr { shift: 6 }))?
            .combined_reservation_bytes,
        arithmetic::resources(spec.arithmetic(32, arithmetic::Gate::Drelu))?
            .combined_reservation_bytes,
        selection::resources(spec.select(32))?.combined_reservation_bytes,
        arithmetic::resources(spec.arithmetic(32, arithmetic::Gate::Drelu))?
            .combined_reservation_bytes,
        selection::resources(spec.select(spec.kind.index_bits()))?.combined_reservation_bytes,
        source_lookup::resources(spec.lookup())?.combined_reservation_bytes,
        selection::resources_for(spec.select(50), selection::Boundary::Deferred)?
            .combined_reservation_bytes,
    ])
}
pub fn activation_resources(spec: ActivationSpec) -> Result<ActivationResources, Error> {
    spec.validate()?;
    let lanes = spec.input.shape().elements();
    let costs = native_costs(spec)?;
    let native = costs
        .iter()
        .try_fold(0usize, |n, &v| n.checked_add(v))
        .ok_or(Error::Resource)?;
    let wrappers = 2 * party_charge(lanes) + dealer_charge(lanes) + workspace_charge(lanes);
    let combined = native.checked_add(wrappers).ok_or(Error::Resource)?;
    let root = 2 * (8 * lanes + ALLOCATOR + common_charge())
        + 8 * lanes
        + ALLOCATOR
        + mask_charge()
        + size_of::<RootInput<'static>>();
    let clear_input = combined.checked_add(root).ok_or(Error::Resource)?;
    if clear_input > crate::shared_allocation::MAX_ALLOCATION_BYTES {
        return Err(Error::Resource);
    }
    let t = arithmetic::resources(spec.arithmetic(38, arithmetic::Gate::Tr { shift: 6 }))?;
    let d = arithmetic::resources(spec.arithmetic(32, arithmetic::Gate::Drelu))?;
    let p = selection::resources(spec.select(32))?;
    let v = selection::resources(spec.select(spec.kind.index_bits()))?;
    let l = source_lookup::resources(spec.lookup())?;
    let f = selection::resources_for(spec.select(50), selection::Boundary::Deferred)?;
    Ok(ActivationResources {
        lanes,
        exchanges: 8,
        total_peer_frame_bytes: 1224 + 144 * lanes,
        random_bytes: t.random_bytes
            + 2 * d.random_bytes
            + p.random_bytes
            + v.random_bytes
            + l.random_bytes
            + f.random_bytes,
        party_material_payload_bytes: t.party_material_payload_bytes
            + 2 * d.party_material_payload_bytes
            + p.party_material_payload_bytes
            + v.party_material_payload_bytes
            + l.party_material_payload_bytes
            + f.party_material_payload_bytes,
        combined_reservation_bytes: combined,
        native_reservation_bytes: native,
        wrapper_reservation_bytes: wrappers,
        clear_input_reservation_bytes: clear_input,
        clear_input_random_bytes: 8 * lanes
            + t.random_bytes
            + 2 * d.random_bytes
            + p.random_bytes
            + v.random_bytes
            + l.random_bytes
            + f.random_bytes,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Fresh,
    Tr,
    Drelu,
    Prefix,
    ClipDrelu,
    ClipSelect,
    Lookup,
    Final,
    Dead,
}

/// Noncloneable party-local owner. No intermediate views or capabilities escape.
///
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::ActivationParty;
/// fn duplicate<'a>(p: ActivationParty<'a>) { let _ = p.clone(); }
/// ```
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::typed::ActivationParty;
/// fn project(p: &ActivationParty<'_>) { let _ = &p.projection; }
/// ```
/// ```compile_fail
/// use pllm_garble::masked_wire_reference::{selection::DeferredShare, typed::ActivationParty};
/// fn share_is_not_common<'a>(p: &mut ActivationParty<'a>, s: DeferredShare<'a>) {
///     let _ = p.begin(s);
/// }
/// ```
pub struct ActivationParty<'b> {
    tr: Option<arithmetic::Party<'b>>,
    drelu: Option<arithmetic::Party<'b>>,
    prefix: Option<selection::Party<'b>>,
    clip_drelu: Option<arithmetic::Party<'b>>,
    clip_select: Option<selection::Party<'b>>,
    lookup: Option<source_lookup::Party<'b>>,
    final_select: Option<selection::Party<'b>>,
    x: Option<CommonMasked<'b, Ring>>,
    y: Option<arithmetic::MaskedOutput<'b>>,
    d: Option<arithmetic::MaskedOutput<'b>>,
    // All private work backing is preallocated under original aggregate lease.
    projection: Zeroizing<Vec<u64>>,
    affine: Zeroizing<Vec<u64>>,
    complement: Zeroizing<Vec<u64>>,
    operand: Zeroizing<Vec<u64>>,
    index: Zeroizing<Vec<u64>>,
    local: Zeroizing<Vec<u64>>,
    input: Binding,
    wires: [Binding; WIRES],
    output: Binding,
    ledger: UseLedger<USES>,
    binding: [u8; 32],
    lookup_binding: [u8; 32],
    select_binding: [u8; 32],
    input_digest: [u8; 32],
    role: u8,
    phase: Phase,
    spec: ActivationSpec,
    // Payload and original native guards drop before wrapper capacity refunds.
    lease: Option<Lease<'b>>,
}
enum FrameBody<'b> {
    Arithmetic(arithmetic::Frame<'b>),
    Selection(selection::Frame<'b>),
    Lookup(source_lookup::Frame<'b>),
    Final(Zeroizing<Vec<u8>>),
}
pub struct ActivationFrame<'b> {
    body: FrameBody<'b>,
    _lease: Lease<'b>,
}
impl ActivationFrame<'_> {
    pub fn as_bytes(&self) -> &[u8] {
        match &self.body {
            FrameBody::Arithmetic(f) => f.as_bytes(),
            FrameBody::Selection(f) => f.as_bytes(),
            FrameBody::Lookup(f) => f.as_bytes(),
            FrameBody::Final(f) => f,
        }
    }
}
#[allow(clippy::large_enum_variant)]
pub enum ActivationStep<'b> {
    Frame(ActivationFrame<'b>),
    Output(CommonMasked<'b, Ring>),
}

fn slot_bindings(input: Binding, wires: &[Binding; WIRES]) -> [Binding; USES] {
    [
        input, wires[XN], wires[Y], wires[D], wires[Y], wires[P], wires[Y], wires[A], wires[C],
        wires[H], wires[C], wires[E], wires[Q], wires[V], wires[J], wires[D], input, wires[F],
        wires[L], wires[O],
    ]
}

fn ledger_digest(ledger: &UseLedger<USES>) -> [u8; 32] {
    ledger.digest_with_domains(
        b"pllm.sigma.typed.activation-private-use.v1\0",
        b"pllm.sigma.typed.activation-private-ledger.v1\0",
    )
}

/// Complete operation issuance, retaining opaque input capabilities for begin.
pub type ActivationIssued<'b> = (
    DealerMask<'b, Ring>,
    [ActivationParty<'b>; 2],
    [CommonMasked<'b, Ring>; 2],
);

impl<'b> LocalDealer<'b> {
    /// Complete clear-input operation boundary. Validate every encoded integer,
    /// declaration and aggregate capacity before allocating or drawing root
    /// material. Dealer namespace creation is outside this operation boundary.
    /// The returned input capabilities can only begin their corresponding party.
    pub fn activation_from_values(
        &mut self,
        spec: ActivationSpec,
        values: &[i64],
    ) -> Result<ActivationIssued<'b>, Error> {
        self.activation_from_values_with(spec, values, |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    fn activation_from_values_with(
        &mut self,
        spec: ActivationSpec,
        values: &[i64],
        mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<ActivationIssued<'b>, Error> {
        let cost = activation_resources(spec)?;
        self.validate(spec.input, false)?;
        self.validate(spec.output, false)?;
        if values.len() != cost.lanes
            || values
                .iter()
                .any(|&x| !(-(1i64 << 37)..(1i64 << 37)).contains(&x))
        {
            return Err(Error::Domain);
        }
        self.counter
            .checked_add(WIRES as u64 + 2)
            .ok_or(Error::Resource)?;
        let promise = BoundRange::new(-(1i128 << 37), 1i128 << 37)?;
        let root_bytes = self.root_reservation_bytes(spec.input, &promise)?;
        let total = cost
            .combined_reservation_bytes
            .checked_add(root_bytes)
            .ok_or(Error::Resource)?;
        let mut aggregate = self.budget.reserve(total)?;
        let region_lease = aggregate.split(cost.combined_reservation_bytes)?;
        let prepared = self.prepare_root_reserved_with(
            spec.input,
            promise,
            spec.operation,
            aggregate,
            &mut fill,
        )?;
        let (input, common) = prepared.encode(values)?;
        let (mask, parties) =
            self.activation_reserved_with(spec, input, region_lease, &mut fill)?;
        Ok((mask, parties, common))
    }

    pub fn activation(
        &mut self,
        spec: ActivationSpec,
        input: DealerMask<'b, Ring>,
    ) -> Result<(DealerMask<'b, Ring>, [ActivationParty<'b>; 2]), Error> {
        self.activation_with(spec, input, |bytes| {
            getrandom::fill(bytes).map_err(|_| Error::Randomness)
        })
    }

    fn activation_with(
        &mut self,
        spec: ActivationSpec,
        input: DealerMask<'b, Ring>,
        mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [ActivationParty<'b>; 2]), Error> {
        let cost = activation_resources(spec)?;
        self.activation_input_range(spec, &input)?;
        // Existing roots/matrix outputs stay continuously charged separately.
        // Complete input checks precede aggregate admission and operation draws.
        let reservation = self.budget.reserve(cost.combined_reservation_bytes)?;
        self.activation_reserved_with(spec, input, reservation, &mut fill)
    }

    fn activation_input_range(
        &self,
        spec: ActivationSpec,
        input: &DealerMask<'b, Ring>,
    ) -> Result<Interval, Error> {
        self.input(input.binding, spec.operation)?;
        if input.binding.descriptor != spec.input {
            return Err(Error::Context);
        }
        let range = input.binding.interval.ok_or(Error::Domain)?;
        if range.lower < -(1i128 << 37) || range.upper > 1i128 << 37 {
            return Err(Error::Domain);
        }
        self.counter
            .checked_add(WIRES as u64 + 1)
            .ok_or(Error::Resource)?;
        if input.backing.values().len() != spec.input.shape().elements()
            || input
                .backing
                .values()
                .iter()
                .any(|&v| v & !super::super::mask(50) != 0)
        {
            return Err(Error::Domain);
        }
        Ok(range)
    }

    fn activation_reserved_with(
        &mut self,
        spec: ActivationSpec,
        input: DealerMask<'b, Ring>,
        mut reservation: Lease<'b>,
        mut fill: impl FnMut(&mut [u8]) -> Result<(), Error>,
    ) -> Result<(DealerMask<'b, Ring>, [ActivationParty<'b>; 2]), Error> {
        let cost = activation_resources(spec)?;
        if reservation.bytes != cost.combined_reservation_bytes
            || !std::ptr::eq(reservation.budget, self.budget)
        {
            return Err(Error::Resource);
        }
        let range = self.activation_input_range(spec, &input)?;
        let descriptors = spec.wires()?;
        let native = native_costs(spec)?;
        let n = cost.lanes;
        // All descriptor/table/range/geometry checks precede capacity, allocation
        // and entropy. Every native receives a split of this original aggregate.
        let left_lease = reservation.split(party_charge(n))?;
        let right_lease = reservation.split(party_charge(n))?;
        let mask_lease = reservation.split(dealer_charge(n))?;
        let mut wires = [input.binding; WIRES];
        for (i, edge) in descriptors.iter().enumerate() {
            let interval = match i {
                XN => Some(range),
                Y => Some(range.shift(6)),
                A => Some(Interval {
                    lower: 0,
                    upper: (1i128 << 31) + 1,
                }),
                C => Some(Interval {
                    lower: -(1i128 << spec.kind.index_bits()),
                    upper: (1i128 << 31) - (1i128 << spec.kind.index_bits()) + 1,
                }),
                _ => None,
            };
            wires[i] = self.binding(*edge, spec.digest(), interval)?;
        }
        let output = self.binding(spec.output, spec.next_consumer, None)?;
        let make_ledger = |role| UseLedger {
            uses: spec.uses(&descriptors),
            bindings: slot_bindings(input.binding, &wires),
            role,
            next: 0,
        };
        let left_ledger = make_ledger(0);
        let right_ledger = make_ledger(1);
        // Dealer is also an independent fixed-slot owner (role tag 2).
        let mut dealer_ledger = make_ledger(2);
        let binding = digest(&[
            DOMAIN.as_bytes(),
            &spec.digest(),
            &input.binding.digest(),
            &output.digest(),
            &ledger_digest(&left_ledger),
            &ledger_digest(&right_ledger),
            &ledger_digest(&dealer_ledger),
        ]);
        let arithmetic_context = |stage: u8, x: Binding, y: Binding| arithmetic::Context {
            plan: self.plan,
            session: self.session,
            operation: digest(&[&binding, &spec.stage(stage)]),
            input_wire: x.digest(),
            output_wire: y.digest(),
            first_tensor_index: spec.input.identity().first_tensor_index(),
        };
        let select_context = |stage: u8, d: Binding, x: Binding, y: Binding| selection::Context {
            plan: self.plan,
            session: self.session,
            operation: digest(&[&binding, &spec.stage(stage)]),
            predicate_wire: d.digest(),
            input_wire: x.digest(),
            output_wire: y.digest(),
            first_tensor_index: spec.input.identity().first_tensor_index(),
        };
        let burn = |ledger: &mut UseLedger<USES>, slot: usize| {
            ledger.burn(slot as u8, slot_bindings(input.binding, &wires)[slot], 2)
        };
        // Declare backing after leases: partial failures erase before refund.
        let mut projection = zeros(n, self.budget)?;
        let mut affine = zeros(n, self.budget)?;
        let mut operand = zeros(n, self.budget)?;
        let mut final_masks = zeros(n, self.budget)?;
        burn(&mut dealer_ledger, 0)?;
        for (out, &mask) in projection.iter_mut().zip(input.backing.values()) {
            *out = mask & super::super::mask(38);
        }
        burn(&mut dealer_ledger, 1)?;
        let (ym, [t0, t1]) = arithmetic::issue_reserved_with(
            spec.arithmetic(38, arithmetic::Gate::Tr { shift: 6 }),
            arithmetic_context(1, wires[XN], wires[Y]),
            &projection,
            reservation.split(native[0])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 2)?;
        let (dm, [d0, d1]) = arithmetic::issue_reserved_with(
            spec.arithmetic(32, arithmetic::Gate::Drelu),
            arithmetic_context(2, wires[Y], wires[D]),
            ym.values(),
            reservation.split(native[1])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 3)?;
        burn(&mut dealer_ledger, 4)?;
        let (pm, [p0, p1]) = selection::issue_reserved_with(
            spec.select(32),
            select_context(3, wires[D], wires[Y], wires[P]),
            dm.values(),
            ym.values(),
            selection::Boundary::Reconstructed,
            reservation.split(native[2])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 5)?;
        burn(&mut dealer_ledger, 6)?;
        for i in 0..n {
            affine[i] = affine_mask(pm.values()[i], ym.values()[i]);
        }
        burn(&mut dealer_ledger, 7)?; // C's public subtraction changes no masks.
        burn(&mut dealer_ledger, 8)?;
        let (hm, [h0, h1]) = arithmetic::issue_reserved_with(
            spec.arithmetic(32, arithmetic::Gate::Drelu),
            arithmetic_context(8, wires[C], wires[H]),
            &affine,
            reservation.split(native[3])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 9)?; // E's complement retains H's XOR mask.
        burn(&mut dealer_ledger, 10)?;
        for i in 0..n {
            operand[i] = affine[i] & super::super::mask(spec.kind.index_bits());
        }
        burn(&mut dealer_ledger, 11)?;
        burn(&mut dealer_ledger, 12)?;
        let (vm, [v0, v1]) = selection::issue_reserved_with(
            spec.select(spec.kind.index_bits()),
            select_context(11, wires[E], wires[Q], wires[V]),
            hm.values(),
            &operand,
            selection::Boundary::Reconstructed,
            reservation.split(native[4])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 13)?; // J's public addition changes no mask.
        burn(&mut dealer_ledger, 14)?;
        let context = source_lookup::Context {
            plan: self.plan,
            session: self.session,
            operation: digest(&[&binding, &spec.stage(14)]),
            input_wire: wires[J].digest(),
            output_wire: wires[L].digest(),
            arithmetic_wire: wires[14].digest(),
            correction_wire: wires[15].digest(),
            select_wire: wires[16].digest(),
            numeric_contract: spec.numeric_contract(),
            first_tensor_index: spec.input.identity().first_tensor_index(),
        };
        let (lm, [l0, l1]) = source_lookup::issue_reserved_with(
            spec.lookup(),
            context,
            spec.kind.table(),
            vm.values(),
            reservation.split(native[5])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 15)?;
        burn(&mut dealer_ledger, 16)?;
        let (fm, [f0, f1]) = selection::issue_reserved_with(
            spec.select(50),
            select_context(15, wires[D], input.binding, wires[F]),
            dm.values(),
            input.backing.values(),
            selection::Boundary::Deferred,
            reservation.split(native[6])?,
            &mut fill,
        )?;
        burn(&mut dealer_ledger, 17)?;
        burn(&mut dealer_ledger, 18)?;
        for i in 0..n {
            final_masks[i] = fm.values()[i].wrapping_sub(lm.values()[i]) & super::super::mask(50);
        }
        burn(&mut dealer_ledger, 19)?;
        let party = |ledger,
                     tr,
                     drelu,
                     prefix,
                     clip_drelu,
                     clip_select,
                     lookup: source_lookup::Party<'b>,
                     final_select: selection::Party<'b>,
                     role,
                     lease|
         -> Result<ActivationParty<'b>, Error> {
            Ok(ActivationParty {
                lookup_binding: lookup.internal_binding(),
                select_binding: final_select.internal_binding(),
                tr: Some(tr),
                drelu: Some(drelu),
                prefix: Some(prefix),
                clip_drelu: Some(clip_drelu),
                clip_select: Some(clip_select),
                lookup: Some(lookup),
                final_select: Some(final_select),
                x: None,
                y: None,
                d: None,
                projection: zeros(n, self.budget)?,
                affine: zeros(n, self.budget)?,
                complement: zeros(n, self.budget)?,
                operand: zeros(n, self.budget)?,
                index: zeros(n, self.budget)?,
                local: zeros(n, self.budget)?,
                input: input.binding,
                wires,
                output,
                ledger,
                binding,
                input_digest: [0; 32],
                role,
                phase: Phase::Fresh,
                spec,
                lease: Some(lease),
            })
        };
        let left = party(left_ledger, t0, d0, p0, h0, v0, l0, f0, 0, left_lease)?;
        let right = party(right_ledger, t1, d1, p1, h1, v1, l1, f1, 1, right_lease)?;
        // Retain original input mask/backing until final wide Select issuance.
        // All temporary masks are erased while workspace reservation remains live.
        drop((
            projection, affine, operand, ym, dm, pm, hm, vm, lm, fm, input,
        ));
        Ok((
            DealerMask {
                backing: MaskBacking::Fresh(final_masks),
                binding: output,
                _kind: PhantomData,
                _lease: mask_lease,
            },
            [left, right],
        ))
    }
}

fn affine_mask(p: u64, y: u64) -> u64 {
    p.wrapping_mul(2).wrapping_sub(y) & super::super::mask(32)
}

impl<'b> ActivationParty<'b> {
    pub fn spec(&self) -> ActivationSpec {
        self.spec
    }
    pub fn cancel(&mut self) {
        self.phase = Phase::Dead;
        self.ledger.next = USES as u8;
        self.tr = None;
        self.drelu = None;
        self.prefix = None;
        self.clip_drelu = None;
        self.clip_select = None;
        self.lookup = None;
        self.final_select = None;
        self.x = None;
        self.y = None;
        self.d = None;
        // Erase and deallocate workspace before refund, not merely Vec::clear.
        self.projection = Zeroizing::new(Vec::new());
        self.affine = Zeroizing::new(Vec::new());
        self.complement = Zeroizing::new(Vec::new());
        self.operand = Zeroizing::new(Vec::new());
        self.index = Zeroizing::new(Vec::new());
        self.local = Zeroizing::new(Vec::new());
        // The cancelled/completed wrapper remains caller-owned. Keep its
        // original metadata reservation until Drop; only retired workspace
        // may be refunded here. Repeated cancellation keeps the same lease.
        super::retain_wrapper_metadata(&mut self.lease, size_of::<ActivationParty<'static>>());
    }
    fn burn(&mut self, slot: usize) -> Result<(), Error> {
        self.ledger.burn(
            slot as u8,
            slot_bindings(self.input, &self.wires)[slot],
            self.role,
        )
    }
    fn frame(&mut self, body: FrameBody<'b>) -> Result<ActivationFrame<'b>, Error> {
        let extra = if matches!(&body, FrameBody::Final(_)) {
            HEADER + 8 * self.spec.input.shape().elements() + ALLOCATOR
        } else {
            0
        };
        Ok(ActivationFrame {
            body,
            _lease: self
                .lease
                .as_mut()
                .ok_or(Error::Consumed)?
                .split(frame_wrapper_charge() + extra)?,
        })
    }
    pub fn begin(&mut self, input: CommonMasked<'b, Ring>) -> Result<ActivationFrame<'b>, Error> {
        let result = (|| {
            // Burn actual binding/role, not caller-independent expected metadata.
            self.ledger.burn(0, input.binding, input.role)?;
            if self.phase != Phase::Fresh {
                return Err(Error::Consumed);
            }
            if input.binding != self.input || input.role != self.role {
                return Err(Error::Context);
            }
            let values = input.backing.values();
            if values.len() != self.projection.len()
                || values.iter().any(|&v| v & !super::super::mask(50) != 0)
            {
                return Err(Error::Domain);
            }
            let mut h = Sha256::new();
            h.update(b"pllm.sigma.activation.common-external-input.v1");
            for (out, &value) in self.projection.iter_mut().zip(values) {
                h.update(value.to_le_bytes());
                *out = value & super::super::mask(38);
            }
            self.input_digest = h.finalize().into();
            self.x = Some(input); // Original owner and matrix guard stay intact.
            self.burn(1)?;
            let f = self
                .tr
                .as_mut()
                .ok_or(Error::Consumed)?
                .begin(&self.projection)?;
            self.phase = Phase::Tr;
            self.frame(FrameBody::Arithmetic(f))
        })();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn advance(&mut self, peer: &[u8]) -> Result<ActivationStep<'b>, Error> {
        let result = self.advance_inner(peer);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn advance_inner(&mut self, peer: &[u8]) -> Result<ActivationStep<'b>, Error> {
        let body = match self.phase {
            Phase::Tr => match self.tr.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
                arithmetic::Step::Frame(f) => FrameBody::Arithmetic(f),
                arithmetic::Step::Output(y) => {
                    self.y = Some(y);
                    self.tr = None;
                    self.burn(2)?;
                    let f = self
                        .drelu
                        .as_mut()
                        .ok_or(Error::Consumed)?
                        .begin(self.y.as_ref().ok_or(Error::Consumed)?.values())?;
                    self.phase = Phase::Drelu;
                    FrameBody::Arithmetic(f)
                }
            },
            Phase::Drelu => match self.drelu.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
                arithmetic::Step::Frame(_) => return Err(Error::Context),
                arithmetic::Step::Output(d) => {
                    self.d = Some(d);
                    self.drelu = None;
                    self.burn(3)?;
                    self.burn(4)?;
                    let f = self.prefix.as_mut().ok_or(Error::Consumed)?.begin(
                        self.d.as_ref().ok_or(Error::Consumed)?.values(),
                        self.y.as_ref().ok_or(Error::Consumed)?.values(),
                    )?;
                    self.phase = Phase::Prefix;
                    FrameBody::Selection(f)
                }
            },
            Phase::Prefix => {
                let p = self.prefix.as_mut().ok_or(Error::Consumed)?.finish(peer)?;
                self.burn(5)?;
                self.burn(6)?;
                let y = self.y.as_ref().ok_or(Error::Consumed)?;
                for i in 0..self.affine.len() {
                    self.affine[i] = affine_mask(p.values()[i], y.values()[i]);
                }
                self.burn(7)?;
                let k = 1u64 << self.spec.kind.index_bits();
                for value in &mut *self.affine {
                    *value = value.wrapping_sub(k) & super::super::mask(32);
                }
                self.burn(8)?;
                let f = self
                    .clip_drelu
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(&self.affine)?;
                self.prefix = None;
                self.y = None;
                self.phase = Phase::ClipDrelu;
                FrameBody::Arithmetic(f)
            }
            Phase::ClipDrelu => match self
                .clip_drelu
                .as_mut()
                .ok_or(Error::Consumed)?
                .advance(peer)?
            {
                arithmetic::Step::Frame(_) => return Err(Error::Context),
                arithmetic::Step::Output(h) => {
                    self.burn(9)?;
                    for (out, &value) in self.complement.iter_mut().zip(h.values()) {
                        *out = value ^ 1;
                    }
                    self.burn(10)?;
                    for (out, &value) in self.operand.iter_mut().zip(self.affine.iter()) {
                        *out =
                            value.wrapping_add(1) & super::super::mask(self.spec.kind.index_bits());
                    }
                    self.burn(11)?;
                    self.burn(12)?;
                    let f = self
                        .clip_select
                        .as_mut()
                        .ok_or(Error::Consumed)?
                        .begin(&self.complement, &self.operand)?;
                    self.clip_drelu = None;
                    self.phase = Phase::ClipSelect;
                    FrameBody::Selection(f)
                }
            },
            Phase::ClipSelect => {
                let v = self
                    .clip_select
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .finish(peer)?;
                self.burn(13)?;
                let k = 1u64 << self.spec.kind.index_bits();
                for (out, &value) in self.index.iter_mut().zip(v.values()) {
                    *out = value.wrapping_add(k - 1) & (k - 1);
                }
                self.burn(14)?;
                let f = self
                    .lookup
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .begin(&self.index)?;
                self.clip_select = None;
                self.phase = Phase::Lookup;
                FrameBody::Lookup(f)
            }
            Phase::Lookup => {
                let l = match self.lookup.as_mut().ok_or(Error::Consumed)?.advance(peer)? {
                    source_lookup::Step::Deferred(l) => l,
                    _ => return Err(Error::Context),
                };
                self.burn(15)?;
                self.burn(16)?;
                let f = self
                    .final_select
                    .take()
                    .ok_or(Error::Consumed)?
                    .into_deferred(
                        self.d.as_ref().ok_or(Error::Consumed)?.values(),
                        self.x.as_ref().ok_or(Error::Consumed)?.backing.values(),
                    )?;
                self.burn(17)?;
                self.burn(18)?;
                if f.role() != self.role
                    || l.role() != self.role
                    || f.bits() != 50
                    || l.bits() != 50
                    || f.lanes() != self.local.len()
                    || l.lanes() != self.local.len()
                    || *f.binding() != self.select_binding
                    || *l.binding() != self.lookup_binding
                {
                    return Err(Error::Context);
                }
                // Keep BOTH distinct deferred backing/lease owners alive through
                // transformation. Every role subtracts its own lookup share.
                for i in 0..self.local.len() {
                    self.local[i] =
                        f.values()[i].wrapping_sub(l.values()[i]) & super::super::mask(50);
                }
                self.lookup = None;
                self.d = None;
                // Original wide input's last use is now complete, guard included.
                self.x = None;
                self.burn(19)?;
                let bytes = self.final_bytes()?;
                self.phase = Phase::Final;
                FrameBody::Final(bytes)
            }
            Phase::Final => {
                self.validate_final(peer)?; // Full frame before any mutation.
                for (out, word) in self.local.iter_mut().zip(peer[HEADER..].chunks_exact(8)) {
                    *out = out.wrapping_add(u64::from_le_bytes(word.try_into().unwrap()))
                        & super::super::mask(50);
                }
                let values = std::mem::replace(&mut self.local, Zeroizing::new(Vec::new()));
                let lease = self
                    .lease
                    .as_mut()
                    .ok_or(Error::Consumed)?
                    .split(common_charge() + 8 * values.len() + ALLOCATOR)?;
                let output = CommonMasked {
                    backing: CommonBacking::Root(values),
                    binding: self.output,
                    role: self.role,
                    _kind: PhantomData,
                    _lease: lease,
                };
                self.cancel();
                return Ok(ActivationStep::Output(output));
            }
            Phase::Fresh | Phase::Dead => return Err(Error::Consumed),
        };
        Ok(ActivationStep::Frame(self.frame(body)?))
    }
    fn final_bytes(&self) -> Result<Zeroizing<Vec<u8>>, Error> {
        let budget = self.lease.as_ref().ok_or(Error::Consumed)?.budget;
        let mut bytes = Zeroizing::new(allocate(HEADER + 8 * self.local.len(), budget)?);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&self.binding);
        bytes.extend_from_slice(&self.input_digest);
        bytes.extend_from_slice(&[self.role, 8]);
        bytes.extend_from_slice(&(self.local.len() as u16).to_le_bytes());
        for value in &*self.local {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }
    fn validate_final(&self, peer: &[u8]) -> Result<(), Error> {
        if peer.len() != HEADER + 8 * self.local.len()
            || &peer[..8] != MAGIC
            || peer[8..40] != self.binding
            || peer[40..72] != self.input_digest
            || peer[72] != 1 - self.role
            || peer[73] != 8
            || peer[74..76] != (self.local.len() as u16).to_le_bytes()
        {
            return Err(Error::Context);
        }
        if peer[HEADER..]
            .chunks_exact(8)
            .any(|word| u64::from_le_bytes(word.try_into().unwrap()) & !super::super::mask(50) != 0)
        {
            return Err(Error::Domain);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
