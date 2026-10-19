//! Source-ordered arithmetic-to-Boolean conversion and signed MSB extraction.
//!
//! Three party-local owners perform the masked split, then public carry circuits.
//! Every AND uses the existing typed Boolean primitive and shared PRSS ledger.
//! No intermediate or final value is reconstructed by this module.

use super::{
    boolean::{self, Share},
    Operand, PartyRuntime, Phase, Status,
};
use crate::puma::prss::{fill_typed, typed_blocks, SourceParty, PRSS_PROFILE};
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "puma.spu-a42c261.a2b-msb.ring64.replicated3.v1";
pub const COMPACT_MSB_PROFILE: &str = "puma.signed-bucket-msb.compact.replicated3.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Bits,
    MostSignificantBit,
    /// Conversion in the public quotient ring Z/(2^bits). Additive components
    /// are reduced locally before the split; no private value chooses the width.
    /// This separate contract projects both XOR components to nominal width;
    /// fresh typed PRSS still consumes every charged backing-type AES block.
    LowBits(u8),
    /// Sign bit in a publicly proved signed bucket; not a membership check.
    /// Only native widths 8/16/32/64 are supported by the shrinking circuit.
    CompactMostSignificantBit(u8),
}

impl Kind {
    fn width(self) -> u8 {
        match self {
            Self::LowBits(bits) | Self::CompactMostSignificantBit(bits) => bits,
            _ => 64,
        }
    }

    fn rounds(self) -> u8 {
        match self {
            Self::LowBits(bits) => (u8::BITS - (bits - 1).leading_zeros()) as u8,
            Self::CompactMostSignificantBit(bits) => bits.ilog2() as u8,
            _ => 6,
        }
    }

    fn sign(self) -> bool {
        matches!(
            self,
            Self::MostSignificantBit | Self::CompactMostSignificantBit(_)
        )
    }

    fn compact(self) -> bool {
        matches!(self, Self::LowBits(_) | Self::CompactMostSignificantBit(_))
    }
}

/// One split plus seven Boolean gate rotations. Vmapped gates hold 2*n lanes,
/// so admission is at most half the underlying primitive's lane ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    operation: Operation,
    setup: [u8; 32],
    kind: Kind,
    pair_start: u64,
    gate_start: u64,
    pair_end: u64,
    admitted_bytes: usize,
    limits: Limits,
}

impl Plan {
    pub fn digest(self) -> [u8; 32] {
        self.operation.digest()
    }
    pub fn sequence(self) -> u64 {
        self.operation.sequence
    }
    pub fn lanes(self) -> usize {
        self.operation.lanes
    }
    pub fn kind(self) -> Kind {
        self.kind
    }
    pub fn admitted_bytes(self) -> usize {
        self.admitted_bytes
    }
    pub fn pair_blocks(self) -> u64 {
        self.pair_end - self.pair_start
    }
    pub fn rotations(self) -> usize {
        usize::from(self.kind.rounds()) + 2
    }

    pub fn input_binding(self) -> Operation {
        let mut op = self.operation;
        let mut h = Sha256::new();
        h.update(b"pllm.puma.conversion-input.v1");
        h.update(self.digest());
        op.consumer = Some(h.finalize().into());
        op
    }

    pub fn share_input(
        self,
        party: Party,
        first: &[u64],
        second: &[u64],
        limits: Limits,
    ) -> Result<ReplicatedShare, Error> {
        ReplicatedShare::from_local(self.input_binding(), party, first, second, limits)
    }

    pub(super) fn gate(self, round: u8) -> Result<boolean::Plan, Error> {
        if round > self.kind.rounds() {
            return Err(Error::Bounds);
        }
        let mut start = self.gate_start;
        for r in 0..round {
            let (lanes, bits) = self.geometry(r);
            start += typed_blocks(lanes, backing(bits))?;
        }
        let (lanes, bits) = self.geometry(round);
        let plan = boolean::Plan::nested(
            self.operation,
            self.setup,
            round,
            lanes,
            bits,
            start,
            self.limits,
        )?;
        Ok(if self.kind.compact() {
            plan.compact()
        } else {
            plan
        })
    }

    pub(super) fn nested_adder(
        operation: Operation,
        setup: [u8; 32],
        gate_start: u64,
        limits: Limits,
    ) -> Result<Self, Error> {
        let n = operation.lanes;
        limits.admit(2 * n)?;
        let pair_end = gate_start
            .checked_add(typed_blocks(n, 8)? + 6 * typed_blocks(2 * n, 8)?)
            .ok_or(Error::Bounds)?;
        Ok(Self {
            operation,
            setup,
            kind: Kind::Bits,
            pair_start: gate_start,
            gate_start,
            pair_end,
            admitted_bytes: FIXED_RESERVE_BYTES + 256 * n,
            limits,
        })
    }

    fn geometry(self, round: u8) -> (usize, u8) {
        if round == 0 {
            return (self.lanes(), self.kind.width());
        }
        let bits = match self.kind {
            Kind::Bits => 64,
            Kind::MostSignificantBit => 64 >> round,
            Kind::CompactMostSignificantBit(bits) => bits >> round,
            Kind::LowBits(bits) => bits,
        };
        (2 * self.lanes(), bits)
    }

    /// Bounded typed import for a declared circuit round (0 is the split).
    /// Receiving state rechecks the exact expected round; transport must supply
    /// authenticated private delivery and retire the runtime on decode failure.
    pub fn message_from_received(
        self,
        round: u8,
        header: Header,
        words: &[u64],
        limits: Limits,
    ) -> Result<Message, Error> {
        if round == 0 {
            if header.phase != MessagePhase::ArithmeticToBoolean {
                return Err(Error::Binding);
            }
            Message::from_received(header, words, self.operation, limits)
        } else {
            self.gate(round - 1)?
                .message_from_received(header, words, limits)
        }
    }
}

// Keep bounded metadata inline. A Box here would add an infallible allocation
// on every gate; the complete runtime is checked against its fixed reserve.
#[allow(clippy::large_enum_variant)]
pub(super) enum State {
    Planned(Plan),
    Split {
        plan: Plan,
        first: Words,
        other: ReplicatedShare,
    },
    Circuit(Work),
    Complete(Share),
}

pub(super) struct Work {
    plan: Plan,
    initial: Share,
    propagate: Share,
    generate: Option<Share>,
    gate: boolean::Gate,
    round: u8,
}

impl State {
    pub(super) fn receive_layout(&self) -> Result<([u8; 32], MessagePhase, usize, u8), Error> {
        match self {
            Self::Split { plan, .. } => Ok((
                plan.digest(),
                MessagePhase::ArithmeticToBoolean,
                plan.lanes(),
                plan.kind.width(),
            )),
            Self::Circuit(work) => {
                let gate = work.plan.gate(work.round)?;
                Ok((
                    gate.digest(),
                    MessagePhase::BooleanAnd,
                    gate.lanes(),
                    gate.output_bits(),
                ))
            }
            _ => Err(Error::Phase),
        }
    }
    pub(super) fn status(&self) -> Status {
        match self {
            Self::Planned(_) => Status::Planned,
            Self::Split { .. } | Self::Circuit(_) => Status::AwaitingRotate,
            Self::Complete(_) => Status::Complete,
        }
    }
}

fn backing(bits: u8) -> usize {
    match bits {
        0..=8 => 1,
        9..=16 => 2,
        17..=32 => 4,
        _ => 8,
    }
}

fn pair(
    usage: &PayloadUsage,
    operation: Operation,
    party: Party,
    bits: u8,
) -> Result<Share, Error> {
    Ok(Share {
        value: ReplicatedShare {
            operation,
            party,
            first: usage.zeros(operation.lanes)?,
            second: usage.zeros(operation.lanes)?,
        },
        bits,
    })
}

fn component(value: &Share, second: bool) -> &[u64] {
    if second {
        value.second()
    } else {
        value.first()
    }
}

fn component_mut(value: &mut Share, second: bool) -> &mut [u64] {
    if second {
        &mut value.value.second
    } else {
        &mut value.value.first
    }
}

// Source bit_scatter applies the public permutation to EVERY backing-type block,
// including padding bits above nominal widths 2/4. Its high half is a typed right
// shift, not a mask to the nominal width. Retaining those randomized bits keeps
// complete source share transcripts, not only reconstructed low bits, exact.
fn scatter(value: u64, bits: u8) -> (u64, u64) {
    let mut reordered = 0_u64;
    for base in (0..8 * backing(bits)).step_by(bits as usize) {
        for i in 0..usize::from(bits) {
            let position = base + i / 2 + (i % 2) * usize::from(bits / 2);
            reordered |= ((value >> (base + i)) & 1) << position;
        }
    }
    let half = bits / 2;
    let low = reordered & ((1_u64 << half) - 1);
    let high = (reordered >> half) & (u64::MAX >> (64 - 8 * backing(half)));
    (low, high)
}

impl PartyRuntime {
    pub fn plan_conversion(&mut self, lanes: usize, kind: Kind) -> Result<Plan, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            let limits = self.source.limits.core;
            if matches!(kind, Kind::LowBits(bits) if !(1..=64).contains(&bits)) {
                return Err(Error::Bounds);
            }
            if matches!(kind, Kind::CompactMostSignificantBit(bits) if ![8, 16, 32, 64].contains(&bits))
            {
                return Err(Error::Bounds);
            }
            limits.admit(lanes.checked_mul(2).ok_or(Error::Bounds)?)?;
            // Prices state, both vmapped operands, expansion and ALL original
            // outgoing frames if the caller retains them until completion.
            let admitted_bytes = FIXED_RESERVE_BYTES + 32 * lanes * 8;
            let usage = self.payload_usage();
            if usage
                .live_bytes()
                .checked_add(admitted_bytes - FIXED_RESERVE_BYTES)
                .is_none_or(|v| v > usage.limit_bytes())
            {
                return Err(Error::Bounds);
            }
            let pair_start = self.source.cursors.pair_blocks;
            let mut count = 2 * typed_blocks(lanes, backing(kind.width()))?; // split + initial AND
            for round in 1..=kind.rounds() {
                let bits = match kind {
                    Kind::Bits => 64,
                    Kind::MostSignificantBit => 64 >> round,
                    Kind::CompactMostSignificantBit(bits) => bits >> round,
                    Kind::LowBits(bits) => bits,
                };
                count += typed_blocks(2 * lanes, backing(bits))?;
            }
            let pair_end = pair_start.checked_add(count).ok_or(Error::Bounds)?;
            self.source.work(2 * count)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                lanes,
                1,
                limits,
            )?;
            let mut hash = Sha256::new();
            hash.update(PROFILE.as_bytes());
            hash.update(PRSS_PROFILE.as_bytes());
            hash.update(self.source.setup);
            match kind {
                Kind::Bits => hash.update([0]),
                Kind::MostSignificantBit => hash.update([1]),
                Kind::CompactMostSignificantBit(bits) => {
                    hash.update([3, bits]);
                    hash.update(COMPACT_MSB_PROFILE.as_bytes());
                }
                Kind::LowBits(bits) => {
                    hash.update([2, bits]);
                    hash.update(b"nominal-xor-domain.compact-wire.v1");
                }
            }
            hash.update(pair_start.to_le_bytes());
            hash.update(pair_end.to_le_bytes());
            operation.consumer = Some(hash.finalize().into());
            let plan = Plan {
                operation,
                setup: self.source.setup,
                kind,
                pair_start,
                gate_start: pair_start + typed_blocks(lanes, backing(kind.width()))?,
                pair_end,
                admitted_bytes,
                limits,
            };
            self.phase = Phase::Conversion(State::Planned(plan));
            Ok(plan)
        })();
        self.checked(result)
    }

    pub fn begin_conversion(&mut self, input: ReplicatedShare) -> Result<Message, Error> {
        let result = self.begin_conversion_inner(input);
        self.checked(result)
    }

    /// Move an existing arithmetic owner into the planned conversion without
    /// exporting/copying its private components. Caller/compiler binds the graph.
    pub fn bind_conversion_input(
        &mut self,
        mut input: ReplicatedShare,
    ) -> Result<ReplicatedShare, Error> {
        let result = (|| {
            self.source.live()?;
            let Phase::Conversion(State::Planned(plan)) = self.phase else {
                return Err(Error::Phase);
            };
            if input.party != self.party() {
                return Err(Error::Role);
            }
            if input.operation.context != self.context() {
                return Err(Error::Binding);
            }
            if input.first.len() != plan.lanes() || input.second.len() != plan.lanes() {
                return Err(Error::Shape);
            }
            input.charge_to(&self.payload_usage())?;
            input.operation = plan.input_binding();
            Ok(input)
        })();
        self.checked(result)
    }

    fn begin_conversion_inner(&mut self, mut input: ReplicatedShare) -> Result<Message, Error> {
        self.source.live()?;
        let Phase::Conversion(State::Planned(plan)) = self.phase else {
            return Err(Error::Phase);
        };
        if input.operation != plan.input_binding() {
            return Err(Error::Binding);
        }
        if input.party() != self.party() {
            return Err(Error::Role);
        }
        if self.source.cursors.pair_blocks != plan.pair_start {
            return Err(Error::Binding);
        }
        self.source.runtime.consume_ordinal(plan.operation)?;
        let n = plan.lanes();
        let bytes = backing(plan.kind.width());
        let q = typed_blocks(n, bytes)?;
        self.source.cursors.aes_work = self.source.work(2 * q)?;
        self.source.cursors.pair_blocks += q;
        let usage = self.payload_usage();
        input.charge_to(&usage)?;
        let mut first = usage.zeros(n)?;
        let mut adjacent = usage.zeros(n)?;
        let seeds = self.source.seeds.as_ref().ok_or(Error::Retired)?;
        fill_typed(&seeds.own, plan.pair_start, &mut first, bytes)?;
        fill_typed(
            seeds.next.as_ref().ok_or(Error::Phase)?,
            plan.pair_start,
            &mut adjacent,
            bytes,
        )?;
        for i in 0..n {
            first[i] ^= adjacent[i];
            if self.party() == Party::P0 {
                first[i] ^= input.first[i].wrapping_add(input.second[i])
                    & (u64::MAX >> (64 - plan.kind.width()));
            }
            if plan.kind.compact() {
                first[i] &= u64::MAX >> (64 - plan.kind.width());
            }
            // Reduction of every additive component is a ring homomorphism;
            // never reinterpret or independently truncate a signed share.
            input.first[i] &= u64::MAX >> (64 - plan.kind.width());
            input.second[i] &= u64::MAX >> (64 - plan.kind.width());
            if self.party() != Party::P2 {
                input.first[i] = 0;
            }
            if self.party() != Party::P1 {
                input.second[i] = 0;
            }
        }
        drop(adjacent);
        let message = Message {
            header: Header {
                binding: plan.digest(),
                from: self.party(),
                to: self.party().next().next(),
                phase: MessagePhase::ArithmeticToBoolean,
            },
            words: usage.copy(&first, n)?,
        };
        self.phase = Phase::Conversion(State::Split {
            plan,
            first,
            other: input,
        });
        Ok(message)
    }

    /// Consume the next-role rotation and emit the next public circuit round.
    /// `None` means complete; output remains opaque until explicit extraction.
    pub fn advance_conversion(&mut self, message: Message) -> Result<Option<Message>, Error> {
        let result = self.advance_conversion_inner(message);
        self.checked(result)
    }

    fn advance_conversion_inner(&mut self, mut message: Message) -> Result<Option<Message>, Error> {
        self.source.live()?;
        let state = std::mem::replace(&mut self.phase, Phase::Retired);
        let (state, message) = match state {
            Phase::Conversion(State::Split { plan, first, other }) => {
                message.validate(
                    plan.operation,
                    self.party().next(),
                    self.party(),
                    MessagePhase::ArithmeticToBoolean,
                )?;
                let mask = if plan.kind.compact() {
                    u64::MAX >> (64 - plan.kind.width())
                } else {
                    u64::MAX >> (64 - 8 * backing(plan.kind.width()))
                };
                if message.words().iter().any(|&word| word & !mask != 0) {
                    return Err(Error::Bounds);
                }
                let usage = self.payload_usage();
                message.words.charge_to(&usage)?;
                let left = Share {
                    value: ReplicatedShare {
                        operation: plan.operation,
                        party: self.party(),
                        first,
                        second: message.words,
                    },
                    bits: plan.kind.width(),
                };
                let right = Share {
                    value: other,
                    bits: plan.kind.width(),
                };
                let (work, outgoing) = start(&mut self.source, plan, left, right)?;
                (State::Circuit(work), Some(outgoing))
            }
            Phase::Conversion(State::Circuit(work)) => advance(&mut self.source, work, message)?,
            _ => return Err(Error::Phase),
        };
        self.phase = Phase::Conversion(state);
        Ok(message)
    }
}

pub(super) fn start(
    source: &mut SourceParty,
    plan: Plan,
    mut left: Share,
    mut right: Share,
) -> Result<(Work, Message), Error> {
    for value in [&left, &right] {
        if value.bits != plan.kind.width()
            || value.first().len() != plan.lanes()
            || value.second().len() != plan.lanes()
        {
            return Err(Error::Shape);
        }
        if value.party() != source.party() || value.value.operation.context != source.context() {
            return Err(Error::Binding);
        }
    }
    let usage = source.payload_usage();
    let gate_plan = plan.gate(0)?;
    left.value.operation = gate_plan.operand(Operand::Left).0;
    right.value.operation = gate_plan.operand(Operand::Right).0;
    let mut propagate = pair(&usage, plan.operation, source.party(), plan.kind.width())?;
    let mut initial = pair(
        &usage,
        plan.operation,
        source.party(),
        if plan.kind.sign() {
            1
        } else {
            plan.kind.width()
        },
    )?;
    for second in [false, true] {
        for i in 0..plan.lanes() {
            let xor = component(&left, second)[i] ^ component(&right, second)[i];
            component_mut(&mut propagate, second)[i] = xor;
            component_mut(&mut initial, second)[i] = if !plan.kind.sign() {
                xor
            } else {
                xor >> (plan.kind.width() - 1)
            };
        }
    }
    let (gate, outgoing) = boolean::issue(source, gate_plan, &mut left, &mut right)?;
    Ok((
        Work {
            plan,
            initial,
            propagate,
            generate: None,
            gate,
            round: 0,
        },
        outgoing,
    ))
}

pub(super) fn advance(
    source: &mut SourceParty,
    work: Work,
    message: Message,
) -> Result<(State, Option<Message>), Error> {
    let Work {
        plan,
        mut initial,
        mut propagate,
        generate,
        gate,
        round,
    } = work;
    let output = gate.receive(source, message)?;
    let usage = source.payload_usage();
    let n = plan.lanes();
    let mut generate = if round == 0 {
        output
    } else {
        let mut generate = generate.ok_or(Error::Phase)?;
        for second in [false, true] {
            for i in 0..n {
                component_mut(&mut propagate, second)[i] = component(&output, second)[i];
                component_mut(&mut generate, second)[i] ^= component(&output, second)[n + i];
            }
        }
        propagate.bits = plan.geometry(round).1;
        generate.bits = propagate.bits;
        drop(output);
        generate
    };
    if round == plan.kind.rounds() {
        for second in [false, true] {
            for i in 0..n {
                let carry = component(&generate, second)[i];
                component_mut(&mut initial, second)[i] ^=
                    if !plan.kind.sign() { carry << 1 } else { carry };
                if let Kind::LowBits(bits) = plan.kind {
                    component_mut(&mut initial, second)[i] &= u64::MAX >> (64 - bits);
                }
                if matches!(plan.kind, Kind::CompactMostSignificantBit(_)) {
                    component_mut(&mut initial, second)[i] &= 1;
                }
            }
        }
        if source.cursors.pair_blocks != plan.pair_end {
            return Err(Error::Binding);
        }
        initial.value.operation = plan.operation;
        return Ok((State::Complete(initial), None));
    }
    let next_round = round + 1;
    let gate_plan = plan.gate(next_round)?;
    let mut left = pair(
        &usage,
        gate_plan.operand(Operand::Left).0,
        source.party(),
        gate_plan.output_bits(),
    )?;
    let mut right = pair(
        &usage,
        gate_plan.operand(Operand::Right).0,
        source.party(),
        gate_plan.output_bits(),
    )?;
    for second in [false, true] {
        for i in 0..n {
            let mut p = component(&propagate, second)[i];
            let mut g = component(&generate, second)[i];
            if !plan.kind.sign() {
                let shift = 1_u32 << round;
                component_mut(&mut left, second)[i] = p;
                component_mut(&mut left, second)[n + i] = p;
                component_mut(&mut right, second)[i] = p << shift;
                component_mut(&mut right, second)[n + i] = g << shift;
            } else {
                if round == 0 {
                    p <<= 1;
                    g <<= 1;
                } // public odd 63-bit carry padding
                let (p0, p1) = scatter(p, propagate.bits);
                let (g0, g1) = scatter(g, generate.bits);
                component_mut(&mut left, second)[i] = p0;
                component_mut(&mut left, second)[n + i] = g0;
                component_mut(&mut right, second)[i] = p1;
                component_mut(&mut right, second)[n + i] = p1;
                component_mut(&mut generate, second)[i] = g1;
            }
        }
    }
    let (gate, outgoing) = boolean::issue(source, gate_plan, &mut left, &mut right)?;
    Ok((
        State::Circuit(Work {
            plan,
            initial,
            propagate,
            generate: Some(generate),
            gate,
            round: next_round,
        }),
        Some(outgoing),
    ))
}

#[cfg(test)]
#[path = "conversion_tests.rs"]
mod tests;
