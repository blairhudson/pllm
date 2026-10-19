//! Pinned SPU B2A selector: three-party OT at 0..=8 bits, PPA at 9..=64.
//!
//! The helper, sender and receiver have separate party-local views. Pairwise
//! PRSS provides the source three-party OT masks; no dealer reconstructs bits.
//! Higher widths reuse the shared carry circuit and the source masked reveal.

use super::{boolean::Share, PartyRuntime, Phase, Status};
use crate::puma::prss::{blocks, fill, SourceParty, PRSS_PROFILE};
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "puma.spu-a42c261.b2a-by-ot.ring64.replicated3.v1";
pub const PPA_PROFILE: &str = "puma.spu-a42c261.b2a-by-ppa.ring64.replicated3.v1";

#[path = "boolean_to_arithmetic_ppa.rs"]
mod ppa;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    operation: Operation,
    input_binding: [u8; 32],
    bits: u8,
    pair_start: u64,
    pair_end: u64,
    admitted_bytes: usize,
    setup: [u8; 32],
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
    pub fn bits(self) -> u8 {
        self.bits
    }
    pub fn helper(self) -> Party {
        self.operation.pivot
    }
    pub fn receiver(self) -> Party {
        self.helper().next()
    }
    pub fn sender(self) -> Party {
        self.receiver().next()
    }
    pub fn pair_blocks(self) -> u64 {
        self.pair_end - self.pair_start
    }
    pub fn admitted_bytes(self) -> usize {
        self.admitted_bytes
    }
    fn expanded(self) -> usize {
        self.lanes() * usize::from(self.bits)
    }

    fn message_operation(self, phase: MessagePhase) -> Result<Operation, Error> {
        if self.bits == 0 {
            return Err(Error::Phase);
        }
        let (tag, lanes) = match phase {
            MessagePhase::BooleanOtMask => (0_u8, self.expanded()),
            MessagePhase::BooleanOtZero => (1, self.expanded()),
            MessagePhase::BooleanOtOne => (2, self.expanded()),
            MessagePhase::BooleanOtReshare => (3, self.lanes()),
            _ => return Err(Error::Phase),
        };
        let mut op = self.operation;
        op.lanes = lanes;
        let mut h = Sha256::new();
        h.update(b"pllm.puma.b2a-message.v1");
        h.update(self.digest());
        h.update([tag]);
        op.consumer = Some(h.finalize().into());
        Ok(op)
    }

    fn endpoints(self, phase: MessagePhase) -> Result<(Party, Party), Error> {
        match phase {
            MessagePhase::BooleanOtMask => Ok((self.helper(), self.receiver())),
            MessagePhase::BooleanOtZero | MessagePhase::BooleanOtOne => {
                Ok((self.sender(), self.receiver()))
            }
            MessagePhase::BooleanOtReshare => Ok((self.receiver(), self.helper())),
            _ => Err(Error::Phase),
        }
    }

    pub fn message_from_received(
        self,
        header: Header,
        words: &[u64],
        limits: Limits,
    ) -> Result<Message, Error> {
        if self.bits > 8 {
            return ppa::import(self, header, words, limits);
        }
        let (from, to) = self.endpoints(header.phase)?;
        if header.from != from || header.to != to {
            return Err(Error::Role);
        }
        Message::from_received(header, words, self.message_operation(header.phase)?, limits)
    }

    fn frame(self, phase: MessagePhase, words: Words) -> Result<Message, Error> {
        let operation = self.message_operation(phase)?;
        if words.len() != operation.lanes {
            return Err(Error::Shape);
        }
        let (from, to) = self.endpoints(phase)?;
        Ok(Message {
            header: Header {
                binding: operation.digest(),
                from,
                to,
                phase,
            },
            words,
        })
    }

    fn accept(
        self,
        phase: MessagePhase,
        mut frame: Message,
        usage: &PayloadUsage,
    ) -> Result<Words, Error> {
        let (from, to) = self.endpoints(phase)?;
        frame.validate(self.message_operation(phase)?, from, to, phase)?;
        frame.words.charge_to(usage)?;
        Ok(frame.words)
    }
}

// Bounded inline metadata avoids an unchecked Box allocation in each circuit.
#[allow(clippy::large_enum_variant)]
pub(super) enum State {
    Planned(Plan),
    Helper {
        plan: Plan,
        first: Words,
    },
    Receiver {
        plan: Plan,
        choices: Words,
        second: Words,
    },
    Complete(ReplicatedShare),
    Ppa(ppa::State),
}

impl State {
    /// Compact equality only uses the one-bit OT path. Validate exact public
    /// role/phase/binding before its full-ring arithmetic payload is allocated.
    pub(super) fn validate_compact_header(&self, header: Header) -> Result<(), Error> {
        let plan = match self {
            Self::Helper { plan, .. } if header.phase == MessagePhase::BooleanOtReshare => plan,
            Self::Receiver { plan, .. }
                if matches!(
                    header.phase,
                    MessagePhase::BooleanOtMask
                        | MessagePhase::BooleanOtZero
                        | MessagePhase::BooleanOtOne
                ) =>
            {
                plan
            }
            _ => return Err(Error::Phase),
        };
        if plan.bits != 1 {
            return Err(Error::Binding);
        }
        let (from, to) = plan.endpoints(header.phase)?;
        if header.from != from || header.to != to {
            return Err(Error::Role);
        }
        if header.binding != plan.message_operation(header.phase)?.digest() {
            return Err(Error::Binding);
        }
        Ok(())
    }

    pub(super) fn status(&self) -> Status {
        match self {
            Self::Planned(_) => Status::Planned,
            Self::Helper { .. } => Status::AwaitingReshare,
            Self::Receiver { .. } => Status::AwaitingOpen,
            Self::Complete(_) => Status::Complete,
            Self::Ppa(state) => state.status(),
        }
    }
}

// Selection bit is 0 or 1. No secret-dependent branches or addresses.
fn choose(zero: u64, one: u64, bit: u64) -> u64 {
    zero ^ ((zero ^ one) & 0_u64.wrapping_sub(bit))
}

fn compose(words: &[u64], plan: Plan, usage: &PayloadUsage) -> Result<Words, Error> {
    let mut out = usage.zeros(plan.lanes())?;
    for (lane, value) in out.iter_mut().enumerate() {
        for bit in 0..usize::from(plan.bits) {
            *value = value.wrapping_add(words[lane * usize::from(plan.bits) + bit] << bit);
        }
    }
    Ok(out)
}

impl PartyRuntime {
    /// Bind an existing party-local Boolean owner, its public width and source
    /// identity. Shape/work/payload admission precedes the source public pivot.
    /// Zero-width values consume no draws; 1..8 uses OT and 9..64 uses source PPA.
    pub fn plan_boolean_to_arithmetic(&mut self, input: &Share) -> Result<Plan, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            if input.party() != self.party() {
                return Err(Error::Role);
            }
            if input.value.operation.context != self.context() || input.bits() > 64 {
                return Err(Error::Binding);
            }
            let lanes = input.value.operation.lanes;
            let expanded = lanes
                .checked_mul(usize::from(input.bits()))
                .ok_or(Error::Bounds)?;
            let ppa = input.bits() > 8;
            self.source
                .limits
                .core
                .admit(if ppa { 2 * lanes } else { lanes.max(expanded) })?;
            let admitted_bytes = FIXED_RESERVE_BYTES
                + if ppa {
                    320 * lanes
                } else {
                    (16 * expanded + 8 * lanes) * 8
                };
            let usage = self.payload_usage();
            if usage
                .live_bytes()
                .checked_add(admitted_bytes - FIXED_RESERVE_BYTES)
                .is_none_or(|v| v > usage.limit_bytes())
            {
                return Err(Error::Bounds);
            }
            let q = if input.bits() == 0 {
                0
            } else if ppa {
                blocks(lanes)?
            } else {
                blocks(expanded)?
            };
            let pair_start = self.source.cursors.pair_blocks;
            let count = 3 * q + if ppa { 6 * blocks(2 * lanes)? } else { 0 };
            let pair_end = pair_start.checked_add(count).ok_or(Error::Bounds)?;
            let public_start = self.source.cursors.public_blocks;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                lanes,
                1,
                self.source.limits.core,
            )?;
            if ppa {
                self.source.work(2 * count)?;
            } else if input.bits() != 0 {
                self.source.work(4 * q + 1)?;
                self.source.cursors.public_blocks =
                    public_start.checked_add(1).ok_or(Error::Bounds)?;
                self.source.cursors.aes_work = self.source.work(1)?;
                let mut word = [0];
                fill(
                    &self.source.public_seed.to_le_bytes(),
                    public_start,
                    &mut word,
                )?;
                // Match source raw+1/raw+2; reject its two overflowing aliases.
                word[0].checked_add(2).ok_or(Error::Bounds)?;
                operation.pivot = [Party::P0, Party::P1, Party::P2][(word[0] % 3) as usize];
            }
            let mut hash = Sha256::new();
            hash.update(if ppa { PPA_PROFILE } else { PROFILE }.as_bytes());
            hash.update(PRSS_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(input.binding());
            hash.update([input.bits()]);
            hash.update(public_start.to_le_bytes());
            hash.update(pair_start.to_le_bytes());
            hash.update(pair_end.to_le_bytes());
            operation.consumer = Some(hash.finalize().into());
            let plan = Plan {
                operation,
                input_binding: input.binding(),
                bits: input.bits(),
                pair_start,
                pair_end,
                admitted_bytes,
                setup: self.source.setup,
                limits: self.source.limits.core,
            };
            self.phase = Phase::BooleanToArithmetic(State::Planned(plan));
            Ok(plan)
        })();
        self.checked(result)
    }

    pub fn begin_boolean_to_arithmetic(
        &mut self,
        input: Share,
    ) -> Result<[Option<Message>; 2], Error> {
        let result = self.begin_boolean_to_arithmetic_inner(input);
        self.checked(result)
    }

    fn begin_boolean_to_arithmetic_inner(
        &mut self,
        mut input: Share,
    ) -> Result<[Option<Message>; 2], Error> {
        self.source.live()?;
        let Phase::BooleanToArithmetic(State::Planned(plan)) = self.phase else {
            return Err(Error::Phase);
        };
        if input.binding() != plan.input_binding
            || input.bits() != plan.bits
            || self.source.cursors.pair_blocks != plan.pair_start
        {
            return Err(Error::Binding);
        }
        if input.party() != self.party() {
            return Err(Error::Role);
        }
        self.source.runtime.consume_ordinal(plan.operation)?;
        let usage = self.payload_usage();
        input.value.charge_to(&usage)?;
        let (state, messages) = issue(&mut self.source, plan, input)?;
        self.phase = Phase::BooleanToArithmetic(state);
        Ok(messages)
    }

    /// Helper receives [c2,None,None]. Receiver receives [mask,m0,m1].
    /// Sender has no receive phase. Invalid phases/routes retire the owner.
    pub fn receive_boolean_to_arithmetic(
        &mut self,
        messages: [Option<Message>; 3],
    ) -> Result<Option<Message>, Error> {
        let result = self.receive_boolean_to_arithmetic_inner(messages);
        self.checked(result)
    }

    fn receive_boolean_to_arithmetic_inner(
        &mut self,
        messages: [Option<Message>; 3],
    ) -> Result<Option<Message>, Error> {
        self.source.live()?;
        let state = std::mem::replace(&mut self.phase, Phase::Retired);
        let usage = self.payload_usage();
        let (plan, first, second, reply) = match (state, messages) {
            (
                Phase::BooleanToArithmetic(State::Helper { plan, first }),
                [Some(reply), None, None],
            ) => {
                let second = plan.accept(MessagePhase::BooleanOtReshare, reply, &usage)?;
                (plan, first, second, None)
            }
            (
                Phase::BooleanToArithmetic(State::Receiver {
                    plan,
                    choices,
                    second,
                }),
                [Some(mc), Some(m0), Some(m1)],
            ) => {
                let mc = plan.accept(MessagePhase::BooleanOtMask, mc, &usage)?;
                let m0 = plan.accept(MessagePhase::BooleanOtZero, m0, &usage)?;
                let m1 = plan.accept(MessagePhase::BooleanOtOne, m1, &usage)?;
                let mut first = usage.zeros(plan.lanes())?;
                for lane in 0..plan.lanes() {
                    for bit in 0..usize::from(plan.bits) {
                        let i = lane * usize::from(plan.bits) + bit;
                        let value = choose(m0[i], m1[i], (choices[lane] >> bit) & 1) ^ mc[i];
                        first[lane] = first[lane].wrapping_add(value << bit);
                    }
                }
                let reply = plan.frame(
                    MessagePhase::BooleanOtReshare,
                    usage.copy(&first, plan.lanes())?,
                )?;
                (plan, first, second, Some(reply))
            }
            _ => return Err(Error::Phase),
        };
        self.phase = Phase::BooleanToArithmetic(State::Complete(ReplicatedShare {
            operation: plan.operation,
            party: self.party(),
            first,
            second,
        }));
        Ok(reply)
    }

    /// Wider PPA conversion consumes one directed frame per step and can emit
    /// both a masked reveal and a final rotation. The OT receive API stays distinct.
    pub fn advance_boolean_to_arithmetic(
        &mut self,
        message: Message,
    ) -> Result<[Option<Message>; 2], Error> {
        let result = (|| {
            self.source.live()?;
            let Phase::BooleanToArithmetic(State::Ppa(state)) =
                std::mem::replace(&mut self.phase, Phase::Retired)
            else {
                return Err(Error::Phase);
            };
            let (state, replies) = ppa::advance(&mut self.source, state, message)?;
            self.phase = Phase::BooleanToArithmetic(state);
            Ok(replies)
        })();
        self.checked(result)
    }
}

fn issue(
    source: &mut SourceParty,
    plan: Plan,
    input: Share,
) -> Result<(State, [Option<Message>; 2]), Error> {
    if plan.bits > 8 {
        return ppa::issue(source, plan, input);
    }
    let usage = source.payload_usage();
    if plan.bits == 0 {
        drop(input);
        return Ok((
            State::Complete(ReplicatedShare {
                operation: plan.operation,
                party: source.party(),
                first: usage.zeros(plan.lanes())?,
                second: usage.zeros(plan.lanes())?,
            }),
            [None, None],
        ));
    }
    let q = blocks(plan.expanded())?;
    let work = source.work(if source.party() == plan.receiver() {
        2 * q
    } else {
        4 * q
    })?;
    source.cursors.pair_blocks = plan.pair_end;
    source.cursors.aes_work = work;
    let seeds = source.seeds.as_ref().ok_or(Error::Retired)?;
    let next_seed = seeds.next.as_ref().ok_or(Error::Phase)?;
    let mut r0 = usage.zeros(plan.expanded())?;
    let mut r1 = usage.zeros(plan.expanded())?;
    fill(&seeds.own, plan.pair_start, &mut r0)?;
    fill(next_seed, plan.pair_start, &mut r1)?;
    if source.party() == plan.helper() {
        let first = compose(&r0, plan, &usage)?;
        drop(r0);
        drop(r1);
        let mut m0 = usage.zeros(plan.expanded())?;
        let mut m1 = usage.zeros(plan.expanded())?;
        fill(&seeds.own, plan.pair_start + q, &mut m0)?;
        fill(&seeds.own, plan.pair_start + 2 * q, &mut m1)?;
        for lane in 0..plan.lanes() {
            for bit in 0..usize::from(plan.bits) {
                let i = lane * usize::from(plan.bits) + bit;
                m0[i] = choose(m0[i], m1[i], (input.second()[lane] >> bit) & 1);
            }
        }
        let message = plan.frame(MessagePhase::BooleanOtMask, m0)?;
        Ok((State::Helper { plan, first }, [Some(message), None]))
    } else if source.party() == plan.receiver() {
        let second = compose(&r1, plan, &usage)?;
        // Source None/None calls consume pair counters without AES work.
        Ok((
            State::Receiver {
                plan,
                choices: input.value.first,
                second,
            },
            [None, None],
        ))
    } else {
        let first = compose(&r0, plan, &usage)?;
        let second = compose(&r1, plan, &usage)?;
        for lane in 0..plan.lanes() {
            for bit in 0..usize::from(plan.bits) {
                let i = lane * usize::from(plan.bits) + bit;
                let value = ((input.first()[lane] ^ input.second()[lane]) >> bit) & 1;
                let sum = r0[i].wrapping_add(r1[i]);
                r0[i] = value.wrapping_sub(sum);
                r1[i] = (value ^ 1).wrapping_sub(sum);
            }
        }
        let mut mask = usage.zeros(plan.expanded())?;
        fill(next_seed, plan.pair_start + q, &mut mask)?;
        for i in 0..plan.expanded() {
            r0[i] ^= mask[i];
        }
        fill(next_seed, plan.pair_start + 2 * q, &mut mask)?;
        for i in 0..plan.expanded() {
            r1[i] ^= mask[i];
        }
        let m0 = plan.frame(MessagePhase::BooleanOtZero, r0)?;
        let m1 = plan.frame(MessagePhase::BooleanOtOne, r1)?;
        Ok((
            State::Complete(ReplicatedShare {
                operation: plan.operation,
                party: source.party(),
                first,
                second,
            }),
            [Some(m0), Some(m1)],
        ))
    }
}

#[cfg(test)]
#[path = "boolean_to_arithmetic_tests.rs"]
mod tests;
