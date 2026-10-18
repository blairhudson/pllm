//! Party-local Boolean products on the compound PUMA source stream.
//!
//! Specification: pinned SPU `aby3/boolean.cc` AndBB. The three replicated
//! cross-products use XOR/AND, one typed Both PRSS fill and one rotation. This
//! module neither reconstructs inputs nor provides a dealer or transport.

use super::{Operand, PartyRuntime, Phase, Status};
use crate::puma::prss::{fill_typed, typed_blocks, PRSS_PROFILE};
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "puma.spu-a42c261.and-bb.typed.replicated3.v1";

fn word_bytes(bits: u8) -> Result<usize, Error> {
    match bits {
        0..=8 => Ok(1),
        9..=16 => Ok(2),
        17..=32 => Ok(4),
        33..=64 => Ok(8),
        _ => Err(Error::Bounds),
    }
}

fn word_mask(bytes: usize) -> u64 {
    u64::MAX >> (64 - 8 * bytes)
}

/// The two local XOR-share components only. Nominal width describes the
/// reconstructed value; individual randomized components fill the backing type.
/// Caller/compiler must establish that public width, without opening a value.
/// Boolean and additive shares are deliberately different types.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::boolean::Share;
/// fn duplicate(value: Share) { let _ = value.clone(); }
/// ```
pub struct Share {
    value: ReplicatedShare,
    bits: u8,
}

impl Share {
    pub fn first(&self) -> &[u64] {
        self.value.first()
    }
    pub fn second(&self) -> &[u64] {
        self.value.second()
    }
    pub fn party(&self) -> Party {
        self.value.party()
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }
    pub fn binding(&self) -> [u8; 32] {
        self.value.operation.digest()
    }
}

/// Immutable geometry/source admission; no secret material or PRSS draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    operation: Operation,
    left_bits: u8,
    right_bits: u8,
    bytes: usize,
    pair_start: u64,
    admitted_bytes: usize,
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
    pub fn output_bits(self) -> u8 {
        self.left_bits.max(self.right_bits)
    }
    pub fn backing_bytes(self) -> usize {
        self.bytes
    }
    pub fn admitted_bytes(self) -> usize {
        self.admitted_bytes
    }

    fn operand(self, operand: Operand) -> (Operation, u8) {
        let (index, bits) = match operand {
            Operand::Left => (0_u8, self.left_bits),
            Operand::Right => (1_u8, self.right_bits),
        };
        let mut op = self.operation;
        let mut hash = Sha256::new();
        hash.update(b"pllm.puma.boolean-operand.v1");
        hash.update(self.digest());
        hash.update([index, bits]);
        op.consumer = Some(hash.finalize().into());
        (op, bits)
    }

    /// Bounded outer import; runtime admission charges the original buffers.
    /// Width validation concerns each backing-type word, not private values.
    pub fn share_operand(
        self,
        operand: Operand,
        party: Party,
        first: &[u64],
        second: &[u64],
        limits: Limits,
    ) -> Result<Share, Error> {
        let (operation, bits) = self.operand(operand);
        let mask = word_mask(word_bytes(bits)?);
        if first.len() != self.lanes() || second.len() != self.lanes() {
            return Err(Error::Shape);
        }
        if first.iter().chain(second).any(|&v| v & !mask != 0) {
            return Err(Error::Bounds);
        }
        Ok(Share {
            value: ReplicatedShare::from_local(operation, party, first, second, limits)?,
            bits,
        })
    }

    /// Typed-frame import, still requiring a confidential authenticated channel.
    /// The outer raw buffer is separately owned; no wire codec is implied.
    pub fn message_from_received(
        self,
        header: Header,
        words: &[u64],
        limits: Limits,
    ) -> Result<Message, Error> {
        if header.phase != MessagePhase::BooleanAnd {
            return Err(Error::Binding);
        }
        if words.len() != self.lanes() {
            return Err(Error::Shape);
        }
        let mask = word_mask(self.bytes);
        if words.iter().any(|&v| v & !mask != 0) {
            return Err(Error::Bounds);
        }
        Message::from_received(header, words, self.operation, limits)
    }
}

pub(super) enum State {
    Planned(Plan),
    Rotating { plan: Plan, product: Words },
    Complete(Share),
}

impl State {
    pub(super) fn status(&self) -> Status {
        match self {
            Self::Planned(_) => Status::Planned,
            Self::Rotating { .. } => Status::AwaitingRotate,
            Self::Complete(_) => Status::Complete,
        }
    }
}

impl PartyRuntime {
    /// Use the same source setup, pair cursor, ordinal and payload ledger as
    /// multiplication/truncation. Widths select public storage, never secrets.
    pub fn plan_and(&mut self, lanes: usize, left_bits: u8, right_bits: u8) -> Result<Plan, Error> {
        let result = (|| {
            self.source.live()?;
            if !matches!(self.phase, Phase::Ready) {
                return Err(Error::Phase);
            }
            word_bytes(left_bits)?;
            word_bytes(right_bits)?;
            let bytes = word_bytes(left_bits.max(right_bits))?;
            let admitted_bytes = self.source.limits.core.admit(lanes)?;
            let usage = self.payload_usage();
            if usage
                .live_bytes()
                .checked_add(admitted_bytes - FIXED_RESERVE_BYTES)
                .is_none_or(|v| v > usage.limit_bytes())
            {
                return Err(Error::Bounds);
            }
            let q = typed_blocks(lanes, bytes)?;
            let pair_start = self.source.cursors.pair_blocks;
            pair_start.checked_add(q).ok_or(Error::Bounds)?;
            self.source.work(2 * q)?;
            let mut operation = Operation::new(
                self.context(),
                self.source.runtime.next_sequence(),
                Party::P0,
                lanes,
                1,
                self.source.limits.core,
            )?;
            let mut hash = Sha256::new();
            hash.update(PROFILE.as_bytes());
            hash.update(PRSS_PROFILE.as_bytes());
            hash.update(self.source.setup);
            hash.update(b"typed-pairwise-boolean-product");
            hash.update(pair_start.to_le_bytes());
            hash.update([left_bits, right_bits, bytes as u8]);
            operation.consumer = Some(hash.finalize().into());
            let plan = Plan {
                operation,
                left_bits,
                right_bits,
                bytes,
                pair_start,
                admitted_bytes,
            };
            self.phase = Phase::Boolean(State::Planned(plan));
            Ok(plan)
        })();
        self.checked(result)
    }

    pub fn begin_and(&mut self, left: Share, right: Share) -> Result<Message, Error> {
        let result = self.begin_and_inner(left, right);
        self.checked(result)
    }

    fn begin_and_inner(&mut self, mut left: Share, mut right: Share) -> Result<Message, Error> {
        self.source.live()?;
        let Phase::Boolean(State::Planned(plan)) = self.phase else {
            return Err(Error::Phase);
        };
        for (value, operand) in [(&left, Operand::Left), (&right, Operand::Right)] {
            let (op, bits) = plan.operand(operand);
            if value.value.operation != op || value.bits != bits {
                return Err(Error::Binding);
            }
            if value.party() != self.party() {
                return Err(Error::Role);
            }
        }
        if self.source.cursors.pair_blocks != plan.pair_start {
            return Err(Error::Binding);
        }
        let n = plan.lanes();
        let q = typed_blocks(n, plan.bytes)?;
        let end = plan.pair_start.checked_add(q).ok_or(Error::Bounds)?;
        let work = self.source.work(2 * q)?;
        self.source.runtime.consume_ordinal(plan.operation)?;
        self.source.cursors.pair_blocks = end;
        self.source.cursors.aes_work = work; // burn before expansion/allocation
        let usage = self.payload_usage();
        left.value.charge_to(&usage)?;
        right.value.charge_to(&usage)?;
        let seeds = self.source.seeds.as_ref().ok_or(Error::Retired)?;
        let next = seeds.next.as_ref().ok_or(Error::Phase)?;
        let mut product = usage.zeros(n)?;
        let mut adjacent = usage.zeros(n)?;
        fill_typed(&seeds.own, plan.pair_start, &mut product, plan.bytes)?;
        fill_typed(next, plan.pair_start, &mut adjacent, plan.bytes)?;
        for i in 0..n {
            product[i] ^= adjacent[i]
                ^ (left.first()[i] & right.first()[i])
                ^ (left.first()[i] & right.second()[i])
                ^ (left.second()[i] & right.first()[i]);
        }
        drop(adjacent);
        let message = Message {
            header: Header {
                binding: plan.digest(),
                from: self.party(),
                to: self.party().next().next(),
                phase: MessagePhase::BooleanAnd,
            },
            words: usage.copy(&product, n)?,
        };
        self.phase = Phase::Boolean(State::Rotating { plan, product });
        Ok(message)
    }

    pub fn receive_and(&mut self, mut message: Message) -> Result<(), Error> {
        let result = (|| {
            self.source.live()?;
            let Phase::Boolean(State::Rotating { plan, product }) =
                std::mem::replace(&mut self.phase, Phase::Retired)
            else {
                return Err(Error::Phase);
            };
            message.validate(
                plan.operation,
                self.party().next(),
                self.party(),
                MessagePhase::BooleanAnd,
            )?;
            let mask = word_mask(plan.bytes);
            if message.words().iter().any(|&v| v & !mask != 0) {
                return Err(Error::Bounds);
            }
            message.words.charge_to(&self.payload_usage())?;
            self.phase = Phase::Boolean(State::Complete(Share {
                value: ReplicatedShare {
                    operation: plan.operation,
                    party: self.party(),
                    first: product,
                    second: message.words,
                },
                bits: plan.output_bits(),
            }));
            Ok(())
        })();
        self.checked(result)
    }

    pub fn take_boolean_output(&mut self) -> Result<Share, Error> {
        let result = (|| {
            self.source.live()?;
            let Phase::Boolean(State::Complete(output)) =
                std::mem::replace(&mut self.phase, Phase::Retired)
            else {
                return Err(Error::Phase);
            };
            self.phase = Phase::Ready;
            Ok(output)
        })();
        self.checked(result)
    }
}

#[cfg(test)]
#[path = "boolean_tests.rs"]
mod tests;
