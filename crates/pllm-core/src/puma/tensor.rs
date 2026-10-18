//! Bounded public tensor programs over party-local replicated shares.
//!
//! All instructions execute for every lane. Comparisons return secret bits;
//! selections use exact ring products, never reconstructed branch decisions.
//! Public intervals must come from the caller/compiler, not private inputs.
//! This executor uses the existing PRSS owner and typed private-channel frames.

use super::{
    arithmetic::{rescaled_bounds, Affine},
    boolean,
    conversion::Kind,
    InputBounds, Interval, Operand, PartyRuntime, Status,
};
use crate::puma::payload::Reservation;
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "pllm.puma.public-tensor-program.q18.v1";
pub const MAX_NODES: usize = 64;
pub const MAX_TILE_LANES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Gelu,
    Silu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value(u8);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Instruction {
    Input,
    Affine {
        left: Value,
        right: Option<Value>,
        coefficients: Affine,
    },
    Product {
        left: Value,
        right: Value,
        rescale: bool,
    },
    Rescale {
        input: Value,
        shift: u8,
    },
    Sign(Value),
    FromBit(Value),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Node {
    instruction: Instruction,
    bounds: Interval,
    bit: bool,
}

/// Immutable public instructions. Construction proves every eager intermediate
/// range, including inactive polynomial branches, before an owner draws PRSS.
#[derive(Clone)]
pub struct Program {
    nodes: [Option<Node>; MAX_NODES],
    len: usize,
    lanes: usize,
    digest: [u8; 32],
}

pub struct Builder(Program);

impl Builder {
    pub fn new(lanes: usize, input: Interval) -> Result<Self, Error> {
        if !(1..=MAX_TILE_LANES).contains(&lanes) {
            return Err(Error::Bounds);
        }
        let mut program = Program {
            nodes: [None; MAX_NODES],
            len: 1,
            lanes,
            digest: [0; 32],
        };
        program.nodes[0] = Some(Node {
            instruction: Instruction::Input,
            bounds: input,
            bit: false,
        });
        Ok(Self(program))
    }

    pub fn input(&self) -> Value {
        Value(0)
    }

    fn node(&self, value: Value, bit: bool) -> Result<Node, Error> {
        self.0
            .nodes
            .get(value.0 as usize)
            .copied()
            .flatten()
            .filter(|n| n.bit == bit)
            .ok_or(Error::Binding)
    }

    fn push(
        &mut self,
        instruction: Instruction,
        bounds: Interval,
        bit: bool,
    ) -> Result<Value, Error> {
        if self.0.len == MAX_NODES {
            return Err(Error::Bounds);
        }
        let value = Value(self.0.len as u8);
        self.0.nodes[self.0.len] = Some(Node {
            instruction,
            bounds,
            bit,
        });
        self.0.len += 1;
        Ok(value)
    }

    pub fn affine(
        &mut self,
        left: Value,
        right: Option<Value>,
        coefficients: Affine,
    ) -> Result<Value, Error> {
        let bounds = InputBounds {
            left: self.node(left, false)?.bounds,
            right: match right {
                Some(v) => self.node(v, false)?.bounds,
                None => Interval::new(0, 0)?,
            },
        };
        if right.is_none() && coefficients.right != 0 {
            return Err(Error::Bounds);
        }
        self.push(
            Instruction::Affine {
                left,
                right,
                coefficients,
            },
            coefficients.output_bounds(bounds)?,
            false,
        )
    }

    pub fn product(&mut self, left: Value, right: Value, rescale: bool) -> Result<Value, Error> {
        let bounds = InputBounds {
            left: self.node(left, false)?.bounds,
            right: self.node(right, false)?.bounds,
        };
        let output = if rescale {
            bounds.output()?
        } else {
            let mut lo = i128::MAX;
            let mut hi = i128::MIN;
            for x in [bounds.left.lower(), bounds.left.upper()] {
                for y in [bounds.right.lower(), bounds.right.upper()] {
                    let v = i128::from(x) * i128::from(y);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            Interval::new(
                i64::try_from(lo).map_err(|_| Error::Bounds)?,
                i64::try_from(hi).map_err(|_| Error::Bounds)?,
            )?
        };
        self.push(
            Instruction::Product {
                left,
                right,
                rescale,
            },
            output,
            false,
        )
    }

    pub fn rescale(&mut self, input: Value, shift: u8) -> Result<Value, Error> {
        let bounds = rescaled_bounds(self.node(input, false)?.bounds, shift)?;
        self.push(Instruction::Rescale { input, shift }, bounds, false)
    }

    pub fn sign(&mut self, input: Value) -> Result<Value, Error> {
        self.node(input, false)?;
        self.push(Instruction::Sign(input), Interval::new(0, 1)?, true)
    }

    pub fn from_bit(&mut self, input: Value) -> Result<Value, Error> {
        self.node(input, true)?;
        self.push(Instruction::FromBit(input), Interval::new(0, 1)?, false)
    }

    pub fn finish(mut self, output: Value) -> Result<Program, Error> {
        self.node(output, false)?;
        if output.0 == 0 || output.0 as usize + 1 != self.0.len {
            return Err(Error::Binding);
        }
        let mut h = Sha256::new();
        h.update(PROFILE.as_bytes());
        h.update(super::NUMERIC_PROFILE.as_bytes());
        h.update((self.0.lanes as u64).to_le_bytes());
        h.update((self.0.len as u64).to_le_bytes());
        for node in self.0.nodes[..self.0.len].iter().flatten() {
            match node.instruction {
                Instruction::Input => h.update([0]),
                Instruction::Affine {
                    left,
                    right,
                    coefficients,
                } => {
                    h.update([1, left.0, right.map_or(u8::MAX, |v| v.0)]);
                    for c in [coefficients.left, coefficients.right, coefficients.constant] {
                        h.update(c.to_le_bytes());
                    }
                }
                Instruction::Product {
                    left,
                    right,
                    rescale,
                } => h.update([2, left.0, right.0, u8::from(rescale)]),
                Instruction::Rescale { input, shift } => h.update([3, input.0, shift]),
                Instruction::Sign(input) => h.update([4, input.0]),
                Instruction::FromBit(input) => h.update([5, input.0]),
            }
            h.update(node.bounds.lower().to_le_bytes());
            h.update(node.bounds.upper().to_le_bytes());
        }
        self.0.digest = h.finalize().into();
        Ok(self.0)
    }
}

impl Program {
    /// The approved source-derived profile, not the unavailable evaluated IR.
    /// Coefficients/pivots are the binary32/toward-zero Q18 constants frozen in
    /// paper/reproduction-contracts/shared-profile-resolution.md, section 3.3.
    pub fn activation(kind: Activation, lanes: usize, input: Interval) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let x = b.input();
        let x2 = b.product(x, x, true)?;
        let x3 = if kind == Activation::Gelu {
            Some(b.product(x2, x, true)?)
        } else {
            None
        };
        let x4 = b.product(x2, x2, true)?;
        let x6 = match x3 {
            Some(v) => b.product(v, v, true)?,
            None => b.product(x2, x4, true)?,
        };
        let (low, high, pivots) = match kind {
            Activation::Gelu => (
                polynomial(
                    &mut b,
                    x,
                    -132488,
                    &[(x, -110694), (x2, -30952), (x3.unwrap(), -2892)],
                )?,
                polynomial(
                    &mut b,
                    x,
                    2235,
                    &[(x, 131072), (x2, 94458), (x4, -9879), (x6, 473)],
                )?,
                [-1048576, -511180, 786432],
            ),
            Activation::Silu => (
                polynomial(&mut b, x, -80413, &[(x, -21489), (x2, -1453)])?,
                polynomial(
                    &mut b,
                    x,
                    2229,
                    &[(x, 131072), (x2, 59806), (x4, -2913), (x6, 71)],
                )?,
                [-2097152, -1048576, 1048576],
            ),
        };
        let mut predicates = [x; 3];
        for (i, pivot) in pivots.into_iter().enumerate() {
            // Last polynomial includes its upper endpoint; strict MSB tests
            // x < upper+1. All decisions stay secret-shared through B2A.
            let delta = b.affine(
                x,
                None,
                Affine {
                    left: 1,
                    right: 0,
                    constant: -pivot - i64::from(i == 2),
                },
            )?;
            let bit = b.sign(delta)?;
            predicates[i] = b.from_bit(bit)?;
        }
        let upper = select(&mut b, predicates[2], high, x)?;
        let middle = select(&mut b, predicates[1], low, upper)?;
        let removed = b.product(predicates[0], middle, false)?;
        let output = b.affine(
            middle,
            Some(removed),
            Affine {
                left: 1,
                right: -1,
                constant: 0,
            },
        )?;
        b.finish(output)
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn instructions(&self) -> usize {
        self.len - 1
    }
    pub fn output_bounds(&self) -> Interval {
        self.nodes[self.len - 1].unwrap().bounds
    }

    /// Conservative retained-register plus primitive/copy/frame allowance.
    /// Original exported buffers retain their own ledger charges as usual.
    pub fn admitted_bytes(&self) -> usize {
        FIXED_RESERVE_BYTES
            + metadata_bytes()
            + 3 * std::mem::size_of::<Frame>()
            + (2 * MAX_NODES + 64) * self.lanes * 8
    }
}

fn polynomial(
    b: &mut Builder,
    input: Value,
    constant: i64,
    terms: &[(Value, i64)],
) -> Result<Value, Error> {
    let mut sum = b.affine(
        input,
        None,
        Affine {
            left: 0,
            right: 0,
            constant,
        },
    )?;
    for &(power, coefficient) in terms {
        let raw = b.affine(
            power,
            None,
            Affine {
                left: coefficient,
                right: 0,
                constant: 0,
            },
        )?;
        let term = b.rescale(raw, 18)?;
        sum = b.affine(
            sum,
            Some(term),
            Affine {
                left: 1,
                right: 1,
                constant: 0,
            },
        )?;
    }
    Ok(sum)
}

fn select(b: &mut Builder, bit: Value, yes: Value, no: Value) -> Result<Value, Error> {
    let difference = b.affine(
        yes,
        Some(no),
        Affine {
            left: 1,
            right: -1,
            constant: 0,
        },
    )?;
    let chosen = b.product(bit, difference, false)?;
    b.affine(
        no,
        Some(chosen),
        Affine {
            left: 1,
            right: 1,
            constant: 0,
        },
    )
}

/// A typed frame exposes routing and public schedule position, never payload.
/// Outer transport must authenticate each private channel. No codec is implied.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::Frame;
/// fn duplicate(frame: Frame) { let _ = frame.clone(); }
/// ```
pub struct Frame {
    program: [u8; 32],
    invocation: u64,
    step: usize,
    inner: Message,
    // Last field: payload erasure/deallocation precedes metadata refund.
    _metadata: Reservation,
}

impl Frame {
    pub fn sender(&self) -> Party {
        self.inner.header().from
    }
    pub fn recipient(&self) -> Party {
        self.inner.header().to
    }
    pub fn payload_bytes(&self) -> usize {
        self.inner.words().len() * 8
    }
}

enum Register {
    Arithmetic(ReplicatedShare),
    Bit(boolean::Share),
}

/// One party's fixed public schedule and opaque live registers. No all-party
/// coordinator or reconstructed value exists in this production owner.
///
/// ```compile_fail
/// use pllm_core::puma::multiply::tensor::Executor;
/// fn duplicate(owner: Executor) { let _ = owner.clone(); }
/// ```
pub struct Executor {
    runtime: Option<PartyRuntime>,
    program: Program,
    registers: [Option<Register>; MAX_NODES],
    inbox: [Option<Message>; 3],
    invocation: u64,
    step: usize,
    active: bool,
    retired: bool,
    metadata: Option<Reservation>,
}

fn metadata_bytes() -> usize {
    // The existing runtime retains its own 4096-byte fixed reserve. Charge this
    // executor's inline program/register/frame metadata as an additional owner.
    std::mem::size_of::<Executor>()
}

impl Executor {
    pub fn new(
        mut runtime: PartyRuntime,
        program: Program,
        mut input: ReplicatedShare,
    ) -> Result<Self, Error> {
        let result = (|| {
            if runtime.status() != Status::Ready {
                return Err(Error::Phase);
            }
            if input.operation.context != runtime.context() {
                return Err(Error::Binding);
            }
            if input.party != runtime.party() {
                return Err(Error::Role);
            }
            if input.operation.lanes != program.lanes {
                return Err(Error::Shape);
            }
            runtime.source.limits.core.admit(2 * program.lanes)?;
            let usage = runtime.payload_usage();
            if usage
                .live_bytes()
                .checked_add(program.admitted_bytes() - FIXED_RESERVE_BYTES)
                .is_none_or(|v| v > usage.limit_bytes())
            {
                return Err(Error::Bounds);
            }
            let invocation = runtime.source.runtime.next_sequence();
            if invocation
                .checked_add(program.instructions() as u64)
                .is_none_or(|v| v > runtime.source.limits.core.max_operations)
            {
                return Err(Error::Bounds);
            }
            // Conservative aggregate work, including every carry circuit and
            // helper split, before any instruction consumes randomness.
            let work = program.instructions() as u64 * (64 * program.lanes as u64 + 4);
            runtime.source.work(work)?;
            for cursor in [
                runtime.cursors().pair_blocks,
                runtime.cursors().public_blocks,
                runtime.cursors().private_blocks,
            ] {
                cursor.checked_add(work).ok_or(Error::Bounds)?;
            }
            let metadata = usage.reserve(metadata_bytes())?;
            input.charge_to(&usage)?;
            Ok((invocation, metadata))
        })();
        let (invocation, metadata) = match result {
            Ok(v) => v,
            Err(e) => {
                runtime.cancel();
                return Err(e);
            }
        };
        let mut registers = std::array::from_fn(|_| None);
        registers[0] = Some(Register::Arithmetic(input));
        Ok(Self {
            runtime: Some(runtime),
            program,
            registers,
            inbox: [None, None, None],
            invocation,
            step: 1,
            active: false,
            retired: false,
            metadata: Some(metadata),
        })
    }

    pub fn party(&self) -> Party {
        self.runtime.as_ref().unwrap().party()
    }
    pub fn cursors(&self) -> super::super::Cursors {
        self.runtime.as_ref().unwrap().cursors()
    }
    pub fn payload_usage(&self) -> PayloadUsage {
        self.runtime.as_ref().unwrap().payload_usage()
    }
    pub fn complete(&self) -> bool {
        !self.retired && !self.active && self.step == self.program.len
    }
    pub fn retired(&self) -> bool {
        self.retired
    }

    pub fn cancel(&mut self) {
        self.retired = true;
        if let Some(runtime) = &mut self.runtime {
            runtime.cancel();
        }
        self.registers = std::array::from_fn(|_| None);
        self.inbox = [None, None, None];
        // Metadata remains charged until this owner actually retires.
    }

    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }

    fn copy(&self, value: Value) -> Result<ReplicatedShare, Error> {
        let Some(Register::Arithmetic(value)) = &self.registers[value.0 as usize] else {
            return Err(Error::Binding);
        };
        let usage = self.payload_usage();
        Ok(ReplicatedShare {
            operation: value.operation,
            party: value.party,
            first: usage.copy(&value.first, self.program.lanes)?,
            second: usage.copy(&value.second, self.program.lanes)?,
        })
    }

    fn bounds(&self, value: Value) -> Interval {
        self.program.nodes[value.0 as usize].unwrap().bounds
    }

    fn wrap(&self, messages: [Option<Message>; 3]) -> Result<[Option<Frame>; 3], Error> {
        let mut out = [None, None, None];
        for (slot, message) in out.iter_mut().zip(messages) {
            if let Some(inner) = message {
                *slot = Some(Frame {
                    program: self.program.digest,
                    invocation: self.invocation,
                    step: self.step,
                    inner,
                    _metadata: self.payload_usage().reserve(std::mem::size_of::<Frame>())?,
                });
            }
        }
        Ok(out)
    }

    /// Start the next public instruction. The router must drain its frames and
    /// call finish_step on every party before beginning the next instruction.
    /// Frames emitted by one protocol round are delivered only after all
    /// parties have reached that round (rotation/open/reshare barriers).
    pub fn start_step(&mut self) -> Result<[Option<Frame>; 3], Error> {
        let result = self.start_inner();
        self.checked(result)
    }

    fn start_inner(&mut self) -> Result<[Option<Frame>; 3], Error> {
        if self.retired {
            return Err(Error::Retired);
        }
        if self.active || self.complete() {
            return Err(Error::Phase);
        }
        self.active = true;
        let mut messages = [None, None, None];
        match self.program.nodes[self.step].unwrap().instruction {
            Instruction::Input => return Err(Error::Phase),
            Instruction::Affine {
                left,
                right,
                coefficients,
            } => {
                let mut a = self.copy(left)?;
                let mut b = right.map(|v| self.copy(v)).transpose()?;
                let bounds = InputBounds {
                    left: self.bounds(left),
                    right: right.map_or(Interval::new(0, 0)?, |v| self.bounds(v)),
                };
                let (value, _) = self.runtime.as_mut().unwrap().affine(
                    &mut a,
                    b.as_mut(),
                    coefficients,
                    bounds,
                )?;
                self.registers[self.step] = Some(Register::Arithmetic(value));
            }
            Instruction::Product {
                left,
                right,
                rescale,
            } => {
                let a = self.copy(left)?;
                let b = self.copy(right)?;
                let bounds = InputBounds {
                    left: self.bounds(left),
                    right: self.bounds(right),
                };
                let all = [bounds; MAX_TILE_LANES];
                let runtime = self.runtime.as_mut().unwrap();
                if rescale {
                    runtime.plan(&all[..self.program.lanes])?;
                } else {
                    runtime.plan_ring_product(&all[..self.program.lanes])?;
                }
                let a = runtime.bind_operand(Operand::Left, a)?;
                let b = runtime.bind_operand(Operand::Right, b)?;
                messages[0] = Some(runtime.begin(a, b)?);
            }
            Instruction::Rescale { input, shift } => {
                let value = self.copy(input)?;
                let bounds = self.bounds(input);
                let (_, [a, b]) = self
                    .runtime
                    .as_mut()
                    .unwrap()
                    .begin_rescale(value, shift, bounds)?;
                messages = [a, b, None];
            }
            Instruction::Sign(input) => {
                let value = self.copy(input)?;
                let runtime = self.runtime.as_mut().unwrap();
                runtime.plan_conversion(self.program.lanes, Kind::MostSignificantBit)?;
                let value = runtime.bind_conversion_input(value)?;
                messages[0] = Some(runtime.begin_conversion(value)?);
            }
            Instruction::FromBit(input) => {
                let Some(Register::Bit(value)) = &self.registers[input.0 as usize] else {
                    return Err(Error::Binding);
                };
                let usage = self.payload_usage();
                let value = boolean::Share {
                    value: ReplicatedShare {
                        operation: value.value.operation,
                        party: value.value.party,
                        first: usage.copy(&value.value.first, self.program.lanes)?,
                        second: usage.copy(&value.value.second, self.program.lanes)?,
                    },
                    bits: value.bits(),
                };
                let runtime = self.runtime.as_mut().unwrap();
                runtime.plan_boolean_to_arithmetic(&value)?;
                let [a, b] = runtime.begin_boolean_to_arithmetic(value)?;
                messages = [a, b, None];
            }
        }
        self.wrap(messages)
    }

    pub fn receive(&mut self, frame: Frame) -> Result<[Option<Frame>; 3], Error> {
        let result = self.receive_inner(frame);
        self.checked(result)
    }

    fn receive_inner(&mut self, frame: Frame) -> Result<[Option<Frame>; 3], Error> {
        if self.retired {
            return Err(Error::Retired);
        }
        if !self.active {
            return Err(Error::Phase);
        }
        if frame.program != self.program.digest
            || frame.invocation != self.invocation
            || frame.step != self.step
        {
            return Err(Error::Binding);
        }
        if frame.recipient() != self.party() {
            return Err(Error::Role);
        }
        let runtime = self.runtime.as_mut().unwrap();
        let instruction = self.program.nodes[self.step].unwrap().instruction;
        let mut out = [None, None, None];
        match instruction {
            Instruction::Sign(_) => out[0] = runtime.advance_conversion(frame.inner)?,
            Instruction::FromBit(_) => {
                let slot = match frame.inner.header().phase {
                    MessagePhase::BooleanOtMask | MessagePhase::BooleanOtReshare => 0,
                    MessagePhase::BooleanOtZero => 1,
                    MessagePhase::BooleanOtOne => 2,
                    _ => return Err(Error::Binding),
                };
                if self.inbox[slot].is_some() {
                    return Err(Error::Binding);
                }
                let reshare = frame.inner.header().phase == MessagePhase::BooleanOtReshare;
                let mut message = frame.inner;
                message.words.charge_to(&runtime.payload_usage())?;
                self.inbox[slot] = Some(message);
                if reshare || self.inbox.iter().all(Option::is_some) {
                    out[0] =
                        runtime.receive_boolean_to_arithmetic(std::mem::take(&mut self.inbox))?;
                }
            }
            Instruction::Product { .. } | Instruction::Rescale { .. } => match runtime.status() {
                Status::AwaitingRotate => {
                    let [a, b] = runtime.rotate(frame.inner)?;
                    out = [a, b, None];
                }
                Status::AwaitingOpen => {
                    let slot = match frame.inner.header().phase {
                        MessagePhase::MaskedInput => 0,
                        MessagePhase::Correlation => 1,
                        _ => return Err(Error::Binding),
                    };
                    if self.inbox[slot].is_some() {
                        return Err(Error::Binding);
                    }
                    let mut message = frame.inner;
                    message.words.charge_to(&runtime.payload_usage())?;
                    self.inbox[slot] = Some(message);
                    if self.inbox[0].is_some() && self.inbox[1].is_some() {
                        out[0] =
                            Some(runtime.open(
                                self.inbox[0].take().unwrap(),
                                self.inbox[1].take().unwrap(),
                            )?);
                    }
                }
                Status::AwaitingReshare => runtime.reshare(frame.inner)?,
                _ => return Err(Error::Phase),
            },
            _ => return Err(Error::Phase),
        }
        self.wrap(out)
    }

    pub fn finish_step(&mut self) -> Result<(), Error> {
        let result = (|| {
            if self.retired {
                return Err(Error::Retired);
            }
            if !self.active || self.inbox.iter().any(Option::is_some) {
                return Err(Error::Phase);
            }
            let runtime = self.runtime.as_mut().unwrap();
            if self.registers[self.step].is_none() {
                self.registers[self.step] = Some(if self.program.nodes[self.step].unwrap().bit {
                    Register::Bit(runtime.take_boolean_output()?)
                } else {
                    Register::Arithmetic(runtime.take_output()?)
                });
            }
            // Bind terminal and intermediate owners to the complete public graph.
            let operation = match self.registers[self.step].as_mut().unwrap() {
                Register::Arithmetic(v) => &mut v.operation,
                Register::Bit(v) => &mut v.value.operation,
            };
            let mut h = Sha256::new();
            h.update(operation.digest());
            h.update(self.program.digest);
            h.update(self.invocation.to_le_bytes());
            h.update((self.step as u64).to_le_bytes());
            operation.consumer = Some(h.finalize().into());
            self.step += 1;
            self.active = false;
            Ok(())
        })();
        self.checked(result)
    }

    /// Return the original runtime for subsequent tensors with unchanged PRSS
    /// cursors. The result keeps its payload lease after this executor retires.
    pub fn finish(mut self) -> Result<(PartyRuntime, ReplicatedShare), Error> {
        if !self.complete() {
            return Err(Error::Phase);
        }
        let Some(Register::Arithmetic(value)) = self.registers[self.program.len - 1].take() else {
            return Err(Error::Phase);
        };
        self.registers = std::array::from_fn(|_| None);
        Ok((self.runtime.take().unwrap(), value))
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.cancel();
        drop(self.metadata.take());
    }
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
