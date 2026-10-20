//! One-role consuming execution and explicit trusted-dealer material feed.
use super::*;
use crate::fusefss_backend::{Frame, Output};
use std::{ops::Deref, sync::Arc};
use zeroize::Zeroizing;

#[path = "wire.rs"]
pub mod wire;

enum Primitive<'a> {
    Conversion(conversions::Party<'a>),
    Predicate(predicates::Party<'a>),
    Lookup(lookup::Party<'a>),
    Gelu(super::super::Party<'a>, bool),
}

/// Role-separated envelope. No public constructor, cloning, or key access.
pub struct Material<'a> {
    plan: [u8; 32],
    operation: [u8; 32],
    pc: usize,
    role: u8,
    primitive: Primitive<'a>,
}

/// Frame ownership and its native budget lease pass directly to caller. Drop
/// each exchanged frame after use to preserve the admitted streaming overlap.
pub enum Event<'a> {
    NeedMaterial { pc: usize },
    Frame(Frame<'a>),
    Output(Output<'a>),
}

struct Execution<'a> {
    values: Zeroizing<Vec<u64>>,
    pc: usize,
    active: Option<Primitive<'a>>,
}

/// Exactly one party's additive/XOR registers. No dealer, sibling evaluator,
/// reconstruction API, checkpoint load, or implicit preprocessing cache.
pub struct Engine<'a, P: Deref<Target = Plan<'a>>> {
    plan: P,
    context: Context,
    role: u8,
    state: Option<Execution<'a>>,
    lease: Lease<'a>,
}

pub type Runtime<'p, 'a> = Engine<'a, &'p Plan<'a>>;
pub type OwnedRuntime<'a> = Engine<'a, Arc<Plan<'a>>>;

pub(super) fn owner_bytes(registers: usize, outputs: usize) -> usize {
    size_of::<Runtime<'_, '_>>() + size_of::<Execution<'_>>() + 4096 + 8 * (registers + outputs)
}

impl<'a> Plan<'a> {
    /// Immutable public issuance order. Local instructions consume no dealer
    /// material; callers must not infer that distinction from execution errors.
    pub fn material_steps(&self) -> impl Iterator<Item = usize> + '_ {
        self.code
            .iter()
            .enumerate()
            .filter_map(|(pc, instruction)| {
                matches!(
                    instruction,
                    Instruction::Convert(..)
                        | Instruction::Multiply(..)
                        | Instruction::Select(..)
                        | Instruction::Predicate(..)
                        | Instruction::Lookup(..)
                        | Instruction::Gelu(..)
                )
                .then_some(pc)
            })
    }

    /// Move the immutable plan into a shared owner. The Plan reservation already
    /// includes its inline metadata and 4096 bytes of owner/alignment slack,
    /// covering Arc's two counters without changing legacy resource identities.
    /// Use this constructor rather than an unmetered external Arc allocation.
    pub fn into_shared(self) -> Result<Arc<Self>, Error> {
        self.lease.budget.allocation()?;
        Ok(Arc::new(self))
    }

    /// Own a shared immutable plan; no self-reference or borrowed outer owner.
    pub fn runtime_owned(
        self: &Arc<Self>,
        role: u8,
        context: Context,
        inputs: &[u64],
    ) -> Result<OwnedRuntime<'a>, Error> {
        Engine::new(Arc::clone(self), role, context, inputs)
    }
    /// Trusted dealer ONLY. Call externally, separately for each requested public
    /// step. Both returned owners use fresh native primitive issuance. Runtime
    /// has no path to this helper. No automatic transport or persistence implied.
    pub fn issue_step(&self, pc: usize, context: Context) -> Result<[Material<'a>; 2], Error> {
        let instruction = *self.code.get(pc).ok_or(Error::Domain)?;
        let context = self.step_context(context, pc);
        let budget = self.lease.budget;
        let pair = match instruction {
            Instruction::Convert(..) | Instruction::Multiply(..) | Instruction::Select(..) => {
                conversions::issue(self.conversion_spec(pc)?, context, budget)?
                    .map(Primitive::Conversion)
            }
            Instruction::Predicate(value, signed, threshold) => {
                budget.allocation()?;
                let descriptor =
                    Descriptor::simple_comparison(self.registers[value].bits, signed, threshold)
                        .map_err(descriptor_error)?;
                predicates::issue(&descriptor, self.source, context, budget)?
                    .map(Primitive::Predicate)
            }
            Instruction::Lookup(_, table) => {
                self.tables[table].issue(1, context)?.map(Primitive::Lookup)
            }
            Instruction::Gelu(_) => {
                let gate = super::super::CompiledGelu::with_backend(
                    budget,
                    super::super::LookupBackend::PublicBoundary,
                )?;
                gate.issue(1, context)?
                    .map(|party| Primitive::Gelu(party, false))
            }
            _ => return Err(Error::Domain),
        };
        let [a, b] = pair;
        let make = |role, primitive| Material {
            plan: self.digest,
            operation: context.operation,
            pc,
            role,
            primitive,
        };
        Ok([make(0, a), make(1, b)])
    }

    /// Construct only one role. Inputs are caller's canonical native ring shares;
    /// semantic range assertions remain fixed public lowering contracts.
    pub fn runtime<'p>(
        &'p self,
        role: u8,
        context: Context,
        inputs: &[u64],
    ) -> Result<Runtime<'p, 'a>, Error> {
        Engine::new(self, role, context, inputs)
    }
}

impl<'a, P: Deref<Target = Plan<'a>>> Engine<'a, P> {
    fn new(plan: P, role: u8, context: Context, inputs: &[u64]) -> Result<Self, Error> {
        if role > 1
            || inputs.len() != plan.inputs.len()
            || inputs
                .iter()
                .zip(&plan.inputs)
                .any(|(&v, &bits)| v > mask(bits))
        {
            return Err(Error::Domain);
        }
        context
            .first_tensor_index
            .checked_add(1)
            .ok_or(Error::Domain)?;
        let lease = plan
            .lease
            .budget
            .reserve(plan.resources.register_owner_bytes_per_party)?;
        let mut values = Zeroizing::new(allocate(plan.registers.len(), plan.lease.budget)?);
        values.extend_from_slice(inputs);
        values.resize(plan.registers.len(), 0);
        Ok(Self {
            plan,
            context,
            role,
            state: Some(Execution {
                values,
                pc: 0,
                active: None,
            }),
            lease,
        })
    }
}

impl<'a, P: Deref<Target = Plan<'a>>> Engine<'a, P> {
    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }

    /// Run public local instructions until next preprocessing request or export.
    /// Calling while waiting on peer burns active primitive and all registers.
    pub fn poll(&mut self) -> Result<Event<'a>, Error> {
        let execution = self.state.take().ok_or(Error::Consumed)?;
        if execution.active.is_some() {
            return Err(Error::Context);
        }
        self.local(execution)
    }

    fn local(&mut self, mut execution: Execution<'a>) -> Result<Event<'a>, Error> {
        while execution.pc < self.plan.code.len() {
            let pc = execution.pc;
            let dest = self.plan.inputs.len() + pc;
            let m = mask(self.plan.registers[dest].bits);
            let v = &execution.values;
            let result = Zeroizing::new(match self.plan.code[pc] {
                Instruction::Constant(value) => {
                    if self.role == 0 {
                        value
                    } else {
                        0
                    }
                }
                Instruction::Add(a, b) => v[a].wrapping_add(v[b]),
                Instruction::Sub(a, b) => v[a].wrapping_sub(v[b]),
                Instruction::PublicMultiply(a, coefficient) => v[a].wrapping_mul(coefficient),
                Instruction::Narrow(a) => v[a],
                _ => {
                    self.state = Some(execution);
                    return Ok(Event::NeedMaterial { pc });
                }
            });
            execution.values[dest] = *result & m;
            execution.pc += 1;
        }
        let mut values = Zeroizing::new(allocate(self.plan.outputs.len(), self.lease.budget)?);
        for &index in &self.plan.outputs {
            values.push(execution.values[index]);
        }
        // Native Output declaration erases values before its split lease refunds.
        Ok(Event::Output(Output {
            values,
            _lease: self.lease.split(8 * self.plan.outputs.len())?,
        }))
    }

    /// Consume exactly expected role/context/plan/pc material. Wrong feed, wrong
    /// order, malformed inputs and allocation failures burn all active state.
    pub fn feed(&mut self, material: Material<'a>) -> Result<Event<'a>, Error> {
        let mut execution = self.state.take().ok_or(Error::Consumed)?;
        if execution.active.is_some()
            || material.role != self.role
            || material.plan != self.plan.digest
            || material.pc != execution.pc
            || material.operation != self.plan.step_context(self.context, execution.pc).operation
        {
            return Err(Error::Context);
        }
        let instruction = *self.plan.code.get(execution.pc).ok_or(Error::Context)?;
        let mut primitive = material.primitive;
        let v = &execution.values;
        let operands = Zeroizing::new(match instruction {
            Instruction::Convert(a, _)
            | Instruction::Predicate(a, _, _)
            | Instruction::Lookup(a, _)
            | Instruction::Gelu(a) => [v[a], 0, 0],
            Instruction::Multiply(a, b) => [v[a], v[b], 0],
            Instruction::Select(bit, yes, no) => [v[bit], v[yes], v[no]],
            _ => return Err(Error::Context),
        });
        let frame = match (&mut primitive, instruction) {
            (Primitive::Conversion(p), Instruction::Convert(_, _)) => p.start(&operands[..1])?,
            (Primitive::Conversion(p), Instruction::Multiply(_, _)) => {
                p.start_product(&operands[..1], &operands[1..2])?
            }
            (Primitive::Conversion(p), Instruction::Select(_, _, _)) => {
                p.start_phi(&operands[..1], &operands[1..2], &operands[2..])?
            }
            (Primitive::Predicate(p), Instruction::Predicate(_, _, _)) => p.start(operands[0])?,
            (Primitive::Lookup(p), Instruction::Lookup(_, _)) => p.start(&operands[..1])?,
            (Primitive::Gelu(p, _), Instruction::Gelu(_)) => p.start(&operands[..1])?,
            _ => return Err(Error::Context),
        };
        execution.active = Some(primitive);
        self.state = Some(execution);
        Ok(Event::Frame(frame))
    }

    /// Feed transport/dealer exhaustion or failure explicitly. An error burns
    /// registers and any active primitive; remote dealer queues remain external.
    pub fn feed_result(
        &mut self,
        material: Result<Material<'a>, Error>,
    ) -> Result<Event<'a>, Error> {
        match material {
            Ok(material) => self.feed(material),
            Err(error) => {
                let _execution = self.state.take().ok_or(Error::Consumed)?;
                Err(error)
            }
        }
    }

    /// Consume one peer primitive frame. All inner frames bind the derived
    /// plan/context/pc and native primitive role/nonce/round. No outer plaintext
    /// openings are introduced. Replayed or wrong-step frames burn this owner.
    pub fn advance(&mut self, peer: &[u8]) -> Result<Event<'a>, Error> {
        let mut execution = self.state.take().ok_or(Error::Consumed)?;
        let mut primitive = execution.active.take().ok_or(Error::Context)?;
        let value = Zeroizing::new(match &mut primitive {
            Primitive::Conversion(p) => {
                let output = p.finish(peer)?;
                *output.shares().first().ok_or(Error::Domain)?
            }
            Primitive::Lookup(p) => {
                let output = p.finish(peer)?;
                *output.shares().first().ok_or(Error::Domain)?
            }
            Primitive::Predicate(p) => match p.advance(peer)? {
                predicates::Step::Frame(frame) => {
                    execution.active = Some(primitive);
                    self.state = Some(execution);
                    return Ok(Event::Frame(frame));
                }
                predicates::Step::Output(output) => {
                    *output.shares().first().ok_or(Error::Domain)?
                }
            },
            Primitive::Gelu(p, advanced) => {
                if !*advanced {
                    let frame = p.advance(peer)?;
                    *advanced = true;
                    execution.active = Some(primitive);
                    self.state = Some(execution);
                    return Ok(Event::Frame(frame));
                }
                let output = p.finish(peer)?;
                *output.shares().first().ok_or(Error::Domain)?
            }
        });
        drop(primitive);
        let dest = self.plan.inputs.len() + execution.pc;
        execution.values[dest] = *value;
        execution.pc += 1;
        self.local(execution)
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
