//! Public dependency scheduling of the existing consuming primitives. Numeric
//! instructions, per-step context and fresh material are unchanged. Only public
//! ready sets determine grouping; no secret value controls a round or a lane.
use super::{
    runtime::{self, Primitive, PrimitiveEvent},
    *,
};
use crate::fusefss_backend::{Frame, Output};
use zeroize::Zeroizing;

pub const MAX_ACTIVE: usize = 256;
pub const MAX_TENSOR_ACTIVE: usize = 4096;
pub const MAX_FRAME_BYTES: usize = 1 << 20;
const MAX_RETAINED_PARTY_BYTES: usize = 2 << 20;
const HEADER: usize = 20;
const RECORD: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
pub struct Resources {
    pub rounds: usize,
    pub primitive_frames: usize,
    pub frame_bytes_per_party: usize,
    pub largest_frame_bytes: usize,
    pub metadata_bytes: usize,
    /// Bounded public depth construction, ordering and schedule hashing work.
    pub construction_work: usize,
    pub runtime_owner_bytes: usize,
    pub retained_party_bytes: usize,
    /// All three public schedules plus both runtime/primitive/frame frontiers.
    /// Added conservatively to legacy streaming admission, without discounts.
    pub additional_reservation_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Step {
    pc: usize,
    depth: usize,
    rounds: usize,
    frame_bytes: usize,
    party_bytes: usize,
}

// One public (depth, local/secret) span of the template. Within that span the
// virtual order is lane-major, exactly matching expanded (depth, secret, PC)
// sorting. Private registers, primitives and one-use PCs are never shared.
struct Span {
    start: usize,
    end: usize,
    logical_end: usize,
}

pub struct Schedule<'p, 'a> {
    plan: &'p Plan<'a>,
    steps: Vec<Step>,
    spans: Vec<Span>,
    resources: Resources,
    digest: [u8; 32],
    active_capacity: usize,
    _lease: Lease<'a>,
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Resource)
}
fn mul(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_mul(b).ok_or(Error::Resource)
}

fn inputs(instruction: Instruction) -> [Option<Value>; 3] {
    use Instruction::*;
    match instruction {
        Constant(_) => [None; 3],
        Add(a, b) | Sub(a, b) | Multiply(a, b) => [Some(a), Some(b), None],
        PublicMultiply(a, _)
        | Narrow(a)
        | Convert(a, _)
        | Predicate(a, _, _)
        | Lookup(a, _)
        | Gelu(a) => [Some(a), None, None],
        Select(a, b, c) => [Some(a), Some(b), Some(c)],
        CompactTokenOneHot(_) => [Some(0), None, None],
    }
}

fn online_cost(plan: &Plan<'_>, pc: usize) -> Result<(usize, usize), Error> {
    Ok(match plan.instruction(pc)? {
        Instruction::Constant(_)
        | Instruction::Add(..)
        | Instruction::Sub(..)
        | Instruction::PublicMultiply(..)
        | Instruction::Narrow(_) => (0, 0),
        Instruction::Convert(..) | Instruction::Multiply(..) | Instruction::Select(..) => {
            let r = conversions::resources(plan.conversion_spec(pc)?)?;
            (1, r.frame_bytes_per_party)
        }
        Instruction::Predicate(v, signed, threshold) => {
            let _scratch = plan.lease.budget.reserve(DESCRIPTOR_SCRATCH)?;
            plan.lease.budget.allocation()?;
            let d = Descriptor::simple_comparison(plan.register(v)?.bits, signed, threshold)
                .map_err(descriptor_error)?;
            let r = predicates::resources(&d)?;
            (r.frames_per_party, r.total_peer_frame_bytes / 2)
        }
        Instruction::Lookup(_, table) => (
            1,
            plan.tables[table].resources(1)?.peer_frame_bytes_per_party,
        ),
        Instruction::Gelu(_) => {
            let gate = super::super::CompiledGelu::with_backend(
                plan.lease.budget,
                super::super::LookupBackend::PublicBoundary,
            )?;
            (2, gate.resources(1)?.issuance.total_peer_frame_bytes / 2)
        }
        Instruction::CompactTokenOneHot(_) => return Err(Error::Unsupported),
    })
}

impl<'p, 'a> Schedule<'p, 'a> {
    pub fn new(plan: &'p Plan<'a>) -> Result<Self, Error> {
        if plan.is_atomic_vector() {
            return Err(Error::Unsupported);
        }
        let budget = plan.lease.budget;
        let instructions = plan.code.len();
        let registers = plan.registers.len();
        let span_capacity = if plan.is_tensor() { instructions } else { 0 };
        let metadata = add(
            add(size_of::<Self>(), 4096)?,
            add(
                mul(instructions, size_of::<Step>())?,
                mul(span_capacity, size_of::<Span>())?,
            )?,
        )?;
        let lease = budget.reserve(metadata)?;
        let _scratch = budget.reserve(mul(registers, size_of::<usize>())?)?;
        let mut depths = allocate(registers, budget)?;
        depths.resize(registers, 0usize);
        let mut steps = allocate(instructions, budget)?;
        let mut spans = allocate(span_capacity, budget)?;
        let template_inputs = if plan.is_tensor() {
            plan.inputs.len() / plan.lanes
        } else {
            plan.inputs.len()
        };
        for pc in 0..instructions {
            let instruction = plan.code[pc];
            let dest = template_inputs + pc;
            let mut depth = 0;
            for input in inputs(instruction).into_iter().flatten() {
                if input >= dest {
                    return Err(Error::Context);
                }
                depth = depth.max(depths[input]);
            }
            let (rounds, frame_bytes) = online_cost(plan, pc)?;
            let party_bytes = if rounds == 0 {
                0
            } else {
                plan.step_transport_resources(pc)?
                    .decoded_party_reservation_bytes
            };
            depths[dest] = add(depth, usize::from(rounds != 0))?;
            steps.push(Step {
                pc,
                depth,
                rounds,
                frame_bytes,
                party_bytes,
            });
        }
        // Same-depth locals preserve SSA order and precede that depth's secrets.
        steps.sort_unstable_by_key(|s| (s.depth, s.rounds != 0, s.pc));
        if plan.is_tensor() {
            let mut start = 0;
            let mut logical_end = 0;
            while start < steps.len() {
                let first = steps[start];
                let mut end = start + 1;
                while end < steps.len()
                    && steps[end].depth == first.depth
                    && (steps[end].rounds != 0) == (first.rounds != 0)
                {
                    end += 1;
                }
                logical_end = add(logical_end, mul(end - start, plan.lanes)?)?;
                spans.push(Span {
                    start,
                    end,
                    logical_end,
                });
                start = end;
            }
            if logical_end != plan.resources.instructions {
                return Err(Error::Context);
            }
        }
        let mut result = Self {
            plan,
            steps,
            spans,
            resources: Resources::default(),
            digest: [0; 32],
            active_capacity: if plan.is_tensor() { 0 } else { MAX_ACTIVE },
            _lease: lease,
        };
        let mut r = Resources {
            metadata_bytes: metadata,
            ..Resources::default()
        };
        r.construction_work = add(
            add(metadata, mul(registers, size_of::<usize>())?)?,
            mul(
                mul(instructions, 64)?,
                (instructions.max(1).ilog2() as usize) + 2,
            )?,
        )?;
        if plan.is_tensor() {
            // Price virtual address lookup, grouping and hashing separately from
            // the compact template's depth construction and sort.
            r.construction_work = add(
                r.construction_work,
                mul(
                    mul(plan.resources.instructions, 64)?,
                    (result.spans.len().max(1).ilog2() as usize) + 2,
                )?,
            )?;
        }
        let mut offset = 0;
        let mut hash = Sha256::new()
            .chain_update(b"pllm.fss.public-wavefront.v1\0")
            .chain_update(plan.digest);
        if plan.is_tensor() {
            hash.update(b"compact-tensor-frontier.v1\0");
        }
        while offset < plan.resources.instructions {
            let end = result.group_end(offset)?;
            hash.update(((end - offset) as u64).to_le_bytes());
            let mut rounds = 0;
            let mut bytes = HEADER;
            let mut parties = 0;
            for index in offset..end {
                let s = result.step(index).ok_or(Error::Context)?;
                hash.update((s.pc as u64).to_le_bytes());
                hash.update((s.rounds as u64).to_le_bytes());
                rounds = rounds.max(s.rounds);
                r.primitive_frames = add(r.primitive_frames, s.rounds)?;
                r.frame_bytes_per_party = add(
                    r.frame_bytes_per_party,
                    add(s.frame_bytes, mul(RECORD, s.rounds)?)?,
                )?;
                bytes = add(bytes, add(RECORD, s.frame_bytes)?)?;
                parties = add(parties, s.party_bytes)?;
            }
            if rounds != 0 {
                result.active_capacity = result.active_capacity.max(end - offset);
                r.rounds = add(r.rounds, rounds)?;
                r.frame_bytes_per_party = add(r.frame_bytes_per_party, mul(HEADER, rounds)?)?;
                r.largest_frame_bytes = r.largest_frame_bytes.max(bytes);
                r.retained_party_bytes = r.retained_party_bytes.max(parties);
            }
            offset = end;
        }
        if r.primitive_frames != plan.resources.frames_per_party {
            return Err(Error::Context);
        }
        r.runtime_owner_bytes = add(
            add(
                size_of::<Runtime<'_, '_, '_>>() + size_of::<State<'_>>() + 4096,
                mul(result.active_capacity, size_of::<Slot<'_>>())?,
            )?,
            mul(8, plan.resources.registers + plan.output_count())?,
        )?;
        r.additional_reservation_bytes = add(
            mul(3, add(metadata, mul(registers, size_of::<usize>())?)?)?,
            mul(
                2,
                add(
                    add(r.runtime_owner_bytes, r.retained_party_bytes)?,
                    mul(3, r.largest_frame_bytes)?,
                )?,
            )?,
        )?;
        if add(
            plan.resources.combined_reservation_bytes,
            r.additional_reservation_bytes,
        )? > MAX_ALLOCATION_BYTES
        {
            return Err(Error::Resource);
        }
        result.resources = r;
        result.digest = hash.finalize().into();
        Ok(result)
    }

    fn step(&self, index: usize) -> Option<Step> {
        if !self.plan.is_tensor() {
            return self.steps.get(index).copied();
        }
        let span = self
            .spans
            .get(self.spans.partition_point(|s| s.logical_end <= index))?;
        let width = span.end - span.start;
        let offset = index.checked_sub(span.logical_end - width * self.plan.lanes)?;
        let mut step = *self.steps.get(span.start + offset % width)?;
        step.pc += (offset / width) * self.plan.code.len();
        Some(step)
    }

    fn group_end(&self, start: usize) -> Result<usize, Error> {
        let first = self.step(start).ok_or(Error::Context)?;
        let secret = first.rounds != 0;
        let mut end = start;
        let mut bytes = HEADER;
        let mut parties = 0;
        while let Some(s) = self.step(end) {
            if s.depth != first.depth
                || (s.rounds != 0) != secret
                || (secret
                    && end - start
                        == if self.plan.is_tensor() {
                            MAX_TENSOR_ACTIVE
                        } else {
                            MAX_ACTIVE
                        })
            {
                break;
            }
            let next = add(bytes, add(RECORD, s.frame_bytes)?)?;
            let retained = add(parties, s.party_bytes)?;
            if secret && (next > MAX_FRAME_BYTES || retained > MAX_RETAINED_PARTY_BYTES) {
                break;
            }
            bytes = next;
            parties = retained;
            end += 1;
        }
        if end == start {
            return Err(Error::Resource);
        }
        Ok(end)
    }

    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn material_steps(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.plan.resources.instructions)
            .filter_map(|index| self.step(index))
            .filter(|s| s.rounds != 0)
            .map(|s| s.pc)
    }
    pub fn runtime<'s>(
        &'s self,
        role: u8,
        context: Context,
        inputs: &[u64],
    ) -> Result<Runtime<'s, 'p, 'a>, Error> {
        if role > 1
            || inputs.len() != self.plan.inputs.len()
            || inputs
                .iter()
                .zip(&self.plan.inputs)
                .any(|(&v, &bits)| v > mask(bits))
        {
            return Err(Error::Domain);
        }
        context
            .first_tensor_index
            .checked_add(1)
            .ok_or(Error::Domain)?;
        let budget = self.plan.lease.budget;
        let lease = budget.reserve(self.resources.runtime_owner_bytes)?;
        let mut values = Zeroizing::new(allocate(self.plan.resources.registers, budget)?);
        values.extend_from_slice(inputs);
        values.resize(self.plan.resources.registers, 0);
        let slots = allocate(self.active_capacity, budget)?;
        Ok(Runtime {
            schedule: self,
            role,
            context,
            state: Some(State {
                values,
                slots,
                offset: 0,
                end: 0,
                round: 0,
                waiting: false,
            }),
            lease,
        })
    }
}

struct Slot<'a> {
    pc: usize,
    remaining: usize,
    primitive: Option<Primitive<'a>>,
    frame: Option<Frame<'a>>,
}
struct State<'a> {
    values: Zeroizing<Vec<u64>>,
    slots: Vec<Slot<'a>>,
    offset: usize,
    end: usize,
    round: usize,
    waiting: bool,
}
pub struct Runtime<'s, 'p, 'a> {
    schedule: &'s Schedule<'p, 'a>,
    role: u8,
    context: Context,
    state: Option<State<'a>>,
    lease: Lease<'a>,
}

impl<'a> Runtime<'_, '_, 'a> {
    /// Still-unfed PCs of this frozen secret group only; no polling, advancing
    /// or speculative lookahead into a later dependency frontier.
    pub fn pending_materials(&self, destination: &mut [usize]) -> Result<usize, Error> {
        let state = self.state.as_ref().ok_or(Error::Consumed)?;
        let first = state.offset + state.slots.len();
        if state.waiting || first >= state.end {
            return Err(Error::Context);
        }
        let count = destination.len().min(state.end - first);
        for (index, pc) in destination[..count].iter_mut().enumerate() {
            let step = self.schedule.step(first + index).ok_or(Error::Context)?;
            if step.rounds == 0 {
                return Err(Error::Context);
            }
            *pc = step.pc;
        }
        Ok(count)
    }

    pub fn cancel(&mut self) {
        self.state = None;
    }
    pub fn is_consumed(&self) -> bool {
        self.state.is_none()
    }
    pub fn poll(&mut self) -> Result<Event<'a>, Error> {
        let state = self.state.take().ok_or(Error::Consumed)?;
        if state.waiting || !state.slots.is_empty() {
            return Err(Error::Context);
        }
        self.local(state)
    }
    fn local(&mut self, mut state: State<'a>) -> Result<Event<'a>, Error> {
        let plan = self.schedule.plan;
        while state.offset < plan.resources.instructions {
            state.end = self.schedule.group_end(state.offset)?;
            let first = self.schedule.step(state.offset).ok_or(Error::Context)?;
            if first.rounds != 0 {
                let pc = first.pc;
                self.state = Some(state);
                return Ok(Event::NeedMaterial { pc });
            }
            for index in state.offset..state.end {
                let step = self.schedule.step(index).ok_or(Error::Context)?;
                let dest = plan.inputs.len() + step.pc;
                let value = Zeroizing::new(
                    runtime::local_value(plan.instruction(step.pc)?, self.role, &state.values)
                        .ok_or(Error::Context)?,
                );
                state.values[dest] = *value & mask(plan.register(dest)?.bits);
            }
            state.offset = state.end;
        }
        if state.round != self.schedule.resources.rounds {
            return Err(Error::Context);
        }
        let mut values = Zeroizing::new(allocate(plan.output_count(), self.lease.budget)?);
        for index in 0..plan.output_count() {
            values.push(state.values[plan.output_index(index)?]);
        }
        Ok(Event::Output(Output {
            values,
            _lease: self.lease.split(8 * plan.output_count())?,
        }))
    }
    pub fn feed_result(
        &mut self,
        material: Result<Material<'a>, Error>,
    ) -> Result<Event<'a>, Error> {
        match material {
            Ok(m) => self.feed(m),
            Err(e) => {
                self.state.take().ok_or(Error::Consumed)?;
                Err(e)
            }
        }
    }
    pub fn feed(&mut self, material: Material<'a>) -> Result<Event<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        let plan = self.schedule.plan;
        let index = state.offset + state.slots.len();
        if state.waiting || index >= state.end {
            return Err(Error::Context);
        }
        let step = self.schedule.step(index).ok_or(Error::Context)?;
        if material.pc != step.pc
            || material.role != self.role
            || material.plan != plan.digest
            || material.operation != plan.step_context(self.context, step.pc).operation
        {
            return Err(Error::Context);
        }
        let mut primitive = material.primitive;
        let instruction = plan.instruction(step.pc)?;
        let values = runtime::operands(instruction, &state.values)?;
        let frame = primitive.start(instruction, &values)?;
        state.slots.push(Slot {
            pc: step.pc,
            remaining: step.rounds,
            primitive: Some(primitive),
            frame: Some(frame),
        });
        if index + 1 < state.end {
            let pc = self.schedule.step(index + 1).ok_or(Error::Context)?.pc;
            self.state = Some(state);
            Ok(Event::NeedMaterial { pc })
        } else {
            self.frame(state)
        }
    }
    fn frame(&mut self, mut state: State<'a>) -> Result<Event<'a>, Error> {
        let count = state.slots.iter().filter(|s| s.remaining != 0).count();
        let len = state
            .slots
            .iter()
            .filter(|s| s.remaining != 0)
            .try_fold(HEADER, |n, s| {
                add(
                    n,
                    add(
                        RECORD,
                        s.frame.as_ref().ok_or(Error::Context)?.as_bytes().len(),
                    )?,
                )
            })?;
        if count == 0 || len > self.schedule.resources.largest_frame_bytes || len > MAX_FRAME_BYTES
        {
            return Err(Error::Resource);
        }
        let lease = self.lease.budget.reserve(len)?;
        let mut bytes = Zeroizing::new(allocate(len, self.lease.budget)?);
        bytes.extend_from_slice(b"FFW1");
        bytes.extend_from_slice(&[self.role, 0, 0, 0]);
        bytes.extend_from_slice(&(state.round as u64).to_le_bytes());
        bytes.extend_from_slice(&(count as u32).to_le_bytes());
        for slot in &mut state.slots {
            if slot.remaining == 0 {
                continue;
            }
            let frame = slot.frame.take().ok_or(Error::Context)?;
            bytes.extend_from_slice(&(slot.pc as u32).to_le_bytes());
            bytes.extend_from_slice(&(frame.as_bytes().len() as u32).to_le_bytes());
            bytes.extend_from_slice(frame.as_bytes());
        }
        state.waiting = true;
        self.state = Some(state);
        Ok(Event::Frame(Frame {
            bytes,
            _lease: lease,
        }))
    }
    pub fn advance(&mut self, peer: &[u8]) -> Result<Event<'a>, Error> {
        let mut state = self.state.take().ok_or(Error::Consumed)?;
        if !state.waiting
            || peer.len() < HEADER
            || peer.len() > self.schedule.resources.largest_frame_bytes
            || peer[..4] != *b"FFW1"
            || peer[4..8] != [1 - self.role, 0, 0, 0]
            || peer[8..16] != (state.round as u64).to_le_bytes()
            || peer[16..20]
                != (state.slots.iter().filter(|s| s.remaining != 0).count() as u32).to_le_bytes()
        {
            return Err(Error::Context);
        }
        let plan = self.schedule.plan;
        let mut offset = HEADER;
        for slot in &mut state.slots {
            if slot.remaining == 0 {
                continue;
            }
            let header = peer
                .get(offset..add(offset, RECORD)?)
                .ok_or(Error::Context)?;
            let pc =
                u32::from_le_bytes(header[..4].try_into().map_err(|_| Error::Context)?) as usize;
            let len =
                u32::from_le_bytes(header[4..].try_into().map_err(|_| Error::Context)?) as usize;
            if pc != slot.pc || len == 0 {
                return Err(Error::Context);
            }
            offset = add(offset, RECORD)?;
            let bytes = peer.get(offset..add(offset, len)?).ok_or(Error::Context)?;
            offset = add(offset, len)?;
            let primitive = slot.primitive.as_mut().ok_or(Error::Context)?;
            slot.remaining = slot.remaining.checked_sub(1).ok_or(Error::Context)?;
            match primitive.advance(bytes)? {
                PrimitiveEvent::Frame(frame) => {
                    if slot.remaining == 0 {
                        return Err(Error::Context);
                    }
                    slot.frame = Some(frame);
                }
                PrimitiveEvent::Output(output) => {
                    if slot.remaining != 0 || output.shares().len() != 1 {
                        return Err(Error::Context);
                    }
                    let dest = plan.inputs.len() + slot.pc;
                    if output.shares()[0] > mask(plan.register(dest)?.bits) {
                        return Err(Error::Context);
                    }
                    state.values[dest] = output.shares()[0];
                    slot.primitive = None;
                }
                PrimitiveEvent::Scalar(value) => {
                    let dest = plan.inputs.len() + slot.pc;
                    if slot.remaining != 0 || *value > mask(plan.register(dest)?.bits) {
                        return Err(Error::Context);
                    }
                    state.values[dest] = *value;
                    slot.primitive = None;
                }
            }
        }
        if offset != peer.len() {
            return Err(Error::Context);
        }
        state.waiting = false;
        state.round = add(state.round, 1)?;
        if state.slots.iter().any(|s| s.remaining != 0) {
            self.frame(state)
        } else {
            state.slots.clear();
            state.offset = state.end;
            self.local(state)
        }
    }
}

#[cfg(test)]
#[path = "wavefront_tests.rs"]
mod tests;
