//! Bounded local-register binary body unit.
//!
//! This executes semantic sites, not a hidden vector of expanded gather/product
//! nodes. Each product uses the original 128-lane chunk order and the existing
//! complete multiply/rescale protocol. Views retain their original owner until
//! last use. This unit intentionally does NOT expose a decoder policy: row,
//! matrix, checked-range and sticky-validity integration must precede that.
use super::{
    arithmetic_frames, multiply, Message, MessagePhase, Party, PartyRuntime, PeerBytes, Status,
};
use crate::replicated_decoder::{
    puma, vec_zero, Error, InputBounds, Interval, Secret, WeightMode, WorkBudget, MAX_NODES,
    MAX_PROGRAM_BYTES, MAX_VALUES,
};
use sha2::{Digest, Sha256};
use std::{mem::size_of, sync::Arc};

const TILE: usize = 128;
const UNUSED: usize = usize::MAX;
const RETAINED: usize = usize::MAX - 1;

#[path = "body_layer.rs"]
pub mod layer;

#[path = "body_integration.rs"]
pub mod integration;

#[path = "body_handoff.rs"]
pub(in crate::replicated_decoder) mod handoff;

#[derive(Clone, Copy)]
pub struct Register {
    pub lanes: usize,
    pub bounds: Interval,
}

/// A checked public affine layout; no descriptor per secret lane.
#[derive(Clone, Copy)]
pub struct View {
    pub owner: usize,
    pub offset: usize,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
}
impl View {
    pub fn contiguous(owner: usize, lanes: usize) -> Self {
        Self {
            owner,
            offset: 0,
            rows: 1,
            columns: lanes,
            row_stride: lanes,
            column_stride: 1,
        }
    }
    fn len(self) -> Result<usize, Error> {
        let n = self.rows.checked_mul(self.columns).ok_or(Error::Bounds)?;
        if n == 0 || n > MAX_VALUES {
            return Err(Error::Shape);
        }
        Ok(n)
    }
    fn validate(self, registers: &[Register]) -> Result<usize, Error> {
        let n = self.len()?;
        let end = self
            .offset
            .checked_add(
                (self.rows - 1)
                    .checked_mul(self.row_stride)
                    .ok_or(Error::Bounds)?,
            )
            .and_then(|v| v.checked_add((self.columns - 1).checked_mul(self.column_stride)?))
            .ok_or(Error::Bounds)?;
        if end >= registers.get(self.owner).ok_or(Error::Binding)?.lanes {
            return Err(Error::Shape);
        }
        Ok(n)
    }
    fn index(self, i: usize) -> usize {
        // All endpoints and products were checked during immutable admission.
        self.offset + (i / self.columns) * self.row_stride + (i % self.columns) * self.column_stride
    }
    fn hash(self, h: &mut Sha256) {
        for v in [
            self.owner,
            self.offset,
            self.rows,
            self.columns,
            self.row_stride,
            self.column_stride,
        ] {
            h.update((v as u64).to_le_bytes());
        }
    }
}

#[derive(Clone, Copy)]
pub enum Binary {
    Add,
    Multiply,
}

#[derive(Clone, Copy)]
pub struct Site {
    pub left: View,
    pub right: View,
    pub operation: Binary,
}

/// Exact owned payload/slot sizes plus separately named conservative scratch
/// allowances. External router queues and caller-retained input copies are not
/// this unit's owners and must be charged by its enclosing executor.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Resources {
    pub immutable_metadata_bytes: usize,
    pub register_storage_bytes: usize,
    pub constructor_slot_bytes: usize,
    pub peak_live_secret_bytes: usize,
    pub gathered_operand_bytes: usize,
    pub primitive_ledger_allowance_bytes: usize,
    pub bounds_stack_bytes: usize,
    pub owner_and_arc_allowance_bytes: usize,
    pub admitted_bytes: usize,
}

pub struct Plan {
    inputs: usize,
    registers: Box<[Register]>,
    sites: Box<[Site]>,
    outputs: Box<[View]>,
    retire: Box<[Vec<usize>]>,
    unused: Box<[usize]>,
    digest: [u8; 32],
    work: WorkBudget,
    resources: Resources,
    product_tiles: u64,
}

impl Plan {
    pub fn new(
        weights: WeightMode,
        inputs: &[Register],
        sites: &[Site],
        outputs: &[View],
    ) -> Result<Self, Error> {
        if weights != WeightMode::Public {
            return Err(Error::Binding);
        }
        let count = inputs.len().checked_add(sites.len()).ok_or(Error::Bounds)?;
        if inputs.is_empty()
            || sites.is_empty()
            || outputs.is_empty()
            || count > MAX_NODES
            || outputs.len() > MAX_NODES
        {
            return Err(Error::Bounds);
        }
        let mut registers = Vec::new();
        registers
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation)?;
        for input in inputs {
            if input.lanes == 0 || input.lanes > MAX_VALUES {
                return Err(Error::Shape);
            }
            registers.push(*input);
        }
        let mut last = vec_zero::<usize>(count)?;
        last.fill(UNUSED);
        let mut work = WorkBudget::default();
        let mut product_tiles = 0_u64;
        let mut hash = Sha256::new();
        hash.update(b"pllm.puma.binary-body-unit.v1");
        hash.update((inputs.len() as u64).to_le_bytes());
        hash.update((sites.len() as u64).to_le_bytes());
        for input in inputs {
            hash.update((input.lanes as u64).to_le_bytes());
            hash.update(input.bounds.lower().to_le_bytes());
            hash.update(input.bounds.upper().to_le_bytes());
        }
        for (step, site) in sites.iter().enumerate() {
            let lanes = site.left.validate(&registers)?;
            if site.right.validate(&registers)? != lanes {
                return Err(Error::Shape);
            }
            let bounds = InputBounds {
                left: registers[site.left.owner].bounds,
                right: registers[site.right.owner].bounds,
            };
            let range = match site.operation {
                Binary::Add => multiply::arithmetic::Affine {
                    left: 1,
                    right: 1,
                    constant: 0,
                }
                .output_bounds(bounds)?,
                Binary::Multiply => {
                    work = work.checked_add(
                        WorkBudget::product(TILE.min(lanes), true)?
                            .checked_mul((lanes / TILE) as u64)?,
                    )?;
                    if lanes % TILE != 0 {
                        // For a short sole tile the preceding multiplier is zero.
                        work = work.checked_add(WorkBudget::product(lanes % TILE, true)?)?;
                    }
                    product_tiles = product_tiles
                        .checked_add(lanes.div_ceil(TILE) as u64)
                        .ok_or(Error::Bounds)?;
                    bounds.output()?
                }
            };
            last[site.left.owner] = step;
            last[site.right.owner] = step;
            last[inputs.len() + step] = step;
            registers.push(Register {
                lanes,
                bounds: range,
            });
            hash.update([match site.operation {
                Binary::Add => 0,
                Binary::Multiply => 1,
            }]);
            site.left.hash(&mut hash);
            site.right.hash(&mut hash);
        }
        hash.update((outputs.len() as u64).to_le_bytes());
        for output in outputs {
            output.validate(&registers)?;
            last[output.owner] = RETAINED;
            output.hash(&mut hash);
        }
        // This unit cannot authorize a larger enclosing program. The caller must
        // still aggregate all units/guards and enforce the original limits.
        work.limits()?;
        let mut retire = Vec::new();
        retire
            .try_reserve_exact(sites.len())
            .map_err(|_| Error::Allocation)?;
        retire.resize_with(sites.len(), Vec::new);
        let mut unused = Vec::new();
        for (owner, &step) in last.iter().enumerate() {
            if step == UNUSED {
                unused.try_reserve(1).map_err(|_| Error::Allocation)?;
                unused.push(owner);
            } else if step != RETAINED {
                retire[step].try_reserve(1).map_err(|_| Error::Allocation)?;
                retire[step].push(owner);
            }
        }
        let bytes = |lanes: usize| lanes.checked_mul(16).ok_or(Error::Bounds);
        let mut live = inputs.iter().try_fold(0usize, |v, r| {
            v.checked_add(bytes(r.lanes)?).ok_or(Error::Bounds)
        })?;
        let mut peak = live;
        for &owner in &unused {
            live -= bytes(registers[owner].lanes)?;
        }
        for (step, reg) in registers[inputs.len()..].iter().enumerate() {
            live = live.checked_add(bytes(reg.lanes)?).ok_or(Error::Bounds)?;
            peak = peak.max(live);
            for &owner in &retire[step] {
                live -= bytes(registers[owner].lanes)?;
            }
        }
        let metadata = size_of::<Self>()
            + count * size_of::<Register>()
            + sites.len() * (size_of::<Site>() + size_of::<Vec<usize>>())
            + std::mem::size_of_val(outputs)
            + retire
                .iter()
                .map(|v| v.capacity() * size_of::<usize>())
                .sum::<usize>()
            + unused.len() * size_of::<usize>();
        let mut resources = Resources {
            immutable_metadata_bytes: metadata,
            register_storage_bytes: count * size_of::<Option<Secret>>(),
            constructor_slot_bytes: inputs.len() * size_of::<Secret>(),
            peak_live_secret_bytes: peak,
            gathered_operand_bytes: 2 * TILE * 16,
            primitive_ledger_allowance_bytes: puma::MAX_ADMITTED_BYTES,
            bounds_stack_bytes: TILE * size_of::<InputBounds>(),
            // Includes both the executing and finished owner, even though they
            // do not coexist, plus conservative Arc/control-block allowance.
            owner_and_arc_allowance_bytes: size_of::<Executor>() + size_of::<Values>() + 64,
            admitted_bytes: 0,
        };
        resources.admitted_bytes = [
            metadata,
            resources.register_storage_bytes,
            resources.constructor_slot_bytes,
            peak,
            resources.gathered_operand_bytes,
            resources.primitive_ledger_allowance_bytes,
            resources.bounds_stack_bytes,
            resources.owner_and_arc_allowance_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |a, b| a.checked_add(b).ok_or(Error::Bounds))?;
        if resources.admitted_bytes > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        Ok(Self {
            inputs: inputs.len(),
            registers: registers.into_boxed_slice(),
            sites: sites.into(),
            outputs: outputs.into(),
            retire: retire.into_boxed_slice(),
            unused: unused.into_boxed_slice(),
            digest: hash.finalize().into(),
            work,
            resources,
            product_tiles,
        })
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn work_budget(&self) -> WorkBudget {
        self.work
    }
    pub fn peak_live_bytes(&self) -> usize {
        self.resources.peak_live_secret_bytes
    }
    pub fn metadata_bytes(&self) -> usize {
        self.resources.immutable_metadata_bytes
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn coordinator_advances(&self) -> u64 {
        self.product_tiles + 1
    }
}

enum Active {
    Ready(PartyRuntime),
    Product(PartyRuntime),
    Retired,
}

/// Owned local registers. No writable alias survives last-use retirement.
pub struct Executor {
    plan: Arc<Plan>,
    values: Vec<Option<Secret>>,
    active: Active,
    inbox: [Option<Message>; 2],
    step: usize,
    offset: usize,
    pid: u32,
}

impl Executor {
    /// Enclosing program must bind plan.digest() into its execution identity.
    /// Borrowed input owners must be charged separately if callers copy them.
    pub fn new(plan: Arc<Plan>, runtime: PartyRuntime, inputs: Vec<Secret>) -> Result<Self, Error> {
        if runtime.status() != Status::Ready || inputs.len() != plan.inputs {
            return Err(Error::Phase);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(plan.registers.len())
            .map_err(|_| Error::Allocation)?;
        for (input, spec) in inputs.into_iter().zip(plan.registers.iter()) {
            if input.first.len() != spec.lanes || input.second.len() != spec.lanes {
                return Err(Error::Shape);
            }
            values.push(Some(input));
        }
        values.resize_with(plan.registers.len(), || None);
        for &owner in &plan.unused {
            values[owner] = None;
        }
        Ok(Self {
            plan,
            values,
            active: Active::Ready(runtime),
            inbox: [None, None],
            step: 0,
            offset: 0,
            pid: std::process::id(),
        })
    }
    pub fn complete(&self) -> bool {
        self.step == self.plan.sites.len() && matches!(self.active, Active::Ready(_))
    }
    #[cfg(test)]
    pub fn live_bytes(&self) -> usize {
        self.values
            .iter()
            .flatten()
            .map(|v| 16 * v.first.len())
            .sum()
    }
    pub fn cancel(&mut self) {
        self.active = Active::Retired;
        self.inbox = [None, None];
        self.values.clear();
    }
    fn live(&self) -> Result<(), Error> {
        if self.pid != std::process::id() || matches!(self.active, Active::Retired) {
            Err(Error::Retired)
        } else {
            Ok(())
        }
    }
    fn checked<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.cancel();
        }
        result
    }
    fn word(&self, view: View, i: usize) -> Result<(u64, u64), Error> {
        let value = self
            .values
            .get(view.owner)
            .and_then(Option::as_ref)
            .ok_or(Error::Binding)?;
        let index = view.index(i);
        Ok((
            *value.first.get(index).ok_or(Error::Shape)?,
            *value.second.get(index).ok_or(Error::Shape)?,
        ))
    }
    fn next_site(&mut self) {
        for &owner in &self.plan.retire[self.step] {
            self.values[owner] = None;
        }
        self.step += 1;
        self.offset = 0;
    }
    pub fn advance(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = self.advance_inner();
        self.checked(result)
    }
    fn advance_inner(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        self.live()?;
        if self.complete() || self.inbox.iter().any(Option::is_some) {
            return Err(Error::Phase);
        }
        let active = std::mem::replace(&mut self.active, Active::Retired);
        let mut runtime = match active {
            Active::Ready(runtime) => runtime,
            Active::Product(mut runtime) => {
                let value = runtime.take_output()?;
                let owner = self.plan.inputs + self.step;
                let output = self.values[owner].as_mut().ok_or(Error::Phase)?;
                let end = self
                    .offset
                    .checked_add(value.first().len())
                    .ok_or(Error::Bounds)?;
                output
                    .first
                    .get_mut(self.offset..end)
                    .ok_or(Error::Shape)?
                    .copy_from_slice(value.first());
                output
                    .second
                    .get_mut(self.offset..end)
                    .ok_or(Error::Shape)?
                    .copy_from_slice(value.second());
                self.offset = end;
                if end == self.plan.registers[owner].lanes {
                    self.next_site();
                }
                runtime
            }
            Active::Retired => return Err(Error::Retired),
        };
        while self.step < self.plan.sites.len() {
            let site = self.plan.sites[self.step];
            let owner = self.plan.inputs + self.step;
            let lanes = self.plan.registers[owner].lanes;
            if self.offset == 0 {
                self.values[owner] = Some(Secret::new(lanes)?);
            }
            match site.operation {
                Binary::Add => {
                    for i in 0..lanes {
                        let (a, b) = self.word(site.left, i)?;
                        let (c, d) = self.word(site.right, i)?;
                        let out = self.values[owner].as_mut().ok_or(Error::Phase)?;
                        out.first[i] = a.wrapping_add(c);
                        out.second[i] = b.wrapping_add(d);
                    }
                    self.next_site();
                }
                Binary::Multiply => {
                    let n = TILE.min(lanes - self.offset);
                    let mut a = Secret::new(n)?;
                    let mut b = Secret::new(n)?;
                    for i in 0..n {
                        (a.first[i], a.second[i]) = self.word(site.left, self.offset + i)?;
                        (b.first[i], b.second[i]) = self.word(site.right, self.offset + i)?;
                    }
                    let bounds = InputBounds {
                        left: self.plan.registers[site.left.owner].bounds,
                        right: self.plan.registers[site.right.owner].bounds,
                    };
                    let plan = runtime.plan(&[bounds; TILE][..n])?;
                    let party = runtime.party();
                    let left = plan.share_operand(
                        multiply::Operand::Left,
                        party,
                        &a.first,
                        &a.second,
                        puma::Limits::default(),
                    )?;
                    let right = plan.share_operand(
                        multiply::Operand::Right,
                        party,
                        &b.first,
                        &b.second,
                        puma::Limits::default(),
                    )?;
                    let frames =
                        arithmetic_frames([Some(runtime.begin(left, right)?), None, None])?;
                    self.active = Active::Product(runtime);
                    return Ok(frames);
                }
            }
        }
        self.active = Active::Ready(runtime);
        Ok(vec![])
    }
    pub fn receive(&mut self, raw: &[u8], sender: Party) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            self.live()?;
            let Active::Product(runtime) = &mut self.active else {
                return Err(Error::Phase);
            };
            let message = Message::decode(raw, sender, runtime.party())?;
            let mut frames = [None, None, None];
            match runtime.status() {
                Status::AwaitingRotate => {
                    let [a, b] = runtime.rotate(message)?;
                    frames = [a, b, None];
                }
                Status::AwaitingOpen => {
                    let slot = match message.header().phase {
                        MessagePhase::MaskedInput => 0,
                        MessagePhase::Correlation => 1,
                        _ => return Err(Error::Phase),
                    };
                    if self.inbox[slot].is_some() {
                        return Err(Error::Binding);
                    }
                    self.inbox[slot] = Some(message);
                    if self.inbox.iter().all(Option::is_some) {
                        frames[0] = Some(runtime.open(
                            self.inbox[0].take().ok_or(Error::Phase)?,
                            self.inbox[1].take().ok_or(Error::Phase)?,
                        )?);
                    }
                }
                Status::AwaitingReshare => runtime.reshare(message)?,
                _ => return Err(Error::Phase),
            }
            arithmetic_frames(frames)
        })();
        self.checked(result)
    }
    pub fn finish(mut self) -> Result<(PartyRuntime, Values), Error> {
        self.live()?;
        if !self.complete() {
            return Err(Error::Phase);
        }
        let Active::Ready(runtime) = std::mem::replace(&mut self.active, Active::Retired) else {
            return Err(Error::Phase);
        };
        Ok((
            runtime,
            Values {
                values: std::mem::take(&mut self.values),
                plan: self.plan.clone(),
            },
        ))
    }
}
impl Drop for Executor {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Output views borrow retained owners; finishing never clones full tensors.
pub struct Values {
    values: Vec<Option<Secret>>,
    plan: Arc<Plan>,
}
impl Values {
    pub fn word(&self, output: usize, index: usize) -> Result<(u64, u64), Error> {
        let view = *self.plan.outputs.get(output).ok_or(Error::Shape)?;
        if index >= view.len()? {
            return Err(Error::Shape);
        }
        let value = self.values[view.owner].as_ref().ok_or(Error::Binding)?;
        Ok((
            value.first[view.index(index)],
            value.second[view.index(index)],
        ))
    }
}

#[cfg(test)]
pub fn from_test_worker(plan: Arc<Plan>, mut worker: super::Worker) -> Executor {
    let super::Active::Idle(runtime) =
        std::mem::replace(&mut worker.active, super::Active::Retired)
    else {
        panic!("expected idle test worker");
    };
    let inputs = worker.values[..worker.program.inputs]
        .iter_mut()
        .map(|value| value.take().unwrap())
        .collect();
    Executor::new(plan, runtime, inputs).unwrap()
}
