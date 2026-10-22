//! Exact binary-subgraph handoff for an already admitted decoder program.
//!
//! No new Code variant, source authority, default policy or execution identity.
//! Only original gather/gather/binary triples are folded; every product retains
//! its original primitive width, order, input bounds and consuming runtime.
use super::*;
use crate::replicated_decoder::{Code, Node, Program, Worker};
use std::collections::BTreeMap;

pub use super::super::BoundaryProgress;

// Borrow dependencies directly: Node::inputs() constructs a Vec, including when
// merely asking whether one exported owner remains live.
fn uses(node: &Node, owner: usize) -> bool {
    match &node.code {
        Code::Input(_) | Code::PublicInput { .. } | Code::PublicValues(_) => false,
        Code::Gather(words) => words.owners().any(|id| id == owner),
        Code::SourceGather(view) => view.owner == owner,
        Code::Affine { left, right, .. } => *left == owner || *right == Some(owner),
        Code::Product { left, right, .. } => *left == owner || *right == owner,
        Code::PublicProduct { input, weight, .. }
        | Code::PublicLookup { input, weight, .. }
        | Code::TrieLookup { input, weight, .. } => *input == owner || *weight == owner,
        Code::Tensor { input, .. }
        | Code::CheckedRange { input, .. }
        | Code::Selection { input, .. } => *input == owner,
        Code::TensorLoop { input, .. } => input.owner() == owner,
    }
}

fn add(a: usize, b: usize) -> Result<usize, Error> {
    a.checked_add(b).ok_or(Error::Bounds)
}
fn bytes(n: usize, width: usize) -> Result<usize, Error> {
    n.checked_mul(width).ok_or(Error::Bounds)
}

/// Allocation-free upper bound, BEFORE lowering or Plan::new. Count repeated
/// external owners repeatedly rather than allocating a deduplication set. Count
/// every interval node as exported, even if no outside consumer exists. Rejecting
/// an otherwise fitting interval is preferable to unpriced constructor overlap.
fn preflight(program: &Program, start: usize, end: usize) -> Result<usize, Error> {
    let span = end.checked_sub(start).ok_or(Error::Bounds)?;
    let sites = span / 3;
    let mut external_owners = 0usize;
    let mut input_lanes = 0usize;
    let mut site_lanes = 0usize;
    let mut exported_lanes = 0usize;
    for step in (start..end).step_by(3) {
        for node in &program.nodes[step..step + 2] {
            let Code::Gather(words) = &node.code else {
                return Err(Error::Binding);
            };
            let (owner, _) = words.iter().next().ok_or(Error::Shape)?;
            // Exact lowering supports only one affine source per gather. Reject
            // mixed owners now, so 2*sites also bounds external owner entries.
            if words.iter().any(|(id, _)| id != owner) {
                return Err(Error::Binding);
            }
            if owner < start {
                let source = program.nodes.get(owner).ok_or(Error::Binding)?;
                external_owners = add(external_owners, 1)?;
                input_lanes = add(input_lanes, source.lanes)?;
            }
        }
        site_lanes = add(site_lanes, program.nodes[step + 2].lanes)?;
        for node in &program.nodes[step..step + 3] {
            if node.lanes == 0 || node.lanes > MAX_VALUES {
                return Err(Error::Bounds);
            }
            exported_lanes = add(exported_lanes, node.lanes)?;
        }
    }
    let registers = add(external_owners, sites)?;
    if registers > MAX_NODES || span > MAX_NODES {
        return Err(Error::Bounds);
    }
    // Program::payload_bytes already charges 16*lanes for EVERY non-file node,
    // not merely imports or a liveness estimate. Thus all live prefix owners,
    // including retained validity/feedback, remain charged throughout the body.
    // Sum, not max: allow all input copies and all site outputs simultaneously,
    // plus all exported copies while retained body owners are still alive.
    let live_payload = bytes(add(input_lanes, site_lanes)?, 16)?;
    let export_overlap = bytes(exported_lanes, 16)?;
    // 4096 bytes per entry is deliberately loose. Covers BOTH growing and
    // boxed copies of input/spec/site/output/view/last/unused/register vectors,
    // Option<Secret> slots, nested retirement Vecs (minimum capacity included),
    // BTreeMap nodes for owners/layouts, Plan immutable metadata, and the final
    // validation's 512-byte-per-entry lowering allowance. Current entries are
    // at most Site/View-sized; no tensor templates or per-lane nodes occur here.
    let entries = add(add(registers, span)?, add(sites, 16)?)?;
    let construction = bytes(entries, 4096)?;
    let fixed_owners = [
        size_of::<Compiled>(),
        size_of::<Invocation>(),
        size_of::<Plan>(),
        size_of::<Executor>(),
        size_of::<Values>(),
        4096, // Arc/control blocks and empty-vector/control-owner allowance.
    ]
    .into_iter()
    .try_fold(0, add)?;
    let primitive_scratch = add(
        puma::MAX_ADMITTED_BYTES,
        add(bytes(2 * TILE, 16)?, bytes(TILE, size_of::<InputBounds>())?)?,
    )?;
    let total = [
        program.payload_bytes(),
        live_payload,
        export_overlap,
        construction,
        fixed_owners,
        primitive_scratch,
    ]
    .into_iter()
    .try_fold(0, add)?;
    if total > MAX_PROGRAM_BYTES {
        return Err(Error::Bounds);
    }
    Ok(total)
}

#[cfg(test)]
std::thread_local! {
    // Counts entry to the first allocating constructor region, not eventual
    // success. Regression tests detect moving preflight behind ANY lowering.
    static CONSTRUCTOR_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn constructor_entries() -> usize {
    CONSTRUCTOR_ENTRIES.with(std::cell::Cell::get)
}

/// Compiler-owned mapping; callers cannot substitute a body with different work.
pub struct Compiled {
    program: Arc<Program>,
    plan: Arc<Plan>,
    start: usize,
    end: usize,
    inputs: Box<[usize]>,
    outputs: Box<[usize]>,
    admitted_bytes: usize,
}

impl Compiled {
    pub fn new(
        program: Arc<Program>,
        weights: WeightMode,
        start: usize,
        end: usize,
    ) -> Result<Arc<Self>, Error> {
        if weights != WeightMode::Public
            || start < program.inputs
            || start >= end
            || end > program.nodes.len()
            || (end - start) % 3 != 0
        {
            return Err(Error::Binding);
        }
        let upper_bytes = preflight(&program, start, end)?;
        #[cfg(test)]
        CONSTRUCTOR_ENTRIES.with(|count| count.set(count.get() + 1));
        let mut inputs = Vec::new();
        let mut specs = Vec::new();
        let mut owners = BTreeMap::new();
        // Discover external owners before assigning local result registers.
        for step in (start..end).step_by(3) {
            for node in &program.nodes[step..step + 2] {
                let Code::Gather(words) = &node.code else {
                    return Err(Error::Binding);
                };
                for (id, _) in words.iter() {
                    if id < start && !owners.contains_key(&id) {
                        let input = program.nodes.get(id).ok_or(Error::Binding)?;
                        // Public file owners remain exclusively source-certified;
                        // this bridge never relabels them as secret input owners.
                        if matches!(input.code, Code::PublicInput { .. }) {
                            return Err(Error::Binding);
                        }
                        owners.insert(id, specs.len());
                        inputs.push(id);
                        specs.push(Register {
                            lanes: input.lanes,
                            bounds: input.bounds,
                        });
                    }
                }
            }
        }
        let mut sites = Vec::new();
        let mut layouts = BTreeMap::new();
        let mut work = WorkBudget::default();
        for step in (start..end).step_by(3) {
            for id in step..step + 2 {
                let Code::Gather(words) = &program.nodes[id].code else {
                    return Err(Error::Binding);
                };
                let mut iter = words.iter();
                let (source, offset) = iter.next().ok_or(Error::Shape)?;
                let owner = *owners.get(&source).ok_or(Error::Binding)?;
                let stride = if words.len() > 1 {
                    let (next, index) = iter.next().ok_or(Error::Shape)?;
                    if next != source {
                        return Err(Error::Binding);
                    }
                    index.checked_sub(offset).ok_or(Error::Binding)?
                } else {
                    1
                };
                let view = View {
                    owner,
                    offset,
                    rows: 1,
                    columns: words.len(),
                    row_stride: words.len(),
                    column_stride: stride,
                };
                if words.iter().enumerate().any(|(i, word)| {
                    i.checked_mul(stride)
                        .and_then(|index| offset.checked_add(index))
                        != Some(word.1)
                        || word.0 != source
                }) || program.nodes[id].bounds != program.nodes[source].bounds
                {
                    return Err(Error::Binding);
                }
                layouts.insert(id, view);
            }
            let node = &program.nodes[step + 2];
            let operation = match &node.code {
                Code::Affine {
                    left,
                    right: Some(right),
                    coefficients,
                } if *left == step
                    && *right == step + 1
                    && coefficients.left == 1
                    && coefficients.right == 1
                    && coefficients.constant == 0 =>
                {
                    Binary::Add
                }
                Code::Product {
                    left,
                    right,
                    shape: None,
                    rescale: true,
                    norm: None,
                } if *left == step && *right == step + 1 && node.lanes <= TILE => Binary::Multiply,
                _ => return Err(Error::Binding),
            };
            sites.push(Site {
                left: layouts[&step],
                right: layouts[&(step + 1)],
                operation,
            });
            let owner = specs.len() + sites.len() - 1;
            owners.insert(step + 2, owner);
            layouts.insert(step + 2, View::contiguous(owner, node.lanes));
            work = work.checked_add(node.work_budget()?)?;
        }
        // Export every original owner used after the folded interval, including
        // gathers. Checked validity and feedback remain ordinary outside nodes.
        let mut outputs = Vec::new();
        for id in start..end {
            if program.outputs.iter().any(|&(owner, _)| owner == id)
                || program.nodes[end..].iter().any(|node| uses(node, id))
            {
                outputs.push(id);
            }
        }
        let views: Vec<_> = outputs.iter().map(|id| layouts[id]).collect();
        let plan = Arc::new(Plan::new(weights, &specs, &sites, &views)?);
        if plan.work_budget() != work {
            return Err(Error::Binding);
        }
        for (index, step) in (start..end).step_by(3).enumerate() {
            let register = plan.registers[specs.len() + index];
            let node = &program.nodes[step + 2];
            if register.lanes != node.lanes || register.bounds != node.bounds {
                return Err(Error::Binding);
            }
        }
        // Charge original retained graph/values PLUS complete body workspace,
        // bridge slots and simultaneous exported copies. Never enlarge cap.
        let bridge_bytes = inputs.len() * size_of::<usize>()
            + outputs.len() * size_of::<usize>()
            + size_of::<Self>()
            + size_of::<Invocation>()
            + 128;
        // Bounded conservative constructor allowance: tree entries, vector
        // growth and temporary sites/layouts coexist with the finished plan.
        let lowering_workspace = (end - start + inputs.len())
            .checked_mul(512)
            .ok_or(Error::Bounds)?;
        let bytes = program
            .payload_bytes()
            .checked_add(plan.resources().admitted_bytes)
            .and_then(|n| n.checked_add(bridge_bytes))
            .and_then(|n| n.checked_add(lowering_workspace))
            .ok_or(Error::Bounds)?;
        if bytes > MAX_PROGRAM_BYTES || bytes > upper_bytes {
            return Err(Error::Bounds);
        }
        Ok(Arc::new(Self {
            program,
            plan,
            start,
            end,
            inputs: inputs.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
            admitted_bytes: upper_bytes,
        }))
    }

    pub fn admitted_bytes(&self) -> usize {
        self.admitted_bytes
    }

    /// Drive this exact Program Arc to the exclusive compiled start boundary.
    /// Route every returned frame across all three roles before advancing again.
    /// Only call `begin` after all roles report `Reached` and the router is empty;
    /// this local result alone does not certify global quiescence.
    pub fn advance_to_start(&self, worker: &mut Worker) -> Result<BoundaryProgress, Error> {
        if !Arc::ptr_eq(&self.program, &worker.program) {
            worker.cancel();
            return Err(Error::Binding);
        }
        worker.advance_before(self.start)
    }

    /// Consumes one quiescent worker. At initial import boundary, retire only
    /// input declarations before entering the exact compiled interval.
    /// Caller must first globally drain frames across all three roles; these
    /// local checks cannot prove that the authenticated router is empty.
    pub fn begin(self: Arc<Self>, mut worker: Worker) -> Result<Invocation, Error> {
        let result = (|| {
            worker.live()?;
            // Expanded test-only checkpoint capture expects every retired owner.
            // Until capture has a compact-body mapping, reject rather than skip
            // boundaries or fabricate missing intermediate observations.
            #[cfg(test)]
            if super::super::oracle_trace::enabled() {
                return Err(Error::Binding);
            }
            if !Arc::ptr_eq(&self.program, &worker.program)
                || worker.imported != worker.program.inputs
                || !worker.boundary_ready()
            {
                return Err(Error::Binding);
            }
            if worker.step == 0 && self.start == worker.program.inputs {
                while worker.step < self.start {
                    worker.retire_inputs();
                }
            }
            if worker.step != self.start {
                return Err(Error::Phase);
            }
            let mut inputs = Vec::new();
            inputs
                .try_reserve_exact(self.inputs.len())
                .map_err(|_| Error::Allocation)?;
            // Preserve outer owners: later ordinary nodes may still use them.
            // Their copies and export overlap are included in aggregate admission.
            for &id in &self.inputs {
                let value = worker.values[id].as_ref().ok_or(Error::Binding)?;
                let mut copy = Secret::new(value.first.len())?;
                copy.first.copy_from_slice(&value.first);
                copy.second.copy_from_slice(&value.second);
                inputs.push(copy);
            }
            let super::super::Active::Idle(runtime) =
                std::mem::replace(&mut worker.active, super::super::Active::Retired)
            else {
                return Err(Error::Phase);
            };
            Executor::new(self.plan.clone(), runtime, inputs)
        })();
        match result {
            Ok(executor) => Ok(Invocation {
                compiled: self,
                worker,
                executor: Some(executor),
            }),
            Err(error) => {
                worker.cancel();
                Err(error)
            }
        }
    }
}

/// Suspended decoder retains ordinary source/validity/feedback owners. No token
/// disclosure or reusable worker is available until the body finishes cleanly.
pub struct Invocation {
    compiled: Arc<Compiled>,
    worker: Worker,
    executor: Option<Executor>,
}
impl Invocation {
    pub fn complete(&self) -> bool {
        self.executor.as_ref().is_some_and(Executor::complete)
    }
    pub fn party(&self) -> Party {
        self.worker.party()
    }
    pub fn cancel(&mut self) {
        self.executor = None;
        self.worker.cancel();
    }
    pub fn advance(&mut self) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = self.executor.as_mut().ok_or(Error::Retired)?.advance();
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn receive(&mut self, raw: &[u8], sender: Party) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = self
            .executor
            .as_mut()
            .ok_or(Error::Retired)?
            .receive(raw, sender);
        if result.is_err() {
            self.cancel();
        }
        result
    }
    pub fn finish(mut self) -> Result<Worker, Error> {
        let executor = self.executor.take().ok_or(Error::Retired)?;
        let (runtime, values) = executor.finish()?;
        for (output, &id) in self.compiled.outputs.iter().enumerate() {
            let mut value = Secret::new(self.worker.program.nodes[id].lanes)?;
            for i in 0..value.first.len() {
                (value.first[i], value.second[i]) = values.word(output, i)?;
            }
            self.worker.values[id] = Some(value);
        }
        // Same last-use ledger as original decoder. No original retirement is
        // skipped, and no outside owner is prematurely destroyed.
        while self.worker.step < self.compiled.end {
            self.worker.retire_inputs();
        }
        self.worker.started = true;
        self.worker.active = super::super::Active::Idle(runtime);
        Ok(self.worker)
    }
}

#[cfg(test)]
pub(crate) fn restored_state(worker: &Worker) -> (usize, bool, usize, bool, bool, Vec<bool>) {
    (
        worker.step,
        worker.started,
        worker.loop_offset,
        worker.inbox.iter().all(Option::is_none),
        matches!(worker.active, super::super::Active::Idle(_)),
        worker.values.iter().map(Option::is_some).collect(),
    )
}
