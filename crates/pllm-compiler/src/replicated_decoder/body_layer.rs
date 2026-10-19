//! Semantic public-body plan. No model/source admission override lives here.
use super::{Register, Resources, RETAINED, TILE, UNUSED};
use crate::replicated_decoder::{
    multiply, puma, tensor, vec_zero, Error, InputBounds, Interval, PublicSource, Secret,
    WeightMode, WorkBudget, MAX_NODES, MAX_PROGRAM_BYTES, MAX_VALUES, ONE,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
    sync::Arc,
};

// Separate coordinator budget, NOT a primitive ordinal or network-round bound.
// This preserves the existing replicated transport's 2^20 advance ceiling.
pub const MAX_COORDINATOR_ADVANCES: u64 = 1 << 20;

#[path = "body_layout.rs"]
mod layout;
pub use layout::Layout;
#[path = "body_machine.rs"]
mod machine;
pub use machine::{Executor, Values};
#[path = "body_families.rs"]
mod families;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowKey {
    pub owner: usize,
    pub offset: usize,
    pub width: usize,
}

/// Read-only, producer-bound complete-row obligation, not a core admission
/// bypass. The parent must bind this plan digest and the exact exported layout.
#[derive(Clone, Copy, Debug)]
pub struct CompleteRow {
    producer: [u8; 32],
    key: RowKey,
    energy: Option<u128>,
    mass: Option<u64>,
}
impl CompleteRow {
    pub fn producer(&self) -> [u8; 32] {
        self.producer
    }
    pub fn key(&self) -> RowKey {
        self.key
    }
    pub fn energy(&self) -> Option<u128> {
        self.energy
    }
    pub fn mass(&self) -> Option<u64> {
        self.mass
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Statistics {
    pub semantic_sites: usize,
    pub register_owners: usize,
    pub range_regions: usize,
    pub tensor_invocations: u64,
    pub tensor_instructions: u64,
    pub matrix_products: u64,
    pub guard_tiles: u64,
    pub logical_public_file_read_bytes: u64,
    pub public_ring64_macs: u64,
    pub public_bias_vector_reads: u64,
}

/// Only the native constructor paths in this module mint certificates. They
/// cannot be supplied as unchecked numeric shortcuts by a manifest caller.
#[derive(Clone, Copy)]
struct Proof {
    key: RowKey,
    energy: Option<u128>,
    mass: Option<u64>,
}
#[derive(Clone, Copy)]
struct Region {
    offset: usize,
    len: usize,
    bounds: Interval,
}

#[derive(Clone)]
pub struct Source {
    owner: usize,
    key: String,
    digest: [u8; 32],
    lanes: usize,
    bounds: Interval,
}
impl Source {
    /// The enclosing source importer must certify this descriptor. Execution
    /// additionally requires an exact enclosing PublicInput descriptor and the
    /// already certified immutable source owner. No supplied norm is trusted.
    pub fn new(
        owner: usize,
        key: String,
        digest: [u8; 32],
        lanes: usize,
        bounds: Interval,
    ) -> Result<Arc<Self>, Error> {
        if key.is_empty()
            || key.len() > 512
            || lanes == 0
            || lanes > super::super::super::MAX_PUBLIC_SOURCE_VALUES
        {
            return Err(Error::Bounds);
        }
        Ok(Arc::new(Self {
            owner,
            key,
            digest,
            lanes,
            bounds,
        }))
    }
    fn hash(&self, hash: &mut Sha256) {
        hash.update((self.owner as u64).to_le_bytes());
        hash.update((self.key.len() as u64).to_le_bytes());
        hash.update(self.key.as_bytes());
        hash.update(self.digest);
        hash.update((self.lanes as u64).to_le_bytes());
        hash.update(self.bounds.lower().to_le_bytes());
        hash.update(self.bounds.upper().to_le_bytes());
    }
}

#[derive(Clone)]
struct MatrixRow {
    left: Layout,
    right: Layout,
    inner: usize,
    columns: usize,
    column_major: bool,
    output: usize,
    tile: usize,
    bounds: Box<[InputBounds]>,
}
impl MatrixRow {
    fn right_index(&self, k: usize, column: usize) -> usize {
        if self.column_major {
            column * self.inner + k
        } else {
            k * self.columns + column
        }
    }
}

enum Op {
    Binary {
        left: Layout,
        right: Layout,
        multiply: bool,
    },
    Tensor {
        input: Layout,
        stride: usize,
        programs: Box<[Arc<tensor::Program>]>,
    },
    Matrix {
        rows: Box<[MatrixRow]>,
        rescale: bool,
    },
    PublicMatrix {
        input: Layout,
        source: Arc<Source>,
        rows: usize,
        inner: usize,
        columns: usize,
        transposed: bool,
    },
    Bias {
        input: Layout,
        source: Arc<Source>,
        width: usize,
    },
    Check {
        input: Layout,
        previous: Option<Layout>,
        accepted: Interval,
        checked: Box<[Arc<tensor::Program>]>,
        valid: Box<[Arc<tensor::Program>]>,
    },
}
impl Op {
    fn layouts(&self) -> Vec<&Layout> {
        match self {
            Self::Binary { left, right, .. } => vec![left, right],
            Self::Tensor { input, .. }
            | Self::PublicMatrix { input, .. }
            | Self::Bias { input, .. } => vec![input],
            Self::Matrix { rows, .. } => rows.iter().flat_map(|r| [&r.left, &r.right]).collect(),
            Self::Check {
                input, previous, ..
            } => std::iter::once(input).chain(previous).collect(),
        }
    }
    fn programs(&self) -> Vec<&Arc<tensor::Program>> {
        match self {
            Self::Tensor { programs, .. } => programs.iter().collect(),
            Self::Check { checked, valid, .. } => checked.iter().chain(valid.iter()).collect(),
            _ => vec![],
        }
    }
}

pub struct Builder {
    inputs: usize,
    registers: Vec<Register>,
    regions: Vec<Box<[Region]>>,
    ops: Vec<Op>,
    proofs: Vec<Proof>,
    templates: BTreeMap<[u8; 32], Arc<tensor::Program>>,
    template_metadata: BTreeMap<[u8; 32], usize>,
    work: WorkBudget,
    advances: u64,
}

pub struct Plan {
    inputs: usize,
    registers: Box<[Register]>,
    regions: Box<[Box<[Region]>]>,
    ops: Box<[Op]>,
    outputs: Box<[Layout]>,
    retire: Box<[Vec<usize>]>,
    unused: Box<[usize]>,
    proofs: Box<[Proof]>,
    digest: [u8; 32],
    work: WorkBudget,
    advances: u64,
    resources: Resources,
}
impl Builder {
    pub fn new(weights: WeightMode, inputs: &[Register]) -> Result<Self, Error> {
        if weights != WeightMode::Public {
            return Err(Error::Binding);
        }
        if inputs.is_empty() || inputs.len() >= MAX_NODES {
            return Err(Error::Bounds);
        }
        for r in inputs {
            if r.lanes == 0 || r.lanes > MAX_VALUES {
                return Err(Error::Shape);
            }
        }
        Ok(Self {
            inputs: inputs.len(),
            registers: inputs.to_vec(),
            regions: inputs
                .iter()
                .map(|r| {
                    Box::from([Region {
                        offset: 0,
                        len: r.lanes,
                        bounds: r.bounds,
                    }])
                })
                .collect(),
            ops: vec![],
            proofs: vec![],
            templates: BTreeMap::new(),
            template_metadata: BTreeMap::new(),
            work: WorkBudget::default(),
            advances: 1,
        })
    }
    pub fn input(&self, owner: usize) -> Result<Layout, Error> {
        if owner >= self.inputs {
            return Err(Error::Binding);
        }
        Layout::contiguous(owner, self.registers[owner].lanes)
    }
    fn bounds(&self, layout: &Layout) -> Result<Interval, Error> {
        layout.validate(&self.registers)?;
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for i in 0..layout.len() {
            let (owner, index) = layout.word(i)?;
            let regions = &self.regions[owner];
            let slot = regions.partition_point(|r| r.offset <= index);
            let range = slot
                .checked_sub(1)
                .and_then(|s| regions.get(s))
                .filter(|r| index - r.offset < r.len)
                .map_or(Interval::new(0, 0)?, |r| r.bounds);
            lo = lo.min(range.lower());
            hi = hi.max(range.upper());
        }
        Interval::new(lo, hi)
    }
    fn intern(
        &mut self,
        program: tensor::Program,
        affine_words: usize,
    ) -> Result<Arc<tensor::Program>, Error> {
        let digest = program.digest();
        // Only our native constructor wrappers call this: RMS owns one lane
        // table, RoPE two, remaining constructors none. Native affine_lanes
        // guarantees exact Vec capacity. No private core field introspection.
        let metadata = size_of::<tensor::Program>()
            + 2 * size_of::<usize>()
            + affine_words * size_of::<multiply::arithmetic::Affine>();
        if let Some(p) = self.templates.get(&digest) {
            if self.template_metadata[&digest] != metadata {
                return Err(Error::Binding);
            }
            return Ok(p.clone());
        }
        let bytes = self
            .template_metadata
            .values()
            .try_fold(metadata, |n, p| n.checked_add(*p).ok_or(Error::Bounds))?;
        if bytes > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        let program = Arc::new(program);
        self.templates.insert(digest, program.clone());
        self.template_metadata.insert(digest, metadata);
        Ok(program)
    }
    fn push(
        &mut self,
        op: Op,
        lanes: usize,
        regions: Vec<Region>,
        work: WorkBudget,
        advances: u64,
    ) -> Result<Layout, Error> {
        if lanes == 0 || lanes > MAX_VALUES || self.registers.len() >= MAX_NODES {
            return Err(Error::Bounds);
        }
        for input in op.layouts() {
            input.validate(&self.registers)?;
        }
        let (mut lo, mut hi, mut previous) = (0_i64, 0_i64, 0_usize);
        for r in &regions {
            if r.len == 0
                || r.offset < previous
                || r.offset.checked_add(r.len).is_none_or(|n| n > lanes)
            {
                return Err(Error::Shape);
            }
            previous = r.offset + r.len;
            lo = lo.min(r.bounds.lower());
            hi = hi.max(r.bounds.upper());
        }
        self.work = self.work.checked_add(work)?;
        self.work.limits()?;
        self.advances = self.advances.checked_add(advances).ok_or(Error::Bounds)?;
        let owner = self.registers.len();
        self.registers
            .try_reserve(1)
            .map_err(|_| Error::Allocation)?;
        self.regions.try_reserve(1).map_err(|_| Error::Allocation)?;
        self.ops.try_reserve(1).map_err(|_| Error::Allocation)?;
        self.registers.push(Register {
            lanes,
            bounds: Interval::new(lo, hi)?,
        });
        self.regions.push(regions.into_boxed_slice());
        self.ops.push(op);
        let retained = self
            .regions
            .iter()
            .map(|r| r.len() * size_of::<Region>())
            .sum::<usize>()
            + self.registers.len() * size_of::<Register>()
            + self.ops.len() * size_of::<Op>();
        if retained > MAX_PROGRAM_BYTES {
            return Err(Error::Bounds);
        }
        Layout::contiguous(owner, lanes)
    }
    fn proof(&self, row: &Layout) -> Option<Proof> {
        let key = row.row_key(row.len())?;
        self.proofs.iter().copied().find(|p| p.key == key)
    }
    pub fn finish(self, outputs: &[Layout]) -> Result<Arc<Plan>, Error> {
        Plan::finish(self, outputs)
    }
}

impl Plan {
    fn finish(builder: Builder, outputs: &[Layout]) -> Result<Arc<Self>, Error> {
        builder.work.limits()?;
        if builder.advances > MAX_COORDINATOR_ADVANCES {
            return Err(Error::Bounds);
        }
        if builder.ops.is_empty() || outputs.is_empty() || outputs.len() > MAX_NODES {
            return Err(Error::Shape);
        }
        outputs.iter().try_fold(0usize, |n, l| {
            n.checked_add(l.len())
                .filter(|&n| n <= MAX_VALUES)
                .ok_or(Error::Bounds)
        })?;
        let count = builder.registers.len();
        let mut last = vec_zero::<usize>(count)?;
        last.fill(UNUSED);
        for (step, op) in builder.ops.iter().enumerate() {
            for layout in op.layouts() {
                for owner in layout.owners() {
                    last[owner] = step;
                }
            }
            last[builder.inputs + step] = step;
        }
        for output in outputs {
            output.validate(&builder.registers)?;
            for owner in output.owners() {
                last[owner] = RETAINED;
            }
        }
        let mut retire = Vec::new();
        retire
            .try_reserve_exact(builder.ops.len())
            .map_err(|_| Error::Allocation)?;
        retire.resize_with(builder.ops.len(), Vec::new);
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
        let bytes = |n: usize| n.checked_mul(16).ok_or(Error::Bounds);
        let mut live = builder.registers[..builder.inputs]
            .iter()
            .try_fold(0usize, |n, r| {
                n.checked_add(bytes(r.lanes)?).ok_or(Error::Bounds)
            })?;
        let mut peak = live;
        for &owner in &unused {
            live -= bytes(builder.registers[owner].lanes)?;
        }
        for (step, r) in builder.registers[builder.inputs..].iter().enumerate() {
            live = live.checked_add(bytes(r.lanes)?).ok_or(Error::Bounds)?;
            peak = peak.max(live);
            for &owner in &retire[step] {
                live -= bytes(builder.registers[owner].lanes)?;
            }
        }
        let mut layouts = BTreeSet::new();
        let mut layout_bytes = 0usize;
        let mut source_ids = BTreeSet::new();
        let mut source_bytes = 0usize;
        let mut op_bytes = builder.ops.len() * size_of::<Op>();
        let mut scratch = puma::MAX_ADMITTED_BYTES;
        let mut gathered = 2 * puma::MAX_LANES * 16 + 4 * TILE * 16;
        let mut hash = Sha256::new();
        hash.update(b"pllm.puma.semantic-body.v1");
        hash.update((builder.inputs as u64).to_le_bytes());
        for r in &builder.registers[..builder.inputs] {
            hash.update((r.lanes as u64).to_le_bytes());
            hash.update(r.bounds.lower().to_le_bytes());
            hash.update(r.bounds.upper().to_le_bytes());
        }
        hash.update((builder.ops.len() as u64).to_le_bytes());
        for op in &builder.ops {
            for l in op.layouts() {
                if layouts.insert(l.identity()) {
                    layout_bytes += l.metadata_bytes();
                }
            }
            match op {
                Op::Binary {
                    left,
                    right,
                    multiply,
                } => {
                    hash.update([0, u8::from(*multiply)]);
                    left.hash(&mut hash);
                    right.hash(&mut hash);
                }
                Op::Tensor {
                    input,
                    stride,
                    programs,
                } => {
                    hash.update([1]);
                    input.hash(&mut hash);
                    hash.update((*stride as u64).to_le_bytes());
                    hash.update((programs.len() as u64).to_le_bytes());
                    op_bytes += programs.len() * size_of::<Arc<tensor::Program>>();
                    for p in programs.iter() {
                        hash.update(p.digest());
                        scratch = scratch.max(p.admitted_bytes());
                    }
                }
                Op::Matrix { rows, rescale } => {
                    hash.update([2, u8::from(*rescale)]);
                    hash.update((rows.len() as u64).to_le_bytes());
                    op_bytes += rows.len() * size_of::<MatrixRow>();
                    for r in rows.iter() {
                        op_bytes += r.bounds.len() * size_of::<InputBounds>();
                        r.left.hash(&mut hash);
                        r.right.hash(&mut hash);
                        for v in [
                            r.inner,
                            r.columns,
                            r.output,
                            r.tile,
                            usize::from(r.column_major),
                        ] {
                            hash.update((v as u64).to_le_bytes());
                        }
                    }
                }
                Op::PublicMatrix {
                    input,
                    source,
                    rows,
                    inner,
                    columns,
                    transposed,
                } => {
                    hash.update([3, u8::from(*transposed)]);
                    input.hash(&mut hash);
                    source.hash(&mut hash);
                    for n in [rows, inner, columns] {
                        hash.update((*n as u64).to_le_bytes());
                    }
                    scratch = scratch.max(65536);
                    if !input.row_key(input.len()).is_some_and(|r| {
                        r.offset == 0 && builder.registers[r.owner].lanes == r.width
                    }) {
                        gathered = gathered.max(input.len() * 16);
                    }
                }
                Op::Bias {
                    input,
                    source,
                    width,
                } => {
                    hash.update([4]);
                    input.hash(&mut hash);
                    source.hash(&mut hash);
                    hash.update((*width as u64).to_le_bytes());
                    scratch = scratch.max(width * 8);
                }
                Op::Check {
                    input,
                    previous,
                    accepted,
                    checked,
                    valid,
                } => {
                    hash.update([5, u8::from(previous.is_some())]);
                    input.hash(&mut hash);
                    if let Some(p) = previous {
                        p.hash(&mut hash);
                    }
                    hash.update(accepted.lower().to_le_bytes());
                    hash.update(accepted.upper().to_le_bytes());
                    hash.update((checked.len() as u64).to_le_bytes());
                    op_bytes += (checked.len() + valid.len()) * size_of::<Arc<tensor::Program>>();
                    for p in checked.iter().chain(valid.iter()) {
                        hash.update(p.digest());
                        scratch = scratch.max(p.admitted_bytes());
                    }
                }
            }
            if let Op::PublicMatrix { source, .. } | Op::Bias { source, .. } = op {
                if source_ids.insert(Arc::as_ptr(source) as usize) {
                    source_bytes += size_of::<Source>() + source.key.capacity() + 16;
                }
            }
        }
        hash.update((outputs.len() as u64).to_le_bytes());
        for output in outputs {
            output.hash(&mut hash);
            if layouts.insert(output.identity()) {
                layout_bytes += output.metadata_bytes();
            }
        }
        // Immutable template payload is separate from active tensor workspace.
        let template_bytes = builder.template_metadata.values().sum::<usize>();
        let metadata = size_of::<Self>()
            + count * (size_of::<Register>() + size_of::<Box<[Region]>>())
            + builder
                .regions
                .iter()
                .map(|r| r.len() * size_of::<Region>())
                .sum::<usize>()
            + op_bytes
            + layout_bytes
            + source_bytes
            + std::mem::size_of_val(outputs)
            + retire.len() * size_of::<Vec<usize>>()
            + retire
                .iter()
                .map(|r| r.capacity() * size_of::<usize>())
                .sum::<usize>()
            + unused.len() * size_of::<usize>()
            + builder.proofs.len() * size_of::<Proof>()
            + template_bytes;
        let allocation_allowance = 64
            * (layouts.len()
                + source_ids.len() * 2
                + builder.templates.len() * 3
                + count * 4
                + builder.ops.len() * 4
                + 8);
        let mut resources = Resources {
            immutable_metadata_bytes: metadata,
            register_storage_bytes: count * size_of::<Option<Secret>>(),
            constructor_slot_bytes: builder.inputs * size_of::<Secret>(),
            peak_live_secret_bytes: peak,
            gathered_operand_bytes: gathered,
            primitive_ledger_allowance_bytes: scratch + puma::MAX_ADMITTED_BYTES,
            bounds_stack_bytes: TILE * size_of::<InputBounds>(),
            owner_and_arc_allowance_bytes: size_of::<Executor>()
                + size_of::<Values>()
                + 65536
                + 128
                + allocation_allowance,
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
        Ok(Arc::new(Self {
            inputs: builder.inputs,
            registers: builder.registers.into_boxed_slice(),
            regions: builder.regions.into_boxed_slice(),
            ops: builder.ops.into_boxed_slice(),
            outputs: outputs.into(),
            retire: retire.into_boxed_slice(),
            unused: unused.into_boxed_slice(),
            proofs: builder.proofs.into_boxed_slice(),
            digest: hash.finalize().into(),
            work: builder.work,
            advances: builder.advances,
            resources,
        }))
    }
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn work_budget(&self) -> WorkBudget {
        self.work
    }
    pub fn resources(&self) -> Resources {
        self.resources
    }
    pub fn coordinator_advances(&self) -> u64 {
        self.advances
    }
    pub fn input_count(&self) -> usize {
        self.inputs
    }
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
    pub fn input_specs(&self) -> &[Register] {
        &self.registers[..self.inputs]
    }
    pub fn output_layout(&self, index: usize) -> Result<&Layout, Error> {
        self.outputs.get(index).ok_or(Error::Shape)
    }
    pub fn exported_lanes(&self) -> usize {
        self.outputs.iter().map(Layout::len).sum()
    }
    pub fn complete_row(&self, layout: &Layout) -> Option<CompleteRow> {
        let key = layout.row_key(layout.len())?;
        let proof = self.proofs.iter().find(|p| p.key == key)?;
        Some(CompleteRow {
            producer: self.digest,
            key,
            energy: proof.energy,
            mass: proof.mass,
        })
    }
    pub fn statistics(&self) -> Statistics {
        let mut s = Statistics {
            semantic_sites: self.ops.len(),
            register_owners: self.registers.len(),
            range_regions: self.regions.iter().map(|r| r.len()).sum(),
            tensor_invocations: 0,
            tensor_instructions: 0,
            matrix_products: 0,
            guard_tiles: 0,
            logical_public_file_read_bytes: 0,
            public_ring64_macs: 0,
            public_bias_vector_reads: 0,
        };
        for op in self.ops.iter() {
            for p in op.programs() {
                s.tensor_invocations += 1;
                s.tensor_instructions += p.instructions() as u64;
            }
            match op {
                Op::Matrix { rows, .. } => {
                    s.matrix_products += rows
                        .iter()
                        .map(|r| r.columns.div_ceil(r.tile) as u64)
                        .sum::<u64>()
                }
                Op::Check { checked, .. } => s.guard_tiles += checked.len() as u64,
                Op::PublicMatrix {
                    source,
                    rows,
                    inner,
                    columns,
                    ..
                } => {
                    s.logical_public_file_read_bytes += source.lanes as u64 * 8;
                    s.public_ring64_macs +=
                        2 * (*rows as u64) * (*inner as u64) * (*columns as u64);
                }
                Op::Bias { width, .. } => {
                    s.logical_public_file_read_bytes += *width as u64 * 8;
                    s.public_bias_vector_reads += 1;
                }
                _ => (),
            }
        }
        s
    }
}

#[cfg(test)]
pub(in crate::replicated_decoder) use machine::{from_worker, try_from_worker};
