//! Consuming semantic-site interpreter. One primitive/tensor invocation at a time.
use super::*;
use crate::replicated_decoder::runtime::{PeerBytes, arithmetic_frames, tensor_frames};
use crate::replicated_decoder::source_registry::SourceRegistry;
use multiply::{Operand, PartyRuntime, Status};
use puma::{Message, MessagePhase, Operation, Party, ReplicatedShare};

enum ProductTarget {
    Binary {
        offset: usize,
        len: usize,
    },
    Matrix {
        row: usize,
        column: usize,
        len: usize,
    },
    Sticky {
        len: usize,
    },
}
enum TensorTarget {
    Row { row: usize },
    Map { offset: usize },
    Checked,
    AllValid,
}
// Fixed one-owner storage is included in Plan::resources. Boxing would add a
// fallible per-invocation allocation and change the audited serving overlap.
#[allow(clippy::large_enum_variant)]
enum Active {
    Ready(PartyRuntime),
    Product(PartyRuntime, ProductTarget),
    Tensor(tensor::Executor, TensorTarget),
    Retired,
}

pub struct Executor {
    plan: Arc<Plan>,
    values: Vec<Option<Secret>>,
    public: Option<Arc<PublicSource>>,
    active: Active,
    inbox: [Option<Message>; 2],
    pending: Option<ReplicatedShare>,
    step: usize,
    cursor: usize,
    column: usize,
    check_phase: u8,
    pid: u32,
    #[cfg(test)]
    observed_peak: usize,
}

/// Borrowed only during construction. Neither contract nor its metadata backing
/// is retained by the consuming executor; the original PublicSource Arc is.
enum SourceContract<'a, 'metadata> {
    Expanded(Option<&'a crate::replicated_decoder::Program>),
    Registry(&'a SourceRegistry<'metadata>),
}

impl SourceContract<'_, '_> {
    fn check_files(&self, public: &PublicSource) -> Result<(), Error> {
        match self {
            Self::Expanded(program) => public.check(program.ok_or(Error::Binding)?),
            Self::Registry(registry) => public.check_registry(registry),
        }
    }

    fn check_descriptor(&self, source: &Source) -> Result<(), Error> {
        match self {
            Self::Expanded(program) => {
                let program = program.ok_or(Error::Binding)?;
                let node = program.recipe_at(source.owner).ok_or(Error::Binding)?;
                let crate::replicated_decoder::Code::PublicInput { key, digest, .. } = &node.code
                else {
                    return Err(Error::Binding);
                };
                if key != &source.key
                    || digest != &source.digest
                    || node.lanes != source.lanes
                    || node.bounds != source.bounds
                {
                    return Err(Error::Binding);
                }
                Ok(())
            }
            Self::Registry(registry) => registry.check_descriptor(
                source.owner,
                &source.key,
                source.digest,
                source.lanes,
                source.bounds,
            ),
        }
    }
}

impl Executor {
    /// Caller binds plan.digest() in its outer program and verifies source owner
    /// against that program before lending this immutable Arc. No seed/context is
    /// replaced and no per-site PRSS cursor is reset here.
    pub fn new(
        plan: Arc<Plan>,
        runtime: PartyRuntime,
        inputs: Vec<Secret>,
        public: Option<Arc<PublicSource>>,
        source_program: Option<&crate::replicated_decoder::Program>,
    ) -> Result<Self, Error> {
        Self::new_with_contract(
            plan,
            runtime,
            inputs,
            public,
            SourceContract::Expanded(source_program),
        )
    }

    /// Validate original source metadata and authenticated files without an
    /// expanded Program. The enclosing compiler still binds the plan identity,
    /// checks complete-request admission and globally drains the router. No
    /// unchecked body Source descriptor or registry backing is retained here.
    pub(in crate::replicated_decoder) fn new_with_registry(
        plan: Arc<Plan>,
        runtime: PartyRuntime,
        inputs: Vec<Secret>,
        public: Option<Arc<PublicSource>>,
        registry: &SourceRegistry<'_>,
    ) -> Result<Self, Error> {
        Self::new_with_contract(
            plan,
            runtime,
            inputs,
            public,
            SourceContract::Registry(registry),
        )
    }

    fn new_with_contract(
        plan: Arc<Plan>,
        runtime: PartyRuntime,
        inputs: Vec<Secret>,
        public: Option<Arc<PublicSource>>,
        contract: SourceContract<'_, '_>,
    ) -> Result<Self, Error> {
        if runtime.status() != Status::Ready || inputs.len() != plan.inputs {
            return Err(Error::Phase);
        }
        if plan
            .ops
            .iter()
            .any(|op| matches!(op, Op::PublicMatrix { .. } | Op::Bias { .. }))
        {
            contract.check_files(public.as_ref().ok_or(Error::Binding)?)?;
        }
        for op in plan.ops.iter() {
            if let Op::PublicMatrix { source, .. } | Op::Bias { source, .. } = op {
                contract.check_descriptor(source)?;
                let file = public.as_ref().ok_or(Error::Binding)?.file(&source.key)?;
                if file.digest() != source.digest || file.size() != source.lanes as u64 * 8 {
                    return Err(Error::Binding);
                }
            }
        }
        #[cfg(test)]
        let observed_peak = inputs.iter().map(|s| s.first.len() * 16).sum();
        let mut values = Vec::new();
        values
            .try_reserve_exact(plan.registers.len())
            .map_err(|_| Error::Allocation)?;
        for (value, spec) in inputs.into_iter().zip(plan.registers.iter()) {
            if value.first.len() != spec.lanes || value.second.len() != spec.lanes {
                return Err(Error::Shape);
            }
            values.push(Some(value));
        }
        values.resize_with(plan.registers.len(), || None);
        for &owner in plan.unused.iter() {
            values[owner] = None;
        }
        Ok(Self {
            plan,
            values,
            public,
            active: Active::Ready(runtime),
            inbox: [None, None],
            pending: None,
            step: 0,
            cursor: 0,
            column: 0,
            check_phase: 0,
            pid: std::process::id(),
            #[cfg(test)]
            observed_peak,
        })
    }
    pub fn complete(&self) -> bool {
        self.step == self.plan.ops.len() && matches!(self.active, Active::Ready(_))
    }
    pub fn cancel(&mut self) {
        self.active = Active::Retired;
        self.values.clear();
        self.public = None;
        self.pending = None;
        self.inbox = [None, None];
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
    fn word(&self, layout: &Layout, index: usize) -> Result<(u64, u64), Error> {
        let (owner, index) = layout.word(index)?;
        let value = self
            .values
            .get(owner)
            .and_then(Option::as_ref)
            .ok_or(Error::Binding)?;
        Ok((
            *value.first.get(index).ok_or(Error::Shape)?,
            *value.second.get(index).ok_or(Error::Shape)?,
        ))
    }
    fn range(&self, layout: &Layout) -> Result<Interval, Error> {
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for i in 0..layout.len() {
            let (owner, index) = layout.word(i)?;
            let regions = &self.plan.regions[owner];
            let slot = regions.partition_point(|r| r.offset <= index);
            let range = slot
                .checked_sub(1)
                .and_then(|n| regions.get(n))
                .filter(|r| index - r.offset < r.len)
                .map_or(Interval::new(0, 0)?, |r| r.bounds);
            lo = lo.min(range.lower());
            hi = hi.max(range.upper());
        }
        Interval::new(lo, hi)
    }
    fn gather(&self, layout: &Layout, start: usize, len: usize) -> Result<Secret, Error> {
        if len == 0
            || len > puma::MAX_LANES
            || start.checked_add(len).is_none_or(|n| n > layout.len())
        {
            return Err(Error::Shape);
        }
        let mut value = Secret::new(len)?;
        for i in 0..len {
            (value.first[i], value.second[i]) = self.word(layout, start + i)?;
        }
        Ok(value)
    }
    fn share(runtime: &PartyRuntime, value: &Secret) -> Result<ReplicatedShare, Error> {
        Self::share_parts(runtime, &value.first, &value.second)
    }
    fn share_parts(
        runtime: &PartyRuntime,
        first: &[u64],
        second: &[u64],
    ) -> Result<ReplicatedShare, Error> {
        let operation = Operation::new(
            runtime.context(),
            0,
            Party::P0,
            first.len(),
            1,
            puma::Limits::default(),
        )?;
        ReplicatedShare::from_local(
            operation,
            runtime.party(),
            first,
            second,
            puma::Limits::default(),
        )
    }
    fn store(&mut self, offset: usize, value: &ReplicatedShare) -> Result<(), Error> {
        let owner = self.plan.inputs + self.step;
        let output = self.values[owner].as_mut().ok_or(Error::Phase)?;
        let end = offset
            .checked_add(value.first().len())
            .ok_or(Error::Bounds)?;
        output
            .first
            .get_mut(offset..end)
            .ok_or(Error::Shape)?
            .copy_from_slice(value.first());
        output
            .second
            .get_mut(offset..end)
            .ok_or(Error::Shape)?
            .copy_from_slice(value.second());
        Ok(())
    }
    fn next(&mut self) {
        #[cfg(test)]
        self.observe();
        for &owner in self.plan.retire[self.step].iter() {
            self.values[owner] = None;
        }
        self.step += 1;
        self.cursor = 0;
        self.column = 0;
        self.check_phase = 0;
    }
    fn finish_check_tile(&mut self, len: usize) -> Result<(), Error> {
        self.cursor = self.cursor.checked_add(len).ok_or(Error::Bounds)?;
        self.check_phase = 0;
        let Op::Check { input, .. } = &self.plan.ops[self.step] else {
            return Err(Error::Binding);
        };
        if self.cursor == input.len() {
            self.next();
        }
        Ok(())
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
            Active::Product(mut runtime, target) => {
                let value = runtime.take_output()?;
                match target {
                    ProductTarget::Binary { offset, len } => {
                        self.store(offset, &value)?;
                        self.cursor = offset + len;
                        if self.cursor == self.plan.registers[self.plan.inputs + self.step].lanes {
                            self.next();
                        }
                    }
                    ProductTarget::Matrix { row, column, len } => {
                        let Op::Matrix { rows, .. } = &self.plan.ops[self.step] else {
                            return Err(Error::Binding);
                        };
                        let offset = rows[row].output + column;
                        let columns = rows[row].columns;
                        let count = rows.len();
                        self.store(offset, &value)?;
                        self.column = column + len;
                        if self.column == columns {
                            self.cursor = row + 1;
                            self.column = 0;
                        }
                        if self.cursor == count {
                            self.next();
                        }
                    }
                    ProductTarget::Sticky { len } => {
                        let Op::Check { input, .. } = &self.plan.ops[self.step] else {
                            return Err(Error::Binding);
                        };
                        let offset = input.len();
                        self.store(offset, &value)?;
                        self.finish_check_tile(len)?;
                    }
                }
                runtime
            }
            Active::Tensor(mut executor, target) => {
                executor.finish_step()?;
                if !executor.complete() {
                    let frames = executor.start_step()?;
                    self.active = Active::Tensor(executor, target);
                    return tensor_frames(frames);
                }
                match target {
                    TensorTarget::Row { row } => {
                        let (runtime, value) = executor.finish()?;
                        let Op::Tensor {
                            stride, programs, ..
                        } = &self.plan.ops[self.step]
                        else {
                            return Err(Error::Binding);
                        };
                        let offset = row * stride;
                        let count = programs.len();
                        self.store(offset, &value)?;
                        self.cursor = row + 1;
                        if self.cursor == count {
                            self.next();
                        }
                        runtime
                    }
                    TensorTarget::Map { offset } => {
                        let (runtime, value) = executor.finish()?;
                        let Op::PreparedTensorMap { prepared, .. } = &self.plan.ops[self.step]
                        else {
                            return Err(Error::Binding);
                        };
                        let lanes = prepared.lanes();
                        let len = prepared.program_at(offset)?.lanes();
                        if value.first().len() != len || value.second().len() != len {
                            return Err(Error::Shape);
                        }
                        self.store(offset, &value)?;
                        self.cursor = offset.checked_add(len).ok_or(Error::Bounds)?;
                        if self.cursor == lanes {
                            self.next();
                        }
                        runtime
                    }
                    TensorTarget::Checked => {
                        let (runtime, safe, valid) = executor.finish_checked()?;
                        self.store(self.cursor, &safe)?;
                        self.pending = Some(valid);
                        self.check_phase = 1;
                        runtime
                    }
                    TensorTarget::AllValid => {
                        let (runtime, value) = executor.finish()?;
                        let Op::Check {
                            input, previous, ..
                        } = &self.plan.ops[self.step]
                        else {
                            return Err(Error::Binding);
                        };
                        let n = tensor::MAX_TILE_LANES.min(input.len() - self.cursor);
                        let sticky_offset = input.len();
                        if self.cursor > 0 || previous.is_some() {
                            self.pending = Some(value);
                            self.check_phase = 2;
                        } else {
                            self.store(sticky_offset, &value)?;
                            self.finish_check_tile(n)?;
                        }
                        runtime
                    }
                }
            }
            Active::Retired => return Err(Error::Retired),
        };
        while self.step < self.plan.ops.len() {
            let owner = self.plan.inputs + self.step;
            if self.values[owner].is_none()
                && !matches!(self.plan.ops[self.step], Op::PublicMatrix { .. })
            {
                self.values[owner] = Some(Secret::new(self.plan.registers[owner].lanes)?);
            }
            #[cfg(test)]
            self.observe();
            match &self.plan.ops[self.step] {
                Op::Binary {
                    left,
                    right,
                    multiply,
                } => {
                    let left = left.clone();
                    let right = right.clone();
                    let multiply = *multiply;
                    if !multiply {
                        for i in 0..left.len() {
                            let (a, b) = self.word(&left, i)?;
                            let (c, d) = self.word(&right, i)?;
                            let out = self.values[owner].as_mut().ok_or(Error::Phase)?;
                            out.first[i] = a.wrapping_add(c);
                            out.second[i] = b.wrapping_add(d);
                        }
                        self.next();
                        continue;
                    }
                    let n = TILE.min(left.len() - self.cursor);
                    let a = self.gather(&left, self.cursor, n)?;
                    let b = self.gather(&right, self.cursor, n)?;
                    let bounds = InputBounds {
                        left: self.range(&left.slice(self.cursor, n)?)?,
                        right: self.range(&right.slice(self.cursor, n)?)?,
                    };
                    let plan = runtime.plan(&[bounds; TILE][..n])?;
                    let frames = begin(&mut runtime, &plan, &a, &b)?;
                    self.active = Active::Product(
                        runtime,
                        ProductTarget::Binary {
                            offset: self.cursor,
                            len: n,
                        },
                    );
                    return Ok(frames);
                }
                Op::Tensor {
                    input,
                    stride,
                    programs,
                } => {
                    let program = programs.get(self.cursor).ok_or(Error::Phase)?.clone();
                    let value = self.gather(input, self.cursor * stride, program.lanes())?;
                    let input = Self::share(&runtime, &value)?;
                    let mut executor = tensor::Executor::new_shared(runtime, program, input)?;
                    let frames = executor.start_step()?;
                    self.active = Active::Tensor(executor, TensorTarget::Row { row: self.cursor });
                    return tensor_frames(frames);
                }
                Op::PreparedTensorMap { input, prepared } => {
                    let program = prepared.program_at(self.cursor)?.clone();
                    let value = self.gather(input, self.cursor, program.lanes())?;
                    let input = Self::share(&runtime, &value)?;
                    let mut executor = tensor::Executor::new_shared(runtime, program, input)?;
                    let frames = executor.start_step()?;
                    self.active = Active::Tensor(
                        executor,
                        TensorTarget::Map {
                            offset: self.cursor,
                        },
                    );
                    return tensor_frames(frames);
                }
                Op::Matrix { rows, rescale } => {
                    let row = &rows[self.cursor];
                    let n = row.tile.min(row.columns - self.column);
                    let a = self.gather(&row.left, 0, row.inner)?;
                    let mut b = Secret::new(row.inner * n)?;
                    for k in 0..row.inner {
                        for c in 0..n {
                            (b.first[k * n + c], b.second[k * n + c]) =
                                self.word(&row.right, row.right_index(k, self.column + c))?;
                        }
                    }
                    let shape = crate::replicated_decoder::MatrixShape::new(1, row.inner, n)?;
                    let bounds = row.bounds[self.column / row.tile];
                    let plan = if *rescale {
                        runtime.plan_matmul(shape, bounds)?
                    } else {
                        runtime.plan_ring_matmul(shape, bounds)?
                    };
                    let frames = begin(&mut runtime, &plan, &a, &b)?;
                    self.active = Active::Product(
                        runtime,
                        ProductTarget::Matrix {
                            row: self.cursor,
                            column: self.column,
                            len: n,
                        },
                    );
                    return Ok(frames);
                }
                Op::PublicMatrix {
                    input,
                    source,
                    rows,
                    inner,
                    columns,
                    transposed,
                } => {
                    let input = input.clone();
                    let source = source.clone();
                    let (rows, inner, columns, transposed) = (*rows, *inner, *columns, *transposed);
                    let public = self.public.as_ref().ok_or(Error::Binding)?.clone();
                    let file = public.file(&source.key)?;
                    let direct = input
                        .row_key(input.len())
                        .filter(|r| r.offset == 0 && self.plan.registers[r.owner].lanes == r.width);
                    let output = if let Some(row) = direct {
                        crate::replicated_decoder::public_source::multiply(
                            file,
                            self.values[row.owner].as_ref().ok_or(Error::Binding)?,
                            rows,
                            inner,
                            columns,
                            transposed,
                        )?
                    } else {
                        let mut gathered = Secret::new(input.len())?;
                        for i in 0..input.len() {
                            (gathered.first[i], gathered.second[i]) = self.word(&input, i)?;
                        }
                        crate::replicated_decoder::public_source::multiply(
                            file, &gathered, rows, inner, columns, transposed,
                        )?
                    };
                    self.values[owner] = Some(output);
                    self.next();
                }
                Op::Bias {
                    input,
                    source,
                    width,
                } => {
                    let input = input.clone();
                    let source = source.clone();
                    let width = *width;
                    let public = self.public.as_ref().ok_or(Error::Binding)?.clone();
                    let file = public.file(&source.key)?;
                    let mut raw = vec_zero::<u8>(width * 8)?;
                    file.read(0, &mut raw).map_err(|_| Error::Binding)?;
                    for i in 0..input.len() {
                        let at = i % width * 8;
                        let value = i64::from_le_bytes(
                            raw[at..at + 8].try_into().map_err(|_| Error::Shape)?,
                        );
                        if value < source.bounds.lower() || value > source.bounds.upper() {
                            return Err(Error::Binding);
                        }
                        let (a, b) = self.word(&input, i)?;
                        let out = self.values[owner].as_mut().ok_or(Error::Phase)?;
                        out.first[i] = a.wrapping_add(if runtime.party() == Party::P0 {
                            value as u64
                        } else {
                            0
                        });
                        out.second[i] = b.wrapping_add(if runtime.party().next() == Party::P0 {
                            value as u64
                        } else {
                            0
                        });
                    }
                    self.next();
                }
                Op::Check {
                    input,
                    previous,
                    checked,
                    valid,
                    ..
                } => {
                    let n = tensor::MAX_TILE_LANES.min(input.len() - self.cursor);
                    let tile = self.cursor / tensor::MAX_TILE_LANES;
                    match self.check_phase {
                        0 => {
                            let program = checked[tile].clone();
                            let value = self.gather(input, self.cursor, n)?;
                            let input = Self::share(&runtime, &value)?;
                            let mut executor =
                                tensor::Executor::new_shared(runtime, program, input)?;
                            let frames = executor.start_step()?;
                            self.active = Active::Tensor(executor, TensorTarget::Checked);
                            return tensor_frames(frames);
                        }
                        1 => {
                            let value = self.pending.take().ok_or(Error::Phase)?;
                            // Original CheckedRange -> Gather(valid) -> Tensor
                            // rebinds ordinal-zero input metadata before the
                            // next invocation. Preserve that boundary too.
                            let input = Self::share_parts(&runtime, value.first(), value.second())?;
                            drop(value);
                            let mut executor =
                                tensor::Executor::new_shared(runtime, valid[tile].clone(), input)?;
                            let frames = executor.start_step()?;
                            self.active = Active::Tensor(executor, TensorTarget::AllValid);
                            return tensor_frames(frames);
                        }
                        2 => {
                            let value = self.pending.take().ok_or(Error::Phase)?;
                            let (a, b) = if self.cursor == 0 {
                                self.word(previous.as_ref().ok_or(Error::Binding)?, 0)?
                            } else {
                                let output = self.values[owner].as_ref().ok_or(Error::Phase)?;
                                (output.first[input.len()], output.second[input.len()])
                            };
                            let bounds = InputBounds {
                                left: Interval::new(0, 1)?,
                                right: Interval::new(0, 1)?,
                            };
                            let plan = runtime.plan_ring_product(&[bounds])?;
                            let party = runtime.party();
                            let left = plan.share_operand(
                                Operand::Left,
                                party,
                                &[a],
                                &[b],
                                puma::Limits::default(),
                            )?;
                            let right = plan.share_operand(
                                Operand::Right,
                                party,
                                value.first(),
                                value.second(),
                                puma::Limits::default(),
                            )?;
                            let frames =
                                arithmetic_frames([Some(runtime.begin(left, right)?), None, None])?;
                            self.active =
                                Active::Product(runtime, ProductTarget::Sticky { len: n });
                            return Ok(frames);
                        }
                        _ => return Err(Error::Phase),
                    }
                }
            }
        }
        self.active = Active::Ready(runtime);
        Ok(vec![])
    }
    pub fn receive(&mut self, raw: &[u8], sender: Party) -> Result<Vec<(Party, PeerBytes)>, Error> {
        let result = (|| {
            self.live()?;
            match &mut self.active {
                Active::Tensor(executor, _) => {
                    tensor_frames(executor.receive_encoded(raw, sender)?)
                }
                Active::Product(runtime, _) => {
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
                }
                _ => Err(Error::Phase),
            }
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
                plan: self.plan.clone(),
                values: std::mem::take(&mut self.values),
            },
        ))
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn live_bytes(&self) -> usize {
        self.values
            .iter()
            .flatten()
            .map(|v| v.first.len() * 16)
            .sum()
    }
    #[cfg(test)]
    fn observe(&mut self) {
        self.observed_peak = self.observed_peak.max(self.live_bytes());
    }
    #[cfg(test)]
    pub(in crate::replicated_decoder) fn observed_register_peak(&self) -> usize {
        self.observed_peak
    }
}
impl Drop for Executor {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn begin(
    runtime: &mut PartyRuntime,
    plan: &multiply::Plan,
    a: &Secret,
    b: &Secret,
) -> Result<Vec<(Party, PeerBytes)>, Error> {
    let party = runtime.party();
    let left = plan.share_operand(
        Operand::Left,
        party,
        &a.first,
        &a.second,
        puma::Limits::default(),
    )?;
    let right = plan.share_operand(
        Operand::Right,
        party,
        &b.first,
        &b.second,
        puma::Limits::default(),
    )?;
    arithmetic_frames([Some(runtime.begin(left, right)?), None, None])
}

pub struct Values {
    plan: Arc<Plan>,
    values: Vec<Option<Secret>>,
}
impl Values {
    pub fn word(&self, output: usize, index: usize) -> Result<(u64, u64), Error> {
        let layout = self.plan.outputs.get(output).ok_or(Error::Shape)?;
        let (owner, index) = layout.word(index)?;
        let value = self.values[owner].as_ref().ok_or(Error::Binding)?;
        Ok((value.first[index], value.second[index]))
    }
    pub fn output_len(&self, output: usize) -> Result<usize, Error> {
        Ok(self.plan.outputs.get(output).ok_or(Error::Shape)?.len())
    }
    pub fn word_flat(&self, mut index: usize) -> Result<(u64, u64), Error> {
        for (i, layout) in self.plan.outputs.iter().enumerate() {
            if index < layout.len() {
                return self.word(i, index);
            }
            index -= layout.len();
        }
        Err(Error::Shape)
    }
}

#[cfg(test)]
pub(in crate::replicated_decoder) fn from_worker(
    plan: Arc<Plan>,
    worker: crate::replicated_decoder::Worker,
    public: Option<Arc<PublicSource>>,
) -> Executor {
    try_from_worker(plan, worker, public).unwrap()
}
#[cfg(test)]
pub(in crate::replicated_decoder) fn try_from_worker(
    plan: Arc<Plan>,
    mut worker: crate::replicated_decoder::Worker,
    public: Option<Arc<PublicSource>>,
) -> Result<Executor, Error> {
    let crate::replicated_decoder::runtime::Active::Idle(runtime) = std::mem::replace(
        &mut worker.active,
        crate::replicated_decoder::runtime::Active::Retired,
    ) else {
        panic!("idle test runtime required");
    };
    let inputs = (0..worker.program.input_count())
        .filter_map(|position| {
            let (owner, _) = worker.program.input_at(position);
            worker.values[worker.program.value_slot(owner)].take()
        })
        .collect();
    let public = public.or_else(|| {
        worker
            .public
            .take()
            .map(crate::replicated_decoder::runtime::PublicOwner::into_shared)
    });
    Executor::new(plan, runtime, inputs, public, Some(&worker.program))
}
