//! Caller-buffered, ordered public-slot row folds and centered energy.
//!
//! This is clear arithmetic, not a CKKS layout/parameter or admission contract.
//! Original storage must already be admitted by its owning caller. No allocation,
//! source import, inverse, normalization, or encrypted capability is supplied here.
//! The entire six-buffer pool stays exclusively borrowed until invocation drop;
//! logical last use permits reuse, never an owner-budget refund.

pub const MAX_SLOTS: usize = 32768;
pub const MAX_OPERATIONS: usize = 126;
pub const MAX_CLEAR_WORK: usize = 1 << 28;
pub const MAX_INVOCATION_WORK: usize = 1 << 25;
pub const CONTROL_BYTES: usize = 65536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Geometry,
    ArithmeticOverflow,
    ShortStorage,
    WorkRefused,
    ControlWorkExhausted,
    NonFinite,
    Poisoned,
    LifetimeMismatch,
}

/// One nonrefundable cumulative ledger, including failed attempts and cleanup.
#[derive(Debug)]
pub struct WorkLedger {
    remaining: usize,
    control_remaining: usize,
}
impl WorkLedger {
    pub fn new(ambient_work: usize, ambient_control: usize) -> Self {
        Self {
            remaining: ambient_work.min(MAX_CLEAR_WORK),
            control_remaining: ambient_control.min(1 << 20),
        }
    }
    pub fn remaining(&self) -> usize {
        self.remaining
    }
    pub fn control_remaining(&self) -> usize {
        self.control_remaining
    }
    /// Oracle callers debit the SAME ledger; this is not a per-row allowance.
    pub fn debit(&mut self, clear: usize, control: usize) -> Result<(), Error> {
        let total = clear
            .checked_add(control)
            .ok_or(Error::ArithmeticOverflow)?;
        if total > self.remaining {
            return Err(Error::WorkRefused);
        }
        if control > self.control_remaining {
            return Err(Error::ControlWorkExhausted);
        }
        self.remaining -= total;
        self.control_remaining -= control;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    rows: usize,
    block: usize,
    width: usize,
    slots: usize,
    stages: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quote {
    pub slots: usize,
    pub operations: usize,
    pub public_vectors: usize,
    pub workspace_cells: usize,
    pub public_cells: usize,
    pub array_work: usize,
    pub control_work: usize,
    pub control_bytes: usize,
}
impl Geometry {
    pub fn new(rows: usize, block: usize, width: usize) -> Result<Self, Error> {
        let slots = rows.checked_mul(block).ok_or(Error::ArithmeticOverflow)?;
        if rows == 0
            || !block.is_power_of_two()
            || block > 1024
            || width == 0
            || width > block
            || slots > MAX_SLOTS
        {
            return Err(Error::Geometry);
        }
        Ok(Self {
            rows,
            block,
            width,
            slots,
            stages: block.trailing_zeros() as usize,
        })
    }
    pub fn quote(self) -> Result<Quote, Error> {
        let control_bytes = std::mem::size_of::<Invocation<'static>>()
            + std::mem::size_of::<Quote>()
            + std::mem::size_of::<WorkLedger>()
            + std::mem::size_of::<[Instruction; MAX_OPERATIONS]>();
        if control_bytes > CONTROL_BYTES {
            return Err(Error::ShortStorage);
        }
        Ok(Quote {
            slots: self.slots,
            operations: 12 * self.stages + 6,
            public_vectors: 2 + 2 * self.stages,
            workspace_cells: 6 * self.slots,
            public_cells: (2 + 2 * self.stages) * self.slots,
            // Conservative full ten-stage envelope; includes public initialization,
            // validation/cleanup and pool initialization/cleanup, even on failure.
            array_work: 542 * self.slots,
            control_work: 4 * MAX_OPERATIONS * MAX_OPERATIONS + 1024,
            control_bytes,
        })
    }
    pub fn rows(self) -> usize {
        self.rows
    }
    pub fn block(self) -> usize {
        self.block
    }
    pub fn width(self) -> usize {
        self.width
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    PublicProduct(usize),
    Rotate(isize),
    Add,
    Subtract,
    /// Ordered row-scalar broadcast subtraction, clear continuation only.
    /// The right operand is the row-head scalar of the declared value.
    SubtractRowHead,
    Square,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub kernel: Kernel,
    /// Input ordinal zero denotes immutable original input. Output ordinals start at one.
    pub left: usize,
    pub right: Option<usize>,
}
const EMPTY: Instruction = Instruction {
    kernel: Kernel::Square,
    left: 0,
    right: None,
};

fn emit(
    p: &mut [Instruction; MAX_OPERATIONS],
    n: &mut usize,
    kernel: Kernel,
    left: usize,
    right: Option<usize>,
) -> usize {
    p[*n] = Instruction {
        kernel,
        left,
        right,
    };
    *n += 1;
    *n
}
fn fold(g: Geometry, p: &mut [Instruction; MAX_OPERATIONS], n: &mut usize, mut t: usize) -> usize {
    for stage in 0..g.stages {
        let d = 1isize << stage;
        let u = emit(p, n, Kernel::Rotate(d), t, None);
        let l = emit(p, n, Kernel::PublicProduct(2 + 2 * stage), u, None);
        let w = emit(p, n, Kernel::Rotate(d - g.block as isize), t, None);
        let h = emit(p, n, Kernel::PublicProduct(3 + 2 * stage), w, None);
        let b = emit(p, n, Kernel::Add, l, Some(h));
        t = emit(p, n, Kernel::Add, t, Some(b));
    }
    t
}
fn program(g: Geometry) -> ([Instruction; MAX_OPERATIONS], usize) {
    let mut p = [EMPTY; MAX_OPERATIONS];
    let mut n = 0;
    let a = emit(&mut p, &mut n, Kernel::PublicProduct(0), 0, None);
    let s = fold(g, &mut p, &mut n, a);
    let x = emit(&mut p, &mut n, Kernel::PublicProduct(1), a, None);
    let z_raw = emit(&mut p, &mut n, Kernel::Subtract, x, Some(s));
    let z = emit(&mut p, &mut n, Kernel::PublicProduct(0), z_raw, None);
    let q = emit(&mut p, &mut n, Kernel::Square, z, None);
    let v = fold(g, &mut p, &mut n, q);
    emit(&mut p, &mut n, Kernel::PublicProduct(0), v, None);
    (p, n)
}

/// Immutable original instruction plan, without clear arrays or an invocation.
/// Construction debits the original cumulative ledger before generating control
/// storage. This does not admit ciphertexts, public operands, or native work.
///
/// ```compile_fail
/// use pllm_core::row_fold::{Geometry, InstructionPlan, WorkLedger};
/// let mut work = WorkLedger::new(65536, 65536);
/// let g = Geometry::new(1, 2, 1).unwrap();
/// let mut plan = InstructionPlan::new(g, &mut work).unwrap();
/// let instructions = plan.instructions();
/// plan = InstructionPlan::new(g, &mut work).unwrap();
/// assert_eq!(instructions.len(), 18); // Cannot replace a borrowed original plan.
/// ```
pub struct InstructionPlan {
    geometry: Geometry,
    instructions: [Instruction; MAX_OPERATIONS],
    count: usize,
}
impl InstructionPlan {
    pub fn new(g: Geometry, ledger: &mut WorkLedger) -> Result<Self, Error> {
        // Fixed-array initialization, generation, moves and terminating advances.
        // Same bounded generator as Invocation; no public/pool initialization.
        ledger.debit(0, 8 * MAX_OPERATIONS + 32)?;
        Ok(Self::prepaid(g))
    }
    fn prepaid(geometry: Geometry) -> Self {
        let (instructions, count) = program(geometry);
        Self { geometry, instructions, count }
    }
    pub fn geometry(&self) -> Geometry { self.geometry }
    pub fn instructions(&self) -> &[Instruction] { &self.instructions[..self.count] }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub enum State {
    #[default]
    Fresh,
    Poisoned,
    Complete,
}

/// Single-use invocation over pre-existing, disjoint original caller regions.
/// Safe exclusive borrows prevent source mutation and escaped workspace aliases.
/// No observer/callback (and hence no unpriced closure capture) runs in executor.
///
/// ```compile_fail
/// use pllm_core::row_fold::*;
/// let g = Geometry::new(1, 2, 1).unwrap();
/// let mut input = [1.0, 0.0];
/// let mut public = [0.0; 8]; let mut pool = [0.0; 12];
/// let mut state = State::Fresh; let mut work = WorkLedger::new(10000, 100000);
/// let mut run = Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut work).unwrap();
/// input[0] = 2.0; // Captured immutable source cannot change.
/// run.execute().unwrap();
/// ```
///
/// ```compile_fail
/// use pllm_core::row_fold::*;
/// let g = Geometry::new(1, 2, 1).unwrap();
/// let input = [1.0, 0.0]; let mut public = [0.0; 8]; let mut pool = [0.0; 12];
/// let mut state = State::Fresh; let mut work = WorkLedger::new(10000, 100000);
/// let result;
/// {
///     let mut run = Invocation::new(g, &input, &mut public, &mut pool, &mut state, &mut work).unwrap();
///     result = run.execute().unwrap();
/// }
/// assert_eq!(result[0], 0.0); // Output cannot outlive whole-pool owner borrow.
/// ```
pub struct Invocation<'a> {
    geometry: Geometry,
    input: &'a [f64],
    public: &'a mut [f64],
    pool: &'a mut [f64],
    state: &'a mut State,
    instructions: [Instruction; MAX_OPERATIONS],
    count: usize,
    locations: [Option<usize>; MAX_OPERATIONS + 1],
    last_use: [usize; MAX_OPERATIONS + 1],
    live: [bool; 6],
    cursor: usize,
    output: Option<usize>,
}
// Crate-private, non-Clone prepayment for a complete clear continuation. Its
// geometry cannot be changed by the consumer; this is not an ambient grant.
pub(crate) struct PrefixPrepayment(Geometry);
impl<'a> Invocation<'a> {
    pub fn new(
        g: Geometry,
        input: &'a [f64],
        public: &'a mut [f64],
        pool: &'a mut [f64],
        state: &'a mut State,
        ledger: &mut WorkLedger,
    ) -> Result<Self, Error> {
        let q = g.quote()?;
        if *state != State::Fresh {
            return Err(Error::Poisoned);
        }
        if input.len() != q.slots
            || public.len() != q.public_cells
            || pool.len() != q.workspace_cells
        {
            return Err(Error::ShortStorage);
        }
        if q.array_work > MAX_INVOCATION_WORK {
            return Err(Error::WorkRefused);
        }
        ledger.debit(q.array_work, q.control_work)?;
        Self::initialize(g, input, public, pool, state)
    }
    pub(crate) fn prepay_continuation(
        g: Geometry,
        ledger: &mut WorkLedger,
        extra_work: usize,
        extra_control: usize,
    ) -> Result<PrefixPrepayment, Error> {
        let q = g.quote()?;
        let work = q
            .array_work
            .checked_add(extra_work)
            .ok_or(Error::ArithmeticOverflow)?;
        let control = q
            .control_work
            .checked_add(extra_control)
            .ok_or(Error::ArithmeticOverflow)?;
        if work > MAX_INVOCATION_WORK {
            return Err(Error::WorkRefused);
        }
        ledger.debit(work, control)?;
        Ok(PrefixPrepayment(g))
    }
    pub(crate) fn new_prepaid(
        payment: PrefixPrepayment,
        input: &'a [f64],
        public: &'a mut [f64],
        pool: &'a mut [f64],
        state: &'a mut State,
    ) -> Result<Self, Error> {
        let g = payment.0;
        let q = g.quote()?;
        if *state != State::Fresh {
            return Err(Error::Poisoned);
        }
        if input.len() != q.slots
            || public.len() != q.public_cells
            || pool.len() != q.workspace_cells
        {
            return Err(Error::ShortStorage);
        }
        Self::initialize(g, input, public, pool, state)
    }
    fn initialize(
        g: Geometry,
        input: &'a [f64],
        public: &'a mut [f64],
        pool: &'a mut [f64],
        state: &'a mut State,
    ) -> Result<Self, Error> {
        let q = g.quote()?;
        *state = State::Poisoned; // Remains poisoned through errors and unwinding.
        // Existing whole-invocation control prepayment covers this extraction.
        let InstructionPlan { instructions, count, .. } = InstructionPlan::prepaid(g);
        let mut result = Self {
            geometry: g,
            input,
            public,
            pool,
            state,
            instructions,
            count,
            locations: [None; MAX_OPERATIONS + 1],
            last_use: [0; MAX_OPERATIONS + 1],
            live: [false; 6],
            cursor: 0,
            output: None,
        };
        for (i, op) in result.instructions[..count].iter().enumerate() {
            result.last_use[op.left] = i;
            if let Some(r) = op.right {
                result.last_use[r] = i;
            }
        }
        result.pool.fill(0.0);
        for vector in 0..q.public_vectors {
            for j in 0..q.slots {
                let c = j % g.block;
                result.public[vector * q.slots + j] = match vector {
                    0 => u8::from(c < g.width) as f64,
                    1 => g.width as f64,
                    v => {
                        let threshold = g.block - (1 << ((v - 2) / 2));
                        u8::from(if v % 2 == 0 {
                            c < threshold
                        } else {
                            c >= threshold
                        }) as f64
                    }
                };
            }
        }
        if result.input.iter().any(|x| !x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(result)
    }
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions[..self.count]
    }
    fn read(&self, ordinal: usize, lane: usize) -> Result<f64, Error> {
        if ordinal == 0 {
            return Ok(self.input[lane]);
        }
        let slot = self.locations[ordinal].ok_or(Error::LifetimeMismatch)?;
        if !self.live[slot] {
            return Err(Error::LifetimeMismatch);
        }
        Ok(self.pool[slot * self.geometry.slots + lane])
    }
    /// Runs once. Failure never exposes a partial output. Returned borrow pins
    /// invocation, original input, public vectors and the ENTIRE physical pool.
    pub fn execute(&mut self) -> Result<&[f64], Error> {
        self.execute_program(false)?;
        let slot = self.output.ok_or(Error::LifetimeMismatch)?;
        let s = self.geometry.slots;
        Ok(&self.pool[slot * s..(slot + 1) * s])
    }
    /// Consume this invocation, retaining the actual centered vector and final
    /// energy from one ordered program. No replay or algebraic substitute.
    /// This scalar-row continuation binds s to the row-head Fold(A) result:
    /// cyclic floating reductions need not round identically at every lane.
    /// Exactly one subtraction uses an explicit row-scalar broadcast view;
    /// legacy execute retains its original per-lane subtraction and ordering.
    /// The continuation pins one additional logical value in the same six-buffer
    /// pool. Legacy `execute` still retires it immediately after square.
    pub fn into_centered_energy(mut self) -> Result<CenteredEnergy<'a>, Error> {
        let subtract = self.instructions[..self.count]
            .iter_mut()
            .find(|op| op.kernel == Kernel::Subtract)
            .ok_or(Error::LifetimeMismatch)?;
        subtract.kernel = Kernel::SubtractRowHead;
        self.execute_program(true)?;
        *self.state = State::Poisoned;
        let ordinal = self.instructions[..self.count]
            .iter()
            .find(|op| op.kernel == Kernel::Square)
            .ok_or(Error::LifetimeMismatch)?
            .left;
        let centered_slot = self.locations[ordinal].ok_or(Error::LifetimeMismatch)?;
        let energy_slot = self.output.ok_or(Error::LifetimeMismatch)?;
        Ok(CenteredEnergy {
            invocation: self,
            centered_slot,
            energy_slot,
            normalized_slot: None,
        })
    }
    fn execute_program(&mut self, retain_centered: bool) -> Result<(), Error> {
        if self.cursor != 0 || self.output.is_some() {
            return Err(Error::Poisoned);
        }
        // Set before the first possible failure, including a failure in operation zero.
        self.cursor = 1;
        let s = self.geometry.slots;
        for i in 0..self.count {
            let op = self.instructions[i];
            let slot = self
                .live
                .iter()
                .position(|&v| !v)
                .ok_or(Error::LifetimeMismatch)?;
            self.live[slot] = true;
            for j in 0..s {
                let source = match op.kernel {
                    Kernel::Rotate(d) => (j as isize + d).rem_euclid(s as isize) as usize,
                    _ => j,
                };
                let left = self.read(op.left, source)?;
                let value = match op.kernel {
                    Kernel::Rotate(_) => left,
                    Kernel::PublicProduct(v) => left * self.public[v * s + j],
                    Kernel::Add => left + self.read(op.right.ok_or(Error::LifetimeMismatch)?, j)?,
                    Kernel::Subtract => {
                        left - self.read(op.right.ok_or(Error::LifetimeMismatch)?, j)?
                    }
                    Kernel::SubtractRowHead => {
                        let head = (j / self.geometry.block) * self.geometry.block;
                        left - self.read(op.right.ok_or(Error::LifetimeMismatch)?, head)?
                    }
                    Kernel::Square => left * left,
                };
                self.pool[slot * s + j] = value;
            }
            if self.pool[slot * s..(slot + 1) * s]
                .iter()
                .any(|x| !x.is_finite())
            {
                return Err(Error::NonFinite);
            }
            self.locations[i + 1] = Some(slot);
            for ordinal in 1..=i {
                let centered = retain_centered
                    && self.instructions[i].kernel == Kernel::Square
                    && self.instructions[i].left == ordinal;
                if self.last_use[ordinal] == i && !centered {
                    let old = self.locations[ordinal]
                        .take()
                        .ok_or(Error::LifetimeMismatch)?;
                    self.live[old] = false;
                }
            }
            self.cursor = i + 2;
        }
        let slot = self.locations[self.count].ok_or(Error::LifetimeMismatch)?;
        self.output = Some(slot);
        *self.state = State::Complete;
        Ok(())
    }
}

/// Consuming clear continuation state. Full input/public/pool borrows remain
/// bound to the original invocation. Not a ciphertext or tensor-source proof.
pub struct CenteredEnergy<'a> {
    invocation: Invocation<'a>,
    centered_slot: usize,
    energy_slot: usize,
    normalized_slot: Option<usize>,
}
pub(crate) type ContinuationParts<'a> = (&'a [f64], &'a [f64], &'a mut [f64]);
impl CenteredEnergy<'_> {
    pub fn geometry(&self) -> Geometry {
        self.invocation.geometry
    }
    pub fn centered(&self) -> &[f64] {
        let s = self.invocation.geometry.slots;
        &self.invocation.pool[self.centered_slot * s..(self.centered_slot + 1) * s]
    }
    pub fn energy(&self) -> &[f64] {
        let s = self.invocation.geometry.slots;
        &self.invocation.pool[self.energy_slot * s..(self.energy_slot + 1) * s]
    }
    pub(crate) fn continuation_parts(&mut self) -> Result<ContinuationParts<'_>, Error> {
        if self.normalized_slot.is_some() {
            return Err(Error::Poisoned);
        }
        let run = &mut self.invocation;
        let output = run
            .live
            .iter()
            .position(|&v| !v)
            .ok_or(Error::LifetimeMismatch)?;
        run.live[output] = true;
        self.normalized_slot = Some(output);
        let (mut z, mut v, mut y) = (None, None, None);
        for (slot, part) in run.pool.chunks_exact_mut(run.geometry.slots).enumerate() {
            if slot == self.centered_slot {
                z = Some(part);
            } else if slot == self.energy_slot {
                v = Some(part);
            } else if slot == output {
                y = Some(part);
            }
        }
        Ok((
            z.ok_or(Error::LifetimeMismatch)?,
            v.ok_or(Error::LifetimeMismatch)?,
            y.ok_or(Error::LifetimeMismatch)?,
        ))
    }
    pub(crate) fn finish_normalized(&mut self) {
        *self.invocation.state = State::Complete;
    }
    pub(crate) fn normalized(&self) -> &[f64] {
        let s = self.invocation.geometry.slots;
        // Only the successful continuation can publish this view.
        let slot = self.normalized_slot.expect("completed clear continuation");
        &self.invocation.pool[slot * s..(slot + 1) * s]
    }
}
impl Drop for Invocation<'_> {
    fn drop(&mut self) {
        // Clear reference hygiene, not a native erasure or allocator-credit claim.
        self.pool.fill(0.0);
        self.public.fill(0.0);
    }
}

#[cfg(test)]
#[path = "row_fold_tests.rs"]
mod tests;
