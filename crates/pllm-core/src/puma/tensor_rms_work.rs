//! Original RMS/checked-row work preflight; no Program/Arc construction.
use super::attention_work::SoftmaxSink;
use super::*;

#[cfg(test)]
thread_local! { static FACTORIES:std::cell::Cell<[usize;3]>=const {std::cell::Cell::new([0;3])}; }
#[cfg(test)]
pub(super) fn entered(kind: usize) {
    FACTORIES.with(|c| {
        let mut n = c.get();
        n[kind] += 1;
        c.set(n)
    });
}
#[cfg(test)]
pub(super) fn attempts() -> [usize; 3] {
    FACTORIES.with(|c| c.get())
}

/// Concrete fixed metadata capacity, shared with the softmax native primitive
/// validator/cost owner. Quote workspace_bytes BEFORE empty(). No gamma/input
/// borrow or public coefficient backing is retained. Native proof/callframes,
/// native constructor/table/execution buffers, allocator and RSS are separate.
/// Success does not predict/prevent a later native gamma-table Allocation error.
pub struct RmsWorkWorkspace {
    metadata: SoftmaxWorkWorkspace,
    tables: usize,
    poisoned: bool,
}

impl RmsWorkWorkspace {
    #[cfg(test)]
    pub(super) fn assert_native_metadata(&self, program: &Program) {
        self.metadata.assert_native_metadata(program);
        assert_eq!(self.tables, program.public_affine_count);
    }
    pub const fn workspace_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
    pub const fn empty() -> Self {
        Self {
            metadata: SoftmaxWorkWorkspace::empty(),
            tables: 0,
            poisoned: false,
        }
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Discard partial metadata after error; no stale successful work report is
    /// exposed. Does not reset/refund a caller's aggregate request WorkBudget.
    pub fn reset(&mut self) {
        self.metadata.clear();
        self.tables = 0;
        self.poisoned = false;
    }
    fn prepare(&mut self, lanes: usize, input: Interval) {
        self.metadata.reset(lanes, input);
        self.tables = 0;
    }
    fn query(&mut self, spec: Query<'_>) -> Result<WorkBudget, Error> {
        if self.poisoned {
            return Err(Error::Phase);
        }
        let result = self.execute(spec);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn execute(&mut self, spec: Query<'_>) -> Result<WorkBudget, Error> {
        let output = match spec {
            Query::Rms {
                rows,
                width,
                input,
                gamma,
                partitioned,
                input_norm,
            } => {
                let lanes = geometry(rows, width, gamma.len())?;
                self.prepare(lanes, input);
                numeric::rms_recipe(self, width, input, gamma, partitioned, input_norm)?.0
            }
            Query::Checked {
                lanes,
                incoming,
                accepted,
                fallback,
            } => {
                validate_tile_lanes(lanes)?;
                self.prepare(lanes, incoming);
                let input = self.input();
                let checked = checked_recipe(self, input, accepted, fallback)?;
                checked_output(self, checked.safe, checked.valid)?;
                checked.safe
            }
            Query::AllValid { lanes } => {
                if !(1..=MAX_ROW_LANES).contains(&lanes) {
                    return Err(Error::Bounds);
                }
                self.prepare(lanes, Interval::new(0, 1)?);
                all_valid_recipe(self, lanes)?
            }
        };
        self.metadata.finish_work(output)
    }
}

enum Query<'a> {
    Rms {
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &'a [i64],
        partitioned: bool,
        input_norm: Option<u128>,
    },
    Checked {
        lanes: usize,
        incoming: Interval,
        accepted: Interval,
        fallback: i64,
    },
    AllValid {
        lanes: usize,
    },
}

pub(super) fn geometry(rows: usize, width: usize, gamma: usize) -> Result<usize, Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if width == 0 || rows == 0 || gamma != width {
        return Err(Error::Shape);
    }
    if !(1..=MAX_ROW_LANES).contains(&lanes) {
        return Err(Error::Bounds);
    }
    Ok(lanes)
}

/// The actual native affine-table bound scan, also used before native table
/// allocation. Private iterators are existing coefficients or exact repeated
/// gamma columns; no generic callback-allocation promise is exposed.
pub(super) fn affine_bounds(
    bounds: Interval,
    coefficients: impl Iterator<Item = Affine>,
) -> Result<Interval, Error> {
    let mut lower = i64::MAX;
    let mut upper = i64::MIN;
    for c in coefficients {
        if c.right != 0 {
            return Err(Error::Bounds);
        }
        let range = c.output_bounds(InputBounds {
            left: bounds,
            right: Interval::new(0, 0)?,
        })?;
        lower = lower.min(range.lower());
        upper = upper.max(range.upper());
    }
    Interval::new(lower, upper)
}

pub(super) trait RmsSink: SoftmaxSink {
    fn gamma(&mut self, input: Value, width: usize, gamma: &[i64]) -> Result<Value, Error>;
    fn checked_profile(&mut self);
    fn constant_xor(&mut self, input: Value, constant: u64) -> Result<Value, Error> {
        self.width(input)?;
        self.push(
            Instruction::BitConstantXor(input, constant),
            Interval::new(0, 0)?,
            true,
        )
    }
    fn outside_bound(&mut self, bits: Value, bound: i64, below: bool) -> Result<Value, Error> {
        // Original signed-order comparison: sign flip, highest difference,
        // public bound selects parity. No signed subtraction/overflow.
        let key = (bound as u64) ^ (1 << 63);
        let ordered = self.constant_xor(bits, 1 << 63)?;
        let diff = self.constant_xor(ordered, key)?;
        let mut spread = diff;
        for shift in [1, 2, 4, 8, 16, 32] {
            let lower = self.bit_map(spread, BitMap::ShiftRight(shift))?;
            spread = self.bit_or(spread, lower)?;
        }
        let lower = self.bit_map(spread, BitMap::ShiftRight(1))?;
        let highest = self.bit_xor(spread, lower)?;
        self.bit_map(highest, BitMap::Parity(if below { key } else { !key }))
    }
}

impl RmsSink for Builder {
    fn gamma(&mut self, input: Value, width: usize, gamma: &[i64]) -> Result<Value, Error> {
        let scales: [Affine; MAX_ROW_LANES] = std::array::from_fn(|i| Affine {
            left: gamma[i % width],
            right: 0,
            constant: 0,
        });
        let lanes = self.node(input, false)?.lanes;
        self.affine_lanes(input, &scales[..lanes])
    }
    fn checked_profile(&mut self) {
        if !self.compact() {
            self.0.numeric_profile = CHECKED_RANGE_PROFILE;
        }
    }
}

impl SoftmaxSink for RmsWorkWorkspace {
    fn node(&self, v: Value, bit: bool) -> Result<Node, Error> {
        SoftmaxSink::node(&self.metadata, v, bit)
    }
    fn push(&mut self, i: Instruction, b: Interval, bit: bool) -> Result<Value, Error> {
        SoftmaxSink::push(&mut self.metadata, i, b, bit)
    }
    fn refine(&mut self, v: Value, b: Interval) {
        self.metadata.refine(v, b);
    }
    fn compact(&self) -> bool {
        false
    }
    fn sign_compact(&mut self, _: Value) -> Result<Value, Error> {
        Err(Error::Phase)
    }
    fn width(&self, v: Value) -> Result<u8, Error> {
        self.metadata.width(v)
    }
}
impl RmsSink for RmsWorkWorkspace {
    fn gamma(&mut self, input: Value, width: usize, gamma: &[i64]) -> Result<Value, Error> {
        let node = self.node(input, false)?;
        if self.tables == 2 || self.metadata.instruction_slots() == MAX_NODES {
            return Err(Error::Bounds);
        }
        let bounds = affine_bounds(
            node.bounds,
            (0..node.lanes).map(|i| Affine {
                left: gamma[i % width],
                right: 0,
                constant: 0,
            }),
        )?;
        let result = self.push(
            Instruction::LaneAffine(input, self.tables as u8),
            bounds,
            false,
        )?;
        self.tables += 1;
        Ok(result)
    }
    fn checked_profile(&mut self) {}
}

pub(super) fn checked_output<S: SoftmaxSink>(
    sink: &S,
    output: Value,
    valid: Value,
) -> Result<(), Error> {
    if output == valid {
        return Err(Error::Binding);
    }
    let bounds = sink.node(valid, false)?.bounds;
    if bounds.lower() < 0 || bounds.upper() > 1 {
        return Err(Error::Bounds);
    }
    Ok(())
}

pub(super) fn checked_recipe<S: RmsSink>(
    b: &mut S,
    input: Value,
    accepted: Interval,
    fallback: i64,
) -> Result<CheckedValue, Error> {
    b.node(input, false)?;
    if fallback < accepted.lower() || fallback > accepted.upper() {
        return Err(Error::Bounds);
    }
    b.checked_profile();
    let bits = b.bits(input)?;
    let below = b.outside_bound(bits, accepted.lower(), true)?;
    let above = b.outside_bound(bits, accepted.upper(), false)?;
    let invalid = b.bit_or(below, above)?;
    let valid_bit = b.constant_xor(invalid, 1)?;
    let valid = b.from_bit(valid_bit)?;
    let mut mask = valid_bit;
    for shift in [1, 2, 4, 8, 16, 32] {
        let upper = b.bit_map(mask, BitMap::ShiftLeft(shift))?;
        mask = b.bit_xor(mask, upper)?;
    }
    let delta = b.constant_xor(bits, fallback as u64)?;
    let union = b.bit_or(mask, delta)?;
    let xor = b.bit_xor(mask, delta)?;
    let intersection = b.bit_xor(union, xor)?;
    let selected = b.constant_xor(intersection, fallback as u64)?;
    let low = b.bit_map(selected, BitMap::LowBits(32))?;
    let low = b.from_boolean(low)?;
    let high = b.bit_map(selected, BitMap::ShiftRight(32))?;
    let high = b.bit_map(high, BitMap::LowBits(31))?;
    let high = b.from_boolean(high)?;
    let sign = b.bit_map(selected, BitMap::ShiftRight(63))?;
    let sign = b.from_bit(sign)?;
    let negative = b.affine(
        sign,
        None,
        Affine {
            left: i64::MIN,
            right: 0,
            constant: 0,
        },
    )?;
    let upper = b.affine(
        high,
        Some(negative),
        Affine {
            left: 1 << 32,
            right: 1,
            constant: 0,
        },
    )?;
    let safe = b.affine(
        upper,
        Some(low),
        Affine {
            left: 1,
            right: 1,
            constant: 0,
        },
    )?;
    b.refine(safe, accepted);
    Ok(CheckedValue { safe, valid })
}

pub(super) fn all_valid_recipe<S: SoftmaxSink>(b: &mut S, lanes: usize) -> Result<Value, Error> {
    let sum = b.row_map(b.input(), lanes, RowMap::Sum)?;
    let difference = b.affine(
        sum,
        None,
        Affine {
            left: 1,
            right: 0,
            constant: -(lanes as i64),
        },
    )?;
    let failed = b.sign(difference)?;
    let failed = b.from_bit(failed)?;
    b.affine(
        failed,
        None,
        Affine {
            left: -1,
            right: 0,
            constant: 1,
        },
    )
}

impl Program {
    /// Work for the original unchecked RMSNorm factory, not compact RMS.
    /// Returns native execution WorkBudget only, not a constructor/source permit.
    /// Native WorkBudget draw/envelope semantics are unchanged. Public proof CPU
    /// and callframes are not these ordinals. Success is reusable; any error
    /// poisons workspace until reset (subsequent queries return Error::Phase).
    /// Caller checked-adds/multiplies complete invocation work globally; workspace
    /// reset never refunds or performs that caller-owned request accounting.
    pub fn rms_norm_work(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        workspace: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(Query::Rms {
            rows,
            width,
            input,
            gamma,
            partitioned: false,
            input_norm: None,
        })
    }
    /// Work for rms_norm_certified. Output energy remains owned by the actual
    /// native factory; this returns no certificate or proof-family conversion.
    pub fn rms_norm_certified_work(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        workspace: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(Query::Rms {
            rows,
            width,
            input,
            gamma,
            partitioned: true,
            input_norm: None,
        })
    }
    /// Exact same u128 *general complete-input-row* obligation as the existing
    /// rms_norm_with_input_norm factory. Native API does not carry a sealed
    /// certificate pointer; this work query neither creates one nor makes an
    /// attention-only proof interchangeable. Caller must bind the same original
    /// legitimate obligation/domain/gamma to construction. Not runtime checking.
    pub fn rms_norm_with_input_norm_work(
        rows: usize,
        width: usize,
        input: Interval,
        gamma: &[i64],
        squared_norm: u128,
        workspace: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(Query::Rms {
            rows,
            width,
            input,
            gamma,
            partitioned: true,
            input_norm: Some(squared_norm),
        })
    }
    /// Separate checked-range codec work. Does not include all_valid, sticky
    /// products, graph gathers, or RMS; caller must checked-add actual invocations.
    pub fn checked_range_work(
        lanes: usize,
        incoming: Interval,
        accepted: Interval,
        fallback: i64,
        workspace: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(Query::Checked {
            lanes,
            incoming,
            accepted,
            fallback,
        })
    }
    /// Separate complete-bit reduction factory, not width-multiplied scalar work.
    pub fn all_valid_work(
        lanes: usize,
        workspace: &mut RmsWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.query(Query::AllValid { lanes })
    }
}
