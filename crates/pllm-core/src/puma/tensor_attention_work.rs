//! Native softmax preflight. The public workspace is only a fixed metadata
//! owner, not a Program, immutable template, execution buffer or Budget permit.
use super::*;

#[cfg(test)]
thread_local! { static FACTORY_ATTEMPTS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
pub(super) fn entered_factory() {
    FACTORY_ATTEMPTS.with(|n| n.set(n.get() + 1));
}
#[cfg(test)]
pub(super) fn factory_attempts() -> usize {
    FACTORY_ATTEMPTS.with(|n| n.get())
}

/// Fixed metadata workspace for the original softmax factories. Quote
/// `workspace_bytes()` before constructing it. No input borrows or heap backing
/// are retained. On error, partial metadata is not a successful work report;
/// every subsequent query resets it. Native proof/callframe/RSS costs remain
/// separate and are NOT bounded by this inline workspace quote.
pub struct SoftmaxWorkWorkspace {
    nodes: [Option<Node>; MAX_NODES],
    widths: [u8; MAX_NODES],
    len: usize,
    lanes: usize,
}

impl SoftmaxWorkWorkspace {
    #[cfg(test)]
    pub(super) fn assert_native_metadata(&self, program: &Program) {
        assert_eq!(self.nodes, program.nodes);
        assert_eq!(self.widths, program.bit_widths);
        assert_eq!((self.len, self.lanes), (program.len, program.lanes));
    }
    pub const fn workspace_bytes() -> usize {
        std::mem::size_of::<Self>()
    }
    pub const fn empty() -> Self {
        Self {
            nodes: [None; MAX_NODES],
            widths: [0; MAX_NODES],
            len: 0,
            lanes: 0,
        }
    }
    pub(super) fn clear(&mut self) {
        self.nodes.fill(None);
        self.widths.fill(0);
        self.len = 0;
        self.lanes = 0;
    }
    pub(super) fn reset(&mut self, lanes: usize, input: Interval) {
        self.clear();
        self.len = 1;
        self.lanes = lanes;
        self.nodes[0] = Some(Node {
            instruction: Instruction::Input,
            bounds: input,
            bit: false,
            lanes,
        });
    }
    pub(super) fn finish_work(&self, output: Value) -> Result<WorkBudget, Error> {
        self.node(output, false)?;
        if output.0 == 0 || output.0 as usize + 1 != self.len {
            return Err(Error::Binding);
        }
        self.nodes[..self.len]
            .iter()
            .flatten()
            .try_fold(WorkBudget::default(), |sum, node| {
                sum.checked_add(instruction_work(*node, &self.widths)?)
            })
    }
    pub(super) fn instruction_slots(&self) -> usize {
        self.len
    }
}

pub(super) fn geometry(rows: usize, width: usize) -> Result<usize, Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if width == 0 || rows == 0 {
        return Err(Error::Bounds);
    }
    if !(1..=MAX_ROW_LANES).contains(&lanes) {
        return Err(Error::Bounds);
    }
    Ok(lanes)
}

pub(super) fn node(
    nodes: &[Option<Node>; MAX_NODES],
    value: Value,
    bit: bool,
) -> Result<Node, Error> {
    nodes
        .get(value.0 as usize)
        .and_then(|n| *n)
        .filter(|n| n.bit == bit)
        .ok_or(Error::Binding)
}

/// Shared original push validation. No mutation until all checks succeed.
#[allow(clippy::too_many_arguments)]
pub(super) fn checked_node(
    nodes: &[Option<Node>; MAX_NODES],
    widths: &[u8; MAX_NODES],
    len: usize,
    capacity: usize,
    instruction: Instruction,
    bounds: Interval,
    bit: bool,
) -> Result<(Value, Node, u8), Error> {
    checked_node_with(
        |v| nodes[v.0 as usize].ok_or(Error::Binding),
        |v| {
            node(nodes, v, true)?;
            Ok(widths[v.0 as usize])
        },
        len,
        capacity,
        instruction,
        bounds,
        bit,
    )
}

// Same native push validation for fixed-array and scalar metadata owners.
pub(super) fn checked_node_with(
    read: impl Fn(Value) -> Result<Node, Error>,
    width: impl Fn(Value) -> Result<u8, Error>,
    len: usize,
    capacity: usize,
    instruction: Instruction,
    bounds: Interval,
    bit: bool,
) -> Result<(Value, Node, u8), Error> {
    if len == MAX_NODES {
        return Err(Error::Bounds);
    }
    let value = Value(len as u8);
    let [first, second] = instruction.inputs();
    let lanes = read(first.ok_or(Error::Binding)?)?.lanes;
    if let Some(second) = second {
        if read(second)?.lanes != lanes {
            return Err(Error::Shape);
        }
    }
    let lanes = match instruction {
        Instruction::RowMap(_, width, RowMap::Sum) if lanes % width == 0 => lanes / width,
        Instruction::RowMap(_, width, RowMap::Broadcast) => {
            lanes.checked_mul(width).ok_or(Error::Bounds)?
        }
        _ => lanes,
    };
    if lanes == 0 || lanes > capacity {
        return Err(Error::Bounds);
    }
    let bits = match instruction {
        Instruction::Sign(_) | Instruction::CompactSign(..) => 1,
        Instruction::Bits(_) => 64,
        Instruction::LowBits(_, bits) => bits,
        Instruction::BitMap(v, map) => map.width(width(v)?)?,
        Instruction::BitOr(a, b) | Instruction::BitXor(a, b) => width(a)?.max(width(b)?),
        Instruction::BitOrLow(_, _, bits) => bits,
        Instruction::BitConstantXor(v, c) => width(v)?.max((64 - c.leading_zeros()) as u8),
        _ => 0,
    };
    Ok((
        value,
        Node {
            instruction,
            bounds,
            bit,
            lanes,
        },
        bits,
    ))
}

pub(super) fn instruction_work(node: Node, widths: &[u8; MAX_NODES]) -> Result<WorkBudget, Error> {
    instruction_work_with(node, |v| widths[v.0 as usize])
}

pub(super) fn instruction_work_with(
    node: Node,
    width: impl Fn(Value) -> u8,
) -> Result<WorkBudget, Error> {
    Ok(match node.instruction {
        Instruction::Input => WorkBudget::default(),
        Instruction::Product { rescale, .. } => WorkBudget::product(node.lanes, rescale)?,
        Instruction::Rescale { .. } => WorkBudget::rescale(node.lanes)?,
        Instruction::Sign(_) => WorkBudget::conversion(node.lanes, true)?,
        Instruction::CompactSign(_, bits) => {
            WorkBudget::conversion_signed_compact(node.lanes, bits)?
        }
        Instruction::Bits(_) => WorkBudget::conversion(node.lanes, false)?,
        Instruction::LowBits(_, bits) => WorkBudget::conversion_bounded(node.lanes, bits)?,
        Instruction::FromBit(input) => WorkBudget::from_boolean(node.lanes, width(input))?,
        Instruction::BitOr(a, b) | Instruction::BitOrLow(a, b, _) => {
            WorkBudget::bit_or(node.lanes, width(a).max(width(b)))?
        }
        Instruction::Affine { .. }
        | Instruction::LaneAffine(..)
        | Instruction::RowMap(..)
        | Instruction::BitMap(..)
        | Instruction::BitConstantXor(..)
        | Instruction::BitXor(..) => WorkBudget::local(),
    })
}

/// Private native primitive operations needed by the softmax recipe. Defaults
/// are also the actual Builder implementations: not a parallel numeric walker.
#[allow(clippy::wrong_self_convention)]
pub(super) trait SoftmaxSink {
    fn node(&self, value: Value, bit: bool) -> Result<Node, Error>;
    fn push(
        &mut self,
        instruction: Instruction,
        bounds: Interval,
        bit: bool,
    ) -> Result<Value, Error>;
    fn refine(&mut self, value: Value, bounds: Interval);
    fn compact(&self) -> bool;
    fn sign_compact(&mut self, input: Value) -> Result<Value, Error>;
    fn input(&self) -> Value {
        Value(0)
    }
    fn width(&self, v: Value) -> Result<u8, Error>;
    fn affine(
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
    fn product(&mut self, left: Value, right: Value, rescale: bool) -> Result<Value, Error> {
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
                    let value = i128::from(x) * i128::from(y);
                    lo = lo.min(value);
                    hi = hi.max(value);
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
                certified: None,
            },
            output,
            false,
        )
    }
    fn square(&mut self, input: Value) -> Result<Value, Error> {
        let bound = self.node(input, false)?.bounds;
        let a = i128::from(bound.lower()).pow(2);
        let b = i128::from(bound.upper()).pow(2);
        let low = if bound.lower() <= 0 && bound.upper() >= 0 {
            0
        } else {
            a.min(b)
        };
        let raw = Interval::new(
            i64::try_from(low).map_err(|_| Error::Bounds)?,
            i64::try_from(a.max(b)).map_err(|_| Error::Bounds)?,
        )?;
        self.push(
            Instruction::Product {
                left: input,
                right: input,
                rescale: true,
                certified: Some(raw),
            },
            rescaled_bounds(raw, 18)?,
            false,
        )
    }
    fn rescale(&mut self, input: Value, shift: u8) -> Result<Value, Error> {
        let bounds = rescaled_bounds(self.node(input, false)?.bounds, shift)?;
        self.push(Instruction::Rescale { input, shift }, bounds, false)
    }
    fn row_map(&mut self, input: Value, width: usize, map: RowMap) -> Result<Value, Error> {
        let output = map.output_bounds(width, self.node(input, false)?.bounds)?;
        if map != RowMap::Broadcast && self.node(input, false)?.lanes % width != 0 {
            return Err(Error::Shape);
        }
        self.push(Instruction::RowMap(input, width, map), output, false)
    }
    fn sign(&mut self, input: Value) -> Result<Value, Error> {
        self.node(input, false)?;
        self.push(Instruction::Sign(input), Interval::new(0, 1)?, true)
    }
    fn bits(&mut self, input: Value) -> Result<Value, Error> {
        self.node(input, false)?;
        self.push(Instruction::Bits(input), Interval::new(0, 0)?, true)
    }
    fn bit_map(&mut self, input: Value, map: BitMap) -> Result<Value, Error> {
        map.width(self.width(input)?)?;
        self.push(Instruction::BitMap(input, map), Interval::new(0, 0)?, true)
    }
    fn bit_or(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.width(left)?;
        self.width(right)?;
        self.push(Instruction::BitOr(left, right), Interval::new(0, 0)?, true)
    }
    fn bit_xor(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.width(left)?;
        self.width(right)?;
        self.push(Instruction::BitXor(left, right), Interval::new(0, 0)?, true)
    }
    fn from_bit(&mut self, input: Value) -> Result<Value, Error> {
        if self.width(input)? > 1 {
            return Err(Error::Bounds);
        }
        self.from_boolean(input)
    }
    fn from_boolean(&mut self, input: Value) -> Result<Value, Error> {
        let bits = self.width(input)?;
        if bits > 62 {
            return Err(Error::Bounds);
        }
        self.push(
            Instruction::FromBit(input),
            Interval::new(0, (1_i64 << bits) - 1)?,
            false,
        )
    }
    fn highest_bit(&mut self, input: Value, width: u8) -> Result<Value, Error> {
        let compact = self.compact();
        let bits = if compact {
            let bounds = self.node(input, false)?.bounds;
            if bounds.lower() <= 0 || i128::from(bounds.upper()) >= 1_i128 << width {
                return Err(Error::Bounds);
            }
            self.push(
                Instruction::LowBits(input, width),
                Interval::new(0, 0)?,
                true,
            )?
        } else {
            self.bits(input)?
        };
        let mut spread = self.bit_map(bits, BitMap::LowBits(width))?;
        for shift in [1, 2, 4, 8, 16, 32] {
            let lower = self.bit_map(spread, BitMap::ShiftRight(shift))?;
            spread = if compact {
                self.push(
                    Instruction::BitOrLow(spread, lower, width),
                    Interval::new(0, 0)?,
                    true,
                )?
            } else {
                self.bit_or(spread, lower)?
            };
        }
        let lower = self.bit_map(spread, BitMap::ShiftRight(1))?;
        self.bit_xor(spread, lower)
    }
}

impl SoftmaxSink for Builder {
    fn node(&self, v: Value, bit: bool) -> Result<Node, Error> {
        Builder::node(self, v, bit)
    }
    fn push(&mut self, i: Instruction, b: Interval, bit: bool) -> Result<Value, Error> {
        Builder::push(self, i, b, bit)
    }
    fn refine(&mut self, v: Value, b: Interval) {
        self.0.nodes[v.0 as usize].as_mut().unwrap().bounds = b;
    }
    fn compact(&self) -> bool {
        self.0.numeric_profile == COMPACT_NONLINEAR_PROFILE
    }
    fn sign_compact(&mut self, v: Value) -> Result<Value, Error> {
        Builder::sign_compact(self, v)
    }
    fn width(&self, v: Value) -> Result<u8, Error> {
        self.bit_width(v)
    }
}

impl SoftmaxSink for SoftmaxWorkWorkspace {
    fn node(&self, v: Value, bit: bool) -> Result<Node, Error> {
        node(&self.nodes, v, bit)
    }
    fn push(&mut self, i: Instruction, b: Interval, bit: bool) -> Result<Value, Error> {
        let (value, node, width) =
            checked_node(&self.nodes, &self.widths, self.len, self.lanes, i, b, bit)?;
        self.nodes[self.len] = Some(node);
        self.widths[self.len] = width;
        self.len += 1;
        Ok(value)
    }
    fn refine(&mut self, v: Value, b: Interval) {
        self.nodes[v.0 as usize].as_mut().unwrap().bounds = b;
    }
    fn compact(&self) -> bool {
        false
    }
    fn sign_compact(&mut self, _: Value) -> Result<Value, Error> {
        Err(Error::Binding)
    }
    fn width(&self, v: Value) -> Result<u8, Error> {
        self.node(v, true)?;
        Ok(self.widths[v.0 as usize])
    }
}

impl Program {
    /// Exact original softmax schedule work, before native construction. Uses
    /// caller-preallocated fixed metadata; executes the original public range
    /// proofs too, whose CPU/callframe costs are NOT these execution ordinals.
    /// No randomness, storage reservation, source proof or Budget authority.
    /// Counters retain WorkBudget semantics: exact ordinals/pair/public draws;
    /// private/AES envelopes cover every public pivot, not observed role traces.
    pub fn softmax_work(
        rows: usize,
        width: usize,
        input: Interval,
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        softmax_work(rows, width, input, workspace, false, false)
    }
    /// Work for `softmax_guarded`, not the separately profiled compact factory.
    pub fn softmax_guarded_work(
        rows: usize,
        width: usize,
        input: Interval,
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        softmax_work(rows, width, input, workspace, true, false)
    }
    /// Work for `softmax_certified`. Reuses its public proof computation but
    /// returns no mass certificate or authority to substitute an operand/domain.
    pub fn softmax_certified_work(
        rows: usize,
        width: usize,
        input: Interval,
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        softmax_work(rows, width, input, workspace, true, true)
    }
}

fn softmax_work(
    rows: usize,
    width: usize,
    input: Interval,
    workspace: &mut SoftmaxWorkWorkspace,
    guard: bool,
    certify: bool,
) -> Result<WorkBudget, Error> {
    let lanes = geometry(rows, width)?;
    workspace.reset(lanes, input);
    let (output, _) = numeric::softmax_recipe(workspace, width, input, guard, certify, false)?;
    workspace.finish_work(output)
}
