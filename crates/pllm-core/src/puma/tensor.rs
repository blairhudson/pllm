//! Bounded public tensor programs over party-local replicated shares.
//!
//! All instructions execute for every lane. Comparisons return secret bits;
//! selections use exact ring products, never reconstructed branch decisions.
//! Public intervals must come from the caller/compiler, not private inputs.
//! This executor uses the existing PRSS owner and typed private-channel frames.

use super::{
    arithmetic::{rescaled_bounds, Affine, BitMap, RowMap},
    boolean,
    conversion::Kind,
    InputBounds, Interval, Operand, PartyRuntime, Status,
};
use crate::puma::payload::Reservation;
use crate::puma::*;
use sha2::{Digest, Sha256};

pub const PROFILE: &str = "pllm.puma.public-tensor-program.q18.v1";
/// Approved normalization adaptation: one positive Q18 unit instead of a
/// source epsilon that encodes to zero. Other source-derived programs retain
/// their original numerical identity.
pub const RMS_NORM_PROFILE: &str = "puma.rmsnorm-epsilon1-q18.v1";
pub const RMS_NORM_EPSILON_Q18: i64 = 1;
pub const MAX_NODES: usize = 192;
pub const MAX_TILE_LANES: usize = 128;
/// Row reductions may carry one complete native-width row; Boolean conversion
/// still admits its own expanded shape before any instruction consumes PRSS.
pub const MAX_ROW_LANES: usize = MAX_LANES;

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
        certified: Option<Interval>,
    },
    Rescale {
        input: Value,
        shift: u8,
    },
    Sign(Value),
    FromBit(Value),
    Bits(Value),
    BitMap(Value, BitMap),
    BitOr(Value, Value),
    BitXor(Value, Value),
    RowMap(Value, usize, RowMap),
    LaneAffine(Value, u8),
}

impl Instruction {
    fn inputs(self) -> [Option<Value>; 2] {
        match self {
            Self::Input => [None, None],
            Self::Affine { left, right, .. } => [Some(left), right],
            Self::Product { left, right, .. }
            | Self::BitOr(left, right)
            | Self::BitXor(left, right) => [Some(left), Some(right)],
            Self::Rescale { input, .. }
            | Self::Sign(input)
            | Self::FromBit(input)
            | Self::Bits(input)
            | Self::BitMap(input, _)
            | Self::RowMap(input, _, _)
            | Self::LaneAffine(input, _) => [Some(input), None],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Node {
    instruction: Instruction,
    bounds: Interval,
    bit: bool,
    lanes: usize,
}

/// Immutable public instructions. Construction proves every eager intermediate
/// range, including inactive polynomial branches, before an owner draws PRSS.
#[derive(Clone)]
pub struct Program {
    nodes: [Option<Node>; MAX_NODES],
    len: usize,
    lanes: usize,
    digest: [u8; 32],
    numeric_profile: &'static str,
    last_use: [u8; MAX_NODES],
    slots: [u8; MAX_NODES],
    register_count: usize,
    bit_widths: [u8; MAX_NODES],
    public_affines: [Vec<Affine>; 2],
    public_affine_count: usize,
    work: WorkBudget,
}

pub struct Builder(Program);

impl Builder {
    pub fn new(lanes: usize, input: Interval) -> Result<Self, Error> {
        if !(1..=MAX_TILE_LANES).contains(&lanes) {
            return Err(Error::Bounds);
        }
        Self::row(lanes, input)
    }

    fn row(lanes: usize, input: Interval) -> Result<Self, Error> {
        if !(1..=MAX_ROW_LANES).contains(&lanes) {
            return Err(Error::Bounds);
        }
        let mut program = Program {
            nodes: [None; MAX_NODES],
            len: 1,
            lanes,
            digest: [0; 32],
            numeric_profile: super::NUMERIC_PROFILE,
            last_use: [0; MAX_NODES],
            slots: [0; MAX_NODES],
            register_count: 0,
            bit_widths: [0; MAX_NODES],
            public_affines: [Vec::new(), Vec::new()],
            public_affine_count: 0,
            work: WorkBudget::default(),
        };
        program.nodes[0] = Some(Node {
            instruction: Instruction::Input,
            bounds: input,
            bit: false,
            lanes,
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
        let [first, second] = instruction.inputs();
        let lanes = self.0.nodes[first.ok_or(Error::Binding)?.0 as usize]
            .ok_or(Error::Binding)?
            .lanes;
        if let Some(second) = second {
            if self.0.nodes[second.0 as usize].ok_or(Error::Binding)?.lanes != lanes {
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
        if lanes == 0 || lanes > self.0.lanes {
            return Err(Error::Bounds);
        }
        self.0.bit_widths[self.0.len] = match instruction {
            Instruction::Sign(_) => 1,
            Instruction::Bits(_) => 64,
            Instruction::BitMap(v, map) => map.width(self.bit_width(v)?)?,
            Instruction::BitOr(a, b) | Instruction::BitXor(a, b) => {
                self.bit_width(a)?.max(self.bit_width(b)?)
            }
            _ => 0,
        };
        self.0.nodes[self.0.len] = Some(Node {
            instruction,
            bounds,
            bit,
            lanes,
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
                certified: None,
            },
            output,
            false,
        )
    }

    pub fn rescale(&mut self, input: Value, shift: u8) -> Result<Value, Error> {
        let bounds = rescaled_bounds(self.node(input, false)?.bounds, shift)?;
        self.push(Instruction::Rescale { input, shift }, bounds, false)
    }

    pub fn square(&mut self, input: Value) -> Result<Value, Error> {
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

    pub fn row_map(&mut self, input: Value, width: usize, map: RowMap) -> Result<Value, Error> {
        let output = map.output_bounds(width, self.node(input, false)?.bounds)?;
        if map != RowMap::Broadcast && self.node(input, false)?.lanes % width != 0 {
            return Err(Error::Shape);
        }
        self.push(Instruction::RowMap(input, width, map), output, false)
    }

    pub fn affine_lanes(&mut self, input: Value, coefficients: &[Affine]) -> Result<Value, Error> {
        if coefficients.len() != self.node(input, false)?.lanes
            || self.0.public_affine_count == 2
            || self.0.len == MAX_NODES
        {
            return Err(Error::Bounds);
        }
        let bounds = self.node(input, false)?.bounds;
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
        let slot = self.0.public_affine_count;
        let bounds = Interval::new(lower, upper)?;
        let mut table = Vec::new();
        table
            .try_reserve_exact(coefficients.len())
            .map_err(|_| Error::Allocation)?;
        if table.capacity() != coefficients.len() {
            return Err(Error::Allocation);
        }
        table.extend_from_slice(coefficients);
        // Failure cannot leave an instruction referring to an absent table.
        let output = self.push(Instruction::LaneAffine(input, slot as u8), bounds, false)?;
        self.0.public_affines[slot] = table;
        self.0.public_affine_count += 1;
        Ok(output)
    }

    pub fn sign(&mut self, input: Value) -> Result<Value, Error> {
        self.node(input, false)?;
        self.push(Instruction::Sign(input), Interval::new(0, 1)?, true)
    }

    pub fn from_bit(&mut self, input: Value) -> Result<Value, Error> {
        if self.bit_width(input)? > 1 {
            return Err(Error::Bounds);
        }
        self.from_boolean(input)
    }

    pub fn from_boolean(&mut self, input: Value) -> Result<Value, Error> {
        let bits = self.bit_width(input)?;
        if bits > 62 {
            return Err(Error::Bounds);
        }
        self.push(
            Instruction::FromBit(input),
            Interval::new(0, (1_i64 << bits) - 1)?,
            false,
        )
    }

    fn bit_width(&self, input: Value) -> Result<u8, Error> {
        self.node(input, true)?;
        Ok(self.0.bit_widths[input.0 as usize])
    }

    pub fn bits(&mut self, input: Value) -> Result<Value, Error> {
        self.node(input, false)?;
        self.push(Instruction::Bits(input), Interval::new(0, 0)?, true)
    }

    pub fn bit_map(&mut self, input: Value, map: BitMap) -> Result<Value, Error> {
        map.width(self.bit_width(input)?)?;
        self.push(Instruction::BitMap(input, map), Interval::new(0, 0)?, true)
    }

    pub fn bit_or(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.bit_width(left)?;
        self.bit_width(right)?;
        self.push(Instruction::BitOr(left, right), Interval::new(0, 0)?, true)
    }

    pub fn bit_xor(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        self.bit_width(left)?;
        self.bit_width(right)?;
        self.push(Instruction::BitXor(left, right), Interval::new(0, 0)?, true)
    }

    fn highest_bit(&mut self, input: Value, width: u8) -> Result<Value, Error> {
        let bits = self.bits(input)?;
        let mut spread = self.bit_map(bits, BitMap::LowBits(width))?;
        for shift in [1, 2, 4, 8, 16, 32] {
            let lower = self.bit_map(spread, BitMap::ShiftRight(shift))?;
            spread = self.bit_or(spread, lower)?;
        }
        let lower = self.bit_map(spread, BitMap::ShiftRight(1))?;
        self.bit_xor(spread, lower)
    }

    pub fn finish(mut self, output: Value) -> Result<Program, Error> {
        self.node(output, false)?;
        if output.0 == 0 || output.0 as usize + 1 != self.0.len {
            return Err(Error::Binding);
        }
        let mut h = Sha256::new();
        h.update(PROFILE.as_bytes());
        h.update(self.0.numeric_profile.as_bytes());
        h.update((self.0.lanes as u64).to_le_bytes());
        h.update((self.0.len as u64).to_le_bytes());
        for (index, node) in self.0.nodes[..self.0.len].iter().flatten().enumerate() {
            for value in node.instruction.inputs().into_iter().flatten() {
                self.0.last_use[value.0 as usize] = index as u8;
            }
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
                    certified,
                } => {
                    h.update([2, left.0, right.0, u8::from(rescale)]);
                    if let Some(raw) = certified {
                        h.update(b"conditional-product-envelope.v1");
                        h.update(raw.lower().to_le_bytes());
                        h.update(raw.upper().to_le_bytes());
                    }
                }
                Instruction::Rescale { input, shift } => h.update([3, input.0, shift]),
                Instruction::Sign(input) => h.update([4, input.0]),
                Instruction::FromBit(input) => h.update([5, input.0]),
                Instruction::Bits(input) => h.update([6, input.0]),
                Instruction::BitMap(input, map) => {
                    h.update([7, input.0]);
                    map.hash(&mut h);
                }
                Instruction::BitOr(a, b) => h.update([8, a.0, b.0]),
                Instruction::BitXor(a, b) => h.update([9, a.0, b.0]),
                Instruction::RowMap(input, width, map) => {
                    h.update([10, input.0]);
                    h.update((width as u64).to_le_bytes());
                    h.update(
                        match map {
                            RowMap::Sum => u64::MAX,
                            RowMap::Broadcast => u64::MAX - 1,
                            RowMap::Rotate(n) => n as u64,
                        }
                        .to_le_bytes(),
                    );
                }
                Instruction::LaneAffine(input, slot) => {
                    h.update([11, input.0, slot]);
                    for c in &self.0.public_affines[slot as usize][..node.lanes] {
                        h.update(c.left.to_le_bytes());
                        h.update(c.constant.to_le_bytes());
                    }
                }
            }
            h.update(node.bounds.lower().to_le_bytes());
            h.update(node.bounds.upper().to_le_bytes());
        }
        self.0.digest = h.finalize().into();
        self.0.last_use[output.0 as usize] = self.0.len as u8;
        let mut available_at = [0; MAX_NODES];
        for i in 0..self.0.len {
            let slot = available_at.iter().position(|&next| next <= i).unwrap();
            self.0.slots[i] = slot as u8;
            self.0.register_count = self.0.register_count.max(slot + 1);
            available_at[slot] = usize::from(self.0.last_use[i]).max(i) + 1;
        }
        self.0.work = self.0.nodes[..self.0.len]
            .iter()
            .flatten()
            .try_fold(WorkBudget::default(), |total, node| {
                total.checked_add(self.0.instruction_work(*node)?)
            })?;
        Ok(self.0)
    }
}

fn activation_predicates(
    b: &mut Builder,
    input: Value,
    pivots: [i64; 3],
) -> Result<[Value; 3], Error> {
    let mut predicates = [input; 3];
    for (i, pivot) in pivots.into_iter().enumerate() {
        // Last polynomial includes its upper endpoint; strict MSB tests
        // x < upper+1. All decisions stay secret-shared through B2A.
        let delta = b.affine(
            input,
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
    Ok(predicates)
}

impl Program {
    /// The approved source-derived profile, not the unavailable evaluated IR.
    /// Coefficients/pivots are the binary32/toward-zero Q18 constants frozen in
    /// paper/reproduction-contracts/shared-profile-resolution.md, section 3.3.
    pub fn activation(kind: Activation, lanes: usize, input: Interval) -> Result<Self, Error> {
        Self::activation_impl(kind, lanes, input, false)
    }

    /// Preserve the source piecewise equations while excluding zero/identity
    /// tails from polynomial powers. Decisions remain shared, with one fresh
    /// exact guard product. The eager program retains its instruction identity.
    pub fn activation_guarded(
        kind: Activation,
        lanes: usize,
        input: Interval,
    ) -> Result<Self, Error> {
        Self::activation_impl(kind, lanes, input, true)
    }

    fn activation_impl(
        kind: Activation,
        lanes: usize,
        input: Interval,
        guarded: bool,
    ) -> Result<Self, Error> {
        let mut b = Builder::new(lanes, input)?;
        let original = b.input();
        let pivots = match kind {
            Activation::Gelu => [-1048576, -511180, 786432],
            Activation::Silu => [-2097152, -1048576, 1048576],
        };
        let early = if guarded {
            Some(activation_predicates(&mut b, original, pivots)?)
        } else {
            None
        };
        let x = if let Some(predicates) = early {
            let inside = b.affine(
                predicates[2],
                Some(predicates[0]),
                Affine {
                    left: 1,
                    right: -1,
                    constant: 0,
                },
            )?;
            // [x<=hi]-[x<lo] is exactly the indicator of lo<=x<=hi.
            b.0.nodes[inside.0 as usize].as_mut().unwrap().bounds = Interval::new(0, 1)?;
            let safe = b.product(original, inside, false)?;
            // Both excluded tails map to zero only inside the polynomial; the
            // final identity branch still uses the untouched original input.
            let lo = input.lower().max(pivots[0]);
            let hi = input.upper().min(pivots[2]);
            b.0.nodes[safe.0 as usize].as_mut().unwrap().bounds = if lo <= hi {
                Interval::new(lo.min(0), hi.max(0))?
            } else {
                Interval::new(0, 0)?
            };
            safe
        } else {
            original
        };
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
        let predicates = match early {
            Some(predicates) => predicates,
            None => activation_predicates(&mut b, original, pivots)?,
        };
        let upper = select(&mut b, predicates[2], high, original)?;
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
        if guarded {
            b.0.nodes[output.0 as usize].as_mut().unwrap().bounds =
                numeric::activation_bounds(kind, input)?;
        }
        b.finish(output)
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
    pub fn numeric_profile(&self) -> &'static str {
        self.numeric_profile
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn instructions(&self) -> usize {
        self.len - 1
    }

    /// Exact top-level ordinals and shape-aware PRSS upper bounds for one
    /// invocation. This is not a CPU, network-round or wall-time measurement.
    pub fn work_budget(&self) -> WorkBudget {
        self.work
    }

    fn instruction_work(&self, node: Node) -> Result<WorkBudget, Error> {
        Ok(match node.instruction {
            Instruction::Input => WorkBudget::default(),
            Instruction::Product { rescale, .. } => WorkBudget::product(node.lanes, rescale)?,
            Instruction::Rescale { .. } => WorkBudget::rescale(node.lanes)?,
            Instruction::Sign(_) => WorkBudget::conversion(node.lanes, true)?,
            Instruction::Bits(_) => WorkBudget::conversion(node.lanes, false)?,
            Instruction::FromBit(input) => {
                WorkBudget::from_boolean(node.lanes, self.bit_widths[input.0 as usize])?
            }
            Instruction::BitOr(a, b) => WorkBudget::bit_or(
                node.lanes,
                self.bit_widths[a.0 as usize].max(self.bit_widths[b.0 as usize]),
            )?,
            Instruction::Affine { .. }
            | Instruction::LaneAffine(..)
            | Instruction::RowMap(..)
            | Instruction::BitMap(..)
            | Instruction::BitXor(..) => WorkBudget::local(),
        })
    }
    pub fn output_bounds(&self) -> Interval {
        self.nodes[self.len - 1].unwrap().bounds
    }

    pub fn output_lanes(&self) -> usize {
        self.nodes[self.len - 1].unwrap().lanes
    }

    /// Conservative retained-register plus primitive/copy/frame allowance.
    /// Original exported buffers retain their own ledger charges as usual.
    pub fn admitted_bytes(&self) -> usize {
        let working = (0..self.len)
            .map(|step| {
                let live: usize = (0..=step)
                    .filter(|&i| i == step || usize::from(self.last_use[i]) >= step)
                    .map(|i| 2 * self.nodes[i].unwrap().lanes)
                    .sum();
                let node = self.nodes[step].unwrap();
                // Primitive allowance plus this executor's operand copies.
                // Small-width OT expands each bit separately and can exceed
                // the old uniform 128-word allowance. Conversion prices its
                // full source schedule, including original exported frames.
                let temporary = match node.instruction {
                    Instruction::Input => 0,
                    Instruction::Product { .. } => 20 * node.lanes,
                    Instruction::Rescale { .. } => 18 * node.lanes,
                    Instruction::Affine { right, .. } => {
                        (if right.is_some() { 10 } else { 6 }) * node.lanes
                    }
                    Instruction::LaneAffine(..) => 6 * node.lanes,
                    Instruction::RowMap(input, ..) => {
                        4 * self.nodes[input.0 as usize].unwrap().lanes + 2 * node.lanes
                    }
                    Instruction::FromBit(input) => {
                        let bits = usize::from(self.bit_widths[input.0 as usize]);
                        (if bits <= 8 { 16 * bits + 10 } else { 42 }) * node.lanes
                    }
                    Instruction::Sign(_) | Instruction::Bits(_) => 34 * node.lanes,
                    Instruction::BitMap(..) => 6 * node.lanes,
                    Instruction::BitOr(..) => 20 * node.lanes,
                    Instruction::BitXor(..) => 10 * node.lanes,
                };
                (live + temporary) * 8
            })
            .max()
            .unwrap_or(0);
        FIXED_RESERVE_BYTES + metadata_bytes(self) + 3 * std::mem::size_of::<Frame>() + working
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
    pub fn encode(self) -> Result<crate::puma::wire::Encoded<Self>, Error> {
        let inner = self.inner.encode_bytes()?;
        let mut bytes = crate::puma::wire::allocate(48 + inner.len())?;
        bytes[..4].copy_from_slice(b"PUT1");
        bytes[4..36].copy_from_slice(&self.program);
        bytes[36..44].copy_from_slice(&self.invocation.to_le_bytes());
        bytes[44..48].copy_from_slice(&(self.step as u32).to_le_bytes());
        bytes[48..].copy_from_slice(&inner);
        Ok(crate::puma::wire::Encoded {
            bytes,
            _owner: self,
        })
    }

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
    registers: Vec<Option<Register>>,
    inbox: [Option<Message>; 3],
    invocation: u64,
    step: usize,
    active: bool,
    retired: bool,
    metadata: Option<Reservation>,
}

fn metadata_bytes(program: &Program) -> usize {
    // The existing runtime retains its own 4096-byte fixed reserve. Charge this
    // executor's inline program/register/frame metadata as an additional owner.
    std::mem::size_of::<Executor>()
        + program.register_count * std::mem::size_of::<Option<Register>>()
        + program
            .public_affines
            .iter()
            .map(|v| v.capacity() * std::mem::size_of::<Affine>())
            .sum::<usize>()
}

impl Executor {
    /// The surrounding authenticated channel supplies the peer identity.
    /// Invalid bytes retire the same owner as an invalid typed frame.
    pub fn receive_encoded(
        &mut self,
        raw: &[u8],
        sender: Party,
    ) -> Result<[Option<Frame>; 3], Error> {
        let result = (|| {
            if self.retired {
                return Err(Error::Retired);
            }
            if raw.len() < 48
                || raw.len() > crate::puma::wire::MAX_FRAME_BYTES
                || &raw[..4] != b"PUT1"
            {
                return Err(Error::Shape);
            }
            let program = raw[4..36].try_into().unwrap();
            let invocation = u64::from_le_bytes(raw[36..44].try_into().unwrap());
            let step = u32::from_le_bytes(raw[44..48].try_into().unwrap()) as usize;
            if program != self.program.digest || invocation != self.invocation || step != self.step
            {
                return Err(Error::Binding);
            }
            let inner = Message::decode(&raw[48..], sender, self.party())?;
            let frame = Frame {
                program,
                invocation,
                step,
                inner,
                _metadata: self.payload_usage().reserve(std::mem::size_of::<Frame>())?,
            };
            self.receive_inner(frame)
        })();
        self.checked(result)
    }

    pub fn new(
        mut runtime: PartyRuntime,
        program: Program,
        mut input: ReplicatedShare,
    ) -> Result<Self, Error> {
        let mut metadata = None;
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
            for node in program.nodes[..program.len].iter().flatten() {
                let expanded = match node.instruction {
                    Instruction::Sign(_) | Instruction::Bits(_) => 2 * node.lanes,
                    Instruction::FromBit(input) => {
                        let bits = usize::from(program.bit_widths[input.0 as usize]);
                        node.lanes * if bits > 8 { 2 } else { bits.max(1) }
                    }
                    _ => node.lanes,
                };
                runtime.source.limits.core.admit(expanded)?;
            }
            let usage = runtime.payload_usage();
            // Consuming the original input moves its existing charge into the
            // graph's live-register envelope; other retained owners stay priced.
            let input_credit =
                input.first.charged_bytes(&usage) + input.second.charged_bytes(&usage);
            if usage
                .live_bytes()
                .checked_sub(input_credit)
                .ok_or(Error::Bounds)?
                .checked_add(program.admitted_bytes() - FIXED_RESERVE_BYTES)
                .is_none_or(|v| v > usage.limit_bytes())
            {
                return Err(Error::Bounds);
            }
            let invocation = runtime.source.runtime.next_sequence();
            // Every nested Boolean fill, public pivot and possible helper split
            // is priced before the first instruction consumes randomness.
            runtime.source.admit_work(program.work_budget())?;
            metadata = Some(usage.reserve(metadata_bytes(&program))?);
            let mut registers = Vec::new();
            registers
                .try_reserve_exact(program.register_count)
                .map_err(|_| Error::Allocation)?;
            if registers.capacity() != program.register_count {
                return Err(Error::Allocation);
            }
            registers.resize_with(program.register_count, || None);
            input.charge_to(&usage)?;
            Ok((invocation, registers))
        })();
        let (invocation, mut registers) = match result {
            Ok(v) => v,
            Err(e) => {
                runtime.cancel();
                drop(input);
                drop(program);
                drop(metadata);
                return Err(e);
            }
        };
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
            metadata,
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
        for register in &mut self.registers {
            *register = None;
        }
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
        let Some(Register::Arithmetic(value)) =
            &self.registers[self.program.slots[value.0 as usize] as usize]
        else {
            return Err(Error::Binding);
        };
        let usage = self.payload_usage();
        Ok(ReplicatedShare {
            operation: value.operation,
            party: value.party,
            first: usage.copy(&value.first, value.operation.lanes)?,
            second: usage.copy(&value.second, value.operation.lanes)?,
        })
    }

    fn bounds(&self, value: Value) -> Interval {
        self.program.nodes[value.0 as usize].unwrap().bounds
    }

    fn copy_bit(&self, input: Value) -> Result<boolean::Share, Error> {
        let Some(Register::Bit(value)) =
            &self.registers[self.program.slots[input.0 as usize] as usize]
        else {
            return Err(Error::Binding);
        };
        let usage = self.payload_usage();
        Ok(boolean::Share {
            value: ReplicatedShare {
                operation: value.value.operation,
                party: value.value.party,
                first: usage.copy(&value.value.first, value.value.operation.lanes)?,
                second: usage.copy(&value.value.second, value.value.operation.lanes)?,
            },
            bits: value.bits(),
        })
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
        let lanes = self.program.nodes[self.step].unwrap().lanes;
        let slot = self.program.slots[self.step] as usize;
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
                self.registers[slot] = Some(Register::Arithmetic(value));
            }
            Instruction::Product {
                left,
                right,
                rescale,
                certified,
            } => {
                let a = self.copy(left)?;
                let b = self.copy(right)?;
                let bounds = InputBounds {
                    left: self.bounds(left),
                    right: self.bounds(right),
                };
                let all = [bounds; MAX_ROW_LANES];
                let runtime = self.runtime.as_mut().unwrap();
                if let Some(raw) = certified {
                    runtime.plan_certified_product(
                        &all[..lanes],
                        rescale,
                        raw,
                        self.program.digest,
                    )?;
                } else if rescale {
                    runtime.plan(&all[..lanes])?;
                } else {
                    runtime.plan_ring_product(&all[..lanes])?;
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
            Instruction::Sign(input) | Instruction::Bits(input) => {
                let value = self.copy(input)?;
                let runtime = self.runtime.as_mut().unwrap();
                let kind = if matches!(
                    self.program.nodes[self.step].unwrap().instruction,
                    Instruction::Sign(_)
                ) {
                    Kind::MostSignificantBit
                } else {
                    Kind::Bits
                };
                runtime.plan_conversion(lanes, kind)?;
                let value = runtime.bind_conversion_input(value)?;
                messages[0] = Some(runtime.begin_conversion(value)?);
            }
            Instruction::FromBit(input) => {
                let value = self.copy_bit(input)?;
                let runtime = self.runtime.as_mut().unwrap();
                runtime.plan_boolean_to_arithmetic(&value)?;
                let [a, b] = runtime.begin_boolean_to_arithmetic(value)?;
                messages = [a, b, None];
            }
            Instruction::BitMap(input, map) => {
                let value = self.copy_bit(input)?;
                self.registers[slot] = Some(Register::Bit(
                    self.runtime.as_mut().unwrap().boolean_map(value, map)?,
                ));
            }
            Instruction::BitXor(left, right) => {
                let mut a = self.copy_bit(left)?;
                let mut b = self.copy_bit(right)?;
                self.registers[slot] = Some(Register::Bit(
                    self.runtime
                        .as_mut()
                        .unwrap()
                        .boolean_xor(&mut a, Some(&mut b), 0)?,
                ));
            }
            Instruction::BitOr(left, right) => {
                let a = self.copy_bit(left)?;
                let b = self.copy_bit(right)?;
                let runtime = self.runtime.as_mut().unwrap();
                runtime.plan_and(lanes, a.bits(), b.bits())?;
                let a = runtime.bind_and_operand(Operand::Left, a)?;
                let b = runtime.bind_and_operand(Operand::Right, b)?;
                messages[0] = Some(runtime.begin_and(a, b)?);
            }
            Instruction::RowMap(input, width, map) => {
                let mut value = self.copy(input)?;
                let bounds = self.bounds(input);
                let output = self
                    .runtime
                    .as_mut()
                    .unwrap()
                    .row_map(&mut value, width, map, bounds)?;
                self.registers[slot] = Some(Register::Arithmetic(output));
            }
            Instruction::LaneAffine(input, table) => {
                let mut value = self.copy(input)?;
                let bounds = self.bounds(input);
                let coefficients = &self.program.public_affines[table as usize][..lanes];
                let output = self.runtime.as_mut().unwrap().affine_lanes(
                    &mut value,
                    coefficients,
                    bounds,
                )?;
                self.registers[slot] = Some(Register::Arithmetic(output));
            }
        }
        self.wrap(messages)
    }

    pub fn receive(&mut self, frame: Frame) -> Result<[Option<Frame>; 3], Error> {
        let result = self.receive_inner(frame);
        self.checked(result)
    }

    fn receive_inner(&mut self, mut frame: Frame) -> Result<[Option<Frame>; 3], Error> {
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
            Instruction::Sign(_) | Instruction::Bits(_) => {
                out[0] = runtime.advance_conversion(frame.inner)?
            }
            Instruction::BitOr(_, _) => runtime.receive_and(frame.inner)?,
            Instruction::FromBit(_) => {
                if matches!(
                    frame.inner.header().phase,
                    MessagePhase::BooleanPpaMask
                        | MessagePhase::BooleanPpaReveal
                        | MessagePhase::BooleanPpaRotate
                        | MessagePhase::BooleanAnd
                ) {
                    if frame.inner.header().phase == MessagePhase::BooleanPpaRotate
                        && runtime.status() == Status::AwaitingOpen
                    {
                        if self.inbox[0].is_some() {
                            return Err(Error::Binding);
                        }
                        // The independent rotation may arrive before P2's
                        // masked reveal. Preserve its charged owner until the
                        // public receive barrier admits it.
                        frame.inner.words.charge_to(&runtime.payload_usage())?;
                        self.inbox[0] = Some(frame.inner);
                        return Ok([None, None, None]);
                    }
                    let [a, b] = runtime.advance_boolean_to_arithmetic(frame.inner)?;
                    out = [a, b, None];
                    if runtime.status() == Status::AwaitingReshare {
                        if let Some(pending) = self.inbox[0].take() {
                            let [a, b] = runtime.advance_boolean_to_arithmetic(pending)?;
                            if a.is_some() || b.is_some() {
                                return Err(Error::Phase);
                            }
                        }
                    }
                    return self.wrap(out);
                }
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
            let slot = self.program.slots[self.step] as usize;
            if let Instruction::BitOr(left, right) =
                self.program.nodes[self.step].unwrap().instruction
            {
                let mut a = self.copy_bit(left)?;
                let mut b = self.copy_bit(right)?;
                let runtime = self.runtime.as_mut().unwrap();
                let mut product = runtime.take_boolean_output()?;
                let mut sum = runtime.boolean_xor(&mut a, Some(&mut b), 0)?;
                self.registers[slot] = Some(Register::Bit(runtime.boolean_xor(
                    &mut sum,
                    Some(&mut product),
                    0,
                )?));
            }
            let runtime = self.runtime.as_mut().unwrap();
            if self.registers[slot].is_none() {
                self.registers[slot] = Some(if self.program.nodes[self.step].unwrap().bit {
                    Register::Bit(runtime.take_boolean_output()?)
                } else {
                    Register::Arithmetic(runtime.take_output()?)
                });
            }
            // Bind terminal and intermediate owners to the complete public graph.
            let operation = match self.registers[slot].as_mut().unwrap() {
                Register::Arithmetic(v) => &mut v.operation,
                Register::Bit(v) => &mut v.value.operation,
            };
            let mut h = Sha256::new();
            h.update(operation.digest());
            h.update(self.program.digest);
            h.update(self.invocation.to_le_bytes());
            h.update((self.step as u64).to_le_bytes());
            operation.consumer = Some(h.finalize().into());
            for i in 0..=self.step {
                let last = usize::from(self.program.last_use[i]);
                if last == self.step || (i == self.step && last < self.step) {
                    self.registers[self.program.slots[i] as usize] = None;
                }
            }
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
        let slot = self.program.slots[self.program.len - 1] as usize;
        let Some(Register::Arithmetic(value)) = self.registers[slot].take() else {
            return Err(Error::Phase);
        };
        for register in &mut self.registers {
            *register = None;
        }
        Ok((self.runtime.take().unwrap(), value))
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.cancel();
        drop(std::mem::take(&mut self.registers));
        drop(std::mem::take(&mut self.program.public_affines));
        drop(self.metadata.take());
    }
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;

#[path = "tensor_numeric.rs"]
mod numeric;
