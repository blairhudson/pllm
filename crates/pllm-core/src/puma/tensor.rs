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
use std::sync::Arc;

pub const PROFILE: &str = "pllm.puma.public-tensor-program.q18.v1";
/// Approved normalization adaptation: one positive Q18 unit instead of a
/// source epsilon that encodes to zero. Other source-derived programs retain
/// their original numerical identity.
pub const RMS_NORM_PROFILE: &str = "puma.rmsnorm-epsilon1-q18.v1";
pub const RMS_NORM_EPSILON_Q18: i64 = 1;
/// Opt-in checked operands; never the default private-weight numeric identity.
pub const CHECKED_RANGE_PROFILE: &str = "puma.checked-range-q18.v1";
pub const BOUNDED_EQUALITY_PROFILE: &str = "puma.public-domain-equality.compact-q18.v1";
pub const COMPACT_NONLINEAR_PROFILE: &str = "puma.signed-nonlinear.compact-q18.v1";
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

/// Secret safe operand and per-lane arithmetic validity bit. Neither is opened.
#[derive(Clone, Copy, Debug)]
pub struct CheckedValue {
    pub safe: Value,
    pub valid: Value,
}

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
    CompactSign(Value, u8),
    FromBit(Value),
    Bits(Value),
    LowBits(Value, u8),
    BitMap(Value, BitMap),
    BitOr(Value, Value),
    BitOrLow(Value, Value, u8),
    BitXor(Value, Value),
    BitConstantXor(Value, u64),
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
            | Self::BitOrLow(left, right, _)
            | Self::BitXor(left, right) => [Some(left), Some(right)],
            Self::Rescale { input, .. }
            | Self::Sign(input)
            | Self::CompactSign(input, _)
            | Self::FromBit(input)
            | Self::Bits(input)
            | Self::LowBits(input, _)
            | Self::BitMap(input, _)
            | Self::BitConstantXor(input, _)
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
    checked_validity: Option<Value>,
    last_use: [u8; MAX_NODES],
    slots: [u8; MAX_NODES],
    register_count: usize,
    bit_widths: [u8; MAX_NODES],
    public_affines: [Vec<Affine>; 2],
    public_affine_count: usize,
    work: WorkBudget,
}

pub struct Builder(Program);

// Shared by native tile construction and its allocation-free equality work
// query. Keep both constructor geometry checks; no Program is needed to price it.
fn validate_tile_lanes(lanes: usize) -> Result<(), Error> {
    if !(1..=MAX_TILE_LANES).contains(&lanes) || !(1..=MAX_ROW_LANES).contains(&lanes) {
        return Err(Error::Bounds);
    }
    Ok(())
}

impl Builder {
    pub fn new(lanes: usize, input: Interval) -> Result<Self, Error> {
        validate_tile_lanes(lanes)?;
        Self::row(lanes, input)
    }

    fn row(lanes: usize, input: Interval) -> Result<Self, Error> {
        #[cfg(test)]
        work_preflight::observed_row_entry();
        if !(1..=MAX_ROW_LANES).contains(&lanes) {
            return Err(Error::Bounds);
        }
        let mut program = Program {
            nodes: [None; MAX_NODES],
            len: 1,
            lanes,
            digest: [0; 32],
            numeric_profile: super::NUMERIC_PROFILE,
            checked_validity: None,
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
        let (value, node, width) = attention_work::checked_node(
            &self.0.nodes,
            &self.0.bit_widths,
            self.0.len,
            self.0.lanes,
            instruction,
            bounds,
            bit,
        )?;
        self.0.bit_widths[self.0.len] = width;
        self.0.nodes[self.0.len] = Some(node);
        self.0.len += 1;
        Ok(value)
    }

    pub fn affine(
        &mut self,
        left: Value,
        right: Option<Value>,
        coefficients: Affine,
    ) -> Result<Value, Error> {
        attention_work::SoftmaxSink::affine(self, left, right, coefficients)
    }

    pub fn product(&mut self, left: Value, right: Value, rescale: bool) -> Result<Value, Error> {
        attention_work::SoftmaxSink::product(self, left, right, rescale)
    }

    pub fn rescale(&mut self, input: Value, shift: u8) -> Result<Value, Error> {
        attention_work::SoftmaxSink::rescale(self, input, shift)
    }

    pub fn square(&mut self, input: Value) -> Result<Value, Error> {
        attention_work::SoftmaxSink::square(self, input)
    }

    pub fn row_map(&mut self, input: Value, width: usize, map: RowMap) -> Result<Value, Error> {
        attention_work::SoftmaxSink::row_map(self, input, width, map)
    }

    pub fn affine_lanes(&mut self, input: Value, coefficients: &[Affine]) -> Result<Value, Error> {
        if coefficients.len() != self.node(input, false)?.lanes
            || self.0.public_affine_count == 2
            || self.0.len == MAX_NODES
        {
            return Err(Error::Bounds);
        }
        let bounds = self.node(input, false)?.bounds;
        let bounds = rms_work::affine_bounds(bounds, coefficients.iter().copied())?;
        let slot = self.0.public_affine_count;
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
        attention_work::SoftmaxSink::sign(self, input)
    }

    pub fn from_bit(&mut self, input: Value) -> Result<Value, Error> {
        attention_work::SoftmaxSink::from_bit(self, input)
    }

    pub fn from_boolean(&mut self, input: Value) -> Result<Value, Error> {
        attention_work::SoftmaxSink::from_boolean(self, input)
    }

    fn bit_width(&self, input: Value) -> Result<u8, Error> {
        self.node(input, true)?;
        Ok(self.0.bit_widths[input.0 as usize])
    }

    pub fn bits(&mut self, input: Value) -> Result<Value, Error> {
        attention_work::SoftmaxSink::bits(self, input)
    }

    pub fn bit_map(&mut self, input: Value, map: BitMap) -> Result<Value, Error> {
        attention_work::SoftmaxSink::bit_map(self, input, map)
    }

    pub fn bit_or(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        attention_work::SoftmaxSink::bit_or(self, left, right)
    }

    pub fn bit_xor(&mut self, left: Value, right: Value) -> Result<Value, Error> {
        attention_work::SoftmaxSink::bit_xor(self, left, right)
    }

    /// Retain secret arithmetic validity alongside the ordinary output. The
    /// caller must AND every lane into request-wide sticky validity and disclose
    /// only that final aggregate to the client before releasing any tokens/state.
    pub fn finish_checked(mut self, output: Value, valid: Value) -> Result<Program, Error> {
        rms_work::checked_output(&self, output, valid)?;
        self.0.checked_validity = Some(valid);
        if self.0.numeric_profile != COMPACT_NONLINEAR_PROFILE {
            self.0.numeric_profile = CHECKED_RANGE_PROFILE;
        }
        self.finish(output)
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
        if let Some(valid) = self.0.checked_validity {
            h.update(b"checked-validity-output.v1");
            h.update([valid.0]);
        }
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
                Instruction::CompactSign(input, bits) => h.update([15, input.0, bits]),
                Instruction::FromBit(input) => h.update([5, input.0]),
                Instruction::Bits(input) => h.update([6, input.0]),
                Instruction::LowBits(input, bits) => h.update([13, input.0, bits]),
                Instruction::BitMap(input, map) => {
                    h.update([7, input.0]);
                    map.hash(&mut h);
                }
                Instruction::BitOr(a, b) => h.update([8, a.0, b.0]),
                Instruction::BitOrLow(a, b, bits) => h.update([14, a.0, b.0, bits]),
                Instruction::BitXor(a, b) => h.update([9, a.0, b.0]),
                Instruction::BitConstantXor(input, constant) => {
                    h.update([12, input.0]);
                    h.update(constant.to_le_bytes());
                }
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
        if let Some(valid) = self.0.checked_validity {
            self.0.last_use[valid.0 as usize] = self.0.len as u8;
        }
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

fn activation_predicates<S: attention_work::SoftmaxSink>(
    b: &mut S,
    input: Value,
    pivots: [i64; 3],
    compact: bool,
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
        let bit = if compact {
            b.sign_compact(delta)?
        } else {
            b.sign(delta)?
        };
        predicates[i] = b.from_bit(bit)?;
    }
    Ok(predicates)
}

impl Program {
    /// The approved source-derived profile, not the unavailable evaluated IR.
    /// Coefficients/pivots are the binary32/toward-zero Q18 constants frozen in
    /// paper/reproduction-contracts/shared-profile-resolution.md, section 3.3.
    pub fn activation(kind: Activation, lanes: usize, input: Interval) -> Result<Self, Error> {
        Self::activation_impl(kind, lanes, input, false, false)
    }

    /// Preserve the source piecewise equations while excluding zero/identity
    /// tails from polynomial powers. Decisions remain shared, with one fresh
    /// exact guard product. The eager program retains its instruction identity.
    pub fn activation_guarded(
        kind: Activation,
        lanes: usize,
        input: Interval,
    ) -> Result<Self, Error> {
        Self::activation_impl(kind, lanes, input, true, false)
    }

    fn activation_impl(
        kind: Activation,
        lanes: usize,
        input: Interval,
        guarded: bool,
        compact: bool,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        activation_work::entered(kind, guarded);
        let mut b = Builder::new(lanes, input)?;
        if compact {
            b.0.numeric_profile = COMPACT_NONLINEAR_PROFILE;
        }
        let output = activation_work::emit(&mut b, kind, input, guarded, compact)?;
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
        attention_work::instruction_work(node, &self.bit_widths)
    }
    pub fn output_bounds(&self) -> Interval {
        self.nodes[self.len - 1].unwrap().bounds
    }

    pub fn output_lanes(&self) -> usize {
        self.nodes[self.len - 1].unwrap().lanes
    }

    /// Protected validity output shape, absent for ordinary one-output programs.
    /// Checked-range callers concatenate `output_lanes()` safe words followed by
    /// these validity words; the aggregate is never opened by this executor.
    pub fn validity_lanes(&self) -> Option<usize> {
        self.checked_validity
            .map(|v| self.nodes[v.0 as usize].unwrap().lanes)
    }

    // Joint register/primitive/copy peak for the original instruction schedule.
    fn working_bytes(&self) -> Result<usize, Error> {
        (0..self.len)
            .map(|step| {
                let live = (0..=step)
                    .filter(|&i| i == step || usize::from(self.last_use[i]) >= step)
                    .try_fold(0usize, |n, i| {
                        self.nodes[i]
                            .unwrap()
                            .lanes
                            .checked_mul(2)
                            .and_then(|v| n.checked_add(v))
                            .ok_or(Error::Bounds)
                    })?;
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
                    Instruction::Sign(_)
                    | Instruction::CompactSign(..)
                    | Instruction::Bits(_)
                    | Instruction::LowBits(..) => 34 * node.lanes,
                    Instruction::BitMap(..) => 6 * node.lanes,
                    Instruction::BitOr(..) | Instruction::BitOrLow(..) => 20 * node.lanes,
                    Instruction::BitXor(..) => 10 * node.lanes,
                    Instruction::BitConstantXor(..) => 6 * node.lanes,
                };
                live.checked_add(temporary)
                    .and_then(|n| n.checked_mul(8))
                    .ok_or(Error::Bounds)
            })
            .try_fold(0usize, |peak, n| Ok(peak.max(n?)))
    }

    /// Owner-defined sizes, not a complete enclosing graph's memory admission.
    /// Shared execution grants no credit for caller-retained template ownership.
    pub fn resources(&self) -> Result<Resources, Error> {
        let add = |a: usize, b: usize| a.checked_add(b).ok_or(Error::Bounds);
        let public_table_bytes = self.public_affines.iter().try_fold(0usize, |n, v| {
            add(
                n,
                v.capacity()
                    .checked_mul(std::mem::size_of::<Affine>())
                    .ok_or(Error::Bounds)?,
            )
        })?;
        let immutable_template_bytes = add(
            add(
                std::mem::size_of::<Program>(),
                2 * std::mem::size_of::<usize>(),
            )?,
            public_table_bytes,
        )?;
        let register_container_bytes = self
            .register_count
            .checked_mul(std::mem::size_of::<Option<Register>>())
            .ok_or(Error::Bounds)?;
        let invocation_container_bytes =
            add(std::mem::size_of::<Executor>(), register_container_bytes)?;
        let frame_bytes = 3usize
            .checked_mul(std::mem::size_of::<Frame>())
            .ok_or(Error::Bounds)?;
        let working_bytes = self.working_bytes()?;
        let common = add(
            add(FIXED_RESERVE_BYTES, invocation_container_bytes)?,
            add(frame_bytes, working_bytes)?,
        )?;
        Ok(Resources {
            runtime_fixed_reserve_bytes: FIXED_RESERVE_BYTES,
            immutable_template_bytes,
            public_table_bytes,
            register_container_bytes,
            invocation_container_bytes,
            frame_bytes,
            working_bytes,
            owned_admitted_bytes: add(common, public_table_bytes)?,
            shared_admitted_bytes: add(common, immutable_template_bytes)?,
        })
    }

    /// Conservative owned-executor allowance, including its inline template.
    /// Original exported buffers retain their own ledger charges as usual.
    pub fn admitted_bytes(&self) -> usize {
        // All dimensions/capacities originate in bounded native constructors.
        self.resources()
            .expect("bounded tensor resources")
            .owned_admitted_bytes
    }
}

/// Conservative single-invocation envelope. Sizes count actual Vec capacities.
/// `invocation_container_bytes` includes `register_container_bytes`; do not add
/// them twice. `working_bytes` is the joint peak of live register payload and
/// primitive/operand-copy temporaries, not immutable storage. Frame payloads are
/// included there; `frame_bytes` prices three outer frame containers separately.
/// The owned API stores Program inline; the shared API additionally retains an
/// Arc allocation (including both counters) and its public coefficient backing.
/// Neither API discounts a caller's baseline or another concurrent invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resources {
    pub runtime_fixed_reserve_bytes: usize,
    pub immutable_template_bytes: usize,
    pub public_table_bytes: usize,
    pub register_container_bytes: usize,
    pub invocation_container_bytes: usize,
    pub frame_bytes: usize,
    pub working_bytes: usize,
    pub owned_admitted_bytes: usize,
    pub shared_admitted_bytes: usize,
}

/// Public geometry of the audited, noncompact native constructor paths only.
/// A quote does not prove numeric domains, certificates, instruction/work limits,
/// or source authority. Those remain the original constructor's obligations.
#[derive(Clone, Copy, Debug)]
pub enum Construction {
    /// Builder::new followed only by scalar-affine/product/rescale operations
    /// and finish. Lane-affine tables and arbitrary extensions are NOT covered.
    Builder {
        lanes: usize,
    },
    /// Program::negative only: new -> sign -> from_bit -> finish.
    /// Does not cover negative_compact or arbitrary Builder extensions.
    Negative {
        lanes: usize,
    },
    /// Program::equal_zero only: full-ring bits, fixed six-shift OR reduction,
    /// LowBits(1), from_bit, scalar affine, finish. Not equal_zero_bounded.
    EqualZero {
        lanes: usize,
    },
    GuardedSilu {
        lanes: usize,
    },
    CheckedRange {
        lanes: usize,
    },
    AllValid {
        lanes: usize,
    },
    RmsNormCertified {
        rows: usize,
        width: usize,
        gamma_len: usize,
    },
    RmsNormWithInputNorm {
        rows: usize,
        width: usize,
        gamma_len: usize,
    },
    RotaryHalf {
        rows: usize,
        width: usize,
        cosine_len: usize,
        sine_len: usize,
    },
    SoftmaxCertified {
        rows: usize,
        width: usize,
    },
    RotaryCoefficients {
        theta: f32,
        position: u32,
        width: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstructionQuoteError {
    Geometry(Error),
    /// rotary_coefficients does not check either returned Vec's capacity. Do not
    /// turn requested lengths into a bound on actual retained backing.
    UncheckedRotaryCoefficientCapacity,
}

/// Logical owner/buffer envelope, NOT allocator/RSS or generated stack usage.
/// `new_*` excludes the caller's baseline. `peak_with_borrowed_public_bytes`
/// adds supplied slice elements once, NOT caller Vec capacities/headers or their
/// generation scratch. A caller must retain its full independently priced owner
/// baseline; these fields never authorize a capacity credit or a subtraction.
/// Program moves transfer affine Vec backing: source/destination inline slots
/// overlap, but coefficient payload is not deep-copied by those moves.
/// Scalar arithmetic locals, ABI spill slots, libm/SHA implementation stack and
/// allocator overhead are not request-owned buffers and are NOT stack-quoted.
/// This quotes one construction plus publication into Arc, not concurrent calls,
/// a compiler frontier, caller coefficient generation, or a whole D26 request.
/// Table bytes rely on affine_lanes' existing exact-capacity acceptance check;
/// allocator-internal slack and failed reservations are not RSS-quoted here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstructionQuote {
    /// Largest audited inline Builder/Program Result (including certificates).
    pub return_slot_bytes: usize,
    pub borrowed_public_bytes: usize,
    pub public_table_bytes: usize,
    pub constructor_buffer_bytes: usize,
    pub proof_buffer_bytes: usize,
    pub finish_buffer_bytes: usize,
    pub immutable_template_bytes: usize,
    pub new_build_bytes: usize,
    pub new_finish_bytes: usize,
    pub new_publish_bytes: usize,
    pub new_peak_bytes: usize,
    pub peak_with_borrowed_public_bytes: usize,
}

impl Construction {
    /// Allocation-free. Never invokes a numeric constructor or generates RoPE
    /// coefficients. Re-audit this inventory when the named native paths change.
    pub fn quote(self) -> Result<ConstructionQuote, ConstructionQuoteError> {
        use ConstructionQuoteError::{Geometry, UncheckedRotaryCoefficientCapacity};
        let invalid = || Geometry(Error::Bounds);
        let sum = |parts: &[usize]| {
            parts
                .iter()
                .try_fold(0usize, |a, b| a.checked_add(*b).ok_or_else(invalid))
        };
        let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(invalid);
        let tile = |lanes: usize| {
            if (1..=MAX_TILE_LANES).contains(&lanes) {
                Ok(lanes)
            } else {
                Err(invalid())
            }
        };
        let row = |rows: usize, width: usize| {
            let lanes = mul(rows, width)?;
            if rows == 0 || width == 0 || lanes > MAX_ROW_LANES {
                Err(invalid())
            } else {
                Ok(lanes)
            }
        };
        let affine = std::mem::size_of::<Affine>();
        let interval = std::mem::size_of::<Interval>();
        // numeric.rs: Ranged owns Value + [Option<Interval>; 36]. Round the
        // field sum up by its maximum possible tail padding, independent of
        // Rust field ordering. No public access to private proof representation.
        let boxes = std::mem::size_of::<[Option<Interval>; 36]>();
        let ranged = sum(&[boxes, std::mem::align_of::<Option<Interval>>()])?;
        let corners = std::mem::size_of::<[i128; 4]>();
        let (borrowed, tables, buffers, proofs) = match self {
            Self::Builder { lanes } | Self::CheckedRange { lanes } => {
                tile(lanes)?;
                (0, 0, 0, 0)
            }
            Self::Negative { lanes } => {
                tile(lanes)?;
                // tensor_numeric.rs: new -> sign -> from_bit/from_boolean ->
                // finish. new/row initialize the inline Program; node/push and
                // bit_width touch only its inline arrays. No bit-map/shift array,
                // coefficient slice, affine Vec allocation/copy or proof buffer.
                // finish's SHA/available_at and owner/return/publication overlap
                // use the common inventory below, not execution scratch.
                (0, 0, 0, 0)
            }
            Self::EqualZero { lanes } => {
                tile(lanes)?;
                // tensor_numeric.rs: new -> bits -> six (ShiftRight, bit_or)
                // pairs -> LowBits(1) -> from_bit/from_boolean -> scalar affine
                // -> finish. bit_map/BitMap::width, bit_or and affine's scalar
                // output_bounds introduce no tables or proof arrays. Charge the
                // [32,16,8,4,2,1] expression slot and consuming iterator owner
                // separately, including its cursor metadata, even if moved.
                // Shifts infer u8 from BitMap::ShiftRight (not i32/usize).
                let buffers = sum(&[
                    std::mem::size_of::<[u8; 6]>(),
                    std::mem::size_of::<std::array::IntoIter<u8, 6>>(),
                ])?;
                (0, 0, buffers, 0)
            }
            Self::AllValid { lanes } => {
                row(1, lanes)?;
                (0, 0, 0, 0)
            }
            Self::GuardedSilu { lanes } => {
                tile(lanes)?;
                // Union of named array slots across the nested calls (not a
                // claim that they all coexist): impl's two pivot bindings,
                // predicate parameter/iterator, bound-proof pivot return/local;
                // low/high proof return/local, branch/iterator, and term
                // expression/local/iterator. Predicate return/early/local slots
                // and low/high executable polynomial term arrays are separate.
                let buffers = sum(&[
                    std::mem::size_of::<[(Value, i64); 2]>(),
                    std::mem::size_of::<[(Value, i64); 4]>(),
                    mul(6, std::mem::size_of::<[i64; 3]>())?,
                    mul(2, std::mem::size_of::<[i64; 4]>())?,
                    mul(2, std::mem::size_of::<[i64; 5]>())?,
                    mul(2, std::mem::size_of::<[(i64, i64, bool); 2]>())?,
                    mul(3, std::mem::size_of::<[(Interval, i64); 4]>())?,
                    mul(4, std::mem::size_of::<[Value; 3]>())?,
                    std::mem::size_of::<Option<[Value; 3]>>(),
                ])?;
                (0, 0, buffers, 0)
            }
            Self::RotaryHalf {
                rows,
                width,
                cosine_len,
                sine_len,
            } => {
                let lanes = tile(mul(rows, width)?)?;
                if rows == 0
                    || width == 0
                    || width % 2 != 0
                    || cosine_len != lanes / 2
                    || sine_len != lanes / 2
                {
                    return Err(Geometry(Error::Shape));
                }
                // Array initialization/move overlap plus the two exact-capacity
                // Vec copies retained by affine_lanes (the table field).
                (
                    mul(lanes, 8)?,
                    mul(mul(2, lanes)?, affine)?,
                    mul(mul(2, MAX_TILE_LANES)?, affine)?,
                    0,
                )
            }
            Self::RmsNormCertified {
                rows,
                width,
                gamma_len,
            }
            | Self::RmsNormWithInputNorm {
                rows,
                width,
                gamma_len,
            } => {
                let lanes = row(rows, width)?;
                if gamma_len != width {
                    return Err(Geometry(Error::Shape));
                }
                // rsqrt_ranged: 12 named Ranged slots; chained rescale temporary
                // and result; RMS normalization: inverse, input/row temporary,
                // product result; one helper return overlap = 19 Ranged slots.
                // Five extra arrays: two helper arrays, denominator array plus
                // consuming iterator copy, and row boxes. The proof-cell term
                // array and iterator copy, borrowed power array and iterator
                // copy, and product corner array are charged separately.
                let proof = sum(&[
                    mul(19, ranged)?,
                    mul(5, boxes)?,
                    corners,
                    mul(2, std::mem::size_of::<[(Interval, i64); 3]>())?,
                    mul(2, std::mem::size_of::<[(usize, i64); 3]>())?,
                ])?;
                // from_fn return plus destination scales array; moving the
                // array does not duplicate the separately charged Vec table.
                (
                    mul(width, 8)?,
                    mul(lanes, affine)?,
                    mul(mul(2, MAX_ROW_LANES)?, affine)?,
                    proof,
                )
            }
            Self::SoftmaxCertified { rows, width } => {
                row(rows, width)?;
                // reciprocal: 11 named Ranged values + final product result +
                // helper return overlap. Two helper arrays cover product/raw
                // or rescale/iterator overlap. Scalar cell proof has no Vec.
                (
                    0,
                    0,
                    0,
                    sum(&[mul(13, ranged)?, mul(2, boxes)?, corners, mul(2, interval)?])?,
                )
            }
            Self::RotaryCoefficients {
                theta,
                position,
                width,
            } => {
                if !theta.is_finite()
                    || theta < 1.0
                    || width == 0
                    || width > MAX_TILE_LANES
                    || width % 2 != 0
                    || position > 512
                {
                    return Err(invalid());
                }
                return Err(UncheckedRotaryCoefficientCapacity);
            }
        };
        // Builder::row local Program and returned Builder overlap. Construction
        // retains one Builder; finish retains its inline Program return slot.
        // Publish moves that Program into Arc; backing is transferred, not copied.
        let inline = std::mem::size_of::<Program>();
        let builder = std::mem::size_of::<Builder>();
        let return_slot = match self {
            Self::RmsNormCertified { .. } | Self::RmsNormWithInputNorm { .. } => {
                std::mem::size_of::<Result<(Program, u128), Error>>()
                    .max(std::mem::size_of::<Result<(Builder, Value, u128), Error>>())
            }
            Self::SoftmaxCertified { .. } => std::mem::size_of::<Result<(Program, u64), Error>>()
                .max(std::mem::size_of::<Result<(Program, Option<u64>), Error>>()),
            _ => std::mem::size_of::<Result<Program, Error>>(),
        }
        .max(std::mem::size_of::<Result<Builder, Error>>());
        let finish = sum(&[
            std::mem::size_of::<Sha256>(),
            std::mem::size_of::<[usize; MAX_NODES]>(),
        ])?;
        let immutable = sum(&[inline, 2 * std::mem::size_of::<usize>(), tables])?;
        let build = sum(&[builder, return_slot, tables, buffers, proofs])?;
        let finish_phase = sum(&[builder, return_slot, tables, buffers, proofs, finish])?;
        let publish = sum(&[return_slot, immutable, buffers, proofs])?;
        let peak = build.max(finish_phase).max(publish);
        Ok(ConstructionQuote {
            return_slot_bytes: return_slot,
            borrowed_public_bytes: borrowed,
            public_table_bytes: tables,
            constructor_buffer_bytes: buffers,
            proof_buffer_bytes: proofs,
            finish_buffer_bytes: finish,
            immutable_template_bytes: immutable,
            new_build_bytes: build,
            new_finish_bytes: finish_phase,
            new_publish_bytes: publish,
            new_peak_bytes: peak,
            peak_with_borrowed_public_bytes: sum(&[peak, borrowed])?,
        })
    }
}

#[cfg(test)]
#[path = "tensor_construction_tests.rs"]
mod construction_tests;

fn polynomial<S: attention_work::SoftmaxSink>(
    b: &mut S,
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

fn select<S: attention_work::SoftmaxSink>(
    b: &mut S,
    bit: Value,
    yes: Value,
    no: Value,
) -> Result<Value, Error> {
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
/// Outer transport must authenticate each private channel. Only bounded equality
/// uses PUT2/PUC1 nominal-bit packing; all other programs retain PUT1/PUM1.
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
    compact_bits: Option<u8>,
    // Last field: payload erasure/deallocation precedes metadata refund.
    _metadata: Reservation,
}

impl Frame {
    pub fn encode(self) -> Result<crate::puma::wire::Encoded<Self>, Error> {
        let inner = match self.compact_bits {
            Some(bits) => self.inner.encode_compact(bits)?,
            None => self.inner.encode_bytes()?,
        };
        let mut bytes = crate::puma::wire::allocate(48 + inner.len())?;
        bytes[..4].copy_from_slice(if self.compact_bits.is_some() {
            b"PUT2"
        } else {
            b"PUT1"
        });
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
    program: ProgramOwner,
    registers: Vec<Option<Register>>,
    inbox: [Option<Message>; 3],
    invocation: u64,
    step: usize,
    active: bool,
    retired: bool,
    metadata: Option<Reservation>,
}

// Owned compatibility path deliberately stays inline: no new infallible heap
// allocation or allocation-failure boundary is introduced by Executor::new.
#[allow(clippy::large_enum_variant)]
enum ProgramOwner {
    Owned(Program),
    Shared(Arc<Program>),
    Retired,
}

impl std::ops::Deref for ProgramOwner {
    type Target = Program;
    fn deref(&self) -> &Program {
        match self {
            Self::Owned(p) => p,
            Self::Shared(p) => p,
            Self::Retired => unreachable!("retired template accessed"),
        }
    }
}

fn metadata_bytes(program: &ProgramOwner) -> Result<usize, Error> {
    let resources = program.resources()?;
    resources
        .invocation_container_bytes
        .checked_add(match program {
            ProgramOwner::Shared(_) => resources.immutable_template_bytes,
            ProgramOwner::Owned(_) => resources.public_table_bytes,
            ProgramOwner::Retired => return Err(Error::Retired),
        })
        .ok_or(Error::Bounds)
}

impl Executor {
    fn compact(&self) -> bool {
        self.program.numeric_profile == BOUNDED_EQUALITY_PROFILE
            || (self.program.numeric_profile == COMPACT_NONLINEAR_PROFILE
                && match self.program.nodes[self.step].map(|n| n.instruction) {
                    Some(
                        Instruction::CompactSign(..)
                        | Instruction::LowBits(..)
                        | Instruction::BitOrLow(..),
                    ) => true,
                    Some(Instruction::FromBit(input)) => {
                        self.program.bit_widths[input.0 as usize] == 1
                    }
                    _ => false,
                })
    }

    fn compact_layout(&self) -> Result<(u8, usize), Error> {
        let node = self
            .program
            .nodes
            .get(self.step)
            .copied()
            .flatten()
            .ok_or(Error::Phase)?;
        match node.instruction {
            Instruction::CompactSign(..) => {
                let super::Phase::Conversion(state) =
                    &self.runtime.as_ref().ok_or(Error::Retired)?.phase
                else {
                    return Err(Error::Phase);
                };
                let (_, _, lanes, bits) = state.receive_layout()?;
                Ok((bits, lanes))
            }
            Instruction::LowBits(_, bits) => {
                let super::Phase::Conversion(state) =
                    &self.runtime.as_ref().ok_or(Error::Retired)?.phase
                else {
                    return Err(Error::Phase);
                };
                let (_, _, lanes, expected_bits) = state.receive_layout()?;
                if bits != expected_bits {
                    return Err(Error::Binding);
                }
                Ok((bits, lanes))
            }
            Instruction::BitOrLow(a, b, _) => Ok((
                self.program.bit_widths[a.0 as usize].max(self.program.bit_widths[b.0 as usize]),
                node.lanes,
            )),
            Instruction::FromBit(_) => Ok((64, node.lanes)),
            _ => Err(Error::Phase),
        }
    }

    fn validate_compact_header(&self, header: Header) -> Result<(), Error> {
        match &self.runtime.as_ref().ok_or(Error::Retired)?.phase {
            super::Phase::Conversion(state) => {
                let (binding, phase, _, _) = state.receive_layout()?;
                if header.binding != binding || header.phase != phase {
                    return Err(Error::Binding);
                }
                if header.from != self.party().next() || header.to != self.party() {
                    return Err(Error::Role);
                }
            }
            super::Phase::Boolean(super::boolean::State::Rotating(gate)) => {
                if header.binding != gate.receive_layout().0
                    || header.phase != MessagePhase::BooleanAnd
                {
                    return Err(Error::Binding);
                }
                if header.from != self.party().next() || header.to != self.party() {
                    return Err(Error::Role);
                }
            }
            super::Phase::BooleanToArithmetic(state) => state.validate_compact_header(header)?,
            _ => return Err(Error::Phase),
        }
        Ok(())
    }

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
                || &raw[..4] != if self.compact() { b"PUT2" } else { b"PUT1" }
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
            let compact_bits = if self.compact() {
                Some(self.compact_layout()?.0)
            } else {
                None
            };
            let inner = if let Some(bits) = compact_bits {
                let (_, lanes) = self.compact_layout()?;
                let header =
                    Message::compact_header(&raw[48..], sender, self.party(), bits, lanes)?;
                self.validate_compact_header(header)?;
                Message::decode_compact(
                    &raw[48..],
                    sender,
                    self.party(),
                    bits,
                    lanes,
                    &self.payload_usage(),
                )?
            } else {
                Message::decode(&raw[48..], sender, self.party())?
            };
            let frame = Frame {
                program,
                invocation,
                step,
                inner,
                compact_bits,
                _metadata: self.payload_usage().reserve(std::mem::size_of::<Frame>())?,
            };
            self.receive_inner(frame)
        })();
        self.checked(result)
    }

    pub fn new(
        runtime: PartyRuntime,
        program: Program,
        input: ReplicatedShare,
    ) -> Result<Self, Error> {
        Self::new_inner(runtime, ProgramOwner::Owned(program), input)
    }

    /// Execute immutable shared instructions without cloning coefficient tables.
    /// Caller constructs the Arc before entering this consuming API. Admission
    /// conservatively includes its full backing for EACH invocation; enclosing
    /// owners must separately retain/charge any external baseline through last use.
    pub fn new_shared(
        runtime: PartyRuntime,
        program: Arc<Program>,
        input: ReplicatedShare,
    ) -> Result<Self, Error> {
        Self::new_inner(runtime, ProgramOwner::Shared(program), input)
    }

    fn new_inner(
        mut runtime: PartyRuntime,
        program: ProgramOwner,
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
                    Instruction::Sign(_)
                    | Instruction::CompactSign(..)
                    | Instruction::Bits(_)
                    | Instruction::LowBits(..) => 2 * node.lanes,
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
                .checked_add(
                    match &program {
                        ProgramOwner::Shared(_) => program.resources()?.shared_admitted_bytes,
                        _ => program.resources()?.owned_admitted_bytes,
                    } - FIXED_RESERVE_BYTES,
                )
                .is_none_or(|v| v > usage.limit_bytes())
            {
                return Err(Error::Bounds);
            }
            let invocation = runtime.source.runtime.next_sequence();
            // Every nested Boolean fill, public pivot and possible helper split
            // is priced before the first instruction consumes randomness.
            runtime.source.admit_work(program.work_budget())?;
            metadata = Some(usage.reserve(metadata_bytes(&program)?)?);
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
                    compact_bits: if self.compact() {
                        Some(self.compact_layout()?.0)
                    } else {
                        None
                    },
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
            Instruction::Sign(input)
            | Instruction::CompactSign(input, _)
            | Instruction::Bits(input)
            | Instruction::LowBits(input, _) => {
                let value = self.copy(input)?;
                let runtime = self.runtime.as_mut().unwrap();
                let kind = match self.program.nodes[self.step].unwrap().instruction {
                    Instruction::Sign(_) => Kind::MostSignificantBit,
                    Instruction::CompactSign(_, bits) => Kind::CompactMostSignificantBit(bits),
                    Instruction::LowBits(_, bits) => Kind::LowBits(bits),
                    _ => Kind::Bits,
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
            Instruction::BitConstantXor(input, constant) => {
                let mut value = self.copy_bit(input)?;
                self.registers[slot] = Some(Register::Bit(
                    self.runtime
                        .as_mut()
                        .unwrap()
                        .boolean_xor(&mut value, None, constant)?,
                ));
            }
            Instruction::BitOr(left, right) | Instruction::BitOrLow(left, right, _) => {
                let a = self.copy_bit(left)?;
                let b = self.copy_bit(right)?;
                let runtime = self.runtime.as_mut().unwrap();
                if matches!(
                    self.program.nodes[self.step].unwrap().instruction,
                    Instruction::BitOrLow(..)
                ) && matches!(
                    self.program.numeric_profile,
                    BOUNDED_EQUALITY_PROFILE | COMPACT_NONLINEAR_PROFILE
                ) {
                    runtime.plan_and_compact(lanes, a.bits(), b.bits())?;
                } else {
                    runtime.plan_and(lanes, a.bits(), b.bits())?;
                }
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
        if self.compact() {
            let (bits, lanes) = self.compact_layout()?;
            if frame.compact_bits != Some(bits) || frame.inner.words().len() != lanes {
                return Err(Error::Binding);
            }
            let mask = u64::MAX >> (64 - bits);
            if frame.inner.words().iter().any(|&word| word & !mask != 0) {
                return Err(Error::Bounds);
            }
            self.validate_compact_header(frame.inner.header())?;
        } else if frame.compact_bits.is_some() {
            return Err(Error::Binding);
        }
        let runtime = self.runtime.as_mut().unwrap();
        let instruction = self.program.nodes[self.step].unwrap().instruction;
        let mut out = [None, None, None];
        match instruction {
            Instruction::Sign(_)
            | Instruction::CompactSign(..)
            | Instruction::Bits(_)
            | Instruction::LowBits(..) => out[0] = runtime.advance_conversion(frame.inner)?,
            Instruction::BitOr(_, _) | Instruction::BitOrLow(..) => {
                runtime.receive_and(frame.inner)?
            }
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
            if let Instruction::BitOr(left, right) | Instruction::BitOrLow(left, right, _) =
                self.program.nodes[self.step].unwrap().instruction
            {
                let mut a = self.copy_bit(left)?;
                let mut b = self.copy_bit(right)?;
                let runtime = self.runtime.as_mut().unwrap();
                let mut product = runtime.take_boolean_output()?;
                let mut sum = runtime.boolean_xor(&mut a, Some(&mut b), 0)?;
                let mut output = runtime.boolean_xor(&mut sum, Some(&mut product), 0)?;
                if let Instruction::BitOrLow(_, _, bits) =
                    self.program.nodes[self.step].unwrap().instruction
                {
                    // Fuse the public XOR-linear low-bit map into the existing
                    // final XOR owner: no extra buffer, PRSS draw or ordinal.
                    let mask = u64::MAX >> (64 - bits);
                    for word in output
                        .value
                        .first
                        .iter_mut()
                        .chain(output.value.second.iter_mut())
                    {
                        *word &= mask;
                    }
                    output.bits = bits;
                }
                self.registers[slot] = Some(Register::Bit(output));
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
        if self.program.checked_validity.is_some() {
            return Err(Error::Binding);
        }
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

    /// Transfer both protected outputs. No validity or failure location is opened
    /// to a worker. A checked program cannot use `finish` to discard validity.
    pub fn finish_checked(
        mut self,
    ) -> Result<(PartyRuntime, ReplicatedShare, ReplicatedShare), Error> {
        if !self.complete() {
            return Err(Error::Phase);
        }
        let valid = self.program.checked_validity.ok_or(Error::Binding)?;
        let output = self.program.len - 1;
        let slot = self.program.slots[output] as usize;
        let valid_slot = self.program.slots[valid.0 as usize] as usize;
        let Some(Register::Arithmetic(validity)) = self.registers[valid_slot].take() else {
            return Err(Error::Phase);
        };
        let Some(Register::Arithmetic(value)) = self.registers[slot].take() else {
            return Err(Error::Phase);
        };
        for register in &mut self.registers {
            *register = None;
        }
        Ok((self.runtime.take().unwrap(), value, validity))
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.cancel();
        drop(std::mem::take(&mut self.registers));
        drop(std::mem::replace(&mut self.program, ProgramOwner::Retired));
        drop(self.metadata.take());
    }
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;

#[path = "tensor_compact.rs"]
mod compact;

#[path = "tensor_compact_rows.rs"]
mod compact_rows;

#[path = "tensor_numeric.rs"]
mod numeric;

#[path = "tensor_negative_work.rs"]
mod negative_work;
pub use negative_work::NegativeWorkResources;
#[cfg(test)]
#[path = "tensor_negative_work_tests.rs"]
mod negative_work_tests;

#[path = "tensor_rotary_work.rs"]
mod rotary_work;
pub use rotary_work::RotaryWorkResources;
#[cfg(test)]
#[path = "tensor_rotary_work_tests.rs"]
mod rotary_work_tests;

#[path = "tensor_work_preflight.rs"]
mod work_preflight;
pub use work_preflight::{
    AdmittedFactory, AdmittedWorkSchedule, CallerWorkLedger, ClosedFactory, FactoryOutput,
    FactoryParameters, FactoryWorkSummary, ProducedCertificate, WorkPreflight, WorkPreflightError,
    WorkPreflightResources, WorkPreflightWorkspace,
};

#[cfg(test)]
#[path = "tensor_work_preflight_tests.rs"]
mod work_preflight_tests;

#[path = "tensor_attention_work.rs"]
mod attention_work;
pub use attention_work::SoftmaxWorkWorkspace;

#[cfg(test)]
#[path = "tensor_attention_work_tests.rs"]
mod attention_work_tests;

#[path = "tensor_rms_work.rs"]
mod rms_work;
pub use rms_work::RmsWorkWorkspace;

#[cfg(test)]
#[path = "tensor_rms_work_tests.rs"]
mod rms_work_tests;

#[path = "activation_work.rs"]
mod activation_work;
pub use activation_work::ActivationWorkWorkspace;
