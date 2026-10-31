//! Expand original ScaledSupport stages, not a second physical schedule.
//! Adds the evaluator's fused product sums, explicit zero accumulators and beta
//! bias. Concrete scales/keys/encoded masks remain mandatory external bindings.
use crate::encrypted_instruction_lowering::{Branch, Instruction, Refusal, Selector};
use pllm_core::slot_matrix_layout::{BlockPcPhysicalPlan, PhysicalMaskBranch, PhysicalOp};

fn branch(b: PhysicalMaskBranch) -> Branch {
    match b { PhysicalMaskBranch::Support => Branch::Support,
        PhysicalMaskBranch::Wrap => Branch::Wrap, PhysicalMaskBranch::Remainder => Branch::Remainder }
}
pub(crate) struct Cursor<'a> {
    plan: &'a BlockPcPhysicalPlan,
    stage: usize,
    output: usize,
    pending_sum: bool,
    bias: usize,
    p0: usize,
    p1: usize,
    sum: usize,
    partial: usize,
    raw: [usize; 2],
    rescaled: [usize; 2],
    rotated: [usize; 2],
    aligned: usize,
    accum: [usize; 2],
}
impl<'a> Cursor<'a> {
    pub(crate) fn new(plan: &'a BlockPcPhysicalPlan) -> Self {
        Self { plan, stage: 0, output: 4, pending_sum: false, bias: 0,
            p0: usize::MAX, p1: usize::MAX, sum: usize::MAX, partial: usize::MAX,
            raw: [usize::MAX; 2], rescaled: [usize::MAX; 2], rotated: [usize::MAX; 2],
            aligned: usize::MAX, accum: [2, 3] }
    }
    pub(crate) fn next(&mut self) -> Result<Option<Instruction>, Refusal> {
        if self.plan.stages().len() != 78 {
            return Err(Refusal::LegacyUnresolved);
        }
        let ordinal = self.output;
        let ins = if self.pending_sum {
            self.pending_sum = false;
            self.sum = ordinal;
            Instruction::add(self.p0, self.p1)
        } else if let Some(s) = self.plan.stages().get(self.stage) {
            self.stage += 1;
            let q = s.output_frame; let a = s.displacement;
            if q >= 2 || a >= 4 { return Err(Refusal::Source); }
            match s.op {
                PhysicalOp::WeightProduct { input_frame: j } => {
                    if j > 1 { return Err(Refusal::Source); }
                    if j == 0 { self.p0 = ordinal; } else { self.p1 = ordinal; self.pending_sum = true; }
                    Instruction::unary(Selector::Weight { q, a, j }, j)
                }
                PhysicalOp::ProductSumBoundary => {
                    self.partial = ordinal;
                    Instruction::unary(Selector::Drop { q, a, boundary: 1, branch: None }, self.sum)
                }
                PhysicalOp::SupportMask { branch: b } => {
                    let b = branch(b); let index = usize::from(b == Branch::Remainder);
                    self.raw[index] = ordinal;
                    // BOTH branches use complete original P, never p0 or U.
                    Instruction::unary(Selector::Mask { q, a, branch: b }, self.partial)
                }
                PhysicalOp::MaskRescale { branch: b } => {
                    let b = branch(b); let index = usize::from(b == Branch::Remainder);
                    self.rescaled[index] = ordinal;
                    if b == Branch::Support { self.aligned = ordinal; }
                    Instruction::unary(Selector::Drop { q, a, boundary: 2, branch: Some(b) }, self.raw[index])
                }
                PhysicalOp::RotateMasked { shift } => {
                    if shift != (4 - a) as i32 || a == 0 { return Err(Refusal::Source); }
                    self.rotated[0] = ordinal;
                    Instruction::unary(Selector::Rotation { q, a, shift: shift as isize }, self.rescaled[0])
                }
                PhysicalOp::RotateRemainder { shift } => {
                    if shift != -(a as i32) || a == 0 { return Err(Refusal::Source); }
                    self.rotated[1] = ordinal;
                    Instruction::unary(Selector::Rotation { q, a, shift: shift as isize }, self.rescaled[1])
                }
                PhysicalOp::AddAligned => {
                    self.aligned = ordinal;
                    Instruction::add(self.rotated[0], self.rotated[1])
                }
                PhysicalOp::Accumulate => {
                    let previous = self.accum[q]; self.accum[q] = ordinal;
                    Instruction::add(previous, self.aligned)
                }
                PhysicalOp::Mask | PhysicalOp::SubtractMasked => return Err(Refusal::LegacyUnresolved),
            }
        } else {
            if self.bias == 2 { return Ok(None); }
            let q = self.bias; self.bias += 1;
            Instruction::unary(Selector::Bias(q), self.accum[q])
        };
        self.output += 1;
        Ok(Some(ins))
    }
}

#[cfg(test)]
#[path = "slot_pc_trace_tests.rs"]
mod tests;
