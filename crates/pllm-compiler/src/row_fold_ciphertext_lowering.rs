//! Narrow hook over the ORIGINAL immutable row-fold generator. No Invocation,
//! clear pool, public-vector filling, finite scan, fusion or row-head substitution.
use crate::encrypted_instruction_lowering::{Arithmetic, Instruction, Refusal, Selector};
use pllm_core::row_fold::{InstructionPlan, Kernel};

pub(crate) fn instruction(plan: &InstructionPlan, index: usize) -> Result<Option<Instruction>, Refusal> {
    let Some(op) = plan.instructions().get(index) else { return Ok(None); };
    let selector = match op.kernel {
        Kernel::PublicProduct(v) => Selector::RowVector(v),
        Kernel::Rotate(d) => Selector::RowRotation(d),
        Kernel::Add => Selector::Arithmetic(Arithmetic::Add),
        Kernel::Subtract => Selector::Arithmetic(Arithmetic::Subtract),
        Kernel::Square => Selector::Arithmetic(Arithmetic::Square),
        // Clear scalar-row continuation is NOT the source encrypted prefix.
        Kernel::SubtractRowHead => return Err(Refusal::Source),
    };
    let binary = matches!(op.kernel, Kernel::Add | Kernel::Subtract);
    if binary != op.right.is_some() { return Err(Refusal::Source); }
    Ok(Some(Instruction { selector, parents: [op.left, op.right.unwrap_or(usize::MAX)],
        arity: if binary { 2 } else { 1 } }))
}

#[cfg(test)]
#[path = "row_fold_ciphertext_lowering_tests.rs"]
mod tests;
