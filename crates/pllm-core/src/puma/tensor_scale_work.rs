//! Original scalar scale composition. Coefficient selection remains caller-owned;
//! this neither chooses a numeric profile nor certifies any input/source range.
use super::attention_work::SoftmaxSink;
use super::*;

// Both native emission and metadata preflight visit the same two original
// operations. In particular zero/one/negative coefficients are NOT shortcuts.
fn emit<S: SoftmaxSink>(sink: &mut S, coefficient: i64) -> Result<Value, Error> {
    let input = sink.input();
    let scaled = sink.affine(
        input,
        None,
        Affine {
            left: coefficient,
            right: 0,
            constant: 0,
        },
    )?;
    sink.rescale(scaled, 18)
}

// Literal ordinary map recipe, distinct from affine scaling above.
fn emit_rescale<S: SoftmaxSink>(sink: &mut S, shift: u8) -> Result<Value, Error> {
    let input = sink.input();
    sink.rescale(input, shift)
}

impl Program {
    /// Exact ordinary Builder composition: input -> rescale(shift).
    pub fn rescale(lanes: usize, input: Interval, shift: u8) -> Result<Self, Error> {
        let mut builder = Builder::new(lanes, input)?;
        let output = emit_rescale(&mut builder, shift)?;
        builder.finish(output)
    }

    /// Factory-free work query over the same rescale emission and native cost
    /// authority. Caller quotes the existing workspace before construction.
    /// No admission, execution ledger or input/source authority is granted.
    pub fn rescale_work(
        lanes: usize,
        input: Interval,
        shift: u8,
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.clear();
        validate_tile_lanes(lanes)?;
        workspace.reset(lanes, input);
        let output = emit_rescale(workspace, shift)?;
        workspace.finish_work(output)
    }

    /// Exact existing Builder composition: input -> affine(coefficient,0,0)
    /// -> rescale18. Caller retains the original coefficient calculation.
    pub fn scale18(lanes: usize, input: Interval, coefficient: i64) -> Result<Self, Error> {
        #[cfg(test)]
        FACTORY_ENTRIES.with(|count| count.set(count.get() + 1));
        let mut builder = Builder::new(lanes, input)?;
        let output = emit(&mut builder, coefficient)?;
        builder.finish(output)
    }

    /// Allocation-free work query over the SAME native emission/validation and
    /// instruction_work authority. Quote existing workspace_bytes BEFORE empty.
    /// No Program/Builder, coefficient choice, Budget or execution authority.
    /// Errors are not usable partial reports; each query resets metadata.
    /// Private callframes/allocator/RSS remain outside this workspace contract.
    pub fn scale18_work(
        lanes: usize,
        input: Interval,
        coefficient: i64,
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.clear();
        validate_tile_lanes(lanes)?;
        workspace.reset(lanes, input);
        let output = emit(workspace, coefficient)?;
        workspace.finish_work(output)
    }
}

#[cfg(test)]
thread_local! { static FACTORY_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
fn entries() -> usize {
    FACTORY_ENTRIES.with(std::cell::Cell::get)
}

#[cfg(test)]
#[path = "tensor_scale_work_tests.rs"]
mod tests;
