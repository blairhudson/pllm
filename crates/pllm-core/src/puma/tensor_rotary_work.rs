//! Original half-split rotary traversal, shared by native emission and work-only
//! metadata. No coefficient owner, instruction array or Program is cloned here.
use super::attention_work::SoftmaxSink;
use super::numeric::{affine, ONE};
use super::*;

#[cfg(test)]
thread_local! { static FACTORY_ATTEMPTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(super) fn entered_factory() {
    FACTORY_ATTEMPTS.with(|count| count.set(count.get() + 1));
}
#[cfg(test)]
pub(super) fn factory_attempts() -> usize {
    FACTORY_ATTEMPTS.with(|count| count.get())
}

// Keep the original multiply -> shape -> Builder tile -> coefficient-domain
// validation order. In particular, a zero/odd/short geometry is Shape, not the
// geometry-only ConstructionQuote's earlier tile Bounds winner.
pub(super) fn geometry(
    rows: usize,
    width: usize,
    cosine_len: usize,
    sine_len: usize,
) -> Result<usize, Error> {
    let lanes = rows.checked_mul(width).ok_or(Error::Bounds)?;
    if rows == 0 || width == 0 || width % 2 != 0 || cosine_len != lanes / 2 || sine_len != lanes / 2
    {
        return Err(Error::Shape);
    }
    Ok(lanes)
}

pub(super) fn validate_coefficients(cosine: &[i64], sine: &[i64]) -> Result<(), Error> {
    if cosine.iter().chain(sine).any(|c| !(-ONE..=ONE).contains(c)) {
        return Err(Error::Bounds);
    }
    Ok(())
}

// Exact original row/half/sign selection. Geometry and coefficient range have
// already been checked; the sine sign flip therefore cannot negate i64::MIN.
// Emission fills its existing native affine table scratch; work scans this same
// borrowed selection directly, without materializing an affine/coefficient array.
struct Coefficients<'a> {
    values: &'a [i64],
    lanes: usize,
    width: usize,
    next: usize,
    signed: bool,
}
impl<'a> Coefficients<'a> {
    fn new(values: &'a [i64], lanes: usize, width: usize, signed: bool) -> Self {
        Self {
            values,
            lanes,
            width,
            next: 0,
            signed,
        }
    }
}
impl Iterator for Coefficients<'_> {
    type Item = Affine;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.lanes {
            return None;
        }
        let i = self.next;
        self.next += 1;
        let half = self.width / 2;
        let c = self.values[(i / self.width) * half + i % half];
        Some(affine(
            if self.signed && i % self.width < half {
                -c
            } else {
                c
            },
            0,
            0,
        ))
    }
}

pub(super) trait RotarySink: SoftmaxSink {
    fn coefficients(
        &mut self,
        input: Value,
        width: usize,
        values: &[i64],
        signed: bool,
    ) -> Result<Value, Error>;
}

pub(super) fn recipe<S: RotarySink>(
    sink: &mut S,
    width: usize,
    cosine: &[i64],
    sine: &[i64],
) -> Result<Value, Error> {
    let x = sink.input();
    let cosine_product = sink.coefficients(x, width, cosine, false)?;
    let cosine_product = sink.rescale(cosine_product, 18)?;
    let swapped = sink.row_map(x, width, RowMap::Rotate(width / 2))?;
    let sine_product = sink.coefficients(swapped, width, sine, true)?;
    let sine_product = sink.rescale(sine_product, 18)?;
    sink.affine(cosine_product, Some(sine_product), affine(1, 1, 0))
}

impl RotarySink for Builder {
    fn coefficients(
        &mut self,
        input: Value,
        width: usize,
        values: &[i64],
        signed: bool,
    ) -> Result<Value, Error> {
        let lanes = self.node(input, false)?.lanes;
        let mut coefficients = [affine(0, 0, 0); MAX_TILE_LANES];
        for (slot, coefficient) in coefficients[..lanes]
            .iter_mut()
            .zip(Coefficients::new(values, lanes, width, signed))
        {
            *slot = coefficient;
        }
        self.affine_lanes(input, &coefficients[..lanes])
    }
}

impl RotarySink for rms_resident::ResidentRecipeSink<'_> {
    fn coefficients(
        &mut self,
        input: Value,
        width: usize,
        values: &[i64],
        signed: bool,
    ) -> Result<Value, Error> {
        let lanes = self.node(input, false)?.lanes;
        let mut coefficients = [affine(0, 0, 0); MAX_TILE_LANES];
        for (slot, coefficient) in coefficients[..lanes]
            .iter_mut()
            .zip(Coefficients::new(values, lanes, width, signed))
        {
            *slot = coefficient;
        }
        // Both tables are acquired through the installed ResidentBuilder. A
        // later-table failure cannot drop the first table or pending backing.
        self.0.affine_lanes(input, &coefficients[..lanes])
    }
}

struct Counter<'a> {
    metadata: &'a mut SoftmaxWorkWorkspace,
    tables: usize,
}
impl SoftmaxSink for Counter<'_> {
    fn node(&self, value: Value, bit: bool) -> Result<Node, Error> {
        self.metadata.node(value, bit)
    }
    fn push(
        &mut self,
        instruction: Instruction,
        bounds: Interval,
        bit: bool,
    ) -> Result<Value, Error> {
        self.metadata.push(instruction, bounds, bit)
    }
    fn refine(&mut self, value: Value, bounds: Interval) {
        self.metadata.refine(value, bounds);
    }
    fn compact(&self) -> bool {
        false
    }
    fn sign_compact(&mut self, _: Value) -> Result<Value, Error> {
        Err(Error::Binding)
    }
    fn width(&self, value: Value) -> Result<u8, Error> {
        self.metadata.width(value)
    }
}
impl RotarySink for Counter<'_> {
    fn coefficients(
        &mut self,
        input: Value,
        width: usize,
        values: &[i64],
        signed: bool,
    ) -> Result<Value, Error> {
        let node = self.node(input, false)?;
        if self.tables == 2 || self.metadata.instruction_slots() == MAX_NODES {
            return Err(Error::Bounds);
        }
        let bounds = rms_work::affine_bounds(
            node.bounds,
            Coefficients::new(values, node.lanes, width, signed),
        )?;
        let output = self.push(
            Instruction::LaneAffine(input, self.tables as u8),
            bounds,
            false,
        )?;
        self.tables += 1;
        Ok(output)
    }
}

/// Named known logical work-only slots. NOT generated stack/ABI spill space,
/// dependency/private callframes, allocator, original coefficient-owner capacity,
/// proof CPU, native constructor storage or RSS. No reservation/admission power.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotaryWorkResources {
    pub workspace_bytes: usize,
    pub controller_bytes: usize,
    pub entry_bytes: usize,
    pub validation_bytes: usize,
    pub recipe_bytes: usize,
    pub coefficient_bytes: usize,
    pub primitive_bytes: usize,
    pub finish_bytes: usize,
    pub known_peak_bytes: usize,
}
impl Program {
    /// Quote BEFORE constructing fixed metadata or entering the work pass. The
    /// named slot union below deliberately overbounds disjoint helper branches;
    /// add caller driver/source-owner controls, then max with the disjoint native
    /// construction phase. Do NOT add/subtract borrowed slice bytes as capacity
    /// credits. Existing Construction::RotaryHalf quotes native storage separately.
    pub const fn rotary_half_work_resources() -> RotaryWorkResources {
        use std::mem::size_of;
        let workspace_bytes = size_of::<SoftmaxWorkWorkspace>();
        let controller_bytes = size_of::<Counter<'static>>();
        let entry_bytes = 2 * size_of::<usize>()
            + size_of::<Interval>()
            + 2 * size_of::<&[i64]>()
            + size_of::<&mut SoftmaxWorkWorkspace>()
            + 2 * size_of::<Result<usize, Error>>()
            + 2 * size_of::<Result<(), Error>>()
            + 2 * size_of::<Result<Value, Error>>()
            + 2 * size_of::<Result<WorkBudget, Error>>();
        let validation_bytes = 6 * size_of::<usize>()
            + 2 * size_of::<&[i64]>()
            + size_of::<
                std::iter::Chain<std::slice::Iter<'static, i64>, std::slice::Iter<'static, i64>>,
            >()
            + size_of::<std::ops::RangeInclusive<i64>>()
            + 2 * size_of::<std::slice::Iter<'static, i64>>()
            + size_of::<Option<&i64>>()
            + 2 * size_of::<&i64>()
            + size_of::<bool>()
            + size_of::<Result<usize, Error>>()
            + 2 * size_of::<Result<(), Error>>();
        let recipe_bytes = size_of::<&mut Counter<'static>>()
            + 2 * size_of::<usize>()
            + 2 * size_of::<&[i64]>()
            + 7 * size_of::<Value>()
            + size_of::<RowMap>()
            + size_of::<Affine>()
            + 8 * size_of::<Result<Value, Error>>();
        // Includes borrowed iterator/next/scan slots and the original scalar
        // Affine::output_bounds/term/rescaled_bounds helpers (no private proof
        // arrays or native table allocation). These are named types, not padding.
        let coefficient_bytes = size_of::<&mut Counter<'static>>()
            + size_of::<&[i64]>()
            // Constructor return, by-value scan argument, IntoIterator binding.
            + 3 * size_of::<Coefficients<'static>>()
            + 5 * size_of::<usize>()
            + 2 * size_of::<bool>()
            + size_of::<Value>()
            + size_of::<Node>()
            + 2 * size_of::<Affine>()
            + size_of::<Option<Affine>>()
            + 4 * size_of::<Interval>()
            + 2 * size_of::<InputBounds>()
            + 10 * size_of::<i128>()
            + 4 * size_of::<i64>()
            + size_of::<u8>()
            + 2 * size_of::<Option<i128>>()
            + size_of::<(i128, i128)>()
            + size_of::<Result<Node, Error>>()
            + 3 * size_of::<Result<Interval, Error>>()
            + 2 * size_of::<Result<Value, Error>>();
        // Counter -> fixed metadata -> original checked_node + instruction_work;
        // argument and Result copies, input pair and node transition are explicit.
        let primitive_bytes = size_of::<&mut Counter<'static>>()
            + 3 * size_of::<&mut SoftmaxWorkWorkspace>()
            // checked_node arguments and its two closure capture references.
            + 2 * size_of::<&[Option<Node>; MAX_NODES]>()
            + 2 * size_of::<&[u8; MAX_NODES]>()
            + 3 * size_of::<Instruction>()
            + 3 * size_of::<Interval>()
            + 3 * size_of::<bool>()
            + 3 * size_of::<Value>()
            + size_of::<[Option<Value>; 2]>()
            + 2 * size_of::<Node>()
            + 4 * size_of::<usize>()
            + 2 * size_of::<u8>()
            + 2 * size_of::<(Value, Node, u8)>()
            + 2 * size_of::<Result<(Value, Node, u8), Error>>()
            + 3 * size_of::<Result<Value, Error>>()
            + 2 * size_of::<Result<Node, Error>>()
            + size_of::<Result<u8, Error>>();
        let finish_bytes = 2 * size_of::<&SoftmaxWorkWorkspace>()
            + size_of::<Value>()
            + size_of::<std::iter::Flatten<std::slice::Iter<'static, Option<Node>>>>()
            + 2 * size_of::<Node>()
            + size_of::<Option<&Node>>()
            // Fold state, cost/checked_add arguments, initializer/return slots.
            + 6 * size_of::<WorkBudget>()
            + 6 * size_of::<u64>()
            + size_of::<u8>()
            + 5 * size_of::<Result<WorkBudget, Error>>()
            + size_of::<Result<Node, Error>>()
            + size_of::<Result<u64, Error>>();
        RotaryWorkResources {
            workspace_bytes,
            controller_bytes,
            entry_bytes,
            validation_bytes,
            recipe_bytes,
            coefficient_bytes,
            primitive_bytes,
            finish_bytes,
            known_peak_bytes: workspace_bytes
                + controller_bytes
                + entry_bytes
                + validation_bytes
                + recipe_bytes
                + coefficient_bytes
                + primitive_bytes
                + finish_bytes,
        }
    }

    /// Exact work for ONE original half-split rotary tile, using borrowed original
    /// coefficient rows. No Program/Builder/Arc, public table, coefficient array,
    /// transcendental generation or heap allocation is used to discover work.
    ///
    /// Quote rotary_half_work_resources BEFORE SoftmaxWorkWorkspace::empty().
    /// Metadata resets on each call, including after errors; partial metadata is
    /// never a successful report. The existing closed-ticket controller poisons
    /// on error until its reset(). Caller checked-aggregates actual full/tail
    /// invocations and outside/graph work BEFORE factories; query does not debit.
    ///
    /// The closed ticket retains original immutable coefficient borrows:
    /// ```compile_fail
    /// use pllm_core::puma::multiply::{Interval, tensor::{Construction, FactoryParameters, WorkPreflightWorkspace}};
    /// let mut cosine = vec![1_i64 << 18]; let sine = vec![0_i64];
    /// let factory = Construction::RotaryHalf { rows: 1, width: 2, cosine_len: 1, sine_len: 1 }
    ///     .bind(FactoryParameters::RotaryHalf { input: Interval::new(-1, 1).unwrap(), cosine: &cosine, sine: &sine }).unwrap();
    /// let _quote = WorkPreflightWorkspace::resources();
    /// let mut workspace = WorkPreflightWorkspace::empty();
    /// let ticket = factory.preflight(1, &mut workspace).unwrap();
    /// cosine[0] = 0;
    /// let _ = ticket.summary();
    /// ```
    ///
    /// Owner cannot be dropped even after numeric admission:
    /// ```compile_fail
    /// use pllm_core::puma::{WorkBudget, multiply::{Interval, tensor::{Construction, FactoryParameters, WorkPreflightWorkspace, CallerWorkLedger}}};
    /// let cosine = vec![1_i64 << 18]; let sine = vec![0_i64];
    /// let factory = Construction::RotaryHalf { rows: 1, width: 2, cosine_len: 1, sine_len: 1 }
    ///     .bind(FactoryParameters::RotaryHalf { input: Interval::new(-1, 1).unwrap(), cosine: &cosine, sine: &sine }).unwrap();
    /// let _quote = WorkPreflightWorkspace::resources();
    /// let mut workspace = WorkPreflightWorkspace::empty();
    /// let ticket = factory.preflight(1, &mut workspace).unwrap();
    /// let mut total = WorkBudget::default();
    /// let admitted = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap().admit(ticket).unwrap();
    /// drop(cosine);
    /// let _ = admitted.into_parts().0.build();
    /// ```
    ///
    /// Consuming build cannot substitute equally sized foreign coefficients:
    /// ```compile_fail
    /// use pllm_core::puma::{WorkBudget, multiply::{Interval, tensor::{Construction, FactoryParameters, WorkPreflightWorkspace, CallerWorkLedger}}};
    /// let cosine = [1_i64 << 18]; let sine = [0_i64]; let foreign = [0_i64];
    /// let factory = Construction::RotaryHalf { rows: 1, width: 2, cosine_len: 1, sine_len: 1 }
    ///     .bind(FactoryParameters::RotaryHalf { input: Interval::new(-1, 1).unwrap(), cosine: &cosine, sine: &sine }).unwrap();
    /// let _quote = WorkPreflightWorkspace::resources();
    /// let mut workspace = WorkPreflightWorkspace::empty();
    /// let ticket = factory.preflight(1, &mut workspace).unwrap();
    /// let mut total = WorkBudget::default();
    /// let admitted = CallerWorkLedger::new(&mut total, u64::MAX, u64::MAX).unwrap().admit(ticket).unwrap();
    /// let _ = admitted.into_parts().0.build(&foreign);
    /// ```
    pub fn rotary_half_work(
        rows: usize,
        width: usize,
        input: Interval,
        cosine: &[i64],
        sine: &[i64],
        workspace: &mut SoftmaxWorkWorkspace,
    ) -> Result<WorkBudget, Error> {
        workspace.clear();
        let lanes = geometry(rows, width, cosine.len(), sine.len())?;
        validate_tile_lanes(lanes)?;
        workspace.reset(lanes, input);
        validate_coefficients(cosine, sine)?;
        let mut counter = Counter {
            metadata: workspace,
            tables: 0,
        };
        let output = recipe(&mut counter, width, cosine, sine)?;
        counter.metadata.finish_work(output)
    }
}
