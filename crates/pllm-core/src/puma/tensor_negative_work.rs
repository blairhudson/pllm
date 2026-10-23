//! Legacy Negative factory traversal; native emission and pure work share it.
//! Work numbers confer no Budget, source/range proof or execution authority.
use super::attention_work::SoftmaxSink;
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

// Use the SAME primitive defaults, node validation and instruction-cost owner
// as Builder. No independently encoded conversion or Boolean-work formula.
pub(super) fn recipe<S: SoftmaxSink>(sink: &mut S) -> Result<Value, Error> {
    let bit = sink.sign(sink.input())?;
    sink.from_bit(bit)
}

// Concrete scalar slots, not a lane/Node array, Program, Builder or growable
// buffer. Every push uses the original shared native validation/cost authority.
struct Counter {
    input: Node,
    sign: Option<Node>,
    output: Option<Node>,
    sign_width: u8,
    output_width: u8,
    len: usize,
    work: WorkBudget,
}

impl Counter {
    fn raw_node(&self, v: Value) -> Result<Node, Error> {
        match v.0 {
            0 => Ok(self.input),
            1 => self.sign.ok_or(Error::Binding),
            2 => self.output.ok_or(Error::Binding),
            _ => Err(Error::Binding),
        }
    }
    fn raw_width(&self, v: Value) -> u8 {
        match v.0 {
            1 => self.sign_width,
            2 => self.output_width,
            _ => 0,
        }
    }
}

impl SoftmaxSink for Counter {
    fn node(&self, v: Value, bit: bool) -> Result<Node, Error> {
        let node = self.raw_node(v)?;
        if node.bit != bit {
            return Err(Error::Binding);
        }
        Ok(node)
    }
    fn push(
        &mut self,
        instruction: Instruction,
        bounds: Interval,
        bit: bool,
    ) -> Result<Value, Error> {
        if self.len >= 3 {
            return Err(Error::Bounds);
        }
        let (v, node, width) = attention_work::checked_node_with(
            |v| self.raw_node(v),
            |v| self.width(v),
            self.len,
            self.input.lanes,
            instruction,
            bounds,
            bit,
        )?;
        let work = self
            .work
            .checked_add(attention_work::instruction_work_with(node, |v| {
                self.raw_width(v)
            })?)?;
        if self.len == 1 {
            self.sign = Some(node);
            self.sign_width = width;
        } else {
            self.output = Some(node);
            self.output_width = width;
        }
        self.len += 1;
        self.work = work;
        Ok(v)
    }
    fn width(&self, v: Value) -> Result<u8, Error> {
        self.node(v, true)?;
        Ok(self.raw_width(v))
    }
    fn refine(&mut self, v: Value, bounds: Interval) {
        let node = match v.0 {
            0 => Some(&mut self.input),
            1 => self.sign.as_mut(),
            2 => self.output.as_mut(),
            _ => None,
        };
        if let Some(node) = node {
            node.bounds = bounds;
        }
    }
    fn compact(&self) -> bool {
        false
    }
    fn sign_compact(&mut self, _: Value) -> Result<Value, Error> {
        Err(Error::Binding)
    }
}

/// Known logical slots, not a compiled call-stack/RSS or complete physical
/// envelope. The input pair below is the shared validator's two Option<Value>
/// return; no retained lane/Node array exists. Closure captures are two shared
/// Counter references. Entries explicitly include argument/Result copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegativeWorkResources {
    pub state_bytes: usize,
    pub entry_bytes: usize,
    pub recipe_bytes: usize,
    pub primitive_bytes: usize,
    pub validation_bytes: usize,
    pub cost_bytes: usize,
    pub known_peak_bytes: usize,
}

impl Program {
    /// Quote BEFORE a work-only count pass. Caller must add its own enclosing
    /// driver/control slots. Native construction is a later, disjoint phase;
    /// max those phases rather than pretending this admits native backing.
    pub const fn negative_work_resources() -> NegativeWorkResources {
        use std::mem::size_of;
        let state_bytes = size_of::<Counter>();
        let entry_bytes = size_of::<usize>()
            + size_of::<Interval>()
            + size_of::<Result<(), Error>>()
            + size_of::<Result<Node, Error>>()
            + 2 * size_of::<Result<WorkBudget, Error>>();
        let recipe_bytes = size_of::<&mut Counter>()
            + 2 * size_of::<Value>()
            + 2 * size_of::<Result<Value, Error>>();
        let primitive_bytes = 2 * size_of::<&mut Counter>()
            + 2 * size_of::<Instruction>()
            + 2 * size_of::<Interval>()
            + 2 * size_of::<bool>()
            + 2 * size_of::<u8>()
            + size_of::<Value>()
            + size_of::<Node>()
            + size_of::<(Value, Node, u8)>()
            + 2 * size_of::<Result<(Value, Node, u8), Error>>()
            + 2 * size_of::<Result<Value, Error>>();
        let validation_bytes = 2 * size_of::<&Counter>()
            + 3 * size_of::<usize>()
            + size_of::<Instruction>()
            + size_of::<Interval>()
            + size_of::<bool>()
            + 2 * size_of::<Value>()
            + size_of::<[Option<Value>; 2]>()
            + size_of::<Node>()
            + size_of::<u8>()
            + 2 * size_of::<Result<Node, Error>>()
            + size_of::<Result<Value, Error>>()
            + size_of::<Result<u8, Error>>()
            + size_of::<(Value, Node, u8)>()
            + size_of::<Result<(Value, Node, u8), Error>>();
        let cost_bytes = size_of::<&Counter>()
            + size_of::<Node>()
            + size_of::<Value>()
            + size_of::<u8>()
            + 3 * size_of::<WorkBudget>()
            + 2 * size_of::<Result<WorkBudget, Error>>();
        let transition = if validation_bytes > cost_bytes {
            validation_bytes
        } else {
            cost_bytes
        };
        NegativeWorkResources {
            state_bytes,
            entry_bytes,
            recipe_bytes,
            primitive_bytes,
            validation_bytes,
            cost_bytes,
            known_peak_bytes: state_bytes
                + entry_bytes
                + recipe_bytes
                + primitive_bytes
                + transition,
        }
    }

    /// Known scalar scratch ONLY. Quote before calling `negative_work`.
    /// Arguments/results, compiled/native dependency callframes, full transition
    /// overlap and RSS are separate; this number is not constructor admission.
    pub const fn negative_work_workspace_bytes() -> usize {
        std::mem::size_of::<Counter>()
    }

    /// Exact work/cursors for ONE original `negative` tile, without constructing
    /// Program, Builder, Arc, native instruction storage or execution buffers.
    /// Quote `negative_work_workspace_bytes()` BEFORE this call. Scratch consists
    /// of three individual node slots, widths, length and work: no retained arrays, heap,
    /// retained input borrow or caller-owned workspace. Fresh state each call.
    ///
    /// The legacy factory always emits full signed conversion and FromBit,
    /// including all-negative, nonnegative and singleton domains. Do not replace
    /// it with a public constant or `negative_compact` based on these bounds.
    /// Every valid signed Interval is accepted; geometry/error checks match the
    /// native constructor. Errors return no report and retain no partial state.
    ///
    /// Caller must checked-add actual full/tail INVOCATIONS, tree selection work,
    /// existing graph and outside request work, then admit the whole result
    /// BEFORE metadata reservations/native factories. This query neither debits
    /// that aggregate nor authenticates its range/source obligations. Workspace
    /// bytes exclude native buffers, dependency callframes, proof CPU and RSS.
    pub fn negative_work(lanes: usize, input: Interval) -> Result<WorkBudget, Error> {
        validate_tile_lanes(lanes)?;
        let mut counter = Counter {
            input: Node {
                instruction: Instruction::Input,
                bounds: input,
                bit: false,
                lanes,
            },
            sign: None,
            output: None,
            sign_width: 0,
            output_width: 0,
            len: 1,
            work: WorkBudget::default(),
        };
        let output = recipe(&mut counter)?;
        counter.node(output, false)?;
        if output.0 as usize + 1 != counter.len || counter.len != 3 {
            return Err(Error::Binding);
        }
        Ok(counter.work)
    }
}
